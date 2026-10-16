import torch
import einops
import numpy as np
import tqdm
from PIL import Image
import torchvision.transforms.v2 as transforms
from diffusers import FlowMatchEulerDiscreteScheduler
# FlowUniPCMultistepScheduler generates more details than FlowMatchEulerDiscreteScheduler
from .fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402
from .flash_scheduler import FlashFlowMatchEulerDiscreteScheduler
from .utils import resize_pilimage, calculate_dimensions, get_rope_index_fix_point, find_closest_resolution
from .utils import create_layout_reference_images, load_layout_bboxes

TIMESTEP_TOKEN_NUM = 1
NOISE_SCALE = 8.0
T_EPS = 0.001
CONDITION_IMAGE_SIZE = 384
PATCH_SIZE = 32

TENSOR_TRANSFORM = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize([0.5], [0.5]),
])

DEFAULT_TIMESTEPS = [
    999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764, 737,
    707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8,
]

def build_t2i_text_sample(prompt, height, width, tokenizer, processor, model_config):
    image_token_id = model_config.image_token_id
    video_token_id = model_config.video_token_id
    vision_start_token_id = model_config.vision_start_token_id
    image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)

    boi_token = getattr(tokenizer, "boi_token", "<|boi_token|>")
    tms_token = getattr(tokenizer, "tms_token", "<|tms_token|>")

    messages = [{"role": "user", "content": prompt}]
    template_caption = (
            processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            + boi_token
            + tms_token * TIMESTEP_TOKEN_NUM
    )
    input_ids = tokenizer.encode(template_caption, return_tensors="pt", add_special_tokens=False)

    image_grid_thw = torch.tensor(
        [1, height // PATCH_SIZE, width // PATCH_SIZE], dtype=torch.int64
    ).unsqueeze(0)

    vision_tokens = torch.zeros((1, image_len), dtype=input_ids.dtype) + image_token_id
    vision_tokens[0, 0] = vision_start_token_id
    input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

    position_ids, _ = get_rope_index_fix_point(
        1, image_token_id, video_token_id, vision_start_token_id,
        input_ids=input_ids_pad, image_grid_thw=image_grid_thw,
        video_grid_thw=None, attention_mask=None, skip_vision_start_token=[1],
    )

    txt_seq_len = input_ids.shape[-1]
    all_seq_len = position_ids.shape[-1]

    token_types = torch.zeros((1, all_seq_len), dtype=input_ids.dtype)
    bgn = txt_seq_len - TIMESTEP_TOKEN_NUM
    token_types[0, bgn: bgn + image_len + TIMESTEP_TOKEN_NUM] = 1
    token_types[0, txt_seq_len - TIMESTEP_TOKEN_NUM: txt_seq_len] = 3

    vinput_mask = (token_types == 1)
    token_types_bin = (token_types > 0).to(token_types.dtype)

    return {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'token_types': token_types_bin,
        'vinput_mask': vinput_mask,
    }

def build_scheduler(num_inference_steps, timesteps_list, shift, device, scheduler_name="default"):
    if scheduler_name == "flash":
        sched = FlashFlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)
    elif scheduler_name == "flow_match":
        sched = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=shift)
    elif scheduler_name == "default":
        sched = FlowUniPCMultistepScheduler(use_dynamic_shifting=False, shift=shift)
    else:
        raise ValueError(f"Unknown scheduler_name={scheduler_name!r}")
    sched.set_timesteps(num_inference_steps, device=device)
    if timesteps_list is not None:
        sched.timesteps = torch.tensor(timesteps_list, device=device, dtype=torch.long)
        sigmas = [t.item() / 1000.0 for t in sched.timesteps]
        sigmas.append(0.0)
        sched.sigmas = torch.tensor(sigmas, device=device)
    return sched


def clamp_tensor(tensor, percentage = 0.1):
    lower_bound = torch.quantile(tensor.float(), percentage)
    upper_bound = torch.quantile(tensor.float(), 1 - percentage)
    src_dtype = tensor.dtype
    return torch.clamp(tensor.float(), min=lower_bound, max=upper_bound).to(src_dtype)

@torch.no_grad()
def generate_image(
        model,
        processor,
        prompt: str,
        ref_image_paths: list = None,
        height: int = 1440,
        width: int = 2560,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        shift: float = 3.0,
        timesteps_list=None,
        scheduler_name: str = "default",
        seed: int = 42,
        noise_scale_start: float = NOISE_SCALE,
        noise_scale_end: float = NOISE_SCALE,
        noise_clip_std: float = 0.0,
        keep_original_aspect: bool = False,
        layout_bboxes: str = None,
        use_flash_attn: bool = True,
        callback=None,
) -> Image.Image:
    device = model.device
    dtype = torch.bfloat16
    model_config = model.config
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

    # When `keep_original_aspect` is enabled and exactly one reference image is
    # provided, resize the reference to max_size=2048 (patch-aligned) and derive
    # the target image dimensions from the resized reference. This bypasses the
    # predefined-resolution snapping so the output preserves the reference's
    # original aspect ratio.
    preresized_ref_pil = None
    if keep_original_aspect and ref_image_paths and len(ref_image_paths) == 1:
        pil_orig = Image.open(ref_image_paths[0]).convert("RGB")
        preresized_ref_pil = resize_pilimage(pil_orig, 2048, PATCH_SIZE)
        width, height = preresized_ref_pil.size
        print(
            f"[info] keep_original_aspect: target size set to {width}x{height} "
            f"from reference image"
        )
    else:
        if keep_original_aspect:
            print(
                "[warning] keep_original_aspect requires exactly one reference "
                "image; falling back to default resolution snapping."
            )
        w, h = find_closest_resolution(width, height)
        if w != width or h != height:
            print(f"[warning] Resolution snapped from {width}x{height} to {w}x{h}")
            width, height = w, h

    h_patches = height // PATCH_SIZE
    w_patches = width // PATCH_SIZE

    if not ref_image_paths:
        cond_sample = build_t2i_text_sample(prompt, height, width, tokenizer, processor, model_config)
        uncond_sample = None
        if guidance_scale > 1.0:
            uncond_sample = build_t2i_text_sample(" ", height, width, tokenizer, processor, model_config)

        def to_device(s):
            return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in s.items()}

        cond_sample = to_device(cond_sample)
        if uncond_sample is not None:
            uncond_sample = to_device(uncond_sample)

        ref_patches = None
        tgt_image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)
        samples = [cond_sample]
        if uncond_sample:
            samples.append(uncond_sample)
    else:
        image_token_id = model_config.image_token_id
        video_token_id = model_config.video_token_id
        vision_start_token_id = model_config.vision_start_token_id
        spatial_merge_size = model_config.vision_config.spatial_merge_size

        if preresized_ref_pil is not None:
            ref_pils = [preresized_ref_pil]
        else:
            ref_pils = [Image.open(p).convert("RGB") for p in ref_image_paths]

        K = len(ref_pils)
        layout_data = None
        if layout_bboxes is not None and len(layout_bboxes) > 0 and preresized_ref_pil is None:
            try:
                layout_data = load_layout_bboxes(layout_bboxes)
                K += 1
            except Exception as e:
                print(f"Incorrect layout_bboxes: {layout_bboxes}, {e}")

        if K == 1: max_size = max(height, width)
        elif K == 2: max_size = max(height, width) * 48 // 64
        elif K <= 4: max_size = max(height, width) // 2
        elif K <= 8: max_size = max(height, width) * 24 // 64
        else: max_size = max(height, width) // 4

        if layout_data is not None:
            ref_pils = create_layout_reference_images(
                ref_pils = ref_pils,
                layout_bboxes = layout_data,
                image_width = width,
                image_height = height,
                ref_max_size = max_size,
                patch_size = PATCH_SIZE,
            )

        ref_pils_resized, ref_images = [], []
        for pil in ref_pils:
            # Skip resizing when caller already produced a patch-aligned ref via
            # `keep_original_aspect` — re-running resize_pilimage on it would
            # upscale (since max_size == max(width, height) of the resized ref).
            if preresized_ref_pil is not None and pil is preresized_ref_pil:
                pil_r = pil
            else:
                pil_r = resize_pilimage(pil, max_size, PATCH_SIZE)
            ref_pils_resized.append(pil_r)
            x = TENSOR_TRANSFORM(pil_r)
            x = einops.rearrange(x, "C (H p1) (W p2) -> (H W) (C p1 p2)", p1=PATCH_SIZE, p2=PATCH_SIZE)
            ref_images.append(x)

        ref_image_lens = [img.shape[0] for img in ref_images]
        total_ref_len = sum(ref_image_lens)
        ref_patches = torch.cat(ref_images, dim=0).unsqueeze(0).to(device, dtype)

        tgt_image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)
        h_patches = height // PATCH_SIZE
        w_patches = width // PATCH_SIZE

        if K <= 4: cond_img_size = CONDITION_IMAGE_SIZE
        elif K <= 8: cond_img_size = CONDITION_IMAGE_SIZE * 48 // 64
        else: cond_img_size = CONDITION_IMAGE_SIZE // 2

        ref_pils_vlm = []
        for pil_r in ref_pils_resized:
            cond_w, cond_h = calculate_dimensions(cond_img_size, pil_r.width / pil_r.height)
            ref_pils_vlm.append(pil_r.resize((cond_w, cond_h), resample=Image.LANCZOS))

        image_grid_thw_tgt = torch.tensor([1, height // PATCH_SIZE, width // PATCH_SIZE], dtype=torch.int64).unsqueeze(0)
        image_grid_thw_ref = torch.zeros((K, 3), dtype=torch.int64)
        for i, pil_r in enumerate(ref_pils_resized):
            rw, rh = pil_r.size
            image_grid_thw_ref[i] = torch.tensor([1, rh // PATCH_SIZE, rw // PATCH_SIZE], dtype=torch.int64)

        samples = []
        captions = [prompt]
        if guidance_scale > 1.0:
            captions.append(" ")

        for caption in captions:
            boi_token = getattr(tokenizer, "boi_token", "<|boi_token|>")
            tms_token = getattr(tokenizer, "tms_token", "<|tms_token|>")

            content = [{"type": "image"} for _ in range(K)]
            content.append({"type": "text", "text": caption})
            messages = [{"role": "user", "content": content}]
            template_caption = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            proc = processor(text=[template_caption], images=ref_pils_vlm, padding="longest", return_tensors="pt")
            input_ids_2 = tokenizer.encode(boi_token + tms_token * TIMESTEP_TOKEN_NUM, return_tensors="pt", add_special_tokens=False)
            input_ids = torch.cat([proc.input_ids, input_ids_2], dim=-1)

            igthw_cond = proc.image_grid_thw.clone()
            for i in range(K):
                igthw_cond[i, 1] //= spatial_merge_size
                igthw_cond[i, 2] //= spatial_merge_size
            igthw_all = torch.cat([igthw_cond, image_grid_thw_tgt, image_grid_thw_ref], dim=0)

            vision_tokens_list = []
            vt_tgt = torch.full((1, tgt_image_len), image_token_id, dtype=input_ids.dtype)
            vt_tgt[0, 0] = vision_start_token_id
            vision_tokens_list.append(vt_tgt)
            for rl in ref_image_lens:
                vt_ref = torch.full((1, rl), image_token_id, dtype=input_ids.dtype)
                vt_ref[0, 0] = vision_start_token_id
                vision_tokens_list.append(vt_ref)
            vision_tokens = torch.cat(vision_tokens_list, dim=1)
            input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

            position_ids, _ = get_rope_index_fix_point(
                1, image_token_id, video_token_id, vision_start_token_id,
                input_ids=input_ids_pad, image_grid_thw=igthw_all,
                video_grid_thw=None, attention_mask=None,
                skip_vision_start_token=[0] * K + [1] + [1] * K,
            )
            txt_seq_len = input_ids.shape[-1]
            all_seq_len = position_ids.shape[-1]

            token_types_raw = torch.zeros((1, all_seq_len), dtype=input_ids.dtype)
            bgn = txt_seq_len - TIMESTEP_TOKEN_NUM
            end = bgn + tgt_image_len + TIMESTEP_TOKEN_NUM
            token_types_raw[0, bgn:end] = 1
            token_types_raw[0, end: end + total_ref_len] = 2
            token_types_raw[0, txt_seq_len - TIMESTEP_TOKEN_NUM: txt_seq_len] = 3

            vinput_mask = torch.logical_or(token_types_raw == 1, token_types_raw == 2)
            token_types_bin = (token_types_raw > 0).to(token_types_raw.dtype)

            samples.append({
                "input_ids": input_ids.to(device),
                "position_ids": position_ids.to(device),
                "token_types": token_types_bin.to(device),
                "vinput_mask": vinput_mask.to(device),
                "pixel_values": proc.pixel_values.to(device, dtype),
                "image_grid_thw": proc.image_grid_thw.to(device),
            })

    noise = noise_scale_start * torch.randn(
        (1, 3, height, width),
        generator=torch.Generator('cpu').manual_seed(seed + 1),
    ).to(device, dtype)
    z = einops.rearrange(noise, 'B C (H p1) (W p2) -> B (H W) (C p1 p2)', p1=PATCH_SIZE, p2=PATCH_SIZE)

    sched = build_scheduler(num_inference_steps, timesteps_list, shift, device, scheduler_name)

    num_steps = len(sched.timesteps)
    if num_steps > 1:
        noise_scale_schedule = [
            noise_scale_start + (noise_scale_end - noise_scale_start) * i / (num_steps - 1)
            for i in range(num_steps)
        ]
    else:
        noise_scale_schedule = [noise_scale_start]

    torch.manual_seed(seed + 1)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed + 1)

    def forward_once(sample, z_in, t_pixeldit, precomputed_image_embeds=None, precomputed_deepstack_image_embeds=None):
        with torch.autocast(device.type, dtype=dtype, cache_enabled=False):
            kwargs = {
                "input_ids": sample['input_ids'],
                "position_ids": sample['position_ids'],
                "vinputs": z_in,
                "timestep": t_pixeldit.reshape(-1).to(device),
                "token_types": sample['token_types'],
                "use_flash_attn": use_flash_attn,
                "precomputed_image_embeds": precomputed_image_embeds,
                "precomputed_deepstack_image_embeds": precomputed_deepstack_image_embeds
            }
            if "pixel_values" in sample: kwargs["pixel_values"] = sample["pixel_values"]
            if "image_grid_thw" in sample: kwargs["image_grid_thw"] = sample["image_grid_thw"]

            outputs = model(**kwargs)

        x_pred = outputs.x_pred
        emb = getattr(outputs, "cond_image_embeds", None)
        ds_emb = getattr(outputs, "cond_deepstack_image_embeds", None)
        # x_pred = clamp_tensor(x_pred, percentage = 0.01)
        if ref_patches is None:
            return x_pred[0, sample['vinput_mask'][0]].unsqueeze(0)
        else:
            return x_pred[0, sample['vinput_mask'][0]][:tgt_image_len].unsqueeze(0), emb, ds_emb

    def _decode_x0_preview(x0_pred):
        """Convert a model-predicted x_0 (patch layout, [-1,1]) to a PIL image."""
        img_t = (x0_pred.float() + 1) / 2
        img_t = einops.rearrange(
            img_t.cpu(), 'B (H W) (C p1 p2) -> B C (H p1) (W p2)',
            H=h_patches, W=w_patches, p1=PATCH_SIZE, p2=PATCH_SIZE,
        )
        arr_p = np.round(np.clip(img_t[0].numpy().transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)
        return Image.fromarray(arr_p).convert("RGB")

    cond_image_embeds = None
    cond_deepstack_image_embeds = None

    for step_idx, step_t in enumerate(tqdm.tqdm(sched.timesteps, desc="Generating")):
        t_pixeldit = 1.0 - step_t.float() / 1000.0
        sigma = (step_t.float() / 1000.0).to(dtype=torch.float32).clamp_min(T_EPS)

        if ref_patches is None:
            x_pred_cond = forward_once(samples[0], z.clone(), t_pixeldit)
            v_cond = (x_pred_cond.to(dtype=torch.float32) - z.to(dtype=torch.float32)) / sigma

            if len(samples) > 1:
                x_pred_uncond = forward_once(samples[1], z.clone(), t_pixeldit)
                v_uncond = (x_pred_uncond.to(dtype=torch.float32) - z.to(dtype=torch.float32)) / sigma
                v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v_guided = v_cond
            preview_x0 = x_pred_cond
        else:
            vinputs = torch.cat([z, ref_patches], dim=1)
            x_vis_list = []
            for sample in samples:
                xp, emb, ds_emb = forward_once(
                    sample,
                    vinputs,
                    t_pixeldit,
                    precomputed_image_embeds=cond_image_embeds,
                    precomputed_deepstack_image_embeds=cond_deepstack_image_embeds
                )
                if emb is not None and ds_emb is not None:
                    cond_image_embeds = emb.detach()
                    cond_deepstack_image_embeds = [d.detach() for d in ds_emb]
                x_vis_list.append(xp)
            #x_vis_list = [forward_once(sample, vinputs, t_pixeldit) for sample in samples]
            x_vis_stacked = torch.cat(x_vis_list, dim=0)

            z_rep = z.expand(len(samples), -1, -1)
            v_pred = (x_vis_stacked.to(dtype=torch.float32) - z_rep.to(dtype=torch.float32)) / sigma

            v_cond = v_pred[0:1]
            if len(samples) > 1:
                v_uncond = v_pred[1:]
                v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v_guided = v_cond
            preview_x0 = x_vis_list[0]

        model_output = -v_guided
        # model_output = clamp_tensor(model_output, percentage = 0.05)
        if scheduler_name == "flash":
            z = sched.step(model_output.float(), step_t.to(dtype=torch.float32), z.float(), s_noise=noise_scale_schedule[step_idx], noise_clip_std=noise_clip_std, return_dict=False)[0].to(dtype)
        else:
            z = sched.step(model_output.float(), step_t.to(dtype=torch.float32), z.float(), return_dict=False)[0].to(dtype)

        if callback is not None:
            try:
                # Pass a closure that captures the current step's x0 prediction.
                # Use a default-arg binding to avoid late-binding issues.
                callback(step_idx, len(sched.timesteps),
                         lambda x0=preview_x0: _decode_x0_preview(x0))
            except Exception:
                pass

    img = (z + 1) / 2
    img = einops.rearrange(img.cpu().float(), 'B (H W) (C p1 p2) -> B C (H p1) (W p2)', H=h_patches, W=w_patches, p1=PATCH_SIZE, p2=PATCH_SIZE)
    arr = np.round(np.clip(img[0].numpy().transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")
