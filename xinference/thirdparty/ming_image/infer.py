#!/usr/bin/env python3
"""Unified Hugging Face inference entry point for Ming image checkpoints."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, List

from .inference_profile import (
    VALID_TASKS,
    load_checkpoint_capabilities,
    resolve_model_directory,
)
from .mllm_device_map import (
    build_mllm_device_plan,
    load_mllm_num_hidden_layers,
    validate_loaded_layer_devices,
)
CODE_DIRECTORY = Path(__file__).resolve().parent

TASK_RESOLUTION_BUCKETS = {
    "text-to-image": (1024, 2048),
    "image-edit": (1024,),
    "layer-decompose": (512, 1024),
}
TASK_DEFAULT_RESOLUTIONS = {
    "text-to-image": 2048,
    "image-edit": 1024,
    "layer-decompose": 1024,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run text-to-image, image editing, or layer decomposition with an "
            "explicit checkpoint capability profile."
        )
    )
    parser.add_argument("--model", required=True, help="Local model directory or HF Hub repo ID")
    parser.add_argument("--task", required=True, choices=sorted(VALID_TASKS))
    parser.add_argument("--prompt", help="Generation/edit prompt; optional for layer decomposition")
    parser.add_argument("--input-image", type=Path, help="Required for edit and layer decomposition")
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--resolution",
        type=int,
        help=(
            "Requested resolution bucket. Defaults: text-to-image 2048, "
            "image-edit 1024, layer-decompose 1024. Requests snap to the "
            "nearest bucket supported by the selected task."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Override the checkpoint-family default (generation: 12, layers: 12)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cfg",
        type=float,
        help="Override the checkpoint-family default (generation: 1.0, layers: 2.0)",
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        # The BailingMoeV2 LLM only implements eager and flash_attention_2
        # attention classes; selecting "sdpa" fails closed at load time.
        default="eager",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--device-map",
        choices=("balanced", "none", "auto"),
        default="balanced",
        help="balanced reserves GPU 0 for fixed image modules and shards the MLLM",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="required visible GPU count for balanced placement (any positive integer)",
    )
    parser.add_argument(
        "--processor",
        help="Optional processor data directory; defaults to <checkpoint>/mllm",
    )
    parser.add_argument("--revision", help="HF Hub model revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate model profile and task arguments without loading weights",
    )
    return parser.parse_args()


def parse_num_layers(text: str) -> int:
    decompose_match = re.search(
        r"decompose this image into\s+(\d+)\s+layers?", text.lower()
    )
    if decompose_match:
        return int(decompose_match.group(1))
    for line in text.splitlines():
        s = line.strip().lower()
        if s.startswith("number of layers:"):
            try:
                return int(s.split(":", 1)[1].strip())
            except ValueError:
                pass
    return 5


def resolve_task_resolution(task: str, requested: int | None) -> int:
    """Resolve a user request to the nearest supported task-level bucket."""
    try:
        buckets = TASK_RESOLUTION_BUCKETS[task]
    except KeyError as exc:
        raise ValueError(f"unsupported task for resolution policy: {task!r}") from exc
    if requested is None:
        return TASK_DEFAULT_RESOLUTIONS[task]
    if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
        raise ValueError("--resolution must be a positive integer")
    return min(buckets, key=lambda value: (abs(value - requested), value))


def _load_prompt(prompt: str) -> str:
    prompt_path = Path(prompt)
    try:
        is_file = prompt_path.is_file()
    except OSError:
        # Long literal prompts can be invalid filesystem paths. In that case,
        # keep treating the argument as prompt text.
        return prompt
    if is_file:
        return prompt_path.read_text(encoding="utf-8")
    return prompt


def _dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _build_messages(task: str, prompt: str, input_image: Path | None):
    content = []
    if input_image is not None:
        content.append({"type": "image", "image": str(input_image)})
    content.append({"type": "text", "text": prompt})
    return [{"role": "HUMAN", "content": content}]


def _normalize_outputs(output) -> List[Image.Image]:
    from PIL import Image

    if isinstance(output, Image.Image):
        return [output]
    if isinstance(output, (list, tuple)) and all(
        isinstance(item, Image.Image) for item in output
    ):
        return list(output)
    raise TypeError(f"model returned unsupported output type: {type(output)!r}")


def _model_input_device(model):
    try:
        return model.device
    except AttributeError:
        return next(parameter.device for parameter in model.parameters() if not parameter.is_meta)


def _validate_balanced_placement(model, plan, torch) -> None:
    layer_devices = []
    for index, layer in enumerate(model.model.model.layers):
        devices = {parameter.device for parameter in layer.parameters()}
        if len(devices) != 1:
            raise RuntimeError(f"MLLM layer {index} spans devices {sorted(map(str, devices))}")
        device = devices.pop()
        if device.type != "cuda" or device.index is None:
            raise RuntimeError(f"MLLM layer {index} loaded on {device}, expected CUDA")
        layer_devices.append(device.index)
    validate_loaded_layer_devices(layer_devices, plan)

    fixed_modules = {
        "vision": model.vision,
        "linear_proj": model.linear_proj,
        "connector": getattr(model, "connector", None),
        "proj_in": getattr(model, "proj_in", None),
        "proj_out": getattr(model, "proj_out", None),
        "proj_directvlm": getattr(model, "proj_directvlm", None),
        "query_tokens": getattr(model, "query_tokens_dict", None),
        "diffusion": getattr(model, "diffusion_loss", None),
    }
    expected = {torch.device("cuda:0")}
    for name, module in fixed_modules.items():
        if module is None:
            continue
        devices = {parameter.device for parameter in module.parameters()}
        if devices and devices != expected:
            raise RuntimeError(
                f"fixed module {name} loaded on {sorted(map(str, devices))}, expected cuda:0"
            )


def _save_outputs(
    images: Iterable, output_dir: Path, task: str
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "layer" if task == "layer-decompose" else "image"
    skip_first = task == "layer-decompose"
    paths = []
    for index, image in enumerate(images):
        if skip_first and index == 0:
            continue
        output_path = output_dir / f"{prefix}_{index:02d}.png"
        image.save(output_path)
        paths.append(output_path)
    return paths


def main() -> None:
    args = parse_args()
    model_directory = resolve_model_directory(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    profile = load_checkpoint_capabilities(model_directory)

    has_reference_image = args.input_image is not None
    profile.validate_task(
        args.task,
        has_reference_image=has_reference_image,
        num_layers=args.num_layers,
    )
    effective_resolution = resolve_task_resolution(args.task, args.resolution)
    if args.resolution is not None and args.resolution != effective_resolution:
        print(
            f"resolution {args.resolution} snapped to {effective_resolution} "
            f"for task {args.task}",
            file=sys.stderr,
        )
    sampling = profile.resolve_sampling_parameters(steps=args.steps, cfg=args.cfg)
    if args.input_image is not None and not args.input_image.is_file():
        raise FileNotFoundError(f"input image does not exist: {args.input_image}")
    if args.task != "layer-decompose" and not args.prompt:
        raise ValueError(f"--prompt is required for {args.task}")

    if args.prompt is not None:
        prompt = _load_prompt(args.prompt)
    else:
        prompt = f"Decompose this image into {args.num_layers} layers."
    num_layers = parse_num_layers(prompt) if args.task == "layer-decompose" else args.num_layers
    if args.validate_only:
        print(
            json.dumps(
                {
                    "model": str(model_directory),
                    "task": args.task,
                    "profile": profile.__dict__,
                    "sampling": sampling.__dict__,
                    "resolution": {
                        "requested": args.resolution,
                        "effective": effective_resolution,
                    },
                },
                indent=2,
            )
        )
        return

    model, processor = load_model_and_processor(model_directory, args)
    images = run_generation(
        model,
        processor,
        profile,
        task=args.task,
        prompt=prompt,
        input_image=args.input_image,
        resolution=effective_resolution,
        sampling=sampling,
        seed=args.seed,
        num_layers=num_layers,
        dtype=_dtype(args.dtype),
    )

    output_paths = _save_outputs(images, args.output_dir, args.task)
    print(json.dumps({"outputs": [str(path.resolve()) for path in output_paths]}, indent=2))


def load_model_and_processor(model_directory: Path, args):
    """Load the model and processor once; shared by the CLI and batch tools."""
    import torch

    from .modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration
    from .processing_bailingmm2 import load_bailingmm2_processor

    # Processor/tokenizer data lives in the package's mllm/ component; the
    # Python implementations stay in this repository (AutoProcessor would
    # require them inside the data directory). --processor overrides the data
    # directory only.
    processor_directory = (
        Path(args.processor).expanduser().resolve()
        if args.processor
        else model_directory / "mllm"
    )
    processor = load_bailingmm2_processor(processor_directory)

    dtype = _dtype(args.dtype)
    load_kwargs = {
        "torch_dtype": dtype,
        "attn_implementation": args.attn_implementation,
        "load_image_gen": True,
        "image_gen_device": args.device,
    }
    device_plan = None
    if args.device_map == "balanced":
        visible_gpus = torch.cuda.device_count()
        if visible_gpus != args.num_gpus:
            raise RuntimeError(
                f"balanced placement requires {args.num_gpus} visible GPUs, got {visible_gpus}; "
                "set CUDA_VISIBLE_DEVICES before starting Python"
            )
        num_hidden_layers = load_mllm_num_hidden_layers(model_directory / "mllm")
        device_plan = build_mllm_device_plan(num_hidden_layers, visible_gpus)
        load_kwargs["device_map"] = device_plan.device_map
        print(
            f"MLLM device plan: {visible_gpus} GPUs, layer_counts={device_plan.layer_counts}"
        )
    elif args.device_map == "auto":
        load_kwargs["device_map"] = "auto"

    model = BailingMM2NativeForConditionalGeneration.from_pretrained(
        str(model_directory), **load_kwargs
    )
    if args.device_map == "none":
        model = model.to(device=args.device, dtype=dtype)
    elif device_plan is not None:
        _validate_balanced_placement(model, device_plan, torch)
    return model, processor


def run_generation(
    model,
    processor,
    profile,
    *,
    task: str,
    prompt: str,
    input_image,
    resolution: int,
    sampling,
    seed: int,
    num_layers: int,
    dtype,
) -> List:
    """Run one inference with an already-loaded model and processor."""
    import torch
    from PIL import Image

    resolution = resolve_task_resolution(task, resolution)

    messages = _build_messages(task, prompt, input_image)
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, video_inputs, _ = processor.process_vision_info(messages)

    reference_image = None
    if input_image is not None:
        reference_mode = "RGB" if profile.vae_input_channels == 3 else "RGBA"
        reference_image = Image.open(input_image).convert(reference_mode)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        image_gen_highres=resolution,
        image_gen_ref_images=reference_image,
        image_gen_input_channels=profile.vae_input_channels,
    )
    input_device = _model_input_device(model)
    inputs = inputs.to(input_device)
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            inputs[key] = value.to(dtype=dtype)

    output = model.generate(
        **inputs,
        image_gen=True,
        image_gen_task=task,
        image_gen_seed=seed,
        image_gen_steps=sampling.steps,
        image_gen_cfg=sampling.cfg,
        num_frames_per_prompt=num_layers+1 if task == "layer-decompose" else num_layers,
    )
    images = _normalize_outputs(output)
    expected_outputs = num_layers + 1 if task == "layer-decompose" else 1
    if len(images) != expected_outputs:
        raise RuntimeError(
            f"expected {expected_outputs} output image(s), got {len(images)}"
        )
    return images


if __name__ == "__main__":
    main()
