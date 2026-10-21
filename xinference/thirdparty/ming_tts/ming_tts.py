# Copyright (c) Ant Group. All rights reserved.
"""High-level inference pipeline for Ming-3.0-tts.

``MingTTS.synthesize`` is the single entry point for every task: zero-shot
cloning, multilingual, attribute/emotion/dialect control, built-in IP
voices, voice design and free-form instructions.
"""

from __future__ import annotations

import os

import torch
import torchaudio

from . import tts_tasks
from .frontend.number_en import normalize_numbers
from .frontend.text_segment_cut import cut_text_by_semantic_length, is_chinese
from .frontend.toolkit import (add_period, normalize_text_kit, preprosess_text,
                              protect_paralinguistic_tags)
from .modeling_bailingmm import BailingMMNativeForConditionalGeneration
from .spkemb_extractor import SpkembExtractor

SPEAKER_EMBEDDING_DIM = 192

# Trailing-silence trim, matching the reference implementation's defaults.
SILENCE_FRAME_SECONDS = 0.1
SILENCE_THRESHOLD = 1e-3
SILENCE_TAIL_SECONDS = 0.3


def trim_trailing_silence(
    speech,
    sample_rate,
    threshold=SILENCE_THRESHOLD,
    tail_seconds=SILENCE_TAIL_SECONDS,
):
    """Cut the trailing silence, keeping a fixed tail.

    The model keeps emitting near-silent frames after the utterance ends, and
    how many it emits varies per fragment. This is the non-streaming
    equivalent of the reference implementation's silence holder: scan 0.1 s
    frames backwards for the last one above ``threshold`` and keep
    ``tail_seconds`` beyond it.
    """
    if speech.numel() == 0:
        return speech

    frame = int(sample_rate * SILENCE_FRAME_SECONDS)
    tail = int(sample_rate * tail_seconds)
    if frame <= 0 or speech.shape[-1] < frame:
        return speech[..., :tail]

    # Drop the sub-frame remainder, as the reference does before unfolding.
    num_frames = (speech.shape[-1] - frame) // frame + 1
    speech = speech[..., :num_frames * frame]

    scores = speech.unfold(-1, frame, frame).abs().mean(dim=-1)
    scores = scores.mean(dim=list(range(scores.dim() - 1)))

    index = int(scores.shape[0]) - 1
    while index >= 0 and scores[index] <= threshold:
        index -= 1
    if index < 0:
        # Silent throughout; emit the fixed tail rather than nothing.
        return speech[..., :tail]
    return speech[..., :index * frame + frame + tail]


class MingTTS:
    def __init__(self, model_path, device="cuda:0", text_normalization=True,
                 cache_prompts=True):
        from transformers import AutoTokenizer

        if os.path.isdir(model_path):
            local_path = model_path
        else:
            from huggingface_hub import snapshot_download
            local_path = snapshot_download(repo_id=model_path)

        self.device = device
        self.local_path = local_path
        self.model = BailingMMNativeForConditionalGeneration.from_pretrained(
            local_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).eval().to(torch.bfloat16).to(device)
        # The setter validates the special tokens against the trained layout.
        self.model.tokenizer = AutoTokenizer.from_pretrained(
            local_path, trust_remote_code=True,
        )

        self.sample_rate = self.model.config.audio_tokenizer_config.sample_rate
        self.patch_size = self.model.patch_size
        self.spkemb_extractor = SpkembExtractor(
            os.path.join(local_path, "campplus.onnx")
        )
        self.text_normalization = text_normalization
        self._normalizer = None
        self.cache_prompts = cache_prompts
        self._prompt_cache = {}

    # -- text -----------------------------------------------------------
    @property
    def normalizer(self):
        """Load the FST normalizer on first use.

        It pulls in ``pynini``, which is awkward to install on some
        platforms, so the rest of the pipeline stays importable without it.
        """
        if self._normalizer is None:
            from .frontend.talker_tn.talker_tn import TalkerTN
            self._normalizer = TalkerTN()
        return self._normalizer

    def _normalize_fragment(self, text):
        if not is_chinese(text):
            text = normalize_numbers(text)
        text = self.normalizer.normalize(text)
        text = normalize_text_kit(text)
        # TN can surface a leading comma when it rewrites a leading token.
        if text[:1] == "，":
            text = text[1:]
        return text

    def prepare_text(self, text, taskname, language_id, max_length):
        """Split into synthesis fragments and normalize the ones that need it."""
        if tts_tasks.preserve_multilingual_text(language_id):
            # The FST normalizer reads any CJK ideograph as Chinese, which
            # would strip the kana out of Japanese. Non-zh/en text is passed
            # through unchanged.
            fragments = cut_text_by_semantic_length(
                preprosess_text(text).strip(), max_length)["fragments"]
            candidates = [(fragment, False) for fragment in fragments]
        elif taskname == "STYLE":
            # Voice design and free-form conditioning apply to the utterance
            # as a whole, so the text is not split.
            candidates = [(add_period(preprosess_text(text)), True)]
        else:
            fragments = cut_text_by_semantic_length(
                add_period(preprosess_text(text)), max_length)["fragments"]
            candidates = [(fragment, True) for fragment in fragments]

        prepared = []
        for fragment, apply_tn in candidates:
            if not fragment.strip():
                continue
            if apply_tn and self.text_normalization:
                # Route around inline paralinguistic tags: normalization
                # would rewrite "[Laughter]" into "，Laughter，".
                fragment = protect_paralinguistic_tags(
                    fragment, self._normalize_fragment)
            if fragment.strip():
                prepared.append(fragment)
        if not prepared:
            raise ValueError(f"text produced no synthesizable fragments: {text!r}")
        return prepared

    # -- audio prompt -----------------------------------------------------
    def _pad_align(self):
        encoder = self.model.audio.encoder
        return encoder.hop_size * max(1, encoder.patch_size) * self.patch_size

    def prepare_prompt(self, prompt_wav_path, use_spk_emb):
        """Return the prompt's VAE latents and, if requested, its speaker embeddings.

        With ``cache_prompts`` a prompt is encoded only the first time it is
        used, as the reference runtime does with its registered prompts.
        Encoding samples the VAE posterior, so under a fixed seed the first
        use of a prompt and every later one draw different noise afterwards.
        """
        paths = (prompt_wav_path if isinstance(prompt_wav_path, list)
                 else [prompt_wav_path])
        key = "|".join(paths)
        entry = self._prompt_cache.get(key)
        if entry is None:
            waveform, spk_emb = self._load_prompt(paths)
            entry = (self.model.encode_prompt_latent(waveform), spk_emb)
            if self.cache_prompts:
                self._prompt_cache[key] = entry
        prompt_latent, spk_emb = entry
        return prompt_latent, (spk_emb if use_spk_emb else None)

    def _load_prompt(self, paths):
        waveforms, spk_emb = [], []
        for path in paths:
            waveform, sample_rate = torchaudio.load(path, backend="soundfile")
            if waveform.numel() == 0:
                raise ValueError(f"prompt audio is empty: {path}")
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            speaker_waveform = waveform
            if sample_rate != 16000:
                speaker_waveform = torchaudio.transforms.Resample(
                    sample_rate, 16000)(waveform)
            spk_emb.append(self.spkemb_extractor(speaker_waveform))

            if sample_rate != self.sample_rate:
                waveform = torchaudio.transforms.Resample(
                    sample_rate, self.sample_rate)(waveform)
            waveforms.append(waveform)

        waveform = torch.cat(waveforms, dim=-1)

        # Pad at the front, not the back. The tail of the prompt seeds the
        # latent history for the first decode step, so trailing silence
        # would be what the model continues from.
        align = self._pad_align()
        remainder = waveform.shape[-1] % align
        if remainder:
            padded = torch.zeros(
                (waveform.shape[0], waveform.shape[-1] + align - remainder),
                dtype=waveform.dtype,
            )
            padded[:, -waveform.shape[-1]:] = waveform
            waveform = padded

        return waveform, spk_emb

    def _speaker_embeddings(self, spk_emb_path):
        paths = (spk_emb_path if isinstance(spk_emb_path, list)
                 else [spk_emb_path])
        embeddings = []
        for path in paths:
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict):
                for key in ("spk_emb", "speaker_embedding", "embedding", "emb"):
                    if key in obj:
                        obj = obj[key]
                        break
                else:
                    raise ValueError(
                        f"cannot find a speaker embedding tensor in {path}")
            tensor = obj if torch.is_tensor(obj) else torch.tensor(obj)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[-1] != SPEAKER_EMBEDDING_DIM:
                raise ValueError(
                    f"{path} holds a {tensor.shape[-1]}-dim embedding, "
                    f"expected {SPEAKER_EMBEDDING_DIM}")
            embeddings.append(tensor)
        return embeddings

    # -- generation -------------------------------------------------------
    @torch.inference_mode()
    def synthesize(
        self,
        task,
        text,
        language_id=None,
        instruction=None,
        controls=None,
        instruction_schema=None,
        prompt_wav_path=None,
        prompt_text=None,
        spk_emb_path=None,
        use_spk_emb=False,
        use_zero_spk_emb=False,
        cfg=2.0,
        sigma=0.25,
        temperature=0.0,
        max_length=50,
        max_decode_steps=1000,
        trim_silence=True,
        raw_text=False,
        output_path=None,
    ):
        """Synthesize ``text`` and return a ``(1, samples)`` waveform.

        By default the text is split into fragments of about ``max_length``
        and normalized. ``raw_text=True`` sends it to the model exactly as
        given, in one piece, which is how the published benchmarks were run.
        """
        request = tts_tasks.build_task_generation_kwargs(
            task,
            text=text,
            language_id=language_id,
            instruction=instruction,
            controls=controls,
            instruction_schema=instruction_schema,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            spk_emb_path=spk_emb_path,
            use_spk_emb=use_spk_emb,
            use_zero_spk_emb=use_zero_spk_emb,
            cfg=cfg,
            sigma=sigma,
            temperature=temperature,
        )

        prompt_latent, spk_emb = None, None
        if request["use_zero_spk_emb"]:
            spk_emb = [torch.zeros(1, SPEAKER_EMBEDDING_DIM)]
        elif spk_emb_path is not None:
            spk_emb = self._speaker_embeddings(spk_emb_path) if use_spk_emb else None
        elif request["prompt_wav_path"] is not None:
            prompt_latent, spk_emb = self.prepare_prompt(
                request["prompt_wav_path"], request["use_spk_emb"])

        # An acoustic prompt is only usable together with its transcript.
        prompt_text = request["prompt_text"] if prompt_latent is not None else None
        if prompt_text is None:
            prompt_latent = None

        if raw_text:
            fragments = [request["text"]]
        else:
            fragments = self.prepare_text(
                request["text"], request["taskname"], request["text_language"],
                max_length,
            )

        chunks = []
        for fragment in fragments:
            chunk = self.model.generate(
                prompt=request["prompt"],
                text=fragment,
                spk_emb=None if spk_emb is None else [e.clone() for e in spk_emb],
                instruction=request["instruction"],
                prompt_latent=prompt_latent,
                prompt_text=prompt_text,
                max_decode_steps=max_decode_steps,
                cfg=request["cfg"],
                sigma=request["sigma"],
                temperature=request["temperature"],
                use_zero_spk_emb=request["use_zero_spk_emb"],
            )
            # Applied per fragment, not once at the end: the reference
            # implementation trims inside each synthesis job, so on split text
            # the trailing silence of every fragment is bounded before the
            # pieces are joined.
            if trim_silence:
                chunk = trim_trailing_silence(chunk, self.sample_rate)
            chunks.append(chunk)

        waveform = torch.cat(chunks, dim=-1) if len(chunks) > 1 else chunks[0]
        if output_path is not None:
            directory = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(directory, exist_ok=True)
            torchaudio.save(output_path, waveform, sample_rate=self.sample_rate)
        return waveform


__all__ = ["MingTTS", "trim_trailing_silence"]
