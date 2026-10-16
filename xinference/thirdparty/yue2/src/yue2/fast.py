"""Optional vLLM AR worker; base imports never require vLLM or Triton.

The standard Qwen3 AR subset is derived from the full MoT checkpoint into a
content-addressed cache. A separate process retains its engine between ABC
and semantic generation. close_vllm() releases the complete GPU process group
before NAR/VAE. The public pipeline currently submits one request at a time.
"""
import contextlib
import dataclasses
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import weakref

from .protocol import ABC_END, CODEC_OFFSET, CODEC_SIZE, CONTEXT, EOD, MUSIC_END, Sampling
from .storage import identity, model_identity, sha256_file, write_json

WIRE = "YUE2_FAST\t"
PENALTY, WINDOW = "yue2_window_penalty", "yue2_penalty_window"


def ar_keys(config):
    names = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for index in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{index}."
        names.update(prefix + name + ".weight" for name in (
            "input_layernorm", "post_attention_layernorm", "self_attn.q_proj",
            "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "self_attn.q_norm", "self_attn.k_norm", "mlp.gate_proj",
            "mlp.up_proj", "mlp.down_proj"))
    return names


def qwen_config(config):
    required = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "vocab_size", "rms_norm_eps", "rope_theta")
    output = {name: config[name] for name in required}
    output.update(architectures=["Qwen3ForCausalLM"], model_type="qwen3", hidden_act="silu",
                  max_position_embeddings=config.get("max_position_embeddings", CONTEXT),
                  tie_word_embeddings=False, attention_bias=False, attention_dropout=0.,
                  use_sliding_window=False, sliding_window=None, max_window_layers=0,
                  torch_dtype="bfloat16")
    return output


def derive_ar_checkpoint(model_dir, cache_dir=None):
    """Extract the exact AR tensors; checksum both source and cached derivative."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    model_dir = Path(model_dir)
    source = model_identity(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    converted = qwen_config(config)
    stamp = {"schema": 1, "source": source, "config": converted}
    key = identity(stamp)
    if cache_dir is None:
        base = Path(os.environ.get("YUE2_CACHE", os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")))
        cache_dir = base / "yue2-ar"
    parent = Path(cache_dir)
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / key
    # Linux is the supported vLLM platform; lock prevents simultaneous writers.
    import fcntl
    with (parent / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            manifest = json.loads((target / "derivation.json").read_text())
            if manifest.get("identity") != key or manifest.get("source") != source:
                raise ValueError("AR cache provenance mismatch; remove this derived cache explicitly")
            for name, digest in manifest["sha256"].items():
                if sha256_file(target / name) != digest:
                    raise ValueError(f"Corrupt derived AR cache: {name}")
            return target
        with tempfile.TemporaryDirectory(prefix=key + ".", dir=parent) as temporary:
            temp = Path(temporary)
            wanted, tensors = ar_keys(config), {}
            with contextlib.ExitStack() as stack:
                for filename in source["files"]:
                    reader = stack.enter_context(safe_open(model_dir / filename, framework="pt", device="cpu"))
                    for name in wanted.intersection(reader.keys()):
                        if name in tensors:
                            raise ValueError(f"Duplicate source tensor: {name}")
                        tensors[name] = reader.get_tensor(name)
                if set(tensors) != wanted:
                    raise ValueError(f"Missing AR tensors: {sorted(wanted - set(tensors))}")
                save_file(tensors, temp / "model.safetensors", metadata={"format": "pt"})
            write_json(temp / "config.json", converted)
            write_json(temp / "derivation.json", {**stamp, "identity": key,
                       "tensor_count": len(tensors), "sha256": {
                           name: sha256_file(temp / name) for name in ("model.safetensors", "config.json")}})
            # Rename only our finished directory; leave TemporaryDirectory an
            # empty replacement to clean up normally.
            os.replace(temp, target)
            temp.mkdir()
    return target


def kv_cache_bytes(config, max_sequences=1, context=CONTEXT, block_size=16):
    positions = math.ceil(context / block_size) * block_size
    return positions * max_sequences * config["num_hidden_layers"] * 2 * config["num_key_value_heads"] * config["head_dim"] * 2


def __getattr__(name):
    if name != "WindowedPenalty":
        raise AttributeError(name)
    # vLLM resolves this FQCN only inside its optional worker processes.
    from vllm.v1.sample.logits_processor import LogitsProcessor
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def penalty_kernel(logits, histories, alphas, stride: tl.constexpr, window: tl.constexpr,
                       vocab: tl.constexpr, width: tl.constexpr):
        row = tl.program_id(0)
        pos = tl.arange(0, width)
        ids = tl.load(histories + row * window + pos, pos < window, other=-1)
        equal = (ids[:, None] == ids[None, :]) & (ids[:, None] >= 0)
        count = tl.sum(equal.to(tl.int32), axis=1)
        earlier = tl.sum((equal & (pos[None, :] < pos[:, None])).to(tl.int32), axis=1)
        valid = (pos < window) & (ids >= 0) & (ids < vocab) & (earlier == 0)
        multiplier = tl.load(alphas + row * (window + 1) + count)
        value = tl.load(logits + row * stride + ids, valid, other=0).to(tl.float32)
        tl.store(logits + row * stride + ids,
                 tl.where(value < 0, value * multiplier, value / multiplier), valid)

    class WindowedPenalty(LogitsProcessor):
        @classmethod
        def validate_params(cls, params):
            extra = params.extra_args or {}
            if not 0 < float(extra.get(PENALTY, 1)) or not 1 <= int(extra.get(WINDOW, 50)) <= 100:
                raise ValueError("Invalid window penalty")

        def __init__(self, vllm_config, device, is_pin_memory):
            self.device, self.states = device, {}
            self.capacity = vllm_config.scheduler_config.max_num_seqs
            self.window = 100
            self.host = torch.full((self.capacity, self.window), -1, dtype=torch.long, pin_memory=is_pin_memory)
            self.histories = torch.empty_like(self.host, device=device)
            self.alphas = torch.ones((self.capacity, self.window + 1), dtype=torch.float32, device=device)
            self.lookup = {}

        def is_argmax_invariant(self):
            return False

        def update_state(self, update):
            if update is None:
                return
            for index in update.removed:
                self.states.pop(index, None)
            for index, params, prompt, output in update.added:
                extra = params.extra_args or {}
                self.states[index] = (output, float(extra.get(PENALTY, 1)), int(extra.get(WINDOW, 50)))
                penalty = self.states[index][1]
                if penalty not in self.lookup:
                    self.lookup[penalty] = torch.pow(torch.tensor(penalty, device=self.device, dtype=torch.float32),
                                                    torch.arange(self.window + 1, device=self.device, dtype=torch.float32))
            for a, b, direction in update.moved:
                if direction.name == "SWAP":
                    va, vb = self.states.pop(a, None), self.states.pop(b, None)
                    if va is not None:
                        self.states[b] = va
                    if vb is not None:
                        self.states[a] = vb
                else:
                    self.states.pop(b, None)
                    if a in self.states:
                        self.states[b] = self.states.pop(a)
            for index, (_, penalty, _) in self.states.items():
                self.alphas[index].copy_(self.lookup[penalty])

        def apply(self, logits):
            if logits.device.type == "cuda":
                n = len(logits)
                self.host[:n].fill_(-1)
                host = self.host.numpy()
                for row, (history, _, window) in self.states.items():
                    recent = history[-window:]
                    if recent and row < n:
                        host[row, :len(recent)] = recent
                self.histories[:n].copy_(self.host[:n], non_blocking=True)
                penalty_kernel[(n,)](logits, self.histories, self.alphas, logits.stride(0),
                                     self.window, logits.shape[-1], 128, num_warps=4)
                return logits
            # CPU oracle for focused state/penalty tests without a GPU.
            for row, (history, penalty, window) in self.states.items():
                if row >= len(logits) or not history or penalty == 1:
                    continue
                ids, counts = torch.tensor(history[-window:], device=logits.device).unique(return_counts=True)
                multiplier = torch.pow(torch.tensor(penalty, device=logits.device, dtype=torch.float32), counts.float())
                values = logits[row, ids].float()
                logits[row, ids] = torch.where(values < 0, values * multiplier, values / multiplier).to(logits.dtype)
            return logits

    WindowedPenalty.__module__ = __name__
    WindowedPenalty.__qualname__ = "WindowedPenalty"
    globals()[name] = WindowedPenalty
    return WindowedPenalty


def _stop_process(process):
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


class _Worker:
    def __init__(self, pipe):
        self.lock = threading.Lock()
        self.temp = tempfile.TemporaryDirectory(prefix="yue2-vllm-")
        self.log_path = Path(self.temp.name) / "worker.log"
        self.log = self.log_path.open("w+")
        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        self.process = subprocess.Popen([sys.executable, "-m", "yue2.fast", "--worker"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                         bufsize=0, env=env, start_new_session=True)
        self.finalizer = weakref.finalize(self, _stop_process, self.process)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.buffer = b""
        self._send({"model_dir": str(pipe.model_dir), "device": str(pipe.device),
                    "memory_budget_gib": pipe.memory_budget_gib})

    def _send(self, value):
        self.process.stdin.write((json.dumps(value) + "\n").encode())
        self.process.stdin.flush()

    def request(self, payload, cancelled=None, on_token=None):
        with self.lock:
            self._send(payload)
            while True:
                if cancelled is not None and cancelled():
                    self.close()
                    raise InterruptedError("Cancelled during vLLM AR generation")
                if self.selector.select(timeout=.1):
                    block = os.read(self.process.stdout.fileno(), 65536)
                    if not block:
                        break
                    self.buffer += block
                    while b"\n" in self.buffer:
                        raw, self.buffer = self.buffer.split(b"\n", 1)
                        line = raw.decode(errors="replace")
                        if not line.startswith(WIRE):
                            self.log.write(line + "\n")
                            continue
                        event = json.loads(line[len(WIRE):])
                        if event.get("error"):
                            self.close()
                            raise RuntimeError(event["error"])
                        if event.get("event") == "token" and on_token is not None:
                            on_token(payload["phase"], event["token"])
                        if event.get("event") == "result":
                            return event["ids"], event["timing"], event["truncated"]
                if self.process.poll() is not None:
                    break
            self.log.flush()
            details = self.log_path.read_text(errors="replace")[-12000:]
            self.close()
            raise RuntimeError(f"vLLM worker exited without a result: {details}")

    def close(self):
        _stop_process(self.process)
        self.finalizer.detach()
        self.selector.close()
        for stream in (self.process.stdin, self.process.stdout, self.log):
            if stream is not None and not stream.closed:
                stream.close()
        self.temp.cleanup()


def close_vllm(pipe):
    worker = getattr(pipe, "_vllm_worker", None)
    if worker is not None:
        worker.close()
        pipe._vllm_worker = None


def generate_vllm(pipe, prefix, sampling, seed, phase, negative=None, cfg_scale=1.,
                  legacy_off=False, cancelled=None, on_token=None):
    if phase not in {"abc", "semantic"}:
        raise ValueError("phase must be abc or semantic")
    if len(prefix) + sampling.max_tokens > CONTEXT:
        raise ValueError("Prefix plus requested generation budget exceeds context")
    reason = None
    if cfg_scale != 1.:
        reason = "paired_cfg_uses_torch"
    elif legacy_off:
        reason = "historical_off_sampling_uses_torch"
    elif pipe.device.type != "cuda":
        reason = "vllm_requires_cuda"
    elif pipe.quantization != "none":
        reason = "experimental_fp8_uses_torch"
    if reason:
        from .sampling import generate_tokens
        close_vllm(pipe)
        ids, timing, truncated = generate_tokens(pipe._load_model(), prefix, sampling, seed, phase,
                    negative=negative, cfg_scale=cfg_scale, legacy_off=legacy_off,
                    cancelled=cancelled, on_token=on_token)
        timing.update(backend_actual="torch", backend_requested="vllm", fallback_reason=reason)
        return ids, timing, truncated
    if cancelled is not None and cancelled():
        raise InterruptedError("Cancelled before vLLM load")
    if importlib.util.find_spec("vllm") is None:
        raise ImportError("Install the optional CUDA backend with pip install 'yue2-infer[fast]'")
    if getattr(pipe, "_vllm_worker", None) is None:
        import torch
        if pipe._model is not None:
            pipe._model.to("cpu")
        if getattr(pipe, "_vae", None) is not None:
            pipe._vae.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        pipe._vllm_worker = _Worker(pipe)
    payload = {"prefix": list(prefix), "sampling": dataclasses.asdict(sampling),
               "seed": seed, "phase": phase, "stream_tokens": on_token is not None}
    try:
        return pipe._vllm_worker.request(payload, cancelled, on_token)
    except BaseException:
        close_vllm(pipe)
        raise


def _emit(value):
    print(WIRE + json.dumps(value), flush=True)


async def _worker_main():
    import asyncio
    import torch
    setup = json.loads(sys.stdin.readline())
    device = torch.device(setup["device"])
    if device.index is not None:
        # vLLM sees one logical device; respect the parent scheduler's mapping.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",")[device.index] if visible else str(device.index)
    import vllm
    if vllm.__version__ != "0.19.0":
        raise RuntimeError(f"This backend requires vllm==0.19.0, found {vllm.__version__}")
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    derived = derive_ar_checkpoint(setup["model_dir"])
    config = json.loads((derived / "config.json").read_text())
    total = torch.cuda.get_device_properties(0).total_memory
    budget = min(setup["memory_budget_gib"] * 2**30, total)
    kv_bytes = kv_cache_bytes(config)
    weights_bytes = (derived / "model.safetensors").stat().st_size
    if kv_bytes + weights_bytes + 3 * 2**30 > budget:
        raise MemoryError("Requested vLLM memory budget cannot fit full-context BF16 AR+KV+3GiB reserve")
    load_start = time.perf_counter()
    args = AsyncEngineArgs(model=str(derived), skip_tokenizer_init=True, dtype="bfloat16",
                           max_model_len=CONTEXT, max_num_seqs=1, max_num_batched_tokens=2048,
                           enable_chunked_prefill=True, enable_prefix_caching=True,
                           kv_cache_memory_bytes=kv_bytes,
                           gpu_memory_utilization=min(.9, (budget - 2 * 2**30) / total),
                           logits_processors=["yue2.fast:WindowedPenalty"], disable_log_stats=True)
    engine = AsyncLLM.from_engine_args(args)
    load_seconds = time.perf_counter() - load_start
    try:
        while line := await asyncio.to_thread(sys.stdin.readline):
            request = json.loads(line)
            sampling = Sampling(**request["sampling"])
            abc = request["phase"] == "abc"
            end = ABC_END if abc else MUSIC_END
            allowed = list(range(EOD)) + [end] if abc else list(range(CODEC_OFFSET, CODEC_OFFSET + CODEC_SIZE)) + [end]
            params = SamplingParams(temperature=sampling.temperature, top_p=sampling.top_p,
                       top_k=sampling.top_k, repetition_penalty=1., min_tokens=sampling.min_tokens,
                       max_tokens=sampling.max_tokens, seed=request["seed"], stop_token_ids=[end],
                       allowed_token_ids=allowed, detokenize=False,
                       extra_args={PENALTY: sampling.repetition_penalty, WINDOW: sampling.penalty_window})
            start, first, output, sent = time.perf_counter(), None, None, 0
            async for result in engine.generate({"prompt_token_ids": request["prefix"]}, params, str(uuid.uuid4())):
                output = result.outputs[0]
                if output.token_ids and first is None:
                    first = time.perf_counter() - start
                if request["stream_tokens"]:
                    for token in output.token_ids[sent:]:
                        _emit({"event": "token", "token": int(token)})
                    sent = len(output.token_ids)
            if output is None:
                raise RuntimeError("vLLM returned no output")
            ids = list(output.token_ids)
            ended = bool(ids and ids[-1] == end) or output.stop_reason == end
            if ids and ids[-1] == end:
                ids.pop()
            elapsed = time.perf_counter() - start
            count = len(ids) + int(ended)
            _emit({"event": "result", "ids": ids, "truncated": not ended,
                   "timing": {"seconds": elapsed, "ttft_seconds": first,
                       "output_tokens": count, "content_tokens": len(ids), "output_tps": count / elapsed,
                       "prefix_tokens": len(request["prefix"]), "cfg_branches": 1,
                       "backend_actual": "vllm", "backend_requested": "vllm",
                       "engine_load_seconds": load_seconds, "kv_cache_memory_bytes": kv_bytes,
                       "ar_derivation_identity": derived.name, "max_num_seqs": 1}})
            load_seconds = 0.
    finally:
        engine.shutdown()


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("Internal worker: use YuE2Pipeline(backend='vllm')")
    import asyncio
    try:
        asyncio.run(_worker_main())
    except BaseException as error:
        _emit({"error": f"{type(error).__name__}: {error}"})
        raise
