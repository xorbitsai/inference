"""Small CLI sharing the Python pipeline's defaults and artifact protocol."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from .storage import write_json, verify_result, identity


def kit_root():
    return Path(os.environ.get("YUE2_KIT", Path(__file__).resolve().parents[2]))


def model_paths(args):
    root = kit_root()
    local_model = root / "models/YuE2-3B"
    local_vae = root / "models" / ("YuE2-Vae-legacy" if args.vae == "legacy" else "YuE2-Vae")
    model = args.model or (str(local_model) if local_model.exists() else "m-a-p/YuE2-3B")
    vae = args.vae if args.vae not in {"standard", "legacy"} else (
        str(local_vae) if local_vae.exists() else "m-a-p/" + local_vae.name)
    return model, vae


def get_pipe(args):
    from .pipeline import YuE2Pipeline
    from .protocol import GenerationConfig
    model, vae = model_paths(args)
    config = GenerationConfig.from_dict(json.loads(Path(args.config).read_text())) if args.config else None
    return YuE2Pipeline.from_pretrained(model, vae=vae, revision=args.revision,
             vae_revision=args.vae_revision, device=args.device, memory_budget_gib=args.budget,
             backend=args.backend, quantization=args.quantization, offload_ar=args.offload_ar,
             local_files_only=args.offline, generation_config=config,
             vae_core_frames=512 if args.budget <= 12 else 1024,
             progress=not getattr(args, "quiet", False))


def request_kwargs(data, base=Path.cwd()):
    allowed = {"style", "tags", "lyrics", "cot", "seed", "abc", "cfg_scale", "id", "abc_sampling", "semantic_sampling"}
    metadata = {"lang", "eval_index", "clip_id", "prompt"}
    unknown = set(data) - allowed - metadata - {"abc_path"}
    if unknown:
        raise ValueError(f"Unknown request fields: {sorted(unknown)}")
    result = {k: v for k, v in data.items() if k in allowed}
    if "abc_path" in data:
        if data.get("abc") is not None:
            raise ValueError("Pass abc or abc_path, not both")
        result["abc"] = (base / data["abc_path"]).read_bytes().decode("utf-8")
    if data.get("prompt") is not None:
        from .protocol import SongRequest
        request = SongRequest(style=result.get("style", result.get("tags")), lyrics=result["lyrics"],
                              cot=result.get("cot", "full"))
        if data["prompt"] != request.text():
            raise ValueError("Historical literal prompt does not match native instruction/style/lyrics")
    return result


def doctor(args):
    import torch
    from .storage import model_identity
    model, vae = model_paths(args)
    versions = {}
    for package in ("torch", "transformers", "huggingface-hub", "safetensors", "tiktoken", "soundfile"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    devices = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        devices.append({"id": i, "name": p.name, "memory_gib": p.total_memory / 2**30,
                        "compute_capability": [p.major, p.minor]})
    report = {"dependencies_ready": all(versions.values()), "versions": versions,
              "cuda": devices, "mps_available": torch.backends.mps.is_available(),
              "model": model, "vae": vae, "default_cot": "full", "default_cfg": {"full": 1., "melody": 1., "off": 1.01},
              "validated": False, "note": "Environment readiness is not quality or real-24GB acceptance."}
    if args.verify_hashes:
        from .storage import resolve_model
        report["weights"] = {"model": model_identity(resolve_model(model, local_files_only=args.offline)),
                             "vae": model_identity(resolve_model(vae, local_files_only=args.offline))}
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["dependencies_ready"] else 1


def generate(args):
    if args.request:
        path = Path(args.request)
        data = json.loads(path.read_text())
        base = path.parent
    else:
        data, base = {}, Path.cwd()
    for key in ("id", "style", "cot", "seed", "cfg_scale"):
        value = getattr(args, key)
        if value is not None:
            data[key] = value
    if args.lyrics_file:
        data["lyrics"] = Path(args.lyrics_file).read_bytes().decode("utf-8")
    elif args.lyrics is not None:
        data["lyrics"] = args.lyrics
    if args.abc_file:
        data["abc"] = Path(args.abc_file).read_bytes().decode("utf-8")
        data.pop("abc_path", None)
    if not data:
        data = {"id": "first_song", "style": "Mandarin, warm piano, acoustic pop, female vocal",
                "lyrics": "[Verse]\n晚风轻轻吹过窗前\n你留下的笑还在昨天\n[Chorus]\n让这首歌陪你走远\n把所有想念唱成明天"}
    kwargs = request_kwargs(data, base)
    pipe = get_pipe(args)
    directory = Path(args.output or "runs/default") / kwargs.get("id", "song")
    if args.resume and (directory / "result.json").exists():
        req_kwargs = {k: v for k, v in kwargs.items() if k not in {"abc_sampling", "semantic_sampling"}}
        request = pipe._request(**req_kwargs)
        config = pipe.effective_config(request, kwargs.get("abc_sampling"), kwargs.get("semantic_sampling"))
        expected = identity({"request": request.to_dict(), "config": config, "weights": pipe.weights})
        result = verify_result(directory, expected)
        print(json.dumps({"resumed": True, "result": str(directory / "result.json"), "truncated": result["truncated"]}))
        return 0
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Nonempty output {directory}; use --resume or a new output directory")
    directory.mkdir(parents=True, exist_ok=True)
    try:
        if args.stage == "plan":
            kwargs.pop("semantic_sampling", None)
            plan = pipe.plan(**kwargs)
            plan.save(directory)
            print(json.dumps({"stage": "plan", "output": str(directory), "truncated": plan.truncated}))
        else:
            result = pipe(**kwargs)
            result.save_artifacts(directory)
            print(json.dumps({"status": "complete", "output": str(directory), "truncated": result.truncated,
                              "seconds": result.timing["e2e_seconds"]}))
        return 0
    except BaseException as exc:
        write_json(directory / "failure.json", {"status": "failed", "type": type(exc).__name__, "reason": str(exc),
                                                "request": data})
        raise
    finally:
        pipe.close()


def batch(args):
    if args.concurrency != 1:
        raise ValueError("The minimal torch pipeline currently supports concurrency=1; do not share a pipeline concurrently")
    path = Path(args.input)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [r.get("id") for r in rows]
    if any(x is None for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("Every batch request must have a unique id")
    pipe = get_pipe(args)
    output = Path(args.output or "runs/batch")
    receipts, failures = [], 0
    try:
        for index, row in enumerate(rows, 1):
            if not args.quiet:
                print(f"Song {index}/{len(rows)}: {row['id']}", file=sys.stderr, flush=True)
            kwargs = request_kwargs(row, path.parent)
            if args.cot is not None:
                kwargs["cot"] = args.cot
            directory = output / row["id"]
            try:
                request = pipe._request(**{k:v for k,v in kwargs.items() if k not in {"abc_sampling", "semantic_sampling"}})
                cfg = pipe.effective_config(request, kwargs.get("abc_sampling"), kwargs.get("semantic_sampling"))
                expected = identity({"request": request.to_dict(), "config": cfg, "weights": pipe.weights})
                if args.resume and (directory / "result.json").exists():
                    receipt = verify_result(directory, expected)
                else:
                    if directory.exists() and any(directory.iterdir()):
                        raise FileExistsError("Nonempty request output; use --resume or a new run")
                    receipt = pipe(**kwargs).save_artifacts(directory)
                receipts.append({"id": row["id"], "status": "complete", "identity": receipt["identity"]})
            except Exception as exc:
                failures += 1
                failure = {"id": row["id"], "status": "failed", "reason": str(exc), "type": type(exc).__name__}
                write_json(directory / "failure.json", failure)
                receipts.append(failure)
            write_json(output / "batch.json", {"complete": len(receipts) == len(rows) and not failures,
                       "expected": len(rows), "failed": failures, "results": receipts})
    finally:
        pipe.close()
    return int(failures > 0)


def parser():
    p = argparse.ArgumentParser(description="YuE2: style + lyrics → symbolic plan → song")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("doctor", "generate", "batch"):
        q = sub.add_parser(name)
        q.add_argument("--model")
        q.add_argument("--vae", default="standard", help="standard (listening), legacy (paper evaluation), local path or HF repo")
        q.add_argument("--revision")
        q.add_argument("--vae-revision")
        q.add_argument("--device", default="auto")
        q.add_argument("--budget", type=float, default=24)
        q.add_argument("--backend", choices=("torch", "torch-eager", "vllm"), default="torch")
        q.add_argument("--quantization", choices=("none", "fp8"), default="none")
        q.add_argument("--offload-ar", action="store_true")
        q.add_argument("--offline", action="store_true")
        q.add_argument("--config")
        q.add_argument("--output")
        if name == "doctor":
            q.add_argument("--verify-hashes", action="store_true")
        else:
            q.add_argument("--cot", choices=("full", "melody", "off"))
            q.add_argument("--resume", action="store_true")
            q.add_argument("--quiet", "--no-progress", action="store_true",
                           help="Hide YuE2 progress on stderr; keep result output on stdout")
            if name == "batch":
                q.add_argument("--input", required=True)
                q.add_argument("--concurrency", type=int, default=1)
            else:
                q.add_argument("--request")
                q.add_argument("--id")
                q.add_argument("--style")
                q.add_argument("--lyrics")
                q.add_argument("--lyrics-file")
                q.add_argument("--abc-file")
                q.add_argument("--seed", type=int)
                q.add_argument("--cfg-scale", type=float)
                q.add_argument("--stage", choices=("plan", "audio"), default="audio")
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parser().parse_args(argv)
    return {"doctor": doctor, "generate": generate, "batch": batch}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
