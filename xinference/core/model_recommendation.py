"""Read-only launch recommendations; no capacity or launch guarantee."""

import re
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional, Union

from .._compat import BaseModel, Field, validator


class RecommendationModelNotFound(LookupError):
    """All workers confirmed the model is absent."""


class RecommendationConstraints(BaseModel):
    model_size_in_billions: Optional[Union[int, str]] = None
    worker_ip: Optional[str] = None
    enable_virtual_env: Optional[bool] = None
    n_gpu: Optional[Union[int, Literal["auto"]]] = "auto"
    gpu_idx: Optional[Union[int, List[int]]] = None

    class Config:
        extra = "forbid"
        smart_union = True

    @validator("model_size_in_billions", pre=True)
    def validate_size(cls, value):
        if (
            type(value) not in (int, str)
            or not re.fullmatch(r"\d+(?:[._]\d+)?", str(value))
            or size_value(value) <= 0
        ):
            raise ValueError(
                "model_size_in_billions must be a positive integer or decimal string"
            )
        return value

    @validator("worker_ip", pre=True)
    def validate_worker(cls, value):
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError("worker_ip must be a non-empty host or actor address")
        return value

    @validator("enable_virtual_env", pre=True)
    def validate_venv(cls, value):
        if type(value) is not bool:
            raise ValueError("enable_virtual_env must be a boolean")
        return value

    @validator("n_gpu", pre=True)
    def validate_gpu_count(cls, value):
        if (
            value is not None
            and value != "auto"
            and not (type(value) is int and value > 0)
        ):
            raise ValueError("n_gpu must be a positive integer, 'auto', or null (CPU)")
        return value

    @validator("gpu_idx", pre=True)
    def validate_gpu_indices(cls, value):
        if value is None:
            return value
        indices = [value] if type(value) is int else value
        if (
            not isinstance(indices, list)
            or not indices
            or any(type(i) is not int or i < 0 for i in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError(
                "gpu_idx must be a non-negative integer or non-empty unique list"
            )
        return value


class ModelRecommendationRequest(BaseModel):
    model_name: str
    model_type: Literal["LLM", "embedding", "rerank", "audio"] = "LLM"
    constraints: RecommendationConstraints = Field(
        default_factory=RecommendationConstraints
    )

    class Config:
        extra = "forbid"

    @validator("constraints")
    def validate_model_constraints(cls, value, values):
        if (
            values.get("model_type", "LLM") != "LLM"
            and value.model_size_in_billions is not None
        ):
            raise ValueError(
                "model_size_in_billions is only supported for LLM recommendations"
            )
        return value

    @validator("model_name", pre=True)
    def validate_name(cls, value):
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError("model_name must be a non-empty string")
        return value


def size_value(size: Union[int, str]) -> Decimal:
    return Decimal(str(size).replace("_", "."))


def message(code: str, text: str) -> Dict[str, str]:
    return {"code": code, "message": text}


def spec_key(param: Dict[str, Any], quantization: str) -> tuple:
    return (
        param["model_format"],
        size_value(param["model_size_in_billions"]),
        quantization.lower(),
    )


def select_recommendation(
    request: ModelRecommendationRequest, workers: List[dict], warnings: List[dict]
) -> dict:
    if request.model_type != "LLM":
        return select_non_llm_recommendation(request, workers, warnings)
    constraints = request.constraints
    candidates = []
    for worker in workers:
        indices = constraints.gpu_idx
        indices = [indices] if isinstance(indices, int) else indices
        devices = worker["gpu_indices"]
        host_gpu_count = worker["gpu_count"]
        if isinstance(constraints.n_gpu, int) and constraints.n_gpu > host_gpu_count:
            continue
        if indices is not None:
            if not set(indices) <= set(devices):
                continue
        elif isinstance(constraints.n_gpu, int) and constraints.n_gpu > len(devices):
            continue
        elif constraints.n_gpu == "auto" and host_gpu_count > 0 and not devices:
            # Launch would try to allocate one GPU and fail; do not silently
            # change an automatic GPU request to an explicit CPU launch.
            continue
        cpu = indices is None and (
            constraints.n_gpu is None or (not devices and worker["device"] != "mps")
        )
        device = "cpu" if cpu else worker["device"]
        order = (
            ["mlx", "llama.cpp", "transformers"]
            if device == "mps"
            else (
                ["vllm", "sglang", "transformers", "llama.cpp"]
                if device == "cuda" and worker["platform"] == "Linux"
                else (
                    ["llama.cpp", "transformers"]
                    if device == "cpu"
                    else ["transformers", "llama.cpp"]
                )
            )
        )
        ready = {
            (engine, spec_key(p, q))
            for engine, params in worker["installed_engines"].items()
            if isinstance(params, list)
            for p in params
            for q in p["quantizations"]
        }
        for engine, params in worker["engines"].items():
            if not isinstance(params, list):
                continue
            if device == "cpu" and engine.lower() not in ("transformers", "llama.cpp"):
                continue
            if engine.lower() == "mlx" and device != "mps":
                continue
            for param in params:
                size = size_value(param["model_size_in_billions"])
                if (
                    constraints.model_size_in_billions is not None
                    and size != size_value(constraints.model_size_in_billions)
                ):
                    continue
                for quant in param["quantizations"]:
                    if (
                        device in ("cpu", "mps")
                        and engine.lower() == "transformers"
                        and (
                            param["model_format"] != "pytorch"
                            or quant.lower() != "none"
                        )
                    ):
                        continue
                    if (
                        device == "cpu"
                        and engine.lower() == "llama.cpp"
                        and param["model_format"] != "ggufv2"
                    ):
                        continue
                    key = spec_key(param, quant)
                    if key not in worker["launch_specs"]:
                        continue
                    cached = key in worker["cached_specs"]
                    installed = (engine, key) in ready
                    if not worker["enable_virtual_env"] and not installed:
                        continue
                    quant_order = [
                        "q4_k_m",
                        "4bit",
                        "int4",
                        "q4_0",
                        "q8_0",
                        "8bit",
                        "int8",
                        "none",
                        "bf16",
                        "fp16",
                    ]
                    qrank = (
                        quant_order.index(quant.lower())
                        if quant.lower() in quant_order
                        else len(quant_order)
                    )
                    erank = (
                        order.index(engine.lower())
                        if engine.lower() in order
                        else len(order)
                    )
                    rank = (
                        size,
                        not installed,
                        erank,
                        qrank,
                        not cached,
                        engine,
                        param["model_format"],
                        quant,
                        worker["worker_ip"],
                    )
                    candidates.append(
                        (rank, worker, param, engine, quant, installed, cached)
                    )
    warnings = list(warnings)
    warnings.append(
        message(
            "memory_not_verified",
            "Memory fit and current resource availability are not verified; no resources are reserved.",
        )
    )
    if not candidates:
        return {
            "status": "no_recommendation",
            "config": None,
            "reasons": [
                message(
                    "no_viable_candidates",
                    "No worker has a supported configuration satisfying all constraints.",
                )
            ],
            "warnings": warnings,
        }
    _, worker, param, engine, quant, installed, cached = min(
        candidates, key=lambda c: c[0]
    )
    config = {
        "model_engine": engine,
        "model_format": param["model_format"],
        "model_size_in_billions": param["model_size_in_billions"],
        "quantization": quant,
        "worker_ip": worker["worker_ip"],
        "enable_virtual_env": worker["enable_virtual_env"],
    }
    config.update(
        {
            k: v
            for k, v in constraints.dict(exclude_unset=True).items()
            if k in ("n_gpu", "gpu_idx")
        }
    )
    reasons = [
        message(
            "selection_policy",
            "Selected the smallest viable size, then installed readiness, platform engine order, quantization preference, exact worker-local cache, and stable lexical ties.",
        ),
        message(
            "worker_platform",
            f"Selected {worker['worker_ip']}: {worker['platform']}/{worker['device']}.",
        ),
        message(
            "engine_installed" if installed else "virtual_env_candidate",
            (
                "Engine discovery passed in the worker environment."
                if installed
                else "Compatible virtual-environment candidate; dependencies may need installation at launch."
            ),
        ),
    ]
    reasons.append(
        message(
            (
                "size_constraint"
                if constraints.model_size_in_billions is not None
                else "smallest_size_default"
            ),
            (
                "Preserved the requested model size."
                if constraints.model_size_in_billions is not None
                else "Used the smallest viable registered size as a conservative default, not a hardware-fit estimate."
            ),
        )
    )
    if cached:
        reasons.append(
            message(
                "exact_spec_cached",
                "The exact launch spec is cached on the selected worker.",
            )
        )
    if worker["enable_virtual_env"]:
        warnings.append(
            message(
                "virtual_env_not_verified",
                "The launch virtual environment has not been created or checked; launch may install dependencies.",
            )
        )
    if constraints.gpu_idx is not None:
        warnings.append(
            message(
                "gpu_idx_overrides_n_gpu",
                "Launch uses gpu_idx in preference to n_gpu; both supplied values are preserved.",
            )
        )
    elif constraints.n_gpu == "auto":
        warnings.append(
            message(
                "n_gpu_auto",
                "n_gpu='auto' uses the existing launch allocation policy; this recommendation does not calculate a required GPU count.",
            )
        )
    return {
        "status": "recommended",
        "config": config,
        "reasons": reasons,
        "warnings": warnings,
    }


def non_llm_candidates(
    model_type: str, model_name: str, engines: dict, installed: dict
) -> List[dict]:
    """Resolve discovery tuples through the same read-only matchers as launch.

    Cache managers are deliberately not constructed here: their initialization
    can create directories. Cache preference remains LLM-only for now.
    """
    if model_type == "embedding":
        from ..model.embedding.embed_family import match_embedding as match
    elif model_type == "rerank":
        from ..model.rerank.rerank_family import match_rerank as match
    elif model_type == "audio":
        from ..model.audio.core import match_audio
    else:
        raise ValueError(f"Unsupported recommendation model type: {model_type}")
    candidates = []
    for engine, params in engines.items():
        if not isinstance(params, list):
            continue
        for param in params:
            config = {"model_engine": engine}
            for key in ("model_format", "quantization"):
                if param.get(key) is not None:
                    config[key] = param[key]
            try:
                if model_type == "audio":
                    # Audio discovery may have no format or quantization. Match
                    # the default variant, but do not invent hidden form fields.
                    family = match_audio(
                        model_name,
                        model_engine=engine,
                        quantization=param.get("quantization"),
                    )
                    if family.quantization is not None:
                        config["quantization"] = family.quantization
                else:
                    match(
                        model_name, param.get("model_format"), param.get("quantization")
                    )
            except (ValueError, IndexError):
                continue
            ready = installed.get(engine)
            candidates.append(
                {
                    "config": config,
                    "installed": isinstance(ready, list)
                    and any(
                        all(
                            p.get(key) == param.get(key)
                            for key in ("model_format", "quantization")
                        )
                        for p in ready
                    ),
                }
            )
    return candidates


def select_non_llm_recommendation(
    request: ModelRecommendationRequest, workers: List[dict], warnings: List[dict]
) -> dict:
    """Select discovered non-LLM tuples without inventing size or quantization."""
    constraints = request.constraints
    candidates = []
    for worker in workers:
        indices = constraints.gpu_idx
        indices = [indices] if isinstance(indices, int) else indices
        devices = worker["gpu_indices"]
        if indices is not None and not set(indices) <= set(devices):
            continue
        if isinstance(constraints.n_gpu, int) and (
            constraints.n_gpu > worker["gpu_count"]
            or (indices is None and constraints.n_gpu > len(devices))
        ):
            continue
        if (
            indices is None
            and constraints.n_gpu == "auto"
            and worker["gpu_count"] > 0
            and not devices
        ):
            continue
        cpu = indices is None and (
            constraints.n_gpu is None or (not devices and worker["device"] != "mps")
        )
        device = "cpu" if cpu else worker["device"]
        order = (
            ["mlx", "transformers", "pytorch", "diffusers", "vllm"]
            if request.model_type == "audio" and device == "mps"
            else (
                ["transformers", "pytorch", "diffusers", "vllm", "mlx"]
                if request.model_type == "audio"
                else ["sentence_transformers", "flag", "llama.cpp", "vllm"]
            )
        )
        for candidate in worker.get("candidates", []):
            config = candidate["config"]
            engine = config["model_engine"].lower()
            if engine == "mlx" and device != "mps":
                continue
            if engine == "vllm" and (device != "cuda" or worker["platform"] != "Linux"):
                continue
            if not candidate["installed"] and not worker["enable_virtual_env"]:
                continue
            quant = config.get("quantization", "none").lower()
            rank = (
                not candidate["installed"],
                order.index(engine) if engine in order else len(order),
                quant not in ("none", "fp32", "fp16", "bf16"),
                engine,
                config.get("model_format", ""),
                quant,
                worker["worker_ip"],
            )
            candidates.append((rank, worker, candidate))
    warnings = list(warnings) + [
        message(
            "memory_not_verified",
            "Memory fit and current resource availability are not verified; no resources are reserved.",
        )
    ]
    if not candidates:
        return {
            "status": "no_recommendation",
            "config": None,
            "reasons": [
                message(
                    "no_viable_candidates",
                    "No worker has a supported configuration satisfying all constraints.",
                )
            ],
            "warnings": warnings,
        }
    _, worker, candidate = min(candidates, key=lambda item: item[0])
    config = {
        **candidate["config"],
        "worker_ip": worker["worker_ip"],
        "enable_virtual_env": worker["enable_virtual_env"],
        **{
            key: value
            for key, value in constraints.dict(exclude_unset=True).items()
            if key in ("n_gpu", "gpu_idx")
        },
    }
    if worker["enable_virtual_env"]:
        warnings.append(
            message(
                "virtual_env_not_verified",
                "The launch virtual environment has not been created or checked; launch may install dependencies.",
            )
        )
    return {
        "status": "recommended",
        "config": config,
        "reasons": [
            message(
                "selection_policy",
                "Selected an available engine using model-type defaults, then unquantized variants and stable lexical ties. No model size or memory capacity is inferred.",
            ),
            message(
                (
                    "engine_installed"
                    if candidate["installed"]
                    else "virtual_env_candidate"
                ),
                (
                    "Engine is available in the worker environment."
                    if candidate["installed"]
                    else "Dependencies may need installation at launch."
                ),
            ),
        ],
        "warnings": warnings,
    }
