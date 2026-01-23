import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

_BUILTIN_FAMILY_CACHE: Optional[List[Dict[str, Any]]] = None


def _resolve_config_and_dir(model_path: str) -> Tuple[str, str]:
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, "config.json")
        model_dir = model_path
    else:
        model_dir = os.path.dirname(model_path) or "."
        if os.path.basename(model_path) == "config.json":
            config_path = model_path
        else:
            config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise ValueError(f"config.json not found under {model_path}.")
    return config_path, model_dir


def _load_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r") as file:
        return json.load(file)


def _load_tokenizer_config(model_dir: str) -> Optional[Dict[str, Any]]:
    tokenizer_config_path = os.path.join(model_dir, "tokenizer_config.json")
    if not os.path.exists(tokenizer_config_path):
        return None
    return _load_json_file(tokenizer_config_path)


def _load_chat_template_file(model_dir: str) -> Optional[str]:
    chat_template_path = os.path.join(model_dir, "chat_template.jinja")
    if not os.path.exists(chat_template_path):
        return None
    with open(chat_template_path, "r") as file:
        content = file.read()
    return content.strip() if content else None


def _get_first_value(config: Dict[str, Any], *keys: str) -> Optional[Any]:
    for key in keys:
        value = config.get(key)
        if value is not None:
            return value
    return None


def _infer_context_length(config: Dict[str, Any]) -> int:
    candidates = [
        _get_first_value(config, "max_sequence_length"),
        _get_first_value(config, "seq_length"),
        _get_first_value(config, "max_position_embeddings"),
        _get_first_value(config, "max_seq_len"),
        _get_first_value(config, "model_max_length"),
    ]
    values = [v for v in candidates if isinstance(v, (int, float))]
    return int(max(values)) if values else 2048


def _normalize_architectures(config: Dict[str, Any]) -> List[str]:
    architectures = config.get("architectures")
    if isinstance(architectures, list):
        return [str(item) for item in architectures if item]
    if isinstance(architectures, str) and architectures:
        return [architectures]
    return []


def _match_family_by_architectures(
    architectures: List[str],
) -> Optional[Dict[str, Any]]:
    if not architectures:
        return None
    families = _load_builtin_families()
    matches: List[Dict[str, Any]] = []
    for family in families:
        if not family.get("architectures"):
            continue
        if any(arch in family["architectures"] for arch in architectures):
            matches.append(family)
    if len(matches) == 1:
        return matches[0]
    return None


def _load_builtin_families() -> List[Dict[str, Any]]:
    global _BUILTIN_FAMILY_CACHE
    if _BUILTIN_FAMILY_CACHE is not None:
        return _BUILTIN_FAMILY_CACHE
    json_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "llm_family.json"
    )
    with open(json_path, "r") as file:
        _BUILTIN_FAMILY_CACHE = json.load(file)
    return _BUILTIN_FAMILY_CACHE


def _infer_languages(config: Dict[str, Any]) -> List[str]:
    lang_value = _get_first_value(config, "language", "languages", "lang")
    if isinstance(lang_value, list) and lang_value:
        return [str(item) for item in lang_value]
    if isinstance(lang_value, str) and lang_value:
        return [lang_value]
    return ["en"]


def _format_size_in_billions(size_in_billions: float) -> Union[int, str]:
    rounded = round(size_in_billions, 1)
    if abs(rounded - round(rounded)) < 0.05:
        return int(round(rounded))
    return str(rounded).replace(".", "_")


def _extract_numeric_size(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        unit = text[-1].lower()
        if unit in {"b", "m"}:
            try:
                number = float(text[:-1])
            except ValueError:
                return None
            if unit == "b":
                return number
            return number / 1000.0
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _infer_model_size_in_billions(config: Dict[str, Any]) -> Optional[Union[int, str]]:
    size_value = _get_first_value(
        config,
        "model_size_in_billions",
        "num_parameters_in_billions",
    )
    size_in_billions = _extract_numeric_size(size_value)
    if size_in_billions:
        return _format_size_in_billions(size_in_billions)

    size_millions = _extract_numeric_size(
        _get_first_value(config, "num_parameters_in_millions", "num_params_in_millions")
    )
    if size_millions:
        return _format_size_in_billions(size_millions / 1000.0)

    param_value = _extract_numeric_size(
        _get_first_value(
            config, "num_parameters", "num_params", "n_params", "total_params"
        )
    )
    if param_value:
        if param_value > 1e6:
            return _format_size_in_billions(param_value / 1e9)
        if param_value > 0:
            return _format_size_in_billions(param_value)

    hidden_size = _get_first_value(config, "hidden_size", "d_model", "n_embd")
    num_layers = _get_first_value(config, "num_hidden_layers", "num_layers", "n_layer")
    if hidden_size is None or num_layers is None:
        return None
    try:
        hidden_size = int(hidden_size)
        num_layers = int(num_layers)
    except (TypeError, ValueError):
        return None

    vocab_size = _get_first_value(config, "vocab_size") or 0
    try:
        vocab_size = int(vocab_size)
    except (TypeError, ValueError):
        vocab_size = 0

    intermediate_size = _get_first_value(
        config, "intermediate_size", "ffn_dim", "n_inner"
    )
    try:
        intermediate_size = (
            int(intermediate_size) if intermediate_size else 4 * hidden_size
        )
    except (TypeError, ValueError):
        intermediate_size = 4 * hidden_size

    embedding_params = vocab_size * hidden_size
    attention_params = 4 * hidden_size * hidden_size
    mlp_params = 3 * hidden_size * intermediate_size
    total_params = embedding_params + num_layers * (attention_params + mlp_params)
    size_in_billions = total_params / 1e9
    if size_in_billions <= 0:
        return None
    return _format_size_in_billions(size_in_billions)


def _infer_quantization(config: Dict[str, Any], model_format: str) -> str:
    if model_format == "pytorch":
        return "none"
    quant_config = config.get("quantization_config") or {}
    if isinstance(quant_config, dict):
        bits = _get_first_value(quant_config, "bits", "wbits")
        if bits is not None:
            try:
                bits = int(bits)
                return f"Int{bits}"
            except (TypeError, ValueError):
                pass
        quant_method = quant_config.get("quant_method") or quant_config.get("method")
        if isinstance(quant_method, str) and quant_method:
            if "gptq" in quant_method.lower():
                return "Int4"
    quant = config.get("quantization")
    if isinstance(quant, str) and quant:
        return quant
    return ""


def _extract_chat_template(tokenizer_config: Optional[Dict[str, Any]]) -> Optional[str]:
    if not tokenizer_config:
        return None
    chat_template = tokenizer_config.get("chat_template")
    if isinstance(chat_template, str) and chat_template.strip():
        return chat_template
    return None


def _infer_model_format(config: Dict[str, Any]) -> str:
    quant_config = config.get("quantization_config") or {}
    quant_method = None
    if isinstance(quant_config, dict):
        quant_method = quant_config.get("quant_method") or quant_config.get("method")
    if isinstance(quant_method, str) and quant_method:
        lowered = quant_method.lower()
        if "awq" in lowered:
            return "awq"
        if "gptq" in lowered:
            return "gptq"
        if "fp8" in lowered:
            return "fp8"
        if "bnb" in lowered or "bitsandbytes" in lowered:
            return "bnb"
    return "pytorch"


def build_llm_registration_from_local_config(
    model_path: str, model_family: str
) -> Dict[str, Any]:

    config_path, model_dir = _resolve_config_and_dir(model_path)
    config = _load_json_file(config_path)
    tokenizer_config = _load_tokenizer_config(model_dir)
    chat_template_file = _load_chat_template_file(model_dir)

    model_lang = _infer_languages(config)
    chat_template = _extract_chat_template(tokenizer_config) or chat_template_file
    model_ability = ["generate"]
    if chat_template:
        model_ability.append("chat")
    if config.get("vision_config") is not None:
        model_ability.append("vision")

    prompt_style = None
    if isinstance(model_family, str) and model_family:
        from .llm_family import BUILTIN_LLM_PROMPT_STYLE

        prompt_style = BUILTIN_LLM_PROMPT_STYLE.get(model_family)
        if prompt_style:
            if prompt_style.get("chat_template") and "chat" not in model_ability:
                model_ability.append("chat")
            if prompt_style.get("reasoning_start_tag") and prompt_style.get(
                "reasoning_end_tag"
            ):
                if "reasoning" not in model_ability:
                    model_ability.append("reasoning")

    context_length = _infer_context_length(config)
    model_size_in_billions = _infer_model_size_in_billions(config)
    if model_size_in_billions is None:
        raise ValueError("Unable to infer model_size_in_billions from config.json.")

    model_format = _infer_model_format(config)
    quantization = _infer_quantization(config, model_format)
    if model_format != "pytorch" and not quantization:
        raise ValueError(
            "Unable to infer quantization for the selected model_format from config.json."
        )

    model_spec = {
        "model_uri": model_dir,
        "model_format": model_format,
        "model_size_in_billions": model_size_in_billions,
        "quantization": quantization,
    }

    result = {
        "context_length": context_length,
        "model_lang": model_lang,
        "model_family": model_family,
        "model_ability": model_ability,
        "model_specs": [model_spec],
    }
    if "chat" in model_ability and prompt_style:
        result["chat_template"] = prompt_style.get("chat_template")
        result["stop"] = prompt_style.get("stop")
        result["stop_token_ids"] = prompt_style.get("stop_token_ids")
    if "reasoning" in model_ability and prompt_style:
        result["reasoning_start_tag"] = prompt_style.get("reasoning_start_tag")
        result["reasoning_end_tag"] = prompt_style.get("reasoning_end_tag")
    return result
