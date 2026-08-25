"""
Proposed fix for xinference CosyVoice HyperPyYAML RCE (CVE pending).

Wraps load_hyperpyyaml with a pre-scan that validates all !new:, !apply:,
!name:, and !module: tags against an allowlist of known-safe modules before
the YAML is parsed. This prevents arbitrary class instantiation / function
calls at parse time.

Drop-in replacement: change `from hyperpyyaml import load_hyperpyyaml`
to `from safe_yaml import safe_load_hyperpyyaml` in cosyvoice.py.
"""
import io
import re
from typing import Optional, Set

HYPERPYYAML_TAG_PATTERN = re.compile(
    r"!(new|apply|name|module):([A-Za-z_][A-Za-z0-9_.]*)"
)

COSYVOICE_ALLOWED_MODULES: Set[str] = {
    "cosyvoice",
    "matcha",
    "omegaconf",
    "whisper",
    "conformer",
    "random",
    "numpy",
    "torch",
}


class UnsafeYAMLError(Exception):
    pass


def _get_top_level_module(fqn: str) -> str:
    return fqn.split(".")[0]


def validate_yaml_tags(
    content: str,
    allowed_modules: Optional[Set[str]] = None,
) -> None:
    if allowed_modules is None:
        allowed_modules = COSYVOICE_ALLOWED_MODULES

    for match in HYPERPYYAML_TAG_PATTERN.finditer(content):
        tag_type = match.group(1)
        fqn = match.group(2)
        top_module = _get_top_level_module(fqn)

        if top_module not in allowed_modules:
            raise UnsafeYAMLError(
                f"Blocked unsafe HyperPyYAML tag !{tag_type}:{fqn} "
                f"(module '{top_module}' is not in the allowlist). "
                f"Allowed top-level modules: {sorted(allowed_modules)}"
            )


def safe_load_hyperpyyaml(
    yaml_stream,
    overrides: Optional[dict] = None,
    allowed_modules: Optional[Set[str]] = None,
    **kwargs,
):
    from hyperpyyaml import load_hyperpyyaml

    if hasattr(yaml_stream, "read"):
        content = yaml_stream.read()
    else:
        content = str(yaml_stream)

    validate_yaml_tags(content, allowed_modules)

    return load_hyperpyyaml(io.StringIO(content), overrides=overrides, **kwargs)
