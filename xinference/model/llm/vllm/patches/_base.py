"""Base dataclass for vLLM patches."""
from dataclasses import dataclass
from typing import Callable, Optional, Set


@dataclass
class VllmPatch:
    """
    Describes a post-install patch for vLLM.

    Attributes:
        name: Unique identifier for the patch.
        fn: Callable(env_path) -> bool. Returns True if patch was applied.
        description: What the patch fixes.
        removal_condition: When this patch should be deleted.
        model_families: Set of model families this patch applies to.
                        None means apply to all vLLM models.
    """

    name: str
    fn: Callable[[str], bool]
    description: str
    removal_condition: str
    model_families: Optional[Set[str]] = None
