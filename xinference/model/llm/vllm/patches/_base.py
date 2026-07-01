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
        architectures: Set of model architectures this patch applies to.
                       None means apply to all vLLM models.
    """

    name: str
    fn: Callable[[str], bool]
    description: str
    removal_condition: str
    architectures: Optional[Set[str]] = None
