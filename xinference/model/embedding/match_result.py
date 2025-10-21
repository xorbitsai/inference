"""
Error handling result structures for embedding model engine matching.

This module provides structured error handling for engine matching operations,
allowing engines to provide detailed failure reasons and suggestions.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class MatchResult:
    """
    Result of engine matching operation with detailed error information.

    This class provides structured information about whether an engine can handle
    a specific model configuration, and if not, why and what alternatives exist.
    """

    is_match: bool
    reason: Optional[str] = None
    error_type: Optional[str] = None
    technical_details: Optional[str] = None

    @classmethod
    def success(cls) -> "MatchResult":
        """Create a successful match result."""
        return cls(is_match=True)

    @classmethod
    def failure(
        cls,
        reason: str,
        error_type: Optional[str] = None,
        technical_details: Optional[str] = None,
    ) -> "MatchResult":
        """Create a failed match result with optional details."""
        return cls(
            is_match=False,
            reason=reason,
            error_type=error_type,
            technical_details=technical_details,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        result: Dict[str, Any] = {"is_match": self.is_match}
        if not self.is_match:
            if self.reason:
                result["reason"] = self.reason
            if self.error_type:
                result["error_type"] = self.error_type
            if self.technical_details:
                result["technical_details"] = self.technical_details
        return result

    def to_error_string(self) -> str:
        """Convert to error string for backward compatibility."""
        if self.is_match:
            return "Available"
        error_msg = self.reason or "Unknown error"
        return error_msg


# Error type constants for better categorization
class ErrorType:
    HARDWARE_REQUIREMENT = "hardware_requirement"
    OS_REQUIREMENT = "os_requirement"
    MODEL_FORMAT = "model_format"
    DEPENDENCY_MISSING = "dependency_missing"
    MODEL_COMPATIBILITY = "model_compatibility"
    DIMENSION_MISMATCH = "dimension_mismatch"
    VERSION_REQUIREMENT = "version_requirement"
    CONFIGURATION_ERROR = "configuration_error"
    ENGINE_UNAVAILABLE = "engine_unavailable"
