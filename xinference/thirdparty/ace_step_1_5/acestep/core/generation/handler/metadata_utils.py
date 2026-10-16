"""Metadata formatting helpers for handler decomposition."""

from typing import Any, Dict, List, Optional, Union


class MetadataMixin:
    """Mixin containing metadata parsing and formatting helpers.

    Depends on host members:
    - No cross-mixin runtime dependencies; pure formatting/parsing helpers.
    """

    def _create_default_meta(self) -> str:
        """Create default metadata string."""
        return (
            "- bpm: N/A\n"
            "- timesignature: N/A\n"
            "- keyscale: N/A\n"
            "- duration: 30 seconds\n"
        )

    def _dict_to_meta_string(self, meta_dict: Dict[str, Any]) -> str:
        """Convert metadata dict to formatted string."""
        bpm = meta_dict.get("bpm", meta_dict.get("tempo", "N/A"))
        timesignature = meta_dict.get("timesignature", meta_dict.get("time_signature", "N/A"))
        keyscale = meta_dict.get("keyscale", meta_dict.get("key", meta_dict.get("scale", "N/A")))
        duration = meta_dict.get("duration", meta_dict.get("length", 30))

        if isinstance(duration, (int, float)):
            duration = f"{int(duration)} seconds"
        elif not isinstance(duration, str):
            duration = "30 seconds"

        return (
            f"- bpm: {bpm}\n"
            f"- timesignature: {timesignature}\n"
            f"- keyscale: {keyscale}\n"
            f"- duration: {duration}\n"
        )

    def _parse_metas(self, metas: List[Union[str, Dict[str, Any]]]) -> List[str]:
        """Parse and normalize metadata values with safe fallbacks."""
        parsed_metas = []
        for meta in metas:
            if meta is None:
                parsed_meta = self._create_default_meta()
            elif isinstance(meta, str):
                parsed_meta = meta
            elif isinstance(meta, dict):
                parsed_meta = self._dict_to_meta_string(meta)
            else:
                parsed_meta = self._create_default_meta()
            parsed_metas.append(parsed_meta)
        return parsed_metas

    def prepare_metadata(
        self, bpm: Optional[Union[int, str]], key_scale: str, time_signature: str
    ) -> Dict[str, Any]:
        """Build metadata dict for generation."""
        return self._build_metadata_dict(bpm, key_scale, time_signature)

    def _build_metadata_dict(
        self,
        bpm: Optional[Union[int, str]],
        key_scale: str,
        time_signature: str,
        duration: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Build metadata dictionary with defaults for missing fields."""
        metadata_dict: Dict[str, Any] = {}
        metadata_dict["bpm"] = bpm if bpm else "N/A"
        metadata_dict["keyscale"] = key_scale if key_scale.strip() else "N/A"
        if time_signature.strip() and time_signature != "N/A" and time_signature:
            metadata_dict["timesignature"] = time_signature
        else:
            metadata_dict["timesignature"] = "N/A"
        if duration is not None:
            metadata_dict["duration"] = f"{int(duration)} seconds"
        return metadata_dict
