from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ConfigBase(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    def get(self, key: str, default=None):
        value = getattr(self, key, default)
        if value is default:
            return value

        fields_set = self.model_fields_set
        if value is None and key not in fields_set:
            return default
        return value

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)

    @classmethod
    def _declared_field_names(cls) -> list[str]:
        return [name for name in cls.model_fields if name != "model_config"]

    @classmethod
    def _serialize_declared_value(cls, value):
        if isinstance(value, ConfigBase):
            return value.to_declared_dict()
        if isinstance(value, list):
            return [cls._serialize_declared_value(item) for item in value]
        if isinstance(value, tuple):
            return [cls._serialize_declared_value(item) for item in value]
        if isinstance(value, dict):
            return {
                key: cls._serialize_declared_value(item) for key, item in value.items()
            }
        return value

    def to_declared_dict(self) -> dict[str, Any]:
        data = {}
        for name in self._declared_field_names():
            value = getattr(self, name, None)
            if value is None:
                continue
            data[name] = self._serialize_declared_value(value)
        return data


class StrictConfigBase(ConfigBase):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )


__all__ = ["ConfigBase", "StrictConfigBase"]
