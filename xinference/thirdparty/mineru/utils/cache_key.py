class _IdentityKey:
    """Keep opaque, unhashable configuration objects alive and compare by identity."""

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return id(self.value)

    def __eq__(self, other):
        return isinstance(other, _IdentityKey) and self.value is other.value


def config_cache_key(value):
    """Snapshot nested options without dropping non-scalar model configuration."""
    if isinstance(value, dict):
        return (
            dict,
            frozenset(
                (config_cache_key(key), config_cache_key(item))
                for key, item in value.items()
            ),
        )
    if isinstance(value, (list, tuple)):
        return (type(value), tuple(config_cache_key(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return (type(value), frozenset(config_cache_key(item) for item in value))
    try:
        hash(value)
    except TypeError:
        return _IdentityKey(value)
    return (type(value), value)
