"""YuE2 song generation. Optional backends are imported only when requested."""
__version__ = "0.1.6"


def __getattr__(name):
    if name in {"YuE2Pipeline", "SymbolicPlan", "SemanticResult", "SongResult"}:
        from . import pipeline
        return getattr(pipeline, name)
    if name in {"YuE2Config", "YuE2ForCausalLM"}:
        from . import modeling_yue2
        return getattr(modeling_yue2, name)
    if name in {"YuE2VAE", "YuE2VAEConfig"}:
        from . import modeling_vae
        return getattr(modeling_vae, name)
    raise AttributeError(name)
