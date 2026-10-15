"""AdaLN modulation helpers."""

import torch
from torch import nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class SharedCoreModulation(nn.Module):
    """One shared modulation core plus a per-block low-rank residual.

    ``m_i(c) = m_shared(c) + U_i V_i c`` with ``V_i: D -> r`` and
    ``U_i: r -> 6D``, both bias-free. The core is one ``Linear(D, 6D)`` per
    stream, owned by the model and shared by every patch block; it is
    deliberately NOT registered as a submodule here, so it appears exactly once
    in ``parameters()`` and once in ``state_dict()`` instead of once per block.

    Drop-in for the ``nn.Linear(dim, 6 * dim)`` it replaces: identical
    ``[..., 6*D]`` output layout (shift/scale/gate for attention, then for the
    MLP).

    Zero-init (``model.adaln_zero_init``): the core and ``adaln_up`` (U) are
    zeroed by the model's leaf-name scan, which leaves the block exactly as
    identity-at-init as the per-block projection. ``down`` (V) deliberately
    keeps its default init -- with both factors zero the residual's gradient is
    identically zero and the branch could never train.
    """

    def __init__(self, dim: int, rank: int, core: nn.Linear):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"model.modulation_rank must be positive, got {rank}")
        self.down = nn.Linear(dim, rank, bias=False)
        self.adaln_up = nn.Linear(rank, 6 * dim, bias=False)
        self._core = (core,)  # tuple hides the shared core from nn.Module registration

    @property
    def core(self) -> nn.Linear:
        return self._core[0]

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.core(cond) + self.adaln_up(self.down(cond))


class SharedCoreBias(nn.Module):
    """One shared modulation core plus a per-block learned bias (adaLN-single).

    ``m_i(c) = m_shared(c) + b_i``. The block-specific term is unconditional, so
    every block reads the same timestep response and only shifts it by a learned
    constant, the most aggressive sharing that still lets blocks differ at all.

    Same contract as :class:`SharedCoreModulation`: the core is owned by the
    model and hidden from registration here, and the ``[..., 6*D]`` output layout
    is unchanged.

    The bias starts at zero, which makes the block identity-at-init whenever the
    core is (``model.adaln_zero_init``) without needing the zero-init scan to
    know about it. Unlike the low-rank residual there is no dead-branch hazard:
    a bias always receives gradient regardless of the core's value.
    """

    def __init__(self, dim: int, core: nn.Linear):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(6 * dim))
        self._core = (core,)  # tuple hides the shared core from nn.Module registration

    @property
    def core(self) -> nn.Linear:
        return self._core[0]

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.core(cond) + self.bias


class ModulationBuilder:
    """Builds a block's per-stream adaLN projection for the configured mode.

    ``per_block`` returns a plain ``nn.Linear(dim, 6 * dim, bias=True)`` per
    block. ``shared_lowrank`` returns a
    :class:`SharedCoreModulation` bound to a per-stream core created on first
    request inside ``cores``, a ``ModuleDict`` owned by the model.
    ``shared_bias`` returns a :class:`SharedCoreBias` bound to that same core.

    Streams are named by the caller: the dual-stream block asks for ``img`` and
    ``txt``, the single-stream block for one ``shared`` projection.
    ``stream_aliases`` redirects one stream name onto another's core, which is
    how a hybrid trunk keeps its single-stream half on the dual half's ``img``
    core instead of allocating a third one.
    """

    MODES = ("per_block", "shared_lowrank", "shared_bias")
    SHARED = ("shared_lowrank", "shared_bias")

    def __init__(
        self,
        dim: int,
        mode: str = "per_block",
        rank: int = 64,
        cores: nn.ModuleDict | None = None,
        stream_aliases: dict[str, str] | None = None,
    ):
        if mode not in self.MODES:
            raise ValueError(f"model.modulation must be one of {self.MODES}, got {mode!r}")
        if mode in self.SHARED and cores is None:
            raise ValueError(f"{mode} modulation needs a core container owned by the model")
        self.dim = dim
        self.mode = mode
        self.rank = rank
        self.cores = cores
        self.stream_aliases = stream_aliases or {}

    def __call__(self, stream: str) -> nn.Module:
        if self.mode == "per_block":
            return nn.Linear(self.dim, 6 * self.dim, bias=True)
        key = f"adaln_{self.stream_aliases.get(stream, stream)}"
        if key not in self.cores:
            self.cores[key] = nn.Linear(self.dim, 6 * self.dim, bias=True)
        core = self.cores[key]
        if self.mode == "shared_bias":
            return SharedCoreBias(self.dim, core)
        return SharedCoreModulation(self.dim, self.rank, core)
