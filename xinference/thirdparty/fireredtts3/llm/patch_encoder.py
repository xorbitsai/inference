import torch
from fireredtts3.llm.modules import (
    DiTBlock,
    FinalLayer,
    RotaryEmbedding,
)


class PatchEncoder(torch.nn.Module):
    def __init__(
        self, 
        # In & out
        in_dim: int,
        out_dim: int,
        # Model config
        patch_size: int = 4,
        hidden_size: int = 1024,
        mlp_ratio: int = 3,
        depth: int = 8,
        num_heads: int = 8,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        # [CLS] token
        self.cls_tok = torch.nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.rotary_embed = RotaryEmbedding(hidden_size // num_heads)
        self.blocks = torch.nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        # Input & output proj
        self.in_proj = (
            torch.nn.Linear(in_dim, hidden_size)
            if in_dim != hidden_size else 
            torch.nn.Identity()
        )
        self.out_proj = FinalLayer(hidden_size, out_dim)

    def forward(self, inputs_embeds: torch.Tensor):
        """Patch encoder aggregating {patch_size} latents into one.

        Args:
            inputs_embeds(torch.Tensor): shape (b=1, t, c).
        Returns:
            hidden_states(torch.Tensor): shape (b=1, t//patch_size, c).
        """
        assert inputs_embeds.shape[1] % self.patch_size == 0, \
            'inputs_embeds.shape={} patch_size={}'.format(inputs_embeds.shape, self.patch_size)
        
        inputs_embeds = self.in_proj(inputs_embeds)
        # Patchify, (b=1, t, c) -> (t//patch_size, patch_size, c)
        hidden_states = inputs_embeds.reshape(-1, self.patch_size, self.hidden_size)
        cls_tok = self.cls_tok.expand(hidden_states.shape[0], -1, -1)   # (b*t//patch_size, 1, c)
        hidden_states = torch.cat([cls_tok, hidden_states], dim=1)  # (b*t//patch_size, 1+patch_size, c)
        # NOTE full attention
        rope = self.rotary_embed.forward_from_seq_len(hidden_states.shape[1])
        for block in self.blocks:
            hidden_states = block(hidden_states, None, rope)
        hidden_states = self.out_proj(hidden_states)
        hidden_states = hidden_states[:, 0]    # (t//patch_size, c)
        hidden_states = hidden_states.unsqueeze(0)
        return hidden_states

