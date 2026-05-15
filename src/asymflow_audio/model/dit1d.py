"""1D Diffusion Transformer (DiT-S/64) for raw waveform generation.

Architecture follows Peebles & Xie (DiT, 2022):
- Patch waveform into fixed-size chunks
- Linear proj in → hidden dim
- N transformer blocks with AdaLN-zero conditioning on timestep
- Linear proj out → patch dim
"""
import math
import torch
import torch.nn as nn
from einops import rearrange


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.freq_dim = freq_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) float in [0, 1]
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t[:, None] * freqs[None]  # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, freq_dim)
        return self.mlp(emb)  # (B, dim)


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )
        # AdaLN-zero: 6 params (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN_modulation(c).chunk(6, dim=-1)
        # Attention
        x_norm = modulate(self.norm1(x), shift1, scale1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + gate1.unsqueeze(1) * attn_out
        # MLP
        x = x + gate2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift2, scale2))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim: int, patch_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True),
        )
        self.proj = nn.Linear(dim, patch_size)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.proj(x)  # (B, T, patch_size)


class DiT1D(nn.Module):
    """
    1D DiT for raw waveform flow matching.

    Input:  x (B, L) waveform, t (B,) time in [0,1]
    Output: velocity estimate (B, L)
    """

    def __init__(self, length: int = 16000, patch_size: int = 64,
                 dim: int = 384, depth: int = 12, heads: int = 6,
                 mlp_ratio: float = 4.0):
        super().__init__()
        assert length % patch_size == 0, "length must be divisible by patch_size"
        self.patch_size = patch_size
        self.num_patches = length // patch_size
        self.dim = dim

        self.patch_embed = nn.Linear(patch_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        self.time_embed = TimestepEmbedder(dim)
        self.blocks = nn.ModuleList([
            DiTBlock(dim, heads, mlp_ratio) for _ in range(depth)
        ])
        self.final = FinalLayer(dim, patch_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, L), t: (B,)
        B, L = x.shape
        # Patchify: (B, T, P)
        x = rearrange(x, 'b (t p) -> b t p', p=self.patch_size)
        x = self.patch_embed(x) + self.pos_embed  # (B, T, dim)
        c = self.time_embed(t)                     # (B, dim)
        for block in self.blocks:
            x = block(x, c)
        x = self.final(x, c)                       # (B, T, patch_size)
        x = rearrange(x, 'b t p -> b (t p)')       # (B, L)
        return x


def build_model(cfg) -> DiT1D:
    return DiT1D(
        length=cfg.data.length,
        patch_size=cfg.model.patch_size,
        dim=cfg.model.dim,
        depth=cfg.model.depth,
        heads=cfg.model.heads,
        mlp_ratio=cfg.model.mlp_ratio,
    )
