"""Flow matching loss variants.

Standard FM:   L = E[‖u - û‖²]          where u = ε - x₀
Asym FM:       L = E[‖u_A - û‖²]        where u_A = P·ε - x₀

Velocity recovery at sample time (paper Eq. 5):
    u = P·u_A + (I-P)·(x_t + u_A) / σ_t

where σ_t = 1 - t (linear flow schedule).
"""
import torch
import torch.nn.functional as F
from einops import rearrange

from .projector import DCTProjector, PCAProjector


def linear_schedule(t: torch.Tensor):
    """Returns (alpha_t, sigma_t) for linear interpolant x_t = alpha_t * x0 + sigma_t * eps."""
    return t, 1.0 - t


def sample_xt(x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Interpolate between x0 and eps at time t. t: (B,)."""
    alpha_t, sigma_t = linear_schedule(t)
    return alpha_t[:, None] * x0 + sigma_t[:, None] * eps


def fm_loss(model, x0: torch.Tensor, projector=None) -> torch.Tensor:
    """
    Standard flow matching loss. projector unused (kept for uniform API).
    x0: (B, L)
    """
    B, L = x0.shape
    t = torch.rand(B, device=x0.device)
    eps = torch.randn_like(x0)
    xt = sample_xt(x0, eps, t)

    u_target = eps - x0  # standard velocity target

    u_pred = model(xt, t)
    return F.mse_loss(u_pred, u_target)


def asym_fm_loss(model, x0: torch.Tensor, projector, patch_size: int) -> torch.Tensor:
    """
    Rank-asymmetric FM loss (paper Eq. 4).
    u_A = P·ε - x₀  (patch-wise projection of noise only)
    x0: (B, L)
    """
    B, L = x0.shape
    t = torch.rand(B, device=x0.device)
    eps = torch.randn_like(x0)
    xt = sample_xt(x0, eps, t)

    # Project eps patch-wise
    eps_patches = rearrange(eps, 'b (n p) -> b n p', p=patch_size)
    Peps_patches = projector.project(eps_patches)
    Peps = rearrange(Peps_patches, 'b n p -> b (n p)')

    u_A_target = Peps - x0  # asymmetric velocity target

    u_A_pred = model(xt, t)
    return F.mse_loss(u_A_pred, u_A_target)


def recover_velocity(u_A: torch.Tensor, xt: torch.Tensor, t: torch.Tensor,
                     projector, patch_size: int, sigma_min: float = 1e-3) -> torch.Tensor:
    """
    Recover full-rank velocity from asymmetric prediction (paper Eq. 5).

    u = P·u_A + (I-P)·(x_t + u_A) / σ_t

    u_A:  (B, L) — model output (asym prediction)
    xt:   (B, L) — noisy sample
    t:    (B,)   — timestep in [0,1]
    """
    sigma_t = (1.0 - t).clamp(min=sigma_min)  # (B,)

    # Patch-wise projections
    u_A_p = rearrange(u_A, 'b (n p) -> b n p', p=patch_size)
    xt_p = rearrange(xt, 'b (n p) -> b n p', p=patch_size)

    Pu_A = projector.project(u_A_p)              # (B, n, p)
    ImP_uA = projector.complement(u_A_p)          # (B, n, p)
    ImP_xt = projector.complement(xt_p)           # (B, n, p)

    # x₀-pred path: (I-P)(x_t + u_A) / σ_t
    sigma_t_bcast = sigma_t[:, None, None]
    x0_part = (ImP_xt + ImP_uA) / sigma_t_bcast

    u_full = Pu_A + x0_part  # (B, n, p)
    return rearrange(u_full, 'b n p -> b (n p)')
