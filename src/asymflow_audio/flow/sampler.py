"""50-step Euler ODE sampler for flow matching.

Standard FM:   dx/dt = û(x_t, t)
Asym FM:       dx/dt = recover_velocity(û_A(x_t, t), x_t, t, projector)

Both use the same interface — projector=None → standard FM.
"""
import torch

from .loss import recover_velocity


@torch.no_grad()
def euler_sample(
    model,
    shape: tuple,
    steps: int = 50,
    projector=None,
    patch_size: int = 64,
    sigma_min: float = 1e-3,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample from the learned flow via Euler integration.

    shape:     (B, L) — output shape
    Returns:   (B, L) waveform samples
    """
    if device is None:
        device = next(model.parameters()).device

    x = torch.randn(shape, device=device, dtype=dtype)
    dt = 1.0 / steps
    ts = torch.linspace(0.0, 1.0 - dt, steps, device=device)

    for t_val in ts:
        t = t_val.expand(shape[0])
        u_pred = model(x, t)

        if projector is not None:
            u = recover_velocity(u_pred, x, t, projector, patch_size, sigma_min)
        else:
            u = u_pred

        x = x + u * dt

    return x.clamp(-1.0, 1.0)
