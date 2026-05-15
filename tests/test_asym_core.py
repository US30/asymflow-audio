"""Core correctness tests for the asym-velocity implementation.

Run: pytest tests/ -v
"""
import math
import torch
import pytest
from einops import rearrange

PATCH = 64
B = 4
L = 16000
N = L // PATCH  # 250 tokens


def make_dct_projector(rank):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from asymflow_audio.flow.projector import DCTProjector
    return DCTProjector(PATCH, rank)


# ── Projector properties ──────────────────────────────────────────────────────

def test_projector_idempotent():
    """P^2 == P (idempotent)."""
    proj = make_dct_projector(8)
    x = torch.randn(10, PATCH)
    Px = proj.project(x)
    PPx = proj.project(Px)
    assert torch.allclose(Px, PPx, atol=1e-5), "P is not idempotent"


def test_projector_symmetric():
    """P is symmetric (P == P^T)."""
    proj = make_dct_projector(8)
    assert torch.allclose(proj.P, proj.P.T, atol=1e-5), "P is not symmetric"


def test_project_complement_sum():
    """P·x + (I-P)·x == x."""
    proj = make_dct_projector(16)
    x = torch.randn(10, PATCH)
    assert torch.allclose(proj.project(x) + proj.complement(x), x, atol=1e-5)


def test_full_rank_projector_is_identity():
    """P with rank=patch_size should be identity → project(x) == x."""
    proj = make_dct_projector(PATCH)
    x = torch.randn(5, PATCH)
    assert torch.allclose(proj.project(x), x, atol=1e-5), "Full-rank projector is not identity"


# ── Loss equivalence at full rank ─────────────────────────────────────────────

def test_full_rank_asym_equals_fm():
    """
    AsymFM with r=patch_size must produce identical loss as standard FM.
    Tests paper claim: P=I → u_A = ε - x₀ = u.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from asymflow_audio.flow.loss import fm_loss, asym_fm_loss, sample_xt, linear_schedule
    from asymflow_audio.flow.projector import DCTProjector

    proj_full = DCTProjector(PATCH, PATCH)

    class MockModel(torch.nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)

    model = MockModel()
    torch.manual_seed(0)
    x0 = torch.randn(B, L)

    # Fix RNG so both losses use same noise
    torch.manual_seed(1)
    loss_fm = fm_loss(model, x0)
    torch.manual_seed(1)
    loss_asym = asym_fm_loss(model, x0, proj_full, PATCH)

    assert torch.allclose(loss_fm, loss_asym, atol=1e-4), \
        f"Full-rank AsymFM ({loss_asym:.6f}) != FM ({loss_fm:.6f})"


# ── Velocity recovery ─────────────────────────────────────────────────────────

def test_velocity_recovery_shape():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from asymflow_audio.flow.loss import recover_velocity
    from asymflow_audio.flow.projector import DCTProjector

    proj = DCTProjector(PATCH, 8)
    u_A = torch.randn(B, L)
    xt = torch.randn(B, L)
    t = torch.rand(B)
    u = recover_velocity(u_A, xt, t, proj, PATCH)
    assert u.shape == (B, L)


def test_velocity_recovery_full_rank_identity():
    """
    With r=patch_size: u = P·u_A + (I-P)·(x_t + u_A)/σ_t
                         = u_A + 0 = u_A   (since P=I, I-P=0)
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from asymflow_audio.flow.loss import recover_velocity
    from asymflow_audio.flow.projector import DCTProjector

    proj = DCTProjector(PATCH, PATCH)
    u_A = torch.randn(B, L)
    xt = torch.randn(B, L)
    t = torch.full((B,), 0.5)
    u = recover_velocity(u_A, xt, t, proj, PATCH)
    assert torch.allclose(u, u_A, atol=1e-5), "Full-rank recovery must equal u_A"


# ── Synthetic sinusoid convergence ─────────────────────────────────────────────

def test_synthetic_convergence_asymfm_matches_fm():
    """
    Quick check: on a single sinusoid batch, both losses are non-NaN and finite.
    Full convergence comparison requires training — done separately.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from asymflow_audio.flow.loss import fm_loss, asym_fm_loss
    from asymflow_audio.flow.projector import DCTProjector
    from asymflow_audio.model.dit1d import DiT1D

    model = DiT1D(length=L, patch_size=PATCH, dim=64, depth=2, heads=4)
    proj = DCTProjector(PATCH, 8)

    # Synthetic: sum of 3 sinusoids (genuinely low-rank in DCT)
    t = torch.linspace(0, 1, L).unsqueeze(0).expand(B, -1)
    x0 = (torch.sin(2 * math.pi * 3 * t) +
          torch.sin(2 * math.pi * 7 * t) +
          torch.sin(2 * math.pi * 13 * t)) / 3.0

    loss_fm = fm_loss(model, x0)
    loss_asym = asym_fm_loss(model, x0, proj, PATCH)

    assert torch.isfinite(loss_fm), f"FM loss NaN/Inf: {loss_fm}"
    assert torch.isfinite(loss_asym), f"AsymFM loss NaN/Inf: {loss_asym}"
