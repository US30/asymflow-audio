"""Rank-r orthogonal projectors P = A A^T for the asym-velocity trick.

DCT projector:  A = first r rows of the orthonormal DCT-II matrix (patch_size × patch_size).
PCA projector:  A = top-r eigenvectors from PCA over training patches.

Both guarantee P^2 = P (idempotent) and P^T = P (symmetric).
"""
import torch
import torch.nn as nn
import numpy as np
from scipy.fft import dct as scipy_dct


class DCTProjector(nn.Module):
    """
    Fixed (non-learnable) projector using DCT-II basis.

    A has shape (r, patch_size). P = A^T A is (patch_size, patch_size).
    """

    def __init__(self, patch_size: int, rank: int):
        super().__init__()
        assert rank <= patch_size
        # Build orthonormal DCT-II basis: rows are basis vectors
        basis = _dct_basis(patch_size)  # (patch_size, patch_size), orthonormal
        A = torch.from_numpy(basis[:rank]).float()  # (r, patch_size)
        # P = A^T A  ->  (patch_size, patch_size)
        P = A.T @ A
        self.register_buffer("A", A)
        self.register_buffer("P", P)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Apply P to last dim. x: (..., patch_size) -> (..., patch_size)."""
        return x @ self.P.T  # P is symmetric, P.T == P

    def complement(self, x: torch.Tensor) -> torch.Tensor:
        """Apply (I - P) to last dim."""
        return x - self.project(x)


class PCAProjector(nn.Module):
    """Learnable projector from PCA over training data patches."""

    def __init__(self, patch_size: int, rank: int):
        super().__init__()
        self.patch_size = patch_size
        self.rank = rank
        # Placeholder — call fit() before training
        self.register_buffer("A", torch.zeros(rank, patch_size))
        self.register_buffer("P", torch.zeros(patch_size, patch_size))
        self._fitted = False

    @torch.no_grad()
    def fit(self, patches: torch.Tensor):
        """
        patches: (N, patch_size) float32 — sampled from training data.
        Computes PCA and sets A, P.
        """
        patches = patches.float()
        patches = patches - patches.mean(0, keepdim=True)
        _, _, Vh = torch.linalg.svd(patches, full_matrices=False)  # Vh: (min, patch_size)
        A = Vh[: self.rank]  # (r, patch_size) — already orthonormal rows
        self.A.copy_(A)
        self.P.copy_(A.T @ A)
        self._fitted = True

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.P.T

    def complement(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.project(x)


def build_projector(projector_type: str, patch_size: int, rank: int) -> nn.Module:
    if projector_type == "dct":
        return DCTProjector(patch_size, rank)
    elif projector_type == "pca":
        return PCAProjector(patch_size, rank)
    elif projector_type is None or projector_type == "none":
        return None
    else:
        raise ValueError(f"Unknown projector: {projector_type}")


def _dct_basis(n: int) -> np.ndarray:
    """Return (n, n) orthonormal DCT-II matrix using scipy."""
    # Each row is the DCT-II basis vector for that frequency
    I = np.eye(n, dtype=np.float64)
    # Apply DCT-II column-wise, with ortho norm
    basis = np.zeros_like(I)
    for i in range(n):
        col = scipy_dct(I[:, i], type=2, norm="ortho")
        basis[i] = col
    # Rows are frequency-ordered basis vectors (already orthonormal by ortho norm)
    return basis.astype(np.float32)
