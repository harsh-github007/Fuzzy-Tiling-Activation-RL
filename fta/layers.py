"""Fuzzy tiling activation (Pan, White and White, 2021) and Range Normalization."""
import torch
from torch import nn


class FTA(nn.Module):
    """Maps each input unit z to a k-dimensional, mostly-zero vector.

    The tiling bound [low, high] is split into k tiles of width delta = (high - low) / k with left edges
    c = (low, low + delta, ..., high - delta). For each tile,

        phi_eta(z) = 1 - I_eta,+( max(c - z, 0) + max(z - delta - c, 0) ),
        I_eta,+(x) = x if x <= eta else 1,

    so a tile is 1 when z falls inside it, falls off linearly over a distance eta outside it (the "fuzzy"
    part, which keeps a gradient), and is 0 further away. With eta = 0 this is the hard tiling activation.
    """

    def __init__(self, low: float = -20.0, high: float = 20.0, tiles: int = 20, eta: float | None = None):
        super().__init__()
        if high <= low or tiles < 1:
            raise ValueError("need high > low and tiles >= 1")
        self.low, self.high, self.tiles = float(low), float(high), int(tiles)
        self.delta = (self.high - self.low) / self.tiles
        self.eta = self.delta if eta is None else float(eta)
        self.register_buffer("c", self.low + self.delta * torch.arange(self.tiles, dtype=torch.float32))

    def fuzzy_indicator(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x <= self.eta, x, torch.ones_like(x)) if self.eta > 0 else (x > 0).float()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.unsqueeze(-1)                                   # (..., n, 1)
        d = torch.clamp(self.c - z, min=0) + torch.clamp(z - self.delta - self.c, min=0)
        out = 1.0 - self.fuzzy_indicator(d)                    # (..., n, k)
        return out.flatten(-2)                                 # (..., n * k)

    def extra_repr(self) -> str:
        return f"low={self.low}, high={self.high}, tiles={self.tiles}, delta={self.delta:g}, eta={self.eta:g}"


class RangeNorm(nn.Module):
    """Rescales each sample's features to [-1, 1]: y = 2 (x - min) / (max - min) - 1."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lo = x.min(dim=-1, keepdim=True).values
        hi = x.max(dim=-1, keepdim=True).values
        return 2 * (x - lo) / (hi - lo + self.eps) - 1
