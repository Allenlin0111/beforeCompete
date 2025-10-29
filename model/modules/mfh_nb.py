import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantileHead(nn.Module):
    """
    Predict per-channel quantiles (q25, q50, q75) from decoder states.

    Args:
        d_model: hidden dimension of decoder output.
        c_out: number of target channels.
        hidden: optional hidden width for the MLP.
    """

    def __init__(self, d_model: int, pred_len: int, c_out: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden or d_model
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3 * pred_len),
        )
        self.pred_len = pred_len
        self.c_out = c_out

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h: decoder states of shape [B, C, D] (token-major).

        Returns:
            Tuple of (q25, q50, q75) each with shape [B, pred_len, C].
        """
        bsz, tokens, _ = h.shape
        quantiles = self.net(h).view(bsz, tokens, self.pred_len, 3)
        quantiles = quantiles.permute(0, 2, 1, 3)  # [B, pred_len, tokens, 3]
        q25 = quantiles[..., 0]
        q50 = quantiles[..., 1]
        q75 = quantiles[..., 2]
        return q25, q50, q75


class MFHNBCalib:
    """Lightweight loader for MFH-NB calibration statistics."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.p20: Optional[float] = None
        self.p80: Optional[float] = None
        if not path:
            return
        try:
            meta = json.loads(Path(path).read_text(encoding="utf-8"))
            self.p20 = float(meta["p20"])
            self.p80 = float(meta["p80"])
        except Exception:
            # Leave values as None to signal missing/invalid calibration.
            self.p20 = None
            self.p80 = None


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Pinball (quantile) loss.

    Args:
        pred: predicted quantile.
        target: ground truth.
        tau: quantile level (0 < tau < 1).
    """
    diff = target - pred
    return torch.mean(torch.maximum(tau * diff, (tau - 1.0) * diff))


def quantile_losses(
    q25: torch.Tensor,
    q50: torch.Tensor,
    q75: torch.Tensor,
    target: torch.Tensor,
    lambda_q: float = 0.5,
    lambda_mono: float = 1e-3,
) -> torch.Tensor:
    """
    Aggregate quantile supervision and monotonicity regularisation.
    """
    l25 = pinball_loss(q25, target, 0.25)
    l50 = pinball_loss(q50, target, 0.5)
    l75 = pinball_loss(q75, target, 0.75)
    mono = F.relu(q25 - q50).mean() + F.relu(q50 - q75).mean()
    return lambda_q * (l25 + l50 + l75) + lambda_mono * mono


def mfh_nb_fuse(
    y_main: torch.Tensor,
    q25: torch.Tensor,
    q50: torch.Tensor,
    q75: torch.Tensor,
    p20: float,
    p80: float,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bias-free fusion: y = (1 - α) y_main + α q50, α = clip((s - p20)/(p80 - p20), 0, 1),
    where s = log(IQR + eps) + log(|y_main - q50| + eps).
    """
    iqr = (q75 - q25).clamp_min(eps).detach()
    spread = (y_main - q50).abs().clamp_min(eps).log()
    s = iqr.log() + spread
    denom = float(p80 - p20)
    if denom <= eps:
        denom = eps
    alpha = ((s - p20) / denom).clamp(0.0, 1.0)
    fused = (1.0 - alpha) * y_main + alpha * q50
    return fused, alpha
