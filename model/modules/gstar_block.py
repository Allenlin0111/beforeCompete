import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from layers.star_block import STARBlock
from model.modules.gated_fusion import GatedFusion


class GStarBlock(nn.Module):
    """
    Wrapper around STARBlock that keeps the original STAR refinement intact
    while allowing each gating stage to be toggled independently:
      - STAR residual gate (inside STARBlock)
      - Fusion gate (GatedFusion wrapper)

    Additional guard options (all off by default). Module expects the feature
    dimension to be the last axis (input layout [B, L, C] or compatible). We
    keep a runtime assertion to detect accidental mismatches early.
      * rms_norm   : L2-normalise the fusion delta (star - dec) per sample.
      * cosine_mod : shrink/expand alpha by cosine(dec, star) in [0, 1].
      * couple_tcn : multiply alpha by (1 - sigmoid(g_tcn)) where g_tcn
                     comes from the TCN prefilter gate (scalar).
    """

    def __init__(
        self,
        c_out: int,
        k: int = 25,
        star_gate_bias: float = -4.0,
        use_ln: bool = True,
        fusion_gate_bias: float = -2.0,
        use_star_gate: bool = True,
        use_fusion_gate: bool = True,
        rms_norm: bool = False,
        cosine_mod: bool = False,
        couple_tcn: bool = False,
    ) -> None:
        super().__init__()
        self.star = STARBlock(c_out=c_out, k=k, gate_bias=star_gate_bias, use_ln=use_ln)
        self.use_star_gate = bool(use_star_gate)
        self.use_fusion_gate = bool(use_fusion_gate)
        self.fusion_gate = (
            GatedFusion(dim=c_out, gate_init_bias=fusion_gate_bias)
            if self.use_fusion_gate
            else None
        )

        self.rms_norm = bool(rms_norm)
        self.cosine_mod = bool(cosine_mod)
        self.couple_tcn = bool(couple_tcn)
        self._eps = 1e-6

    def forward(self, x: torch.Tensor, g_tcn: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: base forecast tensor (B, L, C) or compatible layout.
            g_tcn: optional scalar gate from TCN prefilter (already detached).
        """
        if self.use_star_gate:
            star_out = self.star(x)
        else:
            # Bypass the STAR residual gate: use the smoothed signal directly.
            z = self.star.ln(x)
            star_out = self.star.star(z)

        if self.fusion_gate is None:
            return star_out

        base = x
        aux = star_out

        base_norm = self.fusion_gate.left_norm(base)
        aux_norm = self.fusion_gate.right_norm(aux)
        gate_input = torch.cat([base_norm, aux_norm], dim=-1)
        alpha = torch.sigmoid(self.fusion_gate.gate_proj(gate_input))

        delta = aux - base
        # feature dimension inference: assume last dim, fallback to second if shapes differ
        feat_dim = -1
        if alpha.size(-1) != base.size(-1):
            raise RuntimeError(
                f"GStar alpha last dim {alpha.size(-1)} does not match feature dim {base.size(-1)}"
            )
        if not hasattr(self, "_dbg_shapes_printed"):
            print(
                "[GStar] shapes:",
                tuple(base.shape),
                tuple(aux.shape),
                tuple(alpha.shape),
                "feat_dim=",
                feat_dim,
            )
            self._dbg_shapes_printed = True
        if self.rms_norm:
            rms = delta.pow(2).mean(dim=feat_dim, keepdim=True).sqrt()
            delta = delta / (rms + self._eps)

        if self.cosine_mod:
            cos = F.cosine_similarity(base, aux, dim=feat_dim, eps=self._eps)
            scale = (cos + 1.0) * 0.5
            scale = scale.clamp_(0.0, 1.0)
            # ensure broadcast to delta/alpha shape
            while scale.ndim < delta.ndim:
                scale = scale.unsqueeze(-1)
            alpha = alpha * scale

        if self.couple_tcn and (g_tcn is not None):
            alpha = alpha * (1.0 - g_tcn).to(alpha.dtype)

        return base + alpha * delta
