import torch
import torch.nn as nn

from model.modules.cac_block import CACBlock


class GCACBlock(nn.Module):
    """
    CAC with optional pre-gate (simulating residual gate) and post-gate (fusion gate).
    Pre-gate: x -> x + sigma(.) * (mlp(x) - x)
    Post-gate: blend pre-gated x with CAC output.
    """

    def __init__(
        self,
        d_model: int,
        kernel_sizes: str = "3,5,7",
        dilations: str = "1,2,4",
        topk: int = 5,
        dropout: float = 0.1,
        enable_pre_gate: bool = True,
        enable_post_gate: bool = True,
        pre_gate_bias: float = -1.0,
        post_gate_bias: float = -1.0,
    ):
        super().__init__()
        self.cac = CACBlock(
            d_model=d_model,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            topk=topk,
            dropout=dropout,
        )

        self.enable_pre_gate = enable_pre_gate
        self.enable_post_gate = enable_post_gate

        if self.enable_pre_gate:
            self.pre_mlp = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.pre_in_norm = nn.LayerNorm(d_model)
            self.pre_out_norm = nn.LayerNorm(d_model)
            self.pre_gate_proj = nn.Linear(d_model * 2, d_model, bias=True)
            nn.init.zeros_(self.pre_gate_proj.weight)
            nn.init.constant_(self.pre_gate_proj.bias, pre_gate_bias)

        if self.enable_post_gate:
            self.post_in_norm = nn.LayerNorm(d_model)
            self.post_out_norm = nn.LayerNorm(d_model)
            self.post_gate_proj = nn.Linear(d_model * 2, d_model, bias=True)
            nn.init.zeros_(self.post_gate_proj.weight)
            nn.init.constant_(self.post_gate_proj.bias, post_gate_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x
        if self.enable_pre_gate:
            # Residual-style pre-gate: learn a delta branch and blend it with the
            # original token using a sigmoid gate conditioned on both inputs.
            delta = self.pre_mlp(x)  # [B, L, D]
            gate_pre = torch.sigmoid(
                self.pre_gate_proj(
                    torch.cat([self.pre_in_norm(x), self.pre_out_norm(delta)], dim=-1)
                )
            )
            base = x + gate_pre * (delta - x)

        cac_out = self.cac(base)
        if not self.enable_post_gate:
            return cac_out

        # Post fusion gate: condition on (base, cac_out) and softly mix them.
        gate_post = torch.sigmoid(
            self.post_gate_proj(
                torch.cat([self.post_in_norm(base), self.post_out_norm(cac_out)], dim=-1)
            )
        )
        return base + gate_post * (cac_out - base)
