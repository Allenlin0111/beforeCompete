import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """
    Learnable fusion that keeps the base path intact and lets the gate decide
    how much of the auxiliary signal to inject. When the sigmoid output is 0,
    the module is an identity map.
    """

    def __init__(self, dim: int, gate_init_bias: float = -2.0):
        super().__init__()
        self.left_norm = nn.LayerNorm(dim)
        self.right_norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim * 2, dim, bias=True)

        # Start close to the identity path so optimisation stays stable.
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, gate_init_bias)

    def forward(self, x_base: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        x_base_norm = self.left_norm(x_base)
        x_aux_norm = self.right_norm(x_aux)
        gate = torch.sigmoid(self.gate_proj(torch.cat([x_base_norm, x_aux_norm], dim=-1)))
        # Delta-residual form: identity when gate → 0, full aux when gate → 1.
        return x_base + gate * (x_aux - x_base)
