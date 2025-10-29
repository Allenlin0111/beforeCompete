import torch
import torch.nn as nn

class STAR(nn.Module):
    """
    Seasonal-Trend decomposition via simple EMA, then fuse:
      x -> trend = EMA(x, alpha); seasonal = x - trend
      fuse: 'sum' or 'gate' (learnable scalar gate in [0,1])
    Shape: x: [B, L, C] -> out: [B, L, C]
    """
    def __init__(self, alpha: float = 0.7, fuse: str = "sum", gate_init: float = 0.5):
        super().__init__()
        assert 0.0 < alpha < 1.0
        assert fuse in ("sum", "gate")
        self.alpha = alpha
        self.fuse = fuse
        if fuse == "gate":
            self.gate = nn.Parameter(torch.tensor(gate_init).clamp(0, 1))
        else:
            self.register_parameter("gate", None)

    @torch.no_grad()
    def _ema(self, x):
        # x: [B, L, C]
        b, l, c = x.shape
        out = x.new_empty(b, l, c)
        out[:, 0, :] = x[:, 0, :]
        a = self.alpha
        for t in range(1, l):
            out[:, t, :] = a * x[:, t, :] + (1 - a) * out[:, t - 1, :]
        return out

    def forward(self, x):
        trend = self._ema(x)
        seasonal = x - trend
        if self.fuse == "sum":
            return trend + seasonal  # = x
        else:
            gate = torch.clamp(self.gate, 0.0, 1.0)
            return gate * seasonal + (1 - gate) * trend