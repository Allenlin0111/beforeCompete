import torch
import torch.nn as nn
import torch.nn.functional as F

class _DepthwiseAvg1D(nn.Module):
    """
    通道独立的 1D 平均卷积：对时间维做移动平均，保持长度不变
    in/out: [B, L, C]  (内部转 [B, C, L] 做卷积再转回)
    """
    def __init__(self, c_out: int, k: int = 25):
        super().__init__()
        k = max(int(k), 1)
        self.dw = nn.Conv1d(c_out, c_out, kernel_size=k, padding=k//2, groups=c_out, bias=False)
        with torch.no_grad():
            self.dw.weight.fill_(1.0 / float(k))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: [B, L, C]
        r = (y - y.mean(dim=1, keepdim=True)).transpose(1, 2)  # [B, C, L]
        s = self.dw(r).transpose(1, 2)                         # [B, L, C]
        return s

class STARBlock(nn.Module):
    """
    STAR 残差块（post-forecast）：
      y_out = y_in + sigmoid(g) * ( STAR( LN(y_in) ) - LN(y_in) )
    其中 STAR 是基于移动平均的通道独立平滑；g 为按通道门控（[1,1,C]）。
    输入/输出: [B, L, C]
    """
    def __init__(self, c_out: int, k: int = 25, gate_bias: float = -4.0, use_ln: bool = True):
        super().__init__()
        self.ln = nn.LayerNorm(c_out) if use_ln else nn.Identity()
        self.star = _DepthwiseAvg1D(c_out, k)
        # 按通道门控参数，广播到 [B, L, C]
        self.gate = nn.Parameter(torch.full((1, 1, c_out), float(gate_bias)))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        z = self.ln(y)                  # [B, L, C]
        s = self.star(z)                # [B, L, C]
        beta = torch.sigmoid(self.gate) # [1, 1, C]
        return y + beta * (s - z)

