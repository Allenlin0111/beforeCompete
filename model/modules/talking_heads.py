# model/modules/talking_heads.py
import torch
import torch.nn as nn

class TalkingHeadsPostMix(nn.Module):
    """Head-wise 1x1 mixing applied AFTER attention (context) but BEFORE head fusion.
    context: [B, H, L, Dh] -> mix heads -> [B, H, L, Dh]
    Initialized to identity => exact baseline equivalence when enabled with dropout=0.
    """
    def __init__(self, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.mix = nn.Linear(n_heads, n_heads, bias=False)
        # Identity init to guarantee equivalence at t=0
        with torch.no_grad():
            self.mix.weight.copy_(torch.eye(n_heads))  # 单位阵初始化，确保默认数值等价
        self.dropout = nn.Dropout(dropout) if (dropout is not None and dropout > 0.0) else nn.Identity()

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # context: [B, H, L, Dh]
        # move heads to last dim -> apply head-mix -> move back
        x = context.permute(0, 2, 3, 1)    # [B, L, Dh, H]
        # use weight^T for correct broadcasting: [H] dim
        x = x @ self.mix.weight.t()        # [B, L, Dh, H]
        x = self.dropout(x)
        x = x.permute(0, 3, 1, 2)          # [B, H, L, Dh]
        return x

