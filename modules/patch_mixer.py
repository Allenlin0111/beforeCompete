import torch, torch.nn as nn

class PatchMixer(nn.Module):
    """
    对 (B, L, D) 做分组的 1x1 线性混合：先转为 (B, D, L)，Conv1d groups，再转回。
    """
    def __init__(self, d_model: int, groups: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % groups == 0, "d_model 必须能被 groups 整除"
        self.proj = nn.Conv1d(d_model, d_model, kernel_size=1, groups=groups, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, tokens):  # (B,L,D)
        x = tokens.transpose(1, 2)      # (B,D,L)
        x = self.proj(x)                # (B,D,L)
        x = self.drop(x)
        return x.transpose(1, 2)        # (B,L,D)