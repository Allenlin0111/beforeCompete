import torch, torch.nn as nn
from typing import List, Tuple

class MSPatchEmbed(nn.Module):
    """
    并行多尺度 patch 嵌入的包装器。
    将若干单尺度 PatchEmbed 的输出按 token 维 concat，
    fuse='gate' 时对每个分支加一个分支级可学习权重。
    """
    def __init__(self, single_embed_ctor,  # Callable(P:int, S:int, **kw) -> nn.Module
                 patch_set: List[int], stride_ratio: float,
                 fuse: str = "gate"):
        super().__init__()
        assert len(patch_set) >= 2, "ms_patch 至少两个尺度"
        self.fuse = fuse
        self.branches = nn.ModuleList()
        for P in patch_set:
            S = max(1, int(round(P * stride_ratio)))
            self.branches.append(single_embed_ctor(P=P, S=S))
        if fuse == "gate":
            self.alpha = nn.Parameter(torch.zeros(len(self.branches)))  # softmax 后做分支级缩放

    def forward(self, x, x_mark=None):
        outs = []
        if self.fuse == "gate":
            w = torch.softmax(self.alpha, dim=0)  # [n_branch]
            for i, b in enumerate(self.branches):
                yi, _ = b(x)     # (B, L_i, D)
                outs.append(yi * w[i])
        else:
            for b in self.branches:
                yi, _ = b(x)
                outs.append(yi)
        y = torch.cat(outs, dim=1)    # 按 token 维拼接
        return y