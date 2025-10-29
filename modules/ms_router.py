import torch, torch.nn as nn
from typing import List

class MSPatchRouter(nn.Module):
    """
    多尺度 + 序列级路由器：根据输入序列自适应权衡各尺度分支（分支级缩放 + concat）。
    预算正则(相对均匀分布的KL)通过 get_aux_loss() 对外暴露。
    """
    def __init__(self, single_embed_ctor, patch_set: List[int], stride_ratio: float,
                 hidden: int = 64, budget_weight: float = 0.0):
        super().__init__()
        assert len(patch_set) >= 2
        self.budget_weight = float(budget_weight)
        self.branches = nn.ModuleList()
        for P in patch_set:
            S = max(1, int(round(P * stride_ratio)))
            self.branches.append(single_embed_ctor(P=P, S=S))

        self.router = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(),
            nn.Linear(hidden, len(self.branches))
        )
        self._aux_loss = torch.zeros(1)

    def forward(self, x, x_mark=None):
        # 简单的序列级统计特征：时间维上的 mean/std 后再对 C 做平均 -> (B, 2)
        # x: (B, L, C)
        s_mean = x.mean(dim=(1,2), keepdim=False)
        s_std  = x.std (dim=(1,2), keepdim=False)
        feat = torch.stack([s_mean, s_std], dim=-1)  # (B, 2)

        logits = self.router(feat)                   # (B, n_branch)
        w = torch.softmax(logits, dim=-1)            # 权重
        # 预算正则: KL(w || U) = sum w_i log w_i + log(n)
        if self.budget_weight > 0:
            n = w.size(-1)
            kl = (w * (w.clamp_min(1e-12).log())).sum(dim=-1) + torch.log(torch.tensor(float(n), device=w.device))
            self._aux_loss = self.budget_weight * kl.mean()
        else:
            self._aux_loss = torch.zeros(1, device=x.device)

        outs = []
        for i, b in enumerate(self.branches):
            yi, _ = b(x)               # (B, L_i, D)
            outs.append(yi * w[:, i].view(-1, 1, 1))
        y = torch.cat(outs, dim=1)
        return y

    def get_aux_loss(self):
        return self._aux_loss