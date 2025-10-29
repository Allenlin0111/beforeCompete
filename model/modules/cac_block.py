import torch
import torch.nn as nn

def _parse_int_list(s, default):
    if isinstance(s, (list, tuple)): return list(map(int, s))
    try:
        return [int(x) for x in str(s).split(',') if str(x).strip()!='']
    except Exception:
        return default

class AutoCorrelationLite(nn.Module):
    """
    简化自相关：在若干候选滞后上打分，取 topk，对序列 roll 聚合。轻量、残差增强周期结构。
    """
    def __init__(self, topk=5, max_candidates=32, max_lag_cap=168):
        super().__init__()
        self.topk = topk
        self.max_candidates = max_candidates
        self.max_lag_cap = max_lag_cap

    def forward(self, x):  # x: [B, L, D]
        B, L, D = x.shape
        if L < 3:
            return x
        x_mean = x.mean(dim=-1)  # [B, L]
        max_lag = min(L-1, self.max_lag_cap)
        num = min(max_lag, self.max_candidates)
        if num <= 0:
            return x
        device = x.device
        lags = torch.linspace(1, max_lag, steps=num, dtype=torch.long, device=device)  # [num]
        # 打分
        scores = []
        for lag in lags:
            a = x_mean[:, :-lag]
            b = x_mean[:, lag:]
            s = (a * b).mean(dim=1)  # [B]
            scores.append(s)
        scores = torch.stack(scores, dim=1)  # [B, num]
        k = min(self.topk, scores.size(1))
        vals, idx = torch.topk(scores, k=k, dim=1)  # [B, k]
        # 聚合
        out = torch.zeros_like(x)
        for b in range(B):
            denom = vals[b].sum() + 1e-6
            for j in range(k):
                lag = int(lags[idx[b, j]].item())
                rolled = torch.roll(x[b], shifts=lag, dims=0)  # [L, D]
                out[b] = out[b] + (vals[b, j] / denom) * rolled
        return out

class CACBlock(nn.Module):
    """
    CAC-Block (Convolution + Auto-Correlation)
    时间轴周期归纳偏置：多扩张深度可分离卷积 + 简化自相关
    """
    def __init__(self, d_model, kernel_sizes='3,5,7', dilations='1,2,4',
                 topk=5, dropout=0.1):
        super().__init__()
        ks = _parse_int_list(kernel_sizes, [3,5,7])
        ds = _parse_int_list(dilations, [1,2,4])
        assert len(ks) == len(ds), "cac_kernel_sizes 与 cac_dilations 长度需一致"
        
        # 深度可分离卷积分支（多尺度）
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model,
                      kernel_size=k,
                      padding=(k//2)*d,
                      dilation=d,
                      groups=d_model,
                      bias=False)
            for k, d in zip(ks, ds)
        ])

        self.autocorr = AutoCorrelationLite(topk=topk)
        self.proj = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):  # x: [B, L, D]
        # 卷积分支
        x_T = x.transpose(1, 2)  # [B, D, L]
        y_conv = sum([conv(x_T) for conv in self.convs])
        y_conv = y_conv.transpose(1, 2)  # [B, L, D]
        
        # 自相关分支
        y_auto = self.autocorr(x)

        # 融合
        y = torch.cat([y_conv, y_auto], dim=-1)
        y = self.dropout(self.proj(y))
        return self.norm(x + y)
