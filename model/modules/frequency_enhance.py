import torch
import torch.nn as nn

class FrequencyEnhance(nn.Module):
    """
    频域增强（可学习分段权重），对任意序列长度 L 自适应：
      1) rFFT 到频域: [B, L, D] -> [B, F, D], F = L//2 + 1
      2) 将 F 均分为 n_bands 段，对每段乘以可学习权重
      3) iFFT 回时域，并与输入做残差融合
    """
    def __init__(
        self,
        seq_len: int | None = None,   # 仅为兼容/日志，不再做硬校验
        n_bands: int = 3,
        init_weights=(1.0, 1.0, 0.1),
        mix: float = 0.5,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_bands = int(n_bands)
        init_weights = tuple(init_weights) if not isinstance(init_weights, (list, tuple)) else init_weights
        assert len(init_weights) == self.n_bands, \
            f"init_weights length {len(init_weights)} must equal n_bands={self.n_bands}"
        self.weights = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32))  # [n_bands]
        self.mix = float(mix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]  (L 可能与构造时的 seq_len 不同；例如 encoder 下采样后)
        """
        B, L, D = x.shape
        # 频域
        x_freq = torch.fft.rfft(x, dim=1)              # [B, F, D], F = L//2 + 1
        F = x_freq.size(1)

        # 动态分段加权（避免 inplace 修改原张量视图）
        x_freq_w = x_freq.clone()
        for b in range(self.n_bands):
            start = (b * F) // self.n_bands
            end = ((b + 1) * F) // self.n_bands
            if start < end:
                x_freq_w[:, start:end, :] *= self.weights[b]

        # 回时域并残差
        x_enh = torch.fft.irfft(x_freq_w, n=L, dim=1)  # [B, L, D]
        return x + self.mix * (x_enh - x)

    def extra_repr(self) -> str:
        return f"seq_len={self.seq_len}, n_bands={self.n_bands}, mix={self.mix}"

