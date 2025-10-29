import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_sincos_pos_embed(n_tokens: int, dim: int, device=None):
    # [n_tokens, dim] fixed sinusoidal
    position = torch.arange(n_tokens, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(n_tokens, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [N, D]

class PatchEmbed1D(nn.Module):
    """
    Patchify [B, L, C] along L: unfold with kernel=patch_len, stride=patch_stride.
    Channel-Independent (CI) by default: same projection for all variables, then reduce channels -> [B, N, D].
    Options:
      norm: 'none' | 'layer' | 'instance'   (applied per patch)
      pos_emb: 'none' | 'sincos'            (added on token dimension N)
    Outputs:
      tokens: [B, N, D], meta: dict(N, P, S)
    """
    def __init__(self, d_model: int, patch_len: int = 16, patch_stride: int = 8,
                 norm: str = "instance", pos_emb: str = "sincos", use_ci: bool = True):
        super().__init__()
        assert patch_len > 0 and patch_stride > 0
        self.d_model = d_model
        self.P = patch_len
        self.S = patch_stride
        self.norm = norm
        self.pos_emb = pos_emb
        self.use_ci = use_ci

        # projection from P -> d_model (shared for all channels if CI)
        self.proj_ci = nn.Linear(self.P, d_model)

        # instance normalization over patch-length dimension
        if self.norm == "instance":
            # IN over length P; use num_features=1 and reshape
            self.inorm = nn.InstanceNorm1d(1, affine=True, track_running_stats=False)
        elif self.norm == "layer":
            self.ln = nn.LayerNorm(self.P)

    def _patchify(self, x: torch.Tensor):
        # x: [B, L, C]  -> patches: [B, C, N, P]
        b, l, c = x.shape
        if l < self.P:
            # fallback: single patch by left-pad zeros
            pad = self.P - l
            x = F.pad(x, (0, 0, 0, pad))  # pad length dimension
            l = self.P
        # use 2D Unfold to avoid as_strided complexity
        x2d = x.permute(0, 2, 1).unsqueeze(2)   # [B, C, 1, L]
        unfold = nn.Unfold(kernel_size=(1, self.P), stride=(1, self.S))
        patches = unfold(x2d)                   # [B, C*P, N]
        b, _, n = patches.shape
        patches = patches.view(b, -1, self.P, n).permute(0, 1, 3, 2).contiguous()  # [B, C, N, P]
        return patches  # [B, C, N, P]

    def _apply_norm(self, patches: torch.Tensor):
        # patches: [B, C, N, P]
        if self.norm == "none":
            return patches
        b, c, n, p = patches.shape
        if self.norm == "layer":
            return self.ln(patches)
        # instance norm: reshape to [B*C*N, 1, P]
        x = patches.reshape(b * c * n, 1, p)
        x = self.inorm(x)
        return x.view(b, c, n, p)

    def forward(self, x: torch.Tensor):
        # x: [B, L, C] -> tokens: [B, C, D]
        device = x.device
        patches = self._patchify(x)
        patches = self._apply_norm(patches)  # [B, C, N, P]
        b, c, n, _ = patches.shape

        # CI projection: P -> D for every (b, c, n)
        proj = self.proj_ci(patches)  # [B, C, N, D]

        # 位置编码：先加到 N 维的每个 token 上，再做池化
        if self.pos_emb == "sincos":
            pe = build_sincos_pos_embed(n, self.d_model, device=device)  # [N, D]
            proj = proj + pe.view(1, 1, n, self.d_model)  # 广播到 [B, C, N, D]

        # 关键修复：在 patch 维 N 上做池化，保留"变量为 token"
        tokens = proj.mean(dim=2)  # [B, C, D]

        # 确保"变量为 token"
        assert tokens.size(1) == x.size(2), f"Patch tokens ({tokens.size(1)}) must equal C ({x.size(2)})"

        meta = {"n_tokens": n, "patch_len": self.P, "patch_stride": self.S}
        return tokens, meta