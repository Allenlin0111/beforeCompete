import torch, torch.nn as nn

class PatchFilter(nn.Module):
    """
    基于 token 余弦相似度的稀疏注意力掩码。
    仅保留每个 query 的 topk 连接(含自身)，其余位置加 -inf。
    """
    def __init__(self, topk: int = 8, season: str = ""):
        super().__init__()
        self.topk = int(topk)
        self.season = [int(s) for s in season.split(",") if s.strip()] if season else []

    @torch.no_grad()
    def forward(self, tokens):  # tokens: (B, L, D)
        B, L, D = tokens.shape
        x = torch.nn.functional.normalize(tokens, dim=-1)
        sim = torch.einsum("bid,bjd->bij", x, x)  # (B,L,L)

        # 额外保留季节偏移位置
        keep = torch.arange(L, device=tokens.device).unsqueeze(0).repeat(L,1)  # (L,L) -> col索引
        cand = torch.zeros(L, L, dtype=torch.bool, device=tokens.device)
        for i in range(L):
            cand[i, i] = True
            for d in self.season:
                if 0 <= i-d < L: cand[i, i-d] = True
                if 0 <= i+d < L: cand[i, i+d] = True

        # topk（在 cand 为 False 的位置也允许被选，若想严格仅 cand 内筛选，可将非cand位置置极小）
        # 这里采用：先全局topk，再并上 cand 的 True，保证季节边被保留
        vals, idx = torch.topk(sim, k=min(self.topk, L), dim=-1)  # (B,L,k)
        mask = torch.zeros(B, L, L, dtype=torch.bool, device=tokens.device)
        mask.scatter_(dim=-1, index=idx, value=True)              # 选中的置 True
        if self.season:
            mask = mask | cand.unsqueeze(0)                       # 并上季节边

        # 反转得到需要屏蔽的位置
        drop = ~mask                                              # True 表示要屏蔽
        attn_mask = torch.zeros(B, 1, L, L, device=tokens.device)
        attn_mask[drop.unsqueeze(1)] = float("-inf")
        return attn_mask