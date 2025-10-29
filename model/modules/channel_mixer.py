"""
ChannelMixer-Head: 跨变量全连接混合
在最终输出前做轻量的 channel-wise 混合，补充 Channel-Independent 设计丢失的跨变量信息
"""
import torch
import torch.nn as nn


class ChannelMixerHead(nn.Module):
    """
    跨变量全连接混合（在最终输出前）
    
    输入: [B, L, C]
    - 对时间维做聚合（mean）得到 [B, 1, C]
    - 通过 MLP 做跨通道混合
    - 残差连接回原始序列
    
    这样可以捕获变量间的相关性，同时保持时间维的独立性。
    """
    def __init__(self, c_in: int, hidden_ratio: float = 2.0, dropout: float = 0.1):
        super().__init__()
        hidden = max(int(c_in * hidden_ratio), c_in)
        
        # 双层 MLP
        self.fc1 = nn.Linear(c_in, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, c_in)
        self.drop2 = nn.Dropout(dropout)
        
        # LayerNorm（在残差后）
        self.norm = nn.LayerNorm(c_in)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C]
        return: [B, L, C]
        """
        # 保存原始输入（残差连接）
        res = x
        
        # 时间维聚合，得到全局通道表示
        x = x.mean(dim=1, keepdim=True)  # [B, 1, C]
        
        # MLP 跨通道混合
        x = self.fc1(x)      # [B, 1, hidden]
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)      # [B, 1, C]
        x = self.drop2(x)
        
        # 残差连接（broadcast [B, 1, C] -> [B, L, C]）
        x = res + x
        
        # LayerNorm
        return self.norm(x)
