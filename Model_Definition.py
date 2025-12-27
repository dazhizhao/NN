import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """SwiGLU 激活函数实现"""
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class ResidualSwiGLUBlock(nn.Module):
    """包含 SwiGLU、LayerNorm、Dropout 和 残差连接 的块"""
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super(ResidualSwiGLUBlock, self).__init__()
        
        # SwiGLU 需要两倍的输出维度来进行 split
        self.linear = nn.Linear(in_dim, 2 * out_dim)
        self.swiglu = SwiGLU()
        self.norm = nn.LayerNorm(out_dim) 
        self.dropout = nn.Dropout(dropout)
        
        # 残差路径
        self.shortcut_proj = None
        if in_dim != out_dim:
            self.shortcut_proj = nn.Linear(in_dim, out_dim)
            
    def forward(self, x):
        shortcut = x
        if self.shortcut_proj is not None:
            shortcut = self.shortcut_proj(x)
        
        out = self.linear(x)
        out = self.swiglu(out)
        out = self.norm(out)
        out = self.dropout(out)
        
        return out + shortcut

class SpiderWebModel(nn.Module):
    def __init__(self, input_dim=13, curve_dim=200, scalar_dim=4):
        super(SpiderWebModel, self).__init__()
        
        # Backbone
        self.backbone = nn.Sequential(
            ResidualSwiGLUBlock(input_dim, 128),
            ResidualSwiGLUBlock(128, 512),
            ResidualSwiGLUBlock(512, 512)
        )
        
        # Curve Head
        self.curve_head = nn.Sequential(
            ResidualSwiGLUBlock(512, 256), 
            nn.Linear(256, curve_dim)
        )
        
        # Scalar Head
        self.scalar_head = nn.Sequential(
            ResidualSwiGLUBlock(512, 64),
            nn.Linear(64, scalar_dim)
        )
        
    def forward(self, x):
        feat = self.backbone(x)
        return self.curve_head(feat), self.scalar_head(feat)