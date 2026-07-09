import torch
import torch.nn as nn
import torch.nn.functional as functional

"""
PyTorch 手写完整 Transformer
Encoder 堆叠 N 层 + 完整模型
"""

# 多头自注意力
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim) # 生成Q/K/V三个向量，用于计算多头
        self.out_proj = nn.Linear(embed_dim, embed_dim)     # 多头计算完后过线性层，输出多头计算的最终值

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_score = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -1e9)
        attn_weight = functional.softmax(attn_score, dim=-1)

        out = attn_weight @ v
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        return out

# 前馈网络
class FeedForward(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    def forward(self, x):
        return self.net(x)

# 单层 Transformer Encoder
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim) # 层归一化
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads) # 多头机制
        self.dropout1 = nn.Dropout(dropout) # 防过拟合

        self.norm2 = nn.LayerNorm(embed_dim) # 层归一化
        self.mlp = FeedForward(embed_dim, mlp_ratio) # 前馈网络
        self.dropout2 = nn.Dropout(dropout)  # 防过拟合

    def forward(self, x, mask=None):
        x = x + self.dropout1(self.attn(self.norm1(x), mask))
        x = x + self.dropout2(self.mlp(self.norm2(x)))
        return x

# 堆叠多层 Transformer —— 
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        # 1. 先创建空的 ModuleList
        self.layers = nn.ModuleList()

        # 2. 循环 append 往里加每一层
        for _ in range(num_layers):
            layer = TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
            self.layers.append(layer)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        # 遍历每一层，依次前向传播
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x

# 测试
if __name__ == "__main__":
    batch_size = 4  # 批次
    seq_len = 10    # 句子长度
    embed_dim = 8 # 词嵌入/特征维度
    num_heads = 2   # 注意力头
    num_layers = 3  # 堆叠几层Transformer

    model = TransformerEncoder(embed_dim, num_heads, num_layers)
    x = torch.randn(batch_size, seq_len, embed_dim) # 模拟样本数据
    print(x);
    out = model(x)
    print("输入形状:", x.shape)
    print("输出形状:", out.shape)
