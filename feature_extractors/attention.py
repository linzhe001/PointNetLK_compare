import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import BaseFeatureExtractor, symfn_max

def flatten(x):
    return x.view(x.size(0), -1)

def mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
    """创建MLP层序列"""
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
        else:
            weights = torch.nn.Linear(last, outp)
        layers.append(weights)
        layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum))
        layers.append(torch.nn.ReLU())
        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers

class MLPNet(torch.nn.Module):
    """多层感知机网络"""
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
        super().__init__()
        list_layers = mlp_layers(nch_input, nch_layers, b_shared, bn_momentum, dropout)
        self.layers = torch.nn.Sequential(*list_layers)

    def forward(self, inp):
        return self.layers(inp)

class PositionalEncoding3D(nn.Module):
    """3D位置编码"""
    def __init__(self, d_model):
        super().__init__()
        self.pos_projection = nn.Linear(3, d_model)
        
    def forward(self, points):
        """points: [B, N, 3] -> [B, N, d_model]"""
        return self.pos_projection(points)

class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        residual = x
        
        # 计算Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        output = self.w_o(context)
        return self.layer_norm(output + residual)

class FeedForwardNetwork(nn.Module):
    """前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        residual = x
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.layer_norm(x + residual)

class AttentionBlock(nn.Module):
    """注意力块"""
    def __init__(self, d_model, num_heads=8, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4
            
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        
    def forward(self, x):
        x = self.self_attention(x)
        x = self.feed_forward(x)
        return x

def symfn_attention_pool(x):
    """基于注意力的聚合函数"""
    batch_size, seq_len, d_model = x.size()
    attention_weights = torch.softmax(torch.sum(x, dim=-1), dim=-1)
    attention_weights = attention_weights.unsqueeze(-1)
    return torch.sum(x * attention_weights, dim=1)

class AttentionNet_features(BaseFeatureExtractor):
    """基于注意力机制的特征提取器"""
    
    def __init__(self, dim_k=1024, sym_fn=symfn_max, scale=1, 
                 num_attention_blocks=3, num_heads=8, d_model=256):
        super().__init__(dim_k=dim_k, sym_fn=sym_fn, scale=scale)
        
        self.d_model = int(d_model / scale)
        self.num_attention_blocks = num_attention_blocks
        
        # 输入投影
        self.input_projection = nn.Linear(3, self.d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding3D(self.d_model)
        
        # 注意力块
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(self.d_model, num_heads, dropout=0.1)
            for _ in range(num_attention_blocks)
        ])
        
        # 特征变换
        self.feature_transform = MLPNet(
            self.d_model, 
            [int(256/scale), self.dim_k], 
            b_shared=True
        )
        
    def forward(self, points):
        """
        Args:
            points: [B, N, 3] 点云输入
        Returns:
            features: [B, dim_k] 特征向量
        """
        batch_size, num_points, _ = points.size()
        
        # 输入投影和位置编码
        x = self.input_projection(points)  # [B, N, d_model]
        pos_enc = self.pos_encoding(points)
        x = x + pos_enc
        
        # 保存中间特征（兼容性）
        self.t_out_h1 = x.transpose(1, 2)  # [B, d_model, N]
        
        # 注意力块
        for attention_block in self.attention_blocks:
            x = attention_block(x)
        
        # 特征变换
        x = x.transpose(1, 2)  # [B, d_model, N]
        x = self.feature_transform(x)  # [B, dim_k, N]
        x = x.transpose(1, 2)  # [B, N, dim_k]
        
        # 聚合
        if hasattr(self.sym_fn, '__name__') and 'attention' in self.sym_fn.__name__:
            global_features = symfn_attention_pool(x)
        elif self.sym_fn == symfn_max:
            global_features = torch.max(x, dim=1)[0]
        else:
            global_features = torch.mean(x, dim=1)
        
        return global_features 