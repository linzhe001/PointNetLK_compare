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

class SimplifiedPositionalEncoding(nn.Module):
    """简化的3D位置编码"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.pos_projection = nn.Linear(3, d_model // 4)
        
    def forward(self, points):
        pos_encoding = self.pos_projection(points)
        return pos_encoding

class FastAttention(nn.Module):
    """快速注意力机制 - 单头注意力"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.scale = 1.0 / math.sqrt(d_model)
        
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        residual = x
        
        # 计算Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        context = torch.matmul(attention_weights, V)
        output = self.layer_norm(context + residual)
        
        return output

class SimpleFeedForward(nn.Module):
    """简化的前馈网络"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        d_ff = d_model * 2  # 减少隐藏层维度
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        residual = x
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.layer_norm(x + residual)

class FastAttentionBlock(nn.Module):
    """快速注意力块"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.fast_attention = FastAttention(d_model, dropout)
        self.feed_forward = SimpleFeedForward(d_model, dropout)
        
    def forward(self, x):
        x = self.fast_attention(x)
        x = self.feed_forward(x)
        return x

def symfn_fast_attention_pool(x):
    """快速注意力池化聚合函数"""
    batch_size, seq_len, d_model = x.size()
    global_feat = torch.mean(x, dim=1, keepdim=True)
    attention_scores = torch.sum(x * global_feat, dim=-1)
    attention_weights = F.softmax(attention_scores, dim=-1).unsqueeze(-1)
    aggregated = torch.sum(x * attention_weights, dim=1)
    return aggregated

class FastPointAttention_features(BaseFeatureExtractor):
    """快速点云注意力特征提取器"""
    
    def __init__(self, dim_k=1024, sym_fn=symfn_max, scale=1, 
                 num_attention_blocks=2):
        super().__init__(dim_k=dim_k, sym_fn=sym_fn, scale=scale)
        
        self.d_model = int(64 / scale)
        self.num_attention_blocks = num_attention_blocks
        
        # 输入嵌入层
        self.input_projection = nn.Linear(3, self.d_model - self.d_model // 4)
        
        # 简化的3D位置编码
        self.pos_encoding = SimplifiedPositionalEncoding(self.d_model)
        
        # 快速注意力块
        self.attention_blocks = nn.ModuleList([
            FastAttentionBlock(self.d_model)
            for _ in range(num_attention_blocks)
        ])
        
        # 特征变换层
        self.feature_transform = MLPNet(
            self.d_model, 
            [int(128/scale), self.dim_k], 
            b_shared=True
        )
        
    def forward(self, points):
        """
        Args:
            points: [B, N, 3] 输入点云
        Returns:
            features: [B, dim_k] 全局特征向量
        """
        batch_size, num_points, _ = points.size()
        
        # 输入投影
        x = self.input_projection(points)
        
        # 添加简化位置编码
        pos_encoding = self.pos_encoding(points)
        x = torch.cat([x, pos_encoding], dim=-1)  # [B, N, d_model]
        
        # 保存中间特征（兼容性）
        self.t_out_h1 = x.transpose(1, 2)  # [B, d_model, N]
        
        # 通过快速注意力块
        for attention_block in self.attention_blocks:
            x = attention_block(x)  # [B, N, d_model]
        
        # 特征变换
        x = x.transpose(1, 2)  # [B, d_model, N]
        x = self.feature_transform(x)  # [B, dim_k, N]
        x = x.transpose(1, 2)  # [B, N, dim_k]
        
        # 全局聚合
        if hasattr(self.sym_fn, '__name__') and 'fast_attention' in self.sym_fn.__name__:
            global_features = symfn_fast_attention_pool(x)
        elif self.sym_fn == symfn_max:
            global_features = torch.max(x, dim=1)[0]
        else:
            global_features = torch.mean(x, dim=1)
        
        return global_features 