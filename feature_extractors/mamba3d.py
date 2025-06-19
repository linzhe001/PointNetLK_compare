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

class LightweightS6Layer(nn.Module):
    """轻量级S6层 - 简化版本用于快速训练"""
    
    def __init__(self, d_model, d_state=8, expand=1.5, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state  # 减少状态维度
        self.expand = expand  # 减少扩展系数
        self.d_inner = int(d_model * expand)
        self.dt_min = dt_min
        self.dt_max = dt_max
        
        # 简化的输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 移除卷积层，直接使用线性层
        # 简化的S6参数
        self.dt_proj = nn.Linear(self.d_inner, d_state, bias=True)
        
        # 更稳定的初始化
        self.A_log = nn.Parameter(torch.log(torch.ones(d_state, self.d_inner) * 0.1))
        self.D = nn.Parameter(torch.ones(self.d_inner) * 0.1)
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # 激活函数
        self.activation = nn.SiLU()
        
        # 数值稳定性参数
        self.eps = 1e-8
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # 输入投影
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x, z = xz.chunk(2, dim=-1)  # [B, L, d_inner]
        
        # 激活
        x = self.activation(x)
        
        # 简化的S6机制
        y = self._lightweight_scan(x, seq_len, batch_size)
        
        # 门控机制
        y = y * self.activation(z)
        
        # 输出投影
        output = self.out_proj(y)
        return output
    
    def _lightweight_scan(self, x, seq_len, batch_size):
        """轻量级扫描实现 - 优化速度"""
        # 计算dt
        dt = self.dt_proj(x)  # [B, L, d_state]
        dt = torch.clamp(dt, min=self.dt_min, max=self.dt_max)
        
        # 限制A的范围
        A_log_clamped = torch.clamp(self.A_log, min=-10, max=2)
        A = -torch.exp(A_log_clamped)  # [d_state, d_inner]
        
        # 使用并行化的近似扫描（而非逐步扫描）
        # 这是一个简化版本，牺牲一些精度换取速度
        dt_A = torch.einsum('bld,dh->bldh', dt, A)  # [B, L, d_state, d_inner]
        dt_A = torch.clamp(dt_A, min=-10, max=10)
        
        # 简化的状态更新 - 使用全局近似
        dA = torch.exp(dt_A)  # [B, L, d_state, d_inner]
        
        # 简化的输入整合
        x_expanded = x.unsqueeze(2)  # [B, L, 1, d_inner]
        dt_expanded = dt.unsqueeze(-1)  # [B, L, d_state, 1]
        
        # 近似状态计算（避免循环）
        weighted_input = dt_expanded * x_expanded  # [B, L, d_state, d_inner]
        
        # 使用累积和近似状态传播
        cumulative_effect = torch.cumsum(dA * weighted_input, dim=1)
        
        # 输出计算
        y = torch.sum(cumulative_effect, dim=2)  # [B, L, d_inner]
        
        # 应用D参数
        D_clamped = torch.clamp(self.D, min=-10, max=10)
        y = y + x * D_clamped
        
        return y

class SimpleMamba3DBlock(nn.Module):
    """简化的Mamba3D块 - 轻量级版本"""
    def __init__(self, d_model, d_state=8, expand=1.5, dropout=0.1):
        super().__init__()
        self.s6_layer = LightweightS6Layer(d_model, d_state, expand)
        self.norm = nn.LayerNorm(d_model)
        
        # 简化的前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, int(d_model * 2)),  # 减少扩展系数
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * 2), d_model)
        )
        
    def forward(self, x):
        # S6层处理
        s6_out = self.s6_layer(x)
        x = x + s6_out  # 残差连接
        
        # 归一化和前馈
        x_norm = self.norm(x)
        ff_out = self.feed_forward(x_norm)
        x = x + ff_out  # 残差连接
        
        return x

def symfn_selective(x):
    """基于选择性聚合的函数"""
    weights = torch.softmax(torch.sum(x, dim=-1), dim=-1)
    weights = weights.unsqueeze(-1)
    aggregated = torch.sum(x * weights, dim=1)
    return aggregated

class Mamba3D_features(BaseFeatureExtractor):
    """轻量级3DMamba点云特征提取器 - 优化版本"""
    
    def __init__(self, dim_k=1024, sym_fn=symfn_max, scale=1, 
                 num_mamba_blocks=2, d_state=8, expand=1.5):  # 减少默认参数
        super().__init__(dim_k=dim_k, sym_fn=sym_fn, scale=scale)
        
        # 减少模型维度
        self.d_model = max(64, int(128 / scale))  # 从128/256减少到64/128
        self.num_mamba_blocks = min(num_mamba_blocks, 2)  # 减少层数
        
        # 简化的输入嵌入层
        self.input_projection = nn.Linear(3, self.d_model)
        
        # 简化的位置编码（移除复杂的空间感知序列化）
        self.pos_encoding = nn.Parameter(torch.randn(1, 2048, self.d_model) * 0.02)
        
        # 轻量级Mamba块
        self.mamba_blocks = nn.ModuleList([
            SimpleMamba3DBlock(
                self.d_model, 
                d_state=d_state, 
                expand=expand
            )
            for _ in range(self.num_mamba_blocks)
        ])
        
        # 简化的特征变换层
        self.feature_transform = MLPNet(
            self.d_model, 
            [int(256/scale), self.dim_k],  # 减少中间层
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
        x = self.input_projection(points)  # [B, N, d_model]
        
        # 简化的位置编码
        if num_points <= self.pos_encoding.size(1):
            pos_encoding = self.pos_encoding[:, :num_points, :]
        else:
            pos_encoding = F.interpolate(
                self.pos_encoding.transpose(1, 2), 
                size=num_points, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        
        x = x + pos_encoding
        
        # 保存中间特征（兼容性）
        self.t_out_h1 = x.transpose(1, 2)  # [B, d_model, N]
        
        # 通过轻量级Mamba块
        for mamba_block in self.mamba_blocks:
            x = mamba_block(x)  # [B, N, d_model]
        
        # 特征变换
        x = x.transpose(1, 2)  # [B, d_model, N]
        x = self.feature_transform(x)  # [B, dim_k, N]
        x = x.transpose(1, 2)  # [B, N, dim_k]
        
        # 简化的全局聚合
        if hasattr(self.sym_fn, '__name__') and 'selective' in self.sym_fn.__name__:
            global_features = symfn_selective(x)
        elif self.sym_fn == symfn_max:
            global_features = torch.max(x, dim=1)[0]
        else:
            global_features = torch.mean(x, dim=1)
        
        return global_features 