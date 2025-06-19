import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import BaseFeatureExtractor, symfn_max

def flatten(x):
    return x.view(x.size(0), -1)

def mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
    """创建轻量MLP层序列"""
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
    """轻量多层感知机网络"""
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
        super().__init__()
        list_layers = mlp_layers(nch_input, nch_layers, b_shared, bn_momentum, dropout)
        self.layers = torch.nn.Sequential(*list_layers)

    def forward(self, inp):
        return self.layers(inp)

class LightweightPositionEncoding(nn.Module):
    """轻量级位置编码"""
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        self.d_model = d_model
        
        # 简化的3D位置编码器
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # 简单的归一化
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, points, features):
        """
        Args:
            points: [B, N, 3] 点云坐标
            features: [B, N, C] 点云特征
        Returns:
            position_encoded_features: [B, N, C] 位置编码后的特征
        """
        # 3D空间位置编码
        spatial_encoding = self.spatial_encoder(points)
        
        # 简单相加并归一化
        output = self.norm(features + spatial_encoding)
        
        return output

class SimpleProxyCollector(nn.Module):
    """简化的代理点收集器"""
    def __init__(self, d_model, num_proxies=32):
        super().__init__()
        self.d_model = d_model
        self.num_proxies = num_proxies
        
        # 固定数量的可学习代理中心
        self.proxy_centers = nn.Parameter(
            torch.randn(num_proxies, d_model) * 0.02
        )
        
        # 简化的温度参数
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, features):
        """
        Args:
            features: [B, N, C] 输入特征
        Returns:
            proxy_features: [B, P, C] 代理点特征
            attention_weights: [B, N, P] 注意力权重
        """
        batch_size, num_points, d_model = features.shape
        
        # 扩展代理中心到批次维度
        proxy_centers = self.proxy_centers.unsqueeze(0).expand(batch_size, -1, -1)  # [B, P, C]
        
        # 计算软聚类权重
        similarity = torch.matmul(features, proxy_centers.transpose(1, 2))  # [B, N, P]
        attention_weights = F.softmax(similarity / self.temperature, dim=-1)  # [B, N, P]
        
        # 聚合特征到代理点
        proxy_features = torch.matmul(attention_weights.transpose(1, 2), features)  # [B, P, C]
        
        return proxy_features, attention_weights

class LightweightAttention(nn.Module):
    """轻量级注意力机制"""
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # 简化的单头注意力
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, features):
        """
        Args:
            features: [B, N, C] 输入特征
        Returns:
            enhanced_features: [B, N, C] 增强后的特征
        """
        # 自注意力
        attended_feat, _ = self.attention(features, features, features)
        
        # 残差连接和归一化
        output = self.norm(features + attended_feat)
        
        return output

class SimpleDistributor(nn.Module):
    """简化的特征分发器"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # 简化的跨注意力
        self.cross_attention = nn.MultiheadAttention(
            d_model, num_heads=4, batch_first=True
        )
        
        # 简化的门控机制
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, local_features, proxy_features, attention_weights):
        """
        Args:
            local_features: [B, N, C] 局部点特征
            proxy_features: [B, P, C] 代理点特征
            attention_weights: [B, N, P] 收集时的注意力权重
        Returns:
            enhanced_features: [B, N, C] 增强后的局部特征
        """
        # 跨注意力分发
        distributed_features, _ = self.cross_attention(
            local_features, proxy_features, proxy_features
        )
        
        # 简单的门控融合
        concat_features = torch.cat([local_features, distributed_features], dim=-1)
        gate_weights = self.gate(concat_features)
        
        # 融合特征
        enhanced_features = local_features + gate_weights * distributed_features
        
        # 归一化
        output = self.norm(enhanced_features)
        
        return output

class LightweightCDFormerBlock(nn.Module):
    """轻量级收集分发Transformer块"""
    def __init__(self, d_model, num_proxies=32):
        super().__init__()
        
        # 轻量级组件
        self.proxy_collector = SimpleProxyCollector(d_model, num_proxies)
        self.proxy_attention = LightweightAttention(d_model, num_heads=4)
        self.distributor = SimpleDistributor(d_model)
        
        # 简化的前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),  # 减少扩展比例
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        
        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, features):
        """
        Args:
            features: [B, N, C] 输入特征
        Returns:
            output_features: [B, N, C] 输出特征
        """
        # 收集阶段
        proxy_features, attention_weights = self.proxy_collector(features)
        
        # 代理点特征增强
        enhanced_proxy_features = self.proxy_attention(proxy_features)
        
        # 分发阶段
        distributed_features = self.distributor(
            features, enhanced_proxy_features, attention_weights
        )
        features = self.norm1(features + distributed_features)
        
        # 前馈网络
        ff_output = self.feed_forward(features)
        output = self.norm2(features + ff_output)
        
        return output

def symfn_lightweight_pool(x):
    """轻量级聚合函数"""
    # 简单的注意力加权平均
    weights = torch.mean(x, dim=-1, keepdim=True)  # [B, N, 1]
    weights = F.softmax(weights, dim=1)
    pooled = torch.sum(x * weights, dim=1)  # [B, C]
    return pooled

def symfn_cd_pool(x):
    """基于收集分发的聚合函数 - 向后兼容"""
    return symfn_lightweight_pool(x)

class CFormer_features(BaseFeatureExtractor):
    """轻量化CDFormer点云特征提取器"""
    
    def __init__(self, dim_k=1024, sym_fn=symfn_max, scale=1, 
                 num_blocks=2, num_proxies=32, base_proxies=None, max_proxies=None):  # 添加兼容参数
        super().__init__(dim_k=dim_k, sym_fn=sym_fn, scale=scale)
        
        # 处理兼容性参数
        if base_proxies is not None:
            num_proxies = base_proxies
        
        # 调整模型维度
        self.d_model = max(64, int(128 / scale))  # 减少基础维度
        self.num_blocks = min(num_blocks, 3)  # 最多3层
        
        # 输入嵌入
        self.input_projection = nn.Linear(3, self.d_model)
        
        # 轻量级位置编码
        self.position_encoding = LightweightPositionEncoding(self.d_model)
        
        # 少量轻量CDFormer块
        self.cdformer_blocks = nn.ModuleList([
            LightweightCDFormerBlock(
                self.d_model, 
                num_proxies=max(16, num_proxies//(i+1))  # 递减代理点数
            )
            for i in range(self.num_blocks)
        ])
        
        # 简化的特征变换
        self.feature_transform = MLPNet(
            self.d_model, 
            [int(256/scale), self.dim_k],  # 减少中间层
            b_shared=True
        )
        
        # 简化的全局聚合
        self.global_pool = nn.Sequential(
            nn.Linear(self.dim_k, 1),
            nn.Sigmoid()
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
        
        # 轻量级位置编码
        x = self.position_encoding(points, x)
        
        # 保存中间特征（兼容性）
        self.t_out_h1 = x.transpose(1, 2)  # [B, d_model, N]
        
        # 通过轻量CDFormer块
        for cdformer_block in self.cdformer_blocks:
            x = cdformer_block(x)  # [B, N, d_model]
        
        # 特征变换
        x = x.transpose(1, 2)  # [B, d_model, N]
        x = self.feature_transform(x)  # [B, dim_k, N]
        x = x.transpose(1, 2)  # [B, N, dim_k]
        
        # 轻量级全局聚合
        attention_weights = self.global_pool(x)  # [B, N, 1]
        global_features = torch.sum(x * attention_weights, dim=1)  # [B, dim_k]
        
        # 归一化
        global_features = F.normalize(global_features, p=2, dim=1)
        
        return global_features 