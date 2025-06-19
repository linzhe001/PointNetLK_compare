import torch
import torch.nn as nn
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

class TNet(torch.nn.Module):
    """变换网络 [B, K, N] -> [B, K, K]"""
    def __init__(self, K):
        super().__init__()
        self.mlp1 = torch.nn.Sequential(*mlp_layers(K, [64, 128, 1024], b_shared=True))
        self.mlp2 = torch.nn.Sequential(*mlp_layers(1024, [512, 256], b_shared=False))
        self.lin = torch.nn.Linear(256, K*K)

        # 初始化为零
        for param in self.mlp1.parameters():
            torch.nn.init.constant_(param, 0.0)
        for param in self.mlp2.parameters():
            torch.nn.init.constant_(param, 0.0)
        for param in self.lin.parameters():
            torch.nn.init.constant_(param, 0.0)

    def forward(self, inp):
        K = inp.size(1)
        N = inp.size(2)
        eye = torch.eye(K).unsqueeze(0).to(inp)

        x = self.mlp1(inp)
        x = torch.nn.functional.max_pool1d(x, N).view(-1, 1024)
        x = self.mlp2(x)
        x = self.lin(x)

        x = x.view(-1, K, K) + eye
        return x

class PointNet_features(BaseFeatureExtractor):
    """PointNet特征提取器
    
    与legacy_ptlk/pointnet.py保持完全兼容
    """
    
    def __init__(self, dim_k=1024, use_tnet=False, sym_fn=symfn_max, scale=1):
        super().__init__(dim_k=dim_k, sym_fn=sym_fn, scale=scale)
        
        mlp_h1 = [int(64/scale), int(64/scale)]
        mlp_h2 = [int(64/scale), int(128/scale), int(self.dim_k)]

        self.h1 = MLPNet(3, mlp_h1, b_shared=True).layers
        self.h2 = MLPNet(mlp_h1[-1], mlp_h2, b_shared=True).layers

        self.tnet1 = TNet(3) if use_tnet else None
        self.tnet2 = TNet(mlp_h1[-1]) if use_tnet else None

    def forward(self, points):
        """
        Args:
            points: [B, N, 3] 点云输入
        Returns:
            features: [B, dim_k] 特征向量
        """
        x = points.transpose(1, 2)  # [B, 3, N]
        
        if self.tnet1:
            t1 = self.tnet1(x)
            x = t1.bmm(x)

        x = self.h1(x)
        if self.tnet2:
            t2 = self.tnet2(x)
            self.t_out_t2 = t2
            x = t2.bmm(x)
        self.t_out_h1 = x  # 保存局部特征

        x = self.h2(x)
        x = self.sym_fn(x)  # 聚合函数
        x = flatten(x)  # 展平

        return x 