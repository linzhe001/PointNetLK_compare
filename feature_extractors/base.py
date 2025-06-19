from abc import ABC, abstractmethod
import torch
import torch.nn as nn

def symfn_max(x):
    """最大池化聚合函数 [B, K, N] -> [B, K, 1]"""
    return torch.nn.functional.max_pool1d(x, x.size(-1))

def symfn_avg(x):
    """平均池化聚合函数 [B, K, N] -> [B, K, 1]"""
    return torch.nn.functional.avg_pool1d(x, x.size(-1))

class BaseFeatureExtractor(nn.Module, ABC):
    """特征提取器基础接口
    
    所有特征提取器必须继承此类以确保与PointLK系统兼容
    """
    
    def __init__(self, dim_k=1024, sym_fn=symfn_max, scale=1):
        super().__init__()
        self.dim_k = int(dim_k / scale)
        self.sym_fn = sym_fn if sym_fn is not None else symfn_max
        self.scale = scale
        
        # PointLK兼容性属性
        self.t_out_t2 = None  # TNet输出（如果有）
        self.t_out_h1 = None  # 中间特征输出
    
    @abstractmethod
    def forward(self, points):
        """特征提取前向传播
        
        Args:
            points: [B, N, 3] 输入点云
            
        Returns:
            features: [B, dim_k] 特征向量
        """
        pass
    
    def get_feature_dim(self):
        """获取特征维度"""
        return self.dim_k
        
    def get_local_features(self):
        """获取局部特征（如果支持）"""
        return self.t_out_h1
        
    def get_transform_matrix(self):
        """获取变换矩阵（如果支持TNet）"""
        return self.t_out_t2 