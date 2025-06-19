"""特征提取器模块
支持可替换的点云特征提取器，包括：
- PointNet
- AttentionNet
- CFormer
- FastPointAttention
- Mamba3D
"""

from .base import BaseFeatureExtractor, symfn_max, symfn_avg
from .pointnet import PointNet_features
from .attention import AttentionNet_features, symfn_attention_pool
from .cformer import CFormer_features, symfn_cd_pool
from .fast_attention import FastPointAttention_features, symfn_fast_attention_pool
from .mamba3d import Mamba3D_features, symfn_selective
from .factory import FeatureExtractorFactory, create_feature_extractor

__all__ = [
    # 基础类和函数
    'BaseFeatureExtractor',
    'symfn_max',
    'symfn_avg',
    
    # 特征提取器
    'PointNet_features',
    'AttentionNet_features',
    'CFormer_features', 
    'FastPointAttention_features',
    'Mamba3D_features',
    
    # 专用聚合函数
    'symfn_attention_pool',
    'symfn_cd_pool',
    'symfn_fast_attention_pool',
    'symfn_selective',
    
    # 工厂类和便捷函数
    'FeatureExtractorFactory',
    'create_feature_extractor',
] 