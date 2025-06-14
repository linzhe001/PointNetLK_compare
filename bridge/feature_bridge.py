"""
特征提取器桥接模块 - 统一原版和改进版的特征提取接口
Feature Extractor Bridge - Unified interface for original and improved feature extractors
"""

import torch
import torch.nn as nn
import sys
import os

# 添加路径以导入legacy_ptlk和当前目录的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import legacy_ptlk as ptlk
from model import Pointnet_Features  # PointNetLK_Revisited的改进特征提取器


class FeatureExtractorBridge:
    """统一特征提取接口"""
    
    def __init__(self, model_type='original', dim_k=1024, **kwargs):
        """
        初始化特征提取器桥接
        
        Args:
            model_type: 'original' 或 'improved'
            dim_k: 特征维度
            **kwargs: 其他参数
        """
        self.model_type = model_type
        self.dim_k = dim_k
        
        if model_type == 'original':
            # 使用原版PointNet特征提取器
            use_tnet = kwargs.get('use_tnet', False)
            sym_fn = kwargs.get('sym_fn', ptlk.pointnet.symfn_max)
            self.extractor = ptlk.pointnet.PointNet_features(
                dim_k=dim_k, 
                use_tnet=use_tnet, 
                sym_fn=sym_fn
            )
        elif model_type == 'improved':
            # 使用改进版特征提取器
            self.extractor = Pointnet_Features(dim_k=dim_k)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def extract_features(self, points, iter=0):
        """
        统一的特征提取接口
        
        Args:
            points: 输入点云 [B, N, 3]
            iter: 迭代次数（仅改进版使用）
            
        Returns:
            features: 提取的特征 [B, K]
        """
        if self.model_type == 'original':
            return self.extractor(points)
        else:
            return self.extractor(points, iter=iter)
    
    def get_extractor(self):
        """获取底层特征提取器"""
        return self.extractor
    
    def load_state_dict(self, state_dict):
        """加载预训练权重"""
        self.extractor.load_state_dict(state_dict)
    
    def state_dict(self):
        """获取模型状态字典"""
        return self.extractor.state_dict()
    
    def train(self, mode=True):
        """设置训练模式"""
        self.extractor.train(mode)
        return self
    
    def eval(self):
        """设置评估模式"""
        self.extractor.eval()
        return self
    
    def to(self, device):
        """移动到指定设备"""
        self.extractor.to(device)
        return self
    
    def parameters(self):
        """获取模型参数"""
        return self.extractor.parameters()
    
    def __repr__(self):
        return f"FeatureExtractorBridge(type={self.model_type}, dim_k={self.dim_k})" 