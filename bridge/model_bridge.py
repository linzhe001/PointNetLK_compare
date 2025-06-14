"""
模型桥接模块 - 统一原版和改进版的PointNetLK模型接口
Model Bridge - Unified interface for original and improved PointNetLK models
"""

import torch
import torch.nn as nn
import sys
import os

# 添加路径以导入legacy_ptlk和当前目录的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import legacy_ptlk as ptlk
from model import AnalyticalPointNetLK  # PointNetLK_Revisited的改进模型
from .feature_bridge import FeatureExtractorBridge


class ModelBridge:
    """统一PointNetLK模型接口"""
    
    def __init__(self, model_type='original', dim_k=1024, device='cuda:0', **kwargs):
        """
        初始化模型桥接
        
        Args:
            model_type: 'original' 或 'improved'
            dim_k: 特征维度
            device: 计算设备
            **kwargs: 其他参数
        """
        self.model_type = model_type
        self.dim_k = dim_k
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 创建特征提取器
        self.feature_bridge = FeatureExtractorBridge(model_type, dim_k, **kwargs)
        ptnet = self.feature_bridge.get_extractor()
        
        if model_type == 'original':
            # 使用原版PointNetLK
            delta = kwargs.get('delta', 1.0e-2)
            learn_delta = kwargs.get('learn_delta', False)
            self.model = ptlk.pointlk.PointLK(ptnet, delta, learn_delta)
        elif model_type == 'improved':
            # 使用改进版AnalyticalPointNetLK，传入device参数
            self.model = AnalyticalPointNetLK(ptnet, self.device)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def forward(self, p0, p1, maxiter=10, xtol=1e-7, mode='train'):
        """
        统一的前向传播接口
        
        Args:
            p0: 源点云 [B, N, 3]
            p1: 目标点云 [B, N, 3]
            maxiter: 最大迭代次数
            xtol: 收敛阈值
            mode: 'train' 或 'test'
            
        Returns:
            r: 残差
            g: 变换矩阵 (如果可用)
        """
        if self.model_type == 'original':
            # 原版PointNetLK
            r = ptlk.pointlk.PointLK.do_forward(
                self.model, p0, p1, maxiter, xtol, 
                p0_zero_mean=True, p1_zero_mean=True
            )
            g = self.model.g if hasattr(self.model, 'g') else None
            return r, g
        else:
            # 改进版AnalyticalPointNetLK
            # 使用do_forward静态方法
            r = AnalyticalPointNetLK.do_forward(
                self.model, p0, None, p1, None, 
                maxiter=maxiter, xtol=xtol, 
                p0_zero_mean=True, p1_zero_mean=True, 
                mode=mode, data_type='synthetic'
            )
            g = self.model.g if hasattr(self.model, 'g') else None
            return r, g
    
    def get_model(self):
        """获取底层模型"""
        return self.model
    
    def get_feature_extractor(self):
        """获取特征提取器"""
        return self.feature_bridge.get_extractor()
    
    def load_state_dict(self, state_dict):
        """加载预训练权重"""
        self.model.load_state_dict(state_dict)
    
    def state_dict(self):
        """获取模型状态字典"""
        return self.model.state_dict()
    
    def train(self, mode=True):
        """设置训练模式"""
        self.model.train(mode)
        return self
    
    def eval(self):
        """设置评估模式"""
        self.model.eval()
        return self
    
    def to(self, device):
        """移动到指定设备"""
        self.model.to(device)
        self.device = device
        return self
    
    def parameters(self):
        """获取模型参数"""
        return self.model.parameters()
    
    def compute_loss(self, p0, p1, igt, **kwargs):
        """
        计算损失函数
        
        Args:
            p0: 源点云 [B, N, 3]
            p1: 目标点云 [B, N, 3]
            igt: 真实变换矩阵 [B, 4, 4]
            **kwargs: 其他参数
            
        Returns:
            loss: 损失值
        """
        # 提取mode参数，避免重复传递
        mode = kwargs.pop('mode', 'train')
        
        if self.model_type == 'original':
            # 原版损失计算
            r, g = self.forward(p0, p1, mode=mode, **kwargs)
            if g is not None:
                loss_g = ptlk.pointlk.PointLK.comp(g, igt)
                loss_r = ptlk.pointlk.PointLK.rsq(r)
                loss = loss_r + loss_g
            else:
                loss = ptlk.pointlk.PointLK.rsq(r)
            return loss
        else:
            # 改进版损失计算
            r, g = self.forward(p0, p1, mode=mode, **kwargs)
            if g is not None:
                loss_g = AnalyticalPointNetLK.comp(g, igt)
                loss_r = AnalyticalPointNetLK.rsq(r)
                loss = loss_r + loss_g
            else:
                loss = AnalyticalPointNetLK.rsq(r)
            return loss
    
    def __repr__(self):
        return f"ModelBridge(type={self.model_type}, dim_k={self.dim_k}, device={self.device})" 