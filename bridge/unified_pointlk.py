"""统一PointLK桥接接口
支持Original PointLK（数值雅可比矩阵）和Improved PointLK（解析雅可比矩阵）的无缝切换
同时支持可替换的特征提取器
"""

import torch
import sys
import os

# 添加路径以导入legacy和新模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'legacy_ptlk'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from legacy_ptlk.pointlk_with_features import PointLKWithFeatures
from model_with_features import AnalyticalPointNetLKWithFeatures
from feature_extractors import FeatureExtractorFactory


class UnifiedPointLK:
    """统一PointLK接口类
    
    支持两种PointLK实现：
    1. Original PointLK: 使用数值雅可比矩阵，支持所有特征提取器
    2. Improved PointLK: 使用解析雅可比矩阵，仅PointNet支持解析计算，其他特征提取器回退到数值计算
    """
    
    ORIGINAL = 'original'
    IMPROVED = 'improved'
    
    def __init__(self, pointlk_type='original', feature_extractor_name='pointnet', 
                 feature_config=None, device='cuda', **kwargs):
        """
        Args:
            pointlk_type: PointLK类型 ('original' 或 'improved')
            feature_extractor_name: 特征提取器名称
            feature_config: 特征提取器配置字典
            device: 计算设备
            **kwargs: 其他参数（如delta, learn_delta等）
        """
        self.pointlk_type = pointlk_type
        self.feature_extractor_name = feature_extractor_name
        self.feature_config = feature_config or {}
        self.device = device
        
        # 创建相应的PointLK实例
        if pointlk_type == self.ORIGINAL:
            self.model = PointLKWithFeatures(
                feature_extractor_name=feature_extractor_name,
                feature_config=feature_config,
                **kwargs
            )
        elif pointlk_type == self.IMPROVED:
            improved_kwargs = {k: v for k, v in kwargs.items() if k in ['device']}
            improved_kwargs['device'] = device
            self.model = AnalyticalPointNetLKWithFeatures(
                feature_extractor_name=feature_extractor_name,
                feature_config=feature_config,
                **improved_kwargs
            )
        else:
            raise ValueError(f"不支持的PointLK类型: {pointlk_type}. 支持的类型: {self.ORIGINAL}, {self.IMPROVED}")
        
        # 将模型移到指定设备
        if hasattr(self.model, 'to'):
            self.model.to(device)
    
    def __call__(self, *args, **kwargs):
        """使模型可调用"""
        return self.forward(*args, **kwargs)
    
    def forward(self, p0, p1, *args, **kwargs):
        """前向传播 - 统一接口"""
        if self.pointlk_type == self.ORIGINAL:
            # Original PointLK接口
            return self.model(p0, p1, *args, **kwargs)
        else:
            # Improved PointLK接口需要mode参数
            if 'mode' not in kwargs:
                kwargs['mode'] = 'train' if self.model.training else 'test'
            return self.model(p0, p1, *args, **kwargs)
    
    def train(self, mode=True):
        """设置训练模式"""
        if hasattr(self.model, 'train'):
            self.model.train(mode)
        return self
    
    def eval(self):
        """设置评估模式"""
        if hasattr(self.model, 'eval'):
            self.model.eval()
        return self
    
    def to(self, device):
        """移动到指定设备"""
        if hasattr(self.model, 'to'):
            self.model.to(device)
        self.device = device
        return self
    
    def parameters(self):
        """获取模型参数"""
        if hasattr(self.model, 'parameters'):
            return self.model.parameters()
        return []
    
    def state_dict(self):
        """获取状态字典"""
        if hasattr(self.model, 'state_dict'):
            return self.model.state_dict()
        return {}
    
    def load_state_dict(self, state_dict, strict=True):
        """加载状态字典"""
        if hasattr(self.model, 'load_state_dict'):
            return self.model.load_state_dict(state_dict, strict)
    
    def get_transformation_result(self):
        """获取变换结果"""
        return getattr(self.model, 'g', None)
    
    def get_iterations(self):
        """获取迭代次数"""
        return getattr(self.model, 'itr', 0)
    
    def get_last_error(self):
        """获取最后的错误信息"""
        return getattr(self.model, 'last_err', None)
    
    def get_model_info(self):
        """获取模型信息"""
        info = {
            'pointlk_type': self.pointlk_type,
            'feature_extractor_name': self.feature_extractor_name,
            'feature_config': self.feature_config,
            'device': self.device,
        }
        
        # 添加特征提取器特定信息
        if hasattr(self.model, 'get_feature_extractor_info'):
            info.update(self.model.get_feature_extractor_info())
        
        return info
    
    @staticmethod
    def get_available_configurations():
        """获取可用配置"""
        return {
            'pointlk_types': [UnifiedPointLK.ORIGINAL, UnifiedPointLK.IMPROVED],
            'feature_extractors': FeatureExtractorFactory.get_available_extractors(),
            'aggregation_functions': FeatureExtractorFactory.get_available_sym_functions()
        }
    
    @staticmethod
    def create_original_with_features(feature_extractor_name='pointnet', feature_config=None, **kwargs):
        """便捷方法：创建带可替换特征提取器的Original PointLK"""
        return UnifiedPointLK(
            pointlk_type=UnifiedPointLK.ORIGINAL,
            feature_extractor_name=feature_extractor_name,
            feature_config=feature_config,
            **kwargs
        )
    
    @staticmethod
    def create_improved_with_features(feature_extractor_name='pointnet', feature_config=None, **kwargs):
        """便捷方法：创建带可替换特征提取器的Improved PointLK"""
        return UnifiedPointLK(
            pointlk_type=UnifiedPointLK.IMPROVED,
            feature_extractor_name=feature_extractor_name,
            feature_config=feature_config,
            **kwargs
        )
    
    @staticmethod
    def rsq(r):
        """残差平方和 - 统一静态方法"""
        z = torch.zeros_like(r)
        return torch.nn.functional.mse_loss(r, z, size_average=False)
    
    @staticmethod
    def comp(g, igt):
        """变换比较 - 统一静态方法"""
        assert g.size(0) == igt.size(0)
        assert g.size(1) == igt.size(1) and g.size(1) == 4
        assert g.size(2) == igt.size(2) and g.size(2) == 4
        A = g.matmul(igt)
        I = torch.eye(4).to(A).view(1, 4, 4).expand(A.size(0), 4, 4)
        return torch.nn.functional.mse_loss(A, I, size_average=True) * 16


class PointLKFactory:
    """PointLK工厂类 - 用于创建不同配置的PointLK实例"""
    
    @staticmethod
    def create_pointlk(config):
        """根据配置创建PointLK实例
        
        Args:
            config: 配置字典，包含以下键：
                - pointlk_type: 'original' 或 'improved'
                - feature_extractor_name: 特征提取器名称
                - feature_config: 特征提取器配置
                - device: 计算设备
                - 其他PointLK参数
        
        Returns:
            UnifiedPointLK实例
        """
        pointlk_type = config.get('pointlk_type', 'original')
        feature_extractor_name = config.get('feature_extractor_name', 'pointnet')
        feature_config = config.get('feature_config', None)
        device = config.get('device', 'cuda')
        
        # 提取其他参数
        other_kwargs = {k: v for k, v in config.items() 
                       if k not in ['pointlk_type', 'feature_extractor_name', 'feature_config', 'device']}
        
        return UnifiedPointLK(
            pointlk_type=pointlk_type,
            feature_extractor_name=feature_extractor_name,
            feature_config=feature_config,
            device=device,
            **other_kwargs
        )
    
    @staticmethod
    def get_default_configs():
        """获取预定义的默认配置"""
        return {
            'original_pointnet': {
                'pointlk_type': 'original',
                'feature_extractor_name': 'pointnet',
                'feature_config': {'dim_k': 1024, 'sym_fn': 'max'},
                'device': 'cuda',
                'delta': 1.0e-2,
                'learn_delta': False
            },
            'original_attention': {
                'pointlk_type': 'original',
                'feature_extractor_name': 'attention',
                'feature_config': {'dim_k': 1024, 'sym_fn': 'attention_pool'},
                'device': 'cuda',
                'delta': 1.0e-2,
                'learn_delta': False
            },
            'original_cformer': {
                'pointlk_type': 'original',
                'feature_extractor_name': 'cformer',
                'feature_config': {'dim_k': 1024, 'sym_fn': 'cd_pool'},
                'device': 'cuda',
                'delta': 1.0e-2,
                'learn_delta': False
            },
            'original_fast_attention': {
                'pointlk_type': 'original',
                'feature_extractor_name': 'fast_attention',
                'feature_config': {'dim_k': 1024, 'sym_fn': 'fast_attention_pool'},
                'device': 'cuda',
                'delta': 1.0e-2,
                'learn_delta': False
            },
            'original_mamba3d': {
                'pointlk_type': 'original',
                'feature_extractor_name': 'mamba3d',
                'feature_config': {'dim_k': 1024, 'sym_fn': 'selective'},
                'device': 'cuda',
                'delta': 1.0e-2,
                'learn_delta': False
            },
            'improved_pointnet': {
                'pointlk_type': 'improved',
                'feature_extractor_name': 'pointnet',
                'feature_config': {'dim_k': 1024},
                'device': 'cuda'
            },
            'improved_attention': {
                'pointlk_type': 'improved',
                'feature_extractor_name': 'attention',
                'feature_config': {'dim_k': 1024, 'sym_fn': 'attention_pool'},
                'device': 'cuda'
            },
        }


# 为了向后兼容，提供便捷函数
def create_original_pointlk(feature_extractor_name='pointnet', **kwargs):
    """创建Original PointLK（数值雅可比矩阵）"""
    return UnifiedPointLK.create_original_with_features(feature_extractor_name, **kwargs)

def create_improved_pointlk(feature_extractor_name='pointnet', **kwargs):
    """创建Improved PointLK（解析雅可比矩阵）"""
    return UnifiedPointLK.create_improved_with_features(feature_extractor_name, **kwargs) 