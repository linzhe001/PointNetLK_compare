"""特征提取器工厂 - 支持可替换的特征提取器创建
"""

from .pointnet import PointNet_features
from .attention import AttentionNet_features, symfn_attention_pool
from .cformer import CFormer_features, symfn_cd_pool
from .fast_attention import FastPointAttention_features, symfn_fast_attention_pool
from .mamba3d import Mamba3D_features, symfn_selective
from .base import symfn_max, symfn_avg

# 可用的特征提取器
AVAILABLE_FEATURES = {
    'pointnet': PointNet_features,
    'attention': AttentionNet_features,
    'cformer': CFormer_features,
    'fast_attention': FastPointAttention_features,
    'mamba3d': Mamba3D_features,
}

# 可用的聚合函数
AVAILABLE_SYM_FUNCTIONS = {
    'max': symfn_max,
    'avg': symfn_avg,
    'attention_pool': symfn_attention_pool,
    'cd_pool': symfn_cd_pool,
    'fast_attention_pool': symfn_fast_attention_pool,
    'selective': symfn_selective,
}

# 默认参数配置
DEFAULT_CONFIGS = {
    'pointnet': {
        'dim_k': 1024,
        'sym_fn': 'max',
        'scale': 1,
    },
    'attention': {
        'dim_k': 1024,
        'sym_fn': 'attention_pool',
        'scale': 1,
        'num_attention_blocks': 3,
        'num_heads': 8,
        'd_model': 256,
    },
    'cformer': {
        'dim_k': 1024,
        'sym_fn': 'cd_pool',
        'scale': 1,
        'base_proxies': 8,
        'max_proxies': 64,
        'num_blocks': 2,
    },
    'fast_attention': {
        'dim_k': 1024,
        'sym_fn': 'fast_attention_pool',
        'scale': 1,
        'num_attention_blocks': 2,
    },
    'mamba3d': {
        'dim_k': 1024,
        'sym_fn': 'selective',
        'scale': 1,
        'num_mamba_blocks': 3,
        'd_state': 16,
        'expand': 2,
    },
}

class FeatureExtractorFactory:
    """特征提取器工厂类"""
    
    @staticmethod
    def create_feature_extractor(name, config=None):
        """创建特征提取器
        
        Args:
            name: 特征提取器名称 ('pointnet', 'attention', 'cformer', 'fast_attention', 'mamba3d')
            config: 配置字典，如果为None则使用默认配置
            
        Returns:
            特征提取器实例
        """
        if name not in AVAILABLE_FEATURES:
            raise ValueError(f"不支持的特征提取器类型: {name}. 可用类型: {list(AVAILABLE_FEATURES.keys())}")
        
        # 获取默认配置
        default_config = DEFAULT_CONFIGS[name].copy()
        
        # 合并用户配置
        if config is not None:
            default_config.update(config)
        
        # 处理聚合函数
        sym_fn_name = default_config.pop('sym_fn', 'max')
        if isinstance(sym_fn_name, str):
            if sym_fn_name not in AVAILABLE_SYM_FUNCTIONS:
                raise ValueError(f"不支持的聚合函数: {sym_fn_name}. 可用函数: {list(AVAILABLE_SYM_FUNCTIONS.keys())}")
            sym_fn = AVAILABLE_SYM_FUNCTIONS[sym_fn_name]
        else:
            sym_fn = sym_fn_name  # 假设已经是函数
        
        default_config['sym_fn'] = sym_fn
        
        # 创建特征提取器
        feature_class = AVAILABLE_FEATURES[name]
        return feature_class(**default_config)
    
    @staticmethod
    def get_available_extractors():
        """获取可用的特征提取器列表"""
        return list(AVAILABLE_FEATURES.keys())
    
    @staticmethod
    def get_available_sym_functions():
        """获取可用的聚合函数列表"""
        return list(AVAILABLE_SYM_FUNCTIONS.keys())
    
    @staticmethod
    def get_default_config(name):
        """获取特征提取器的默认配置"""
        if name not in DEFAULT_CONFIGS:
            raise ValueError(f"不支持的特征提取器类型: {name}")
        return DEFAULT_CONFIGS[name].copy()

def create_feature_extractor(name, config=None):
    """便捷函数：创建特征提取器"""
    return FeatureExtractorFactory.create_feature_extractor(name, config) 