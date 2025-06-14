"""
桥接模块 - 统一PointNetLK和PointNetLK_Revisited的接口
Bridge Module - Unified interface for PointNetLK and PointNetLK_Revisited
"""

from .feature_bridge import FeatureExtractorBridge
from .model_bridge import ModelBridge
from .data_bridge import DataBridge

__all__ = ['FeatureExtractorBridge', 'ModelBridge', 'DataBridge'] 