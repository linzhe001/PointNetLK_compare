"""
对比分析模块 - 提供原版和改进版PointNetLK的对比分析功能
Comparison Module - Provides comparison analysis between original and improved PointNetLK
"""

from .model_comparison import ModelComparison
# 移除不存在的模块导入
# from .performance_benchmark import PerformanceBenchmark
# from .result_analyzer import ResultAnalyzer

__all__ = ['ModelComparison'] 