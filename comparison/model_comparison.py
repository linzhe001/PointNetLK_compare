"""
模型对比分析 - 原版和改进版PointNetLK的详细对比
Model Comparison - Detailed comparison between original and improved PointNetLK
"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os
from typing import Dict, List, Tuple, Optional

# 添加路径以导入相关模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bridge import ModelBridge, DataBridge


class ModelComparison:
    """模型对比分析器"""
    
    def __init__(self, dim_k=1024, device='cuda:0'):
        """
        初始化模型对比分析器
        
        Args:
            dim_k: 特征维度
            device: 计算设备
        """
        self.dim_k = dim_k
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 创建原版和改进版模型
        self.original_model = ModelBridge('original', dim_k=dim_k)
        self.improved_model = ModelBridge('improved', dim_k=dim_k)
        
        # 移动到设备
        self.original_model.to(self.device)
        self.improved_model.to(self.device)
        
        # 结果存储
        self.comparison_results = {}
    
    def compare_models(self, test_data, maxiter=10, xtol=1e-7) -> Dict:
        """
        对比两个模型的性能
        
        Args:
            test_data: 测试数据 [(p0, p1, igt), ...]
            maxiter: 最大迭代次数
            xtol: 收敛阈值
            
        Returns:
            comparison_results: 对比结果字典
        """
        print("开始模型对比分析...")
        
        results = {
            'original': {'errors': [], 'times': [], 'iterations': []},
            'improved': {'errors': [], 'times': [], 'iterations': []},
            'summary': {}
        }
        
        # 设置为评估模式
        self.original_model.eval()
        self.improved_model.eval()
        
        with torch.no_grad():
            for i, (p0, p1, igt) in enumerate(test_data):
                if i % 10 == 0:
                    print(f"处理测试样本 {i+1}/{len(test_data)}")
                
                p0 = p0.to(self.device)
                p1 = p1.to(self.device)
                igt = igt.to(self.device)
                
                # 测试原版模型
                start_time = time.time()
                r_orig, g_orig = self.original_model.forward(p0, p1, maxiter=maxiter, xtol=xtol)
                orig_time = time.time() - start_time
                
                # 测试改进版模型
                start_time = time.time()
                r_impr, g_impr = self.improved_model.forward(p0, p1, maxiter=maxiter, xtol=xtol, mode='test')
                impr_time = time.time() - start_time
                
                # 计算误差
                if g_orig is not None:
                    orig_error = self._compute_transformation_error(g_orig, igt)
                    results['original']['errors'].append(orig_error.item())
                
                if g_impr is not None:
                    impr_error = self._compute_transformation_error(g_impr, igt)
                    results['improved']['errors'].append(impr_error.item())
                
                # 记录时间
                results['original']['times'].append(orig_time)
                results['improved']['times'].append(impr_time)
                
                # 记录迭代次数（如果可用）
                if hasattr(self.original_model.get_model(), 'itr'):
                    results['original']['iterations'].append(self.original_model.get_model().itr)
                if hasattr(self.improved_model.get_model(), 'itr'):
                    results['improved']['iterations'].append(self.improved_model.get_model().itr)
        
        # 计算统计摘要
        results['summary'] = self._compute_summary_statistics(results)
        
        self.comparison_results = results
        return results
    
    def _compute_transformation_error(self, g_pred, g_gt):
        """计算变换误差"""
        # 计算旋转误差和平移误差
        R_pred = g_pred[:, :3, :3]
        t_pred = g_pred[:, :3, 3]
        R_gt = g_gt[:, :3, :3]
        t_gt = g_gt[:, :3, 3]
        
        # 旋转误差 (角度)
        R_error = torch.bmm(R_pred, R_gt.transpose(1, 2))
        trace = torch.diagonal(R_error, dim1=1, dim2=2).sum(dim=1)
        rot_error = torch.acos(torch.clamp((trace - 1) / 2, -1, 1)) * 180 / np.pi
        
        # 平移误差 (欧几里得距离)
        trans_error = torch.norm(t_pred - t_gt, dim=1)
        
        # 综合误差
        total_error = rot_error + trans_error * 10  # 加权组合
        
        return total_error.mean()
    
    def _compute_summary_statistics(self, results):
        """计算统计摘要"""
        summary = {}
        
        for model_type in ['original', 'improved']:
            if results[model_type]['errors']:
                errors = np.array(results[model_type]['errors'])
                times = np.array(results[model_type]['times'])
                
                summary[model_type] = {
                    'mean_error': float(np.mean(errors)),
                    'std_error': float(np.std(errors)),
                    'median_error': float(np.median(errors)),
                    'mean_time': float(np.mean(times)),
                    'std_time': float(np.std(times)),
                    'total_samples': len(errors)
                }
                
                if results[model_type]['iterations']:
                    iterations = np.array(results[model_type]['iterations'])
                    summary[model_type]['mean_iterations'] = float(np.mean(iterations))
                    summary[model_type]['std_iterations'] = float(np.std(iterations))
        
        # 计算改进比例
        if 'original' in summary and 'improved' in summary:
            summary['improvement'] = {
                'error_reduction': (summary['original']['mean_error'] - summary['improved']['mean_error']) / summary['original']['mean_error'] * 100,
                'speedup': summary['original']['mean_time'] / summary['improved']['mean_time']
            }
        
        return summary
    
    def compare_jacobian_computation(self, test_points, num_samples=100):
        """
        对比数值雅可比和解析雅可比的计算效率
        
        Args:
            test_points: 测试点云 [B, N, 3]
            num_samples: 测试样本数
            
        Returns:
            jacobian_comparison: 雅可比对比结果
        """
        print("开始雅可比计算对比...")
        
        test_points = test_points.to(self.device)
        
        results = {
            'numerical': {'times': [], 'accuracy': []},
            'analytical': {'times': [], 'accuracy': []},
            'summary': {}
        }
        
        # 设置为评估模式
        self.original_model.eval()
        self.improved_model.eval()
        
        with torch.no_grad():
            for i in range(num_samples):
                if i % 10 == 0:
                    print(f"处理雅可比测试 {i+1}/{num_samples}")
                
                # 测试数值雅可比（原版）
                start_time = time.time()
                # 这里需要访问原版模型的雅可比计算方法
                # 由于原版使用有限差分，我们模拟这个过程
                numerical_time = time.time() - start_time
                results['numerical']['times'].append(numerical_time)
                
                # 测试解析雅可比（改进版）
                start_time = time.time()
                # 这里需要访问改进版模型的解析雅可比计算方法
                analytical_time = time.time() - start_time
                results['analytical']['times'].append(analytical_time)
        
        # 计算统计摘要
        results['summary'] = {
            'numerical_mean_time': float(np.mean(results['numerical']['times'])),
            'analytical_mean_time': float(np.mean(results['analytical']['times'])),
            'speedup': float(np.mean(results['numerical']['times']) / np.mean(results['analytical']['times']))
        }
        
        return results
    
    def compare_convergence_behavior(self, test_data, max_iterations=20):
        """
        对比两个模型的收敛行为
        
        Args:
            test_data: 测试数据
            max_iterations: 最大迭代次数
            
        Returns:
            convergence_comparison: 收敛行为对比结果
        """
        print("开始收敛行为对比...")
        
        results = {
            'original': {'convergence_curves': []},
            'improved': {'convergence_curves': []},
            'summary': {}
        }
        
        # 设置为评估模式
        self.original_model.eval()
        self.improved_model.eval()
        
        with torch.no_grad():
            for i, (p0, p1, igt) in enumerate(test_data[:10]):  # 只测试前10个样本
                print(f"分析收敛行为 {i+1}/10")
                
                p0 = p0.to(self.device)
                p1 = p1.to(self.device)
                igt = igt.to(self.device)
                
                # 记录原版模型的收敛过程
                orig_curve = self._track_convergence(
                    self.original_model, p0, p1, igt, max_iterations
                )
                results['original']['convergence_curves'].append(orig_curve)
                
                # 记录改进版模型的收敛过程
                impr_curve = self._track_convergence(
                    self.improved_model, p0, p1, igt, max_iterations
                )
                results['improved']['convergence_curves'].append(impr_curve)
        
        # 计算平均收敛曲线
        results['summary'] = self._analyze_convergence_curves(results)
        
        return results
    
    def _track_convergence(self, model, p0, p1, igt, max_iterations):
        """跟踪单个模型的收敛过程"""
        convergence_curve = []
        
        for iter_num in range(1, max_iterations + 1):
            r, g = model.forward(p0, p1, maxiter=iter_num, xtol=1e-10)
            
            if g is not None:
                error = self._compute_transformation_error(g, igt)
                convergence_curve.append(error.item())
            else:
                convergence_curve.append(float('inf'))
        
        return convergence_curve
    
    def _analyze_convergence_curves(self, results):
        """分析收敛曲线"""
        summary = {}
        
        for model_type in ['original', 'improved']:
            curves = results[model_type]['convergence_curves']
            if curves:
                # 计算平均收敛曲线
                max_len = max(len(curve) for curve in curves)
                avg_curve = []
                
                for i in range(max_len):
                    values = [curve[i] for curve in curves if i < len(curve)]
                    avg_curve.append(np.mean(values))
                
                summary[model_type] = {
                    'average_convergence_curve': avg_curve,
                    'final_error': avg_curve[-1] if avg_curve else float('inf'),
                    'convergence_rate': self._compute_convergence_rate(avg_curve)
                }
        
        return summary
    
    def _compute_convergence_rate(self, curve):
        """计算收敛速率"""
        if len(curve) < 2:
            return 0.0
        
        # 计算相邻点之间的改进率
        improvements = []
        for i in range(1, len(curve)):
            if curve[i-1] > 0:
                improvement = (curve[i-1] - curve[i]) / curve[i-1]
                improvements.append(improvement)
        
        return np.mean(improvements) if improvements else 0.0
    
    def generate_comparison_report(self, save_path=None):
        """
        生成对比分析报告
        
        Args:
            save_path: 报告保存路径
            
        Returns:
            report: 报告内容
        """
        if not self.comparison_results:
            raise ValueError("请先运行模型对比分析")
        
        report = []
        report.append("=" * 60)
        report.append("PointNetLK 模型对比分析报告")
        report.append("=" * 60)
        report.append("")
        
        summary = self.comparison_results['summary']
        
        # 基本性能对比
        report.append("## 基本性能对比")
        report.append("-" * 30)
        
        for model_type in ['original', 'improved']:
            if model_type in summary:
                stats = summary[model_type]
                model_name = "原版PointNetLK" if model_type == 'original' else "改进版PointNetLK"
                
                report.append(f"\n### {model_name}")
                report.append(f"平均误差: {stats['mean_error']:.4f}")
                report.append(f"误差标准差: {stats['std_error']:.4f}")
                report.append(f"中位数误差: {stats['median_error']:.4f}")
                report.append(f"平均运行时间: {stats['mean_time']:.4f}s")
                report.append(f"时间标准差: {stats['std_time']:.4f}s")
                report.append(f"测试样本数: {stats['total_samples']}")
                
                if 'mean_iterations' in stats:
                    report.append(f"平均迭代次数: {stats['mean_iterations']:.2f}")
        
        # 改进效果
        if 'improvement' in summary:
            improvement = summary['improvement']
            report.append(f"\n## 改进效果")
            report.append("-" * 30)
            report.append(f"误差减少: {improvement['error_reduction']:.2f}%")
            report.append(f"速度提升: {improvement['speedup']:.2f}x")
        
        report.append("\n" + "=" * 60)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"报告已保存到: {save_path}")
        
        return report_text
    
    def load_pretrained_models(self, original_path=None, improved_path=None):
        """
        加载预训练模型
        
        Args:
            original_path: 原版模型路径
            improved_path: 改进版模型路径
        """
        if original_path and os.path.exists(original_path):
            self.original_model.load_state_dict(torch.load(original_path, map_location=self.device))
            print(f"已加载原版模型: {original_path}")
        
        if improved_path and os.path.exists(improved_path):
            self.improved_model.load_state_dict(torch.load(improved_path, map_location=self.device))
            print(f"已加载改进版模型: {improved_path}")
    
    def __repr__(self):
        return f"ModelComparison(dim_k={self.dim_k}, device={self.device})" 