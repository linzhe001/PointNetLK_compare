"""
综合测试脚本 - 结合两种测试方法
Comprehensive Testing Script - Combining both testing methodologies

这个脚本结合了：
1. 原版PointNetLK的系统性扰动测试（鲁棒性评估）
2. PointNetLK_Revisited的单一场景测试（精度评估）
"""

import argparse
import os
import sys
import logging
import numpy as np
import torch
import torch.utils.data
import torchvision
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

# 导入桥接模块和对比分析模块
from bridge import ModelBridge, DataBridge
from comparison import ModelComparison

# 设置日志
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='综合PointNetLK测试脚本')
    
    # 基本参数
    parser.add_argument('--model-type', default='both', choices=['original', 'improved', 'both'],
                        help='模型类型: original, improved, 或 both')
    parser.add_argument('--model-path', required=True, type=str,
                        help='模型文件路径（单模型测试时）')
    parser.add_argument('--original-model-path', default='', type=str,
                        help='原版模型路径（对比测试时）')
    parser.add_argument('--improved-model-path', default='', type=str,
                        help='改进版模型路径（对比测试时）')
    
    # 数据集参数
    parser.add_argument('--dataset-path', required=True, type=str,
                        help='数据集路径')
    parser.add_argument('--categoryfile', required=True, type=str,
                        help='类别文件路径')
    parser.add_argument('--num-points', default=1024, type=int,
                        help='点云中的点数')
    
    # 扰动测试参数
    parser.add_argument('--perturbation-angles', default='5,10,15,30,45,60,75,90', type=str,
                        help='扰动角度列表（度），用逗号分隔')
    parser.add_argument('--num-samples-per-angle', default=100, type=int,
                        help='每个角度的测试样本数')
    parser.add_argument('--perturbation-type', default='both', choices=['rotation', 'both'],
                        help='扰动类型: rotation（仅旋转）或 both（旋转+平移）')
    
    # 模型参数
    parser.add_argument('--dim-k', default=1024, type=int,
                        help='特征向量维度')
    parser.add_argument('--max-iter', default=10, type=int,
                        help='LK算法最大迭代次数')
    parser.add_argument('--xtol', default=1e-7, type=float,
                        help='收敛阈值')
    
    # 测试设置
    parser.add_argument('--batch-size', default=32, type=int,
                        help='批次大小（精度测试）')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='计算设备')
    parser.add_argument('--workers', default=4, type=int,
                        help='数据加载工作进程数')
    
    # 输出设置
    parser.add_argument('-o', '--output-dir', required=True, type=str,
                        help='输出目录')
    parser.add_argument('--save-plots', action='store_true',
                        help='保存性能曲线图')
    parser.add_argument('--save-detailed-results', action='store_true',
                        help='保存详细测试结果')
    
    return parser.parse_args()


class PerturbationGenerator:
    """扰动生成器 - 模拟原版PointNetLK的扰动生成方法"""
    
    @staticmethod
    def generate_perturbations_by_angle(angle_degrees, num_samples, perturbation_type='both'):
        """
        按角度生成扰动
        
        Args:
            angle_degrees: 扰动角度（度）
            num_samples: 样本数量
            perturbation_type: 'rotation' 或 'both'
        
        Returns:
            perturbations: [num_samples, 6] 的扰动向量
        """
        angle_rad = np.deg2rad(angle_degrees)
        
        if perturbation_type == 'rotation':
            # 仅旋转扰动
            w = torch.randn(num_samples, 3)
            w = w / w.norm(p=2, dim=1, keepdim=True) * angle_rad
            v = torch.zeros(num_samples, 3)
            x = torch.cat((w, v), dim=1)
        else:
            # 旋转+平移扰动
            x = torch.randn(num_samples, 6)
            x = x / x.norm(p=2, dim=1, keepdim=True) * angle_rad
        
        return x.numpy()
    
    @staticmethod
    def apply_perturbation(points, perturbation, device=None):
        """
        对点云应用扰动
        
        Args:
            points: [N, 3] 点云
            perturbation: [6] 扰动向量
            device: 目标设备
        
        Returns:
            transformed_points: 变换后的点云
            transformation_matrix: 变换矩阵
        """
        import legacy_ptlk as ptlk
        
        # 确保设备一致性
        if device is None:
            device = points.device
        
        twist = torch.from_numpy(perturbation).float().view(1, 6).to(device)
        g = ptlk.se3.exp(twist)  # [1, 4, 4]
        
        # 确保points也在正确的设备上
        points = points.to(device)
        
        # 处理点云维度：确保是 [B, N, 3] 格式
        if points.dim() == 2:
            points = points.unsqueeze(0)  # [1, N, 3]
        
        # 变换点云：需要转置为 [B, 3, N] 格式进行矩阵乘法
        points_transposed = points.transpose(1, 2)  # [1, 3, N]
        transformed_points = ptlk.se3.transform(g, points_transposed)  # [1, 3, N]
        transformed_points = transformed_points.transpose(1, 2)  # [1, N, 3]
        
        return transformed_points.squeeze(0), g.squeeze(0)


class ComprehensiveTester:
    """综合测试器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 设置日志
        log_file = os.path.join(args.output_dir, f'comprehensive_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)
        
        LOGGER.info("开始综合测试")
        LOGGER.info(f"参数: {vars(args)}")
        
        # 解析扰动角度
        self.perturbation_angles = [float(x.strip()) for x in args.perturbation_angles.split(',')]
        
        # 初始化模型
        self.models = self._create_models()
        
        # 创建基础数据集（用于扰动测试）
        self.base_dataset = self._create_base_dataset()
        
        # 创建精度测试数据加载器
        self.precision_loader = self._create_precision_loader()
    
    def _create_models(self):
        """创建测试模型"""
        models = {}
        
        if self.args.model_type in ['original', 'both']:
            LOGGER.info("创建原版模型...")
            original_path = self.args.original_model_path or self.args.model_path
            models['original'] = self._load_model('original', original_path)
        
        if self.args.model_type in ['improved', 'both']:
            LOGGER.info("创建改进版模型...")
            improved_path = self.args.improved_model_path or self.args.model_path
            models['improved'] = self._load_model('improved', improved_path)
        
        return models
    
    def _load_model(self, model_type, model_path):
        """加载单个模型"""
        model_kwargs = {'dim_k': self.args.dim_k}
        model = ModelBridge(model_type, **model_kwargs)
        model.to(self.device)
        
        # 加载权重
        LOGGER.info(f"加载{model_type}模型权重: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def _create_base_dataset(self):
        """创建基础数据集（用于扰动测试）"""
        LOGGER.info("创建基础数据集...")
        
        # 读取类别信息
        categories = [line.rstrip('\n') for line in open(self.args.categoryfile)]
        categories.sort()
        c_to_idx = {categories[i]: i for i in range(len(categories))}
        cinfo = (categories, c_to_idx)
        
        # 创建数据变换
        import legacy_ptlk as ptlk
        transform = torchvision.transforms.Compose([
            ptlk.data.transforms.Mesh2Points(),
            ptlk.data.transforms.OnUnitCube(),
        ])
        
        # 创建ModelNet数据集
        dataset = ptlk.data.datasets.ModelNet(
            self.args.dataset_path, 
            train=0,  # 测试集
            transform=transform, 
            classinfo=cinfo
        )
        
        return dataset
    
    def _create_precision_loader(self):
        """创建精度测试数据加载器"""
        LOGGER.info("创建精度测试数据加载器...")
        
        data_bridge = DataBridge(
            dataset_type='modelnet',
            data_source='improved'  # 使用改进版数据加载
        )
        
        dataset_kwargs = {
            'dataset_path': self.args.dataset_path,
            'num_points': self.args.num_points,
            'categoryfile': self.args.categoryfile
        }
        
        _, testset = data_bridge.get_datasets(**dataset_kwargs)
        
        loader = data_bridge.get_dataloader(
            testset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers
        )
        
        return loader
    
    def run_robustness_test(self):
        """运行鲁棒性测试（系统性扰动）"""
        LOGGER.info("=" * 60)
        LOGGER.info("开始鲁棒性测试（系统性扰动）")
        LOGGER.info("=" * 60)
        
        results = {}
        
        for model_name, model in self.models.items():
            LOGGER.info(f"测试 {model_name} 模型的鲁棒性...")
            
            model_results = {
                'angles': [],
                'mean_errors': [],
                'std_errors': [],
                'success_rates': [],
                'mean_times': []
            }
            
            for angle in self.perturbation_angles:
                LOGGER.info(f"  测试角度: {angle}°")
                
                # 生成扰动
                perturbations = PerturbationGenerator.generate_perturbations_by_angle(
                    angle, self.args.num_samples_per_angle, self.args.perturbation_type
                )
                
                # 测试该角度下的性能
                angle_results = self._test_angle_performance(model, perturbations, angle, model_name)
                
                model_results['angles'].append(angle)
                model_results['mean_errors'].append(angle_results['mean_error'])
                model_results['std_errors'].append(angle_results['std_error'])
                model_results['success_rates'].append(angle_results['success_rate'])
                model_results['mean_times'].append(angle_results['mean_time'])
                
                LOGGER.info(f"    平均误差: {angle_results['mean_error']:.4f}°")
                LOGGER.info(f"    成功率: {angle_results['success_rate']:.2%}")
            
            results[model_name] = model_results
        
        return results
    
    def _test_angle_performance(self, model, perturbations, angle, model_name=None):
        """测试特定角度下的性能"""
        errors = []
        times = []
        successes = 0
        
        # 随机选择测试样本
        num_test_samples = min(self.args.num_samples_per_angle, len(self.base_dataset))
        sample_indices = np.random.choice(len(self.base_dataset), num_test_samples, replace=False)
        
        # 根据模型类型决定是否需要梯度
        need_grad = (model_name == 'improved')
        
        if need_grad:
            # 改进版模型需要梯度
            for i, pert in enumerate(perturbations[:num_test_samples]):
                # 获取原始点云
                p0, _ = self.base_dataset[sample_indices[i]]
                p0 = p0.to(self.device).unsqueeze(0)  # [1, N, 3]
                
                # 应用扰动生成目标点云
                p1, igt = PerturbationGenerator.apply_perturbation(p0.squeeze(0), pert, self.device)
                p1 = p1.to(self.device).unsqueeze(0)  # [1, N, 3]
                igt = igt.to(self.device).unsqueeze(0)  # [1, 4, 4]
                
                # 启用梯度
                p0.requires_grad_(True)
                p1.requires_grad_(True)
                
                # 前向传播
                start_time = time.time()
                with torch.enable_grad():
                    r, g = model.forward(p0, p1, maxiter=self.args.max_iter, 
                                       xtol=self.args.xtol, mode='test')
                inference_time = time.time() - start_time
                
                times.append(inference_time)
                
                # 计算误差
                if g is not None:
                    # 计算旋转误差
                    dg = g.bmm(igt)  # 如果正确，dg应该是单位矩阵
                    
                    # 提取旋转部分并计算角度误差
                    R_error = dg[:, :3, :3]
                    trace = torch.diagonal(R_error, dim1=1, dim2=2).sum(dim=1)
                    rot_error = torch.acos(torch.clamp((trace - 1) / 2, -1, 1)) * 180 / np.pi
                    
                    error_deg = float(rot_error[0])
                    errors.append(error_deg)
                    
                    # 判断是否成功（误差小于5度认为成功）
                    if error_deg < 5.0:
                        successes += 1
                else:
                    errors.append(float('inf'))
        else:
            # 原版模型不需要梯度
            with torch.no_grad():
                for i, pert in enumerate(perturbations[:num_test_samples]):
                    # 获取原始点云
                    p0, _ = self.base_dataset[sample_indices[i]]
                    p0 = p0.to(self.device).unsqueeze(0)  # [1, N, 3]
                    
                    # 应用扰动生成目标点云
                    p1, igt = PerturbationGenerator.apply_perturbation(p0.squeeze(0), pert, self.device)
                    p1 = p1.to(self.device).unsqueeze(0)  # [1, N, 3]
                    igt = igt.to(self.device).unsqueeze(0)  # [1, 4, 4]
                    
                    # 前向传播
                    start_time = time.time()
                    r, g = model.forward(p0, p1, maxiter=self.args.max_iter, 
                                       xtol=self.args.xtol, mode='test')
                    inference_time = time.time() - start_time
                    
                    times.append(inference_time)
                    
                    # 计算误差
                    if g is not None:
                        # 计算旋转误差
                        dg = g.bmm(igt)  # 如果正确，dg应该是单位矩阵
                        
                        # 提取旋转部分并计算角度误差
                        R_error = dg[:, :3, :3]
                        trace = torch.diagonal(R_error, dim1=1, dim2=2).sum(dim=1)
                        rot_error = torch.acos(torch.clamp((trace - 1) / 2, -1, 1)) * 180 / np.pi
                        
                        error_deg = float(rot_error[0])
                        errors.append(error_deg)
                        
                        # 判断是否成功（误差小于5度认为成功）
                        if error_deg < 5.0:
                            successes += 1
                    else:
                        errors.append(float('inf'))
        
        return {
            'mean_error': np.mean(errors) if errors else float('inf'),
            'std_error': np.std(errors) if errors else 0,
            'success_rate': successes / len(errors) if errors else 0,
            'mean_time': np.mean(times) if times else 0
        }
    
    def run_precision_test(self):
        """运行精度测试（单一场景）"""
        LOGGER.info("=" * 60)
        LOGGER.info("开始精度测试（单一场景）")
        LOGGER.info("=" * 60)
        
        results = {}
        
        for model_name, model in self.models.items():
            LOGGER.info(f"测试 {model_name} 模型的精度...")
            
            model_results = {
                'errors': [],
                'trans_errors': [],
                'times': [],
                'iterations': []
            }
            
            total_time = 0
            total_samples = 0
            
            with torch.no_grad():
                for batch_idx, data in enumerate(self.precision_loader):
                    if batch_idx % 10 == 0:
                        LOGGER.info(f"  处理批次 {batch_idx+1}/{len(self.precision_loader)}")
                    
                    # 解析数据
                    p0, p1, igt = self._parse_batch_data(data)
                    p0 = p0.to(self.device)
                    p1 = p1.to(self.device)
                    igt = igt.to(self.device)
                    
                    # 前向传播
                    start_time = time.time()
                    
                    if model_name == 'improved':
                        p0.requires_grad_(True)
                        p1.requires_grad_(True)
                        with torch.enable_grad():
                            r, g = model.forward(p0, p1, maxiter=self.args.max_iter, 
                                               xtol=self.args.xtol, mode='test')
                    else:
                        r, g = model.forward(p0, p1, maxiter=self.args.max_iter, 
                                           xtol=self.args.xtol, mode='test')
                    
                    inference_time = time.time() - start_time
                    
                    # 计算误差
                    if g is not None:
                        rot_error, trans_error = self._compute_transformation_error(g, igt)
                        model_results['errors'].extend(rot_error.cpu().numpy())
                        model_results['trans_errors'].extend(trans_error.cpu().numpy())
                    
                    model_results['times'].append(inference_time)
                    total_time += inference_time
                    total_samples += p0.size(0)
                    
                    # 记录迭代次数
                    if hasattr(model.get_model(), 'itr'):
                        model_results['iterations'].append(model.get_model().itr)
            
            # 计算统计信息
            summary = self._compute_precision_summary(model_results, total_time, total_samples)
            model_results['summary'] = summary
            
            LOGGER.info(f"  {model_name} 精度测试完成:")
            LOGGER.info(f"    平均误差: {summary['mean_error']:.6f}°")
            LOGGER.info(f"    平均时间: {summary['mean_time']:.6f}s")
            
            results[model_name] = model_results
        
        return results
    
    def _parse_batch_data(self, data):
        """解析批次数据"""
        if len(data) == 3:
            return data
        elif isinstance(data, (list, tuple)) and len(data) >= 2:
            p0, p1 = data[0], data[1]
            batch_size = p0.size(0)
            igt = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
            return p0, p1, igt
        else:
            raise ValueError(f"不支持的数据格式: {type(data)}")
    
    def _compute_transformation_error(self, g_pred, g_gt):
        """计算变换误差"""
        R_pred = g_pred[:, :3, :3]
        t_pred = g_pred[:, :3, 3]
        R_gt = g_gt[:, :3, :3]
        t_gt = g_gt[:, :3, 3]
        
        # 旋转误差（角度）
        R_error = torch.bmm(R_pred, R_gt.transpose(1, 2))
        trace = torch.diagonal(R_error, dim1=1, dim2=2).sum(dim=1)
        rot_error = torch.acos(torch.clamp((trace - 1) / 2, -1, 1)) * 180 / np.pi
        
        # 平移误差（欧几里得距离）
        trans_error = torch.norm(t_pred - t_gt, dim=1)
        
        return rot_error, trans_error
    
    def _compute_precision_summary(self, results, total_time, total_samples):
        """计算精度测试摘要"""
        summary = {
            'total_samples': total_samples,
            'total_time': total_time,
            'mean_time': total_time / len(results['times']) if results['times'] else 0,
        }
        
        if results['errors']:
            errors = np.array(results['errors'])
            summary.update({
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
                'median_error': float(np.median(errors)),
                'min_error': float(np.min(errors)),
                'max_error': float(np.max(errors))
            })
        
        if results['iterations']:
            iterations = np.array(results['iterations'])
            summary.update({
                'mean_iterations': float(np.mean(iterations)),
                'std_iterations': float(np.std(iterations))
            })
        
        return summary
    
    def save_results(self, robustness_results, precision_results):
        """保存测试结果"""
        LOGGER.info("保存测试结果...")
        
        # 保存鲁棒性测试结果
        if robustness_results:
            robustness_file = os.path.join(self.args.output_dir, 'robustness_results.csv')
            self._save_robustness_csv(robustness_results, robustness_file)
        
        # 保存精度测试结果
        if precision_results:
            precision_file = os.path.join(self.args.output_dir, 'precision_results.json')
            self._save_precision_json(precision_results, precision_file)
        
        # 保存详细结果
        if self.args.save_detailed_results:
            detailed_file = os.path.join(self.args.output_dir, 'detailed_results.npz')
            np.savez(detailed_file, 
                    robustness=robustness_results,
                    precision=precision_results)
    
    def _save_robustness_csv(self, results, filename):
        """保存鲁棒性结果为CSV"""
        data = []
        for model_name, model_results in results.items():
            for i, angle in enumerate(model_results['angles']):
                data.append({
                    'model': model_name,
                    'angle': angle,
                    'mean_error': model_results['mean_errors'][i],
                    'std_error': model_results['std_errors'][i],
                    'success_rate': model_results['success_rates'][i],
                    'mean_time': model_results['mean_times'][i]
                })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        LOGGER.info(f"鲁棒性结果已保存到: {filename}")
    
    def _save_precision_json(self, results, filename):
        """保存精度结果为JSON"""
        import json
        
        # 转换numpy数组为列表
        json_results = {}
        for model_name, model_results in results.items():
            json_results[model_name] = {
                'summary': model_results['summary'],
                'num_samples': len(model_results['errors'])
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        LOGGER.info(f"精度结果已保存到: {filename}")
    
    def generate_plots(self, robustness_results):
        """生成性能曲线图"""
        if not self.args.save_plots or not robustness_results:
            return
        
        LOGGER.info("生成性能曲线图...")
        
        plt.style.use('seaborn-v0_8')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        for model_name, results in robustness_results.items():
            angles = results['angles']
            
            # 误差曲线
            ax1.plot(angles, results['mean_errors'], 'o-', label=f'{model_name}', linewidth=2, markersize=6)
            ax1.fill_between(angles, 
                           np.array(results['mean_errors']) - np.array(results['std_errors']),
                           np.array(results['mean_errors']) + np.array(results['std_errors']),
                           alpha=0.3)
            
            # 成功率曲线
            ax2.plot(angles, [r*100 for r in results['success_rates']], 'o-', 
                    label=f'{model_name}', linewidth=2, markersize=6)
            
            # 时间曲线
            ax3.plot(angles, results['mean_times'], 'o-', 
                    label=f'{model_name}', linewidth=2, markersize=6)
            
            # 误差分布（最后一个角度）
            if model_name in ['original', 'improved']:
                # 这里可以添加误差分布的直方图
                pass
        
        # 设置图表
        ax1.set_xlabel('扰动角度 (度)')
        ax1.set_ylabel('平均旋转误差 (度)')
        ax1.set_title('鲁棒性测试 - 误差 vs 扰动角度')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('扰动角度 (度)')
        ax2.set_ylabel('成功率 (%)')
        ax2.set_title('鲁棒性测试 - 成功率 vs 扰动角度')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.set_xlabel('扰动角度 (度)')
        ax3.set_ylabel('平均推理时间 (秒)')
        ax3.set_title('性能测试 - 时间 vs 扰动角度')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 移除第四个子图或用于其他用途
        ax4.axis('off')
        ax4.text(0.5, 0.5, f'综合测试报告\n\n测试时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
                           f'扰动角度: {self.args.perturbation_angles}\n'
                           f'每角度样本数: {self.args.num_samples_per_angle}',
                 ha='center', va='center', fontsize=12, 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = os.path.join(self.args.output_dir, 'performance_curves.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        LOGGER.info(f"性能曲线图已保存到: {plot_file}")
    
    def generate_report(self, robustness_results, precision_results):
        """生成综合测试报告"""
        LOGGER.info("生成综合测试报告...")
        
        report = []
        report.append("=" * 80)
        report.append("PointNetLK 综合测试报告")
        report.append("=" * 80)
        report.append("")
        report.append(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"测试模型: {self.args.model_type}")
        report.append(f"数据集: {self.args.dataset_path}")
        report.append("")
        
        # 鲁棒性测试结果
        if robustness_results:
            report.append("## 鲁棒性测试结果（系统性扰动）")
            report.append("-" * 50)
            
            for model_name, results in robustness_results.items():
                report.append(f"\n### {model_name.upper()} 模型")
                report.append(f"扰动角度范围: {min(results['angles'])}° - {max(results['angles'])}°")
                report.append(f"平均误差范围: {min(results['mean_errors']):.3f}° - {max(results['mean_errors']):.3f}°")
                report.append(f"成功率范围: {min(results['success_rates']):.1%} - {max(results['success_rates']):.1%}")
                report.append(f"平均推理时间: {np.mean(results['mean_times']):.4f}s")
                
                # 详细角度结果
                report.append("\n详细结果:")
                report.append("角度(°) | 误差(°) | 成功率 | 时间(s)")
                report.append("-" * 40)
                for i, angle in enumerate(results['angles']):
                    report.append(f"{angle:6.1f} | {results['mean_errors'][i]:6.3f} | "
                                f"{results['success_rates'][i]:6.1%} | {results['mean_times'][i]:6.4f}")
        
        # 精度测试结果
        if precision_results:
            report.append("\n\n## 精度测试结果（单一场景）")
            report.append("-" * 50)
            
            for model_name, results in precision_results.items():
                summary = results['summary']
                report.append(f"\n### {model_name.upper()} 模型")
                report.append(f"测试样本数: {summary['total_samples']}")
                report.append(f"平均误差: {summary.get('mean_error', 'N/A'):.6f}°")
                report.append(f"误差标准差: {summary.get('std_error', 'N/A'):.6f}°")
                report.append(f"中位数误差: {summary.get('median_error', 'N/A'):.6f}°")
                report.append(f"平均推理时间: {summary['mean_time']:.6f}s")
                if 'mean_iterations' in summary:
                    report.append(f"平均迭代次数: {summary['mean_iterations']:.2f}")
        
        # 对比分析
        if len(robustness_results) > 1 or len(precision_results) > 1:
            report.append("\n\n## 对比分析")
            report.append("-" * 50)
            
            if len(robustness_results) > 1:
                orig_results = robustness_results.get('original', {})
                impr_results = robustness_results.get('improved', {})
                
                if orig_results and impr_results:
                    report.append("\n### 鲁棒性对比")
                    avg_error_orig = np.mean(orig_results['mean_errors'])
                    avg_error_impr = np.mean(impr_results['mean_errors'])
                    improvement = (avg_error_orig - avg_error_impr) / avg_error_orig * 100
                    
                    report.append(f"平均误差改进: {improvement:.1f}%")
                    report.append(f"原版平均误差: {avg_error_orig:.3f}°")
                    report.append(f"改进版平均误差: {avg_error_impr:.3f}°")
            
            if len(precision_results) > 1:
                orig_summary = precision_results.get('original', {}).get('summary', {})
                impr_summary = precision_results.get('improved', {}).get('summary', {})
                
                if orig_summary and impr_summary:
                    report.append("\n### 精度对比")
                    orig_error = orig_summary.get('mean_error', 0)
                    impr_error = impr_summary.get('mean_error', 0)
                    if orig_error > 0:
                        improvement = (orig_error - impr_error) / orig_error * 100
                        report.append(f"精度改进: {improvement:.1f}%")
                    
                    orig_time = orig_summary.get('mean_time', 0)
                    impr_time = impr_summary.get('mean_time', 0)
                    if orig_time > 0:
                        speedup = orig_time / impr_time if impr_time > 0 else 1
                        report.append(f"速度提升: {speedup:.2f}x")
        
        report.append("\n" + "=" * 80)
        
        # 保存报告
        report_file = os.path.join(self.args.output_dir, 'comprehensive_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        LOGGER.info(f"综合测试报告已保存到: {report_file}")
        
        # 打印摘要到控制台
        print("\n" + "=" * 60)
        print("综合测试完成！")
        print("=" * 60)
        for line in report[:20]:  # 打印前20行
            print(line)
        print(f"\n详细报告请查看: {report_file}")
    
    def run(self):
        """运行综合测试"""
        start_time = time.time()
        
        # 运行鲁棒性测试
        robustness_results = self.run_robustness_test()
        
        # 运行精度测试
        precision_results = self.run_precision_test()
        
        # 保存结果
        self.save_results(robustness_results, precision_results)
        
        # 生成图表
        self.generate_plots(robustness_results)
        
        # 生成报告
        self.generate_report(robustness_results, precision_results)
        
        total_time = time.time() - start_time
        LOGGER.info(f"综合测试完成，总耗时: {total_time:.2f}秒")
        
        return robustness_results, precision_results


def main():
    """主函数"""
    args = parse_arguments()
    
    # 创建综合测试器
    tester = ComprehensiveTester(args)
    
    # 运行测试
    robustness_results, precision_results = tester.run()
    
    return robustness_results, precision_results


if __name__ == '__main__':
    main() 