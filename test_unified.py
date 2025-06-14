"""
统一测试脚本 - 支持原版和改进版PointNetLK的测试
Unified Testing Script - Support testing for both original and improved PointNetLK
"""

import argparse
import os
import sys
import logging
import numpy as np
import torch
import torch.utils.data
import time
from datetime import datetime

# 导入桥接模块和对比分析模块
from bridge import ModelBridge, DataBridge
from comparison import ModelComparison

# 设置日志
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='统一PointNetLK测试脚本')
    
    # 测试模式
    parser.add_argument('--test-mode', default='single', choices=['single', 'comparison'],
                        help='测试模式: single(单模型测试) 或 comparison(对比测试)')
    parser.add_argument('--model-type', default='improved', choices=['original', 'improved'],
                        help='模型类型 (仅单模型测试时使用)')
    
    # 必需参数
    parser.add_argument('--model-path', required=True, type=str,
                        help='模型文件路径')
    parser.add_argument('--dataset-path', required=True, type=str,
                        help='数据集路径')
    parser.add_argument('-o', '--outfile', required=True, type=str,
                        help='输出文件前缀')
    
    # 对比测试参数
    parser.add_argument('--original-model-path', default='', type=str,
                        help='原版模型路径 (对比测试时使用)')
    parser.add_argument('--improved-model-path', default='', type=str,
                        help='改进版模型路径 (对比测试时使用)')
    
    # 数据集设置
    parser.add_argument('--dataset-type', default='modelnet', 
                        choices=['modelnet', 'shapenet2', 'kitti', '3dmatch'],
                        help='数据集类型')
    parser.add_argument('--num-points', default=1024, type=int,
                        help='点云中的点数')
    parser.add_argument('--categoryfile', default='', type=str,
                        help='类别文件路径（ModelNet需要）')
    
    # 模型设置
    parser.add_argument('--dim-k', default=1024, type=int,
                        help='特征向量维度')
    parser.add_argument('--max-iter', default=10, type=int,
                        help='LK算法最大迭代次数')
    parser.add_argument('--xtol', default=1e-7, type=float,
                        help='收敛阈值')
    
    # 测试设置
    parser.add_argument('--batch-size', default=32, type=int,
                        help='批次大小')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='计算设备')
    parser.add_argument('--workers', default=4, type=int,
                        help='数据加载工作进程数')
    parser.add_argument('--num-test-samples', default=-1, type=int,
                        help='测试样本数量 (-1表示全部)')
    
    # 分析设置
    parser.add_argument('--save-results', action='store_true',
                        help='保存详细测试结果')
    parser.add_argument('--generate-report', action='store_true',
                        help='生成测试报告')
    parser.add_argument('--analyze-convergence', action='store_true',
                        help='分析收敛行为')
    parser.add_argument('--benchmark-jacobian', action='store_true',
                        help='基准测试雅可比计算')
    
    return parser.parse_args()


class UnifiedTester:
    """统一测试器"""
    
    def __init__(self, args):
        """
        初始化测试器
        
        Args:
            args: 命令行参数
        """
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        self.output_dir = os.path.dirname(args.outfile) if os.path.dirname(args.outfile) else './results'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志文件
        log_file = os.path.join(self.output_dir, f'test_{args.test_mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)
        
        LOGGER.info(f"开始 {args.test_mode} 测试")
        LOGGER.info(f"参数: {vars(args)}")
        
        # 初始化数据
        self.test_loader = self._create_data_loader()
        
        # 初始化模型或对比器
        if args.test_mode == 'single':
            self.model = self._create_single_model()
        else:
            self.comparator = self._create_comparator()
    
    def _create_single_model(self):
        """创建单个模型"""
        LOGGER.info(f"创建 {self.args.model_type} 模型...")
        
        # 创建模型
        model_kwargs = {'dim_k': self.args.dim_k}
        model = ModelBridge(self.args.model_type, **model_kwargs)
        model.to(self.device)
        
        # 加载权重
        LOGGER.info(f"加载模型权重: {self.args.model_path}")
        checkpoint = torch.load(self.args.model_path, map_location=self.device, weights_only=False)
        
        # 处理不同的检查点格式
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def _create_comparator(self):
        """创建模型对比器"""
        LOGGER.info("创建模型对比器...")
        
        comparator = ModelComparison(dim_k=self.args.dim_k, device=self.device)
        
        # 加载预训练模型
        original_path = self.args.original_model_path or self.args.model_path
        improved_path = self.args.improved_model_path or self.args.model_path
        
        comparator.load_pretrained_models(original_path, improved_path)
        
        return comparator
    
    def _create_data_loader(self):
        """创建数据加载器"""
        LOGGER.info(f"创建 {self.args.dataset_type} 测试数据加载器...")
        
        # 根据测试模式选择数据源
        if self.args.test_mode == 'single':
            data_source = 'original' if self.args.model_type == 'original' else 'improved'
        else:
            data_source = 'improved'  # 对比测试使用改进版数据加载
        
        data_bridge = DataBridge(
            dataset_type=self.args.dataset_type,
            data_source=data_source
        )
        
        # 准备数据集参数
        dataset_kwargs = {
            'dataset_path': self.args.dataset_path,
            'num_points': self.args.num_points,
        }
        
        if self.args.categoryfile:
            dataset_kwargs['categoryfile'] = self.args.categoryfile
        
        # 获取测试数据集
        _, testset = data_bridge.get_datasets(**dataset_kwargs)
        
        # 限制测试样本数量
        if self.args.num_test_samples > 0 and len(testset) > self.args.num_test_samples:
            indices = torch.randperm(len(testset))[:self.args.num_test_samples]
            testset = torch.utils.data.Subset(testset, indices)
        
        # 创建数据加载器
        test_loader = data_bridge.get_dataloader(
            testset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers
        )
        
        LOGGER.info(f"测试集大小: {len(testset)}")
        return test_loader
    
    def test_single_model(self):
        """测试单个模型"""
        LOGGER.info("开始单模型测试...")
        
        results = {
            'errors': [],      # 旋转误差
            'trans_errors': [], # 平移误差
            'times': [],
            'iterations': [],
            'convergence_info': []
        }
        
        total_time = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                if batch_idx % 10 == 0:
                    LOGGER.info(f"处理批次 {batch_idx+1}/{len(self.test_loader)}")
                
                # 解析数据
                p0, p1, igt = self._parse_batch_data(data)
                p0 = p0.to(self.device)
                p1 = p1.to(self.device)
                igt = igt.to(self.device)
                
                # 前向传播
                start_time = time.time()
                
                # 对于改进版模型，需要启用梯度计算
                if self.args.model_type == 'improved':
                    p0.requires_grad_(True)
                    p1.requires_grad_(True)
                    with torch.enable_grad():
                        r, g = self.model.forward(
                            p0, p1, 
                            maxiter=self.args.max_iter, 
                            xtol=self.args.xtol,
                            mode='test'
                        )
                else:
                    r, g = self.model.forward(
                        p0, p1, 
                        maxiter=self.args.max_iter, 
                        xtol=self.args.xtol,
                        mode='test'
                    )
                
                inference_time = time.time() - start_time
                
                # 计算误差
                if g is not None:
                    rot_error, trans_error = self._compute_transformation_error(g, igt)
                    results['errors'].extend(rot_error.cpu().numpy())
                    results['trans_errors'].extend(trans_error.cpu().numpy())
                
                # 记录时间和迭代信息
                results['times'].append(inference_time)
                total_time += inference_time
                total_samples += p0.size(0)
                
                # 记录迭代次数（如果可用）
                if hasattr(self.model.get_model(), 'itr'):
                    results['iterations'].append(self.model.get_model().itr)
        
        # 计算统计信息
        summary = self._compute_test_summary(results, total_time, total_samples)
        
        LOGGER.info("单模型测试完成")
        LOGGER.info(f"平均误差: {summary['mean_error']:.6f}")
        LOGGER.info(f"平均时间: {summary['mean_time']:.6f}s")
        LOGGER.info(f"总样本数: {summary['total_samples']}")
        
        return results, summary
    
    def test_comparison(self):
        """对比测试"""
        LOGGER.info("开始对比测试...")
        
        # 准备测试数据
        test_data = []
        for data in self.test_loader:
            p0, p1, igt = self._parse_batch_data(data)
            test_data.append((p0, p1, igt))
        
        # 运行对比分析
        comparison_results = self.comparator.compare_models(
            test_data, 
            maxiter=self.args.max_iter, 
            xtol=self.args.xtol
        )
        
        # 可选的额外分析
        if self.args.analyze_convergence:
            LOGGER.info("分析收敛行为...")
            convergence_results = self.comparator.compare_convergence_behavior(test_data)
            comparison_results['convergence'] = convergence_results
        
        if self.args.benchmark_jacobian:
            LOGGER.info("基准测试雅可比计算...")
            # 使用第一个批次的数据进行雅可比测试
            if test_data:
                jacobian_results = self.comparator.compare_jacobian_computation(test_data[0][0])
                comparison_results['jacobian'] = jacobian_results
        
        LOGGER.info("对比测试完成")
        return comparison_results
    
    def _parse_batch_data(self, data):
        """解析批次数据"""
        if len(data) == 3:
            return data
        elif isinstance(data, (list, tuple)) and len(data) >= 2:
            p0, p1 = data[0], data[1]
            # 生成随机变换作为真实值
            batch_size = p0.size(0)
            igt = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
            return p0, p1, igt
        else:
            raise ValueError(f"不支持的数据格式: {type(data)}")
    
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
        
        return rot_error, trans_error
    
    def _compute_test_summary(self, results, total_time, total_samples):
        """计算测试摘要"""
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
    
    def save_results(self, results, summary=None):
        """保存测试结果"""
        if not self.args.save_results:
            return
        
        results_file = f"{self.args.outfile}_results.npz"
        
        if self.args.test_mode == 'single':
            np.savez(results_file, 
                    errors=results['errors'],
                    trans_errors=results['trans_errors'],
                    times=results['times'],
                    iterations=results['iterations'],
                    summary=summary)
        else:
            # 保存对比结果
            np.savez(results_file, **results)
        
        LOGGER.info(f"结果已保存到: {results_file}")
    
    def generate_report(self, results, summary=None):
        """生成测试报告"""
        if not self.args.generate_report:
            return
        
        report_file = f"{self.args.outfile}_report.txt"
        
        if self.args.test_mode == 'single':
            report = self._generate_single_model_report(summary)
        else:
            report = self.comparator.generate_comparison_report()
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        LOGGER.info(f"报告已保存到: {report_file}")
        return report
    
    def _generate_single_model_report(self, summary):
        """生成单模型测试报告"""
        report = []
        report.append("=" * 60)
        report.append(f"{self.args.model_type.upper()} PointNetLK 测试报告")
        report.append("=" * 60)
        report.append("")
        
        report.append("## 测试配置")
        report.append(f"模型类型: {self.args.model_type}")
        report.append(f"数据集: {self.args.dataset_type}")
        report.append(f"测试样本数: {summary['total_samples']}")
        report.append(f"最大迭代次数: {self.args.max_iter}")
        report.append(f"收敛阈值: {self.args.xtol}")
        report.append("")
        
        report.append("## 测试结果")
        if 'mean_error' in summary:
            report.append(f"平均误差: {summary['mean_error']:.6f}")
            report.append(f"误差标准差: {summary['std_error']:.6f}")
            report.append(f"中位数误差: {summary['median_error']:.6f}")
            report.append(f"最小误差: {summary['min_error']:.6f}")
            report.append(f"最大误差: {summary['max_error']:.6f}")
        
        report.append(f"平均推理时间: {summary['mean_time']:.6f}s")
        report.append(f"总推理时间: {summary['total_time']:.2f}s")
        
        if 'mean_iterations' in summary:
            report.append(f"平均迭代次数: {summary['mean_iterations']:.2f}")
            report.append(f"迭代次数标准差: {summary['std_iterations']:.2f}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def run(self):
        """运行测试"""
        if self.args.test_mode == 'single':
            results, summary = self.test_single_model()
            self.save_results(results, summary)
            self.generate_report(results, summary)
            return results, summary
        else:
            results = self.test_comparison()
            self.save_results(results)
            self.generate_report(results)
            return results


def main():
    """主函数"""
    args = parse_arguments()
    
    # 创建测试器
    tester = UnifiedTester(args)
    
    # 运行测试
    results = tester.run()
    
    print("测试完成！")
    return results


if __name__ == '__main__':
    main() 