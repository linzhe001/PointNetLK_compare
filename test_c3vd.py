#!/usr/bin/env python3
"""
C3VD数据集专用测试脚本
Dedicated testing script for C3VD dataset with voxelization support
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.utils.data
import time
from pathlib import Path
import logging

# 导入数据处理模块
from data_utils import create_c3vd_dataset
from bridge import ModelBridge

# 设置日志
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def parse_arguments():
    """解析C3VD测试参数"""
    parser = argparse.ArgumentParser(description='C3VD数据集PointNetLK测试脚本')
    
    # 必需参数
    parser.add_argument('--c3vd-root', required=True, type=str,
                        help='C3VD数据集根目录路径')
    parser.add_argument('--model-path', required=True, type=str,
                        help='训练好的模型路径')
    parser.add_argument('--output-dir', required=True, type=str,
                        help='测试结果输出目录')
    
    # 模型配置
    parser.add_argument('--model-type', default='improved', choices=['original', 'improved'],
                        help='模型类型: original(原版) 或 improved(改进版)')
    parser.add_argument('--dim-k', default=1024, type=int,
                        help='特征向量维度')
    parser.add_argument('--max-iter', default=10, type=int,
                        help='LK算法最大迭代次数')
    
    # 原版模型特有参数
    parser.add_argument('--delta', default=1.0e-2, type=float,
                        help='数值雅可比步长（仅原版）')
    
    # C3VD数据集配置
    parser.add_argument('--source-subdir', default='C3VD_ply_source', type=str,
                        help='源点云子目录名称')
    parser.add_argument('--target-subdir', default='visible_point_cloud_ply_depth', type=str,
                        help='目标点云子目录名称')
    parser.add_argument('--pairing-strategy', default='one_to_one',
                        choices=['one_to_one', 'scene_reference', 'source_to_source', 'target_to_target', 'all'],
                        help='点云配对策略')
    
    # 测试配置
    parser.add_argument('--test-transform-mags', default='0.2,0.4,0.6,0.8', type=str,
                        help='测试变换幅度列表（逗号分隔）')
    parser.add_argument('--batch-size', default=8, type=int,
                        help='测试批次大小')
    parser.add_argument('--num-points', default=1024, type=int,
                        help='采样点数')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='计算设备')
    parser.add_argument('--workers', default=4, type=int,
                        help='数据加载工作进程数')
    
    # 体素化配置
    parser.add_argument('--voxel-size', default=0.05, type=float,
                        help='体素大小')
    parser.add_argument('--voxel-grid-size', default=32, type=int,
                        help='体素网格大小')
    parser.add_argument('--max-voxel-points', default=100, type=int,
                        help='每个体素最大点数')
    parser.add_argument('--max-voxels', default=20000, type=int,
                        help='最大体素数量')
    parser.add_argument('--min-voxel-points-ratio', default=0.1, type=float,
                        help='最小体素点数比例')
    
    # 评估配置
    parser.add_argument('--save-results', action='store_true',
                        help='是否保存详细测试结果')
    parser.add_argument('--visualize', action='store_true',
                        help='是否生成可视化结果')
    
    return parser.parse_args()


class C3VDTester:
    """C3VD测试器"""
    
    def __init__(self, args):
        """初始化测试器"""
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 设置日志
        log_file = os.path.join(args.output_dir, 'test_results.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)
        
        LOGGER.info(f"开始C3VD数据集测试")
        LOGGER.info(f"参数: {vars(args)}")
        
        # 验证数据集
        self.source_path, self.target_path = self._validate_dataset()
        
        # 加载模型
        self.model = self._load_model()
        
        # 解析测试变换幅度
        self.test_mags = [float(x.strip()) for x in args.test_transform_mags.split(',')]
        LOGGER.info(f"测试变换幅度: {self.test_mags}")
    
    def _validate_dataset(self):
        """验证数据集结构"""
        c3vd_path = Path(self.args.c3vd_root)
        
        if not c3vd_path.exists():
            raise FileNotFoundError(f"C3VD数据集根目录不存在: {self.args.c3vd_root}")
        
        source_path = c3vd_path / self.args.source_subdir
        target_path = c3vd_path / self.args.target_subdir
        
        if not source_path.exists():
            raise FileNotFoundError(f"源点云目录不存在: {source_path}")
        
        if not target_path.exists():
            raise FileNotFoundError(f"目标点云目录不存在: {target_path}")
        
        LOGGER.info(f"数据集验证通过: {source_path}, {target_path}")
        return str(source_path), str(target_path)
    
    def _load_model(self):
        """加载训练好的模型"""
        LOGGER.info(f"加载模型: {self.args.model_path}")
        
        # 创建模型
        model_kwargs = {
            'dim_k': self.args.dim_k,
        }
        
        if self.args.model_type == 'original':
            model_kwargs.update({
                'delta': self.args.delta,
                'learn_delta': False,  # 测试时不学习
            })
        
        model = ModelBridge(self.args.model_type, **model_kwargs)
        
        # 加载权重
        if not os.path.exists(self.args.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.args.model_path}")
        
        checkpoint = torch.load(self.args.model_path, map_location=self.device)
        
        # 处理不同的保存格式
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        LOGGER.info(f"模型加载完成并移动到 {self.device}")
        return model
    
    def _create_test_dataset(self, transform_mag):
        """创建测试数据集"""
        # 体素化配置
        voxel_config = {
            'voxel_size': self.args.voxel_size,
            'voxel_grid_size': self.args.voxel_grid_size,
            'max_voxel_points': self.args.max_voxel_points,
            'max_voxels': self.args.max_voxels,
            'min_voxel_points_ratio': self.args.min_voxel_points_ratio
        }
        
        # 智能采样配置
        sampling_config = {
            'target_points': self.args.num_points,
            'intersection_priority': True,
            'min_intersection_ratio': 0.3,
            'max_intersection_ratio': 0.7
        }
        
        # 创建测试数据集
        testset = create_c3vd_dataset(
            source_root=self.source_path,
            target_root=self.target_path,
            pairing_strategy=self.args.pairing_strategy,
            mag=transform_mag,
            train=False,  # 测试模式
            vis=self.args.visualize,
            voxel_config=voxel_config,
            sampling_config=sampling_config
        )
        
        # 创建数据加载器
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True,
            drop_last=False
        )
        
        return testset, test_loader
    
    def _compute_metrics(self, pred_transform, gt_transform):
        """计算评估指标"""
        # 转换为numpy数组
        if torch.is_tensor(pred_transform):
            pred_transform = pred_transform.detach().cpu().numpy()
        if torch.is_tensor(gt_transform):
            gt_transform = gt_transform.detach().cpu().numpy()
        
        batch_size = pred_transform.shape[0]
        
        # 计算旋转和平移误差
        rotation_errors = []
        translation_errors = []
        
        for i in range(batch_size):
            pred_R = pred_transform[i, :3, :3]
            pred_t = pred_transform[i, :3, 3]
            gt_R = gt_transform[i, :3, :3]
            gt_t = gt_transform[i, :3, 3]
            
            # 旋转误差（角度）
            R_diff = np.dot(pred_R, gt_R.T)
            trace_R = np.trace(R_diff)
            # 确保trace在有效范围内
            trace_R = np.clip(trace_R, -1.0, 3.0)
            rotation_error = np.arccos((trace_R - 1) / 2) * 180 / np.pi
            rotation_errors.append(rotation_error)
            
            # 平移误差（欧几里得距离）
            translation_error = np.linalg.norm(pred_t - gt_t)
            translation_errors.append(translation_error)
        
        return np.array(rotation_errors), np.array(translation_errors)
    
    def test_single_magnitude(self, transform_mag):
        """测试单个变换幅度"""
        LOGGER.info(f"测试变换幅度: {transform_mag}")
        
        # 创建测试数据集
        testset, test_loader = self._create_test_dataset(transform_mag)
        
        # 测试统计
        total_samples = 0
        total_rotation_error = 0.0
        total_translation_error = 0.0
        all_rotation_errors = []
        all_translation_errors = []
        all_intersection_ratios = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                # 解析数据
                source = data['source'].to(self.device)
                target = data['target'].to(self.device)
                igt = data['igt'].to(self.device)
                
                # 获取交集比例（如果有）
                if 'intersection_ratio' in data:
                    intersection_ratios = data['intersection_ratio'].cpu().numpy()
                    all_intersection_ratios.extend(intersection_ratios)
                
                try:
                    # 模型推理
                    if self.args.model_type == 'improved':
                        # 改进版需要启用梯度计算
                        source.requires_grad_(True)
                        target.requires_grad_(True)
                        with torch.enable_grad():
                            pred_transform = self.model.forward(source, target, maxiter=self.args.max_iter)
                    else:
                        pred_transform = self.model.forward(source, target, maxiter=self.args.max_iter)
                    
                    # 计算误差
                    rotation_errors, translation_errors = self._compute_metrics(pred_transform, igt)
                    
                    # 统计
                    total_samples += len(rotation_errors)
                    total_rotation_error += np.sum(rotation_errors)
                    total_translation_error += np.sum(translation_errors)
                    all_rotation_errors.extend(rotation_errors)
                    all_translation_errors.extend(translation_errors)
                    
                    # 打印进度
                    if batch_idx % 10 == 0:
                        LOGGER.info(f"  批次 {batch_idx}/{len(test_loader)}, "
                                   f"平均旋转误差: {np.mean(rotation_errors):.3f}°, "
                                   f"平均平移误差: {np.mean(translation_errors):.4f}")
                
                except Exception as e:
                    LOGGER.warning(f"批次 {batch_idx} 处理失败: {e}")
                    continue
        
        # 计算最终统计
        test_time = time.time() - start_time
        
        if total_samples > 0:
            avg_rotation_error = total_rotation_error / total_samples
            avg_translation_error = total_translation_error / total_samples
            
            # 计算中位数和标准差
            median_rotation_error = np.median(all_rotation_errors)
            std_rotation_error = np.std(all_rotation_errors)
            median_translation_error = np.median(all_translation_errors)
            std_translation_error = np.std(all_translation_errors)
            
            # 计算成功率（旋转误差<5度，平移误差<0.1）
            success_rotation = np.sum(np.array(all_rotation_errors) < 5.0) / len(all_rotation_errors)
            success_translation = np.sum(np.array(all_translation_errors) < 0.1) / len(all_translation_errors)
            success_overall = np.sum((np.array(all_rotation_errors) < 5.0) & 
                                   (np.array(all_translation_errors) < 0.1)) / len(all_rotation_errors)
            
            results = {
                'transform_mag': transform_mag,
                'total_samples': total_samples,
                'avg_rotation_error': avg_rotation_error,
                'avg_translation_error': avg_translation_error,
                'median_rotation_error': median_rotation_error,
                'median_translation_error': median_translation_error,
                'std_rotation_error': std_rotation_error,
                'std_translation_error': std_translation_error,
                'success_rotation': success_rotation,
                'success_translation': success_translation,
                'success_overall': success_overall,
                'test_time': test_time,
                'all_rotation_errors': all_rotation_errors,
                'all_translation_errors': all_translation_errors,
                'all_intersection_ratios': all_intersection_ratios
            }
            
            LOGGER.info(f"变换幅度 {transform_mag} 测试完成:")
            LOGGER.info(f"  样本数: {total_samples}")
            LOGGER.info(f"  平均旋转误差: {avg_rotation_error:.3f}° (中位数: {median_rotation_error:.3f}°)")
            LOGGER.info(f"  平均平移误差: {avg_translation_error:.4f} (中位数: {median_translation_error:.4f})")
            LOGGER.info(f"  成功率: {success_overall:.1%} (旋转<5°且平移<0.1)")
            LOGGER.info(f"  测试时间: {test_time:.2f}s")
            
            return results
        else:
            LOGGER.error(f"变换幅度 {transform_mag} 没有有效样本")
            return None
    
    def run_comprehensive_test(self):
        """运行全面测试"""
        LOGGER.info("开始全面测试...")
        
        all_results = []
        
        for mag in self.test_mags:
            result = self.test_single_magnitude(mag)
            if result:
                all_results.append(result)
        
        # 保存结果
        if self.args.save_results:
            self._save_results(all_results)
        
        # 打印总结
        self._print_summary(all_results)
        
        return all_results
    
    def _save_results(self, results):
        """保存测试结果"""
        import json
        
        # 保存JSON格式结果
        json_results = []
        for result in results:
            json_result = result.copy()
            # 转换numpy数组为列表
            json_result['all_rotation_errors'] = [float(x) for x in result['all_rotation_errors']]
            json_result['all_translation_errors'] = [float(x) for x in result['all_translation_errors']]
            json_result['all_intersection_ratios'] = [float(x) for x in result['all_intersection_ratios']]
            json_results.append(json_result)
        
        json_file = os.path.join(self.args.output_dir, 'test_results.json')
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        LOGGER.info(f"测试结果已保存到: {json_file}")
        
        # 保存CSV格式摘要
        csv_file = os.path.join(self.args.output_dir, 'test_summary.csv')
        with open(csv_file, 'w') as f:
            f.write("transform_mag,samples,avg_rot_error,avg_trans_error,median_rot_error,median_trans_error,success_rate\n")
            for result in results:
                f.write(f"{result['transform_mag']},{result['total_samples']},"
                       f"{result['avg_rotation_error']:.3f},{result['avg_translation_error']:.4f},"
                       f"{result['median_rotation_error']:.3f},{result['median_translation_error']:.4f},"
                       f"{result['success_overall']:.3f}\n")
        
        LOGGER.info(f"测试摘要已保存到: {csv_file}")
    
    def _print_summary(self, results):
        """打印测试总结"""
        print("\n" + "="*80)
        print("🎯 C3VD数据集测试总结")
        print("="*80)
        
        print(f"模型类型: {self.args.model_type}")
        print(f"模型路径: {self.args.model_path}")
        print(f"数据集路径: {self.args.c3vd_root}")
        print(f"配对策略: {self.args.pairing_strategy}")
        
        print("\n📊 测试结果:")
        print("-"*80)
        print(f"{'变换幅度':<10} {'样本数':<8} {'旋转误差(°)':<12} {'平移误差':<10} {'成功率':<8}")
        print("-"*80)
        
        for result in results:
            print(f"{result['transform_mag']:<10.1f} {result['total_samples']:<8} "
                  f"{result['avg_rotation_error']:<12.3f} {result['avg_translation_error']:<10.4f} "
                  f"{result['success_overall']:<8.1%}")
        
        print("="*80)


def main():
    """主函数"""
    args = parse_arguments()
    
    try:
        # 创建测试器
        tester = C3VDTester(args)
        
        # 运行测试
        results = tester.run_comprehensive_test()
        
        if results:
            print("\n✅ 测试完成!")
        else:
            print("\n❌ 测试失败!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 