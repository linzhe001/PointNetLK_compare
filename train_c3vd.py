#!/usr/bin/env python3
"""
C3VD数据集专用训练脚本
Dedicated training script for C3VD dataset with voxelization support
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def parse_arguments():
    """解析C3VD专用参数"""
    parser = argparse.ArgumentParser(description='C3VD数据集PointNetLK训练脚本')
    
    # 必需参数
    parser.add_argument('--c3vd-root', required=True, type=str,
                        help='C3VD数据集根目录路径')
    parser.add_argument('--output-prefix', required=True, type=str,
                        help='输出文件前缀')
    
    # 模型选择
    parser.add_argument('--model-type', default='improved', choices=['original', 'improved'],
                        help='模型类型: original(原版) 或 improved(改进版)')
    
    # C3VD数据集配置
    parser.add_argument('--source-subdir', default='C3VD_ply_source', type=str,
                        help='源点云子目录名称')
    parser.add_argument('--target-subdir', default='visible_point_cloud_ply_depth', type=str,
                        help='目标点云子目录名称')
    parser.add_argument('--pairing-strategy', default='one_to_one',
                        choices=['one_to_one', 'scene_reference', 'source_to_source', 'target_to_target', 'all'],
                        help='点云配对策略')
    parser.add_argument('--transform-mag', default=0.8, type=float,
                        help='Ground Truth变换幅度 (0.5-1.0)')
    
    # 体素化配置（C3VD推荐设置）
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
    
    # 训练配置
    parser.add_argument('--epochs', default=200, type=int,
                        help='训练轮数')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='批次大小（C3VD推荐较小批次）')
    parser.add_argument('--learning-rate', default=0.001, type=float,
                        help='学习率')
    parser.add_argument('--num-points', default=1024, type=int,
                        help='采样点数')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='计算设备')
    parser.add_argument('--workers', default=4, type=int,
                        help='数据加载工作进程数')
    
    # 模型配置
    parser.add_argument('--dim-k', default=1024, type=int,
                        help='特征向量维度')
    parser.add_argument('--max-iter', default=10, type=int,
                        help='LK算法最大迭代次数')
    
    # 原版模型特有参数
    parser.add_argument('--delta', default=1.0e-2, type=float,
                        help='数值雅可比步长（仅原版）')
    parser.add_argument('--learn-delta', action='store_true',
                        help='是否学习步长参数（仅原版）')
    
    # 训练控制
    parser.add_argument('--save-interval', default=10, type=int,
                        help='模型保存间隔')
    parser.add_argument('--log-interval', default=10, type=int,
                        help='日志输出间隔')
    parser.add_argument('--eval-interval', default=5, type=int,
                        help='验证间隔')
    
    # 恢复训练
    parser.add_argument('--pretrained', default='', type=str,
                        help='预训练模型路径')
    parser.add_argument('--resume', default='', type=str,
                        help='恢复训练检查点路径')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='开始轮数')
    
    return parser.parse_args()


def validate_c3vd_dataset(c3vd_root, source_subdir, target_subdir):
    """验证C3VD数据集结构"""
    c3vd_path = Path(c3vd_root)
    
    if not c3vd_path.exists():
        raise FileNotFoundError(f"C3VD数据集根目录不存在: {c3vd_root}")
    
    source_path = c3vd_path / source_subdir
    target_path = c3vd_path / target_subdir
    
    if not source_path.exists():
        raise FileNotFoundError(f"源点云目录不存在: {source_path}")
    
    if not target_path.exists():
        raise FileNotFoundError(f"目标点云目录不存在: {target_path}")
    
    # 检查是否有场景目录
    source_scenes = [d for d in source_path.iterdir() if d.is_dir()]
    target_scenes = [d for d in target_path.iterdir() if d.is_dir()]
    
    if not source_scenes:
        raise ValueError(f"源点云目录中没有找到场景子目录: {source_path}")
    
    if not target_scenes:
        raise ValueError(f"目标点云目录中没有找到场景子目录: {target_path}")
    
    print(f"✅ C3VD数据集验证通过:")
    print(f"   源点云场景数: {len(source_scenes)}")
    print(f"   目标点云场景数: {len(target_scenes)}")
    print(f"   源点云路径: {source_path}")
    print(f"   目标点云路径: {target_path}")
    
    return str(source_path), str(target_path)


def build_training_command(args, source_path, target_path):
    """构建训练命令"""
    cmd = [
        'python', 'train_unified.py',
        '--dataset-type', 'c3vd',
        '--dataset-path', args.c3vd_root,
        '--c3vd-source-root', source_path,
        '--c3vd-target-root', target_path,
        '--c3vd-pairing-strategy', args.pairing_strategy,
        '--c3vd-transform-mag', str(args.transform_mag),
        '--model-type', args.model_type,
        '--outfile', args.output_prefix,
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--learning-rate', str(args.learning_rate),
        '--num-points', str(args.num_points),
        '--device', args.device,
        '--workers', str(args.workers),
        '--dim-k', str(args.dim_k),
        '--max-iter', str(args.max_iter),
        '--save-interval', str(args.save_interval),
        '--log-interval', str(args.log_interval),
        '--eval-interval', str(args.eval_interval),
        '--voxel-size', str(args.voxel_size),
        '--voxel-grid-size', str(args.voxel_grid_size),
        '--max-voxel-points', str(args.max_voxel_points),
        '--max-voxels', str(args.max_voxels),
        '--min-voxel-points-ratio', str(args.min_voxel_points_ratio),
    ]
    
    # 原版模型特有参数
    if args.model_type == 'original':
        cmd.extend(['--delta', str(args.delta)])
        if args.learn_delta:
            cmd.append('--learn-delta')
    
    # 恢复训练参数
    if args.pretrained:
        cmd.extend(['--pretrained', args.pretrained])
    
    if args.resume:
        cmd.extend(['--resume', args.resume])
        cmd.extend(['--start-epoch', str(args.start_epoch)])
    
    return cmd


def print_training_info(args, source_path, target_path):
    """打印训练信息"""
    print("\n" + "="*80)
    print("🚀 C3VD数据集PointNetLK训练")
    print("="*80)
    print(f"📊 数据集配置:")
    print(f"   根目录: {args.c3vd_root}")
    print(f"   源点云: {source_path}")
    print(f"   目标点云: {target_path}")
    print(f"   配对策略: {args.pairing_strategy}")
    print(f"   变换幅度: {args.transform_mag}")
    
    print(f"\n🧠 模型配置:")
    print(f"   模型类型: {args.model_type}")
    print(f"   特征维度: {args.dim_k}")
    print(f"   最大迭代: {args.max_iter}")
    if args.model_type == 'original':
        print(f"   数值步长: {args.delta}")
        print(f"   学习步长: {args.learn_delta}")
    
    print(f"\n🔧 体素化配置:")
    print(f"   体素大小: {args.voxel_size}")
    print(f"   网格大小: {args.voxel_grid_size}")
    print(f"   最大体素点数: {args.max_voxel_points}")
    print(f"   最大体素数: {args.max_voxels}")
    print(f"   最小点数比例: {args.min_voxel_points_ratio}")
    
    print(f"\n⚙️ 训练配置:")
    print(f"   训练轮数: {args.epochs}")
    print(f"   批次大小: {args.batch_size}")
    print(f"   学习率: {args.learning_rate}")
    print(f"   采样点数: {args.num_points}")
    print(f"   计算设备: {args.device}")
    print(f"   工作进程: {args.workers}")
    
    print(f"\n💾 输出配置:")
    print(f"   输出前缀: {args.output_prefix}")
    print(f"   保存间隔: {args.save_interval} epochs")
    print(f"   日志间隔: {args.log_interval} batches")
    print(f"   验证间隔: {args.eval_interval} epochs")
    
    if args.pretrained:
        print(f"\n🔄 预训练模型: {args.pretrained}")
    
    if args.resume:
        print(f"\n🔄 恢复训练: {args.resume} (从第{args.start_epoch}轮开始)")
    
    print("="*80)


def main():
    """主函数"""
    args = parse_arguments()
    
    try:
        # 验证数据集
        print("🔍 验证C3VD数据集结构...")
        source_path, target_path = validate_c3vd_dataset(
            args.c3vd_root, args.source_subdir, args.target_subdir
        )
        
        # 打印训练信息
        print_training_info(args, source_path, target_path)
        
        # 构建训练命令
        cmd = build_training_command(args, source_path, target_path)
        
        # 确认开始训练
        print(f"\n🎯 即将执行训练命令:")
        print(f"   {' '.join(cmd)}")
        
        response = input("\n是否开始训练? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ 训练已取消")
            return
        
        # 执行训练
        print("\n🚀 开始训练...")
        print("-"*80)
        
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode == 0:
            print("\n✅ 训练完成!")
        else:
            print(f"\n❌ 训练失败，退出码: {result.returncode}")
            sys.exit(result.returncode)
            
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 