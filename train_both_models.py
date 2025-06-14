#!/usr/bin/env python3
"""
批量训练脚本 - 自动训练原版和改进版PointNetLK模型
Batch Training Script - Automatically train both original and improved PointNetLK models
"""

import os
import sys
import subprocess
import argparse
import time
from datetime import datetime


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='批量训练原版和改进版PointNetLK模型')
    
    # 数据集设置
    parser.add_argument('--dataset-path', required=True, type=str,
                        help='数据集路径')
    parser.add_argument('--dataset-type', default='modelnet', 
                        choices=['modelnet', 'shapenet2', 'kitti', '3dmatch'],
                        help='数据集类型')
    parser.add_argument('--num-points', default=1024, type=int,
                        help='点云中的点数')
    parser.add_argument('--categoryfile', default='', type=str,
                        help='类别文件路径（ModelNet需要）')
    
    # 训练设置
    parser.add_argument('--epochs', default=50, type=int,
                        help='训练轮数')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='批次大小')
    parser.add_argument('--learning-rate', default=0.001, type=float,
                        help='学习率')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='计算设备')
    
    # 输出设置
    parser.add_argument('--output-dir', default='logs', type=str,
                        help='输出目录')
    parser.add_argument('--model-prefix', default='model', type=str,
                        help='模型文件前缀')
    
    # 训练选项
    parser.add_argument('--models', default='both', choices=['original', 'improved', 'both'],
                        help='训练哪些模型: original, improved, 或 both')
    parser.add_argument('--sequential', action='store_true',
                        help='顺序训练（默认并行训练）')
    
    return parser.parse_args()


def build_training_command(model_type, args):
    """构建训练命令"""
    cmd = [
        'python', 'train_unified.py',
        '--model-type', model_type,
        '--outfile', os.path.join(args.output_dir, f'{args.model_prefix}_{model_type}'),
        '--dataset-path', args.dataset_path,
        '--dataset-type', args.dataset_type,
        '--num-points', str(args.num_points),
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--learning-rate', str(args.learning_rate),
        '--device', args.device,
    ]
    
    if args.categoryfile:
        cmd.extend(['--categoryfile', args.categoryfile])
    
    return cmd


def run_training(model_type, args):
    """运行单个模型的训练"""
    print(f"\n{'='*60}")
    print(f"开始训练 {model_type.upper()} 模型")
    print(f"{'='*60}")
    
    start_time = time.time()
    cmd = build_training_command(model_type, args)
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        # 运行训练命令
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{model_type.upper()} 模型训练完成!")
        print(f"训练时间: {duration:.2f}秒 ({duration/60:.2f}分钟)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {model_type.upper()} 模型训练失败!")
        print(f"错误代码: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  {model_type.upper()} 模型训练被用户中断!")
        return False


def run_parallel_training(args):
    """并行训练两个模型"""
    import threading
    import queue
    
    print("\n🚀 开始并行训练两个模型...")
    
    results = queue.Queue()
    
    def train_worker(model_type):
        success = run_training(model_type, args)
        results.put((model_type, success))
    
    # 创建训练线程
    original_thread = threading.Thread(target=train_worker, args=('original',))
    improved_thread = threading.Thread(target=train_worker, args=('improved',))
    
    # 启动训练
    start_time = time.time()
    original_thread.start()
    improved_thread.start()
    
    # 等待完成
    original_thread.join()
    improved_thread.join()
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # 收集结果
    training_results = {}
    while not results.empty():
        model_type, success = results.get()
        training_results[model_type] = success
    
    print(f"\n{'='*60}")
    print("并行训练完成!")
    print(f"总时间: {total_duration:.2f}秒 ({total_duration/60:.2f}分钟)")
    print(f"原版模型: {'✅ 成功' if training_results.get('original', False) else '❌ 失败'}")
    print(f"改进版模型: {'✅ 成功' if training_results.get('improved', False) else '❌ 失败'}")
    print(f"{'='*60}")
    
    return training_results


def run_sequential_training(args):
    """顺序训练两个模型"""
    print("\n🔄 开始顺序训练两个模型...")
    
    total_start_time = time.time()
    results = {}
    
    # 训练原版模型
    results['original'] = run_training('original', args)
    
    # 训练改进版模型
    results['improved'] = run_training('improved', args)
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print(f"\n{'='*60}")
    print("顺序训练完成!")
    print(f"总时间: {total_duration:.2f}秒 ({total_duration/60:.2f}分钟)")
    print(f"原版模型: {'✅ 成功' if results['original'] else '❌ 失败'}")
    print(f"改进版模型: {'✅ 成功' if results['improved'] else '❌ 失败'}")
    print(f"{'='*60}")
    
    return results


def main():
    """主函数"""
    args = parse_arguments()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("PointNetLK 批量训练脚本")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据集路径: {args.dataset_path}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"点云点数: {args.num_points}")
    print(f"输出目录: {args.output_dir}")
    
    if args.models == 'both':
        # 训练两个模型
        if args.sequential:
            results = run_sequential_training(args)
        else:
            results = run_parallel_training(args)
    elif args.models == 'original':
        # 只训练原版模型
        success = run_training('original', args)
        results = {'original': success}
    elif args.models == 'improved':
        # 只训练改进版模型
        success = run_training('improved', args)
        results = {'improved': success}
    
    # 输出最终总结
    print(f"\n🏁 所有训练任务完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 显示模型文件位置
    if results.get('original', False):
        print(f"📁 原版模型文件: {args.output_dir}/{args.model_prefix}_original_best.pth")
    if results.get('improved', False):
        print(f"📁 改进版模型文件: {args.output_dir}/{args.model_prefix}_improved_best.pth")


if __name__ == '__main__':
    main() 