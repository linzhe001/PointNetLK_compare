"""
统一训练脚本 - 支持原版和改进版PointNetLK的训练
Unified Training Script - Support training for both original and improved PointNetLK
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

# 导入桥接模块
from bridge import ModelBridge, DataBridge

# 设置日志
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='统一PointNetLK训练脚本')
    
    # 模型选择
    parser.add_argument('--model-type', default='improved', choices=['original', 'improved'],
                        help='模型类型: original(原版) 或 improved(改进版)')
    
    # 必需参数
    parser.add_argument('-o', '--outfile', required=True, type=str,
                        help='输出文件前缀')
    parser.add_argument('--dataset-path', required=True, type=str,
                        help='数据集路径')
    
    # 数据集设置
    parser.add_argument('--dataset-type', default='modelnet', 
                        choices=['modelnet', 'shapenet2', 'kitti', '3dmatch', 'c3vd'],
                        help='数据集类型')
    parser.add_argument('--num-points', default=1024, type=int,
                        help='点云中的点数')
    parser.add_argument('--categoryfile', default='', type=str,
                        help='类别文件路径（ModelNet需要）')
    
    # C3VD数据集特定参数
    parser.add_argument('--c3vd-source-root', default='', type=str,
                        help='C3VD源点云根目录路径')
    parser.add_argument('--c3vd-target-root', default='', type=str,
                        help='C3VD目标点云根目录路径（可选）')
    parser.add_argument('--c3vd-source-subdir', default='C3VD_ply_source', type=str,
                        help='C3VD源点云子目录名称')
    parser.add_argument('--c3vd-target-subdir', default='visible_point_cloud_ply_depth', type=str,
                        help='C3VD目标点云子目录名称')
    parser.add_argument('--c3vd-pairing-strategy', default='one_to_one',
                        choices=['one_to_one', 'scene_reference', 'source_to_source', 'target_to_target', 'all'],
                        help='C3VD配对策略')
    parser.add_argument('--c3vd-transform-mag', default=0.8, type=float,
                        help='C3VD Ground Truth变换幅度')
    
    # 体素化参数
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
    parser.add_argument('--voxel-after-transf', action='store_true', default=True,
                        help='是否在变换后进行体素化（默认True）')
    parser.add_argument('--voxel-before-transf', dest='voxel_after_transf', action='store_false',
                        help='在变换前进行体素化（与--voxel-after-transf相反）')
    
    # 模型设置
    parser.add_argument('--dim-k', default=1024, type=int,
                        help='特征向量维度')
    parser.add_argument('--max-iter', default=10, type=int,
                        help='LK算法最大迭代次数')
    parser.add_argument('--delta', default=1.0e-2, type=float,
                        help='数值雅可比步长（仅原版）')
    parser.add_argument('--learn-delta', action='store_true',
                        help='是否学习步长参数（仅原版）')
    
    # 训练设置
    parser.add_argument('--epochs', default=200, type=int,
                        help='训练轮数')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='批次大小')
    parser.add_argument('--learning-rate', default=0.001, type=float,
                        help='学习率')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        help='优化器类型')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='计算设备')
    parser.add_argument('--workers', default=4, type=int,
                        help='数据加载工作进程数')
    
    # 预训练和恢复
    parser.add_argument('--pretrained', default='', type=str,
                        help='预训练模型路径')
    parser.add_argument('--resume', default='', type=str,
                        help='恢复训练检查点路径')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='开始轮数')
    
    # 其他设置
    parser.add_argument('--save-interval', default=10, type=int,
                        help='模型保存间隔')
    parser.add_argument('--log-interval', default=10, type=int,
                        help='日志输出间隔')
    parser.add_argument('--eval-interval', default=5, type=int,
                        help='验证间隔')
    
    return parser.parse_args()


class UnifiedTrainer:
    """统一训练器"""
    
    def __init__(self, args):
        """
        初始化训练器
        
        Args:
            args: 命令行参数
        """
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        self.output_dir = os.path.dirname(args.outfile) if os.path.dirname(args.outfile) else './logs'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志文件
        log_file = os.path.join(self.output_dir, f'train_{args.model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)
        
        LOGGER.info(f"开始训练 {args.model_type} 模型")
        LOGGER.info(f"参数: {vars(args)}")
        
        # 初始化模型
        self.model = self._create_model()
        
        # 初始化数据
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # 初始化优化器
        self.optimizer = self._create_optimizer()
        
        # 训练状态
        self.best_loss = float('inf')
        self.start_epoch = args.start_epoch
        
        # 加载预训练模型或恢复训练
        self._load_checkpoint()
    
    def _create_model(self):
        """创建模型"""
        LOGGER.info(f"创建 {self.args.model_type} 模型...")
        
        model_kwargs = {
            'dim_k': self.args.dim_k,
        }
        
        if self.args.model_type == 'original':
            model_kwargs.update({
                'delta': self.args.delta,
                'learn_delta': self.args.learn_delta,
            })
        
        model = ModelBridge(self.args.model_type, **model_kwargs)
        model.to(self.device)
        
        LOGGER.info(f"模型已创建并移动到 {self.device}")
        return model
    
    def _create_data_loaders(self):
        """创建数据加载器"""
        LOGGER.info(f"创建 {self.args.dataset_type} 数据加载器...")
        
        # C3VD数据集特殊处理
        if self.args.dataset_type == 'c3vd':
            return self._create_c3vd_data_loaders()
        
        # 根据模型类型选择数据源
        data_source = 'original' if self.args.model_type == 'original' else 'improved'
        
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
        
        # 获取数据集
        trainset, testset = data_bridge.get_datasets(**dataset_kwargs)
        
        # 创建数据加载器
        train_loader = data_bridge.get_dataloader(
            trainset, 
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.workers
        )
        
        test_loader = data_bridge.get_dataloader(
            testset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers
        )
        
        LOGGER.info(f"训练集大小: {len(trainset)}, 测试集大小: {len(testset)}")
        return train_loader, test_loader
    
    def _create_c3vd_data_loaders(self):
        """创建C3VD数据加载器"""
        from data_utils import create_c3vd_dataset
        
        LOGGER.info("创建C3VD数据集...")
        
        # 确定数据路径
        if self.args.c3vd_source_root:
            source_root = self.args.c3vd_source_root
        else:
            source_root = os.path.join(self.args.dataset_path, self.args.c3vd_source_subdir)
        
        if self.args.c3vd_target_root:
            target_root = self.args.c3vd_target_root
        else:
            target_root = os.path.join(self.args.dataset_path, self.args.c3vd_target_subdir)
        
        # 验证路径存在
        if not os.path.exists(source_root):
            raise FileNotFoundError(f"C3VD源点云路径不存在: {source_root}")
        if not os.path.exists(target_root):
            raise FileNotFoundError(f"C3VD目标点云路径不存在: {target_root}")
        
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
        
        # 创建训练集
        trainset = create_c3vd_dataset(
            source_root=source_root,
            target_root=target_root,
            pairing_strategy=self.args.c3vd_pairing_strategy,
            mag=self.args.c3vd_transform_mag,
            train=True,
            vis=False,
            voxel_config=voxel_config,
            sampling_config=sampling_config,
            voxel_after_transf=self.args.voxel_after_transf
        )
        
        # 创建测试集（使用较小的变换幅度）
        testset = create_c3vd_dataset(
            source_root=source_root,
            target_root=target_root,
            pairing_strategy='one_to_one',  # 测试时使用简单配对
            mag=self.args.c3vd_transform_mag * 0.5,  # 测试时使用较小变换
            train=False,
            vis=False,
            voxel_config=voxel_config,
            sampling_config=sampling_config,
            voxel_after_transf=self.args.voxel_after_transf
        )
        
        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.workers,
            pin_memory=True,
            drop_last=True
        )
        
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=True,
            drop_last=False
        )
        
        LOGGER.info(f"C3VD训练集大小: {len(trainset)}, 测试集大小: {len(testset)}")
        LOGGER.info(f"源点云路径: {source_root}")
        LOGGER.info(f"目标点云路径: {target_root}")
        LOGGER.info(f"配对策略: {self.args.c3vd_pairing_strategy}, 变换幅度: {self.args.c3vd_transform_mag}")
        LOGGER.info(f"体素化配置: 网格大小={self.args.voxel_grid_size}, 体素大小={self.args.voxel_size}")
        
        return train_loader, test_loader
    
    def _create_optimizer(self):
        """创建优化器"""
        learnable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        if self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(learnable_params, lr=self.args.learning_rate)
        else:
            optimizer = torch.optim.SGD(learnable_params, lr=self.args.learning_rate)
        
        LOGGER.info(f"创建 {self.args.optimizer} 优化器，学习率: {self.args.learning_rate}")
        return optimizer
    
    def _load_checkpoint(self):
        """加载检查点或预训练模型"""
        if self.args.resume and os.path.exists(self.args.resume):
            LOGGER.info(f"恢复训练从: {self.args.resume}")
            checkpoint = torch.load(self.args.resume, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            
        elif self.args.pretrained and os.path.exists(self.args.pretrained):
            LOGGER.info(f"加载预训练模型: {self.args.pretrained}")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=self.device))
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        start_time = time.time()
        
        for batch_idx, data in enumerate(self.train_loader):
            # 解析数据
            if len(data) == 3:
                p0, p1, igt = data
            else:
                # 如果数据格式不同，需要适配
                p0, p1, igt = self._parse_batch_data(data)
            
            p0 = p0.to(self.device)
            p1 = p1.to(self.device)
            igt = igt.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            loss = self.model.compute_loss(p0, p1, igt, maxiter=self.args.max_iter)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_samples += p0.size(0)
            
            # 日志输出
            if batch_idx % self.args.log_interval == 0:
                LOGGER.info(f'训练 Epoch: {epoch} [{batch_idx * len(p0)}/{len(self.train_loader.dataset)} '
                           f'({100. * batch_idx / len(self.train_loader):.0f}%)]\\t'
                           f'Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(self.train_loader)
        epoch_time = time.time() - start_time
        
        LOGGER.info(f'训练 Epoch: {epoch}, 平均损失: {avg_loss:.6f}, 时间: {epoch_time:.2f}s')
        return avg_loss
    
    def evaluate(self, epoch):
        """评估模型"""
        if not hasattr(self, 'val_loader') or self.val_loader is None:
            LOGGER.warning("没有验证数据集，跳过评估")
            return 0.0
            
        self.model.eval()
        val_loss = 0.0
        num_examples = 0
        
        with torch.no_grad():
            for batch_idx, data in enumerate(self.val_loader):
                # 处理不同数据格式
                if isinstance(data, dict):
                    # C3VD数据集格式
                    p0 = data['source'].to(self.device)
                    p1 = data['target'].to(self.device) 
                    igt = data['igt'].to(self.device)
                else:
                    # ModelNet等数据集格式
                    p0, p1, igt = data
                    p0 = p0.to(self.device)
                    p1 = p1.to(self.device)
                    igt = igt.to(self.device)
                
                # 前向传播
                if self.args.model_type == 'original':
                    r, t = self.model(p0, p1)
                    loss = self.model.compute_loss(r, t, igt, maxiter=self.args.max_iter)
                else:
                    loss = self.model.compute_loss(p0, p1, igt, maxiter=self.args.max_iter)
                
                val_loss += loss.item()
                num_examples += p0.size(0)
                
                # 限制验证批次数量，避免过长时间
                if batch_idx >= 10:  # 只验证前10个批次
                    break
        
        avg_val_loss = val_loss / max(num_examples, 1)
        LOGGER.info(f'验证 Epoch: {epoch}, 平均损失: {avg_val_loss:.6f}')
        
        return avg_val_loss
    
    def _parse_batch_data(self, data):
        """解析批次数据（适配不同的数据格式）"""
        # C3VD数据集返回字典格式
        if isinstance(data, dict):
            p0 = data['source']
            p1 = data['target'] 
            igt = data['igt']
            return p0, p1, igt
        
        # 其他数据集的格式
        if isinstance(data, (list, tuple)) and len(data) >= 2:
            p0, p1 = data[0], data[1]
            # 生成随机变换作为真实值（用于训练）
            batch_size = p0.size(0)
            igt = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
            return p0, p1, igt
        else:
            raise ValueError(f"不支持的数据格式: {type(data)}")
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'args': self.args
        }
        
        # 保存最新检查点
        checkpoint_path = f"{self.args.outfile}_last.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = f"{self.args.outfile}_best.pth"
            torch.save(checkpoint, best_path)
            LOGGER.info(f"保存最佳模型到: {best_path}")
        
        # 定期保存
        if epoch % self.args.save_interval == 0:
            epoch_path = f"{self.args.outfile}_epoch_{epoch}.pth"
            torch.save(checkpoint, epoch_path)
    
    def train(self):
        """主训练循环"""
        LOGGER.info("开始训练...")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            if epoch % self.args.eval_interval == 0:
                val_loss = self.evaluate(epoch)
                
                # 检查是否为最佳模型
                is_best = val_loss < self.best_loss
                if is_best:
                    self.best_loss = val_loss
                
                # 保存检查点
                self.save_checkpoint(epoch, val_loss, is_best)
            else:
                # 只保存训练检查点
                self.save_checkpoint(epoch, train_loss)
        
        LOGGER.info(f"训练完成！最佳验证损失: {self.best_loss:.6f}")


def main():
    """主函数"""
    args = parse_arguments()
    
    # 创建训练器
    trainer = UnifiedTrainer(args)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main() 