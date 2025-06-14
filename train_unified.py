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
        self.train_loader, self.test_loader = self._create_data_loaders()
        
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
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                if len(data) == 3:
                    p0, p1, igt = data
                else:
                    p0, p1 = data
                    igt = torch.eye(4).unsqueeze(0).repeat(p0.size(0), 1, 1)
                
                p0, p1, igt = p0.to(self.device), p1.to(self.device), igt.to(self.device)
                
                try:
                    # 在评估时，对于改进版模型，需要启用梯度计算来计算雅可比
                    if self.args.model_type == 'improved':
                        # 临时启用梯度计算
                        p0.requires_grad_(True)
                        p1.requires_grad_(True)
                        with torch.enable_grad():
                            loss = self.model.compute_loss(p0, p1, igt, maxiter=self.args.max_iter, mode='test')
                    else:
                        loss = self.model.compute_loss(p0, p1, igt, maxiter=self.args.max_iter)
                    
                    total_loss += loss.item()
                    num_batches += 1
                except Exception as e:
                    LOGGER.warning(f"评估批次 {batch_idx} 时出错: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        LOGGER.info(f'评估 Epoch: {epoch}, 平均损失: {avg_loss:.6f}')
        
        return avg_loss
    
    def _parse_batch_data(self, data):
        """解析批次数据（适配不同的数据格式）"""
        # 这里需要根据实际的数据格式进行适配
        # 暂时返回默认格式
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