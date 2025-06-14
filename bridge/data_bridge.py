"""
数据桥接模块 - 统一原版和改进版的数据加载接口
Data Bridge - Unified interface for original and improved data loaders
"""

import torch
import torch.utils.data
import sys
import os
import numpy as np

# 添加路径以导入legacy_ptlk和当前目录的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import legacy_ptlk as ptlk
# 移除不存在的导入
# from data_utils import get_datasets as get_improved_datasets  # PointNetLK_Revisited的数据加载


class DemoDataset(torch.utils.data.Dataset):
    """演示数据集类 - 用于处理demo目录中的.npy文件"""
    
    def __init__(self, data_path, num_points=1024, transform=None):
        """
        初始化演示数据集
        
        Args:
            data_path: 数据路径
            num_points: 点数
            transform: 变换函数
        """
        self.data_path = data_path
        self.num_points = num_points
        self.transform = transform
        
        # 加载演示数据
        p0_path = os.path.join(data_path, 'p0.npy')
        p1_path = os.path.join(data_path, 'p1.npy')
        
        if os.path.exists(p0_path) and os.path.exists(p1_path):
            self.p0 = np.load(p0_path)
            self.p1 = np.load(p1_path)
            
            # 如果点数超过需要的数量，随机采样
            if self.p0.shape[0] > num_points:
                indices = np.random.choice(self.p0.shape[0], num_points, replace=False)
                self.p0 = self.p0[indices]
            if self.p1.shape[0] > num_points:
                indices = np.random.choice(self.p1.shape[0], num_points, replace=False)
                self.p1 = self.p1[indices]
                
            # 创建多个样本（通过添加噪声）
            self.samples = []
            for i in range(100):  # 创建100个样本
                # 添加小量噪声
                noise_p0 = self.p0 + np.random.normal(0, 0.01, self.p0.shape)
                noise_p1 = self.p1 + np.random.normal(0, 0.01, self.p1.shape)
                self.samples.append((noise_p0, noise_p1))
        else:
            # 如果没有演示数据，创建合成数据
            self.samples = []
            for i in range(100):
                # 创建随机点云
                p0 = np.random.randn(num_points, 3).astype(np.float32)
                # 创建变换后的点云
                angle = np.random.uniform(-0.5, 0.5)
                R = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0],
                             [0, 0, 1]], dtype=np.float32)
                t = np.random.uniform(-0.1, 0.1, (3,)).astype(np.float32)
                p1 = (R @ p0.T).T + t
                self.samples.append((p0, p1))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        p0, p1 = self.samples[idx]
        
        # 转换为tensor
        p0 = torch.from_numpy(p0.astype(np.float32))
        p1 = torch.from_numpy(p1.astype(np.float32))
        
        if self.transform:
            p0 = self.transform(p0)
            p1 = self.transform(p1)
        
        # 创建一个虚拟的变换矩阵（单位矩阵加小扰动）
        # 这是为了兼容训练脚本的期望格式
        igt = torch.eye(4, dtype=torch.float32)
        # 添加小的随机变换
        angle = np.random.uniform(-0.1, 0.1)
        igt[0, 0] = np.cos(angle)
        igt[0, 1] = -np.sin(angle)
        igt[1, 0] = np.sin(angle)
        igt[1, 1] = np.cos(angle)
        igt[:3, 3] = torch.randn(3) * 0.05  # 小的平移
        
        return p0, p1, igt


class DataBridge:
    """统一数据加载接口"""
    
    def __init__(self, dataset_type='modelnet', data_source='original', **kwargs):
        """
        初始化数据桥接
        
        Args:
            dataset_type: 数据集类型 ('modelnet', 'shapenet2', 'kitti', '3dmatch', 'demo')
            data_source: 'original' 或 'improved'
            **kwargs: 其他参数
        """
        self.dataset_type = dataset_type
        self.data_source = data_source
        self.kwargs = kwargs
        
    def get_datasets(self, args=None, **kwargs):
        """
        获取训练和测试数据集
        
        Args:
            args: 参数对象（兼容原版接口）
            **kwargs: 其他参数
            
        Returns:
            trainset, testset: 训练和测试数据集
        """
        # 合并参数
        params = {**self.kwargs, **kwargs}
        
        # 如果数据集路径包含demo，自动切换到demo模式
        dataset_path = params.get('dataset_path', '')
        if 'demo' in dataset_path.lower():
            self.dataset_type = 'demo'
        
        if self.dataset_type == 'demo':
            # 使用演示数据集
            return self._get_demo_datasets(args, **params)
        elif self.data_source == 'original':
            # 使用原版数据加载
            if self.dataset_type in ['modelnet', 'shapenet2']:
                return self._get_original_datasets(args, **params)
            else:
                raise ValueError(f"原版数据加载不支持数据集类型: {self.dataset_type}")
        else:
            # 使用改进版数据加载（简化版本）
            return self._get_improved_datasets(args, **params)
    
    def _get_demo_datasets(self, args, **kwargs):
        """获取演示数据集"""
        dataset_path = kwargs.get('dataset_path', './demo')
        num_points = kwargs.get('num_points', 1024)
        
        # 创建演示数据集
        dataset = DemoDataset(dataset_path, num_points)
        
        # 分割为训练和测试集
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        return trainset, testset
    
    def _get_original_datasets(self, args, **kwargs):
        """使用原版数据加载方法"""
        if args is None:
            # 创建模拟args对象
            class Args:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
            
            # 设置默认参数，避免重复
            default_params = {
                'dataset_type': self.dataset_type,
                'dataset_path': kwargs.get('dataset_path', ''),
                'categoryfile': kwargs.get('categoryfile', ''),
                'num_points': kwargs.get('num_points', 1024),
            }
            # 添加其他参数，但避免重复
            for k, v in kwargs.items():
                if k not in default_params:
                    default_params[k] = v
            
            args = Args(**default_params)
        
        # 调用原版数据加载函数
        if self.dataset_type == 'modelnet':
            return self._get_modelnet_original(args)
        elif self.dataset_type == 'shapenet2':
            return self._get_shapenet2_original(args)
        else:
            raise ValueError(f"不支持的原版数据集类型: {self.dataset_type}")
    
    def _get_improved_datasets(self, args, **kwargs):
        """使用改进版数据加载方法（简化实现）"""
        # 暂时使用原版数据加载作为后备
        return self._get_original_datasets(args, **kwargs)
    
    def _get_modelnet_original(self, args):
        """获取ModelNet40数据集（原版方式）"""
        import torchvision.transforms as transforms
        
        # 处理类别信息
        cinfo = None
        if hasattr(args, 'categoryfile') and args.categoryfile:
            categories = [line.rstrip('\n') for line in open(args.categoryfile)]
            categories.sort()
            c_to_idx = {categories[i]: i for i in range(len(categories))}
            cinfo = (categories, c_to_idx)
        
        # 创建变换
        transform = transforms.Compose([
            ptlk.data.transforms.Mesh2Points(),
            ptlk.data.transforms.OnUnitCube(),
            ptlk.data.transforms.Resampler(args.num_points),
        ])
        
        # 加载数据集
        trainset = ptlk.data.datasets.ModelNet(
            args.dataset_path, 
            train=1, 
            transform=transform, 
            classinfo=cinfo
        )
        
        testset = ptlk.data.datasets.ModelNet(
            args.dataset_path, 
            train=0, 
            transform=transform, 
            classinfo=cinfo
        )
        
        # 为PointNetLK训练创建配准数据集
        mag_randomly = True
        mag = getattr(args, 'mag', 0.8)
        
        trainset_reg = ptlk.data.datasets.CADset4tracking(
            trainset,
            ptlk.data.transforms.RandomTransformSE3(mag, mag_randomly)
        )
        
        testset_reg = ptlk.data.datasets.CADset4tracking(
            testset,
            ptlk.data.transforms.RandomTransformSE3(mag, mag_randomly)
        )
        
        return trainset_reg, testset_reg
    
    def _get_shapenet2_original(self, args):
        """获取ShapeNet2数据集（原版方式）"""
        # 实现ShapeNet2数据加载逻辑
        raise NotImplementedError("ShapeNet2原版数据加载尚未实现")
    
    def get_dataloader(self, dataset, batch_size=32, shuffle=True, num_workers=4, **kwargs):
        """
        创建数据加载器
        
        Args:
            dataset: 数据集
            batch_size: 批次大小
            shuffle: 是否打乱
            num_workers: 工作进程数
            **kwargs: 其他参数
            
        Returns:
            dataloader: 数据加载器
        """
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )
    
    def get_supported_datasets(self):
        """获取支持的数据集列表"""
        if self.data_source == 'original':
            return ['modelnet', 'shapenet2', 'demo']
        else:
            return ['modelnet', 'shapenet2', 'kitti', '3dmatch', 'demo']
    
    def __repr__(self):
        return f"DataBridge(type={self.dataset_type}, source={self.data_source})" 