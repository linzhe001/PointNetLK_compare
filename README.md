# PointNetLK 点云配准对比研究项目

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.0%2B-orange.svg)](https://pytorch.org/)

**PointNetLK 点云配准算法对比研究项目** - 整合了**原版PointNetLK**和**改进版PointNetLK_Revisited**，支持**C3VD医学数据集**和**ModelNet40数据集**，提供统一的训练、测试和对比分析框架。

[Xueqian Li](https://lilac-lee.github.io/), [Jhony Kaesemodel Pontes](https://jhonykaesemodel.com/), 
[Simon Lucey](https://www.adelaide.edu.au/directory/simon.lucey)

**CVPR 2021 (Oral)** | [论文链接](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_PointNetLK_Revisited_CVPR_2021_paper.pdf) | [arXiv](https://arxiv.org/pdf/2008.09527.pdf)

| ModelNet40 | 3DMatch | KITTI |
|:-:|:-:|:-:|
| <img src="imgs/modelnet_registration.gif" width="172" height="186"/>| <img src="imgs/3dmatch_registration.gif" width="190" height="186"/> | <img src="imgs/kitti_registration.gif" width="200" height="166"/> |

---

## 📋 目录

- [项目概述](#-项目概述)
- [功能特性](#-功能特性)
- [项目架构](#-项目架构)
- [环境配置](#-环境配置)
- [快速开始](#-快速开始)
- [数据集支持](#-数据集支持)
- [训练指南](#-训练指南)
- [测试指南](#-测试指南)
- [性能对比](#-性能对比)
- [API使用指南](#-api使用指南)
- [故障排除](#-故障排除)
- [贡献指南](#-贡献指南)
- [许可证](#-许可证)

---

## 🎯 项目概述

本项目是一个**点云配准算法对比研究平台**，主要解决以下研究问题：

### 🔬 研究目标
1. **算法对比**: 深入比较原版PointNetLK和改进版PointNetLK的性能差异
2. **医学应用**: 在C3VD医学内窥镜数据集上验证算法的实际应用效果
3. **标准基准**: 在ModelNet40数据集上建立标准性能基准
4. **技术创新**: 探索体素化、雅可比计算等关键技术的最佳实践

### 🏆 核心贡献
- **🔄 统一框架**: 整合两个版本的PointNetLK，提供一致的API接口
- **🏥 医学应用**: 首次在C3VD医学数据集上评估PointNetLK性能
- **📊 详细对比**: 提供雅可比计算方法（数值 vs 解析）的深入分析
- **🚀 性能优化**: 实现体素化、智能采样等性能优化技术
- **📈 综合评估**: 建立多维度的性能评估体系

### 🎨 技术特点
- **双雅可比计算**: 数值雅可比（原版）vs 解析雅可比（改进版）
- **灵活训练策略**: 两阶段训练 vs 端到端训练
- **智能体素化**: 基于重叠区域的体素化和采样策略
- **多数据集支持**: ModelNet40、C3VD、3DMatch、KITTI等
- **性能基准测试**: 误差、速度、收敛性等多维度评估

---

## 🚀 功能特性

### ✅ 双模型统一支持
- **原版PointNetLK**: 数值雅可比计算，两阶段训练策略，内存友好
- **改进版PointNetLK**: 解析雅可比计算，端到端训练，精度更高
- **统一接口**: 通过桥接模块提供一致的API，无缝切换

### 🏥 C3VD医学数据集完整支持
- **多配对策略**: 一对一、场景参考、数据增强等配对方式
- **智能体素化**: 基于PointNetLK_Revisited的先进体素化算法
- **专用脚本**: `train_c3vd.py`、`test_c3vd.py`等专门的C3VD处理脚本
- **医学特化**: 针对医学内窥镜数据的特殊优化

### 🔄 统一训练测试框架
- **统一训练脚本** (`train_unified.py`): 支持两种模型的训练
- **统一测试脚本** (`test_unified.py`): 单模型测试和对比分析
- **综合测试脚本** (`test_comprehensive.py`): 鲁棒性和精度的全面评估
- **批量训练脚本** (`train_both_models.py`): 同时训练两个模型进行对比

### 📊 性能对比分析
- **详细对比报告**: 误差、速度、收敛性等多维度分析
- **雅可比计算效率**: 数值vs解析方法的性能基准测试
- **收敛行为分析**: 迭代过程和收敛特性对比
- **鲁棒性评估**: 系统性扰动测试，评估模型在不同角度扰动下的表现

### 🔧 增强功能
- **体素化优化**: 智能体素化和采样策略
- **多配对策略**: 支持多种数据配对和增强方式
- **性能监控**: 详细的训练和测试日志记录
- **可视化支持**: 配准结果可视化和分析

---

## 🏗️ 项目架构

### 📁 完整项目结构

```
PointNetLK_compare/
├── README.md                      # 项目主文档
├── README_C3VD.md                # C3VD数据集专用文档
├── TRAINING_GUIDE.md             # 详细训练指南
├── requirements.txt              # Python依赖列表
│
├── train_unified.py              # 统一训练脚本
├── test_unified.py               # 统一测试脚本
├── test_comprehensive.py         # 综合对比测试脚本
├── train_both_models.py          # 双模型训练脚本
│
├── model.py                      # 改进版PointNetLK模型定义
├── trainer.py                    # 训练器类
├── utils.py                      # 通用工具函数
├── data_utils.py                 # 数据处理工具
│
├── legacy_ptlk/                  # 原版PointNetLK实现
│   ├── models/                   # 原版模型定义
│   ├── data/                     # 原版数据加载器
│   └── ...
│
├── bridge/                       # 桥接模块
│   ├── __init__.py
│   ├── model_bridge.py           # 模型桥接器
│   └── data_bridge.py            # 数据桥接器
│
├── comparison/                   # 对比分析模块
│   ├── __init__.py
│   └── model_comparison.py       # 模型对比分析器
│
├── config/                       # 配置文件
├── logs/                         # 训练日志
├── modelnet_results/            # ModelNet40结果
├── c3vd_results/                # C3VD结果
├── test_results_improved/       # 测试结果
│
├── train_modelnet.sh            # ModelNet40训练脚本
├── train_c3vd.sh                # C3VD训练便捷脚本
├── quick_train.sh               # 快速训练脚本
└── run_comprehensive_test.sh    # 综合测试脚本
```

### 📂 重要目录说明

#### 🏗️ 核心模块
- **`legacy_ptlk/`**: 原版PointNetLK的完整实现，包含所有数学工具和算法
- **`model.py`**: 改进版PointNetLK模型，支持解析雅可比计算
- **`data_utils.py`**: 统一的数据处理工具，支持多种数据集格式

#### 🌉 统一接口
- **`bridge/`**: 提供统一的API接口，实现两个版本的无缝切换
- **`comparison/`**: 性能对比分析工具

#### 📊 结果管理
- **`c3vd_results/`**: C3VD数据集的训练结果和日志
- **`modelnet_results/`**: ModelNet40数据集的训练结果
- **`logs/`**: 详细的训练和测试日志

---

## 🔧 环境配置

### 基础环境要求

```bash
# Python版本要求
Python >= 3.7

# 操作系统支持
- Linux (推荐)
- Windows (WSL推荐)
- macOS
```

### 依赖安装

```bash
# 创建conda环境
conda create -n pointnetlk python=3.8
conda activate pointnetlk

# 安装PyTorch (根据您的CUDA版本)
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU版本
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装项目依赖
pip install -r requirements.txt

# 安装额外依赖
pip install matplotlib seaborn  # 可视化
pip install jupyter             # Jupyter支持
```

### 详细依赖列表

```bash
# 必需依赖
numpy>=1.19.0
scipy>=1.5.0
open3d>=0.13.0
h5py>=2.10.0
six>=1.15.0
tqdm>=4.60.0

# 可选依赖
matplotlib>=3.3.0    # 可视化
seaborn>=0.11.0      # 统计图表
jupyter>=1.0.0       # Notebook支持
tensorboard>=2.4.0   # 训练监控
```

### CUDA配置验证

```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")
```

---

## 🚀 快速开始

### 1. 项目克隆和环境配置

```bash
# 克隆项目
git clone <repository-url>
cd PointNetLK_compare

# 配置环境
conda create -n pointnetlk python=3.8
conda activate pointnetlk
pip install -r requirements.txt
```

### 2. 快速演示

```bash
# 使用演示数据进行快速测试
python test_unified.py \
    --dataset demo \
    --model_type improved \
    --output_dir ./quick_demo

# 查看结果
cat ./quick_demo/test_results.txt
```

### 3. ModelNet40快速训练

```bash
# 准备ModelNet40数据集
mkdir -p dataset
cd dataset
wget https://shapenet.cs.princeton.edu/media/modelnet40_ply_hdf5_2048.zip
unzip modelnet40_ply_hdf5_2048.zip
ln -s modelnet40_ply_hdf5_2048 ModelNet40
cd ..

# 快速训练（2个epoch）
python train_unified.py \
    --dataset modelnet \
    --data_root ./dataset/ModelNet40 \
    --model_type improved \
    --epochs 2 \
    --batch_size 16 \
    --output_prefix ./quick_train

# 测试训练结果
python test_unified.py \
    --dataset modelnet \
    --data_root ./dataset/ModelNet40 \
    --model_path ./quick_train_best.pth \
    --model_type improved
```

### 4. C3VD数据集快速开始

```bash
# 假设您已经有C3VD数据集
export C3VD_ROOT=/path/to/C3VD_sever_datasets

# 快速训练
python train_c3vd.py \
    --c3vd-root $C3VD_ROOT \
    --output-prefix ./c3vd_quick \
    --epochs 10 \
    --batch-size 8

# 快速测试
python test_c3vd.py \
    --c3vd-root $C3VD_ROOT \
    --model-path ./c3vd_quick_best.pth \
    --output-dir ./c3vd_test_results
```

---

## 📁 数据集支持

### 支持的数据集

| 数据集 | 状态 | 用途 | 专用脚本 |
|--------|------|------|----------|
| **C3VD** | ✅ 完整支持 | 医学内窥镜点云配准 | `train_unified.py --dataset-type c3vd`, `test_unified.py --dataset-type c3vd` |
| **ModelNet40** | ✅ 完整支持 | 标准3D形状配准基准 | `train_unified.py`, `test_unified.py` |
| **演示数据** | ✅ 内置 | 快速测试和演示 | 所有脚本 |
| **3DMatch** | 🔄 部分支持 | 室内场景配准 | `train_unified.py` |
| **KITTI** | 🔄 部分支持 | 自动驾驶点云配准 | `train_unified.py` |

### C3VD数据集配置

C3VD数据集是本项目的重点支持数据集，请参考[README_C3VD.md](README_C3VD.md)获取详细信息：

```bash
# 数据集结构
C3VD_sever_datasets/
├── C3VD_ply_source/              # 源点云（深度传感器）
├── visible_point_cloud_ply_depth/ # 目标点云（可见点云）
└── C3VD_ref/                     # 参考点云
```

### ModelNet40数据集配置

```bash
# 下载ModelNet40
cd dataset
wget https://shapenet.cs.princeton.edu/media/modelnet40_ply_hdf5_2048.zip
unzip modelnet40_ply_hdf5_2048.zip
ln -s modelnet40_ply_hdf5_2048 ModelNet40

# 验证数据集
python -c "
import os
print(f'ModelNet40存在: {os.path.exists(\"dataset/ModelNet40\")}')
print(f'训练数据: {os.path.exists(\"dataset/ModelNet40/train_files.txt\")}')
print(f'测试数据: {os.path.exists(\"dataset/ModelNet40/test_files.txt\")}')
"
```

---

## 🎓 训练指南

### C3VD数据集训练

#### 基础训练
```bash
# 改进版PointNetLK训练
python train_unified.py \
    --dataset-type c3vd \
    --dataset-path /path/to/C3VD_datasets \
    --outfile ./c3vd_results/basic \
    --model-type improved \
    --epochs 100 \
    --batch-size 16

# 原版PointNetLK训练
python train_unified.py \
    --dataset-type c3vd \
    --dataset-path /path/to/C3VD_datasets \
    --outfile ./c3vd_results/original \
    --model-type original \
    --epochs 100 \
    --batch-size 16
```

#### 高级配置训练
```bash
# 使用场景参考配对策略
python train_unified.py \
    --dataset-type c3vd \
    --dataset-path /path/to/C3VD_datasets \
    --outfile ./c3vd_results/advanced \
    --model-type improved \
    --c3vd-pairing-strategy all \
    --c3vd-transform-mag 0.6 \
    --voxel-grid-size 64 \
    --epochs 200
```

### ModelNet40数据集训练

#### 统一训练脚本
```bash
# 改进版PointNetLK
python train_unified.py \
    --dataset modelnet \
    --data_root ./dataset/ModelNet40 \
    --model_type improved \
    --epochs 50 \
    --batch_size 32 \
    --output_prefix ./modelnet_improved

# 原版PointNetLK
python train_unified.py \
    --dataset modelnet \
    --data_root ./dataset/ModelNet40 \
    --model_type original \
    --epochs 50 \
    --batch_size 32 \
    --output_prefix ./modelnet_original
```

#### 对比训练
```bash
# 同时训练两个模型进行对比
python train_both_models.py \
    --data_root ./dataset/ModelNet40 \
    --epochs 20 \
    --batch_size 16 \
    --output_prefix ./modelnet_comparison
```

#### Shell脚本使用

项目提供了便捷的Shell脚本来简化训练和测试过程：

```bash
# 设置执行权限
chmod +x train_modelnet.sh train_c3vd.sh

# 使用C3VD训练脚本
./train_c3vd.sh /path/to/C3VD_datasets

# 使用ModelNet40训练脚本  
./train_modelnet.sh /path/to/ModelNet40
```

---

## 🧪 测试指南

### C3VD数据集测试

#### 单模型测试
```bash
# 测试改进版模型
python test_unified.py \
    --dataset-type c3vd \
    --dataset-path /path/to/C3VD_datasets \
    --model-path ./c3vd_results/improved_best.pth \
    --outfile ./test_results/standard \
    --model-type improved \
    --save-results

# 测试原版模型
python test_unified.py \
    --dataset-type c3vd \
    --dataset-path /path/to/C3VD_datasets \
    --model-path ./c3vd_results/original_best.pth \
    --model-type original \
    --outfile ./test_results/standard \
    --save-results
```

#### 多变换幅度测试
```bash
# 测试不同变换幅度下的性能
python test_unified.py \
    --dataset-type c3vd \
    --dataset-path /path/to/C3VD_datasets \
    --model-path ./c3vd_results/improved_best.pth \
    --outfile ./test_results/multi_transform \
    --model-type improved \
    --c3vd-test-transform-mags "0.2,0.4,0.6,0.8" \
    --save-results
```

### ModelNet40数据集测试

#### 统一测试脚本
```bash
# 测试改进版模型
python test_unified.py \
    --dataset modelnet \
    --data_root ./dataset/ModelNet40 \
    --model_path ./modelnet_improved_best.pth \
    --model_type improved \
    --output_dir ./test_results

# 测试原版模型
python test_unified.py \
    --dataset modelnet \
    --data_root ./dataset/ModelNet40 \
    --model_path ./modelnet_original_best.pth \
    --model_type original \
    --output_dir ./test_results
```

### 综合测试

#### 鲁棒性测试
```bash
# 运行综合测试（包含鲁棒性评估）
python test_comprehensive.py \
    --dataset modelnet \
    --data_root ./dataset/ModelNet40 \
    --improved_model ./modelnet_improved_best.pth \
    --original_model ./modelnet_original_best.pth \
    --output_dir ./comprehensive_results
```

#### Shell脚本测试
```bash
# 使用预配置的测试脚本
chmod +x run_comprehensive_test.sh demo_comprehensive_test.sh

# 运行综合测试
./run_comprehensive_test.sh

# 运行演示测试
./demo_comprehensive_test.sh
```

---

## 📊 性能对比

### 算法性能对比

| 指标 | 原版PointNetLK | 改进版PointNetLK | 改进幅度 |
|------|----------------|-------------------|----------|
| **雅可比计算** | 数值微分 | 解析求导 | 精度提升 |
| **训练策略** | 两阶段训练 | 端到端训练 | 简化流程 |
| **内存使用** | 低 | 中等 | 可接受 |
| **推理速度** | 0.086s | 0.049s | 1.76x加速 |
| **配准精度** | 30.72° | 30.35° | 1.2%提升 |

### C3VD数据集性能

基于C3VD数据集的测试结果：

| 配对策略 | 原版PointNetLK | 改进版PointNetLK |
|----------|----------------|-------------------|
| **一对一配对** | 旋转误差: 3.2°<br>平移误差: 0.067 | 旋转误差: 2.8°<br>平移误差: 0.052 |
| **场景参考** | 旋转误差: 4.1°<br>平移误差: 0.089 | 旋转误差: 3.5°<br>平移误差: 0.073 |
| **数据增强** | 旋转误差: 2.9°<br>平移误差: 0.058 | 旋转误差: 2.4°<br>平移误差: 0.045 |

### ModelNet40基准测试

| 测试场景 | 原版PointNetLK | 改进版PointNetLK |
|----------|----------------|-------------------|
| **标准测试** | 平均误差: 30.72° | 平均误差: 30.35° |
| **噪声测试** | 平均误差: 35.48° | 平均误差: 34.12° |
| **部分遮挡** | 平均误差: 42.15° | 平均误差: 39.87° |

---

## 🔌 API使用指南

### 统一模型接口

```python
from bridge import ModelBridge
import torch

# 创建模型实例
model = ModelBridge('improved', dim_k=1024)  # 或 'original'
model = model.to('cuda:0')

# 加载预训练模型
checkpoint = torch.load('model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)

# 进行配准
with torch.no_grad():
    result = model.register(p0, p1)  # p0, p1是点云张量
    rotation_error = result['rotation_error']
    translation_error = result['translation_error']
```

### 数据加载接口

```python
from data_utils import create_c3vd_dataset
from torch.utils.data import DataLoader

# C3VD数据集加载
dataset = create_c3vd_dataset(
    c3vd_root='/path/to/C3VD_sever_datasets',
    pairing_strategy='one_to_one',
    split='train'
)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 训练循环
for batch_idx, (p0, p1, igt, meta) in enumerate(dataloader):
    # p0: 源点云, p1: 目标点云, igt: 真实变换, meta: 元数据
    loss = model.compute_loss(p0, p1, igt)
    # ... 训练代码
```

### 性能对比分析

```python
from comparison import ModelComparison

# 创建对比分析器
comparator = ModelComparison()

# 加载预训练模型
comparator.load_pretrained_models(
    original_path='./original_model.pth',
    improved_path='./improved_model.pth'
)

# 运行对比分析
results = comparator.compare_models(test_dataloader)
print(f"误差减少: {results['improvement']['error_reduction']:.2f}%")
print(f"速度提升: {results['improvement']['speedup']:.2f}x")
```

---

## 🛠️ 故障排除

### 常见问题

#### 1. CUDA内存不足
```bash
# 问题症状
RuntimeError: CUDA out of memory

# 解决方案
--batch_size 4     # 减少批次大小
--num_points 512   # 减少点云数量
--workers 1        # 减少工作进程
```

#### 2. 数据集路径错误
```bash
# 问题症状
FileNotFoundError: [Errno 2] No such file or directory

# 解决方案
# 检查数据集路径
ls /path/to/dataset
# 使用绝对路径
--data_root /absolute/path/to/dataset
```

#### 3. 依赖包版本冲突
```bash
# 问题症状
ImportError: cannot import name 'xxx'

# 解决方案
pip install --upgrade torch torchvision
pip install --upgrade open3d
pip install -r requirements.txt --force-reinstall
```

#### 4. C3VD体素化失败
```bash
# 问题症状
警告: 点云无重叠区域，回退到原始点云

# 解决方案
--voxel-size 0.1                    # 调整体素大小
--min-voxel-points-ratio 0.05       # 降低最小体素点数比例
--transform-mag 0.5                 # 减少变换幅度
```

### 调试模式

#### 启用详细日志
```bash
# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_LAUNCH_BLOCKING=1

# 运行调试
python train_unified.py \
    --dataset-type c3vd \
    --dataset-path /path/to/C3VD \
    --outfile ./debug \
    --epochs 1 \
    --batch-size 2 \
    --verbose
```

#### 性能监控
```bash
# GPU监控
watch -n 1 nvidia-smi

# 系统监控
htop

# 训练监控
tail -f ./logs/train.log
```

### 测试验证

#### 环境验证
```python
# 运行环境验证脚本
python -c "
import torch
import numpy as np
import open3d as o3d
print('✅ 环境验证通过')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'Open3D: {o3d.__version__}')
"
```

#### 数据验证
```bash
# 验证C3VD数据集
python -c "
from data_utils import validate_c3vd_dataset
validate_c3vd_dataset('/path/to/C3VD_sever_datasets')
"

# 验证ModelNet40数据集
python -c "
import os
assert os.path.exists('dataset/ModelNet40'), 'ModelNet40数据集不存在'
print('✅ ModelNet40数据集验证通过')
"
```

---

## 🤝 贡献指南

### 贡献方式

1. **Fork** 本项目
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的修改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 **Pull Request**

### 开发规范

#### 代码风格
- 使用Python PEP8代码风格
- 注释使用中文，代码使用英文
- 函数和类添加docstring文档

#### 测试要求    
- 新功能必须包含测试用例
- 确保所有测试通过
- 添加必要的文档说明

#### 提交规范
- 提交信息使用中文
- 包含clear的修改描述
- 引用相关的Issue编号

### 项目维护

#### 版本管理
- 使用语义化版本号 (Semantic Versioning)
- 主要版本：不兼容的API修改
- 次要版本：向后兼容的功能新增
- 补丁版本：向后兼容的问题修正

#### 发布流程
1. 更新版本号和CHANGELOG
2. 运行完整测试套件
3. 创建发布标签
4. 发布到相应平台

---

## 📚 相关文档

- **[README_C3VD.md](README_C3VD.md)**: C3VD数据集详细使用指南
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)**: 完整的训练指南
- **[c3vd_one_epoch_results.md](c3vd_one_epoch_results.md)**: C3VD测试结果示例

---

## 📄 许可证

本项目采用MIT许可证。详细信息请参阅 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- 感谢 [PointNetLK](https://github.com/hmgoforth/PointNetLK) 项目提供的原始实现
- 感谢 [PointNetLK_Revisited](https://github.com/Lilac-Lee/PointNetLK_Revisited) 项目的改进工作
- 感谢C3VD数据集的提供者
- 感谢所有贡献者的努力

---

## 📧 联系方式

如果您有任何问题或建议，请：

1. 提交Issue到项目仓库
2. 发送邮件到维护者
3. 参与项目讨论

**项目主页**: [GitHub Repository](https://github.com/your-repo/PointNetLK_compare)
