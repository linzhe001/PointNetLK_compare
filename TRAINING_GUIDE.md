# PointNetLK 训练与测试完整指南

## 📖 目录

1. [项目概述](#项目概述)
2. [环境配置](#环境配置)
3. [数据集准备](#数据集准备)
4. [模型架构](#模型架构)
5. [训练指南](#训练指南)
6. [测试指南](#测试指南)
7. [参数详解](#参数详解)
8. [工作流程](#工作流程)
9. [结果分析](#结果分析)
10. [故障排除](#故障排除)

---

## 📋 项目概述

本项目实现了两个版本的PointNetLK点云配准算法：

- **原版PointNetLK** (`legacy_ptlk/`): Lucas-Kanade风格的迭代配准
- **改进PointNetLK** (主目录): 增强的特征提取和配准网络

支持两个主要数据集：
- **C3VD**: 医学内窥镜点云数据集
- **ModelNet40**: 标准3D形状配准基准

---

## 🔧 环境配置

### 基础环境

```bash
# 创建conda环境
conda create -n pointnetlk python=3.8
conda activate pointnetlk

# 安装依赖
pip install torch torchvision torchaudio
pip install numpy scipy matplotlib
pip install tqdm argparse
pip install open3d  # 可选：用于点云可视化
```

### 项目依赖

```bash
# 安装项目依赖
pip install -r requirements.txt
```

---

## 📁 数据集准备

### C3VD数据集

#### 数据集结构
```
C3VD_sever_datasets/
├── C3VD_ply_source/              # 源点云（深度传感器数据）
│   ├── cecum_t1_a/
│   │   ├── 0000_depth_pcd.ply
│   │   ├── 0001_depth_pcd.ply
│   │   └── ...
├── visible_point_cloud_ply_depth/ # 目标点云（可见点云数据）
│   ├── cecum_t1_a/
│   │   ├── frame_0000_visible.ply
│   │   ├── frame_0001_visible.ply
│   │   └── ...
├── C3VD_ref/                     # 参考数据
│   └── coverage_mesh.ply         # 场景参考网格
└── pairing_files/                # 配对文件
    ├── one_to_one_pairing.txt
    ├── scene_reference_pairing.txt
    └── ...
```

#### 配对策略说明

1. **one_to_one** (推荐): 时间同步的一对一配对
2. **scene_reference**: 使用coverage_mesh.ply作为参考
3. **source_to_source**: 深度点云间的数据增强
4. **target_to_target**: 可见点云间的数据增强
5. **all**: 包含所有配对方式

### ModelNet40数据集

#### 数据获取
```bash
# 下载ModelNet40数据集
wget https://modelnet.cs.princeton.edu/ModelNet40.zip
unzip ModelNet40.zip
```

#### 数据预处理
```bash
# 运行数据预处理
python dataset/modelnet40_preprocess.py --data_root ./ModelNet40
```

---

## 🏗️ 模型架构

### 原版PointNetLK (Legacy)

```python
# legacy_ptlk/pointlk.py
class PointLK:
    - 基于Lucas-Kanade的迭代配准
    - PointNet特征提取器
    - SE3李群优化
```

### 改进PointNetLK (Improved)

```python
# model.py
class PointNetLK_improved:
    - 增强的特征提取网络
    - 残差连接
    - 多尺度特征融合
    - 改进的损失函数
```

---

## 🚀 训练指南

### C3VD数据集训练

#### 原版PointNetLK训练

```bash
# 基础训练
python train_unified.py \
    --dataset-type c3vd \
    --dataset-path /path/to/C3VD \
    --outfile ./c3vd_results/basic_model \
    --model-type improved \
    --epochs 100
```

#### 改进PointNetLK训练

```bash
# 高级训练
python train_unified.py \
    --dataset-type c3vd \
    --dataset-path /path/to/C3VD \
    --outfile ./c3vd_results/advanced_model \
    --model-type improved \
    --c3vd-pairing-strategy all \
    --c3vd-transform-mag 0.6 \
    --voxel-grid-size 64 \
    --max-voxel-points 150 \
    --epochs 200 \
    --batch-size 12
```

### ModelNet40数据集训练

#### 原版PointNetLK训练

```bash
# ModelNet40基础训练
python train_modelnet.py \
    --data_root ./ModelNet40 \
    --model_type legacy \
    --category airplane \
    --batch_size 16 \
    --epochs 50

# 多类别训练
python train_modelnet.py \
    --data_root ./ModelNet40 \
    --model_type legacy \
    --category all \
    --batch_size 32 \
    --epochs 100 \
    --max_angle 45 \
    --max_trans 0.5
```

#### 改进PointNetLK训练

```bash
# 改进版本训练
python train_modelnet.py \
    --data_root ./ModelNet40 \
    --model_type improved \
    --category airplane \
    --batch_size 16 \
    --epochs 75 \
    --lr 0.0005 \
    --feature_dim 1024
```

### 统一训练脚本

```bash
# 恢复训练
python train_unified.py \
    --dataset-type c3vd \
    --dataset-path /path/to/C3VD \
    --outfile ./c3vd_results/resumed_model \
    --model-type improved \
    --resume ./c3vd_results/basic_model_epoch_50.pth \
    --start-epoch 50 \
    --epochs 100
```

---

## 🧪 测试指南

### C3VD数据集测试

#### 单模型测试

```bash
# 基础测试
python test_unified.py \
    --dataset-type c3vd \
    --dataset-path /path/to/C3VD \
    --model-path ./c3vd_results/basic_model_best.pth \
    --outfile ./test_results/basic \
    --model-type improved

# 详细测试
python test_unified.py \
    --dataset-type c3vd \
    --dataset-path /path/to/C3VD \
    --model-path ./c3vd_results/advanced_model_best.pth \
    --outfile ./test_results/detailed \
    --model-type improved \
    --c3vd-test-transform-mags "0.2,0.4,0.6,0.8" \
    --save-results \
    --visualize
```

#### 对比测试

```bash
# 运行双模型对比测试
python test_comprehensive.py \
    --c3vd_root /path/to/C3VD \
    --legacy_model ./results/c3vd_legacy/best_model.pth \
    --improved_model ./results/c3vd_improved/best_model.pth \
    --output_dir ./comparison_results
```

### ModelNet40数据集测试

```bash
# ModelNet40测试
python test_modelnet.py \
    --data_root ./ModelNet40 \
    --model_type legacy \
    --model_path ./results/modelnet_legacy/best_model.pth \
    --category airplane \
    --test_unseen
```

### 统一测试脚本

```bash
# 使用统一测试脚本
python test_unified.py \
    --dataset-type c3vd \
    --dataset-path /path/to/data \
    --model-path ./results/model.pth \
    --outfile ./test_results \
    --model-type improved
```

---

## ⚙️ 参数详解

### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_root` | str | 必填 | 数据集根目录路径 |
| `--model_type` | str | `legacy` | 模型类型：`legacy`或`improved` |
| `--batch_size` | int | 4 | 批次大小 |
| `--epochs` | int | 100 | 训练轮数 |
| `--lr` | float | 0.001 | 学习率 |
| `--device` | str | `cuda:0` | 设备：`cuda:0`或`cpu` |
| `--seed` | int | 1234 | 随机种子 |
| `--output_dir` | str | `./results` | 输出目录 |

### C3VD特定参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--c3vd_root` | str | 必填 | C3VD数据集根目录 |
| `--pairing_strategy` | str | `one_to_one` | 配对策略 |
| `--voxel_size` | float | 4.0 | 体素大小(cm) |
| `--num_points` | int | 1024 | 采样点数 |
| `--min_intersection_ratio` | float | 0.3 | 最小交集比例 |
| `--max_intersection_ratio` | float | 0.7 | 最大交集比例 |

### ModelNet40特定参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--category` | str | `airplane` | 物体类别或`all` |
| `--num_points` | int | 1024 | 采样点数 |
| `--max_angle` | float | 45.0 | 最大旋转角度(度) |
| `--max_trans` | float | 0.5 | 最大平移距离 |
| `--noise_level` | float | 0.0 | 噪声水平 |
| `--partial_ratio` | float | 1.0 | 部分点云比例 |

### 训练特定参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--lr_decay` | float | 0.7 | 学习率衰减因子 |
| `--lr_decay_step` | int | 40 | 学习率衰减步长 |
| `--weight_decay` | float | 0.0001 | 权重衰减 |
| `--save_freq` | int | 10 | 模型保存频率 |
| `--log_freq` | int | 100 | 日志输出频率 |
| `--val_freq` | int | 5 | 验证频率 |

### 模型特定参数

#### 原版PointNetLK参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--feature_dim` | int | 1024 | PointNet特征维度 |
| `--delta` | float | 1e-2 | LK算法步长 |
| `--learn_delta` | bool | True | 是否学习步长 |
| `--maxiter` | int | 10 | 最大迭代次数 |
| `--xtol` | float | 1e-7 | 收敛阈值 |

#### 改进PointNetLK参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--feature_dim` | int | 1024 | 特征维度 |
| `--use_residual` | bool | False | 使用残差连接 |
| `--multi_scale` | bool | False | 多尺度特征融合 |
| `--attention` | bool | False | 使用注意力机制 |

### 测试特定参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_path` | str | 必填 | 预训练模型路径 |
| `--save_results` | bool | False | 保存测试结果 |
| `--visualize` | bool | False | 可视化结果 |
| `--test_unseen` | bool | False | 测试未见过的类别 |

---

## 🔄 工作流程

### Workflow 1: C3VD快速验证

```bash
# 1. 快速数据验证
python data_utils.py --c3vd_root /path/to/C3VD --validate

# 2. 单个epoch训练测试
python train_unified.py \
    --dataset-type c3vd \
    --dataset-path /path/to/C3VD \
    --outfile ./test \
    --model-type improved \
    --epochs 1 \
    --batch-size 2 \
    --max-samples 50

# 3. 快速测试
python test_unified.py \
    --dataset-type c3vd \
    --dataset-path /path/to/C3VD \
    --model-path ./test \
    --batch-size 1 \
    --max-samples 20
```

### Workflow 2: C3VD完整训练

```bash
# 1. 数据预处理
python data_utils.py --c3vd_root /path/to/C3VD --preprocess

# 2. 训练原版模型
python train_unified.py \
    --dataset-type c3vd \
    --dataset-path /path/to/C3VD \
    --outfile ./results/c3vd_basic \
    --model-type improved \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 0.001

# 3. 训练改进模型
python train_unified.py \
    --dataset-type c3vd \
    --dataset-path /path/to/C3VD \
    --outfile ./results/c3vd_advanced \
    --model-type improved \
    --c3vd-pairing-strategy all \
    --c3vd-transform-mag 0.6 \
    --voxel-grid-size 64 \
    --max-voxel-points 150 \
    --epochs 200 \
    --batch-size 12

# 4. 对比测试
python test_comprehensive.py \
    --c3vd_root /path/to/C3VD \
    --legacy_model ./results/c3vd_legacy/best_model.pth \
    --improved_model ./results/c3vd_improved/best_model.pth
```

### Workflow 3: ModelNet40基准测试

```bash
# 1. 数据准备
python dataset/modelnet40_preprocess.py --data_root ./ModelNet40

# 2. 单类别训练
python train_modelnet.py \
    --data_root ./ModelNet40 \
    --category airplane \
    --model_type legacy \
    --epochs 50

# 3. 多类别训练
python train_modelnet.py \
    --data_root ./ModelNet40 \
    --category all \
    --model_type improved \
    --epochs 100

# 4. 基准测试
python test_modelnet.py \
    --data_root ./ModelNet40 \
    --model_path ./results/best_model.pth \
    --test_unseen
```

### Workflow 4: 双数据集交叉验证

```bash
# 1. C3VD训练，ModelNet测试
python train_unified.py \
    --dataset-type c3vd \
    --dataset-path /path/to/C3VD \
    --outfile ./results/c3vd_model \
    --model-type improved \
    --epochs 100
python test_modelnet.py \
    --data_root ./ModelNet40 \
    --model_path ./results/c3vd_model_best.pth \
    --cross_domain

# 2. ModelNet训练，C3VD测试
python train_modelnet.py --data_root ./ModelNet40 --epochs 100
python test_unified.py \
    --dataset-type c3vd \
    --dataset-path /path/to/C3VD \
    --model-path ./results/modelnet_model_best.pth \
    --cross_domain
```

### Workflow 5: 超参数搜索

```bash
# 使用网格搜索
python hyperparameter_search.py \
    --dataset c3vd \
    --data_root /path/to/C3VD \
    --search_space configs/search_space.yaml \
    --trials 50
```

---

## 📊 结果分析

### 评估指标

1. **配准误差** (Registration Error)
   - 旋转误差 (Rotation Error)
   - 平移误差 (Translation Error)
   - 总体配准误差 (Overall Registration Error)

2. **成功率** (Success Rate)
   - 基于阈值的成功率
   - 不同精度要求下的成功率

3. **计算效率**
   - 训练时间
   - 推理时间
   - GPU内存使用

### 结果可视化

```bash
# 生成结果报告
python analysis/generate_report.py \
    --result_dir ./results \
    --output_dir ./analysis_results

# 可视化配准结果
python analysis/visualize_registration.py \
    --data_root /path/to/data \
    --model_path ./results/model.pth \
    --sample_ids 1,2,3,4,5
```

---

## 🛠️ 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减小批次大小
   --batch_size 2
   
   # 减少点云数量
   --num_points 512
   ```

2. **数据加载失败**
   ```bash
   # 检查数据路径
   python data_utils.py --c3vd_root /path/to/C3VD --validate
   
   # 检查文件权限
   chmod -R 755 /path/to/data
   ```

3. **训练不收敛**
   ```bash
   # 调整学习率
   --lr 0.0001
   
   # 增加学习率衰减
   --lr_decay 0.5 --lr_decay_step 20
   ```

4. **体素化失效**
   ```bash
   # 调整体素大小
   --voxel_size 2.0  # 对于密集点云
   --voxel_size 8.0  # 对于稀疏点云
   ```

### 调试模式

```bash
# 启用调试模式
python train_unified.py \
    --debug \
    --verbose \
    --save_intermediate \
    --max_samples 10
```

### 日志分析

```bash
# 查看训练日志
tail -f ./results/train.log

# 分析tensorboard日志
tensorboard --logdir ./results/logs
```

---

## 📚 参考文档

- [README_C3VD.md](./README_C3VD.md) - C3VD数据集详细说明
- [c3vd_one_epoch_results.md](./c3vd_one_epoch_results.md) - 单轮训练结果分析
- [README.md](./README.md) - 项目总体介绍

---

## 📧 支持

如果遇到问题，请检查：
1. 环境配置是否正确
2. 数据路径是否有效
3. GPU内存是否充足
4. 参数设置是否合理

更多帮助请参考项目README或提交Issue。 