#!/bin/bash

# PointNetLK C3VD训练脚本
# 描述: 用于在C3VD医学点云数据集上训练PointNetLK模型的脚本

set -e  # 遇到错误时退出

# 激活conda环境
echo "[INFO] 激活conda环境: revisited"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate revisited

# 显示帮助信息
show_help() {
    cat << EOF
PointNetLK C3VD训练脚本

用法: $0 [选项]

选项:
    -h, --help              显示此帮助信息
    -d, --dataset-path      C3VD数据集路径 (默认: /mnt/f/Datasets/C3VD_sever_datasets)
    -m, --model-type        模型类型: original|improved (默认: improved)
    -e, --epochs            训练轮数 (默认: 10)
    -b, --batch-size        批次大小 (默认: 8)
    -s, --pairing-strategy  配对策略 (默认: one_to_one)
    -t, --transform-mag     变换幅度 (默认: 0.8)
    -o, --output-dir        输出目录 (默认: c3vd_results)
    -g, --gpu-id            GPU设备ID (默认: 0)
    -p, --num-points        点云点数 (默认: 1024)
    --quick-test            快速测试模式 (1轮训练)
    --voxel-size            体素大小 (默认: 0.05)
    --voxel-after-transf    变换后体素化 (默认行为)
    --voxel-before-transf   变换前体素化 (新功能)
    --scene-split           启用场景划分训练
    --split-ratio           训练集比例 (默认: 0.8)
    --random-seed           随机种子 (默认: 42)
    --test-scenes           指定测试场景 (逗号分隔)

体素化时机说明:
    --voxel-after-transf    先应用变换再体素化 (默认，适合标准训练)
    --voxel-before-transf   先体素化再应用变换 (适合大幅度变换或数据质量差的情况)

配对策略说明:
    one_to_one              一对一配对 (推荐)
    scene_reference         场景参考配对
    source_to_source        源到源配对
    target_to_target        目标到目标配对
    all                     全部配对

示例:
    $0                                          # 使用默认参数训练
    $0 -m original -e 50 -b 4                  # 训练原始版模型50轮
    $0 --quick-test                             # 快速测试
    $0 --voxel-before-transf -t 1.0             # 使用变换前体素化，大变换幅度
    $0 --voxel-after-transf -t 0.5              # 使用变换后体素化，小变换幅度
    $0 --scene-split --split-ratio 0.7
    $0 --scene-split --test-scenes cecum_trial1_seq1,desc_trial2_seq3

EOF
}

# 默认参数
DATASET_PATH="/mnt/f/Datasets/C3VD_sever_datasets"
MODEL_TYPE="improved"
EPOCHS=10
BATCH_SIZE=8
PAIRING_STRATEGY="one_to_one"
TRANSFORM_MAG=0.8
OUTPUT_DIR="c3vd_results"
GPU_ID=0
NUM_POINTS=1024
VOXEL_SIZE=4
VOXEL_AFTER_TRANSF=false  # 默认为变换后体素化
QUICK_TEST=false

# 场景划分参数
SCENE_SPLIT=false
SPLIT_RATIO=0.8
RANDOM_SEED=42
TEST_SCENES=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        -m|--model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -s|--pairing-strategy)
            PAIRING_STRATEGY="$2"
            shift 2
            ;;
        -t|--transform-mag)
            TRANSFORM_MAG="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -g|--gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        -p|--num-points)
            NUM_POINTS="$2"
            shift 2
            ;;
        --voxel-size)
            VOXEL_SIZE="$2"
            shift 2
            ;;
        --voxel-after-transf)
            VOXEL_AFTER_TRANSF=true
            shift
            ;;
        --voxel-before-transf)
            VOXEL_AFTER_TRANSF=false
            shift
            ;;
        --quick-test)
            QUICK_TEST=true
            EPOCHS=1
            BATCH_SIZE=4
            shift
            ;;
        --scene-split)
            SCENE_SPLIT=true
            shift
            ;;
        --split-ratio)
            SPLIT_RATIO="$2"
            echo "设置划分比例: $SPLIT_RATIO"
            shift 2
            ;;
        --random-seed)
            RANDOM_SEED="$2"
            echo "设置随机种子: $RANDOM_SEED"
            shift 2
            ;;
        --test-scenes)
            TEST_SCENES="$2"
            echo "指定测试场景: $TEST_SCENES"
            shift 2
            ;;
        *)
            echo "[ERROR] 未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 验证模型类型
if [[ ! "$MODEL_TYPE" =~ ^(original|improved)$ ]]; then
    echo "[ERROR] 无效的模型类型: $MODEL_TYPE. 必须是 original 或 improved"
    exit 1
fi

# 验证配对策略
if [[ ! "$PAIRING_STRATEGY" =~ ^(one_to_one|scene_reference|source_to_source|target_to_target|all)$ ]]; then
    echo "[ERROR] 无效的配对策略: $PAIRING_STRATEGY"
    exit 1
fi

# 根据体素化时机给出建议
if [[ "$VOXEL_AFTER_TRANSF" == "true" ]]; then
    VOXEL_TIMING_DESC="变换后体素化"
    if (( $(echo "$TRANSFORM_MAG > 0.8" | bc -l) )); then
        echo "[WARNING] 使用变换后体素化时，建议变换幅度不超过0.8"
    fi
else
    VOXEL_TIMING_DESC="变换前体素化"
    if (( $(echo "$TRANSFORM_MAG < 0.8" | bc -l) )); then
        echo "[INFO] 使用变换前体素化时，可以尝试更大的变换幅度(0.8-1.2)"
    fi
fi

# 打印配置信息
echo "[INFO] === PointNetLK C3VD训练配置 ==="
echo "数据集路径: $DATASET_PATH"
echo "模型类型: $MODEL_TYPE"
echo "训练轮数: $EPOCHS"
echo "批次大小: $BATCH_SIZE"
echo "配对策略: $PAIRING_STRATEGY"
echo "变换幅度: $TRANSFORM_MAG"
echo "输出目录: $OUTPUT_DIR"
echo "GPU设备: cuda:$GPU_ID"
echo "点云点数: $NUM_POINTS"
echo "体素大小: $VOXEL_SIZE"
echo "体素化时机: $VOXEL_TIMING_DESC"
echo "快速测试: $QUICK_TEST"
echo "场景划分: $SCENE_SPLIT"
echo "训练集比例: $SPLIT_RATIO"
echo "随机种子: $RANDOM_SEED"
echo "测试场景: $TEST_SCENES"
echo "=================================="

# 环境检查函数
check_environment() {
    echo "[INFO] 检查训练环境..."
    
    # 检查Python环境
    if ! command -v python &> /dev/null; then
        echo "[ERROR] Python未找到，请确保Python已安装"
        exit 1
    fi
    
    # 检查CUDA
    if command -v nvidia-smi &> /dev/null; then
        echo "[INFO] CUDA环境检测成功"
    else
        echo "[WARNING] 未检测到CUDA，将使用CPU训练"
    fi
    
    # 检查PyTorch
    python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')" 2>/dev/null || {
        echo "[ERROR] PyTorch未正确安装"
        exit 1
    }
    
    # 检查bc命令（用于浮点数比较）
    if ! command -v bc &> /dev/null; then
        echo "[WARNING] bc命令未找到，跳过变换幅度建议检查"
    fi
    
    echo "[INFO] 环境检查完成"
}

# 数据集检查函数
check_dataset() {
    echo "[INFO] 检查C3VD数据集..."
    
    if [[ ! -d "$DATASET_PATH" ]]; then
        echo "[ERROR] 数据集路径不存在: $DATASET_PATH"
        exit 1
    fi
    
    # 检查必要的子目录
    local source_dir="$DATASET_PATH/C3VD_ply_source"
    local target_dir="$DATASET_PATH/visible_point_cloud_ply_depth"
    
    if [[ ! -d "$source_dir" ]]; then
        echo "[ERROR] 源点云目录不存在: $source_dir"
        exit 1
    fi
    
    if [[ ! -d "$target_dir" ]]; then
        echo "[ERROR] 目标点云目录不存在: $target_dir"
        exit 1
    fi
    
    # 统计场景数量
    local source_scenes=$(find "$source_dir" -maxdepth 1 -type d | grep -v "^$source_dir$" | wc -l)
    local target_scenes=$(find "$target_dir" -maxdepth 1 -type d | grep -v "^$target_dir$" | wc -l)
    
    echo "[INFO] 源点云场景数: $source_scenes"
    echo "[INFO] 目标点云场景数: $target_scenes"
    echo "[INFO] C3VD数据集检查完成"
}

# 创建输出目录
create_output_dir() {
    echo "[INFO] 创建输出目录..."
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/logs"
    mkdir -p "$OUTPUT_DIR/models"
    echo "[INFO] 输出目录创建完成: $OUTPUT_DIR"
}

# 训练模型
train_model() {
    local output_prefix="$OUTPUT_DIR/${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S)"
    
    echo "[INFO] 开始训练 $MODEL_TYPE 模型..."
    
    # 构建基础训练命令
    local cmd="python train_unified.py \
        --dataset-type c3vd \
        --dataset-path $DATASET_PATH \
        --outfile $output_prefix \
        --model-type $MODEL_TYPE \
        --c3vd-pairing-strategy $PAIRING_STRATEGY \
        --c3vd-transform-mag $TRANSFORM_MAG \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --num-points $NUM_POINTS \
        --device cuda:$GPU_ID \
        --voxel-size $VOXEL_SIZE \
        --learning-rate 0.001 \
        --workers 4 \
        --save-interval 10 \
        --log-interval 10 \
        --eval-interval 5"
    
    # 添加体素化时机参数
    if [[ "$VOXEL_AFTER_TRANSF" == "true" ]]; then
        cmd="$cmd --voxel-after-transf"
    else
        cmd="$cmd --voxel-before-transf"
    fi
    
    # 添加场景划分参数
    if [[ "$SCENE_SPLIT" == "true" ]]; then
        cmd="$cmd --c3vd-scene-split"
        cmd="$cmd --c3vd-split-ratio $SPLIT_RATIO"
        cmd="$cmd --c3vd-random-seed $RANDOM_SEED"
        
        if [[ -n "$TEST_SCENES" ]]; then
            cmd="$cmd --c3vd-test-scenes $TEST_SCENES"
        fi
    fi
    
    echo "[INFO] 执行训练命令..."
    echo "[INFO] 体素化时机: $VOXEL_TIMING_DESC"
    
    # 记录开始时间
    local start_time=$(date +%s)
    
    # 执行训练
    if eval "$cmd"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "[SUCCESS] $MODEL_TYPE 模型训练完成，用时 ${duration}秒"
        
        # 保存训练信息
        echo "模型类型: $MODEL_TYPE" > "$output_prefix.info"
        echo "训练时间: $(date)" >> "$output_prefix.info"
        echo "训练用时: ${duration}秒" >> "$output_prefix.info"
        echo "数据集: $DATASET_PATH" >> "$output_prefix.info"
        echo "配对策略: $PAIRING_STRATEGY" >> "$output_prefix.info"
        echo "变换幅度: $TRANSFORM_MAG" >> "$output_prefix.info"
        echo "体素化时机: $VOXEL_TIMING_DESC" >> "$output_prefix.info"
        
        # 保存模型路径供测试使用
        echo "$output_prefix" > "$OUTPUT_DIR/latest_model_path.txt"
        
        return 0
    else
        echo "[ERROR] $MODEL_TYPE 模型训练失败"
        return 1
    fi
}

# 运行测试
run_test() {
    echo "[INFO] 开始模型测试..."
    
    # 查找最新的模型文件
    if [[ ! -f "$OUTPUT_DIR/latest_model_path.txt" ]]; then
        echo "[WARNING] 未找到模型路径信息，跳过测试"
        return 0
    fi
    
    local model_prefix=$(cat "$OUTPUT_DIR/latest_model_path.txt")
    local model_path="${model_prefix}_best.pth"
    
    if [[ ! -f "$model_path" ]]; then
        echo "[WARNING] 未找到训练好的模型文件: $model_path，跳过测试"
        return 0
    fi
    
    echo "[INFO] 运行模型测试: $MODEL_TYPE"
    echo "[INFO] 测试体素化时机: $VOXEL_TIMING_DESC"
    
    local test_output="$OUTPUT_DIR/test_$(date +%Y%m%d_%H%M%S)"
    
    # 构建测试命令
    local test_cmd="python test_unified.py \
        --test-mode single \
        --dataset-type c3vd \
        --dataset-path $DATASET_PATH \
        --model-path $model_path \
        --outfile $test_output \
        --model-type $MODEL_TYPE \
        --c3vd-pairing-strategy $PAIRING_STRATEGY \
        --c3vd-test-transform-mags 0.2,0.4,0.6,0.8 \
        --batch-size 4 \
        --num-points $NUM_POINTS \
        --device cuda:$GPU_ID \
        --voxel-size $VOXEL_SIZE \
        --workers 4 \
        --save-results \
        --generate-report"
    
    # 添加体素化时机参数
    if [[ "$VOXEL_AFTER_TRANSF" == "true" ]]; then
        test_cmd="$test_cmd --voxel-after-transf"
    else
        test_cmd="$test_cmd --voxel-before-transf"
    fi
    
    eval "$test_cmd"
    
    echo "[SUCCESS] 测试完成"
}

# 生成训练报告
generate_report() {
    echo "[INFO] 生成训练报告..."
    
    local report_file="$OUTPUT_DIR/training_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" << EOF
=== PointNetLK C3VD训练报告 ===
生成时间: $(date)

训练配置:
- 数据集路径: $DATASET_PATH
- 模型类型: $MODEL_TYPE
- 训练轮数: $EPOCHS
- 批次大小: $BATCH_SIZE
- 配对策略: $PAIRING_STRATEGY
- 变换幅度: $TRANSFORM_MAG
- 输出目录: $OUTPUT_DIR
- GPU设备: cuda:$GPU_ID
- 点云点数: $NUM_POINTS
- 体素大小: $VOXEL_SIZE
- 体素化时机: $VOXEL_TIMING_DESC

训练结果:
EOF
    
    # 添加训练日志摘要
    find "$OUTPUT_DIR" -name "*.log" -type f | while read -r log_file; do
        if [[ -f "$log_file" ]]; then
            echo "--- $(basename "$log_file") ---" >> "$report_file"
            tail -10 "$log_file" >> "$report_file"
            echo "" >> "$report_file"
        fi
    done
    
    # 添加模型文件信息
    echo "生成的模型文件:" >> "$report_file"
    find "$OUTPUT_DIR" -name "*.pth" -type f -exec ls -lh {} \; >> "$report_file"
    
    # 添加测试结果
    echo "" >> "$report_file"
    echo "测试结果:" >> "$report_file"
    find "$OUTPUT_DIR" -name "*test*" -type d | while read -r test_dir; do
        if [[ -d "$test_dir" ]]; then
            echo "--- $(basename "$test_dir") ---" >> "$report_file"
            find "$test_dir" -name "*.txt" -type f | head -1 | xargs cat >> "$report_file" 2>/dev/null || echo "无测试结果文件" >> "$report_file"
            echo "" >> "$report_file"
        fi
    done
    
    echo "[SUCCESS] 训练报告已生成: $report_file"
}

# 主函数
main() {
    echo "[INFO] 开始PointNetLK C3VD训练流程..."
    
    # 检查环境
    check_environment
    
    # 检查数据集
    check_dataset
    
    # 创建输出目录
    create_output_dir
    
    # 记录开始时间
    local total_start_time=$(date +%s)
    
    # 训练模型
    train_model
    
    # 运行测试
    run_test
    
    # 生成报告
    generate_report
    
    # 计算总用时
    local total_end_time=$(date +%s)
    local total_duration=$((total_end_time - total_start_time))
    
    echo "[SUCCESS] === 训练流程完成 ==="
    echo "[SUCCESS] 总用时: ${total_duration}秒"
    echo "[SUCCESS] 结果保存在: $OUTPUT_DIR"
    
    # 显示GPU使用情况
    if command -v nvidia-smi &> /dev/null; then
        echo "[INFO] 当前GPU状态:"
        nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
    fi
}

# 信号处理
trap 'echo "[ERROR] 训练被中断"; exit 1' INT TERM

# 执行主函数
main "$@" 