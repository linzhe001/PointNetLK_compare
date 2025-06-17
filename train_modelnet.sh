#!/bin/bash

# PointNetLK ModelNet训练脚本
# 描述: 用于在ModelNet40数据集上训练PointNetLK模型的简化脚本

set -e  # 遇到错误时退出

# 激活conda环境
echo "[INFO] 激活conda环境: revisited"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate revisited

# 显示帮助信息
show_help() {
    cat << EOF
PointNetLK ModelNet训练脚本

用法: $0 [选项]

选项:
    -h, --help              显示此帮助信息
    -d, --dataset-path      ModelNet40数据集路径 (默认: /mnt/f/Datasets/ModelNet40)
    -m, --model-type        模型类型: original|improved (默认: improved)
    -e, --epochs            训练轮数 (默认: 10)
    -b, --batch-size        批次大小 (默认: 16)
    -c, --category-file     类别文件 (默认: dataset/modelnet40_half1.txt)
    -o, --output-dir        输出目录 (默认: modelnet_results)
    -g, --gpu-id            GPU设备ID (默认: 0)
    -p, --num-points        点云点数 (默认: 1024)
    --quick-test            快速测试模式 (1轮训练)

示例:
    $0                                          # 使用默认参数训练
    $0 -m original -e 50 -b 32                 # 训练原始版模型50轮
    $0 --quick-test                             # 快速测试

EOF
}

# 默认参数
DATASET_PATH="/mnt/f/Datasets/ModelNet40"
MODEL_TYPE="improved"
EPOCHS=2
BATCH_SIZE=16
CATEGORY_FILE="dataset/modelnet40_half1.txt"
OUTPUT_DIR="modelnet_results"
GPU_ID=0
NUM_POINTS=1024
QUICK_TEST=false

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
        -c|--category-file)
            CATEGORY_FILE="$2"
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
        --quick-test)
            QUICK_TEST=true
            EPOCHS=1
            BATCH_SIZE=8
            shift
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

# 打印配置信息
echo "[INFO] === PointNetLK ModelNet训练配置 ==="
echo "数据集路径: $DATASET_PATH"
echo "模型类型: $MODEL_TYPE"
echo "训练轮数: $EPOCHS"
echo "批次大小: $BATCH_SIZE"
echo "类别文件: $CATEGORY_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "GPU设备: cuda:$GPU_ID"
echo "点云点数: $NUM_POINTS"
echo "快速测试: $QUICK_TEST"
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
    
    echo "[INFO] 环境检查完成"
}

# 数据集检查函数
check_dataset() {
    echo "[INFO] 检查数据集..."
    
    if [[ ! -d "$DATASET_PATH" ]]; then
        echo "[ERROR] 数据集路径不存在: $DATASET_PATH"
        exit 1
    fi
    
    # 检查类别文件
    if [[ ! -f "$CATEGORY_FILE" ]]; then
        echo "[ERROR] 类别文件不存在: $CATEGORY_FILE"
        exit 1
    fi
    
    local num_categories=$(wc -l < "$CATEGORY_FILE")
    echo "[INFO] 使用 $num_categories 个类别进行训练"
    echo "[INFO] 数据集检查完成"
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
    
    local cmd="python train_unified.py \
        --model-type $MODEL_TYPE \
        --outfile $output_prefix \
        --dataset-path $DATASET_PATH \
        --dataset-type modelnet \
        --categoryfile $CATEGORY_FILE \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --num-points $NUM_POINTS \
        --device cuda:$GPU_ID"
    
    echo "[INFO] 执行训练命令..."
    
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
        echo "类别数: $(wc -l < "$CATEGORY_FILE")" >> "$output_prefix.info"
        
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
    
    local test_output="$OUTPUT_DIR/test_$(date +%Y%m%d_%H%M%S)"
    
    python test_unified.py \
        --test-mode single \
        --model-type $MODEL_TYPE \
        --model-path "$model_path" \
        --dataset-path "$DATASET_PATH" \
        --dataset-type modelnet \
        --categoryfile "$CATEGORY_FILE" \
        --outfile "$test_output" \
        --generate-report \
        --device cuda:$GPU_ID
    
    echo "[SUCCESS] 测试完成"
}

# 生成训练报告
generate_report() {
    echo "[INFO] 生成训练报告..."
    
    local report_file="$OUTPUT_DIR/training_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" << EOF
=== PointNetLK ModelNet训练报告 ===
生成时间: $(date)

训练配置:
- 数据集路径: $DATASET_PATH
- 模型类型: $MODEL_TYPE
- 训练轮数: $EPOCHS
- 批次大小: $BATCH_SIZE
- 类别文件: $CATEGORY_FILE
- 输出目录: $OUTPUT_DIR
- GPU设备: cuda:$GPU_ID
- 点云点数: $NUM_POINTS

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
    find "$OUTPUT_DIR" -name "*test*.txt" -type f | while read -r test_file; do
        if [[ -f "$test_file" ]]; then
            echo "--- $(basename "$test_file") ---" >> "$report_file"
            cat "$test_file" >> "$report_file"
            echo "" >> "$report_file"
        fi
    done
    
    echo "[SUCCESS] 训练报告已生成: $report_file"
}

# 主函数
main() {
    echo "[INFO] 开始PointNetLK ModelNet训练流程..."
    
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
    
}

# 信号处理
trap 'echo "[ERROR] 训练被中断"; exit 1' INT TERM

# 执行主函数
main "$@" 