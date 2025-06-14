#!/bin/bash

# PointNetLK ModelNet训练脚本
# 作者: AI Assistant
# 版本: 1.0
# 描述: 用于在ModelNet40数据集上训练PointNetLK模型的完整脚本

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
PointNetLK ModelNet训练脚本

用法: $0 [选项]

选项:
    -h, --help              显示此帮助信息
    -d, --dataset-path      ModelNet40数据集路径 (默认: /mnt/f/Datasets/ModelNet40)
    -m, --model-type        模型类型: original|improved|both (默认: both)
    -e, --epochs            训练轮数 (默认: 10)
    -b, --batch-size        批次大小 (默认: 16)
    -c, --category-file     类别文件 (默认: dataset/modelnet40_half1.txt)
    -o, --output-dir        输出目录 (默认: modelnet_results)
    -g, --gpu-id            GPU设备ID (默认: 0)
    -p, --num-points        点云点数 (默认: 1024)
    -l, --log-interval      日志间隔 (默认: 50)
    --quick-test            快速测试模式 (1轮训练)
    --full-dataset          使用完整数据集 (所有40个类别)
    --sequential            顺序训练 (不并行)
    --skip-test             跳过测试阶段
    --cleanup               训练后清理临时文件

示例:
    $0                                          # 使用默认参数训练
    $0 -m improved -e 50 -b 32                 # 训练改进版模型50轮
    $0 --quick-test                             # 快速测试
    $0 --full-dataset -e 100                   # 使用完整数据集训练100轮

EOF
}

# 默认参数
DATASET_PATH="/mnt/f/Datasets/ModelNet40"
MODEL_TYPE="both"
EPOCHS=10
BATCH_SIZE=16
CATEGORY_FILE="dataset/modelnet40_half1.txt"
OUTPUT_DIR="modelnet_results"
GPU_ID=0
NUM_POINTS=1024
LOG_INTERVAL=50
QUICK_TEST=false
FULL_DATASET=false
SEQUENTIAL=false
SKIP_TEST=false
CLEANUP=false

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
        -l|--log-interval)
            LOG_INTERVAL="$2"
            shift 2
            ;;
        --quick-test)
            QUICK_TEST=true
            EPOCHS=1
            BATCH_SIZE=8
            shift
            ;;
        --full-dataset)
            FULL_DATASET=true
            CATEGORY_FILE="dataset/modelnet40_all.txt"
            shift
            ;;
        --sequential)
            SEQUENTIAL=true
            shift
            ;;
        --skip-test)
            SKIP_TEST=true
            shift
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        *)
            print_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 验证模型类型
if [[ ! "$MODEL_TYPE" =~ ^(original|improved|both)$ ]]; then
    print_error "无效的模型类型: $MODEL_TYPE. 必须是 original, improved, 或 both"
    exit 1
fi

# 打印配置信息
print_info "=== PointNetLK ModelNet训练配置 ==="
echo "数据集路径: $DATASET_PATH"
echo "模型类型: $MODEL_TYPE"
echo "训练轮数: $EPOCHS"
echo "批次大小: $BATCH_SIZE"
echo "类别文件: $CATEGORY_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "GPU设备: cuda:$GPU_ID"
echo "点云点数: $NUM_POINTS"
echo "日志间隔: $LOG_INTERVAL"
echo "快速测试: $QUICK_TEST"
echo "完整数据集: $FULL_DATASET"
echo "顺序训练: $SEQUENTIAL"
echo "跳过测试: $SKIP_TEST"
echo "清理文件: $CLEANUP"
echo "=================================="

# 环境检查函数
check_environment() {
    print_info "检查训练环境..."
    
    # 检查Python环境
    if ! command -v python &> /dev/null; then
        print_error "Python未找到，请确保Python已安装"
        exit 1
    fi
    
    # 检查CUDA
    if command -v nvidia-smi &> /dev/null; then
        print_success "CUDA环境检测成功"
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | head -1
    else
        print_warning "未检测到CUDA，将使用CPU训练"
    fi
    
    # 检查PyTorch
    python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')" 2>/dev/null || {
        print_error "PyTorch未正确安装"
        exit 1
    }
    
    print_success "环境检查完成"
}

# 数据集检查函数
check_dataset() {
    print_info "检查数据集..."
    
    if [[ ! -d "$DATASET_PATH" ]]; then
        print_error "数据集路径不存在: $DATASET_PATH"
        exit 1
    fi
    
    # 检查数据集结构
    local sample_dirs=(airplane bathtub bed)
    for dir in "${sample_dirs[@]}"; do
        if [[ ! -d "$DATASET_PATH/$dir" ]]; then
            print_error "数据集结构不正确，缺少目录: $dir"
            exit 1
        fi
    done
    
    # 统计数据集信息
    local total_classes=$(find "$DATASET_PATH" -maxdepth 1 -type d | grep -v "^$DATASET_PATH$" | wc -l)
    print_success "数据集检查完成，共找到 $total_classes 个类别"
    
    # 检查类别文件
    if [[ ! -f "$CATEGORY_FILE" ]]; then
        print_warning "类别文件不存在: $CATEGORY_FILE"
        if [[ "$FULL_DATASET" == true ]]; then
            print_info "创建完整数据集类别文件..."
            mkdir -p "$(dirname "$CATEGORY_FILE")"
            find "$DATASET_PATH" -maxdepth 1 -type d -exec basename {} \; | grep -v "^ModelNet40$" | sort > "$CATEGORY_FILE"
            print_success "已创建类别文件: $CATEGORY_FILE"
        else
            print_error "请提供有效的类别文件"
            exit 1
        fi
    fi
    
    local num_categories=$(wc -l < "$CATEGORY_FILE")
    print_info "使用 $num_categories 个类别进行训练"
}

# 创建输出目录
create_output_dir() {
    print_info "创建输出目录..."
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/logs"
    mkdir -p "$OUTPUT_DIR/models"
    mkdir -p "$OUTPUT_DIR/reports"
    print_success "输出目录创建完成: $OUTPUT_DIR"
}

# 训练单个模型
train_single_model() {
    local model_type=$1
    local output_prefix="$OUTPUT_DIR/${model_type}_$(date +%Y%m%d_%H%M%S)"
    
    print_info "开始训练 $model_type 模型..."
    
    local cmd="python train_unified.py \
        --model-type $model_type \
        --outfile $output_prefix \
        --dataset-path $DATASET_PATH \
        --dataset-type modelnet \
        --categoryfile $CATEGORY_FILE \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --num-points $NUM_POINTS \
        --device cuda:$GPU_ID \
        --log-interval $LOG_INTERVAL"
    
    print_info "执行命令: $cmd"
    
    # 记录开始时间
    local start_time=$(date +%s)
    
    # 执行训练
    if eval "$cmd"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "$model_type 模型训练完成，用时 ${duration}秒"
        
        # 保存训练信息
        echo "模型类型: $model_type" > "$output_prefix.info"
        echo "训练时间: $(date)" >> "$output_prefix.info"
        echo "训练用时: ${duration}秒" >> "$output_prefix.info"
        echo "数据集: $DATASET_PATH" >> "$output_prefix.info"
        echo "类别数: $(wc -l < "$CATEGORY_FILE")" >> "$output_prefix.info"
        
        return 0
    else
        print_error "$model_type 模型训练失败"
        return 1
    fi
}

# 批量训练两个模型
train_both_models() {
    print_info "开始批量训练两个模型..."
    
    local output_prefix="$OUTPUT_DIR/comparison_$(date +%Y%m%d_%H%M%S)"
    local sequential_flag=""
    
    if [[ "$SEQUENTIAL" == true ]]; then
        sequential_flag="--sequential"
    fi
    
    local cmd="python train_both_models.py \
        --dataset-path $DATASET_PATH \
        --dataset-type modelnet \
        --categoryfile $CATEGORY_FILE \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --num-points $NUM_POINTS \
        --output-dir $OUTPUT_DIR \
        --model-prefix comparison_$(date +%Y%m%d_%H%M%S) \
        --device cuda:$GPU_ID \
        $sequential_flag"
    
    print_info "执行命令: $cmd"
    
    # 记录开始时间
    local start_time=$(date +%s)
    
    # 执行训练
    if eval "$cmd"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "批量训练完成，用时 ${duration}秒"
        return 0
    else
        print_error "批量训练失败"
        return 1
    fi
}

# 运行测试
run_test() {
    if [[ "$SKIP_TEST" == true ]]; then
        print_info "跳过测试阶段"
        return 0
    fi
    
    print_info "开始模型测试..."
    
    # 查找最新的模型文件
    local latest_models=($(find "$OUTPUT_DIR" -name "*_best.pth" -type f -printf '%T@ %p\n' | sort -n | tail -2 | cut -d' ' -f2-))
    
    if [[ ${#latest_models[@]} -eq 0 ]]; then
        print_warning "未找到训练好的模型文件，跳过测试"
        return 0
    fi
    
    if [[ ${#latest_models[@]} -eq 1 ]]; then
        # 单模型测试
        local model_path=${latest_models[0]}
        local model_type="improved"
        if [[ "$model_path" == *"original"* ]]; then
            model_type="original"
        fi
        
        print_info "运行单模型测试: $model_type"
        
        python test_unified.py \
            --test-mode single \
            --model-type $model_type \
            --model-path "$model_path" \
            --dataset-path "$DATASET_PATH" \
            --dataset-type modelnet \
            --categoryfile "$CATEGORY_FILE" \
            --outfile "$OUTPUT_DIR/test_$(date +%Y%m%d_%H%M%S)" \
            --generate-report \
            --device cuda:$GPU_ID
    else
        # 对比测试
        print_info "运行对比测试"
        
        local original_model=""
        local improved_model=""
        
        for model in "${latest_models[@]}"; do
            if [[ "$model" == *"original"* ]]; then
                original_model="$model"
            else
                improved_model="$model"
            fi
        done
        
        if [[ -n "$original_model" && -n "$improved_model" ]]; then
            python test_unified.py \
                --test-mode comparison \
                --original-model-path "$original_model" \
                --improved-model-path "$improved_model" \
                --dataset-path "$DATASET_PATH" \
                --dataset-type modelnet \
                --categoryfile "$CATEGORY_FILE" \
                --outfile "$OUTPUT_DIR/comparison_test_$(date +%Y%m%d_%H%M%S)" \
                --analyze-convergence \
                --benchmark-jacobian \
                --device cuda:$GPU_ID
        fi
    fi
    
    print_success "测试完成"
}

# 生成训练报告
generate_report() {
    print_info "生成训练报告..."
    
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
        echo "--- $(basename "$log_file") ---" >> "$report_file"
        if [[ -f "$log_file" ]]; then
            tail -5 "$log_file" >> "$report_file"
        fi
        echo "" >> "$report_file"
    done
    
    # 添加模型文件信息
    echo "生成的模型文件:" >> "$report_file"
    find "$OUTPUT_DIR" -name "*.pth" -type f -exec ls -lh {} \; >> "$report_file"
    
    print_success "训练报告已生成: $report_file"
}

# 清理临时文件
cleanup_files() {
    if [[ "$CLEANUP" == true ]]; then
        print_info "清理临时文件..."
        
        # 清理__pycache__目录
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        
        # 清理.pyc文件
        find . -name "*.pyc" -delete 2>/dev/null || true
        
        print_success "临时文件清理完成"
    fi
}

# 主函数
main() {
    print_info "开始PointNetLK ModelNet训练流程..."
    
    # 检查环境
    check_environment
    
    # 检查数据集
    check_dataset
    
    # 创建输出目录
    create_output_dir
    
    # 记录开始时间
    local total_start_time=$(date +%s)
    
    # 根据模型类型执行训练
    case "$MODEL_TYPE" in
        "original")
            train_single_model "original"
            ;;
        "improved")
            train_single_model "improved"
            ;;
        "both")
            train_both_models
            ;;
    esac
    
    # 运行测试
    run_test
    
    # 生成报告
    generate_report
    
    # 清理文件
    cleanup_files
    
    # 计算总用时
    local total_end_time=$(date +%s)
    local total_duration=$((total_end_time - total_start_time))
    
    print_success "=== 训练流程完成 ==="
    print_success "总用时: ${total_duration}秒"
    print_success "结果保存在: $OUTPUT_DIR"
    
    # 显示GPU使用情况
    if command -v nvidia-smi &> /dev/null; then
        print_info "当前GPU状态:"
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    fi
}

# 信号处理
trap 'print_error "训练被中断"; exit 1' INT TERM

# 执行主函数
main "$@" 