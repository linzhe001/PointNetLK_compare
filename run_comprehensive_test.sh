#!/bin/bash

# 综合测试运行脚本
# Comprehensive Test Runner Script

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  PointNetLK 综合测试脚本${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 默认参数
DATASET_PATH="/mnt/f/Datasets/ModelNet40"
CATEGORY_FILE="$DATASET_PATH/modelnet40_shape_names.txt"
OUTPUT_DIR="./comprehensive_test_results"
MODEL_TYPE="both"
ORIGINAL_MODEL=""
IMPROVED_MODEL=""

# 检查参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --category-file)
            CATEGORY_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --original-model)
            ORIGINAL_MODEL="$2"
            shift 2
            ;;
        --improved-model)
            IMPROVED_MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --dataset-path PATH     数据集路径 (默认: $DATASET_PATH)"
            echo "  --category-file PATH    类别文件路径 (默认: 自动检测)"
            echo "  --output-dir PATH       输出目录 (默认: $OUTPUT_DIR)"
            echo "  --model-type TYPE       模型类型: original/improved/both (默认: both)"
            echo "  --original-model PATH   原版模型路径"
            echo "  --improved-model PATH   改进版模型路径"
            echo "  -h, --help             显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 --model-type both --original-model model_original.pth --improved-model model_improved.pth"
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            exit 1
            ;;
    esac
done

# 环境检查
echo -e "${YELLOW}检查环境...${NC}"

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo -e "${RED}错误: 未找到Python${NC}"
    exit 1
fi

# 检查PyTorch
if ! python -c "import torch" &> /dev/null; then
    echo -e "${RED}错误: 未找到PyTorch${NC}"
    exit 1
fi

# 检查CUDA
if python -c "import torch; print('CUDA可用:', torch.cuda.is_available())" | grep -q "True"; then
    echo -e "${GREEN}✓ CUDA可用${NC}"
    DEVICE="cuda:0"
else
    echo -e "${YELLOW}⚠ CUDA不可用，使用CPU${NC}"
    DEVICE="cpu"
fi

# 检查数据集
if [ ! -d "$DATASET_PATH" ]; then
    echo -e "${RED}错误: 数据集路径不存在: $DATASET_PATH${NC}"
    exit 1
fi

# 自动检测类别文件
if [ ! -f "$CATEGORY_FILE" ]; then
    echo -e "${YELLOW}类别文件不存在，尝试自动检测...${NC}"
    
    # 常见的类别文件名
    POSSIBLE_FILES=(
        "$DATASET_PATH/modelnet40_shape_names.txt"
        "$DATASET_PATH/shape_names.txt"
        "$DATASET_PATH/categories.txt"
        "$DATASET_PATH/modelnet_id.txt"
    )
    
    for file in "${POSSIBLE_FILES[@]}"; do
        if [ -f "$file" ]; then
            CATEGORY_FILE="$file"
            echo -e "${GREEN}✓ 找到类别文件: $CATEGORY_FILE${NC}"
            break
        fi
    done
    
    if [ ! -f "$CATEGORY_FILE" ]; then
        echo -e "${RED}错误: 未找到类别文件${NC}"
        echo "请手动指定类别文件路径: --category-file PATH"
        exit 1
    fi
fi

# 检查模型文件
if [ "$MODEL_TYPE" = "both" ]; then
    if [ -z "$ORIGINAL_MODEL" ] || [ -z "$IMPROVED_MODEL" ]; then
        echo -e "${RED}错误: 对比测试需要指定两个模型文件${NC}"
        echo "请使用: --original-model PATH --improved-model PATH"
        exit 1
    fi
    
    if [ ! -f "$ORIGINAL_MODEL" ]; then
        echo -e "${RED}错误: 原版模型文件不存在: $ORIGINAL_MODEL${NC}"
        exit 1
    fi
    
    if [ ! -f "$IMPROVED_MODEL" ]; then
        echo -e "${RED}错误: 改进版模型文件不存在: $IMPROVED_MODEL${NC}"
        exit 1
    fi
    
    MODEL_PATH="$ORIGINAL_MODEL"  # 主模型路径
elif [ "$MODEL_TYPE" = "original" ]; then
    if [ -z "$ORIGINAL_MODEL" ]; then
        echo -e "${RED}错误: 请指定原版模型路径: --original-model PATH${NC}"
        exit 1
    fi
    MODEL_PATH="$ORIGINAL_MODEL"
elif [ "$MODEL_TYPE" = "improved" ]; then
    if [ -z "$IMPROVED_MODEL" ]; then
        echo -e "${RED}错误: 请指定改进版模型路径: --improved-model PATH${NC}"
        exit 1
    fi
    MODEL_PATH="$IMPROVED_MODEL"
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}✓ 环境检查完成${NC}"
echo ""

# 显示测试配置
echo -e "${BLUE}测试配置:${NC}"
echo "  数据集路径: $DATASET_PATH"
echo "  类别文件: $CATEGORY_FILE"
echo "  输出目录: $OUTPUT_DIR"
echo "  模型类型: $MODEL_TYPE"
echo "  计算设备: $DEVICE"
if [ "$MODEL_TYPE" = "both" ]; then
    echo "  原版模型: $ORIGINAL_MODEL"
    echo "  改进版模型: $IMPROVED_MODEL"
elif [ "$MODEL_TYPE" = "original" ]; then
    echo "  原版模型: $ORIGINAL_MODEL"
elif [ "$MODEL_TYPE" = "improved" ]; then
    echo "  改进版模型: $IMPROVED_MODEL"
fi
echo ""

# 询问用户确认
read -p "是否开始综合测试? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}测试已取消${NC}"
    exit 0
fi

echo -e "${GREEN}开始综合测试...${NC}"
echo ""

# 构建Python命令
PYTHON_CMD="python test_comprehensive.py"
PYTHON_CMD="$PYTHON_CMD --dataset-path '$DATASET_PATH'"
PYTHON_CMD="$PYTHON_CMD --categoryfile '$CATEGORY_FILE'"
PYTHON_CMD="$PYTHON_CMD --output-dir '$OUTPUT_DIR'"
PYTHON_CMD="$PYTHON_CMD --model-type '$MODEL_TYPE'"
PYTHON_CMD="$PYTHON_CMD --model-path '$MODEL_PATH'"
PYTHON_CMD="$PYTHON_CMD --device '$DEVICE'"

if [ "$MODEL_TYPE" = "both" ]; then
    PYTHON_CMD="$PYTHON_CMD --original-model-path '$ORIGINAL_MODEL'"
    PYTHON_CMD="$PYTHON_CMD --improved-model-path '$IMPROVED_MODEL'"
fi

# 添加其他参数
PYTHON_CMD="$PYTHON_CMD --perturbation-angles '5,10,15,30,45,60,75,90'"
PYTHON_CMD="$PYTHON_CMD --num-samples-per-angle 50"  # 减少样本数以加快测试
PYTHON_CMD="$PYTHON_CMD --batch-size 16"
PYTHON_CMD="$PYTHON_CMD --max-iter 10"
PYTHON_CMD="$PYTHON_CMD --save-plots"
PYTHON_CMD="$PYTHON_CMD --save-detailed-results"

# 记录开始时间
START_TIME=$(date +%s)

echo -e "${BLUE}执行命令:${NC}"
echo "$PYTHON_CMD"
echo ""

# 执行测试
eval $PYTHON_CMD

# 检查执行结果
if [ $? -eq 0 ]; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  综合测试完成！${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}总耗时: ${DURATION}秒${NC}"
    echo -e "${GREEN}结果保存在: $OUTPUT_DIR${NC}"
    echo ""
    
    # 显示生成的文件
    if [ -d "$OUTPUT_DIR" ]; then
        echo -e "${BLUE}生成的文件:${NC}"
        ls -la "$OUTPUT_DIR"
        echo ""
        
        # 如果有报告文件，显示摘要
        REPORT_FILE="$OUTPUT_DIR/comprehensive_report.txt"
        if [ -f "$REPORT_FILE" ]; then
            echo -e "${BLUE}测试报告摘要:${NC}"
            head -n 30 "$REPORT_FILE"
            echo ""
            echo -e "${YELLOW}完整报告请查看: $REPORT_FILE${NC}"
        fi
    fi
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  测试失败！${NC}"
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}请检查错误信息并重试${NC}"
    exit 1
fi 