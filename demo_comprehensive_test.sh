#!/bin/bash

# 综合测试演示脚本
# Comprehensive Test Demo Script

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  PointNetLK 综合测试演示${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# 检查模型文件
ORIGINAL_MODEL="./modelnet40_results/modelnet40_comparison_original_best.pth"
IMPROVED_MODEL="./modelnet40_results/modelnet40_comparison_improved_best.pth"

if [ ! -f "$ORIGINAL_MODEL" ]; then
    echo -e "${RED}错误: 原版模型文件不存在: $ORIGINAL_MODEL${NC}"
    echo -e "${YELLOW}请先运行训练脚本生成模型文件${NC}"
    exit 1
fi

if [ ! -f "$IMPROVED_MODEL" ]; then
    echo -e "${RED}错误: 改进版模型文件不存在: $IMPROVED_MODEL${NC}"
    echo -e "${YELLOW}请先运行训练脚本生成模型文件${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 找到训练好的模型文件${NC}"
echo "  原版模型: $ORIGINAL_MODEL"
echo "  改进版模型: $IMPROVED_MODEL"
echo ""

# 演示菜单
echo -e "${BLUE}请选择演示类型:${NC}"
echo "1. 快速演示 (少量样本，快速完成)"
echo "2. 标准演示 (中等样本，平衡速度和准确性)"
echo "3. 完整演示 (大量样本，详细分析)"
echo "4. 自定义演示 (手动配置参数)"
echo ""

read -p "请输入选择 (1-4): " choice

case $choice in
    1)
        echo -e "${YELLOW}运行快速演示...${NC}"
        ANGLES="10,30,60"
        SAMPLES=20
        BATCH_SIZE=8
        OUTPUT_DIR="./demo_results/quick_demo"
        ;;
    2)
        echo -e "${YELLOW}运行标准演示...${NC}"
        ANGLES="5,15,30,45,75"
        SAMPLES=50
        BATCH_SIZE=16
        OUTPUT_DIR="./demo_results/standard_demo"
        ;;
    3)
        echo -e "${YELLOW}运行完整演示...${NC}"
        ANGLES="5,10,15,30,45,60,75,90"
        SAMPLES=100
        BATCH_SIZE=32
        OUTPUT_DIR="./demo_results/full_demo"
        ;;
    4)
        echo -e "${YELLOW}自定义演示配置...${NC}"
        read -p "扰动角度 (用逗号分隔，如 10,30,60): " ANGLES
        read -p "每角度样本数 (如 50): " SAMPLES
        read -p "批次大小 (如 16): " BATCH_SIZE
        read -p "输出目录 (如 ./demo_results/custom): " OUTPUT_DIR
        ;;
    *)
        echo -e "${RED}无效选择，退出${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}演示配置:${NC}"
echo "  扰动角度: $ANGLES"
echo "  每角度样本数: $SAMPLES"
echo "  批次大小: $BATCH_SIZE"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 确认开始
read -p "是否开始演示? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}演示已取消${NC}"
    exit 0
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}开始综合测试演示...${NC}"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 运行综合测试
python test_comprehensive.py \
    --model-type both \
    --model-path "$ORIGINAL_MODEL" \
    --original-model-path "$ORIGINAL_MODEL" \
    --improved-model-path "$IMPROVED_MODEL" \
    --dataset-path "/mnt/f/Datasets/ModelNet40" \
    --categoryfile "/mnt/f/Datasets/ModelNet40/modelnet40_shape_names.txt" \
    --output-dir "$OUTPUT_DIR" \
    --perturbation-angles "$ANGLES" \
    --num-samples-per-angle $SAMPLES \
    --batch-size $BATCH_SIZE \
    --max-iter 10 \
    --save-plots \
    --save-detailed-results \
    --device "cuda:0"

# 检查执行结果
if [ $? -eq 0 ]; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  演示完成！${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}总耗时: ${DURATION}秒${NC}"
    echo ""
    
    # 显示结果文件
    echo -e "${BLUE}生成的文件:${NC}"
    if [ -d "$OUTPUT_DIR" ]; then
        ls -la "$OUTPUT_DIR"
        echo ""
        
        # 显示报告摘要
        REPORT_FILE="$OUTPUT_DIR/comprehensive_report.txt"
        if [ -f "$REPORT_FILE" ]; then
            echo -e "${CYAN}========================================${NC}"
            echo -e "${CYAN}  测试报告摘要${NC}"
            echo -e "${CYAN}========================================${NC}"
            
            # 提取关键信息
            echo -e "${YELLOW}鲁棒性测试结果:${NC}"
            grep -A 10 "鲁棒性测试结果" "$REPORT_FILE" | head -15
            echo ""
            
            echo -e "${YELLOW}精度测试结果:${NC}"
            grep -A 10 "精度测试结果" "$REPORT_FILE" | head -15
            echo ""
            
            echo -e "${YELLOW}对比分析:${NC}"
            grep -A 10 "对比分析" "$REPORT_FILE" | head -15
            echo ""
            
            echo -e "${BLUE}完整报告: $REPORT_FILE${NC}"
        fi
        
        # 显示性能曲线
        PLOT_FILE="$OUTPUT_DIR/performance_curves.png"
        if [ -f "$PLOT_FILE" ]; then
            echo -e "${BLUE}性能曲线图: $PLOT_FILE${NC}"
        fi
        
        # 显示CSV结果
        CSV_FILE="$OUTPUT_DIR/robustness_results.csv"
        if [ -f "$CSV_FILE" ]; then
            echo ""
            echo -e "${YELLOW}鲁棒性测试数据预览:${NC}"
            head -10 "$CSV_FILE"
            echo -e "${BLUE}完整数据: $CSV_FILE${NC}"
        fi
    fi
    
    echo ""
    echo -e "${GREEN}演示成功完成！${NC}"
    echo -e "${CYAN}你可以查看 $OUTPUT_DIR 目录中的详细结果${NC}"
    
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  演示失败！${NC}"
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}请检查错误信息${NC}"
    exit 1
fi 