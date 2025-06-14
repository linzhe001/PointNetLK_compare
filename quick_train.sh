#!/bin/bash

# PointNetLK 快速训练脚本
# 提供常用的训练配置选项

echo "🚀 PointNetLK ModelNet快速训练脚本"
echo "=================================="

# 检查是否在正确目录
if [[ ! -f "train_modelnet.sh" ]]; then
    echo "❌ 错误: 请在PointNetLK_Revisited目录中运行此脚本"
    exit 1
fi

echo "请选择训练模式:"
echo "1) 快速测试 (1轮训练，小批次)"
echo "2) 标准训练 (10轮训练，中等批次)"
echo "3) 完整训练 (50轮训练，大批次)"
echo "4) 对比训练 (训练两个模型进行对比)"
echo "5) 自定义参数"

read -p "请输入选择 (1-5): " choice

case $choice in
    1)
        echo "🔥 启动快速测试模式..."
        ./train_modelnet.sh --quick-test --cleanup
        ;;
    2)
        echo "🔥 启动标准训练模式..."
        ./train_modelnet.sh -e 10 -b 16 -m both --cleanup
        ;;
    3)
        echo "🔥 启动完整训练模式..."
        ./train_modelnet.sh -e 50 -b 32 -m both --full-dataset --cleanup
        ;;
    4)
        echo "🔥 启动对比训练模式..."
        ./train_modelnet.sh -e 20 -b 16 -m both --sequential --cleanup
        ;;
    5)
        echo "📋 自定义参数模式"
        echo "可用参数:"
        echo "  -m: 模型类型 (original/improved/both)"
        echo "  -e: 训练轮数"
        echo "  -b: 批次大小"
        echo "  --full-dataset: 使用完整40类数据集"
        echo "  --quick-test: 快速测试"
        echo "  --cleanup: 训练后清理"
        echo ""
        read -p "请输入自定义参数: " custom_params
        ./train_modelnet.sh $custom_params
        ;;
    *)
        echo "❌ 无效选择，退出"
        exit 1
        ;;
esac

echo ""
echo "✅ 训练完成！查看结果:"
echo "   - 模型文件: modelnet_results/"
echo "   - 训练日志: modelnet_results/logs/"
echo "   - 训练报告: modelnet_results/training_report_*.txt" 