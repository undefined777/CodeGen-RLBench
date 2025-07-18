#!/bin/bash


set -e  # 遇到错误立即退出


# 📁 路径配置 - 请根据实际情况修改
MODEL_PATH="/home/cxy/Qwen2.5-Coder/finetuning/sft/checkpoints/qwen0.5b-lr5e-5-wr10-wd0.0-bsz1024-maxlen1280"
DATA_PATH="data"
OUTPUT_PATH="./outputs"
TENSORBOARD_DIR="${OUTPUT_PATH}/tensorboard"

# 🎛️ A100优化训练参数
SOURCE_LANG="java"
TARGET_LANG="cpp"
TRAIN_BATCH_SIZE=4        # A100可以支持更大的batch size
TEST_BATCH_SIZE=4         # 测试时可以用更大的batch
MAX_SOURCE_LENGTH=400      # 适当增加序列长度
MAX_TARGET_LENGTH=400
LEARNING_RATE=1.5e-5       # 稍微增大学习率配合大batch size
TRAIN_EPOCHS=1000000       # 大量训练轮次
KL_COEF=0.05              # KL散度系数
VF_COEF=1e-3              # 价值函数系数
SAVE_STEPS=5              # 每5个epoch保存一次
MAX_CHECKPOINTS=20        # A100有大存储，可以保留更多检查点

# 🔍 创建输出目录
echo "📁 创建输出目录: ${OUTPUT_PATH}"
mkdir -p "${OUTPUT_PATH}"
mkdir -p "${TENSORBOARD_DIR}"

# 📝 保存配置信息
CONFIG_FILE="${OUTPUT_PATH}/training_config.txt"
cat > "${CONFIG_FILE}" << EOF
=============================================================================
A100 训练配置信息
=============================================================================
训练开始时间: $(date)
GPU信息: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
CUDA版本: $(nvcc --version | grep "release" | awk '{print $5,$6}')
PyTorch版本: $(python -c "import torch; print(torch.__version__)")

模型配置:
- 模型路径: ${MODEL_PATH}
- 源语言: ${SOURCE_LANG}
- 目标语言: ${TARGET_LANG}

训练参数:
- 训练批次大小: ${TRAIN_BATCH_SIZE}
- 测试批次大小: ${TEST_BATCH_SIZE}
- 最大源序列长度: ${MAX_SOURCE_LENGTH}
- 最大目标序列长度: ${MAX_TARGET_LENGTH}
- 学习率: ${LEARNING_RATE}
- 训练轮次: ${TRAIN_EPOCHS}
- KL系数: ${KL_COEF}
- 价值函数系数: ${VF_COEF}
- 保存间隔: ${SAVE_STEPS} epochs
- 最大检查点数: ${MAX_CHECKPOINTS}

输出路径: ${OUTPUT_PATH}
Tensorboard路径: ${TENSORBOARD_DIR}
=============================================================================
EOF

echo "📊 训练配置信息已保存到: ${CONFIG_FILE}"
cat "${CONFIG_FILE}"

# 🚀 检查GPU状态
echo ""
echo "🔍 GPU状态检查:"
nvidia-smi

# 🎯 启动训练
echo ""
echo "🚀 开始A100优化训练..."
echo "📈 Tensorboard监控: tensorboard --logdir=${TENSORBOARD_DIR} --port=6006"
echo ""

# 使用nohup在后台运行，输出重定向到日志文件
python optimized_rl_trainer.py \
  --source_lang "${SOURCE_LANG}" \
  --target_lang "${TARGET_LANG}" \
  --model_path "${MODEL_PATH}" \
  --data_path "${DATA_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --max_source_length ${MAX_SOURCE_LENGTH} \
  --max_target_length ${MAX_TARGET_LENGTH} \
  --train_batch_size ${TRAIN_BATCH_SIZE} \
  --test_batch_size ${TEST_BATCH_SIZE} \
  --train_epochs ${TRAIN_EPOCHS} \
  --learning_rate ${LEARNING_RATE} \
  --kl_coef ${KL_COEF} \
  --vf_coef ${VF_COEF} \
  --save_steps ${SAVE_STEPS} \
  --max_checkpoints ${MAX_CHECKPOINTS} \
  --use_tensorboard \
  --tensorboard_log_dir "${TENSORBOARD_DIR}" \
  --log_every_n_steps 1 \
  --seed 42 \