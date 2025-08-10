#!/bin/bash


set -e  # Exit on error


# 📁 Path configuration - Please modify according to actual situation
MODEL_PATH="/home/cxy/Qwen2.5-Coder/finetuning/sft/checkpoints/qwen0.5b-lr5e-5-wr10-wd0.0-bsz1024-maxlen1280"
DATA_PATH="data"
OUTPUT_PATH="./outputs"
TENSORBOARD_DIR="${OUTPUT_PATH}/tensorboard"

# 🎛️ A100 optimization training parameters
SOURCE_LANG="java"
TARGET_LANG="cpp"
TRAIN_BATCH_SIZE=8        # A100 can support larger batch size
TEST_BATCH_SIZE=1         # Test with larger batch
MAX_SOURCE_LENGTH=700      # Increase sequence length
MAX_TARGET_LENGTH=700
LEARNING_RATE=5e-6        # 降低学习率，避免梯度爆炸
TRAIN_EPOCHS=10       # Large number of training epochs
KL_COEF=0.1               # 增加KL系数，加强参考模型约束
VF_COEF=1e-3              # Value function coefficient
SAVE_EVERY_N_STEPS=100    # Save every 100 training steps
MAX_CHECKPOINTS=20        # A100 has large storage, can retain more checkpoints
MINIBATCH_SIZE=1          # 保持为1，通过梯度累积模拟更大batch
GRADIENT_ACCUMULATION_STEPS=4  # 4步累积 = 有效batch为16/4=4次更新
CRITIC_WARMUP_STEPS=50    # Critic预热步数，让价值网络先稳定

# 🔍 Create output directory
echo "📁 创建输出目录: ${OUTPUT_PATH}"
mkdir -p "${OUTPUT_PATH}"
mkdir -p "${TENSORBOARD_DIR}"

# 📝 Save configuration information
CONFIG_FILE="${OUTPUT_PATH}/training_config.txt"
cat > "${CONFIG_FILE}" << EOF
=============================================================================
A100 training configuration information
=============================================================================
Training start time: $(date)
GPU information: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
CUDA version: $(nvcc --version | grep "release" | awk '{print $5,$6}')
PyTorch version: $(python -c "import torch; print(torch.__version__)")

Model configuration:
- Model path: ${MODEL_PATH}
- Source language: ${SOURCE_LANG}
- Target language: ${TARGET_LANG}

Training parameters:
- Training batch size: ${TRAIN_BATCH_SIZE}
- Test batch size: ${TEST_BATCH_SIZE}
- Maximum source sequence length: ${MAX_SOURCE_LENGTH}
- Maximum target sequence length: ${MAX_TARGET_LENGTH}
- Learning rate: ${LEARNING_RATE}
- Training epochs: ${TRAIN_EPOCHS}
- KL coefficient: ${KL_COEF}
- Value function coefficient: ${VF_COEF}
- Minibatch size: ${MINIBATCH_SIZE}
- Gradient accumulation steps: ${GRADIENT_ACCUMULATION_STEPS}
- Critic warmup steps: ${CRITIC_WARMUP_STEPS}
- Save interval: ${SAVE_EVERY_N_STEPS} training steps
- Maximum checkpoints: ${MAX_CHECKPOINTS}

Output path: ${OUTPUT_PATH}
Tensorboard path: ${TENSORBOARD_DIR}
=============================================================================
EOF

echo "📊 Training configuration information saved to: ${CONFIG_FILE}"
cat "${CONFIG_FILE}"

# 🚀 Check GPU status
echo ""
echo "🔍 GPU status check:"
nvidia-smi

# 🎯 Start training
echo ""
echo "🚀 Start A100 optimization training..."
echo "📈 Tensorboard monitoring: tensorboard --logdir=${TENSORBOARD_DIR} --port=6006"
echo ""

# Use nohup to run in the background, redirect output to log file
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
  --save_every_n_steps ${SAVE_EVERY_N_STEPS} \
  --minibatch_size ${MINIBATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
  --critic_warmup_steps ${CRITIC_WARMUP_STEPS} \
  --max_checkpoints ${MAX_CHECKPOINTS} \
  --use_tensorboard \
  --tensorboard_log_dir "${TENSORBOARD_DIR}" \
  --log_every_n_steps 1 \
  --seed 44