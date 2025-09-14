#!/bin/bash

# Model evaluation startup script
# Used to compare the performance of models before and after fine-tuning

# Set parameters
MODEL_BEFORE="/home/cxy/CodeGen-RLBench/baseline_model/checkpoint-200"
MODEL_AFTER="/home/cxy/CodeGen-RLBench/outputs/checkpoints/checkpoint-step-260-java2cpp-grpo"

# Supported data paths for multiple language pairs
# Optional: Java-Python, Java-C++, C++-Python
DATA_PATH="data/qwen/Java-C++/val.jsonl"  # Default Java->Python
OUTPUT_DIR="evaluation_results_$(date +%Y%m%d_%H%M%S)"

echo "🚀 Start model evaluation"
echo "📁 Model before: $MODEL_BEFORE"
echo "📁 Model after: $MODEL_AFTER" 
echo "📊 Data set: $DATA_PATH"

# Automatically infer language pair
if [[ "$DATA_PATH" == *"Java-Python"* ]]; then
    echo "🌐 Translation task: Java → Python"
elif [[ "$DATA_PATH" == *"Java-C++"* ]]; then
    echo "🌐 Translation task: Java → C++"
elif [[ "$DATA_PATH" == *"C++-Python"* ]]; then
    echo "🌐 Translation task: C++ → Python"
else
    echo "🌐 Translation task: Unknown language pair"
fi

echo "📂 Output directory: $OUTPUT_DIR"
echo ""

# Check if files exist
if [ ! -d "$MODEL_BEFORE" ]; then
    echo "❌ Model path does not exist: $MODEL_BEFORE"
    exit 1
fi

if [ ! -d "$MODEL_AFTER" ]; then
    echo "❌ Model path does not exist: $MODEL_AFTER"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Data file does not exist: $DATA_PATH"
    exit 1
fi

echo
    echo "📊 Running complete evaluation..."
    python evaluation_script.py \
        --model_before "$MODEL_BEFORE" \
        --model_after "$MODEL_AFTER" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size 64
    
    echo ""
    echo "✅ Complete evaluation completed!"
    echo "📁 Detailed results saved in: $OUTPUT_DIR"

echo ""
echo "🎉 Evaluation tasks completed!"