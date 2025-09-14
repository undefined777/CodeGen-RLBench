# CodeGen-RLBench

A comprehensive reinforcement learning benchmark for code generation, featuring code parsing, compilation, evaluation, and multiple RL training algorithms. This project is specifically designed for training and evaluating code translation models using Qwen2.5-Coder.

## 🚀 Features

### Core Capabilities
- **Multi-Language Code Parsing**: Support for Python, Java, C++, C, C#, PHP, JavaScript using Tree-sitter
- **Code Compilation & Validation**: Real-time compilation testing with error detection
- **Advanced Code Evaluation**: Integrated CodeBLEU metrics with AST and DFG matching
- **Multiple RL Algorithms**: PPO, GRPO, and RLOO for code generation training
- **Qwen2.5-Coder Integration**: Optimized for Qwen2.5-Coder model training and fine-tuning

### Supported Language Pairs
- Java → C++
- Java → Python  
- C++ → Python
- And more through flexible configuration

## 📁 Project Structure

```
CodeGen-RLBench/
├── code_parser/              # Code parsing modules
│   ├── my-languages.so      # Tree-sitter language library
│   ├── DFG_*.py            # Data Flow Graph generators
│   └── utils.py            # Parser utilities
├── code_prepro/            # Code preprocessing
│   └── lang_processors/    # Language-specific processors
├── compiler/               # Code compilation
│   ├── compilers.py        # Compiler interfaces
│   └── terminal_compiler.py # Terminal compiler
├── codebleu/               # CodeBLEU evaluation (local implementation)
├── baseline_model/         # Pre-trained model checkpoints
├── outputs/                # Training outputs and checkpoints
├── data/                   # Training and evaluation datasets
├── evaluation_results_*/   # Evaluation results
├── rl_trainer.py          # Main RL training framework
├── model.py               # Qwen2.5-Coder model wrapper
├── reward.py              # Reward calculation functions
├── ppo.py                 # PPO algorithm implementation
├── grpo.py                # GRPO algorithm implementation
├── rloo.py                # RLOO algorithm implementation
├── evaluation_script.py   # Model evaluation script
├── run_training.py        # Training launcher
├── run_evaluation.sh      # Evaluation launcher
└── requirements.txt       # Python dependencies
```

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd CodeGen-RLBench
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python -c "from code_parser import DFG_python; print('Installation successful!')"
```

## 🚀 Quick Start

### Training a Code Translation Model

#### Using Command Line
```bash
python rl_trainer.py \
    --source_lang java \
    --target_lang cpp \
    --model_path /path/to/qwen2.5-coder \
    --data_path data/qwen/Java-C++/train.jsonl \
    --output_path outputs/ \
    --rl_algorithm ppo \
    --train_batch_size 16 \
    --learning_rate 1e-5
```

#### Using Configuration File
```bash
python run_training.py --config config.json
```

### Evaluating Models

#### Quick Evaluation
```bash
python quick_eval.py \
    --model_before /path/to/baseline/model \
    --model_after /path/to/finetuned/model \
    --data data/qwen/Java-C++/val.jsonl
```

#### Complete Evaluation
```bash
bash run_evaluation.sh
```

## 🧠 Supported RL Algorithms

### 1. PPO (Proximal Policy Optimization)
- **Best for**: General code generation tasks
- **Features**: Stable training, good sample efficiency
- **Usage**: `--rl_algorithm ppo`

### 2. GRPO (Group Relative Policy Optimization)
- **Best for**: Multi-sample generation tasks
- **Features**: Group-based optimization, better for diverse outputs
- **Usage**: `--rl_algorithm grpo --group_size 4`

### 3. RLOO (Reinforcement Learning with Leave-One-Out)
- **Best for**: High-quality single output generation
- **Features**: Leave-one-out baseline, reduced variance
- **Usage**: `--rl_algorithm rloo --group_size 4`

## 📊 Evaluation Metrics

### Compilation Success Rate
- Tests syntactic correctness of generated code
- Real-time compilation with error detection

### CodeBLEU Score
- **BLEU**: N-gram overlap with reference
- **Weighted BLEU**: Weighted n-gram scoring
- **AST Match**: Abstract Syntax Tree similarity
- **DFG Match**: Data Flow Graph similarity

### Example Output
```
================================================================
📊 Evaluation Results
================================================================
📁 Model: qwen-model-finetuned
📊 Samples: 100

🔨 Compilation Rate: 85.00% (85/100)
📊 CodeBLEU: 0.7234
   • BLEU: 0.6891
   • Weighted BLEU: 0.7123
   • AST Match: 0.7456
   • DFG Match: 0.7567
================================================================
```

## 🔧 Configuration

### Training Configuration
```python
@dataclass
class TrainingConfig:
    # Required
    source_lang: str = "java"
    target_lang: str = "cpp"
    model_path: str = "/path/to/qwen2.5-coder"
    data_path: str = "data/qwen/Java-C++/train.jsonl"
    output_path: str = "outputs/"
    
    # Model settings
    max_source_length: int = 400
    max_target_length: int = 400
    
    # Training settings
    train_batch_size: int = 16
    learning_rate: float = 1e-5
    train_epochs: int = 1000000
    
    # RL settings
    rl_algorithm: str = "ppo"  # "ppo", "grpo", "rloo"
    kl_coef: float = 0.05
    kl_target: float = 0.1
    vf_coef: float = 1e-3
```

### Data Format
The project uses Qwen format JSONL files:

```json
{
    "messages": [
        {
            "role": "system", 
            "content": "You are a helpful assistant for code translation."
        },
        {
            "role": "user", 
            "content": "Translate the following Java code to C++:\n\n```java\npublic class Hello {\n    public static void main(String[] args) {\n        System.out.println(\"Hello World\");\n    }\n}\n```"
        },
        {
            "role": "assistant", 
            "content": "Here's the C++ translation:\n\n```cpp\n#include <iostream>\nusing namespace std;\n\nint main() {\n    cout << \"Hello World\" << endl;\n    return 0;\n}\n```"
        }
    ]
}
```

## 📈 Training Scripts

### A100 Training Script
```bash
bash run_a100_training.sh
```

### Custom Training
```bash
python rl_trainer.py \
    --source_lang java \
    --target_lang cpp \
    --model_path ./models/qwen2.5-coder-7b \
    --data_path ./data/qwen/Java-C++/train.jsonl \
    --output_path ./outputs/java2cpp_ppo \
    --rl_algorithm ppo \
    --train_batch_size 32 \
    --learning_rate 5e-6 \
    --kl_coef 0.1 \
    --train_epochs 1000
```

## 🎯 Usage Examples

### Basic Code Generation
```python
from reward import get_reward
from utils import extract_code_from_qwen_response

# Generate code using your model
generated_code = model.generate(input_text)

# Calculate rewards
rewards, compile_rate, ast_match, dfg_match, errors, errors_ref, nodes, nodes_ref = get_reward(
    code_ids=generated_code_ids,
    code_ref_ids=reference_code_ids,
    gold_ids=gold_standard_ids,
    tokenizer=tokenizer
)
```

### Code Parsing
```python
from code_parser import DFG_python
from tree_sitter import Language, Parser

# Initialize parser
LANGUAGE = Language('code_parser/my-languages.so', 'python')
parser = Parser()
parser.set_language(LANGUAGE)

# Parse code
tree = parser.parse(bytes(code, 'utf-8'))
root_node = tree.root_node
```

### Code Compilation
```python
from compiler.terminal_compiler import TerminalCompiler

# Create compiler
compiler = TerminalCompiler("Python")

# Compile code
error, output, success = compiler.compile_code_string(code_string)
```

## 🔍 Advanced Features

### TensorBoard Integration
- Real-time training monitoring
- Loss curves and reward tracking
- KL divergence monitoring

### Checkpoint Management
- Automatic checkpoint saving
- Model versioning
- Resume training from checkpoints

### Multi-GPU Support
- Distributed training support
- Memory optimization
- Batch processing

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python rl_trainer.py --train_batch_size 4 --minibatch_size 1
   ```

2. **Tree-sitter Library Issues**
   ```bash
   # Rebuild language library
   cd code_parser && python build_languages.py
   ```

3. **Model Loading Errors**
   ```bash
   # Check model path and format
   ls -la /path/to/your/model/
   ```

## 📋 Requirements

- Python 3.8+
- PyTorch 1.9.0+
- Transformers 4.20.0+
- CUDA 11.0+ (for GPU training)
- GCC/Clang (for code compilation)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

Please check the LICENSE file in the project root directory.

## 🙏 Acknowledgments

- Qwen2.5-Coder team for the base model
- Tree-sitter for code parsing
- Hugging Face Transformers for model support
- The open-source community for various utilities

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in `EVALUATION_README.md`
- Review the training logs and error messages

---

*Last updated: 2024*