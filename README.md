# CodeGen-RLBench

这是一个用于代码生成的强化学习基准测试项目，包含了代码解析、编译、评估等完整功能。

## 项目结构

```
CodeGen-RLBench/
├── parser/                 # 代码解析器
│   ├── my-languages.so    # Tree-sitter语言库
│   ├── DFG.py            # 数据流图生成
│   └── utils.py          # 解析工具函数
├── code_prepro/          # 代码预处理
│   └── lang_processors/  # 各种编程语言处理器
├── compiler/             # 代码编译器
│   ├── compilers.py      # 编译器接口
│   └── terminal_compiler.py # 终端编译器
├── codebleu/             # CodeBLEU评估工具（本地实现）
├── reward.py             # 奖励计算函数
├── utils.py              # 通用工具函数
└── requirements.txt      # Python依赖
```

## 安装

1. 克隆仓库：
```bash
git clone <repository-url>
cd CodeGen-RLBench
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

注意：CodeBLEU评估工具已经包含在 `codebleu/` 目录中，无需额外安装。

## 主要功能

### 1. 代码解析
- 支持多种编程语言：Python, Java, C++, C, C#, PHP, JavaScript
- 使用Tree-sitter进行语法解析
- 生成抽象语法树(AST)和数据流图(DFG)

### 2. 代码编译
- 支持多种语言的代码编译
- 提供编译错误检测
- 支持临时文件编译

### 3. 代码评估
- 集成CodeBLEU评估指标
- 支持AST匹配和DFG匹配
- 提供语法正确性检查

### 4. 强化学习奖励
- 基于编译成功率的奖励
- AST和DFG匹配度奖励
- 语法错误惩罚

## 使用方法

### 基本使用

```python
from reward import get_reward
from utils import *

# 计算代码生成奖励
rewards, compile_rate, ast_match, dfg_match, errors, errors_ref, nodes, nodes_ref = get_reward(
    lang='python',
    code_ids=generated_code_ids,
    code_ref_ids=reference_code_ids,
    gold_ids=gold_standard_ids,
    tokenizer=tokenizer
)
```

### 代码解析

```python
from parser import DFG_python
from tree_sitter import Language, Parser

# 初始化解析器
LANGUAGE = Language('parser/my-languages.so', 'python')
parser = Parser()
parser.set_language(LANGUAGE)

# 解析代码
tree = parser.parse(bytes(code, 'utf-8'))
root_node = tree.root_node
```

### 代码编译

```python
from compiler.terminal_compiler import TerminalCompiler

# 创建编译器
compiler = TerminalCompiler("Python")

# 编译代码
error, output, success = compiler.compile_code_string(code_string)
```

## 支持的编程语言

- **Python**: 完整的语法解析和编译支持
- **Java**: 语法解析和编译
- **C++**: 语法解析和编译
- **C**: 语法解析和编译
- **C#**: 语法解析和编译
- **PHP**: 语法解析和编译
- **JavaScript**: 语法解析

## 注意事项

1. 确保系统已安装相应的编程语言编译器（如gcc, javac等）
2. Tree-sitter语言库文件 `my-languages.so` 已包含在项目中
3. CodeBLEU评估工具为本地实现，无需外部依赖

## 许可证

请查看项目根目录的LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request来改进项目。 