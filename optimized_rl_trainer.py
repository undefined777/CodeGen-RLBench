#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的PPO代码生成强化学习训练程序

主要功能：
1. 代码翻译任务的PPO训练 - 专为Qwen2.5-Coder设计
2. 支持多种编程语言对
3. 基于编译成功率和代码结构的奖励计算
4. 自适应KL控制和策略裁剪
5. 详细的训练监控和日志记录

作者：AI Assistant
版本：2.0 - Qwen专用版本
"""

import os
import sys
import torch
import numpy as np
import datetime
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import json

# 🔧 新增：Tensorboard 支持
from torch.utils.tensorboard import SummaryWriter

# 项目内部导入
from code_parser import (DFG_python, DFG_java, DFG_ruby, DFG_go, 
                        DFG_php, DFG_javascript, DFG_csharp)
from code_parser import (tree_to_token_index, tree_to_token_nodes,
                        index_to_code_token, tree_to_variable_index, 
                        detokenize_code)
from tree_sitter import Language, Parser
from reward import remove_special_tokens, tree_sitter_full_compile, get_reward
from torch.utils.data import DataLoader, TensorDataset
from model import respond_to_batch, QwenCoderHeadWithValueModelLocal
from transformers import AutoTokenizer
from ppo import PPOTrainer
from utils import (extract_structure, Example, InputFeatures)
from code_prepro.lang_processors import (py_tokenizer, java_tokenizer, cpp_tokenizer,
                                        c_tokenizer, js_tokenizer, php_tokenizer, cs_tokenizer,
                                        py_detokenizer, java_detokenizer, cpp_detokenizer,
                                        c_detokenizer, js_detokenizer, php_detokenizer, cs_detokenizer)
from compiler.terminal_compiler import TerminalCompiler


def extract_code_from_qwen_response(response: str, target_lang: str = "cpp") -> str:
    """
    从Qwen模型的回复中提取纯代码
    
    Args:
        response: Qwen模型的完整回复
        target_lang: 目标语言，用于匹配代码块
    
    Returns:
        提取的纯代码字符串
    """
    # 语言名称映射，支持不同的变体
    lang_patterns = {
        'cpp': ['cpp', 'c++', 'cxx'],
        'java': ['java'],
        'python': ['python', 'py'],
        'javascript': ['javascript', 'js'],
        'c': ['c'],
        'php': ['php'],
        'c_sharp': ['csharp', 'c#', 'cs']
    }
    
    # 获取目标语言的所有可能模式
    target_patterns = lang_patterns.get(target_lang, [target_lang])
    
    # 尝试匹配代码块
    for pattern in target_patterns:
        # 匹配 ```lang\ncode\n``` 格式，转义特殊字符
        escaped_pattern = re.escape(pattern)
        code_match = re.search(rf'```{escaped_pattern}\s*\n(.*?)\n```', response, re.DOTALL | re.IGNORECASE)
        if code_match:
            return code_match.group(1).strip()
    
    # 如果没找到特定语言的代码块，尝试匹配通用代码块
    code_match = re.search(r'```\s*\n(.*?)\n```', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    # 如果没有代码块，尝试提取"translation:"后的内容
    translation_match = re.search(r'translation:\s*\n\n(.+)', response, re.DOTALL | re.IGNORECASE)
    if translation_match:
        return translation_match.group(1).strip()
    
    # 最后的备选方案：返回去除常见前缀后的内容
    response = response.strip()
    prefixes_to_remove = [
        "Here's the C++ translation:",
        "Here's the Java translation:",
        "Here's the Python translation:",
        "Here's the translation:",
        "Translation:",
        "```",
    ]
    
    for prefix in prefixes_to_remove:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
    
    # 移除末尾的 ```
    if response.endswith("```"):
        response = response[:-3].strip()
    
    return response


def read_qwen_examples(filename: str, args) -> List[Example]:
    """
    从Qwen格式的JSONL文件中读取训练样例
    
    Args:
        filename: JSONL文件路径
        args: 包含语言配置的参数对象
    
    Returns:
        Example对象列表
    """
    examples = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line)
                messages = data.get('messages', [])
                
                # 查找 system / user / assistant 消息
                system_message = None
                user_message = None
                assistant_message = None
                
                for message in messages:
                    role = message.get('role')
                    if role == 'system':
                        system_message = message.get('content', '')
                    elif role == 'user':
                        user_message = message.get('content', '')
                    elif role == 'assistant':
                        assistant_message = message.get('content', '')
                
                if not user_message or not assistant_message:
                    continue
                
                # 从user消息中提取源代码 - 用于构建Example.source
                source_code = extract_code_from_qwen_response(user_message, args.source_lang)
                
                # 从assistant消息中提取目标代码 - 用于构建Example.target
                target_code = extract_code_from_qwen_response(assistant_message, args.target_lang)
                
                if not source_code or not target_code:
                    continue
                
                e = Example(
                    idx=idx,
                    source=source_code,
                    target=target_code,
                    source_orig=user_message,      # 先存 user；system 单独挂
                    target_orig=assistant_message
                )
                # 动态挂载 system（若无则空串）
                setattr(e, "system_orig", system_message or "")
                examples.append(e)

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"跳过第{idx+1}行，解析错误: {e}")
                continue
    
    return examples


def convert_qwen_examples_to_features(examples, tokenizer, args, stage=None):
    """
    将Qwen样例转换为模型输入特征
    专门处理对话格式的tokenization
    """
    features = []
    for example_index, example in enumerate(examples):
        # 对于Qwen，我们使用完整的对话消息
        # source_orig包含完整的user prompt
        # target_orig包含完整的assistant回复
        
        # 可以使用tokenizer的chat template，或者简单拼接
        if hasattr(tokenizer, 'apply_chat_template'):
            # 尝试使用chat template
            try:
                if hasattr(example, "system_orig") and example.system_orig:
                    # 使用样本自带 system
                    messages = [
                        {"role": "system", "content": example.system_orig},
                        {"role": "user", "content": example.source_orig},
                    ]
                else:
                    messages = [
                        {"role": "user", "content": example.source_orig},
                    ]
                source_text = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                ) 
            except:
                # 如果失败，使用原始内容
                source_text = example.source_orig
        else:
            source_text = example.source_orig
            
        # tokenize source - 直接编码，不需要特殊token
        source_ids = tokenizer.encode(source_text, max_length=args.max_source_length, 
                                     truncation=True, add_special_tokens=True)
        source_mask = [1] * len(source_ids)
        padding_length = args.max_source_length - len(source_ids)
        source_ids = [tokenizer.pad_token_id] * padding_length + source_ids  # ✅ left-padding
        source_mask = [0] * padding_length + source_mask  # ✅ left-padding
        
        # tokenize target
        if stage == "test":
            target_text = "None"
        else:
            target_text = example.target_orig
            
        target_ids = tokenizer.encode(target_text, max_length=args.max_target_length,
                                     truncation=True, add_special_tokens=True)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids = [tokenizer.pad_token_id] * padding_length + target_ids  # ✅ left-padding
        target_mask = [0] * padding_length + target_mask  # ✅ left-padding
        
        features.append(InputFeatures(
            example_index,
            source_ids,
            target_ids,
            source_mask,
            target_mask,
            example.target_orig))  # 保存完整回复用于后续处理
            
    return features


def create_reward_wrapper(original_get_reward):
    """
    Wrap the original `get_reward()` so that *each* of (policy, ref, gold)
    is decoded up to **its own** EOS, code-block extracted, re-tokenized,
    EOS-appended, and padded to a common length *before* reward computation.
    这样避免将 policy 的 eos 位置误用于 ref/gold（原实现的问题）。
    """
    def get_reward_with_extraction(lang, code_ids=None, code_ref_ids=None, gold_ids=None, tokenizer=None):
        # ---------- helpers ----------
        def _decode_rows(t: torch.Tensor):
            """
            Return (texts, eos_pos_list, max_seq_len) for given token ids tensor.
            """
            arr = t.detach().cpu().numpy()
            max_len = arr.shape[1]
            texts, eos_pos_list = [], []
            eos_id = tokenizer.eos_token_id
            for row in arr:
                # find EOS; if none, use max_len
                eos_pos = int((row == eos_id).argmax()) if eos_id in row else max_len
                eos_pos_list.append(eos_pos)
                texts.append(
                    tokenizer.decode(
                        row[:eos_pos],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                )
            return texts, eos_pos_list, max_len

        # ---------- decode raw responses ----------
        raw_responses, eos_resp, max_resp = _decode_rows(code_ids)
        raw_responses_ref, eos_ref, max_ref = _decode_rows(code_ref_ids)
        raw_gold, eos_gold, max_gold = _decode_rows(gold_ids)

        # ---------- extract code blocks ----------
        extracted_codes = [extract_code_from_qwen_response(txt, lang) for txt in raw_responses]
        extracted_codes_ref = [extract_code_from_qwen_response(txt, lang) for txt in raw_responses_ref]
        extracted_codes_gold = [extract_code_from_qwen_response(txt, lang) for txt in raw_gold]

        # ---------- re-tokenize & append EOS ----------
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        triplets = []
        for c, r, g in zip(extracted_codes, extracted_codes_ref, extracted_codes_gold):
            toks_c = tokenizer.encode(c, add_special_tokens=False) + [eos_id]
            toks_r = tokenizer.encode(r, add_special_tokens=False) + [eos_id]
            toks_g = tokenizer.encode(g, add_special_tokens=False) + [eos_id]
            triplets.append((toks_c, toks_r, toks_g))

        # 统一长度（不超过原 policy 输出长度上限，以节约显存）
        # 你也可以用全局 max(len)；这里采用 min(global_max, policy_original_max)
        global_max = max(len(x) for tri in triplets for x in tri) if triplets else 1
        max_len = max_resp

        def _pad(seq):
            if len(seq) >= max_len:
                return seq[:max_len]
            return seq + [pad_id] * (max_len - len(seq))

        policy_padded = [_pad(x[0]) for x in triplets]
        ref_padded    = [_pad(x[1]) for x in triplets]
        gold_padded   = [_pad(x[2]) for x in triplets]

        # ---------- to tensors ----------
        code_ids_tensor     = torch.tensor(policy_padded, dtype=torch.long, device=code_ids.device)
        code_ref_ids_tensor = torch.tensor(ref_padded,    dtype=torch.long, device=code_ref_ids.device)
        gold_ids_tensor     = torch.tensor(gold_padded,   dtype=torch.long, device=gold_ids.device)

        # ---------- call original reward ----------
        return original_get_reward(
            lang=lang,
            code_ids=code_ids_tensor,
            code_ref_ids=code_ref_ids_tensor,
            gold_ids=gold_ids_tensor,
            tokenizer=tokenizer,
        )
    
    return get_reward_with_extraction


@dataclass
class TrainingConfig:
    """训练配置数据类 - Qwen专用版本"""
    # 语言配置
    source_lang: str
    target_lang: str
    
    # 别名，用于兼容旧代码
    @property
    def l1(self):
        return self.source_lang
    
    @property
    def l2(self):
        return self.target_lang
    
    # 模型配置
    model_path: str
    max_source_length: int = 400
    max_target_length: int = 400
    
    # 训练配置
    train_batch_size: int = 16
    test_batch_size: int = 48
    train_epochs: int = 1000000
    learning_rate: float = 1e-5
    kl_coef: float = 0.05
    kl_target: float = 1.0
    vf_coef: float = 1e-3
    
    # 生成配置
    action_space: int = 2  # top_k
    num_syn_samples: int = 5
    
    # 路径配置
    data_path: str = None
    output_path: str = None
    baseline_output_path: str = None
    
    # 🔧 新增：检查点保存控制
    save_steps: int = 1  # 每N轮保存一次检查点，默认每轮都保存
    max_checkpoints: int = 10  # 最多保留N个检查点，0表示不限制
    
    # 🔧 新增：Tensorboard 支持
    use_tensorboard: bool = True  # 是否启用Tensorboard日志
    tensorboard_log_dir: str = None  # Tensorboard日志目录，None表示使用默认路径
    log_every_n_steps: int = 1  # 每N个训练步骤记录一次指标
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 运行配置
    run_id: int = 1
    seed: int = 42


class CodeTranslationTrainer:
    """代码翻译PPO训练器 - Qwen专用版本"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_logging()
        self.setup_device()
        self.setup_language_mappings()
        self.setup_parsers()
        self.setup_models()
        self.setup_data_loaders()
        self.setup_ppo_trainer()
        self.setup_training_stats()
        
        # 创建奖励函数包装器
        self.get_reward_func = create_reward_wrapper(get_reward)
        
    def setup_logging(self):
        """设置日志系统"""
        log_dir = Path(self.config.output_path) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"training_{self.config.run_id}.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"训练配置: {self.config}")
        
    def setup_device(self):
        """设置计算设备"""
        if self.config.device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA不可用，切换到CPU")
            self.config.device = "cpu"
        
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            
        self.logger.info(f"使用设备: {self.config.device}")
        
    def setup_language_mappings(self):
        """设置语言映射"""
        self.dir_dict = {
            'javascript': 'Javascript', 'java': 'Java', 'c_sharp': 'C#', 
            'php': 'PHP', 'python': 'Python', 'c': 'C', 'cpp': 'C++'
        }
        
    def setup_parsers(self):
        """设置代码解析器"""
        self.dfg_function = {
            'python': DFG_python, 'java': DFG_java, 'php': DFG_php,
            'javascript': DFG_javascript, 'c_sharp': DFG_csharp,
            'c': DFG_csharp, 'cpp': DFG_csharp,
        }
        
        self.parsers = {}
        for lang in self.dfg_function:
            try:
                LANGUAGE = Language('code_parser/my-languages.so', lang)
                parser = Parser()
                parser.set_language(LANGUAGE)
                parser = [parser, self.dfg_function[lang]]
                self.parsers[lang] = parser
            except Exception as e:
                self.logger.warning(f"无法加载{lang}解析器: {e}")
                
    def setup_models(self):
        """设置模型和分词器"""
        # 获取模型文件所在目录
        self.model_dir = Path(self.config.model_path)
        
        # 检查并准备tokenizer和配置文件
        self._check_model_files()
        
        print(f"正在加载模型到设备: {self.config.device}")
        print(f"加载模型文件: {self.config.model_path}")
        
        # 直接加载微调好的完整模型（包含架构与权重）——不要再分 config/weight 两步。:contentReference[oaicite:4]{index=4}
        self.model = QwenCoderHeadWithValueModelLocal(
            self.config.model_path,
            torch_dtype=None,              # 保持默认dtype; 下行统一 .to()
            device=self.config.device,
        )
        self.model.to(self.config.device)
        self.model.train() 
        
        # 加载参考模型（固定不变）
        self.model_ref = QwenCoderHeadWithValueModelLocal(
            self.config.model_path,
            torch_dtype=None,
            device=self.config.device,
        )
        #self.model_ref.load_model_weights(self.config.model_path, self.config.device)
        self.model_ref.to(self.config.device)
        for p in self.model_ref.parameters():
            p.requires_grad = False
        self.model_ref.eval()
        
        # 从本地加载tokenizer
        print("正在从本地加载tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, 
            local_files_only=True,
            trust_remote_code=True,
            padding_side='left'  # Decoder-only 模型使用 left-padding
)
            # 打印调试信息
            print("tokenizer从本地加载完成！")
        except Exception as e:
            raise RuntimeError(f"从本地加载tokenizer失败: {e}")
        
        self.logger.info("模型和分词器加载完成")
        
    def _check_model_files(self):
        """检查模型必要文件是否存在"""
        print("检查模型文件...")
        
        # 检查模型权重文件
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"模型权重文件不存在: {self.config.model_path}")
        
        # 检查必要文件是否存在
        required_files = [
            'config.json',
            'tokenizer.json',
            'vocab.json', 
            'merges.txt',
            'special_tokens_map.json'
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = self.model_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            raise FileNotFoundError(
                f"缺少必要的模型文件: {missing_files}\n"
                f"请确保模型目录 {self.model_dir} 包含所有必要文件:\n"
                f"  - config.json (模型配置)\n"
                f"  - tokenizer.json (分词器配置)\n"
                f"  - vocab.json (词汇表)\n"
                f"  - merges.txt (BPE合并规则)\n"
                f"  - special_tokens_map.json (特殊token映射)\n"
                f"  - {Path(self.config.model_path).name} (模型权重)"
            )
        
        print("✓ 所有必要文件检查通过")
        
    def setup_data_loaders(self):
        """设置数据加载器"""
        # 构建数据文件路径
        self.data_files = self._build_data_paths()
        
        # 加载数据
        self.train_examples = read_qwen_examples(self.data_files['train'], self.config)
        self.dev_examples = read_qwen_examples(self.data_files['dev'], self.config)
        self.test_examples = read_qwen_examples(self.data_files['test'], self.config)
        
        # 转换为特征
        self.train_features = convert_qwen_examples_to_features(
            self.train_examples, self.tokenizer, self.config, stage='train'
        )
        self.dev_features = convert_qwen_examples_to_features(
            self.dev_examples, self.tokenizer, self.config, stage='train'
        )
        self.test_features = convert_qwen_examples_to_features(
            self.test_examples, self.tokenizer, self.config, stage='train'
        )
        
        # 创建数据加载器
        self.train_dataloader = self._create_dataloader(
            self.train_features, self.config.train_batch_size, shuffle=True
        )
        self.dev_dataloader = self._create_dataloader(
            self.dev_features, self.config.train_batch_size, shuffle=False
        )
        self.test_dataloader = self._create_dataloader(
            self.test_features, self.config.test_batch_size, shuffle=False
        )
        
        self.logger.info(f"数据加载完成 - 训练: {len(self.train_features)}, "
                        f"验证: {len(self.dev_features)}, 测试: {len(self.test_features)}")
        
    def _build_data_paths(self) -> Dict[str, str]:
        """构建Qwen格式数据文件路径"""
        l1, l2 = self.config.source_lang, self.config.target_lang
        
        # 尝试不同的路径组合
        possible_paths = [
            f"{self.config.data_path}/qwen/{self.dir_dict[l1]}-{self.dir_dict[l2]}/",
            f"{self.config.data_path}/qwen/{self.dir_dict[l2]}-{self.dir_dict[l1]}/",
            f"{self.config.data_path}/{self.dir_dict[l1]}-{self.dir_dict[l2]}/",  # 备选路径
            f"{self.config.data_path}/{self.dir_dict[l2]}-{self.dir_dict[l1]}/"   # 备选路径
        ]
        
        data_dir = None
        for path in possible_paths:
            if os.path.exists(path):
                data_dir = path
                break
                
        if data_dir is None:
            raise FileNotFoundError(f"找不到Qwen格式数据目录: {possible_paths}")
            
        return {
            'train': f"{data_dir}train.jsonl",
            'dev': f"{data_dir}val.jsonl",
            'test': f"{data_dir}test.jsonl"
        }
        
    def _create_dataloader(self, features: List[InputFeatures], 
                          batch_size: int, shuffle: bool = False) -> DataLoader:
        """创建数据加载器"""
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in features], dtype=torch.long)
        indices = torch.arange(len(features))
        
        dataset = TensorDataset(all_source_ids, all_source_mask, 
                               all_target_ids, all_target_mask, indices)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
    def setup_ppo_trainer(self):
        """设置PPO训练器"""
        ppo_config = {
            "batch_size": self.config.train_batch_size,
            'eos_token_id': self.tokenizer.eos_token_id,
            'lr': self.config.learning_rate,
            "adap_kl_ctrl": True,
            'init_kl_coef': self.config.kl_coef,
            "target": self.config.kl_target,
            "vf_coef": self.config.vf_coef
        }
        
        self.ppo_trainer = PPOTrainer(self.model, self.model_ref, **ppo_config)
        self.logger.info("PPO训练器设置完成")
        
    def setup_training_stats(self):
        """设置训练统计"""
        self.training_stats = {
            'nsteps': 0,
            'total_nerrors': 0,
            'total_rewards': 0,
            'total_nnodes': 0,
            'total_nerrors_ref': 0,
            'total_nnodes_ref': 0,
            'total_seen': 0
        }
        
        # 创建结果目录
        self.results_dir = Path(self.config.output_path) / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建检查点目录
        self.checkpoint_dir = Path(self.config.output_path) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 🔧 新增：初始化 Tensorboard
        self.tensorboard_writer = None
        if self.config.use_tensorboard:
            # 设置Tensorboard日志目录
            if self.config.tensorboard_log_dir:
                tb_log_dir = Path(self.config.tensorboard_log_dir)
            else:
                tb_log_dir = Path(self.config.output_path) / "tensorboard"
            
            # 添加时间戳和运行ID到日志目录
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            tb_log_dir = tb_log_dir / f"run_{self.config.run_id}_{timestamp}"
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            
            # 初始化SummaryWriter
            self.tensorboard_writer = SummaryWriter(log_dir=str(tb_log_dir))
            self.logger.info(f"Tensorboard日志保存到: {tb_log_dir}")
            
            # 记录配置信息
            config_text = str(self.config).replace(',', '\n')
            self.tensorboard_writer.add_text("Config", config_text, 0)
        else:
            self.logger.info("Tensorboard日志已禁用")
            
    def train(self):
        """主训练循环"""
        self.logger.info("开始训练...")
        
        for epoch in range(self.config.train_epochs):
            self.logger.info(f"开始第 {epoch} 轮训练")
            
            # 每轮进行多次采样
            for sample_idx in range(self.config.num_syn_samples):
                self._train_epoch(epoch, sample_idx)
                
            # 保存模型和评估
            self._save_checkpoint(epoch)
            self._evaluate(epoch)
            
    def _train_epoch(self, epoch: int, sample_idx: int):
        """训练一个epoch"""
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}, Sample {sample_idx}")
        
        for batch_idx, batch in enumerate(pbar):
            # 处理批次数据
            batch = tuple(t.to(self.config.device) for t in batch)
            # DataLoader 返回 (source_ids, source_mask, target_ids, target_mask, indices)
            source_ids, source_mask, target_ids, target_mask, ind = batch
            # 生成代码
            response_ids = self._generate_code(source_ids, source_mask)
            response_ids_ref = self._generate_code_ref(source_ids, source_mask)

            # 计算奖励
            reward, metrics = self._compute_reward(response_ids, response_ids_ref, target_ids)
            
            # 更新统计信息
            self._update_stats(reward, metrics, len(source_ids))
            
            # PPO训练步骤
            train_stats = self.ppo_trainer.step(
                source_ids, source_mask, response_ids, response_ids_ref, 
                reward.to(self.config.device)
            )
            
            # 更新进度条
            pbar.set_description(
                f"Epoch {epoch}, Sample {sample_idx}, "
                f"Avg Errors: {self.training_stats['total_nerrors']/self.training_stats['total_seen']:.5f}"
            )
            
            # 记录训练统计
            self._log_training_step(epoch, sample_idx, batch_idx, reward, metrics, train_stats)
            
            self.training_stats['nsteps'] += 1
            
    def _generate_code(self, source_ids: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        """生成代码"""
        full = respond_to_batch(
            self.model, source_ids, source_mask,
            max_target_length=self.config.max_target_length,
            top_k=self.config.action_space, top_p=1.0,
            tokenizer=self.tokenizer
        ).detach()
        # full包含 [prompt | generated]；仅保留generated部分
        gen_start = source_ids.size(1)
        return torch.clone(full[:, gen_start:])  # [B, <=max_new_tokens]
        
    def _generate_code_ref(self, source_ids: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        """生成参考代码"""
        full = respond_to_batch(
            self.model_ref, source_ids, source_mask,
            max_target_length=self.config.max_target_length,
            top_k=self.config.action_space, top_p=1.0,
            tokenizer=self.tokenizer
        ).detach()
        # full包含 [prompt | generated]；仅保留generated部分
        gen_start = source_ids.size(1)
        return torch.clone(full[:, gen_start:])  # [B, <=max_new_tokens]
        
    def _compute_reward(self, response_ids: torch.Tensor, response_ids_ref: torch.Tensor, 
                       target_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """计算奖励"""
        reward, mean_rate, mean_ast_match, mean_dfg_match, num_errors, num_errors_ref, num_nodes, num_nodes_ref = self.get_reward_func(
            lang=self.config.target_lang,
            code_ids=response_ids,
            code_ref_ids=response_ids_ref,
            gold_ids=target_ids,
            tokenizer=self.tokenizer
        )
        
        metrics = {
            'mean_rate': mean_rate,
            'mean_ast_match': mean_ast_match,
            'mean_dfg_match': mean_dfg_match,
            'num_errors': num_errors,
            'num_errors_ref': num_errors_ref,
            'num_nodes': num_nodes,
            'num_nodes_ref': num_nodes_ref
        }
        
        return reward, metrics
        
    def _update_stats(self, reward: torch.Tensor, metrics: Dict, batch_size: int):
        """更新训练统计"""
        self.training_stats['total_rewards'] += float(sum(reward.sum(axis=-1).tolist()))
        self.training_stats['total_nerrors'] += sum(metrics['num_errors'])
        self.training_stats['total_nnodes'] += sum(metrics['num_nodes'])
        self.training_stats['total_nerrors_ref'] += sum(metrics['num_errors_ref'])
        self.training_stats['total_nnodes_ref'] += sum(metrics['num_nodes_ref'])
        self.training_stats['total_seen'] += batch_size
        
    def _log_training_step(self, epoch: int, sample_idx: int, batch_idx: int,
                          reward: torch.Tensor, metrics: Dict, train_stats: Dict):
        """记录训练步骤"""
        # 计算平均指标
        avg_reward = float(sum(reward.sum(axis=-1).tolist())) / len(reward)
        avg_errors = sum(metrics['num_errors']) / len(metrics['num_errors'])
        avg_errors_ref = sum(metrics['num_errors_ref']) / len(metrics['num_errors_ref'])
        avg_nodes = sum(metrics['num_nodes']) / len(metrics['num_nodes'])
        avg_nodes_ref = sum(metrics['num_nodes_ref']) / len(metrics['num_nodes_ref'])
        
        # 🔧 新增：记录到 Tensorboard
        if (self.tensorboard_writer and 
            self.training_stats['nsteps'] % self.config.log_every_n_steps == 0):
            
            global_step = self.training_stats['nsteps']
            
            # 奖励相关指标
            self.tensorboard_writer.add_scalar("Training/Average_Reward", avg_reward, global_step)
            self.tensorboard_writer.add_scalar("Training/Compilation_Success_Rate", metrics['mean_rate'], global_step)
            self.tensorboard_writer.add_scalar("Training/AST_Match_Score", metrics['mean_ast_match'], global_step)
            self.tensorboard_writer.add_scalar("Training/DFG_Match_Score", metrics['mean_dfg_match'], global_step)
            
            # 代码质量指标
            self.tensorboard_writer.add_scalar("Code_Quality/Avg_Errors", avg_errors, global_step)
            self.tensorboard_writer.add_scalar("Code_Quality/Avg_Errors_Ref", avg_errors_ref, global_step)
            self.tensorboard_writer.add_scalar("Code_Quality/Avg_Nodes", avg_nodes, global_step)
            self.tensorboard_writer.add_scalar("Code_Quality/Avg_Nodes_Ref", avg_nodes_ref, global_step)
            
            # PPO训练指标
            if 'objective/kl' in train_stats:
                self.tensorboard_writer.add_scalar(
                    "PPO/KL_Divergence", float(train_stats['objective/kl']), global_step
                )
            if 'objective/entropy' in train_stats:
                self.tensorboard_writer.add_scalar(
                    "PPO/Entropy", float(train_stats['objective/entropy']), global_step
                )
            if 'ppo/loss/total' in train_stats:
                self.tensorboard_writer.add_scalar("PPO/Total_Loss", train_stats['ppo/loss/total'].item(), global_step)
            if 'ppo/loss/policy' in train_stats:
                self.tensorboard_writer.add_scalar("PPO/Policy_Loss", train_stats['ppo/loss/policy'].item(), global_step)
            if 'ppo/loss/value' in train_stats:
                self.tensorboard_writer.add_scalar("PPO/Value_Loss", train_stats['ppo/loss/value'].item(), global_step)
            if 'ppo/policy/advantages_mean' in train_stats:
                self.tensorboard_writer.add_scalar("PPO/Advantages_Mean", train_stats['ppo/policy/advantages_mean'].item(), global_step)
            if 'ppo/returns/mean' in train_stats:
                self.tensorboard_writer.add_scalar("PPO/Returns_Mean", train_stats['ppo/returns/mean'].item(), global_step)
            if 'ppo/val/mean' in train_stats:
                self.tensorboard_writer.add_scalar("PPO/Value_Mean", train_stats['ppo/val/mean'].item(), global_step)
            
            # 学习率（如果可获取）
            try:
                current_lr = self.ppo_trainer.optimizer.param_groups[0]['lr']
                self.tensorboard_writer.add_scalar("Training/Learning_Rate", current_lr, global_step)
            except:
                pass
        
        # 记录到CSV文件
        csv_line = [
            datetime.datetime.now().strftime("%H:%M:%S"),
            str(self.config.run_id),
            str(self.config.train_batch_size),
            str(self.config.max_source_length),
            str(self.config.max_target_length),
            str(self.config.learning_rate),
            str(epoch),
            str(self.training_stats['nsteps']),
            f"{avg_reward:.4f}",
            f"{avg_errors:.4f}",
            f"{avg_errors_ref:.4f}",
            f"{avg_nodes:.4f}",
            f"{avg_nodes_ref:.4f}",
            str(float(train_stats['objective/kl'])),
            str(float(train_stats['objective/entropy'])),
            str(train_stats['ppo/loss/total'].item()),
            str(train_stats['ppo/loss/policy'].item()),
            str(train_stats['ppo/loss/value'].item()),
            str(train_stats['ppo/policy/advantages_mean'].item()),
            str(train_stats['ppo/returns/mean'].item()),
            str(train_stats['ppo/val/mean'].item()),
            str(metrics['mean_rate']),
            str(metrics['mean_ast_match']),
            str(metrics['mean_dfg_match'])
        ]
        
        csv_file = self.results_dir / f"{self.config.source_lang}-{self.config.target_lang}.csv"
        with open(csv_file, 'a') as f:
            f.write(','.join(csv_line) + '\n')
            
    def _save_checkpoint(self, epoch: int):
        """保存检查点"""
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        checkpoint_path = self.checkpoint_dir / f"pytorch_model_ep{epoch}.bin"
        torch.save(model_to_save.state_dict(), checkpoint_path)
        self.logger.info(f"模型已保存到: {checkpoint_path}")
        
        # 🔧 新增：清理旧检查点
        self._cleanup_old_checkpoints()
        
    def _cleanup_old_checkpoints(self):
        """清理旧的检查点文件，只保留最新的N个"""
        if self.config.max_checkpoints <= 0:
            return  # 不限制检查点数量
            
        # 获取所有检查点文件
        checkpoint_pattern = "pytorch_model_ep*.bin"
        checkpoint_files = list(self.checkpoint_dir.glob(checkpoint_pattern))
        
        if len(checkpoint_files) <= self.config.max_checkpoints:
            return  # 数量未超限
            
        # 按照修改时间排序（最新的在前）
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # 删除超出限制的旧文件
        files_to_delete = checkpoint_files[self.config.max_checkpoints:]
        
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                self.logger.info(f"删除旧检查点: {file_path}")
            except Exception as e:
                self.logger.warning(f"删除检查点失败 {file_path}: {e}")
                
        if files_to_delete:
            self.logger.info(f"已清理 {len(files_to_delete)} 个旧检查点，保留最新的 {self.config.max_checkpoints} 个")
                
    def _evaluate(self, epoch: int):
        """评估模型"""
        self.model.eval()
        self.logger.info(f"开始第 {epoch} 轮评估")
        
        # 训练集评估
        train_errors, train_errors_ref = self._evaluate_dataset(
            epoch, self.train_features, self.train_dataloader, 'train'
        )
        self.model.train()
        
            # 测试集评估
        test_errors, test_errors_ref = self._evaluate_dataset(
            epoch, self.test_features, self.test_dataloader, 'test'
        )
        self.model.train()
        
        self.logger.info(f"Epoch {epoch} 评估结果:")
        self.logger.info(f"  训练集 - 模型错误: {train_errors}, 参考模型错误: {train_errors_ref}")
        self.logger.info(f"  测试集 - 模型错误: {test_errors}, 参考模型错误: {test_errors_ref}")
        
        # 🔧 新增：记录评估指标到 Tensorboard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar("Evaluation/Train_Errors", train_errors, epoch)
            self.tensorboard_writer.add_scalar("Evaluation/Train_Errors_Ref", train_errors_ref, epoch)
            self.tensorboard_writer.add_scalar("Evaluation/Test_Errors", test_errors, epoch)
            self.tensorboard_writer.add_scalar("Evaluation/Test_Errors_Ref", test_errors_ref, epoch)
            
            # 计算错误率
            if len(self.train_features) > 0:
                train_error_rate = train_errors / len(self.train_features)
                self.tensorboard_writer.add_scalar("Evaluation/Train_Error_Rate", train_error_rate, epoch)
            
            if len(self.test_features) > 0:
                test_error_rate = test_errors / len(self.test_features)
                self.tensorboard_writer.add_scalar("Evaluation/Test_Error_Rate", test_error_rate, epoch)
        self.model.train()
            
    def _evaluate_dataset(self, epoch: int, features: List[InputFeatures], 
                         dataloader: DataLoader, prefix: str) -> Tuple[int, int]:
        """评估数据集"""
        pred_ids = []
        pred_ids_ref = []
        indices = []
        nerrors = 0
        nerrors_ref = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = tuple(t.to(self.config.device) for t in batch)
                source_ids, source_mask, target_ids, target_mask, ind = batch
                
                # 生成预测
                full_preds = respond_to_batch(
                    self.model, source_ids, source_mask,
                    max_target_length=self.config.max_target_length,
                    top_k=self.config.action_space, top_p=1.0,
                    tokenizer=self.tokenizer
                )
                preds = full_preds[:, source_ids.size(1):]
                
                full_preds_ref = respond_to_batch(
                    self.model_ref, source_ids, source_mask,
                    max_target_length=self.config.max_target_length,
                    top_k=self.config.action_space, top_p=1.0,
                    tokenizer=self.tokenizer
                )
                preds_ref = full_preds_ref[:, source_ids.size(1):]
                
                # 计算错误数
                nerrors += sum(self.get_reward_func(
                    lang=self.config.target_lang,
                    code_ids=preds,
                    code_ref_ids=preds_ref,
                    gold_ids=target_ids,
                    tokenizer=self.tokenizer
                )[4])
                
                nerrors_ref += sum(self.get_reward_func(
                    lang=self.config.target_lang,
                    code_ids=preds_ref,
                    code_ref_ids=preds_ref,
                    gold_ids=target_ids,
                    tokenizer=self.tokenizer
                )[5])
                
                # 保存预测结果
                pred_ids.extend(list(preds.cpu().numpy()))
                pred_ids_ref.extend(list(preds_ref.cpu().numpy()))
                indices.extend(list(ind.cpu().numpy()))
                
        # 解码并保存结果
        self._save_predictions(epoch, prefix, pred_ids, pred_ids_ref, indices, features)
        
        return nerrors, nerrors_ref
        
    def _save_predictions(self, epoch: int, prefix: str, pred_ids: List, 
                         pred_ids_ref: List, indices: List, features: List[InputFeatures]):
        """保存预测结果"""
        # 解码预测结果
        raw_predictions = [
            self.tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for id in pred_ids
        ]
        raw_predictions_ref = [
            self.tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for id in pred_ids_ref
        ]
        
        # 从Qwen响应中提取代码
        predictions = [
            extract_code_from_qwen_response(pred, self.config.target_lang)
            for pred in raw_predictions
        ]
        predictions_ref = [
            extract_code_from_qwen_response(pred, self.config.target_lang)
            for pred in raw_predictions_ref
        ]
        
        # 保存到文件
        model_file = self.checkpoint_dir / f"{prefix}.model_ep{epoch}"
        ref_file = self.checkpoint_dir / f"{prefix}.model_ref_ep{epoch}"
        gold_file = self.checkpoint_dir / f"{prefix}.gold_ep{epoch}"
        
        with open(model_file, 'w') as f_model, \
             open(ref_file, 'w') as f_ref, \
             open(gold_file, 'w') as f_gold:
            
            for pred, ref, i in zip(predictions, predictions_ref, indices):
                f_model.write(pred + '\n')
                f_ref.write(ref + '\n')
                # 对于gold，也需要提取代码
                gold_code = extract_code_from_qwen_response(features[i].target, self.config.target_lang)
                f_gold.write(gold_code + '\n')


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Qwen2.5-Coder PPO代码生成训练程序")
    
    # 必需参数
    parser.add_argument("--source_lang", required=True, type=str,
                       help="源代码语言")
    parser.add_argument("--target_lang", required=True, type=str,
                       help="目标代码语言")
    parser.add_argument("--model_path", required=True, type=str,
                       help="Qwen2.5-Coder模型路径")
    parser.add_argument("--data_path", required=True, type=str,
                       help="Qwen格式数据目录路径")
    parser.add_argument("--output_path", required=True, type=str,
                       help="输出目录路径")
    
    # 可选参数
    parser.add_argument("--max_source_length", default=400, type=int,
                       help="最大源代码长度")
    parser.add_argument("--max_target_length", default=400, type=int,
                       help="最大目标代码长度")
    parser.add_argument("--train_batch_size", default=16, type=int,
                       help="训练批次大小")
    parser.add_argument("--test_batch_size", default=48, type=int,
                       help="测试批次大小")
    parser.add_argument("--train_epochs", default=1000000, type=int,
                       help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="学习率")
    parser.add_argument("--kl_coef", type=float, default=0.05,
                       help="KL系数")
    parser.add_argument("--kl_target", type=float, default=1.0,
                       help="KL目标值")
    parser.add_argument("--vf_coef", type=float, default=1e-3,
                       help="价值函数系数")
    parser.add_argument("--action_space", default=2, type=int,
                       help="动作空间大小（top_k）")
    parser.add_argument("--num_syn_samples", default=5, type=int,
                       help="每轮采样次数")
    parser.add_argument("--run_id", default=1, type=int,
                       help="运行ID")
    parser.add_argument("--seed", default=42, type=int,
                       help="随机种子")
    
    # 🔧 新增：检查点保存控制参数
    parser.add_argument("--save_steps", default=1, type=int,
                       help="每N轮保存一次检查点（默认每轮都保存）")
    parser.add_argument("--max_checkpoints", default=10, type=int,
                       help="最多保留N个检查点，0表示不限制（默认保留10个）")
    
    # 🔧 新增：Tensorboard 支持参数
    parser.add_argument("--use_tensorboard", action="store_true", default=True,
                       help="启用Tensorboard日志记录（默认启用）")
    parser.add_argument("--no_tensorboard", action="store_false", dest="use_tensorboard",
                       help="禁用Tensorboard日志记录")
    parser.add_argument("--tensorboard_log_dir", default=None, type=str,
                       help="Tensorboard日志目录（默认为output_path/tensorboard）")
    parser.add_argument("--log_every_n_steps", default=1, type=int,
                       help="每N个训练步骤记录一次指标到Tensorboard（默认每步都记录）")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建配置对象
    config = TrainingConfig(
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_path,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        train_epochs=args.train_epochs,
        learning_rate=args.learning_rate,
        kl_coef=args.kl_coef,
        kl_target=args.kl_target,
        vf_coef=args.vf_coef,
        action_space=args.action_space,
        num_syn_samples=args.num_syn_samples,
        run_id=args.run_id,
        seed=args.seed,
        save_steps=args.save_steps,
        max_checkpoints=args.max_checkpoints,
        use_tensorboard=args.use_tensorboard,
        tensorboard_log_dir=args.tensorboard_log_dir,
        log_every_n_steps=args.log_every_n_steps
    )
    
    print("=" * 60)
    print("🚀 Qwen2.5-Coder PPO代码翻译训练程序")
    print("=" * 60)
    print(f"📝 源语言: {config.source_lang}")
    print(f"🎯 目标语言: {config.target_lang}")
    print(f"🤖 模型路径: {config.model_path}")
    print(f"📂 数据路径: {config.data_path}")
    print(f"💾 输出路径: {config.output_path}")
    print(f"🔧 设备: {config.device}")
    print("=" * 60)
    
    # 创建训练器并开始训练
    trainer = CodeTranslationTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main() 