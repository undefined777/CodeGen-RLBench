#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import shutil


# 🔧 新增：Tensorboard 支持
from torch.utils.tensorboard import SummaryWriter

# Internal imports
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
    Extract pure code from Qwen model's response
    
    Args:
        response: Qwen model's complete response
        target_lang: Target language, used to match code blocks
    
    Returns:
        Extracted pure code string
    """
    # Language name mapping, support different variants
    lang_patterns = {
        'cpp': ['cpp', 'c++', 'cxx'],
        'java': ['java'],
        'python': ['python', 'py'],
        'javascript': ['javascript', 'js'],
        'c': ['c'],
        'php': ['php'],
        'c_sharp': ['csharp', 'c#', 'cs']
    }
    
    # Get all possible patterns for the target language
    target_patterns = lang_patterns.get(target_lang, [target_lang])
    
    # Try to match code blocks
    for pattern in target_patterns:
        # Match ```lang\ncode\n``` format, escape special characters
        escaped_pattern = re.escape(pattern)
        code_match = re.search(rf'```{escaped_pattern}\s*\n(.*?)\n```', response, re.DOTALL | re.IGNORECASE)
        if code_match:
            return code_match.group(1).strip()
    
    # If no specific language code block is found, try to match generic code blocks
    code_match = re.search(r'```\s*\n(.*?)\n```', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    # If no code block is found, try to extract content after "translation:"
    translation_match = re.search(r'translation:\s*\n\n(.+)', response, re.DOTALL | re.IGNORECASE)
    if translation_match:
        return translation_match.group(1).strip()
    
    # Last fallback: return content after removing common prefixes
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
    
    # Remove trailing ```
    if response.endswith("```"):
        response = response[:-3].strip()
    
    return response


def read_qwen_examples(filename: str, args) -> List[Example]:
    """
    Read training examples from Qwen format JSONL file
    
    Args:
        filename: JSONL file path
        args: Parameter object containing language configuration
    
    Returns:
        List of Example objects
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
                
                # Extract source code from user message - for building Example.source
                source_code = extract_code_from_qwen_response(user_message, args.source_lang)
                
                # Extract target code from assistant message - for building Example.target
                target_code = extract_code_from_qwen_response(assistant_message, args.target_lang)
                
                if not source_code or not target_code:
                    continue
                
                e = Example(
                    idx=idx,
                    source=source_code,
                    target=target_code,
                    source_orig=user_message,      # First store user; system is separate
                    target_orig=assistant_message
                )
                # Dynamically mount system (if none, empty string)
                setattr(e, "system_orig", system_message or "")
                examples.append(e)

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Skipping line {idx+1}, parsing error: {e}")
                continue
    
    return examples


def convert_qwen_examples_to_features(examples, tokenizer, args, stage=None):
    """
    Convert Qwen examples to model input features
    Special handling for tokenization of dialog format
    """
    features = []
    for example_index, example in enumerate(examples):
        # For Qwen, we use complete dialog messages
        # source_orig contains complete user prompt
        # target_orig contains complete assistant response
        
        # Can use tokenizer's chat template, or simple concatenation
        if hasattr(tokenizer, 'apply_chat_template'):
            # Try using chat template
            try:
                # 强制添加system消息，确保所有输入都包含system指令
                messages = []
                if hasattr(example, "system_orig") and example.system_orig:
                    # 使用数据中的自定义system消息
                    system_content = example.system_orig
                else:
                    # 使用默认的system消息
                    system_content = "You are a helpful assistant for code translation. You specialize in translating Java code to C++ code while maintaining functionality and best practices."
                
                messages.append({"role": "system", "content": system_content})
                messages.append({"role": "user", "content": example.source_orig})
                source_text = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                ) 
                # 🔧 新增：验证apply_chat_template结果
                if hasattr(example, "system_orig") and example.system_orig and "system" not in source_text:
                    print(f"⚠️ 警告：apply_chat_template结果中缺少system内容")
                    print(f"   原始system: {example.system_orig[:50]}...")
                    print(f"   生成结果: {source_text[:100]}...")
            except Exception as e:
                # If failed, use original content
                print(f"❌ apply_chat_template失败: {e}")
                print(f"   使用原始内容作为fallback")
                source_text = example.source_orig
        else:
            source_text = example.source_orig
            
        # tokenize source - directly encode, no special tokens
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
    This avoids using policy's EOS position for ref/gold (original implementation issue).
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

        # 使用全局最大长度,保留所有信息
        max_len = max(len(x) for tri in triplets for x in tri) if triplets else 1

        def _pad(seq):
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
    """Training configuration class - Qwen专用版本"""
    # Language configuration
    source_lang: str
    target_lang: str
    
    # Aliases, for compatibility with old code
    @property
    def l1(self):
        return self.source_lang
    
    @property
    def l2(self):
        return self.target_lang
    
    # Model configuration
    model_path: str
    max_source_length: int = 400
    max_target_length: int = 400
    
    # Training configuration
    train_batch_size: int = 16
    test_batch_size: int = 48
    train_epochs: int = 1000000
    learning_rate: float = 1e-5
    kl_coef: float = 0.05
    kl_target: float = 1
    vf_coef: float = 1e-3
    
    # Generation configuration
    action_space: int = 2  # top_k
    num_syn_samples: int = 5
    
    # Path configuration
    data_path: str = None
    output_path: str = None
    baseline_output_path: str = None
    
    # 🔧 Enhanced: Checkpoint saving control
    save_every_n_steps: int = 0  # Save checkpoint every N training steps (0 means disabled)
    max_checkpoints: int = 10  # Maximum number of checkpoints to retain, 0 means no limit
    
    # 🔧 New: Performance-based saving
    save_best_only: bool = False  # Only save when performance improves
    save_metric: str = "reward"  # Metric to track for best model: "reward", "compilation_rate", "ast_match", "dfg_match"
    save_threshold: float = 0.0  # Minimum improvement threshold for saving
    
    # 🔧 New: Emergency saving
    save_on_error: bool = True  # Save checkpoint when training error occurs
    
    # 🔧 New: Tensorboard support
    use_tensorboard: bool = True  # Whether to enable Tensorboard logging
    tensorboard_log_dir: str = None  # Tensorboard log directory, None means use default path
    log_every_n_steps: int = 1  # Log metrics every N training steps

    # PPO configuration
    minibatch_size: int = 1
    
    # 🔧 New: Gradient accumulation configuration
    gradient_accumulation_steps: int = 4  # 梯度累积步数，推荐：batch_size=16时用4步，有效batch=16*4=64
    
    # 🔧 New: Critic warmup configuration
    critic_warmup_steps: int = 0  # critic预热步数，推荐：50-100步让critic先稳定
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Running configuration
    run_id: int = 1
    seed: int = 42


class CodeTranslationTrainer:
    """Code translation PPO trainer"""
    
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
        
        # Create reward function wrapper
        self.get_reward_func = create_reward_wrapper(get_reward)
        
    def setup_logging(self):
        """Setup logging system"""
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
        self.logger.info(f"Training configuration: {self.config}")
        
    def setup_device(self):
        """Setup computing device"""
        if self.config.device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA not available, switching to CPU")
            self.config.device = "cpu"
        
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            
        self.logger.info(f"Using device: {self.config.device}")
        
    def setup_language_mappings(self):
        """Setup language mappings"""
        self.dir_dict = {
            'javascript': 'Javascript', 'java': 'Java', 'c_sharp': 'C#', 
            'php': 'PHP', 'python': 'Python', 'c': 'C', 'cpp': 'C++'
        }
        
    def setup_parsers(self):
        """Setup code parsers"""
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
                self.logger.warning(f"Failed to load {lang} parser: {e}")
                
    def setup_models(self):
        """Setup models and tokenizers"""
        # Get model file directory
        self.model_dir = Path(self.config.model_path)
        
        # Check and prepare tokenizer and config files
        self._check_model_files()
        
        print(f"Loading model to device: {self.config.device}")
        print(f"Loading model file: {self.config.model_path}")
        # Load fine-tuned complete model (including architecture and weights)
        self.model = QwenCoderHeadWithValueModelLocal(
            self.config.model_path,
            torch_dtype=torch.bfloat16,              # Keep default dtype; below unified .to()
            device=self.config.device,
        )
        self.model.to(self.config.device)
        self.model.train() 
        
        # Load reference model (fixed)
        self.model_ref = QwenCoderHeadWithValueModelLocal(
            self.config.model_path,
            torch_dtype=torch.bfloat16,
            device=self.config.device,
        )
        #self.model_ref.load_model_weights(self.config.model_path, self.config.device)
        self.model_ref.to(self.config.device)
        for p in self.model_ref.parameters():
            p.requires_grad = False
        self.model_ref.eval()

        #self.model.model.gradient_checkpointing_enable()
        self.model.model.config.use_cache = False          # Already double-checked in forward
        self.model_ref.model.config.use_cache = False
        
        # Load tokenizer from local
        print("Loading tokenizer from local...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, 
            local_files_only=True,
            trust_remote_code=True,
            padding_side='right'
)
            # Print debug information
            print("Tokenizer loaded from local!")
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer from local: {e}")
        
        self.logger.info("Models and tokenizers loaded")
        
    def _check_model_files(self):
        """Check if model necessary files exist"""
        print("Checking model files...")
        
        # Check model weight file
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model weight file does not exist: {self.config.model_path}")
        
        # Check necessary files
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
                f"Missing necessary model files: {missing_files}\n"
                f"Please ensure model directory {self.model_dir} contains all necessary files:\n"
                f"  - config.json (model configuration)\n"
                f"  - tokenizer.json (tokenizer configuration)\n"
                f"  - vocab.json (vocabulary)\n"
                f"  - merges.txt (BPE merge rules)\n"
                f"  - special_tokens_map.json (special token mapping)\n"
                f"  - {Path(self.config.model_path).name} (model weights)"
            )
        
        print("✓ All necessary files checked")
        
    def setup_data_loaders(self):
        """Setup data loaders"""
        # Build data file paths
        self.data_files = self._build_data_paths()
        
        # Load data
        self.train_examples = read_qwen_examples(self.data_files['train'], self.config)
        self.dev_examples = read_qwen_examples(self.data_files['dev'], self.config)
        self.test_examples = read_qwen_examples(self.data_files['test'], self.config)
        
        # Convert to features
        self.train_features = convert_qwen_examples_to_features(
            self.train_examples, self.tokenizer, self.config, stage='train'
        )
        self.dev_features = convert_qwen_examples_to_features(
            self.dev_examples, self.tokenizer, self.config, stage='train'
        )
        self.test_features = convert_qwen_examples_to_features(
            self.test_examples, self.tokenizer, self.config, stage='train'
        )
        
        # Create data loaders
        self.train_dataloader = self._create_dataloader(
            self.train_features, self.config.train_batch_size, shuffle=True
        )
        self.dev_dataloader = self._create_dataloader(
            self.dev_features, self.config.train_batch_size, shuffle=False
        )
        self.test_dataloader = self._create_dataloader(
            self.test_features, self.config.test_batch_size, shuffle=False
        )
        
        self.logger.info(f"Data loaded - train: {len(self.train_features)}, "
                        f"dev: {len(self.dev_features)}, test: {len(self.test_features)}")
        
    def _build_data_paths(self) -> Dict[str, str]:
        """Build Qwen format data file paths"""
        l1, l2 = self.config.source_lang, self.config.target_lang
        
        # Try different path combinations
        possible_paths = [
            f"{self.config.data_path}/qwen/{self.dir_dict[l1]}-{self.dir_dict[l2]}/",
            f"{self.config.data_path}/qwen/{self.dir_dict[l2]}-{self.dir_dict[l1]}/",
            f"{self.config.data_path}/{self.dir_dict[l1]}-{self.dir_dict[l2]}/",  # Alternative paths
            f"{self.config.data_path}/{self.dir_dict[l2]}-{self.dir_dict[l1]}/"   # Alternative paths
        ]
        
        data_dir = None
        for path in possible_paths:
            if os.path.exists(path):
                data_dir = path
                break
                
        if data_dir is None:
            raise FileNotFoundError(f"Qwen format data directory not found: {possible_paths}")
            
        return {
            'train': f"{data_dir}train.jsonl",
            'dev': f"{data_dir}val.jsonl",
            'test': f"{data_dir}test.jsonl"
        }
        
    def _create_dataloader(self, features: List[InputFeatures], 
                          batch_size: int, shuffle: bool = False) -> DataLoader:
        """Create data loader"""
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in features], dtype=torch.long)
        indices = torch.arange(len(features))
        
        dataset = TensorDataset(all_source_ids, all_source_mask, 
                               all_target_ids, all_target_mask, indices)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
    def setup_ppo_trainer(self):
        """Setup PPO trainer"""
        ppo_config = {
            "batch_size": self.config.train_batch_size,
            'eos_token_id': self.tokenizer.eos_token_id,
            'lr': self.config.learning_rate,
            "adap_kl_ctrl": True,
            'init_kl_coef': self.config.kl_coef,
            "target": self.config.kl_target,
            "vf_coef": self.config.vf_coef,
            "minibatch_size": self.config.minibatch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,  # 🔧 New: 传递梯度累积参数
            "critic_warmup_steps": self.config.critic_warmup_steps,  # 🔧 New: 传递critic预热参数
            "tokenizer": self.tokenizer
        }
        
        self.ppo_trainer = PPOTrainer(self.model, self.model_ref, **ppo_config)
        self.logger.info("PPO trainer setup complete")
        
    def setup_training_stats(self):
        """Setup training statistics"""
        self.training_stats = {
            'nsteps': 0,
            'total_nerrors': 0,
            'total_rewards': 0,
            'total_rewards_ref': 0,
            'total_nnodes': 0,
            'total_nerrors_ref': 0,
            'total_nnodes_ref': 0,
            'total_seen': 0
        }
        
        # 🔧 New: Performance tracking for best model saving
        self.best_metrics = {
            'reward': -float('inf'),
            'compilation_rate': -float('inf'),
            'ast_match': -float('inf'),
            'dfg_match': -float('inf')
        }
        self.last_save_step = 0  # Track last save step to avoid duplicate saves
        
        # Create results directory
        self.results_dir = Path(self.config.output_path) / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(self.config.output_path) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 🔧 New: Initialize Tensorboard
        self.tensorboard_writer = None
        if self.config.use_tensorboard:
            # Set Tensorboard log directory
            if self.config.tensorboard_log_dir:
                tb_log_dir = Path(self.config.tensorboard_log_dir)
            else:
                tb_log_dir = Path(self.config.output_path) / "tensorboard"
            
            # Add timestamp and run ID to log directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            tb_log_dir = tb_log_dir / f"run_{self.config.run_id}_{timestamp}"
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize SummaryWriter
            self.tensorboard_writer = SummaryWriter(log_dir=str(tb_log_dir))
            self.logger.info(f"Tensorboard logs saved to: {tb_log_dir}")
            
            # Record configuration information
            config_text = str(self.config).replace(',', '\n')
            self.tensorboard_writer.add_text("Config", config_text, 0)
        else:
            self.logger.info("Tensorboard logging disabled")
            
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        # 🔧 New: 记录训练配置信息
        if self.config.critic_warmup_steps > 0:
            self.logger.info(f"🔥 Critic warmup enabled: {self.config.critic_warmup_steps} steps")
            self.logger.info(f"   - Phase 1: Steps 0-{self.config.critic_warmup_steps-1} (Critic only)")
            self.logger.info(f"   - Phase 2: Steps {self.config.critic_warmup_steps}+ (Joint training)")
        else:
            self.logger.info("🔥 Joint actor-critic training from start (no warmup)")
        
        for epoch in range(self.config.train_epochs):
            self.logger.info(f"Starting epoch {epoch}")
            
            # For each epoch, perform multiple samples
            for sample_idx in range(self.config.num_syn_samples):
                self._train_epoch(epoch, sample_idx)
                
            # Save model and evaluate (every epoch by default)
            self._save_checkpoint(epoch, save_type="epoch")
            self._evaluate(epoch)
            
    def _train_epoch(self, epoch: int, sample_idx: int):
        """Train one epoch"""
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}, Sample {sample_idx}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Process batch data
                batch = tuple(t.to(self.config.device) for t in batch)
                # DataLoader returns (source_ids, source_mask, target_ids, target_mask, indices)
                source_ids, source_mask, target_ids, target_mask, ind = batch
                # Generate code
                response_ids = self._generate_code(source_ids, source_mask)
                response_ids_ref = self._generate_code_ref(source_ids, source_mask)

                response_mask = self._get_response_mask(response_ids)

                # Compute reward
                reward, metrics = self._compute_reward(response_ids, response_ids_ref, target_ids)

                
                # 🔧 新增：保存模型输出状态到文件
                self._save_model_output_state(epoch, sample_idx, batch_idx, source_ids, response_ids, response_ids_ref, reward, metrics)
                
                # Update statistics
                self._update_stats(reward, metrics, len(source_ids))
                
                # PPO training step
                train_stats = self.ppo_trainer.step(
                    source_ids, source_mask, response_ids, response_ids_ref, 
                    reward.to(self.config.device), response_mask.to(self.config.device)
                )
                
                # Update progress bar
                avg_errors = self.training_stats['total_nerrors']/self.training_stats['total_seen']
                avg_reward = self.training_stats['total_rewards']/self.training_stats['total_seen']
                
                # 🔧 New: 添加训练阶段信息到进度条
                if self.config.critic_warmup_steps > 0:
                    if self.training_stats['nsteps'] < self.config.critic_warmup_steps:
                        phase_info = f"[Critic Warmup {self.training_stats['nsteps']}/{self.config.critic_warmup_steps}]"
                    else:
                        phase_info = "[Joint Training]"
                else:
                    phase_info = ""
                
                pbar.set_description(
                    f"Epoch {epoch}, Sample {sample_idx} {phase_info}, "
                    f"Avg Errors: {avg_errors:.3f}, Avg Reward: {avg_reward:.3f}"
                )
                
                # Record training statistics
                self._log_training_step(epoch, sample_idx, batch_idx, reward, metrics, train_stats)
                
                self.training_stats['nsteps'] += 1
                
                # 🔧 New: Step-based saving
                if (self.config.save_every_n_steps > 0 and 
                    self.training_stats['nsteps'] % self.config.save_every_n_steps == 0):
                    self._save_checkpoint(epoch, save_type="step", step=self.training_stats['nsteps'])
                
                # 🔧 New: Performance-based saving
                if self._should_save_best_model(metrics):
                    self._save_checkpoint(epoch, save_type="best", step=self.training_stats['nsteps'])
                    
            except Exception as e:
                self.logger.error(f"Error during training step: {e}")
                if self.config.save_on_error:
                    self._save_checkpoint(epoch, save_type="emergency", step=self.training_stats['nsteps'])
                raise e
        
    def _generate_code(self, source_ids: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        """Generate code"""
        full = respond_to_batch(
            self.model, source_ids, source_mask,
            max_target_length=self.config.max_target_length,
            top_k=self.config.action_space, top_p=1.0,
            tokenizer=self.tokenizer
        ).detach()
        # full contains [prompt | generated]; only keep generated part
        gen_start = source_ids.size(1)
        return torch.clone(full[:, gen_start:])  # [B, <=max_new_tokens]
        
    def _generate_code_ref(self, source_ids: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        """Generate reference code"""
        full = respond_to_batch(
            self.model_ref, source_ids, source_mask,
            max_target_length=self.config.max_target_length,
            top_k=self.config.action_space, top_p=1.0,
            tokenizer=self.tokenizer
        ).detach()
        # full contains [prompt | generated]; only keep generated part
        gen_start = source_ids.size(1)
        return torch.clone(full[:, gen_start:])  # [B, <=max_new_tokens]
        
    def _get_response_mask(self, response_ids: torch.Tensor) -> torch.Tensor:
        """
        Create mask for response sequences based on EOS token positions
        
        Args:
            response_ids: Response token sequences [batch_size, seq_len]
            
        Returns:
            mask: Mask tensor where 1 indicates valid positions, 0 indicates padding
        """
        batch_size, seq_len = response_ids.shape
        masks = torch.zeros_like(response_ids, dtype=torch.float)
        
        for i, seq in enumerate(response_ids):
            # Find EOS token position
            eos_pos = (seq == self.tokenizer.eos_token_id).nonzero()
            if len(eos_pos) > 0:
                eos_pos = eos_pos[0].item()
            else:
                eos_pos = seq_len - 1
            
            # Create mask: valid positions (including EOS) are 1, padding positions are 0
            valid_length = eos_pos + 1
            masks[i, :valid_length] = 1.0
        
        return masks
        
    def _compute_reward(self, response_ids: torch.Tensor, response_ids_ref: torch.Tensor, 
                       target_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute reward"""
        reward, reward_ref, mean_rate, mean_ast_match, mean_dfg_match, mean_rate_ref, mean_ast_match_ref, mean_dfg_match_ref, num_errors, num_errors_ref, num_nodes, num_nodes_ref, sample_details = self.get_reward_func(
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
            'mean_rate_ref': mean_rate_ref,
            'mean_ast_match_ref': mean_ast_match_ref,
            'mean_dfg_match_ref': mean_dfg_match_ref,
            'num_errors': num_errors,
            'num_errors_ref': num_errors_ref,
            'num_nodes': num_nodes,
            'num_nodes_ref': num_nodes_ref,
            'reward_ref': reward_ref,
            'sample_details': sample_details  # 新增：每个样本的详细信息
        }
        
        return reward, metrics
        
    def _update_stats(self, reward: torch.Tensor, metrics: Dict, batch_size: int):
        """Update training statistics"""
        self.training_stats['total_rewards'] += float(sum(reward.sum(axis=-1).tolist()))
        self.training_stats['total_nerrors'] += sum(metrics['num_errors'])
        self.training_stats['total_nnodes'] += sum(metrics['num_nodes'])
        self.training_stats['total_nerrors_ref'] += sum(metrics['num_errors_ref'])
        self.training_stats['total_nnodes_ref'] += sum(metrics['num_nodes_ref'])
        self.training_stats['total_seen'] += batch_size
        
        # Add ref rewards statistics
        if 'total_rewards_ref' not in self.training_stats:
            self.training_stats['total_rewards_ref'] = 0
        self.training_stats['total_rewards_ref'] += float(sum(metrics['reward_ref'].sum(axis=-1).tolist()))
        
    def _log_training_step(self, epoch: int, sample_idx: int, batch_idx: int,
                          reward: torch.Tensor, metrics: Dict, train_stats: Dict):
        """Record training step"""
        # Calculate average metrics
        avg_reward = float(sum(reward.sum(axis=-1).tolist())) / len(reward)
        avg_reward_ref = float(sum(metrics['reward_ref'].sum(axis=-1).tolist())) / len(metrics['reward_ref'])
        avg_errors = sum(metrics['num_errors']) / len(metrics['num_errors'])
        avg_errors_ref = sum(metrics['num_errors_ref']) / len(metrics['num_errors_ref'])
        avg_nodes = sum(metrics['num_nodes']) / len(metrics['num_nodes'])
        avg_nodes_ref = sum(metrics['num_nodes_ref']) / len(metrics['num_nodes_ref'])
        
        # 🔧 New: Record to Tensorboard
        if (self.tensorboard_writer and 
            self.training_stats['nsteps'] % self.config.log_every_n_steps == 0):
            
            global_step = self.training_stats['nsteps']
            
            # Reward-related metrics
            self.tensorboard_writer.add_scalar("Training/Average_Reward", avg_reward, global_step)
            self.tensorboard_writer.add_scalar("Training/Average_Reward_Ref", avg_reward_ref, global_step)
            self.tensorboard_writer.add_scalar("Training/Compilation_Success_Rate", metrics['mean_rate'], global_step)
            self.tensorboard_writer.add_scalar("Training/Compilation_Success_Rate_Ref", metrics['mean_rate_ref'], global_step)
            self.tensorboard_writer.add_scalar("Training/AST_Match_Score", metrics['mean_ast_match'], global_step)
            self.tensorboard_writer.add_scalar("Training/AST_Match_Score_Ref", metrics['mean_ast_match_ref'], global_step)
            self.tensorboard_writer.add_scalar("Training/DFG_Match_Score", metrics['mean_dfg_match'], global_step)
            self.tensorboard_writer.add_scalar("Training/DFG_Match_Score_Ref", metrics['mean_dfg_match_ref'], global_step)
            
            # Comparison metrics: Policy vs Ref
            self.tensorboard_writer.add_scalar("Comparison/Reward_Difference", avg_reward - avg_reward_ref, global_step)
            self.tensorboard_writer.add_scalar("Comparison/Compilation_Rate_Difference", metrics['mean_rate'] - metrics['mean_rate_ref'], global_step)
            self.tensorboard_writer.add_scalar("Comparison/AST_Match_Difference", metrics['mean_ast_match'] - metrics['mean_ast_match_ref'], global_step)
            self.tensorboard_writer.add_scalar("Comparison/DFG_Match_Difference", metrics['mean_dfg_match'] - metrics['mean_dfg_match_ref'], global_step)
            self.tensorboard_writer.add_scalar("Comparison/Error_Difference", avg_errors - avg_errors_ref, global_step)
            
            # Code quality metrics
            self.tensorboard_writer.add_scalar("Code_Quality/Avg_Errors", avg_errors, global_step)
            self.tensorboard_writer.add_scalar("Code_Quality/Avg_Errors_Ref", avg_errors_ref, global_step)
            self.tensorboard_writer.add_scalar("Code_Quality/Avg_Nodes", avg_nodes, global_step)
            self.tensorboard_writer.add_scalar("Code_Quality/Avg_Nodes_Ref", avg_nodes_ref, global_step)

            self.tensorboard_writer.add_scalar("Code_Quality/Avg_NonScore_Rwd", train_stats['ppo/mean_non_score_reward'], global_step)
            self.tensorboard_writer.add_scalar("Code_Quality/Avg_Score", train_stats['ppo/mean_score_reward'], global_step)
            
            # PPO training metrics
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
            
            # 🔧 New: Advantages详细统计信息
            # 原始advantages (标准化前)
            if 'ppo/advantages_raw_mean' in train_stats:
                self.tensorboard_writer.add_scalar("PPO/Advantages_Raw_Mean", train_stats['ppo/advantages_raw_mean'], global_step)
            if 'ppo/advantages_raw_std' in train_stats:
                self.tensorboard_writer.add_scalar("PPO/Advantages_Raw_Std", train_stats['ppo/advantages_raw_std'], global_step)
            if 'ppo/advantages_raw_max' in train_stats:
                self.tensorboard_writer.add_scalar("PPO/Advantages_Raw_Max", train_stats['ppo/advantages_raw_max'], global_step)
            if 'ppo/advantages_raw_min' in train_stats:
                self.tensorboard_writer.add_scalar("PPO/Advantages_Raw_Min", train_stats['ppo/advantages_raw_min'], global_step)
            
            # 标准化后advantages (用于训练)
            if 'ppo/advantages_normalized_mean' in train_stats:
                self.tensorboard_writer.add_scalar("PPO/Advantages_Normalized_Mean", train_stats['ppo/advantages_normalized_mean'], global_step)
            if 'ppo/advantages_normalized_std' in train_stats:
                self.tensorboard_writer.add_scalar("PPO/Advantages_Normalized_Std", train_stats['ppo/advantages_normalized_std'], global_step)
            if 'ppo/advantages_normalized_max' in train_stats:
                self.tensorboard_writer.add_scalar("PPO/Advantages_Normalized_Max", train_stats['ppo/advantages_normalized_max'], global_step)
            if 'ppo/advantages_normalized_min' in train_stats:
                self.tensorboard_writer.add_scalar("PPO/Advantages_Normalized_Min", train_stats['ppo/advantages_normalized_min'], global_step)
            
            if 'ppo/returns/mean' in train_stats:
                self.tensorboard_writer.add_scalar("PPO/Returns_Mean", train_stats['ppo/returns/mean'].item(), global_step)
            if 'ppo/val/mean' in train_stats:
                self.tensorboard_writer.add_scalar("PPO/Value_Mean", train_stats['ppo/val/mean'].item(), global_step)
            
            # Learning rate (if available)
            try:
                current_lr = self.ppo_trainer.optimizer.param_groups[0]['lr']
                self.tensorboard_writer.add_scalar("Training/Learning_Rate", current_lr, global_step)
            except:
                pass
            
            # 🔧 New: 训练阶段标识
            if self.config.critic_warmup_steps > 0:
                is_warmup_phase = self.training_stats['nsteps'] < self.config.critic_warmup_steps
                self.tensorboard_writer.add_scalar("Training/Is_Critic_Warmup_Phase", float(is_warmup_phase), global_step)
                self.tensorboard_writer.add_scalar("Training/Warmup_Progress", 
                                                 min(self.training_stats['nsteps'] / self.config.critic_warmup_steps, 1.0), global_step)
        
        # Record to CSV file
        # 🔧 New: 添加训练阶段信息
        training_phase = "warmup" if (self.config.critic_warmup_steps > 0 and 
                                     self.training_stats['nsteps'] < self.config.critic_warmup_steps) else "joint"
        
        csv_line = [
            datetime.datetime.now().strftime("%H:%M:%S"),
            str(self.config.run_id),
            str(self.config.train_batch_size),
            str(self.config.max_source_length),
            str(self.config.max_target_length),
            str(self.config.learning_rate),
            str(epoch),
            str(self.training_stats['nsteps']),
            training_phase,  # 🔧 New: 训练阶段标识
            f"{avg_reward:.4f}",
            f"{avg_reward_ref:.4f}",
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
            str(train_stats.get('ppo/advantages_raw_mean', 0.0)),
            str(train_stats.get('ppo/advantages_raw_std', 0.0)),
            str(train_stats.get('ppo/advantages_raw_max', 0.0)),
            str(train_stats.get('ppo/advantages_raw_min', 0.0)),
            str(train_stats.get('ppo/advantages_normalized_mean', 0.0)),
            str(train_stats.get('ppo/advantages_normalized_std', 0.0)),
            str(train_stats.get('ppo/advantages_normalized_max', 0.0)),
            str(train_stats.get('ppo/advantages_normalized_min', 0.0)),
            str(train_stats['ppo/returns/mean'].item()),
            str(train_stats['ppo/val/mean'].item()),
            str(metrics['mean_rate']),
            str(metrics['mean_rate_ref']),
            str(metrics['mean_ast_match']),
            str(metrics['mean_ast_match_ref']),
            str(metrics['mean_dfg_match']),
            str(metrics['mean_dfg_match_ref'])
        ]
        
        csv_file = self.results_dir / f"{self.config.source_lang}-{self.config.target_lang}.csv"
        with open(csv_file, 'a') as f:
            f.write(','.join(csv_line) + '\n')
            
    def _save_checkpoint(self, epoch: int, save_type: str = "epoch", step: int = None):
        """Save checkpoint with flexible naming and conditions - 与Qwen2.5-Coder保持一致"""
        # Skip if this step was already saved
        if step is not None and step == self.last_save_step:
            return
            
        # Generate checkpoint directory name based on save type
        if save_type == "epoch":
            checkpoint_dir = self.checkpoint_dir / f"checkpoint-epoch-{epoch}"
        elif save_type == "step":
            checkpoint_dir = self.checkpoint_dir / f"checkpoint-step-{step}"
        elif save_type == "best":
            checkpoint_dir = self.checkpoint_dir / f"checkpoint-best-{self.config.save_metric}"
        elif save_type == "emergency":
            checkpoint_dir = self.checkpoint_dir / f"checkpoint-emergency-step-{step}"
        else:
            checkpoint_dir = self.checkpoint_dir / f"checkpoint-{save_type}-ep{epoch}-step{step}"
        
        # Create checkpoint directory
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 🔧 修改：保存完整的模型目录，与Qwen2.5-Coder保持一致
        try:
            # 获取要保存的模型（处理DataParallel）
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            
            # 保存模型（使用Hugging Face的标准保存方式）
            model_to_save.save_pretrained(checkpoint_dir)
            
            # 保存tokenizer
            self.tokenizer.save_pretrained(checkpoint_dir)
            
            # 保存训练配置信息
            config_info = {
                "epoch": epoch,
                "step": step,
                "save_type": save_type,
                "training_config": self.config.__dict__,
                "model_path": str(self.config.model_path),
                "save_time": datetime.datetime.now().isoformat()
            }
            
            config_file = checkpoint_dir / "training_info.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_info, f, indent=2, ensure_ascii=False)
            
            # Update last save step
            if step is not None:
                self.last_save_step = step
                
            self.logger.info(f"✅ 完整模型保存到: {checkpoint_dir} (类型: {save_type})")
            
        except Exception as e:
            self.logger.error(f"❌ 保存模型失败: {e}")
            # 如果完整保存失败，尝试只保存权重作为备用
            try:
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                weights_path = checkpoint_dir / "pytorch_model.bin"
                torch.save(model_to_save.state_dict(), weights_path)
                self.logger.info(f"⚠️  备用权重保存到: {weights_path}")
            except Exception as e2:
                self.logger.error(f"❌ 备用保存也失败: {e2}")
                return
        
        # 🔧 New: Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints, only keep the latest N - 适配新的目录保存方式"""
        if self.config.max_checkpoints <= 0:
            return  # Do not limit checkpoint number
            
        # Get all checkpoint directories
        checkpoint_patterns = [
            "checkpoint-epoch-*",
            "checkpoint-step-*", 
            "checkpoint-best-*",
            "checkpoint-emergency-*"
        ]
        
        checkpoint_dirs = []
        for pattern in checkpoint_patterns:
            checkpoint_dirs.extend(list(self.checkpoint_dir.glob(pattern)))
        
        if len(checkpoint_dirs) <= self.config.max_checkpoints:
            return  # Number not exceeded
            
        # Sort by modification time (latest first)
        checkpoint_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Delete old directories exceeding limit
        dirs_to_delete = checkpoint_dirs[self.config.max_checkpoints:]
        
        for dir_path in dirs_to_delete:
            try:
                shutil.rmtree(dir_path)
                self.logger.info(f"🗑️  删除旧checkpoint目录: {dir_path}")
            except Exception as e:
                self.logger.warning(f"⚠️  删除checkpoint目录失败 {dir_path}: {e}")
                
        if dirs_to_delete:
            self.logger.info(f"🧹 清理了 {len(dirs_to_delete)} 个旧checkpoint目录，保留最新的 {self.config.max_checkpoints} 个")
                
    def _evaluate(self, epoch: int):
        """Evaluate model"""
        self.model.eval()
        self.logger.info(f"Starting epoch {epoch} evaluation")
        
        # Train set evaluation
        train_errors, train_errors_ref = self._evaluate_dataset(
            epoch, self.train_features, self.train_dataloader, 'train'
        )
        self.model.train()
        
        # Test set evaluation
        test_errors, test_errors_ref = self._evaluate_dataset(
            epoch, self.test_features, self.test_dataloader, 'test'
        )
        self.model.train()
        
        self.logger.info(f"Epoch {epoch} evaluation results:")
        self.logger.info(f"  Train - Model errors: {train_errors}, Ref model errors: {train_errors_ref}")
        self.logger.info(f"  Test - Model errors: {test_errors}, Ref model errors: {test_errors_ref}")
        
        # 🔧 New: Record evaluation metrics to Tensorboard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar("Evaluation/Train_Errors", train_errors, epoch)
            self.tensorboard_writer.add_scalar("Evaluation/Train_Errors_Ref", train_errors_ref, epoch)
            self.tensorboard_writer.add_scalar("Evaluation/Test_Errors", test_errors, epoch)
            self.tensorboard_writer.add_scalar("Evaluation/Test_Errors_Ref", test_errors_ref, epoch)
            
            # Calculate error rate
            if len(self.train_features) > 0:
                train_error_rate = train_errors / len(self.train_features)
                self.tensorboard_writer.add_scalar("Evaluation/Train_Error_Rate", train_error_rate, epoch)
            
            if len(self.test_features) > 0:
                test_error_rate = test_errors / len(self.test_features)
                self.tensorboard_writer.add_scalar("Evaluation/Test_Error_Rate", test_error_rate, epoch)
        self.model.train()
            
    def _evaluate_dataset(self, epoch: int, features: List[InputFeatures], 
                         dataloader: DataLoader, prefix: str) -> Tuple[int, int]:
        """Evaluate dataset"""
        pred_ids = []
        pred_ids_ref = []
        indices = []
        nerrors = 0
        nerrors_ref = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = tuple(t.to(self.config.device) for t in batch)
                source_ids, source_mask, target_ids, target_mask, ind = batch
                
                # Generate predictions
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
                
                # Calculate number of errors
                reward_result = self.get_reward_func(
                    lang=self.config.target_lang,
                    code_ids=preds,
                    code_ref_ids=preds_ref,
                    gold_ids=target_ids,
                    tokenizer=self.tokenizer
                )
                nerrors += sum(reward_result[8])  # num_errors in the 8th position
                
                # For ref errors, we can directly use the same result in num_errors_ref
                nerrors_ref += sum(reward_result[9])  # num_errors_ref in the 9th position
                
                # Save predictions
                pred_ids.extend(list(preds.cpu().numpy()))
                pred_ids_ref.extend(list(preds_ref.cpu().numpy()))
                indices.extend(list(ind.cpu().numpy()))
                
        # Decode and save results
        self._save_predictions(epoch, prefix, pred_ids, pred_ids_ref, indices, features)
        
        return nerrors, nerrors_ref
        
    def _save_predictions(self, epoch: int, prefix: str, pred_ids: List, 
                         pred_ids_ref: List, indices: List, features: List[InputFeatures]):
        """Save predictions"""
        # Decode predictions
        raw_predictions = [
            self.tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for id in pred_ids
        ]
        raw_predictions_ref = [
            self.tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for id in pred_ids_ref
        ]
        
        # Extract code from Qwen response
        predictions = [
            extract_code_from_qwen_response(pred, self.config.target_lang)
            for pred in raw_predictions
        ]
        predictions_ref = [
            extract_code_from_qwen_response(pred, self.config.target_lang)
            for pred in raw_predictions_ref
        ]
        
        # Save to file
        model_file = self.checkpoint_dir / f"{prefix}.model_ep{epoch}"
        ref_file = self.checkpoint_dir / f"{prefix}.model_ref_ep{epoch}"
        gold_file = self.checkpoint_dir / f"{prefix}.gold_ep{epoch}"
        
        with open(model_file, 'w') as f_model, \
             open(ref_file, 'w') as f_ref, \
             open(gold_file, 'w') as f_gold:
            
            for pred, ref, i in zip(predictions, predictions_ref, indices):
                f_model.write(pred + '\n')
                f_ref.write(ref + '\n')
                # For gold, also extract code
                gold_code = extract_code_from_qwen_response(features[i].target, self.config.target_lang)
                f_gold.write(gold_code + '\n')

    def _save_model_output_state(self, epoch: int, sample_idx: int, batch_idx: int, 
                                source_ids: torch.Tensor, response_ids: torch.Tensor, 
                                response_ids_ref: torch.Tensor, reward: torch.Tensor, metrics: Dict):
        """保存模型输出状态到文件，用于调试和分析 - 每步都保存但每20步保存为一个文件"""
        try:
            # 🔧 修改：每步都保存，但每20步保存为一个文件
            steps_per_file = 20
            
            # 创建输出目录
            output_dir = Path(self.config.output_path) / "model_outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 计算当前应该保存到哪个文件
            file_batch_idx = batch_idx // steps_per_file
            step_in_file = batch_idx % steps_per_file
            
            # 创建文件名 - 每20步一个文件
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_output_ep{epoch}_sample{sample_idx}_batch{file_batch_idx*steps_per_file}-{(file_batch_idx+1)*steps_per_file-1}_{timestamp}.json"
            filepath = output_dir / filename
            
            # 检查是否需要创建新文件或追加到现有文件
            if step_in_file == 0:
                # 新文件，创建完整的输出数据结构
                output_data = {
                    "metadata": {
                        "epoch": epoch,
                        "sample_idx": sample_idx,
                        "file_batch_range": f"{file_batch_idx*steps_per_file}-{(file_batch_idx+1)*steps_per_file-1}",
                        "timestamp": timestamp,
                        "run_id": self.config.run_id,
                        "source_lang": self.config.source_lang,
                        "target_lang": self.config.target_lang,
                        "steps_per_file": steps_per_file,
                        "total_steps_in_file": 0
                    },
                    "steps": []
                }
            else:
                # 追加到现有文件
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        output_data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    # 如果文件不存在或损坏，创建新的
                    output_data = {
                        "metadata": {
                            "epoch": epoch,
                            "sample_idx": sample_idx,
                            "file_batch_range": f"{file_batch_idx*steps_per_file}-{(file_batch_idx+1)*steps_per_file-1}",
                            "timestamp": timestamp,
                            "run_id": self.config.run_id,
                            "source_lang": self.config.source_lang,
                            "target_lang": self.config.target_lang,
                            "steps_per_file": steps_per_file,
                            "total_steps_in_file": 0
                        },
                        "steps": []
                    }
            
            # 解码token序列为文本
            source_texts = []
            response_texts = []
            response_ref_texts = []
            
            for i in range(source_ids.shape[0]):
                # 解码source_ids（去掉padding）
                source_tokens = source_ids[i].cpu().numpy()
                source_text = self.tokenizer.decode(source_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                source_texts.append(source_text)
                
                # 解码response_ids
                response_tokens = response_ids[i].cpu().numpy()
                response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                response_texts.append(response_text)
                
                # 解码response_ids_ref
                response_ref_tokens = response_ids_ref[i].cpu().numpy()
                response_ref_text = self.tokenizer.decode(response_ref_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                response_ref_texts.append(response_ref_text)
            
            # 提取代码块
            extracted_codes = [extract_code_from_qwen_response(text, self.config.target_lang) for text in response_texts]
            extracted_codes_ref = [extract_code_from_qwen_response(text, self.config.target_lang) for text in response_ref_texts]
            
            # 🔧 修改：准备当前步骤的数据
            step_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            step_data = {
                "step_info": {
                    "batch_idx": batch_idx,
                    "step_in_file": step_in_file,
                    "timestamp": step_timestamp,
                    "batch_size": source_ids.shape[0],
                    "source_length": source_ids.shape[1],
                    "response_length": response_ids.shape[1],
                    "response_ref_length": response_ids_ref.shape[1]
                },
                "rewards": {
                    "reward_values": reward.cpu().numpy().tolist(),
                    "mean_reward": float(reward.mean().cpu().numpy().item()),
                    "total_reward": float(reward.sum().cpu().numpy().item())
                },
                "metrics": {
                    "mean_rate": float(metrics['mean_rate']) if isinstance(metrics['mean_rate'], (int, float, np.number)) else float(metrics['mean_rate'].item()),
                    "mean_ast_match": float(metrics['mean_ast_match']) if isinstance(metrics['mean_ast_match'], (int, float, np.number)) else float(metrics['mean_ast_match'].item()),
                    "mean_dfg_match": float(metrics['mean_dfg_match']) if isinstance(metrics['mean_dfg_match'], (int, float, np.number)) else float(metrics['mean_dfg_match'].item()),
                    "mean_rate_ref": float(metrics['mean_rate_ref']) if isinstance(metrics['mean_rate_ref'], (int, float, np.number)) else float(metrics['mean_rate_ref'].item()),
                    "mean_ast_match_ref": float(metrics['mean_ast_match_ref']) if isinstance(metrics['mean_ast_match_ref'], (int, float, np.number)) else float(metrics['mean_ast_match_ref'].item()),
                    "mean_dfg_match_ref": float(metrics['mean_dfg_match_ref']) if isinstance(metrics['mean_dfg_match_ref'], (int, float, np.number)) else float(metrics['mean_dfg_match_ref'].item()),
                    "num_errors": metrics['num_errors'],
                    "num_errors_ref": metrics['num_errors_ref'],
                    "num_nodes": metrics['num_nodes'],
                    "num_nodes_ref": metrics['num_nodes_ref']
                },
                "samples": []
            }
            
            # 为每个样本添加详细信息
            for i in range(source_ids.shape[0]):
                # 安全地获取reward值
                if reward.shape[0] > i:
                    reward_value = float(reward[i].cpu().numpy().item()) if reward[i].numel() == 1 else float(reward[i].mean().cpu().numpy().item())
                else:
                    reward_value = 0.0
                
                # 获取每个样本的具体编译成功信息
                sample_details = metrics.get('sample_details', {})
                compilation_success = sample_details.get('compilation_success', [False] * source_ids.shape[0])
                ast_match = sample_details.get('ast_match', [0.0] * source_ids.shape[0])
                dfg_match = sample_details.get('dfg_match', [0.0] * source_ids.shape[0])
                compilation_success_ref = sample_details.get('compilation_success_ref', [False] * source_ids.shape[0])
                ast_match_ref = sample_details.get('ast_match_ref', [0.0] * source_ids.shape[0])
                dfg_match_ref = sample_details.get('dfg_match_ref', [0.0] * source_ids.shape[0])
                
                sample_data = {
                    "sample_id": i,
                    "source": {
                        "text": source_texts[i],
                        "length": len(source_ids[i])
                    },
                    "response": {
                        "text": response_texts[i],
                        "length": len(response_ids[i]),
                        "extracted_code": extracted_codes[i],
                        "code_length": len(extracted_codes[i])
                    },
                    "response_ref": {
                        "text": response_ref_texts[i],
                        "length": len(response_ids_ref[i]),
                        "extracted_code": extracted_codes_ref[i],
                        "code_length": len(extracted_codes_ref[i])
                    },
                    "reward": {
                        "value": reward_value,
                        "compilation_success": compilation_success[i] if i < len(compilation_success) else False,
                        "ast_match": ast_match[i] if i < len(ast_match) else 0.0,
                        "dfg_match": dfg_match[i] if i < len(dfg_match) else 0.0
                    },
                    "reward_ref": {
                        "compilation_success": compilation_success_ref[i] if i < len(compilation_success_ref) else False,
                        "ast_match": ast_match_ref[i] if i < len(ast_match_ref) else 0.0,
                        "dfg_match": dfg_match_ref[i] if i < len(dfg_match_ref) else 0.0
                    }
                }
                step_data["samples"].append(sample_data)
            
            # 🔧 修改：将当前步骤添加到输出数据中
            output_data["steps"].append(step_data)
            output_data["metadata"]["total_steps_in_file"] = len(output_data["steps"])
            
            # 保存到文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            # 🔧 修改：同时创建一个简化的日志文件，记录关键信息
            log_filename = f"model_output_summary_ep{epoch}_sample{sample_idx}_batch{file_batch_idx*steps_per_file}-{(file_batch_idx+1)*steps_per_file-1}.txt"
            log_filepath = output_dir / log_filename
            
            with open(log_filepath, 'a', encoding='utf-8') as f:
                f.write(f"=== Step {step_in_file+1}/{steps_per_file} (Batch {batch_idx}) at {step_timestamp} ===\n")
                f.write(f"Mean Reward: {step_data['rewards']['mean_reward']:.4f}\n")
                f.write(f"Compilation Rate: {step_data['metrics']['mean_rate']:.4f}\n")
                f.write(f"AST Match: {step_data['metrics']['mean_ast_match']:.4f}\n")
                f.write(f"DFG Match: {step_data['metrics']['mean_dfg_match']:.4f}\n")
                f.write(f"Errors: {sum(step_data['metrics']['num_errors'])}\n")
                f.write(f"Nodes: {sum(step_data['metrics']['num_nodes'])}\n")
                f.write("-" * 50 + "\n")
                
                # 为每个样本添加简要信息
                for i, sample in enumerate(step_data["samples"]):
                    f.write(f"Sample {i}:\n")
                    f.write(f"  Source length: {sample['source']['length']} tokens\n")
                    f.write(f"  Response length: {sample['response']['length']} tokens\n")
                    f.write(f"  Extracted code length: {sample['response']['code_length']} chars\n")
                    f.write(f"  Reward: {sample['reward']['value']:.4f}\n")
                    f.write(f"  Compilation: {'✅' if sample['reward']['compilation_success'] else '❌'}\n")
                    f.write(f"  AST Match: {sample['reward']['ast_match']:.4f}\n")
                    f.write(f"  DFG Match: {sample['reward']['dfg_match']:.4f}\n")
                    f.write(f"  Ref Compilation: {'✅' if sample['reward_ref']['compilation_success'] else '❌'}\n")
                    f.write(f"  Code preview: {sample['response']['extracted_code'][:100]}...\n")
                    f.write("\n")
            
            # 🔧 修改：记录到日志，每20步记录一次
            if step_in_file == 0:  # 每20步记录一次，避免日志过多
                self.logger.info(f"📝 模型输出状态已保存: {filename} (包含20个步骤)")
            else:
                self.logger.debug(f"📝 步骤 {step_in_file+1}/20 已添加到文件: {filename}")
                
        except Exception as e:
            self.logger.warning(f"⚠️  保存模型输出状态失败: {e}")
            # 不抛出异常，避免影响训练流程

    def _should_save_best_model(self, metrics: Dict) -> bool:
        """Check if we should save the best model based on performance metrics"""
        if not self.config.save_best_only:
            return False
            
        current_metric = self.config.save_metric
        if current_metric not in metrics:
            self.logger.warning(f"Save metric '{current_metric}' not found in metrics: {list(metrics.keys())}")
            return False
            
        current_value = metrics[current_metric]
        best_value = self.best_metrics[current_metric]
        
        # Check if current performance is better than best
        improvement = current_value - best_value
        if improvement > self.config.save_threshold:
            self.best_metrics[current_metric] = current_value
            self.logger.info(f"New best {current_metric}: {current_value:.4f} (improvement: {improvement:.4f})")
            return True
            
        return False


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Qwen2.5-Coder PPO code generation training program")
    
    # Required arguments
    parser.add_argument("--source_lang", required=True, type=str,
                       help="Source code language")
    parser.add_argument("--target_lang", required=True, type=str,
                       help="Target code language")
    parser.add_argument("--model_path", required=True, type=str,
                       help="Qwen2.5-Coder model path")
    parser.add_argument("--data_path", required=True, type=str,
                       help="Qwen format data directory path")
    parser.add_argument("--output_path", required=True, type=str,
                       help="Output directory path")
    
    # Optional arguments
    parser.add_argument("--max_source_length", default=400, type=int,
                       help="Maximum source code length")
    parser.add_argument("--max_target_length", default=400, type=int,
                       help="Maximum target code length")
    parser.add_argument("--train_batch_size", default=16, type=int,
                       help="Training batch size")
    parser.add_argument("--test_batch_size", default=48, type=int,
                       help="Test batch size")
    parser.add_argument("--minibatch_size", default=1, type=int,
                       help="Minibatch size")
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int,
                       help="Gradient accumulation steps (default: 4, effective_batch = train_batch_size * gradient_accumulation_steps)")
    parser.add_argument("--critic_warmup_steps", default=0, type=int,
                       help="Critic warmup steps - train only critic for N steps before joint training (default: 0, recommended: 50-100)")
    parser.add_argument("--train_epochs", default=1000000, type=int,
                       help="Training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--kl_coef", type=float, default=0.05,
                       help="KL coefficient")
    parser.add_argument("--kl_target", type=float, default=1.0,
                       help="KL target value")
    parser.add_argument("--vf_coef", type=float, default=1e-3,
                       help="Value function coefficient")
    parser.add_argument("--action_space", default=2, type=int,
                       help="Action space size (top_k)")
    parser.add_argument("--num_syn_samples", default=5, type=int,
                       help="Number of samples per epoch")
    parser.add_argument("--run_id", default=1, type=int,
                       help="Run ID")
    parser.add_argument("--seed", default=42, type=int,
                       help="Random seed")
    
    # 🔧 New: Checkpoint saving control parameters
    parser.add_argument("--save_every_n_steps", default=0, type=int,
                       help="Save checkpoint every N training steps (0 means disabled, default: 0)")
    parser.add_argument("--max_checkpoints", default=10, type=int,
                       help="Maximum number of checkpoints to retain (default: 10)")
    
    # 🔧 New: Performance-based saving parameters
    parser.add_argument("--save_best_only", action="store_true", default=False,
                       help="Only save when performance improves (default: False)")
    parser.add_argument("--save_metric", default="reward", type=str,
                       choices=["reward", "compilation_rate", "ast_match", "dfg_match"],
                       help="Metric to track for best model saving (default: reward)")
    parser.add_argument("--save_threshold", default=0.0, type=float,
                       help="Minimum improvement threshold for saving (default: 0.0)")
    
    # 🔧 New: Emergency saving parameter
    parser.add_argument("--save_on_error", action="store_true", default=True,
                       help="Save checkpoint when training error occurs (default: True)")
    parser.add_argument("--no_save_on_error", action="store_false", dest="save_on_error",
                       help="Disable emergency saving on error")
    
    # 🔧 New: Tensorboard support parameters
    parser.add_argument("--use_tensorboard", action="store_true", default=True,
                       help="Enable Tensorboard logging (default: enabled)")
    parser.add_argument("--no_tensorboard", action="store_false", dest="use_tensorboard",
                       help="Disable Tensorboard logging")
    parser.add_argument("--tensorboard_log_dir", default=None, type=str,
                       help="Tensorboard log directory (default: output_path/tensorboard)")
    parser.add_argument("--log_every_n_steps", default=1, type=int,
                       help="Record metrics to Tensorboard every N training steps (default: every step)")
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Create configuration object
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
        minibatch_size=args.minibatch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        critic_warmup_steps=args.critic_warmup_steps,
        train_epochs=args.train_epochs,
        learning_rate=args.learning_rate,
        kl_coef=args.kl_coef,
        kl_target=args.kl_target,
        vf_coef=args.vf_coef,
        action_space=args.action_space,
        num_syn_samples=args.num_syn_samples,
        run_id=args.run_id,
        seed=args.seed,
        save_every_n_steps=args.save_every_n_steps,
        max_checkpoints=args.max_checkpoints,
        save_best_only=args.save_best_only,
        save_metric=args.save_metric,
        save_threshold=args.save_threshold,
        save_on_error=args.save_on_error,
        use_tensorboard=args.use_tensorboard,
        tensorboard_log_dir=args.tensorboard_log_dir,
        log_every_n_steps=args.log_every_n_steps
    )
    
    print("=" * 60)
    print("🚀 Qwen2.5-Coder PPO code translation training program")
    print("=" * 60)
    print(f"📝 Source language: {config.source_lang}")
    print(f"🎯 Target language: {config.target_lang}")
    print(f"🤖 Model path: {config.model_path}")
    print(f"📂 Data path: {config.data_path}")
    print(f"💾 Output path: {config.output_path}")
    print(f"🔧 Device: {config.device}")
    print("=" * 60)
    
    # Create trainer and start training
    trainer = CodeTranslationTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main() 