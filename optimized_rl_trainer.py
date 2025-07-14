#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的PPO代码生成强化学习训练程序

主要功能：
1. 代码翻译任务的PPO训练
2. 支持多种编程语言对
3. 基于编译成功率和代码结构的奖励计算
4. 自适应KL控制和策略裁剪
5. 详细的训练监控和日志记录

作者：AI Assistant
版本：1.0
"""

import os
import sys
import torch
import numpy as np
import datetime
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import json

# 项目内部导入
from code_parser import (DFG_python, DFG_java, DFG_ruby, DFG_go, 
                        DFG_php, DFG_javascript, DFG_csharp)
from code_parser import (tree_to_token_index, tree_to_token_nodes,
                        index_to_code_token, tree_to_variable_index, 
                        detokenize_code)
from tree_sitter import Language, Parser
from reward import remove_special_tokens, tree_sitter_full_compile, get_reward
from torch.utils.data import DataLoader, TensorDataset
from model import respond_to_batch, CodeT5HeadWithValueModelLocal
from transformers import AutoTokenizer
from ppo import PPOTrainer
from utils import (extract_structure, Example, read_examples, 
                  convert_examples_to_features, InputFeatures)
from code_prepro.lang_processors import (py_tokenizer, java_tokenizer, cpp_tokenizer,
                                        c_tokenizer, js_tokenizer, php_tokenizer, cs_tokenizer,
                                        py_detokenizer, java_detokenizer, cpp_detokenizer,
                                        c_detokenizer, js_detokenizer, php_detokenizer, cs_detokenizer)
from compiler.terminal_compiler import TerminalCompiler


@dataclass
class TrainingConfig:
    """训练配置数据类"""
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
    max_source_length: int = 300  # 减少输入长度，为输出留出空间
    max_target_length: int = 200  # 减少输出长度，确保总长度不超过512
    
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
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 运行配置
    run_id: int = 1
    seed: int = 42


class CodeTranslationTrainer:
    """代码翻译PPO训练器"""
    
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
        self.end_dict = {
            'javascript': 'js', 'java': 'java', 'c_sharp': 'cs', 
            'php': 'php', 'python': 'py', 'c': 'c', 'cpp': 'cpp'
        }
        
        # 代码处理器映射
        self.code_tokenizers = {
            "java": java_tokenizer, "cpp": cpp_tokenizer, "c": c_tokenizer, 
            "python": py_tokenizer, "javascript": js_tokenizer, 
            "php": php_tokenizer, "c_sharp": cs_tokenizer
        }
        self.code_detokenizers = {
            "java": java_detokenizer, "cpp": cpp_detokenizer, "c": c_detokenizer,
            "python": py_detokenizer, "javascript": js_detokenizer,
            "php": php_detokenizer, "c_sharp": cs_detokenizer
        }
        
        # 编译器映射
        self.lang2compiler = {
            "python": TerminalCompiler("Python"),
            "java": TerminalCompiler("Java"),
            "cpp": TerminalCompiler("C++"),
            "c_sharp": TerminalCompiler("C#"),
            "c": TerminalCompiler("C"),
            "php": TerminalCompiler("PHP"),
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
        
        # 初始化模型结构（不加载预训练权重）
        config_path = self.model_dir / 'config.json'
        
        # 加载主模型
        self.model = CodeT5HeadWithValueModelLocal(config_path)
        self.model.load_model_weights(self.config.model_path, self.config.device)
        self.model.to(self.config.device)
        
        # 加载参考模型（固定不变）
        self.model_ref = CodeT5HeadWithValueModelLocal(config_path)
        self.model_ref.load_model_weights(self.config.model_path, self.config.device)
        self.model_ref.to(self.config.device)
        
        # 从本地加载tokenizer
        print("正在从本地加载tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
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
        self.train_examples = read_examples(self.data_files['train'], self.config)
        self.dev_examples = read_examples(self.data_files['dev'], self.config)
        self.test_examples = read_examples(self.data_files['test'], self.config)
        
        # 转换为特征
        self.train_features = convert_examples_to_features(
            self.train_examples, self.tokenizer, self.config, stage='train'
        )
        self.dev_features = convert_examples_to_features(
            self.dev_examples, self.tokenizer, self.config, stage='train'
        )
        self.test_features = convert_examples_to_features(
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
        """构建数据文件路径"""
        l1, l2 = self.config.source_lang, self.config.target_lang
        
        # 尝试不同的路径组合
        possible_paths = [
            f"{self.config.data_path}/{self.dir_dict[l1]}-{self.dir_dict[l2]}/",
            f"{self.config.data_path}/{self.dir_dict[l2]}-{self.dir_dict[l1]}/"
        ]
        
        data_dir = None
        for path in possible_paths:
            if os.path.exists(path):
                data_dir = path
                break
                
        if data_dir is None:
            raise FileNotFoundError(f"找不到数据目录: {possible_paths}")
            
        # 构建文件名模板
        template = (f"{data_dir}train-{self.dir_dict[l1]}-{self.dir_dict[l2]}-tok."
                   f"{self.end_dict[l1]},{data_dir}train-{self.dir_dict[l1]}-{self.dir_dict[l2]}-tok."
                   f"{self.end_dict[l2]}")
        
        return {
            'train': template,
            'dev': template.replace('train', 'val'),
            'test': template.replace('train', 'test')
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
            source_ids, source_mask, target_ids, target_mask, _ = batch
            
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
        return torch.clone(respond_to_batch(
            self.model, source_ids, source_mask,
            max_target_length=self.config.max_target_length,
            top_k=self.config.action_space, top_p=1.0
        ).detach()[:, 1:])
        
    def _generate_code_ref(self, source_ids: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        """生成参考代码"""
        return torch.clone(respond_to_batch(
            self.model_ref, source_ids, source_mask,
            max_target_length=self.config.max_target_length,
            top_k=self.config.action_space, top_p=1.0
        ).detach()[:, 1:])
        
    def _compute_reward(self, response_ids: torch.Tensor, response_ids_ref: torch.Tensor, 
                       target_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """计算奖励"""
        reward, mean_rate, mean_ast_match, mean_dfg_match, num_errors, num_errors_ref, num_nodes, num_nodes_ref = get_reward(
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
            str(train_stats['objective/kl']),
            str(train_stats['objective/entropy']),
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
        
    def _evaluate(self, epoch: int):
        """评估模型"""
        self.logger.info(f"开始第 {epoch} 轮评估")
        
        # 训练集评估
        train_errors, train_errors_ref = self._evaluate_dataset(
            epoch, self.train_features, self.train_dataloader, 'train'
        )
        
        # 测试集评估
        test_errors, test_errors_ref = self._evaluate_dataset(
            epoch, self.test_features, self.test_dataloader, 'test'
        )
        
        self.logger.info(f"Epoch {epoch} 评估结果:")
        self.logger.info(f"  训练集 - 模型错误: {train_errors}, 参考模型错误: {train_errors_ref}")
        self.logger.info(f"  测试集 - 模型错误: {test_errors}, 参考模型错误: {test_errors_ref}")
        
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
                preds = respond_to_batch(
                    self.model, source_ids, source_mask,
                    max_target_length=self.config.max_target_length,
                    top_k=self.config.action_space, top_p=1.0
                )[:, 1:]
                
                preds_ref = respond_to_batch(
                    self.model_ref, source_ids, source_mask,
                    max_target_length=self.config.max_target_length,
                    top_k=self.config.action_space, top_p=1.0
                )[:, 1:]
                
                # 计算错误数
                nerrors += sum(get_reward(
                    lang=self.config.target_lang,
                    code_ids=preds,
                    code_ref_ids=preds_ref,
                    gold_ids=target_ids,
                    tokenizer=self.tokenizer
                )[4])
                
                nerrors_ref += sum(get_reward(
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
        predictions = [
            self.tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for id in pred_ids
        ]
        predictions_ref = [
            self.tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for id in pred_ids_ref
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
                f_gold.write(features[i].target + '\n')


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="优化的PPO代码生成训练程序")
    
    # 必需参数
    parser.add_argument("--source_lang", required=True, type=str,
                       help="源代码语言")
    parser.add_argument("--target_lang", required=True, type=str,
                       help="目标代码语言")
    parser.add_argument("--model_path", required=True, type=str,
                       help="预训练模型路径")
    parser.add_argument("--data_path", required=True, type=str,
                       help="数据目录路径")
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
        seed=args.seed
    )
    
    # 创建训练器并开始训练
    trainer = CodeTranslationTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main() 