#!/usr/bin/env python3
"""Qwen2.5-Coder RL Trainer - Supports PPO, GRPO, RLOO algorithms"""

import os
import torch
import json
import shutil
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from tree_sitter import Language, Parser

# Internal imports
from code_parser import (DFG_python, DFG_java, DFG_php, DFG_javascript, DFG_csharp)
from reward import get_reward
from model import respond_to_batch, QwenCoderHeadWithValueModelLocal
from utils import (
    Example, InputFeatures, extract_code_from_qwen_response,
    read_qwen_examples, convert_qwen_examples_to_features, create_reward_wrapper
)


@dataclass
class TrainingConfig:
    """RL Training Configuration Class"""
    # Required parameters
    source_lang: str
    target_lang: str
    model_path: str
    data_path: str
    output_path: str
    
    # Model configuration
    max_source_length: int = 400
    max_target_length: int = 400
    
    # Training configuration
    train_batch_size: int = 16
    test_batch_size: int = 48
    train_epochs: int = 1000000
    learning_rate: float = 1e-5
    
    # RL algorithm configuration
    rl_algorithm: str = "ppo"  # "ppo", "grpo", "rloo"
    kl_coef: float = 0.05
    kl_target: float = 0.1
    vf_coef: float = 1e-3
    minibatch_size: int = 1
    gradient_accumulation_steps: int = 4
    group_size: int = 4  # GRPO/RLOO group size
    
    # RLOO specific configuration
    rloo_use_clipping: bool = False
    rloo_baseline_mode: str = "leave_one_out"
    
    # Generation configuration
    action_space: int = 5
    num_syn_samples: int = 5
    
    # Checkpoint configuration
    save_every_n_steps: int = 0
    max_checkpoints: int = 10
    
    # System configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    run_id: int = 1
    seed: int = 42
    use_tensorboard: bool = True
    
    # Backward compatibility properties
    @property
    def l1(self): return self.source_lang
    @property  
    def l2(self): return self.target_lang


class BaseRLTrainer:
    """Base RL Trainer Class, supports multiple RL algorithms"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_logging()
        self.setup_device()
        self.setup_language_mappings()
        self.setup_parsers()
        self.setup_models()
        self.setup_data_loaders()
        self.setup_rl_trainer()
        self.setup_training_stats()
        self.get_reward_func = create_reward_wrapper(get_reward)
        
    def setup_logging(self):
        """Setup logging system"""
        log_dir = Path(self.config.output_path) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler
        file_handler = logging.FileHandler(log_dir / f"training_{self.config.run_id}.log")
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Training config: {self.config.source_lang} -> {self.config.target_lang}")
        self.logger.info(f"RL algorithm: {self.config.rl_algorithm.upper()}")
        self.logger.info(f"Device: {self.config.device}")
        
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
        self.dir_dict = {'javascript': 'Javascript', 'java': 'Java', 'c_sharp': 'C#', 
                        'php': 'PHP', 'python': 'Python', 'c': 'C', 'cpp': 'C++'}
        
    def setup_parsers(self):
        """Setup code parsers"""
        self.dfg_function = {'python': DFG_python, 'java': DFG_java, 'php': DFG_php,
                           'javascript': DFG_javascript, 'c_sharp': DFG_csharp, 'c': DFG_csharp, 'cpp': DFG_csharp}
        
        self.parsers = {}
        for lang in self.dfg_function:
            try:
                LANGUAGE = Language('code_parser/my-languages.so', lang)
                parser = Parser()
                parser.set_language(LANGUAGE)
                self.parsers[lang] = [parser, self.dfg_function[lang]]
            except Exception as e:
                self.logger.warning(f"Failed to load {lang} parser: {e}")
                
    def setup_models(self):
        """Setup models and tokenizer"""
        self.model_dir = Path(self.config.model_path)
        self._check_model_files()
        
        print(f"Loading model to device: {self.config.device}")
        # Load training model
        self.model = QwenCoderHeadWithValueModelLocal(self.config.model_path, torch_dtype=torch.bfloat16, device=self.config.device)
        self.model.to(self.config.device).train()
        
        # Load reference model
        self.model_ref = QwenCoderHeadWithValueModelLocal(self.config.model_path, torch_dtype=torch.bfloat16, device=self.config.device)
        self.model_ref.to(self.config.device)
        for p in self.model_ref.parameters():
            p.requires_grad = False
        self.model_ref.eval()

        # Set model configuration
        self.model.model.config.use_cache = False
        self.model_ref.model.config.use_cache = False
        
        # Load tokenizer
        print("Loading tokenizer from local...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True, trust_remote_code=True, padding_side='right')
            print("Tokenizer loaded from local!")
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer from local: {e}")
        
        self.logger.info("Models and tokenizers loaded")
        
    def _check_model_files(self):
        """Check if necessary model files exist"""
        print("Checking model files...")
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model weight file does not exist: {self.config.model_path}")
        
        required_files = ['config.json', 'tokenizer.json', 'vocab.json', 'merges.txt', 'special_tokens_map.json']
        missing_files = [f for f in required_files if not (self.model_dir / f).exists()]
        
        if missing_files:
            raise FileNotFoundError(f"Missing necessary model files: {missing_files}")
        print("✓ All necessary files checked")
        
    def setup_data_loaders(self):
        """Setup data loaders"""
        self.data_files = self._build_data_paths()
        
        # Load and convert data
        datasets = {}
        for split in ['train', 'dev', 'test']:
            examples = read_qwen_examples(self.data_files[split], self.config)
            features = convert_qwen_examples_to_features(examples, self.tokenizer, self.config, stage='train')
            setattr(self, f"{split}_examples", examples)
            setattr(self, f"{split}_features", features)
            datasets[split] = features
        
        # Create data loaders
        self.train_dataloader = self._create_dataloader(datasets['train'], self.config.train_batch_size, shuffle=True)
        self.dev_dataloader = self._create_dataloader(datasets['dev'], self.config.train_batch_size, shuffle=False)
        self.test_dataloader = self._create_dataloader(datasets['test'], self.config.test_batch_size, shuffle=False)
        
        self.logger.info(f"Data loaded - train: {len(datasets['train'])}, dev: {len(datasets['dev'])}, test: {len(datasets['test'])}")
        
    def _build_data_paths(self) -> Dict[str, str]:
        """Build data file paths"""
        l1, l2 = self.config.source_lang, self.config.target_lang
        possible_paths = [
            f"{self.config.data_path}/qwen/{self.dir_dict[l1]}-{self.dir_dict[l2]}/",
            f"{self.config.data_path}/qwen/{self.dir_dict[l2]}-{self.dir_dict[l1]}/",
            f"{self.config.data_path}/{self.dir_dict[l1]}-{self.dir_dict[l2]}/",
            f"{self.config.data_path}/{self.dir_dict[l2]}-{self.dir_dict[l1]}/"
        ]
        
        data_dir = next((path for path in possible_paths if os.path.exists(path)), None)
        if data_dir is None:
            raise FileNotFoundError(f"Data directory not found: {possible_paths}")
            
        return {'train': f"{data_dir}train.jsonl", 'dev': f"{data_dir}val.jsonl", 'test': f"{data_dir}test.jsonl"}
        
    def _create_dataloader(self, features: List[InputFeatures], batch_size: int, shuffle: bool = False) -> DataLoader:
        """Create data loader"""
        tensors = [torch.tensor([getattr(f, attr) for f in features], dtype=torch.long) 
                  for attr in ['source_ids', 'source_mask', 'target_ids', 'target_mask']]
        tensors.append(torch.arange(len(features)))
        return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)
        
    def setup_rl_trainer(self):
        """Setup RL trainer (supports multiple algorithms)"""
        rl_config = {
            "batch_size": self.config.train_batch_size, 'eos_token_id': self.tokenizer.eos_token_id,
            'lr': self.config.learning_rate, "adap_kl_ctrl": True, 'init_kl_coef': self.config.kl_coef,
            "target": self.config.kl_target, "vf_coef": self.config.vf_coef, "minibatch_size": self.config.minibatch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps, "tokenizer": self.tokenizer,
            # GRPO specific configuration
            "group_size": self.config.group_size, "kl_coef": self.config.kl_coef, "device": self.config.device
        }
        
        if self.config.rl_algorithm == "ppo":
            from ppo import PPOTrainer
            self.rl_trainer = PPOTrainer(self.model, self.model_ref, **rl_config)
        elif self.config.rl_algorithm == "grpo":
            from grpo import GRPOTrainer
            self.rl_trainer = GRPOTrainer(self.model, self.model_ref, **rl_config)
        elif self.config.rl_algorithm == "rloo":
            from rloo import RLOOTrainer, RLOOConfig
            # RLOO specific configuration
            rloo_specific_config = {
                "lr": self.config.learning_rate,
                "group_size": self.config.group_size,
                "kl_coef": self.config.kl_coef,
                "use_clipping": self.config.rloo_use_clipping,
                "baseline_mode": self.config.rloo_baseline_mode,
                "device": self.config.device
            }
            rloo_config = RLOOConfig(**rloo_specific_config)
            self.rl_trainer = RLOOTrainer(self.model, self.model_ref, rloo_config)
        else:
            raise ValueError(f"Unsupported RL algorithm: {self.config.rl_algorithm}")
        
        self.logger.info(f"{self.config.rl_algorithm.upper()} trainer setup complete")
        
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
        
        # Performance tracking
        self.best_metrics = {'reward': -float('inf')}
        self.last_save_step = 0
        
        # Create output directories
        self.output_dir = Path(self.config.output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(self.config.output_path) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Tensorboard
        self.tensorboard_writer = None
        if self.config.use_tensorboard:
            tb_log_dir = Path(self.config.output_path) / "tensorboard" / f"run_{self.config.run_id}"
            tb_log_dir.mkdir(parents=True, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=str(tb_log_dir))
            self.logger.info(f"Tensorboard logs: {tb_log_dir}")
            
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        self.logger.info("🔥 Joint actor-critic training from start")
        
        for epoch in range(self.config.train_epochs):
            self.logger.info(f"Starting epoch {epoch}")
            for sample_idx in range(self.config.num_syn_samples):
                self._train_epoch(epoch, sample_idx)
            self._save_checkpoint(epoch)
            self._evaluate(epoch)
            
    def _train_epoch(self, epoch: int, sample_idx: int):
        """Train one epoch"""
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}, Sample {sample_idx}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Process batch data
                batch = tuple(t.to(self.config.device) for t in batch)
                source_ids, source_mask, target_ids, target_mask, ind = batch
                
                # Generate code
                if self.config.rl_algorithm == "ppo":
                    response_ids = self._generate_code(source_ids, source_mask)
                    response_ids_ref = self._generate_code_ref(source_ids, source_mask)
                    expanded_source_ids = source_ids
                    expanded_source_mask = source_mask
                    expanded_target_ids = target_ids
                elif self.config.rl_algorithm in ["grpo", "rloo"]:
                    # GRPO/RLOO mode: generate multiple candidate codes
                    response_ids = self._generate_code_group(source_ids, source_mask, self.config.group_size)
                    response_ids_ref = self._generate_code_ref(source_ids, source_mask)
                    # Keep original input, let trainer handle group expansion internally to save memory
                    expanded_source_ids = source_ids  # Don't expand, save memory
                    expanded_source_mask = source_mask  # Don't expand, save memory
                    expanded_target_ids = target_ids.repeat_interleave(self.config.group_size, dim=0)
                response_mask = self._get_response_mask(response_ids)

                # Compute reward and update statistics
                reward, metrics = self._compute_reward(response_ids, response_ids_ref, expanded_target_ids, response_mask)
                self._update_stats(reward, metrics, len(source_ids))
                
                # RL training step
                train_stats = self.rl_trainer.step(expanded_source_ids, expanded_source_mask, response_ids, response_ids_ref, 
                                                  reward.to(self.config.device), response_mask.to(self.config.device))
                
                # Update progress bar and logging
                avg_errors = self.training_stats['total_nerrors']/self.training_stats['total_seen']
                avg_reward = self.training_stats['total_rewards']/self.training_stats['total_seen']
                pbar.set_description(f"Epoch {epoch}, Sample {sample_idx}, Avg Errors: {avg_errors:.3f}, Avg Reward: {avg_reward:.3f}")
                
                self._log_training_step(epoch, sample_idx, batch_idx, reward, metrics, train_stats)
                self.training_stats['nsteps'] += 1
                
                # Checkpoint saving
                if (self.config.save_every_n_steps > 0 and 
                    self.training_stats['nsteps'] % self.config.save_every_n_steps == 0):
                    self._save_checkpoint(epoch, step=self.training_stats['nsteps'])
                    
            except Exception as e:
                self.logger.error(f"Training step error: {e}")
                self._save_checkpoint(epoch, step=self.training_stats['nsteps'])
                raise e
        
    def _generate_code(self, source_ids: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        """Generate code"""
        full = respond_to_batch(self.model, source_ids, source_mask, max_target_length=self.config.max_target_length,
                               top_k=self.config.action_space, top_p=0.9, tokenizer=self.tokenizer).detach()
        return torch.clone(full[:, source_ids.size(1):])
        
    def _generate_code_ref(self, source_ids: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        """Generate reference code"""
        full = respond_to_batch(self.model_ref, source_ids, source_mask, max_target_length=self.config.max_target_length,
                               top_k=self.config.action_space, top_p=0.9, tokenizer=self.tokenizer).detach()
        return torch.clone(full[:, source_ids.size(1):])
    
    def _generate_code_group(self, source_ids: torch.Tensor, source_mask: torch.Tensor, group_size: int) -> torch.Tensor:
        """
        Generate multiple code candidates for GRPO algorithm
        
        Args:
            source_ids: Source code sequence [batch_size, source_length]
            source_mask: Source code mask [batch_size, source_length]
            group_size: Number of candidates to generate per input
            
        Returns:
            response_ids: Generated code sequence [batch_size * group_size, target_length]
        """
        batch_size = source_ids.size(0)
        
        # Optimization: directly expand the entire batch instead of processing one by one
        # This better utilizes GPU parallelism
        expanded_source_ids = source_ids.repeat_interleave(group_size, dim=0)  # [batch_size * group_size, source_length]
        expanded_source_mask = source_mask.repeat_interleave(group_size, dim=0)  # [batch_size * group_size, source_length]
        
        # Generate all candidate codes at once
        full = respond_to_batch(
            self.model, 
            expanded_source_ids, 
            expanded_source_mask, 
            max_target_length=self.config.max_target_length,
            top_k=self.config.action_space, 
            top_p=0.9, 
            tokenizer=self.tokenizer
        ).detach()
        
        # Extract generated part (remove source code part)
        response_ids = full[:, expanded_source_ids.size(1):]  # [batch_size * group_size, target_length]
        
        return torch.clone(response_ids)
        
    def _get_response_mask(self, response_ids: torch.Tensor) -> torch.Tensor:
        """Create response sequence mask"""
        batch_size, seq_len = response_ids.shape
        masks = torch.zeros_like(response_ids, dtype=torch.float)
        
        for i, seq in enumerate(response_ids):
            eos_pos = (seq == self.tokenizer.eos_token_id).nonzero()
            eos_pos = eos_pos[0].item() if len(eos_pos) > 0 else seq_len - 1
            masks[i, :eos_pos + 1] = 1.0
        return masks
        
    def _compute_reward(self, response_ids: torch.Tensor, response_ids_ref: torch.Tensor, target_ids: torch.Tensor, response_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute reward and place reward values at EOS positions"""
        
        # Check if it's GRPO mode (dimension mismatch case)
        batch_size, seq_len = response_ids.shape
        is_grpo_mode = (response_ids_ref is not None and 
                       response_ids.size(0) != response_ids_ref.size(0) and
                       response_ids.size(0) % response_ids_ref.size(0) == 0)
        
        if is_grpo_mode:
            # GRPO mode: separately compute rewards for main response and reference response
            group_size = response_ids.size(0) // response_ids_ref.size(0)
            original_batch_size = response_ids_ref.size(0)
            
            # Construct corresponding target_ids for reference response (using original target_ids)
            target_ids_ref = target_ids[:original_batch_size]
            
            # Separately compute rewards
            result_main = self.get_reward_func(
                lang=self.config.target_lang, 
                code_ids=response_ids, 
                code_ref_ids=None,  # GRPO doesn't need reference comparison
                gold_ids=target_ids, 
                tokenizer=self.tokenizer
            )
            
            result_ref = self.get_reward_func(
                lang=self.config.target_lang,
                code_ids=response_ids_ref,
                code_ref_ids=None,
                gold_ids=target_ids_ref,
                tokenizer=self.tokenizer
            )
            
            # Unpack main results
            reward_scores, _, mean_rate, mean_ast_match, mean_dfg_match, _, _, _, num_errors, _, num_nodes, _, sample_details = result_main
            
            # Unpack reference results
            _, reward_ref, mean_rate_ref, mean_ast_match_ref, mean_dfg_match_ref, _, _, _, num_errors_ref, _, num_nodes_ref, _, _ = result_ref
            
        else:
            # Normal PPO mode: unified computation
            result = self.get_reward_func(lang=self.config.target_lang, code_ids=response_ids, 
                                         code_ref_ids=response_ids_ref, gold_ids=target_ids, tokenizer=self.tokenizer)
            reward_scores, reward_ref, mean_rate, mean_ast_match, mean_dfg_match, mean_rate_ref, mean_ast_match_ref, mean_dfg_match_ref, num_errors, num_errors_ref, num_nodes, num_nodes_ref, sample_details = result

        # Create reward tensor with same shape as response_ids
        rewards = torch.zeros_like(response_ids, dtype=torch.float, device=response_ids.device)
        
        # Place reward values at EOS position of each sequence
        for i in range(batch_size):
            # Find EOS position
            eos_positions = (response_ids[i] == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                eos_pos = eos_positions[0].item()
            else:
                # If no EOS, use last valid position
                valid_positions = (response_mask[i] > 0.5).nonzero(as_tuple=True)[0]
                eos_pos = valid_positions[-1].item() if len(valid_positions) > 0 else seq_len - 1
            
            # Extract reward value and place at EOS position
            if isinstance(reward_scores, torch.Tensor):
                if reward_scores.ndim > 1:
                    # Multi-dimensional reward tensor, find non-zero values
                    if i < reward_scores.size(0):  # Ensure index is within range
                        non_zero_indices = (reward_scores[i] != 0).nonzero(as_tuple=True)[0]
                        reward_value = reward_scores[i][non_zero_indices[0]].item() if len(non_zero_indices) > 0 else reward_scores[i].sum().item()
                    else:
                        reward_value = 0.0
                else:
                    # One-dimensional reward tensor
                    if i < reward_scores.size(0):
                        reward_value = reward_scores[i].item()
                    else:
                        reward_value = 0.0
            else:
                if hasattr(reward_scores, '__getitem__') and i < len(reward_scores):
                    reward_value = float(reward_scores[i])
                else:
                    reward_value = float(reward_scores) if not hasattr(reward_scores, '__getitem__') else 0.0
            
            rewards[i, eos_pos] = reward_value

        return rewards, {
            'mean_rate': mean_rate, 'mean_ast_match': mean_ast_match, 'mean_dfg_match': mean_dfg_match,
            'mean_rate_ref': mean_rate_ref, 'mean_ast_match_ref': mean_ast_match_ref, 'mean_dfg_match_ref': mean_dfg_match_ref,
            'num_errors': num_errors, 'num_errors_ref': num_errors_ref, 'num_nodes': num_nodes, 'num_nodes_ref': num_nodes_ref,
            'reward_ref': reward_ref, 'sample_details': sample_details
        }
        
    def _update_stats(self, reward: torch.Tensor, metrics: Dict, batch_size: int):
        """Update training statistics"""
        self.training_stats.update({
            'total_rewards': self.training_stats.get('total_rewards', 0) + float(sum(reward.sum(axis=-1).tolist())),
            'total_nerrors': self.training_stats.get('total_nerrors', 0) + sum(metrics['num_errors']),
            'total_nnodes': self.training_stats.get('total_nnodes', 0) + sum(metrics['num_nodes']),
            'total_nerrors_ref': self.training_stats.get('total_nerrors_ref', 0) + sum(metrics['num_errors_ref']),
            'total_nnodes_ref': self.training_stats.get('total_nnodes_ref', 0) + sum(metrics['num_nodes_ref']),
            'total_rewards_ref': self.training_stats.get('total_rewards_ref', 0) + float(sum(metrics['reward_ref'].sum(axis=-1).tolist())),
            'total_seen': self.training_stats.get('total_seen', 0) + batch_size
        })
        
    def _log_training_step(self, epoch: int, sample_idx: int, batch_idx: int, reward: torch.Tensor, metrics: Dict, train_stats: Dict):
        """Log training step"""
        # Calculate average metrics
        avg_reward = float(sum(reward.sum(axis=-1).tolist())) / len(reward)
        avg_reward_ref = float(sum(metrics['reward_ref'].sum(axis=-1).tolist())) / len(metrics['reward_ref'])
        avg_errors = sum(metrics['num_errors']) / len(metrics['num_errors'])
        avg_errors_ref = sum(metrics['num_errors_ref']) / len(metrics['num_errors_ref'])
        
        def _req(key: str) -> float:
            if key not in train_stats:
                raise KeyError(f"Missing required train_stats key: {key}")
            val = train_stats[key]
            return float(val if not hasattr(val, 'item') else val.item())

        # TensorBoard logging
        if self.tensorboard_writer:
            global_step = self.training_stats['nsteps']
            
            # Basic metrics
            self.tensorboard_writer.add_scalar("Training/Reward", avg_reward, global_step)
            self.tensorboard_writer.add_scalar("Training/Reward_Ref", avg_reward_ref, global_step)
            self.tensorboard_writer.add_scalar("Training/Compilation_Rate", metrics['mean_rate'], global_step)
            self.tensorboard_writer.add_scalar("Training/AST_Match", metrics['mean_ast_match'], global_step)
            self.tensorboard_writer.add_scalar("Training/DFG_Match", metrics['mean_dfg_match'], global_step)
            self.tensorboard_writer.add_scalar("Training/Errors", avg_errors, global_step)
            
            # RL algorithm specific metrics
            for key in ['ppo/mean_kl', 'ppo/loss/total', 'ppo/loss/policy', 'ppo/loss/value',
                       'grpo/mean_kl', 'grpo/total_loss', 'grpo/policy_loss',
                       'rloo/mean_kl', 'rloo/total_loss', 'rloo/policy_loss']:
                if key in train_stats:
                    val = train_stats[key]
                    value = val.item() if hasattr(val, 'item') else float(val)
                    self.tensorboard_writer.add_scalar(f"RL/{key.replace('/', '_')}", value, global_step)
            
            # Learning rate
            try:
                current_lr = self.rl_trainer.optimizer.param_groups[0]['lr']
                self.tensorboard_writer.add_scalar("Training/Learning_Rate", current_lr, global_step)
            except:
                pass

        # Log training metrics
        self.logger.info(f"Step {self.training_stats['nsteps']}: Reward={avg_reward:.4f}, Errors={avg_errors:.4f}, KL={train_stats.get('ppo/mean_kl', train_stats.get('grpo/mean_kl', train_stats.get('rloo/mean_kl', 0.0))):.4f}")
            
    def _save_checkpoint(self, epoch: int, step: int = None):
        """Save checkpoint"""
        if step is not None and step == self.last_save_step:
            return
            
        checkpoint_dir = self.checkpoint_dir / f"checkpoint-epoch-{epoch}-step-{step or 0}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model and tokenizer
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_save.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            
            # Save training information
            config_info = {
                "epoch": epoch,
                "step": step,
                "training_config": self.config.__dict__,
                "save_time": datetime.datetime.now().isoformat()
            }
            
            with open(checkpoint_dir / "training_info.json", 'w', encoding='utf-8') as f:
                json.dump(config_info, f, indent=2, ensure_ascii=False)
            
            if step is not None:
                self.last_save_step = step
                
            self.logger.info(f"Checkpoint saved: {checkpoint_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints"""
        if self.config.max_checkpoints <= 0:
            return
            
        checkpoint_dirs = list(self.checkpoint_dir.glob("checkpoint-*"))
        if len(checkpoint_dirs) <= self.config.max_checkpoints:
            return
            
        # Sort by modification time, delete oldest
        checkpoint_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        dirs_to_delete = checkpoint_dirs[self.config.max_checkpoints:]
        
        for dir_path in dirs_to_delete:
            try:
                shutil.rmtree(dir_path)
                self.logger.info(f"Deleted old checkpoint: {dir_path.name}")
            except Exception as e:
                self.logger.warning(f"Failed to delete checkpoint {dir_path}: {e}")
                
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
        
        # Record evaluation metrics to Tensorboard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar("Evaluation/Train_Errors", train_errors, epoch)
            self.tensorboard_writer.add_scalar("Evaluation/Test_Errors", test_errors, epoch)
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




# Backward compatibility alias
CodeTranslationTrainer = BaseRLTrainer


def create_trainer(config: TrainingConfig) -> BaseRLTrainer:
    """Factory function to create trainer"""
    return BaseRLTrainer(config)


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

    # RL算法选择参数
    parser.add_argument("--rl_algorithm", default="ppo", type=str, choices=["ppo", "grpo", "rloo"],
                       help="RL算法选择 (default: ppo)")
    parser.add_argument("--group_size", default=4, type=int,
                       help="GRPO/RLOO算法的组大小 (default: 4)")
    
    # RLOO特有参数
    parser.add_argument("--rloo_use_clipping", action="store_true", default=False,
                       help="RLOO是否使用PPO clipping (default: False)")
    parser.add_argument("--rloo_baseline_mode", default="leave_one_out", type=str,
                       choices=["leave_one_out", "mean"],
                       help="RLOO基线计算模式 (default: leave_one_out)")

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
    
    # Checkpoint configuration
    parser.add_argument("--save_every_n_steps", default=0, type=int,
                       help="Save checkpoint every N steps (0 means disabled)")
    parser.add_argument("--max_checkpoints", default=10, type=int,
                       help="Maximum number of checkpoints to retain")
    
    # System configuration
    parser.add_argument("--use_tensorboard", action="store_true", default=True,
                       help="Enable Tensorboard logging")
    parser.add_argument("--no_tensorboard", action="store_false", dest="use_tensorboard",
                       help="Disable Tensorboard logging")
    
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
        train_epochs=args.train_epochs,
        learning_rate=args.learning_rate,
        kl_coef=args.kl_coef,
        kl_target=args.kl_target,
        vf_coef=args.vf_coef,
        minibatch_size=args.minibatch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        rl_algorithm=args.rl_algorithm,
        group_size=args.group_size,
        rloo_use_clipping=args.rloo_use_clipping,
        rloo_baseline_mode=args.rloo_baseline_mode,
        action_space=args.action_space,
        num_syn_samples=args.num_syn_samples,
        run_id=args.run_id,
        seed=args.seed,
        save_every_n_steps=args.save_every_n_steps,
        max_checkpoints=args.max_checkpoints,
        use_tensorboard=args.use_tensorboard
    )
    
    print("=" * 60)
    print("🚀 Qwen2.5-Coder RL Code Translation Training Program")
    print("=" * 60)
    print(f"📝 Source Language: {config.source_lang} | 🎯 Target Language: {config.target_lang}")
    print(f"🤖 Model: {config.model_path}")
    print(f"📂 Data: {config.data_path} | 💾 Output: {config.output_path}")
    print(f"🔧 Device: {config.device} | 🧠 Algorithm: {config.rl_algorithm.upper()}")
    print("=" * 60)
    
    # Create and start trainer
    create_trainer(config).train()


if __name__ == "__main__":
    main() 