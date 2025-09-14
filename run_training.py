#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified training launch script

Usage:
python run_training.py --source_lang python --target_lang java --model_path ./models/codet5-base --data_path ./data --output_path ./outputs

Or use configuration file:
python run_training.py --config config.json
"""

import argparse
import json
import sys
from pathlib import Path
from rl_trainer import CodeTranslationTrainer, TrainingConfig


def load_config_from_file(config_path: str) -> TrainingConfig:
    """Load configuration from configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    training_config = config_data['training_config']
    
    return TrainingConfig(
        source_lang=training_config['source_lang'],
        target_lang=training_config['target_lang'],
        model_path=training_config['model_path'],
        data_path=training_config['data_path'],
        output_path=training_config['output_path'],
        max_source_length=training_config.get('max_source_length', 400),
        max_target_length=training_config.get('max_target_length', 400),
        train_batch_size=training_config.get('train_batch_size', 16),
        test_batch_size=training_config.get('test_batch_size', 48),
        train_epochs=training_config.get('train_epochs', 1000000),
        learning_rate=training_config.get('learning_rate', 1e-5),
        kl_coef=training_config.get('kl_coef', 0.05),
        kl_target=training_config.get('kl_target', 1.0),
        vf_coef=training_config.get('vf_coef', 1e-3),
        action_space=training_config.get('action_space', 2),
        num_syn_samples=training_config.get('num_syn_samples', 5),
        run_id=training_config.get('run_id', 1),
        seed=training_config.get('seed', 42)
    )


def validate_config(config: TrainingConfig):
    """Validate configuration validity"""
    # Check required files and directories
    if not Path(config.model_path).exists():
        raise FileNotFoundError(f"Model path does not exist: {config.model_path}")
    
    if not Path(config.data_path).exists():
        raise FileNotFoundError(f"Data path does not exist: {config.data_path}")
    
    # Check supported languages
    supported_langs = ['python', 'java', 'cpp', 'c', 'javascript', 'php', 'c_sharp']
    if config.source_lang not in supported_langs:
        raise ValueError(f"Unsupported source code language: {config.source_lang}")
    if config.target_lang not in supported_langs:
        raise ValueError(f"Unsupported target code language: {config.target_lang}")
    
    # Check parameter range
    if config.learning_rate <= 0:
        raise ValueError(f"Learning rate must be greater than 0: {config.learning_rate}")
    if config.train_batch_size <= 0:
        raise ValueError(f"Training batch size must be greater than 0: {config.train_batch_size}")
    if config.train_epochs <= 0:
        raise ValueError(f"Training epochs must be greater than 0: {config.train_epochs}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="PPO code generation training launch script")
    
    # Add configuration options
    parser.add_argument("--config", type=str, help="Configuration file path")
    
    # Add command line arguments (when not using configuration file)
    parser.add_argument("--source_lang", type=str, help="Source code language")
    parser.add_argument("--target_lang", type=str, help="Target code language")
    parser.add_argument("--model_path", type=str, help="Pre-trained model path")
    parser.add_argument("--data_path", type=str, help="Data directory path")
    parser.add_argument("--output_path", type=str, help="Output directory path")
    
    # Optional parameters
    parser.add_argument("--max_source_length", type=int, default=400, help="Maximum source code length")
    parser.add_argument("--max_target_length", type=int, default=400, help="Maximum target code length")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--test_batch_size", type=int, default=48, help="Test batch size")
    parser.add_argument("--train_epochs", type=int, default=1000000, help="Training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--kl_coef", type=float, default=0.05, help="KL coefficient")
    parser.add_argument("--kl_target", type=float, default=1.0, help="KL target value")
    parser.add_argument("--vf_coef", type=float, default=1e-3, help="Value function coefficient")
    parser.add_argument("--action_space", type=int, default=2, help="Action space size")
    parser.add_argument("--num_syn_samples", type=int, default=5, help="Number of samples per epoch")
    parser.add_argument("--run_id", type=int, default=1, help="Run ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        # Load from configuration file
        config = load_config_from_file(args.config)
        print(f"Loaded configuration from: {args.config}")
    else:
        # Create configuration from command line arguments
        if not all([args.source_lang, args.target_lang, args.model_path, args.data_path, args.output_path]):
            print("Error: When not using configuration file, all required parameters must be provided")
            print("Usage:")
            print("  python run_training.py --config config.json")
            print("  or")
            print("  python run_training.py --source_lang python --target_lang java --model_path ./models/codet5-base --data_path ./data --output_path ./outputs")
            sys.exit(1)
        
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
    
    # Validate configuration
    try:
        validate_config(config)
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        sys.exit(1)
    
    # Create output directory
    Path(config.output_path).mkdir(parents=True, exist_ok=True)
    
    # Print configuration information
    print("=" * 60)
    print("PPO code generation training configuration")
    print("=" * 60)
    print(f"Source code language: {config.source_lang}")
    print(f"Target code language: {config.target_lang}")
    print(f"Model path: {config.model_path}")
    print(f"Data path: {config.data_path}")
    print(f"Output path: {config.output_path}")
    print(f"Training batch size: {config.train_batch_size}")
    print(f"Test batch size: {config.test_batch_size}")
    print(f"Training epochs: {config.train_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"KL coefficient: {config.kl_coef}")
    print(f"KL target value: {config.kl_target}")
    print(f"Value function coefficient: {config.vf_coef}")
    print(f"Action space size: {config.action_space}")
    print(f"Number of samples per epoch: {config.num_syn_samples}")
    print(f"Run ID: {config.run_id}")
    print(f"Random seed: {config.seed}")
    print("=" * 60)
    
    # Start training
    try:
        trainer = CodeTranslationTrainer(config)
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 