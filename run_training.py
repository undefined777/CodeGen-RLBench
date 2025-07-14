#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的训练启动脚本

使用方法：
python run_training.py --source_lang python --target_lang java --model_path ./models/codet5-base --data_path ./data --output_path ./outputs

或者使用配置文件：
python run_training.py --config config.json
"""

import argparse
import json
import sys
from pathlib import Path
from optimized_rl_trainer import CodeTranslationTrainer, TrainingConfig


def load_config_from_file(config_path: str) -> TrainingConfig:
    """从配置文件加载配置"""
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
    """验证配置的有效性"""
    # 检查必需的文件和目录
    if not Path(config.model_path).exists():
        raise FileNotFoundError(f"模型路径不存在: {config.model_path}")
    
    if not Path(config.data_path).exists():
        raise FileNotFoundError(f"数据路径不存在: {config.data_path}")
    
    # 检查支持的语言
    supported_langs = ['python', 'java', 'cpp', 'c', 'javascript', 'php', 'c_sharp']
    if config.source_lang not in supported_langs:
        raise ValueError(f"不支持的源代码语言: {config.source_lang}")
    if config.target_lang not in supported_langs:
        raise ValueError(f"不支持的目标代码语言: {config.target_lang}")
    
    # 检查参数范围
    if config.learning_rate <= 0:
        raise ValueError(f"学习率必须大于0: {config.learning_rate}")
    if config.train_batch_size <= 0:
        raise ValueError(f"训练批次大小必须大于0: {config.train_batch_size}")
    if config.train_epochs <= 0:
        raise ValueError(f"训练轮数必须大于0: {config.train_epochs}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PPO代码生成训练启动脚本")
    
    # 添加配置选项
    parser.add_argument("--config", type=str, help="配置文件路径")
    
    # 添加命令行参数（当不使用配置文件时）
    parser.add_argument("--source_lang", type=str, help="源代码语言")
    parser.add_argument("--target_lang", type=str, help="目标代码语言")
    parser.add_argument("--model_path", type=str, help="预训练模型路径")
    parser.add_argument("--data_path", type=str, help="数据目录路径")
    parser.add_argument("--output_path", type=str, help="输出目录路径")
    
    # 可选参数
    parser.add_argument("--max_source_length", type=int, default=400, help="最大源代码长度")
    parser.add_argument("--max_target_length", type=int, default=400, help="最大目标代码长度")
    parser.add_argument("--train_batch_size", type=int, default=16, help="训练批次大小")
    parser.add_argument("--test_batch_size", type=int, default=48, help="测试批次大小")
    parser.add_argument("--train_epochs", type=int, default=1000000, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--kl_coef", type=float, default=0.05, help="KL系数")
    parser.add_argument("--kl_target", type=float, default=1.0, help="KL目标值")
    parser.add_argument("--vf_coef", type=float, default=1e-3, help="价值函数系数")
    parser.add_argument("--action_space", type=int, default=2, help="动作空间大小")
    parser.add_argument("--num_syn_samples", type=int, default=5, help="每轮采样次数")
    parser.add_argument("--run_id", type=int, default=1, help="运行ID")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        # 从配置文件加载
        config = load_config_from_file(args.config)
        print(f"从配置文件加载配置: {args.config}")
    else:
        # 从命令行参数创建配置
        if not all([args.source_lang, args.target_lang, args.model_path, args.data_path, args.output_path]):
            print("错误：当不使用配置文件时，必须提供所有必需参数")
            print("使用方法：")
            print("  python run_training.py --config config.json")
            print("  或者")
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
    
    # 验证配置
    try:
        validate_config(config)
    except Exception as e:
        print(f"配置验证失败: {e}")
        sys.exit(1)
    
    # 创建输出目录
    Path(config.output_path).mkdir(parents=True, exist_ok=True)
    
    # 打印配置信息
    print("=" * 60)
    print("PPO代码生成训练配置")
    print("=" * 60)
    print(f"源代码语言: {config.source_lang}")
    print(f"目标代码语言: {config.target_lang}")
    print(f"模型路径: {config.model_path}")
    print(f"数据路径: {config.data_path}")
    print(f"输出路径: {config.output_path}")
    print(f"训练批次大小: {config.train_batch_size}")
    print(f"测试批次大小: {config.test_batch_size}")
    print(f"训练轮数: {config.train_epochs}")
    print(f"学习率: {config.learning_rate}")
    print(f"KL系数: {config.kl_coef}")
    print(f"KL目标值: {config.kl_target}")
    print(f"价值函数系数: {config.vf_coef}")
    print(f"动作空间大小: {config.action_space}")
    print(f"每轮采样次数: {config.num_syn_samples}")
    print(f"运行ID: {config.run_id}")
    print(f"随机种子: {config.seed}")
    print("=" * 60)
    
    # 开始训练
    try:
        trainer = CodeTranslationTrainer(config)
        trainer.train()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 