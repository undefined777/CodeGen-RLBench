#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试训练环境中的train_dataloader
直接使用训练环境中的数据处理逻辑
"""

import os
import sys
import torch
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
import time
from transformers import AutoTokenizer

# 导入训练环境中的模块
from optimized_rl_trainer import (
    read_qwen_examples,
    convert_qwen_examples_to_features,
    TrainingConfig,
    CodeTranslationTrainer
)
from model import QwenCoderHeadWithValueModelLocal


def test_train_dataloader():
    """测试训练环境中的train_dataloader"""
    print("🚀 测试训练环境中的train_dataloader")
    print("=" * 80)
    
    # 1. 创建训练配置
    print("📋 创建训练配置...")
    config = TrainingConfig(
        source_lang="java",
        target_lang="cpp",
        model_path="/home/cxy/CodeGen-RLBench/test_model/checkpoint-200",
        data_path="data",
        output_path="./test_outputs",
        max_source_length=600,  # 与训练环境一致
        max_target_length=600,  # 与训练环境一致
        train_batch_size=2,     # 与训练环境一致
        test_batch_size=2,      # 与训练环境一致
        action_space=2,         # 与训练环境一致
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("✅ 训练配置创建成功")
    
    # 2. 创建训练器实例
    print("\n🔧 创建训练器实例...")
    trainer = CodeTranslationTrainer(config)
    print("✅ 训练器实例创建成功")
    
    # 3. 设置模型和tokenizer
    print("\n📥 设置模型和tokenizer...")
    trainer.setup_models()
    print("✅ 模型和tokenizer设置成功")
    
    # 4. 设置数据加载器
    print("\n📂 设置数据加载器...")
    trainer.setup_data_loaders()
    print("✅ 数据加载器设置成功")
    
    # 5. 测试数据加载
    print(f"\n📊 数据统计:")
    print(f"  训练样本数: {len(trainer.train_features)}")
    print(f"  验证样本数: {len(trainer.dev_features)}")
    print(f"  测试样本数: {len(trainer.test_features)}")
    
    # 6. 测试生成
    print(f"\n🤖 测试模型生成...")
    
    # 获取第一个batch
    for batch_idx, batch in enumerate(trainer.train_dataloader):
        if batch_idx >= 2:  # 只测试前2个batch
            break
            
        source_ids, source_mask, target_ids, target_mask, indices = batch
        
        print(f"\n📝 Batch {batch_idx + 1}:")
        print(f"  Source IDs shape: {source_ids.shape}")
        print(f"  Source mask shape: {source_mask.shape}")
        print(f"  Target IDs shape: {target_ids.shape}")
        print(f"  Target mask shape: {target_mask.shape}")
        print(f"  Indices: {indices.tolist()}")
        
        # 测试生成
        try:
            print(f"  🔄 开始生成...")
            start_time = time.time()
            
            # 修复：将tensor移动到正确的设备（与训练环境一致）
            source_ids = source_ids.to(trainer.config.device)
            source_mask = source_mask.to(trainer.config.device)
            
            # 使用训练环境中的生成方法
            response_ids = trainer._generate_code(source_ids, source_mask)
            response_ids_ref = trainer._generate_code_ref(source_ids, source_mask)
            
            generation_time = time.time() - start_time
            
            print(f"  ⏱️  生成时间: {generation_time:.2f}秒")
            print(f"  📊 生成结果shape: {response_ids.shape}")
            print(f"  📊 参考结果shape: {response_ids_ref.shape}")
            
            # 🔧 新增：测试reward计算
            print(f"  🎯 开始计算reward...")
            reward_start_time = time.time()
            
            # 将target_ids也移动到设备上
            target_ids = target_ids.to(trainer.config.device)
            
            # 使用训练环境中的reward计算方法
            reward, metrics = trainer._compute_reward(response_ids, response_ids_ref, target_ids)
            
            reward_time = time.time() - reward_start_time
            print(f"  ⏱️  Reward计算时间: {reward_time:.3f}秒")
            
            # 解析reward结果
            if len(metrics) >= 6:
                mean_rate = metrics.get('mean_rate', 0.0)
                mean_ast_match = metrics.get('mean_ast_match', 0.0)
                mean_dfg_match = metrics.get('mean_dfg_match', 0.0)
                mean_rate_ref = metrics.get('mean_rate_ref', 0.0)
                mean_ast_match_ref = metrics.get('mean_ast_match_ref', 0.0)
                mean_dfg_match_ref = metrics.get('mean_dfg_match_ref', 0.0)
                
                print(f"  📈 Reward指标:")
                print(f"    编译奖励: {mean_rate:.3f} (成功=1.0, 失败=-1.0)")
                print(f"    AST匹配度: {mean_ast_match:.3f}")
                print(f"    DFG匹配度: {mean_dfg_match:.3f}")
                print(f"    参考编译奖励: {mean_rate_ref:.3f}")
                print(f"    参考AST匹配度: {mean_ast_match_ref:.3f}")
                print(f"    参考DFG匹配度: {mean_dfg_match_ref:.3f}")
                
                # 计算总reward
                if hasattr(reward, 'item'):
                    total_reward = reward.item()
                else:
                    total_reward = float(reward)
                print(f"    总Reward: {total_reward:.3f}")
                
                # 编译状态判断
                compile_success = mean_rate > 0
                print(f"    编译状态: {'✅ 成功' if compile_success else '❌ 失败'}")
                
                # 代码质量评估
                if total_reward > 1:
                    quality = "🌟 优秀"
                elif total_reward > 0:
                    quality = "✅ 良好"
                elif total_reward > -1:
                    quality = "⚠️  一般"
                else:
                    quality = "❌ 差"
                print(f"    代码质量: {quality}")
            else:
                print(f"  ⚠️  Reward指标不完整: {metrics}")
            
            # 解码生成结果
            for i in range(response_ids.shape[0]):
                print(f"\n  📝 样本 {i + 1}:")
                
                # 解码输入
                input_text = trainer.tokenizer.decode(source_ids[i], skip_special_tokens=True)
                print(f"    输入预览: {input_text[:100]}...")
                
                # 解码生成结果
                generated_text = trainer.tokenizer.decode(response_ids[i], skip_special_tokens=True)
                print(f"    生成结果预览: {generated_text[:100]}...")
                
                # 解码参考结果
                ref_text = trainer.tokenizer.decode(response_ids_ref[i], skip_special_tokens=True)
                print(f"    参考结果预览: {ref_text[:100]}...")
                
                # 解码目标结果
                target_text = trainer.tokenizer.decode(target_ids[i], skip_special_tokens=True)
                print(f"    目标结果预览: {target_text[:100]}...")
                
                # 提取代码
                from optimized_rl_trainer import extract_code_from_qwen_response
                generated_code = extract_code_from_qwen_response(generated_text, 'cpp')
                ref_code = extract_code_from_qwen_response(ref_text, 'cpp')
                target_code = extract_code_from_qwen_response(target_text, 'cpp')
                
                print(f"    生成代码长度: {len(generated_code)} 字符")
                print(f"    参考代码长度: {len(ref_code)} 字符")
                print(f"    目标代码长度: {len(target_code)} 字符")
                
                if len(generated_code) > 0:
                    print(f"    生成代码预览: {generated_code[:100]}...")
                else:
                    print(f"    ⚠️  没有提取到生成代码")
                
                if len(ref_code) > 0:
                    print(f"    参考代码预览: {ref_code[:100]}...")
                else:
                    print(f"    ⚠️  没有提取到参考代码")
                
                if len(target_code) > 0:
                    print(f"    目标代码预览: {target_code[:100]}...")
                else:
                    print(f"    ⚠️  没有提取到目标代码")
                
                # 🔧 简化：只显示完整的生成代码
                print(f"\n  🔍 完整生成代码:")
                print("-" * 60)
                if len(generated_code) > 0:
                    print(generated_code)
                else:
                    print("没有提取到生成代码")
                print("-" * 60)
                
                # 🔧 简化：只显示完整的目标代码
                print(f"\n  🎯 完整目标代码:")
                print("-" * 60)
                if len(target_code) > 0:
                    print(target_code)
                else:
                    print("没有提取到目标代码")
                print("-" * 60)
                
        except Exception as e:
            print(f"  ❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 train_dataloader测试完成!")


def main():
    """主函数"""
    print("🚀 训练环境train_dataloader测试")
    print("=" * 80)
    
    # 设置随机种子
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行测试
    try:
        test_train_dataloader()
        
        print("\n🎉 测试成功完成!")
        
    except KeyboardInterrupt:
        print("\n⏹️  测试被用户中断")
    except Exception as e:
        print(f"\n💥 测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 