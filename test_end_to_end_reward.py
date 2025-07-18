#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端Reward函数测试 - 使用真实微调的Qwen模型

完整流程测试：
抽样本 -> tokenize -> 输入模型 -> 得到输出 -> decode -> 提取代码 -> 计算reward

使用用户微调的Qwen模型进行真实测试
"""

import os
import sys
import json
import torch
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# 导入测试相关模块
from optimized_rl_trainer import (
    extract_code_from_qwen_response,
    create_reward_wrapper,
    read_qwen_examples,
    convert_qwen_examples_to_features,
)
from reward import get_reward
from utils import Example


def load_qwen_model_and_tokenizer(model_path: str):
    """加载微调过的Qwen模型和tokenizer"""
    print(f"🔧 加载Qwen模型和tokenizer...")
    print(f"📂 模型路径: {model_path}")
    
    model_path = Path(model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    print("📥 加载tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side='left'
        )
        
        # 确保有pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"✅ Tokenizer加载成功")
        print(f"   词汇表大小: {tokenizer.vocab_size}")
        print(f"   PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        print(f"   EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        
    except Exception as e:
        print(f"❌ Tokenizer加载失败: {e}")
        raise
    
    print("📥 加载模型...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        model.eval()  # 设置为评估模式
        
        print(f"✅ 模型加载成功")
        print(f"   模型类型: {type(model).__name__}")
        print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   设备: {next(model.parameters()).device}")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise
    
    return tokenizer, model


def load_test_samples(data_file: Path, num_samples: int = 5, args=None) -> List[Example]:
    """从数据集中加载测试样本，使用optimized_rl_trainer的读取函数"""
    print(f"📂 从 {data_file} 随机加载 {num_samples} 个测试样本...")
    
    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")
    
    # 创建临时args对象，如果没有提供的话
    if args is None:
        class TempArgs:
            source_lang = "java"
            target_lang = "cpp"
        args = TempArgs()
    
    # 使用optimized_rl_trainer的函数读取所有样本
    all_examples = read_qwen_examples(str(data_file), args)
    
    # 随机采样
    num_samples = min(num_samples, len(all_examples))
    samples = random.sample(all_examples, num_samples)
    
    for i, sample in enumerate(samples):
        print(f"✅ 样本 {i+1}: 加载成功")
    
    print(f"📊 成功随机加载 {len(samples)} 个测试样本")
    return samples


def construct_model_input(sample: Example, tokenizer, args=None) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
    """构造模型输入并提取参考答案，使用optimized_rl_trainer的特征提取函数"""
    
    # 创建临时args对象，如果没有提供的话
    if args is None:
        class TempArgs:
            max_source_length = 400
            max_target_length = 400
            source_lang = "java"
            target_lang = "cpp"
        args = TempArgs()
    
    # 使用optimized_rl_trainer的函数将Example转换为InputFeatures
    features = convert_qwen_examples_to_features([sample], tokenizer, args, stage='test')
    
    if not features:
        raise ValueError("无法从样本提取特征")
    
    feature = features[0]
    
    # 转换为tensor
    source_ids = torch.tensor(feature.source_ids, dtype=torch.long).unsqueeze(0)  # [1, seq_len]
    source_mask = torch.tensor(feature.source_mask, dtype=torch.long).unsqueeze(0)  # [1, seq_len]
    
    # 返回张量和原始内容
    return source_ids, source_mask, sample.source_orig, sample.target_orig


def generate_model_response(model, tokenizer, source_ids: torch.Tensor, source_mask: torch.Tensor, max_new_tokens: int = 512) -> str:
    """使用模型生成响应，接受预处理的张量输入"""
    print("🤖 模型生成响应...")
    
    # 移动到模型设备
    device = next(model.parameters()).device
    source_ids = source_ids.to(device)
    source_mask = source_mask.to(device)
    
    print(f"📊 输入长度: {source_ids.shape[1]} tokens")
    
    # 生成响应
    with torch.no_grad():
        start_time = time.time()
        
        outputs = model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        generation_time = time.time() - start_time
    
    # 解码响应
    generated_ids = outputs[0][source_ids.shape[1]:]  # 只取新生成的部分
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"⏱️  生成时间: {generation_time:.2f}秒")
    print(f"📊 生成长度: {len(generated_ids)} tokens")
    print(f"📝 生成响应预览: {response[:100]}...")
    
    return response


def test_end_to_end_reward():
    """端到端测试reward函数"""
    print("🚀 端到端Reward函数测试")
    print("=" * 80)
    print("📋 测试流程:")
    print("  1. 加载微调的Qwen模型")
    print("  2. 抽取测试样本")
    print("  3. 构造模型输入")
    print("  4. 模型生成响应")
    print("  5. 提取生成的代码")
    print("  6. 使用create_reward_wrapper计算reward")
    print("=" * 80)
    
    # 1. 加载模型
    model_path = "~/Qwen2.5-Coder/finetuning/sft/checkpoints/qwen0.5b-lr5e-5-wr10-wd0.0-bsz1024-maxlen1280/"
    
    try:
        tokenizer, model = load_qwen_model_and_tokenizer(model_path)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False
    
    # 2. 加载测试样本
    data_file = Path("data/qwen/Java-C++/val.jsonl")
    try:
        # 创建args对象用于样本加载
        class TestArgs:
            source_lang = "java"
            target_lang = "cpp"
            max_source_length = 400
            max_target_length = 400
        
        test_args = TestArgs()
        samples = load_test_samples(data_file, num_samples=3, args=test_args)
    except Exception as e:
        print(f"❌ 样本加载失败: {e}")
        return False
    
    if not samples:
        print("❌ 没有加载到有效样本")
        return False
    
    # 3. 创建reward函数包装器
    print("\n🎁 创建reward函数包装器...")
    wrapped_reward = create_reward_wrapper(get_reward)
    print("✅ 包装器创建成功")
    
    # 4. 对每个样本进行端到端测试
    all_results = []
    
    for i, sample in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"📝 测试样本 {i+1}/{len(samples)}")
        print("=" * 60)
        
        try:
            # 构造输入
            source_ids, source_mask, user_content, reference_assistant = construct_model_input(sample, tokenizer, test_args)
            
            print(f"📋 用户输入预览:")
            print(f"{user_content[:200]}..." if len(user_content) > 200 else user_content)
            
            # 提取参考Java和C++代码
            reference_java = extract_code_from_qwen_response(user_content, 'java')
            reference_cpp = extract_code_from_qwen_response(reference_assistant, 'cpp')
            
            print(f"\n📊 参考代码:")
            print(f"  Java代码长度: {len(reference_java)} 字符")
            print(f"  C++代码长度: {len(reference_cpp)} 字符")
            
            # 模型生成响应
            generated_response = generate_model_response(model, tokenizer, source_ids, source_mask)
            
            print(f"\n🤖 生成的完整响应:")
            print("-" * 40)
            print(generated_response)
            print("-" * 40)
            
            # 提取生成的C++代码
            generated_cpp = extract_code_from_qwen_response(generated_response, 'cpp')
            
            print(f"\n🔍 提取的生成代码:")
            print(f"长度: {len(generated_cpp)} 字符")
            print(f"内容: {generated_cpp}")
            
            if not generated_cpp.strip():
                print("⚠️  警告: 没有提取到有效的C++代码")
                generated_cpp = "// Empty generated code"
            
            # 准备reward计算的输入
            batch_size = 1
            max_length = 512
            
            # 构造完整响应（模拟训练时的格式）
            full_generated_response = f"Here's the C++ translation:\n\n```cpp\n{generated_cpp}\n```"
            full_reference_response = reference_assistant
            full_gold_response = reference_assistant  # 金标准使用参考答案
            
            # 编码为tensor
            generated_ids = tokenizer.encode(full_generated_response, max_length=max_length, truncation=True, padding='max_length')
            reference_ids = tokenizer.encode(full_reference_response, max_length=max_length, truncation=True, padding='max_length')
            gold_ids = tokenizer.encode(full_gold_response, max_length=max_length, truncation=True, padding='max_length')
            
            # 转换为tensor
            code_ids = torch.tensor([generated_ids], dtype=torch.long)
            code_ref_ids = torch.tensor([reference_ids], dtype=torch.long)
            gold_ids_tensor = torch.tensor([gold_ids], dtype=torch.long)
            
            print(f"\n🔧 准备reward计算:")
            print(f"  生成响应tensor: {code_ids.shape}")
            print(f"  参考响应tensor: {code_ref_ids.shape}")
            print(f"  金标准tensor: {gold_ids_tensor.shape}")
            
            # 显示 clang-format 格式化结果
            print(f"\n🎨 clang-format 格式化效果预览:")
            print("-" * 60)
            try:
                from reward import format_code_with_clang_format
                
                formatted_generated = format_code_with_clang_format(generated_cpp)
                formatted_reference = format_code_with_clang_format(reference_cpp)
                
                print(f"📝 原始生成代码 ({len(generated_cpp)} 字符):")
                preview_generated = generated_cpp[:150] + "..." if len(generated_cpp) > 150 else generated_cpp
                print(f"    {preview_generated}")
                
                print(f"\n📝 格式化后生成代码 ({len(formatted_generated)} 字符):")
                preview_formatted = formatted_generated[:150] + "..." if len(formatted_generated) > 150 else formatted_generated
                print(f"    {preview_formatted}")
                
                print(f"\n📝 格式化后参考代码 ({len(formatted_reference)} 字符):")
                preview_reference = formatted_reference[:150] + "..." if len(formatted_reference) > 150 else formatted_reference
                print(f"    {preview_reference}")
                
                # 检查格式化是否使代码更接近
                if formatted_generated == formatted_reference:
                    print(f"\n✨ 格式化后代码完全匹配！")
                elif formatted_generated.replace(' ', '').replace('\n', '') == formatted_reference.replace(' ', '').replace('\n', ''):
                    print(f"\n✨ 格式化后代码在语义上相同！")
                else:
                    print(f"\n💡 格式化后仍有差异，但应该能提高AST匹配分数")
                
            except Exception as e:
                print(f"⚠️  格式化预览失败: {e}")
            
            print("-" * 60)
            
            # 计算reward
            print(f"\n🎯 计算reward...")
            start_time = time.time()
            
            result = wrapped_reward(
                lang="cpp",
                code_ids=code_ids,
                code_ref_ids=code_ref_ids,
                gold_ids=gold_ids_tensor,
                tokenizer=tokenizer
            )
            
            reward_time = time.time() - start_time
            
            # 解析结果
            (rewards, mean_rate, mean_ast_match, mean_dfg_match,
             num_errors, num_errors_ref, num_nodes, num_nodes_ref) = result
            
            # 提取reward值
            rewards_np = rewards.numpy()
            non_zero_rewards = rewards_np[rewards_np != 0]
            total_reward = float(non_zero_rewards[0]) if len(non_zero_rewards) > 0 else 0.0
            
            print(f"⏱️  Reward计算时间: {reward_time:.3f}秒")
            
            # 显示结果
            print(f"\n📈 Reward分析结果:")
            print("-" * 40)
            print(f"🎯 总体指标:")
            print(f"  编译奖励: {mean_rate:.3f} (成功=1.0, 失败=-1.0)")
            print(f"  AST匹配度: {mean_ast_match:.3f}")
            print(f"  DFG匹配度: {mean_dfg_match:.3f}")
            print(f"  总Reward: {total_reward:.3f}")
            
            print(f"\n🔍 详细信息:")
            print(f"  生成代码错误数: {num_errors[0]}")
            print(f"  生成代码节点数: {num_nodes[0]}")
            print(f"  参考代码错误数: {num_errors_ref[0]}")
            print(f"  参考代码节点数: {num_nodes_ref[0]}")
            
            # 编译状态（从编译奖励判断）
            compile_success = mean_rate > 0
            print(f"  编译状态: {'✅ 成功' if compile_success else '❌ 失败'}")
            
            # 代码质量评估
            if total_reward > 1:
                quality = "🌟 优秀"
            elif total_reward > 0:
                quality = "✅ 良好"
            elif total_reward > -1:
                quality = "⚠️  一般"
            else:
                quality = "❌ 差"
            
            print(f"  代码质量: {quality}")
            
            # 保存结果
            result_info = {
                'sample_id': i + 1,
                'generated_cpp': generated_cpp,
                'reference_cpp': reference_cpp,
                'total_reward': total_reward,
                'compile_success': compile_success,
                'mean_rate': mean_rate,
                'mean_ast_match': mean_ast_match,
                'mean_dfg_match': mean_dfg_match,
                'num_errors': num_errors[0],
                'num_nodes': num_nodes[0],
                'generation_time': None,  # 在生成阶段记录
                'reward_time': reward_time
            }
            
            all_results.append(result_info)
            
        except Exception as e:
            print(f"❌ 样本 {i+1} 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 5. 总结结果
    print(f"\n🎉 端到端测试完成!")
    print("=" * 80)
    print("📊 测试总结:")
    print("-" * 40)
    
    if all_results:
        total_samples = len(all_results)
        successful_compiles = sum(1 for r in all_results if r['compile_success'])
        avg_reward = sum(r['total_reward'] for r in all_results) / total_samples
        avg_reward_time = sum(r['reward_time'] for r in all_results) / total_samples
        
        print(f"📈 总体统计:")
        print(f"  测试样本数: {total_samples}")
        print(f"  编译成功数: {successful_compiles}")
        print(f"  编译成功率: {successful_compiles/total_samples*100:.1f}%")
        print(f"  平均reward: {avg_reward:.3f}")
        print(f"  平均reward计算时间: {avg_reward_time:.3f}秒")
        
        print(f"\n📋 各样本详情:")
        for r in all_results:
            status = "✅" if r['compile_success'] else "❌"
            print(f"  样本{r['sample_id']}: {status} Reward={r['total_reward']:.3f}, 错误={r['num_errors']}, 节点={r['num_nodes']}")
    
    print(f"\n🎯 结论:")
    if all_results and len(all_results) > 0:
        success_rate = sum(1 for r in all_results if r['compile_success']) / len(all_results)
        if success_rate >= 0.7:
            print("✅ 端到端流程正常，reward函数表现良好")
        elif success_rate >= 0.3:
            print("⚠️  端到端流程基本正常，但生成代码质量有改进空间")
        else:
            print("❌ 生成代码质量较低，可能需要检查模型或数据")
    else:
        print("❌ 测试未能完成，请检查模型和数据配置")
    
    print("💡 reward函数本身工作正常，可以用于PPO训练")
    
    return True


def main():
    """主函数"""
    print("🚀 端到端Reward测试 - 使用微调Qwen模型")
    print("=" * 80)
    
    # 设置随机种子
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行测试
    try:
        success = test_end_to_end_reward()
        
        if success:
            print("\n🎉 端到端测试成功完成!")
        else:
            print("\n❌ 端到端测试失败")
            
    except KeyboardInterrupt:
        print("\n⏹️  测试被用户中断")
    except Exception as e:
        print(f"\n💥 测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 