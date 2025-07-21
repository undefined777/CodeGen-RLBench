#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试PPO修复效果的简单脚本
"""
import torch
import numpy as np
from reward import get_reward
from optimized_rl_trainer import create_reward_wrapper

def test_reward_distribution():
    """测试新的奖励分布"""
    print("🧪 测试奖励分布修复效果")
    print("=" * 50)
    
    # 创建模拟数据
    batch_size = 3
    seq_len = 10
    
    # 模拟generated code (带EOS)
    code_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    # 在位置7设置EOS (假设)
    eos_token_id = 151645
    code_ids[0, 7] = eos_token_id
    code_ids[1, 5] = eos_token_id  
    code_ids[2, 9] = eos_token_id
    
    # 模拟参考代码
    code_ref_ids = torch.zeros_like(code_ids)
    code_ref_ids[0, 6] = eos_token_id
    code_ref_ids[1, 8] = eos_token_id
    code_ref_ids[2, 7] = eos_token_id
    
    # 模拟金标准
    gold_ids = torch.zeros_like(code_ids)
    gold_ids[0, 8] = eos_token_id
    gold_ids[1, 6] = eos_token_id
    gold_ids[2, 8] = eos_token_id
    
    # 创建mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.eos_token_id = eos_token_id
            self.pad_token_id = 0
            
        def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            # 简单的解码，返回模拟的C++代码
            return "int main() { return 0; }"
            
        def encode(self, text, add_special_tokens=False):
            return [1, 2, 3, 151645]  # 简单的编码
    
    tokenizer = MockTokenizer()
    
    # 创建奖励包装器
    wrapped_reward = create_reward_wrapper(get_reward)
    
    try:
        # 计算奖励
        result = wrapped_reward(
            lang="cpp",
            code_ids=code_ids,
            code_ref_ids=code_ref_ids,
            gold_ids=gold_ids,
            tokenizer=tokenizer
        )
        
        rewards, mean_rate, mean_ast_match, mean_dfg_match, num_errors, num_errors_ref, num_nodes, num_nodes_ref = result
        
        print(f"📊 奖励分布测试结果:")
        print(f"  Rewards shape: {rewards.shape}")
        print(f"  Non-zero elements: {torch.count_nonzero(rewards).item()}")
        print(f"  Total elements: {rewards.numel()}")
        print(f"  Non-zero ratio: {torch.count_nonzero(rewards).item() / rewards.numel():.3f}")
        print(f"  Rewards mean: {rewards.mean().item():.6f}")
        print(f"  Rewards std: {rewards.std().item():.6f}")
        
        print(f"\n🎯 具体奖励值分布:")
        for i in range(batch_size):
            non_zero_positions = torch.nonzero(rewards[i]).flatten()
            non_zero_values = rewards[i][non_zero_positions]
            print(f"  Batch {i}: 非零位置 {non_zero_positions.tolist()}, 值 {non_zero_values.tolist()}")
            
        print(f"\n✅ 测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advantage_normalization():
    """测试advantage归一化修复"""
    print("\n🧪 测试Advantage归一化修复效果")
    print("=" * 50)
    
    from utils import whiten
    
    # 测试不同标准差的情况
    test_cases = [
        ("高方差", torch.tensor([1.0, 5.0, -2.0, 3.0, -1.0])),
        ("低方差", torch.tensor([1.001, 1.002, 0.999, 1.000, 1.001])),
        ("极低方差", torch.tensor([1.0000001, 1.0000002, 0.9999999, 1.0000000, 1.0000001])),
        ("全零", torch.zeros(5)),
        ("常数", torch.ones(5) * 2.5)
    ]
    
    for name, values in test_cases:
        print(f"\n📊 测试案例: {name}")
        print(f"  原始值: {values}")
        print(f"  原始 mean: {values.mean().item():.8f}")
        print(f"  原始 std: {values.std().item():.8f}")
        
        whitened = whiten(values)
        print(f"  白化后: {whitened}")
        print(f"  白化后 mean: {whitened.mean().item():.8f}")
        print(f"  白化后 std: {whitened.std().item():.8f}")

if __name__ == "__main__":
    print("🚀 PPO修复效果测试")
    print("=" * 60)
    
    # 测试奖励分布
    success1 = test_reward_distribution()
    
    # 测试advantage归一化
    test_advantage_normalization()
    
    print(f"\n🎉 测试总结:")
    print(f"  奖励分布测试: {'✅ 通过' if success1 else '❌ 失败'}")
    print(f"  建议: 运行实际训练观察policy loss和advantages是否有改善") 