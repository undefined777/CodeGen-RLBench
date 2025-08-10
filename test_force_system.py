#!/usr/bin/env python3
"""
测试强制添加system消息的功能
"""

import json
import torch
from transformers import AutoTokenizer
import os
from optimized_rl_trainer import read_qwen_examples, convert_qwen_examples_to_features, Example

def test_force_system():
    """测试强制添加system消息"""
    
    # 设置模型路径
    model_path = "/home/cxy/Qwen2.5-Coder/finetuning/sft/checkpoints/qwen0.5b-lr5e-5-wr10-wd0.0-bsz1024-maxlen1280"
    
    print("🔍 测试强制添加system消息")
    print(f"模型路径: {model_path}")
    
    try:
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 创建测试数据
        test_examples = []
        
        # 示例1：有system消息的数据
        example1 = Example(
            idx=0,
            source="test source code",
            target="test target code", 
            source_orig="Translate this Java code to C++",
            target_orig="Here's the C++ translation"
        )
        setattr(example1, "system_orig", "You are a helpful assistant for code translation.")
        test_examples.append(example1)
        
        # 示例2：无system消息的数据
        example2 = Example(
            idx=1,
            source="test source code 2",
            target="test target code 2",
            source_orig="Translate this Python code to Java", 
            target_orig="Here's the Java translation"
        )
        # 不设置system_orig，模拟没有system消息的情况
        test_examples.append(example2)
        
        # 创建简单的args对象
        class TempArgs:
            def __init__(self):
                self.max_source_length = 400
                self.max_target_length = 400
                self.source_lang = "java"
                self.target_lang = "cpp"
        
        args = TempArgs()
        
        print(f"\n🧪 测试convert_qwen_examples_to_features:")
        
        # 转换特征
        features = convert_qwen_examples_to_features(test_examples, tokenizer, args, stage='train')
        
        print(f"转换了 {len(features)} 个特征")
        
        # 检查每个特征
        for i, feature in enumerate(features):
            print(f"\n📝 特征 {i+1}:")
            
            # 解码source_ids查看内容
            source_text = tokenizer.decode(feature.source_ids, skip_special_tokens=True)
            print(f"   Source文本长度: {len(source_text)}")
            print(f"   Source文本前200字符: {source_text[:200]}...")
            
            # 检查是否包含system内容
            has_system = "system" in source_text.lower() or "assistant" in source_text.lower()
            print(f"   包含system内容: {has_system}")
            
            if has_system:
                print(f"   ✅ 成功添加了system消息")
            else:
                print(f"   ❌ 缺少system消息")
        
        print(f"\n🎯 测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_force_system() 