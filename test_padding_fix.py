#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试padding修复是否有效

验证right-padding配置的一致性
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from optimized_rl_trainer import convert_qwen_examples_to_features, Example


def test_padding_consistency():
    """测试padding配置的一致性"""
    print("🔍 测试padding配置一致性")
    print("=" * 60)
    
    # 加载模型和tokenizer
    model_path = "/home/cxy/Qwen2.5-Coder/finetuning/sft/checkpoints/qwen0.5b-lr5e-5-wr10-wd0.0-bsz1024-maxlen1280"
    
    print(f"📂 加载模型: {model_path}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='right'  # 使用right-padding
    )
    
    # 确保有pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Tokenizer配置:")
    print(f"   Padding side: {tokenizer.padding_side}")
    print(f"   Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"   EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    
    # 测试基本padding功能
    print(f"\n🧪 测试基本padding功能:")
    test_texts = [
        "Hello world",
        "Translate Java to C++",
        "This is a longer text for testing padding functionality"
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\n   测试 {i+1}: {text}")
        
        # 编码
        encoded = tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=50, 
            return_tensors="pt"
        )
        
        input_ids = encoded['input_ids'][0]
        attention_mask = encoded['attention_mask'][0]
        
        # 检查padding位置
        pad_positions = (input_ids == tokenizer.pad_token_id)
        mask_positions = (attention_mask == 0)
        
        print(f"   输入长度: {len(input_ids)}")
        print(f"   有效长度: {attention_mask.sum().item()}")
        print(f"   Padding位置: {pad_positions.sum().item()} tokens")
        
        if torch.equal(pad_positions, mask_positions):
            print(f"   ✅ Padding与attention mask一致")
        else:
            print(f"   ❌ Padding与attention mask不一致！")
            
        # 解码验证
        decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
        print(f"   解码结果: {decoded}")
    
    # 测试特征提取函数
    print(f"\n🧪 测试特征提取函数:")
    
    # 创建测试样本
    class TempArgs:
        max_source_length = 100
        max_target_length = 100
        source_lang = "java"
        target_lang = "cpp"
    
    test_example = Example(
        source_orig="Translate Java to C++: class Test { }",
        target_orig="Here's the C++ translation:\n\n```cpp\nclass Test { };\n```",
        system_orig="You are a helpful assistant for code translation."
    )
    
    features = convert_qwen_examples_to_features([test_example], tokenizer, TempArgs(), stage='test')
    
    if features:
        feature = features[0]
        print(f"   特征提取成功")
        print(f"   Source长度: {len(feature.source_ids)}")
        print(f"   Target长度: {len(feature.target_ids)}")
        
        # 检查padding
        source_pad_count = sum(1 for x in feature.source_ids if x == tokenizer.pad_token_id)
        target_pad_count = sum(1 for x in feature.target_ids if x == tokenizer.pad_token_id)
        
        print(f"   Source padding: {source_pad_count} tokens")
        print(f"   Target padding: {target_pad_count} tokens")
        
        # 检查padding位置（应该是末尾）
        source_pad_positions = [i for i, x in enumerate(feature.source_ids) if x == tokenizer.pad_token_id]
        target_pad_positions = [i for i, x in enumerate(feature.target_ids) if x == tokenizer.pad_token_id]
        
        print(f"   Source padding位置: {source_pad_positions}")
        print(f"   Target padding位置: {target_pad_positions}")
        
        # 验证right-padding（padding应该在末尾）
        if source_pad_positions and max(source_pad_positions) == len(feature.source_ids) - 1:
            print(f"   ✅ Source使用right-padding")
        else:
            print(f"   ❌ Source padding位置不正确")
            
        if target_pad_positions and max(target_pad_positions) == len(feature.target_ids) - 1:
            print(f"   ✅ Target使用right-padding")
        else:
            print(f"   ❌ Target padding位置不正确")
    else:
        print(f"   ❌ 特征提取失败")
    
    print(f"\n🎉 Padding一致性测试完成")


def test_model_generation():
    """测试模型生成功能"""
    print(f"\n🤖 测试模型生成功能")
    print("=" * 60)
    
    model_path = "/home/cxy/Qwen2.5-Coder/finetuning/sft/checkpoints/qwen0.5b-lr5e-5-wr10-wd0.0-bsz1024-maxlen1280"
    
    try:
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side='right'
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✅ 模型和tokenizer加载成功")
        
        # 测试生成
        test_prompt = "Translate Java to C++:\n\n```java\nclass Test {\n    public static void main(String[] args) {\n        System.out.println(\"Hello\");\n    }\n}\n```"
        
        print(f"📝 测试提示: {test_prompt[:100]}...")
        
        # 编码
        inputs = tokenizer(
            test_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=200
        )
        
        # 移动到GPU
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        print(f"📊 输入形状: {inputs['input_ids'].shape}")
        print(f"📊 Attention mask形状: {inputs['attention_mask'].shape}")
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🤖 生成结果: {generated_text}")
        
        # 检查生成质量
        if "class" in generated_text.lower() and "main" in generated_text.lower():
            print(f"✅ 生成内容看起来合理")
        else:
            print(f"⚠️  生成内容可能有问题")
            
    except Exception as e:
        print(f"❌ 模型生成测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 开始padding修复验证测试")
    print("=" * 80)
    
    # 测试padding一致性
    test_padding_consistency()
    
    # 测试模型生成
    test_model_generation()
    
    print(f"\n🎉 所有测试完成！")
    print("=" * 80) 