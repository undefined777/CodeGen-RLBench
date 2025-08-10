#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试输入构造方式
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from optimized_rl_trainer import convert_qwen_examples_to_features, Example


def debug_input_construction():
    """调试输入构造方式"""
    print("🔍 调试输入构造方式")
    print("=" * 60)
    
    # 加载模型和tokenizer
    model_path = "/home/cxy/Qwen2.5-Coder/finetuning/sft/checkpoints/qwen0.5b-lr5e-5-wr10-wd0.0-bsz1024-maxlen1280"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='right'
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 测试提示
    test_prompt = "Translate Java to C++:\n\n```java\nclass Test {\n    public static void main(String[] args) {\n        System.out.println(\"Hello\");\n    }\n}\n```"
    
    print(f"📝 测试提示:")
    print(f"{test_prompt}")
    
    # 方法1: 直接使用tokenizer (快速测试中的方法)
    print(f"\n🔧 方法1: 直接使用tokenizer")
    print("-" * 40)
    
    inputs1 = tokenizer(
        test_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=200
    )
    
    inputs1 = {k: v.to(model.device) for k, v in inputs1.items()}
    
    print(f"   输入形状: {inputs1['input_ids'].shape}")
    print(f"   Attention mask形状: {inputs1['attention_mask'].shape}")
    print(f"   有效长度: {inputs1['attention_mask'].sum().item()}")
    
    # 检查padding
    pad_positions1 = (inputs1['input_ids'][0] == tokenizer.pad_token_id)
    mask_positions1 = (inputs1['attention_mask'][0] == 0)
    
    print(f"   Padding tokens: {pad_positions1.sum().item()}")
    print(f"   Mask zeros: {mask_positions1.sum().item()}")
    
    if torch.equal(pad_positions1, mask_positions1):
        print(f"   ✅ Padding与attention mask一致")
    else:
        print(f"   ❌ Padding与attention mask不一致！")
    
    # 生成测试1
    print(f"\n🤖 生成测试1...")
    with torch.no_grad():
        outputs1 = model.generate(
            **inputs1,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text1 = tokenizer.decode(outputs1[0], skip_special_tokens=True)
    new_tokens1 = outputs1.shape[1] - inputs1['input_ids'].shape[1]
    
    print(f"   生成长度: {new_tokens1} tokens")
    print(f"   生成内容: {generated_text1}")
    
    # 方法2: 使用convert_qwen_examples_to_features (端到端测试中的方法)
    print(f"\n🔧 方法2: 使用convert_qwen_examples_to_features")
    print("-" * 40)
    
    # 创建Example对象
    example = Example(
        idx=0,
        source=test_prompt,
        target="",
        source_orig=test_prompt,
        target_orig=""
    )
    
    # 创建args对象
    class TempArgs:
        max_source_length = 400
        max_target_length = 400
        source_lang = "java"
        target_lang = "cpp"
    
    args = TempArgs()
    
    # 使用特征提取函数
    features = convert_qwen_examples_to_features([example], tokenizer, args, stage='test')
    
    if features:
        feature = features[0]
        
        # 转换为tensor
        source_ids = torch.tensor(feature.source_ids, dtype=torch.long).unsqueeze(0)
        source_mask = torch.tensor(feature.source_mask, dtype=torch.long).unsqueeze(0)
        
        # 移动到设备
        source_ids = source_ids.to(model.device)
        source_mask = source_mask.to(model.device)
        
        print(f"   输入形状: {source_ids.shape}")
        print(f"   Attention mask形状: {source_mask.shape}")
        print(f"   有效长度: {source_mask.sum().item()}")
        
        # 检查padding
        pad_positions2 = (source_ids[0] == tokenizer.pad_token_id)
        mask_positions2 = (source_mask[0] == 0)
        
        print(f"   Padding tokens: {pad_positions2.sum().item()}")
        print(f"   Mask zeros: {mask_positions2.sum().item()}")
        
        if torch.equal(pad_positions2, mask_positions2):
            print(f"   ✅ Padding与attention mask一致")
        else:
            print(f"   ❌ Padding与attention mask不一致！")
        
        # 生成测试2
        print(f"\n🤖 生成测试2...")
        with torch.no_grad():
            outputs2 = model.generate(
                input_ids=source_ids,
                attention_mask=source_mask,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_text2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)
        new_tokens2 = outputs2.shape[1] - source_ids.shape[1]
        
        print(f"   生成长度: {new_tokens2} tokens")
        print(f"   生成内容: {generated_text2}")
        
        # 比较两种方法
        print(f"\n📊 方法比较:")
        print(f"   方法1生成长度: {new_tokens1} tokens")
        print(f"   方法2生成长度: {new_tokens2} tokens")
        
        if new_tokens1 > 1 and new_tokens2 <= 1:
            print(f"   ❌ 方法2有问题！")
        elif new_tokens1 <= 1 and new_tokens2 > 1:
            print(f"   ❌ 方法1有问题！")
        elif new_tokens1 > 1 and new_tokens2 > 1:
            print(f"   ✅ 两种方法都正常")
        else:
            print(f"   ❌ 两种方法都有问题！")
    
    print(f"\n🎉 调试完成！")


if __name__ == "__main__":
    debug_input_construction() 