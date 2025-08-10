#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试生成功能
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def test_generation():
    """测试模型生成功能"""
    print("🚀 快速测试模型生成功能")
    print("=" * 60)
    
    # 加载模型和tokenizer
    model_path = "/home/cxy/Qwen2.5-Coder/finetuning/sft/checkpoints/qwen0.5b-lr5e-5-wr10-wd0.0-bsz1024-maxlen1280"
    
    print(f"📂 加载模型: {model_path}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='right'  # 与SFT训练保持一致
    )
    
    # 确保有pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Tokenizer配置:")
    print(f"   Padding side: {tokenizer.padding_side}")
    print(f"   Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"   EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"✅ 模型加载成功")
    
    # 测试生成
    test_prompt = "Translate Java to C++:\n\n```java\nclass Test {\n    public static void main(String[] args) {\n        System.out.println(\"Hello\");\n    }\n}\n```"
    
    print(f"\n📝 测试提示:")
    print(f"{test_prompt}")
    
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
    
    print(f"\n📊 输入信息:")
    print(f"   输入形状: {inputs['input_ids'].shape}")
    print(f"   Attention mask形状: {inputs['attention_mask'].shape}")
    print(f"   有效长度: {inputs['attention_mask'].sum().item()}")
    
    # 检查padding
    pad_positions = (inputs['input_ids'][0] == tokenizer.pad_token_id)
    mask_positions = (inputs['attention_mask'][0] == 0)
    
    print(f"   Padding tokens: {pad_positions.sum().item()}")
    print(f"   Mask zeros: {mask_positions.sum().item()}")
    
    if torch.equal(pad_positions, mask_positions):
        print(f"   ✅ Padding与attention mask一致")
    else:
        print(f"   ❌ Padding与attention mask不一致！")
    
    # 生成
    print(f"\n🤖 开始生成...")
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
    
    print(f"\n📝 生成结果:")
    print(f"{generated_text}")
    
    # 检查生成质量
    print(f"\n🔍 生成质量分析:")
    if "class" in generated_text.lower():
        print(f"   ✅ 包含class关键字")
    else:
        print(f"   ❌ 不包含class关键字")
        
    if "main" in generated_text.lower():
        print(f"   ✅ 包含main关键字")
    else:
        print(f"   ❌ 不包含main关键字")
        
    if "cout" in generated_text.lower() or "printf" in generated_text.lower():
        print(f"   ✅ 包含C++输出语句")
    else:
        print(f"   ❌ 不包含C++输出语句")
    
    # 检查生成长度
    original_length = inputs['input_ids'].shape[1]
    generated_length = outputs.shape[1]
    new_tokens = generated_length - original_length
    
    print(f"\n📊 生成长度分析:")
    print(f"   原始长度: {original_length} tokens")
    print(f"   生成后长度: {generated_length} tokens")
    print(f"   新生成: {new_tokens} tokens")
    
    if new_tokens > 1:
        print(f"   ✅ 生成了新内容")
    else:
        print(f"   ❌ 没有生成新内容")
    
    print(f"\n🎉 测试完成！")


if __name__ == "__main__":
    test_generation() 