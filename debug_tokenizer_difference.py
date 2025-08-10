#!/usr/bin/env python3
"""
调试tokenizer差异的脚本
检查本地和服务器环境的apply_chat_template行为
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def test_tokenizer_behavior():
    """测试tokenizer的apply_chat_template行为"""
    
    # 模拟数据
    system_message = "You are a helpful assistant for code translation. You specialize in translating Java code to C++ code while maintaining functionality and best practices."
    user_message = "Translate the following Java code to C++:\n\n```java\nclass Node { int data ; Node next ; Node ( int d ) { data = d ; next = null ; } } class LinkedList { Node head ; void push ( int new_data ) { Node new_node = new Node ( new_data ) ; new_node . next = head ; head = new_node ; } }```"
    
    print("🔍 环境信息:")
    print(f"   Transformers版本: {torch.__version__}")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    
    # 检查模型路径
    model_path = "/home/cxy/Qwen2.5-Coder/finetuning/sft/checkpoints/qwen0.5b-lr5e-5-wr10-wd0.0-bsz1024-maxlen1280"
    print(f"\n📁 模型路径: {model_path}")
    print(f"   路径存在: {os.path.exists(model_path)}")
    
    # 检查tokenizer配置文件
    tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
    print(f"   tokenizer_config.json存在: {os.path.exists(tokenizer_config_path)}")
    
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, 'r') as f:
            config = json.load(f)
        print(f"   chat_template存在: {'chat_template' in config}")
        if 'chat_template' in config:
            print(f"   chat_template长度: {len(config['chat_template'])}")
    
    try:
        # 加载tokenizer
        print(f"\n🔧 加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        print(f"   hasattr(tokenizer, 'apply_chat_template'): {hasattr(tokenizer, 'apply_chat_template')}")
        print(f"   tokenizer.chat_template存在: {hasattr(tokenizer, 'chat_template')}")
        
        if hasattr(tokenizer, 'chat_template'):
            print(f"   chat_template长度: {len(tokenizer.chat_template)}")
            print(f"   chat_template前100字符: {tokenizer.chat_template[:100]}...")
        
        # 测试apply_chat_template
        print(f"\n🧪 测试apply_chat_template:")
        
        # 方法1: 有system消息
        messages_with_system = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        
        try:
            result_with_system = tokenizer.apply_chat_template(
                messages_with_system, add_generation_prompt=True, tokenize=False
            )
            print(f"   ✅ 有system消息 - 成功")
            print(f"   结果长度: {len(result_with_system)}")
            print(f"   结果前200字符: {result_with_system[:200]}...")
            print(f"   包含'system': {'system' in result_with_system}")
        except Exception as e:
            print(f"   ❌ 有system消息 - 失败: {e}")
        
        # 方法2: 无system消息
        messages_without_system = [
            {"role": "user", "content": user_message},
        ]
        
        try:
            result_without_system = tokenizer.apply_chat_template(
                messages_without_system, add_generation_prompt=True, tokenize=False
            )
            print(f"   ✅ 无system消息 - 成功")
            print(f"   结果长度: {len(result_without_system)}")
            print(f"   结果前200字符: {result_without_system[:200]}...")
        except Exception as e:
            print(f"   ❌ 无system消息 - 失败: {e}")
        
        # 方法3: 直接使用原始内容
        print(f"\n📝 直接使用原始内容:")
        print(f"   原始user_message长度: {len(user_message)}")
        print(f"   原始user_message前200字符: {user_message[:200]}...")
        
        # 对比结果
        if 'result_with_system' in locals() and 'result_without_system' in locals():
            print(f"\n🔍 结果对比:")
            print(f"   有system vs 无system长度差异: {len(result_with_system) - len(result_without_system)}")
            print(f"   有system vs 原始内容长度差异: {len(result_with_system) - len(user_message)}")
            
    except Exception as e:
        print(f"❌ 加载tokenizer失败: {e}")

if __name__ == "__main__":
    test_tokenizer_behavior() 