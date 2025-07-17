#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将现有的Java-C++数据转换为Qwen2.5-Coder SFT格式
"""

import json
import os
import argparse
from pathlib import Path

def convert_data_to_qwen_format(java_file, cpp_file, output_file, source_lang="Java", target_lang="C++"):
    """
    将配对的Java和C++文件转换为Qwen SFT格式
    """
    system_message = f"You are a helpful assistant for code translation. You specialize in translating {source_lang} code to {target_lang} code while maintaining functionality and best practices."
    
    converted_data = []
    
    with open(java_file, 'r', encoding='utf-8') as f_java, \
         open(cpp_file, 'r', encoding='utf-8') as f_cpp:
        
        for line_num, (java_line, cpp_line) in enumerate(zip(f_java, f_cpp), 1):
            java_code = java_line.strip().replace('▁', '_')
            cpp_code = cpp_line.strip().replace('▁', '_')
            
            if not java_code or not cpp_code:
                continue
                
            # 处理代码格式
            java_code = java_code.replace('NEW_LINE', '\n')
            cpp_code = cpp_code.replace('NEW_LINE', '\n')
            
            # 构建消息格式
            messages = [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user", 
                    "content": f"Translate the following {source_lang} code to {target_lang}:\n\n```{source_lang.lower()}\n{java_code}\n```"
                },
                {
                    "role": "assistant",
                    "content": f"Here's the {target_lang} translation:\n\n```{target_lang.lower()}\n{cpp_code}\n```"
                }
            ]
            
            # 添加到转换数据
            converted_data.append({
                "messages": messages,
                "format": "chatml"
            })
    
    # 保存转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"转换完成: {len(converted_data)} 个样本保存到 {output_file}")

def main():
    parser = argparse.ArgumentParser(description='转换数据为Qwen2.5-Coder SFT格式')
    parser.add_argument('--data_dir', type=str, default='data/Java-C++', help='数据目录')
    parser.add_argument('--output_dir', type=str, default='data/qwen_sft', help='输出目录')
    parser.add_argument('--source_lang', type=str, default='Java', help='源语言')
    parser.add_argument('--target_lang', type=str, default='C++', help='目标语言')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 转换训练集、验证集和测试集
    for split in ['train', 'val', 'test']:
        java_file = f"{args.data_dir}/{split}-Java-C++-tok.java"
        cpp_file = f"{args.data_dir}/{split}-Java-C++-tok.cpp"
        output_file = f"{args.output_dir}/{split}.jsonl"
        
        if os.path.exists(java_file) and os.path.exists(cpp_file):
            convert_data_to_qwen_format(java_file, cpp_file, output_file, args.source_lang, args.target_lang)
        else:
            print(f"警告: 找不到文件 {java_file} 或 {cpp_file}")

if __name__ == "__main__":
    main()