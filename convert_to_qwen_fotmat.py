#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert the existing Java-C++ data to Qwen2.5-Coder SFT format
"""

import json
import os
import argparse
from pathlib import Path

def convert_data_to_qwen_format(source_file, target_file, output_file, source_lang, target_lang):
    """
    Convert the paired source language and target language files to Qwen SFT format
    Support multiple language pairs: Java-C++, C-Python, Python-Javascript
    """
    system_message = f"You are a helpful assistant for code translation. You specialize in translating {source_lang} code to {target_lang} code while maintaining functionality and best practices."
    
    converted_data = []
    
    with open(source_file, 'r', encoding='utf-8') as f_source, \
         open(target_file, 'r', encoding='utf-8') as f_target:
        
        for line_num, (source_line, target_line) in enumerate(zip(f_source, f_target), 1):
            source_code = source_line.strip().replace('▁', '_')
            target_code = target_line.strip().replace('▁', '_')
            
            if not source_code or not target_code:
                continue
                
            # Process code format - adapt to different language's special marks
            source_code = process_code_format(source_code, source_lang)
            target_code = process_code_format(target_code, target_lang)
            
            # Build message format
            messages = [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user", 
                    "content": f"Translate the following {source_lang} code to {target_lang}:\n\n```{get_code_extension(source_lang)}\n{source_code}\n```"
                },
                {
                    "role": "assistant",
                    "content": f"Here's the {target_lang} translation:\n\n```{get_code_extension(target_lang)}\n{target_code}\n```"
                }
            ]
            
            # Add to converted data
            converted_data.append({
                "messages": messages,
                "format": "chatml"
            })
    
    # Save converted data
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Conversion completed: {len(converted_data)} samples saved to {output_file}")

def process_code_format(code, lang):
    """
    Process different language's code format marks
    """
    # Process common format marks
    code = code.replace('NEW_LINE', '\n')
    
    # Python's special indent marks
    if lang.lower() == 'python':
        # Process Python's indent marks
        lines = code.split('\n')
        processed_lines = []
        indent_level = 0
        
        for line in lines:
            # Process INDENT and DEDENT marks
            while 'DEDENT' in line:
                indent_level = max(0, indent_level - 1)
                line = line.replace('DEDENT', '', 1).strip()
            
            if 'INDENT' in line:
                line = line.replace('INDENT', '').strip()
                if line:  # If this line has other content
                    processed_lines.append('    ' * indent_level + line)
                indent_level += 1
            else:
                if line.strip():  # Only process non-empty lines
                    processed_lines.append('    ' * indent_level + line.strip())
        
        code = '\n'.join(processed_lines)
    
    return code

def get_code_extension(lang):
    """
    Get the code block identifier corresponding to the language
    """
    lang_map = {
        'Java': 'java',
        'C++': 'cpp', 
        'C': 'c',
        'Python': 'python',
        'Javascript': 'javascript'
    }
    return lang_map.get(lang, lang.lower())

def get_file_extension(lang):
    """
    Get the file extension corresponding to the language
    """
    ext_map = {
        'Java': 'java',
        'C++': 'cpp',
        'C': 'c', 
        'Python': 'py',
        'Javascript': 'js'
    }
    return ext_map.get(lang, lang.lower())

def detect_and_convert_datasets(data_root='data'):
    """
    Automatically detect all available datasets and convert them
    """
    print("🔍 Scanning available code translation datasets...")
    
    # Define supported language pairs
    language_pairs = [
        ('C++', 'Python'),
        ('Java', 'Python')
    ]
    
    converted_count = 0
    
    for source_lang, target_lang in language_pairs:
        data_dir = f"{data_root}/{source_lang}-{target_lang}"
        
        if not os.path.exists(data_dir):
            print(f"⚠️  Dataset directory does not exist: {data_dir}")
            continue
            
        print(f"\n📂 Processing dataset: {source_lang} → {target_lang}")
        
        # Create output directory
        output_dir = f"{data_root}/qwen/{source_lang}-{target_lang}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert training, validation and test sets
        for split in ['train', 'val', 'test']:
            source_ext = get_file_extension(source_lang)
            target_ext = get_file_extension(target_lang)
            
            source_file = f"{data_dir}/{split}-{source_lang}-{target_lang}-tok.{source_ext}"
            target_file = f"{data_dir}/{split}-{source_lang}-{target_lang}-tok.{target_ext}"
            output_file = f"{output_dir}/{split}.jsonl"
            
            if os.path.exists(source_file) and os.path.exists(target_file):
                print(f"  ✅ Converting {split} set...")
                convert_data_to_qwen_format(source_file, target_file, output_file, source_lang, target_lang)
                converted_count += 1
            else:
                print(f"  ❌ File not found: {source_file} or {target_file}")
    
    print(f"\n🎉 Conversion completed! Total {converted_count} data files processed")
    return converted_count

def main():
    parser = argparse.ArgumentParser(description='Convert code translation data to Qwen2.5-Coder SFT format')
    parser.add_argument('--data_root', type=str, default='data', help='Data root directory')
    parser.add_argument('--auto_detect', action='store_true', help='Automatically detect and convert all available datasets')
    parser.add_argument('--data_dir', type=str, help='Specific data directory (e.g. data/Java-C++)')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--source_lang', type=str, help='Source language (e.g. Java, C, Python)')
    parser.add_argument('--target_lang', type=str, help='Target language (e.g. C++, Python, Javascript)')
    
    args = parser.parse_args()
    
    if args.auto_detect:
        # Automatically detect and convert all available datasets
        detect_and_convert_datasets(args.data_root)
    elif args.data_dir and args.source_lang and args.target_lang:
        # Manually specify a single dataset
        if not args.output_dir:
            args.output_dir = f"{args.data_root}/qwen/{args.source_lang}-{args.target_lang}"
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Convert training, validation and test sets
        for split in ['train', 'val', 'test']:
            source_ext = get_file_extension(args.source_lang)
            target_ext = get_file_extension(args.target_lang)
            
            source_file = f"{args.data_dir}/{split}-{args.source_lang}-{args.target_lang}-tok.{source_ext}"
            target_file = f"{args.data_dir}/{split}-{args.source_lang}-{args.target_lang}-tok.{target_ext}"
            output_file = f"{args.output_dir}/{split}.jsonl"
            
            if os.path.exists(source_file) and os.path.exists(target_file):
                convert_data_to_qwen_format(source_file, target_file, output_file, args.source_lang, args.target_lang)
            else:
                print(f"Warning: File not found: {source_file} or {target_file}")
    else:
        print("Please use --auto_detect to automatically convert all available datasets, or specify --data_dir, --source_lang, --target_lang parameters")
        parser.print_help()

if __name__ == "__main__":
    main()