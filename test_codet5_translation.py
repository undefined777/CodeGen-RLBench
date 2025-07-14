#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CodeT5模型Java到C++代码翻译测试脚本
测试微调过的CodeT5模型在Java到C++代码翻译任务上的性能
"""

import torch
import argparse
import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, T5Config
from torch import nn
from model import respond_to_batch, CodeT5HeadWithValueModelLocal

# 添加CodeBLEU路径
sys.path.append('./codebleu')
try:
    from codebleu.calc_code_bleu import calc_code_bleu
    CODEBLEU_AVAILABLE = True
    print("CodeBLEU模块加载成功")
except ImportError as e:
    print(f"CodeBLEU模块加载失败: {e}")
    print("将跳过CodeBLEU评估")
    CODEBLEU_AVAILABLE = False


class CodeTranslationTester:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化测试器
        
        Args:
            model_path: 模型文件路径 (.bin, .pt 或 .pth)，必须提供
            device: 运行设备
        """
        self.device = device
        self.model_path = model_path
        
        # 获取模型文件所在目录
        self.model_dir = Path(model_path).parent
        
        # 检查并准备tokenizer和配置文件
        self._prepare_model_files()
        
        print(f"正在加载模型到设备: {device}")
        print(f"加载模型文件: {model_path}")
        
        # 初始化模型结构（不加载预训练权重）
        config_path = self.model_dir / 'config.json'
        self.model = CodeT5HeadWithValueModelLocal(config_path)
        
        # 加载用户提供的模型权重
        self.model.load_model_weights(model_path, device)
            
        self.model.to(device)
        self.model.eval()
        print("模型加载完成！")
        
    def _prepare_model_files(self):
        """检查模型必要文件是否存在"""
        print("检查模型文件...")
        
        # 检查模型权重文件
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型权重文件不存在: {self.model_path}")
        
        # 检查必要文件是否存在
        required_files = [
            'config.json',
            'tokenizer.json',
            'vocab.json',
            'merges.txt',
            'special_tokens_map.json'
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = self.model_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            raise FileNotFoundError(
                f"缺少必要的模型文件: {missing_files}\n"
                f"请确保模型目录 {self.model_dir} 包含所有必要文件:\n"
                f"  - config.json (模型配置)\n"
                f"  - tokenizer.json (分词器配置)\n"
                f"  - vocab.json (词汇表)\n"
                f"  - merges.txt (BPE合并规则)\n"
                f"  - special_tokens_map.json (特殊token映射)\n"
                f"  - {Path(self.model_path).name} (模型权重)"
            )
        
        print("✓ 所有必要文件检查通过")
        
        # 从本地加载tokenizer
        print("正在从本地加载tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
            print("tokenizer从本地加载完成！")
        except Exception as e:
            raise RuntimeError(f"从本地加载tokenizer失败: {e}")
        
    def prepare_test_samples(self):
        """准备测试样本 - 专门用于Java到C++翻译，包含gold代码"""
        # 准备Java到C++的代码翻译测试样本，每个样本包含gold代码
        test_samples = [
            # 简单加法函数
            {
                'source': 'public int add(int a, int b) { return a + b; }',
                'gold': 'int add ( int a, int b ) { return a + b ; }',
                'description': 'Java简单加法函数转C++'
            },
            # 阶乘函数
            {
                'source': 'public int factorial(int n) { if (n <= 1) return 1; return n * factorial(n - 1); }',
                'gold': 'int factorial(int n) { if (n <= 1) return 1; return n * factorial(n - 1); }',
                'description': 'Java递归阶乘函数转C++'
            },
            # 斐波那契函数
            {
                'source': 'public int fibonacci(int n) { if (n <= 1) return n; return fibonacci(n-1) + fibonacci(n-2); }',
                'gold': 'int fibonacci(int n) { if (n <= 1) return n; return fibonacci(n - 1) + fibonacci(n - 2); }',
                'description': 'Java斐波那契函数转C++'
            },
            # 数组最大值
            {
                'source': 'public int findMax(int[] arr) { int max = arr[0]; for (int i = 1; i < arr.length; i++) { if (arr[i] > max) max = arr[i]; } return max; }',
                'gold': 'int findMax(int arr[], int n) { int max = arr[0]; for (int i = 1; i < n; i++) { if (arr[i] > max) max = arr[i]; } return max; }',
                'description': 'Java数组最大值函数转C++'
            },
            # 冒泡排序
            {
                'source': 'public void bubbleSort(int[] arr) { int n = arr.length; for (int i = 0; i < n-1; i++) { for (int j = 0; j < n-i-1; j++) { if (arr[j] > arr[j+1]) { int temp = arr[j]; arr[j] = arr[j+1]; arr[j+1] = temp; } } } }',
                'gold': 'void bubbleSort(int arr[], int n) { for (int i = 0; i < n - 1; i++) { for (int j = 0; j < n - i - 1; j++) { if (arr[j] > arr[j + 1]) { int temp = arr[j]; arr[j] = arr[j + 1]; arr[j + 1] = temp; } } } }',
                'description': 'Java冒泡排序函数转C++'
            },
            # 字符串长度
            {
                'source': 'public int getStringLength(String str) { return str.length(); }',
                'gold': 'int getStringLength(string str) { return str.length(); }',
                'description': 'Java字符串长度函数转C++'
            },
            # 循环求和
            {
                'source': 'public int sum(int[] numbers) { int total = 0; for (int num : numbers) { total += num; } return total; }',
                'gold': 'int sum(int numbers[], int n) { int total = 0; for (int i = 0; i < n; i++) { total += numbers[i]; } return total; }',
                'description': 'Java数组求和函数转C++'
            },
            # 判断偶数
            {
                'source': 'public boolean isEven(int number) { return number % 2 == 0; }',
                'gold': 'bool isEven(int number) { return number % 2 == 0; }',
                'description': 'Java判断偶数函数转C++'
            },
            # 测试长代码
            {
                'source': 'import java . util . * ; class GFG { static class pair { int first , second ; public pair ( int first , int second ) { this . first = first ; this . second = second ; } } static int __gcd ( int a , int b ) { return b == 0 ? a : __gcd ( b , a % b ) ; } static void sumOfSquares ( int n , Vector < pair > vp ) { for ( int i = 1 ; i <= Math . sqrt ( n ) ; i ++ ) { int h = n - i * i , h1 = ( int ) Math . sqrt ( h ) ; if ( h1 * h1 == h ) { int a = Math . max ( h1 , i ) , b = Math . min ( h1 , i ) ; if ( vp . size ( ) == 1 && a != vp . get ( 0 ) . first ) vp . add ( new pair ( a , b ) ) ; if ( vp . size ( ) == 0 ) vp . add ( new pair ( a , b ) ) ; if ( vp . size ( ) == 2 ) return ; } } } static void findFactors ( int n ) { Vector < pair > vp = new Vector < > ( ) ; sumOfSquares ( n , vp ) ; if ( vp . size ( ) != 2 ) System . out . print ( " Factors ▁ Not ▁ Possible " ) ; int a , b , c , d ; a = vp . get ( 0 ) . first ; b = vp . get ( 0 ) . second ; c = vp . get ( 1 ) . first ; d = vp . get ( 1 ) . second ; if ( a < c ) { int t = a ; a = c ; c = t ; t = b ; b = d ; d = t ; } int k , h , l , m ; k = __gcd ( a - c , d - b ) ; h = __gcd ( a + c , d + b ) ; l = ( a - c ) / k ; m = ( d - b ) / k ; System . out . print ( " a ▁ = ▁ " + a + " TABSYMBOL TABSYMBOL ( A ) ▁ a ▁ - ▁ c ▁ = ▁ " + ( a - c ) + " TABSYMBOL TABSYMBOL k ▁ = ▁ gcd [ A , ▁ C ] ▁ = ▁ " + k + "NEW_LINE"); System . out . print ( " b ▁ = ▁ " + b + " TABSYMBOL TABSYMBOL ( B ) ▁ a ▁ + ▁ c ▁ = ▁ " + ( a + c ) + " TABSYMBOL TABSYMBOL h ▁ = ▁ gcd [ B , ▁ D ] ▁ = ▁ " + h + "NEW_LINE"); System . out . print ( " c ▁ = ▁ " + c + " TABSYMBOL TABSYMBOL ( C ) ▁ d ▁ - ▁ b ▁ = ▁ " + ( d - b ) + " TABSYMBOL TABSYMBOL l ▁ = ▁ A / k ▁ = ▁ " + l + "NEW_LINE"); System . out . print ( " d ▁ = ▁ " + d + " TABSYMBOL TABSYMBOL ( D ) ▁ d ▁ + ▁ b ▁ = ▁ " + ( d + b ) + " TABSYMBOL TABSYMBOL m ▁ = ▁ c / k ▁ = ▁ " + m + "NEW_LINE"); if ( k % 2 == 0 && h % 2 == 0 ) { k = k / 2 ; h = h / 2 ; System . out . print ( " Factors ▁ are : ▁ " + ( ( k ) * ( k ) + ( h ) * ( h ) ) + " ▁ " + ( l * l + m * m ) + "NEW_LINE"); } else { l = l / 2 ; m = m / 2 ; System . out . print ( " Factors ▁ are : ▁ " + ( ( l ) * ( l ) + ( m ) * ( m ) ) + " ▁ " + ( k * k + h * h ) + "NEW_LINE"); } } public static void main ( String [ ] args ) { int n = 100000 ; findFactors ( n ) ; } }',
                'gold':'#include <bits/stdc++.h> NEW_LINE using namespace std ; void sumOfSquares ( int n , vector < pair < int , int > > & vp ) { for ( int i = 1 ; i <= sqrt ( n ) ; i ++ ) { int h = n - i * i , h1 = sqrt ( h ) ; if ( h1 * h1 == h ) { int a = max ( h1 , i ) , b = min ( h1 , i ) ; if ( vp . size ( ) == 1 && a != vp [ 0 ] . first ) vp . push_back ( make_pair ( a , b ) ) ; if ( vp . size ( ) == 0 ) vp . push_back ( make_pair ( a , b ) ) ; if ( vp . size ( ) == 2 ) return ; } } } void findFactors ( int n ) { vector < pair < int , int > > vp ; sumOfSquares ( n , vp ) ; if ( vp . size ( ) != 2 ) cout << " Factors ▁ Not ▁ Possible " ; int a , b , c , d ; a = vp [ 0 ] . first ; b = vp [ 0 ] . second ; c = vp [ 1 ] . first ; d = vp [ 1 ] . second ; if ( a < c ) { int t = a ; a = c ; c = t ; t = b ; b = d ; d = t ; } int k , h , l , m ; k = __gcd ( a - c , d - b ) ; h = __gcd ( a + c , d + b ) ; l = ( a - c ) / k ; m = ( d - b ) / k ; cout << " a ▁ = ▁ " << a << " TABSYMBOL TABSYMBOL ( A ) ▁ a ▁ - ▁ c ▁ = ▁ " << ( a - c ) << " TABSYMBOL TABSYMBOL k ▁ = ▁ gcd [ A , ▁ C ] ▁ = ▁ " << k << endl ; cout << " b ▁ = ▁ " << b << " TABSYMBOL TABSYMBOL ( B ) ▁ a ▁ + ▁ c ▁ = ▁ " << ( a + c ) << " TABSYMBOL TABSYMBOL h ▁ = ▁ gcd [ B , ▁ D ] ▁ = ▁ " << h << endl ; cout << " c ▁ = ▁ " << c << " TABSYMBOL TABSYMBOL ( C ) ▁ d ▁ - ▁ b ▁ = ▁ " << ( d - b ) << " TABSYMBOL TABSYMBOL l ▁ = ▁ A / k ▁ = ▁ " << l << endl ; cout << " d ▁ = ▁ " << d << " TABSYMBOL TABSYMBOL ( D ) ▁ d ▁ + ▁ b ▁ = ▁ " << ( d + b ) << " TABSYMBOL TABSYMBOL m ▁ = ▁ c / k ▁ = ▁ " << m << endl ; if ( k % 2 == 0 && h % 2 == 0 ) { k = k / 2 ; h = h / 2 ; cout << " Factors ▁ are : ▁ " << ( ( k ) * ( k ) + ( h ) * ( h ) ) << " ▁ " << ( l * l + m * m ) << endl ; } else { l = l / 2 ; m = m / 2 ; cout << " Factors ▁ are : ▁ " << ( ( l ) * ( l ) + ( m ) * ( m ) ) << " ▁ " << ( k * k + h * h ) << endl ; } } int main ( ) { int n = 100000 ; findFactors ( n ) ; return 0 ; }',
                'description': 'Java到C++的复杂代码翻译'
            },
        ]
        return test_samples
        
    def translate_code(self, source_text, max_length=400, top_k=5, top_p=0.9):
        """
        翻译代码
        
        Args:
            source_text: 源代码文本
            max_length: 最大生成长度
            top_k: Top-k采样
            top_p: Top-p采样
            
        Returns:
            翻译后的代码
        """
        # 对输入文本进行tokenization
        inputs = self.tokenizer(
            source_text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # 使用模型生成代码
        with torch.no_grad():
            generated_ids = respond_to_batch(
                self.model,
                input_ids,
                attention_mask,
                max_target_length=max_length,
                top_k=top_k,
                top_p=top_p
            )
        
        # 解码生成的token
        generated_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return generated_text
        
    def evaluate_with_codebleu(self, predicted_code, gold_code, lang='cpp'):
        """
        使用CodeBLEU评估代码翻译质量
        
        Args:
            predicted_code: 模型预测的代码
            gold_code: 标准答案代码
            lang: 编程语言 (cpp)
            
        Returns:
            dict: 包含各项评估指标的字典
        """
        if not CODEBLEU_AVAILABLE:
            return {
                'bleu': 0.0,
                'bleu_weighted': 0.0,
                'ast_match': 0.0,
                'dfg_match': 0.0,
                'error': 'CodeBLEU not available'
            }
        
        try:
            # 准备keywords目录路径
            keywords_dir = './codebleu/keywords/'
            if not os.path.exists(keywords_dir):
                keywords_dir = './CodeBLEU/keywords/'
            
            # 调用CodeBLEU计算
            result = calc_code_bleu([[gold_code]], [predicted_code], lang, keywords_dir)
            
            return {
                'bleu': result[0],           # BLEU分数
                'bleu_weighted': result[1],  # 加权BLEU分数
                'ast_match': result[2],      # AST匹配分数
                'dfg_match': result[3],      # Dataflow匹配分数
                'error': None
            }
            
        except Exception as e:
            return {
                'bleu': 0.0,
                'bleu_weighted': 0.0,
                'ast_match': 0.0,
                'dfg_match': 0.0,
                'error': str(e)
            }
        
    def run_tests(self):
        """运行所有测试"""
        test_samples = self.prepare_test_samples()
        
        print("=" * 80)
        print("开始CodeT5代码翻译测试 (Java -> C++)")
        if CODEBLEU_AVAILABLE:
            print("✓ CodeBLEU评估已启用")
        else:
            print("✗ CodeBLEU评估未启用")
        print("=" * 80)
        
        # 统计信息
        total_samples = len(test_samples)
        successful_tests = 0
        total_bleu = 0.0
        total_ast_match = 0.0
        total_dfg_match = 0.0
        
        for i, sample in enumerate(test_samples, 1):
            print(f"\n测试 {i}/{total_samples}: {sample['description']}")
            print("-" * 60)
            print(f"输入: {sample['source']}")
            print("-" * 60)
            
            try:
                # 进行代码翻译
                translated = self.translate_code(sample['source'])
                print(f"输出: {translated}")
                print("-" * 60)
                
                # 显示期望输出
                if 'gold' in sample:
                    print(f"期望: {sample['gold']}")
                    print("-" * 60)
                
                # 进行CodeBLEU评估
                if CODEBLEU_AVAILABLE and 'gold' in sample:
                    eval_result = self.evaluate_with_codebleu(translated, sample['gold'])
                    
                    if eval_result['error'] is None:
                        print(f"📊 CodeBLEU评估:")
                        print(f"   BLEU: {eval_result['bleu']:.4f}")
                        print(f"   AST匹配: {eval_result['ast_match']:.4f}")
                        print(f"   Dataflow匹配: {eval_result['dfg_match']:.4f}")
                        
                        # 累计统计
                        total_bleu += eval_result['bleu']
                        total_ast_match += eval_result['ast_match']
                        total_dfg_match += eval_result['dfg_match']
                        successful_tests += 1
                    else:
                        print(f"❌ CodeBLEU评估失败: {eval_result['error']}")
                    print("-" * 60)
                
            except Exception as e:
                print(f"错误: {str(e)}")
                print("-" * 60)
                
        print("\n" + "=" * 80)
        print("测试完成!")
        
        # 显示统计信息
        if successful_tests > 0 and CODEBLEU_AVAILABLE:
            print(f"\n📈 总体评估结果 (基于{successful_tests}个成功测试):")
            print(f"平均BLEU分数: {total_bleu/successful_tests:.4f}")
            print(f"平均AST匹配分数: {total_ast_match/successful_tests:.4f}")
            print(f"平均Dataflow匹配分数: {total_dfg_match/successful_tests:.4f}")
            print(f"成功率: {successful_tests}/{total_samples} ({successful_tests/total_samples*100:.1f}%)")
            
            # 评估等级
            avg_ast = total_ast_match / successful_tests
            avg_dfg = total_dfg_match / successful_tests
            
            print(f"\n🎯 模型性能评估:")
            if avg_ast >= 0.8:
                print(f"AST结构理解: 优秀 ({avg_ast:.4f})")
            elif avg_ast >= 0.6:
                print(f"AST结构理解: 良好 ({avg_ast:.4f})")
            elif avg_ast >= 0.4:
                print(f"AST结构理解: 一般 ({avg_ast:.4f})")
            else:
                print(f"AST结构理解: 需要改进 ({avg_ast:.4f})")
                
            if avg_dfg >= 0.8:
                print(f"数据流理解: 优秀 ({avg_dfg:.4f})")
            elif avg_dfg >= 0.6:
                print(f"数据流理解: 良好 ({avg_dfg:.4f})")
            elif avg_dfg >= 0.4:
                print(f"数据流理解: 一般 ({avg_dfg:.4f})")
            else:
                print(f"数据流理解: 需要改进 ({avg_dfg:.4f})")
        
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='测试CodeT5模型Java到C++代码翻译能力')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型文件路径 (.bin, .pt 或 .pth)，必须提供')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='运行设备 (cuda 或 cpu)')
    parser.add_argument('--max_length', type=int, default=400,
                        help='最大生成长度')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Top-k采样参数')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p采样参数')
    parser.add_argument('--disable_codebleu', action='store_true',
                        help='禁用CodeBLEU评估')
    
    args = parser.parse_args()
    
    # 如果用户选择禁用CodeBLEU，则设置为不可用
    if args.disable_codebleu:
        global CODEBLEU_AVAILABLE
        CODEBLEU_AVAILABLE = False
        print("CodeBLEU评估已被用户禁用")
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件 {args.model_path} 不存在!")
        return
    
    # 创建测试器
    tester = CodeTranslationTester(
        model_path=args.model_path,
        device=args.device
    )
    
    # 运行测试
    tester.run_tests()


if __name__ == "__main__":
    main() 