#!/usr/bin/env python3
"""
模型评估测试脚本
用于比较微调前后模型在代码翻译任务上的性能

评估维度：
1. 编译通过率评分
2. AST匹配评分  
3. DFG匹配评分
4. CodeBLEU评分

数据集：data/qwen/Java-C++/val.jsonl
"""

import os
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer

# 导入项目模块
from model import QwenCoderHeadWithValueModelLocal, respond_to_batch
from utils import (read_qwen_examples, convert_qwen_examples_to_features, 
                   extract_code_from_qwen_response, Example)
from reward import get_reward
from codebleu.calc_code_bleu import calc_code_bleu
from compiler.terminal_compiler import TerminalCompiler


@dataclass
class EvaluationConfig:
    """评估配置"""
    # 模型路径
    model_before: str  # 微调前模型路径
    model_after: str   # 微调后模型路径
    
    # 数据配置
    data_path: str = "data/qwen/Java-Python/val.jsonl"
    source_lang: str = "java"
    target_lang: str = "python"
    
    # 生成配置
    max_source_length: int = 700
    max_target_length: int = 700
    batch_size: int = 8
    top_k: int = 1
    top_p: float = 1
    temperature: float = 0
    do_sample: bool = False
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 输出配置
    output_dir: str = "evaluation_results"
    save_predictions: bool = True
    
    # McNemar测试配置
    enable_mcnemar: bool = True  # 是否启用McNemar测试
    mcnemar_alpha: float = 0.05  # McNemar测试显著性水平


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.setup_output_dir()
        self.load_data()
        self.setup_compiler()
        
    def setup_output_dir(self):
        """设置输出目录"""
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 输出目录: {self.output_dir}")
        
    def load_data(self):
        """加载评估数据"""
        print(f"📊 加载数据: {self.config.data_path}")
        
        # 自动推断语言对
        self.source_lang, self.target_lang = self._infer_languages_from_path(self.config.data_path)
        print(f"🌐 推断翻译任务: {self.source_lang} → {self.target_lang}")
        
        # 创建临时配置对象用于数据加载
        class TempArgs:
            def __init__(self, source_lang, target_lang):
                self.source_lang = source_lang
                self.target_lang = target_lang
                
        temp_args = TempArgs(self.source_lang, self.target_lang)
        self.examples = read_qwen_examples(self.config.data_path, temp_args)
        print(f"✅ 加载了 {len(self.examples)} 个样本")
        
    def _infer_languages_from_path(self, data_path: str) -> tuple:
        """从数据路径推断源语言和目标语言"""
        if "Java-Python" in data_path:
            return "java", "python"
        elif "Java-C++" in data_path:
            return "java", "cpp"
        elif "C++-Python" in data_path:
            return "cpp", "python"
        else:
            # 默认Java->Python
            return "java", "python"
        
    def setup_compiler(self):
        """设置编译器"""
        # 语言映射
        lang_mapping = {
            "cpp": "C++",
            "java": "Java", 
            "python": "Python",
            "c": "C",
            "php": "PHP",
            "c_sharp": "C#"
        }
        compiler_lang = lang_mapping.get(self.target_lang, "Python")
        self.compiler = TerminalCompiler(compiler_lang)
        print(f"🔧 设置编译器: {self.target_lang} -> {compiler_lang}")
        
    def load_model_and_tokenizer(self, model_path: str):
        """加载模型和分词器"""
        print(f"🤖 加载模型: {model_path}")
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            padding_side='right'
        )
        
        # 加载模型
        model = QwenCoderHeadWithValueModelLocal(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device=self.config.device
        )
        model.to(self.config.device)
        model.eval()
        
        return model, tokenizer
        
    def generate_predictions(self, model, tokenizer, examples: List[Example]) -> List[str]:
        """生成模型预测"""
        predictions = []
        
        # 转换为特征
        features = convert_qwen_examples_to_features(
            examples, tokenizer, self.config, stage='test'
        )
        
        # 批量生成
        print(f"🔄 生成预测结果...")
        for i in tqdm(range(0, len(features), self.config.batch_size)):
            batch_features = features[i:i + self.config.batch_size]
            
            # 准备批次数据
            source_ids = torch.tensor([f.source_ids for f in batch_features], dtype=torch.long).to(self.config.device)
            source_mask = torch.tensor([f.source_mask for f in batch_features], dtype=torch.long).to(self.config.device)
            
            # 生成代码
            with torch.no_grad():
                full_outputs = respond_to_batch(
                    model, source_ids, source_mask,
                    max_target_length=self.config.max_target_length,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    tokenizer=tokenizer,
                    temperature=self.config.temperature,
                    do_sample=self.config.do_sample
                )
                
                # 提取生成的部分
                generated_ids = full_outputs[:, source_ids.size(1):]
                
                # 解码
                for gen_ids in generated_ids:
                    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    extracted_code = extract_code_from_qwen_response(decoded, self.target_lang)
                    predictions.append(extracted_code)
        
        return predictions
        
    def evaluate_compilation(self, predictions: List[str]) -> Dict:
        """评估编译通过率"""
        print("🔨 评估编译通过率...")
        
        compilation_results = []
        success_count = 0
        
        for i, code in enumerate(tqdm(predictions)):
            try:
                # 编译代码
                compile_result = self.compiler.compile_code_string(code)
                success = compile_result[2] if len(compile_result) > 2 else False
                
                compilation_results.append({
                    'index': i,
                    'success': success,
                    'code': code,
                    'error': compile_result[1] if not success and len(compile_result) > 1 else None
                })
                
                if success:
                    success_count += 1
                    
            except Exception as e:
                compilation_results.append({
                    'index': i,
                    'success': False,
                    'code': code,
                    'error': str(e)
                })
        
        compile_rate = success_count / len(predictions) if predictions else 0.0
        
        return {
            'compile_rate': compile_rate,
            'success_count': success_count,
            'total_count': len(predictions),
            'details': compilation_results
        }
        
    def evaluate_codebleu(self, predictions: List[str], targets: List[str]) -> Dict:
        """评估CodeBLEU指标"""
        print("📊 评估CodeBLEU指标...")
        
        # 准备keywords目录
        keywords_dir = './codebleu/keywords/'
        if not os.path.exists(keywords_dir):
            keywords_dir = './CodeBLEU/keywords/'
            
        try:
            # 调用CodeBLEU计算
            result = calc_code_bleu([targets], predictions, self.target_lang, keywords_dir)
            
            return {
                'bleu': result[0],                    # BLEU分数
                'weighted_bleu': result[1],           # 加权BLEU分数  
                'ast_match': result[2],               # AST匹配分数
                'dfg_match': result[3],               # DFG匹配分数
                'codebleu': result[4],                # 综合CodeBLEU分数
                'error': None
            }
            
        except Exception as e:
            print(f"⚠️ CodeBLEU计算失败: {e}")
            return {
                'bleu': 0.0,
                'weighted_bleu': 0.0,
                'ast_match': 0.0,
                'dfg_match': 0.0,
                'codebleu': 0.0,
                'error': str(e)
            }
    
    def evaluate_ast_dfg_individual(self, predictions: List[str], targets: List[str]) -> Dict:
        """单独评估AST和DFG匹配度（逐个样本）"""
        print("🌳 评估AST和DFG匹配度...")
        
        from codebleu.calc_code_bleu import calc_code_bleu
        
        ast_scores = []
        dfg_scores = []
        keywords_dir = './codebleu/keywords/'
        
        for pred, target in tqdm(zip(predictions, targets), total=len(predictions)):
            try:
                # 单个样本的CodeBLEU计算
                result = calc_code_bleu([[target]], [pred], self.target_lang, keywords_dir)
                ast_scores.append(result[2])
                dfg_scores.append(result[3])
            except Exception as e:
                ast_scores.append(0.0)
                dfg_scores.append(0.0)
        
        return {
            'ast_mean': np.mean(ast_scores),
            'ast_std': np.std(ast_scores),
            'ast_scores': ast_scores,
            'dfg_mean': np.mean(dfg_scores),
            'dfg_std': np.std(dfg_scores),
            'dfg_scores': dfg_scores
        }
    
    def calculate_mcnemar_compilation(self, before_compilation: List[bool], after_compilation: List[bool]) -> Dict:
        """
        使用McNemar测试评估两个模型的编译通过率差异
        
        混淆矩阵:
        - n00: 两个模型都编译失败
        - n01: 微调前失败，微调后成功 (关键指标)
        - n10: 微调前成功，微调后失败 (关键指标)  
        - n11: 两个模型都编译成功
        
        重点关注n01 vs n10的差异，n01 > n10表示微调有效
        """
        print("📊 计算McNemar编译通过率测试...")
        
        if len(before_compilation) != len(after_compilation):
            raise ValueError("编译结果列表长度不匹配")
        
        # 构建混淆矩阵
        n00 = n01 = n10 = n11 = 0
        
        for before_success, after_success in zip(before_compilation, after_compilation):
            if not before_success and not after_success:
                n00 += 1
            elif not before_success and after_success:
                n01 += 1  # 微调前失败，微调后成功
            elif before_success and not after_success:
                n10 += 1  # 微调前成功，微调后失败
            else:  # before_success and after_success
                n11 += 1
        
        # 计算McNemar统计量 (连续性校正)
        if n01 + n10 > 0:
            mcnemar_statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
            # 自由度为1的卡方分布
            from scipy.stats import chi2
            p_value = 1 - chi2.cdf(mcnemar_statistic, df=1)
        else:
            mcnemar_statistic = 0.0
            p_value = 1.0
        
        # 计算改进效果
        improvement_rate = (n01 - n10) / (n01 + n10) if (n01 + n10) > 0 else 0.0
        
        # 判断统计显著性
        is_significant = p_value < self.config.mcnemar_alpha
        
        result = {
            'confusion_matrix': {
                'n00': n00,  # 都失败
                'n01': n01,  # 前失败后成功 (关键指标)
                'n10': n10,  # 前成功后失败 (关键指标)
                'n11': n11   # 都成功
            },
            'mcnemar_statistic': mcnemar_statistic,
            'p_value': p_value,
            'is_significant': is_significant,
            'improvement_rate': improvement_rate,
            'interpretation': self._interpret_mcnemar_compilation(n01, n10, p_value, is_significant)
        }
        
        return result
    
    def _interpret_mcnemar_compilation(self, n01: int, n10: int, p_value: float, is_significant: bool) -> str:
        """解释McNemar编译通过率测试结果"""
        if n01 == 0 and n10 == 0:
            return "两个模型编译表现完全一致，无变化"
        
        if n01 > n10:
            direction = "正向"
            improvement = f"微调后编译成功率提升了 {n01 - n10} 个样本"
        elif n01 < n10:
            direction = "负向"
            improvement = f"微调后编译成功率下降了 {n10 - n01} 个样本"
        else:
            direction = "无变化"
            improvement = "微调前后编译成功率相同"
        
        significance = "统计显著" if is_significant else "统计不显著"
        
        if is_significant:
            if n01 > n10:
                conclusion = "微调效果显著，模型编译能力有实质性提升"
            elif n01 < n10:
                conclusion = "微调效果显著，但模型编译能力出现实质性下降"
            else:
                conclusion = "微调效果显著，但方向需要进一步分析"
        else:
            conclusion = "微调效果不显著，需要更多数据或调整训练策略"
        
        return f"{direction}变化，{improvement}。{significance}，{conclusion}"
    
    def print_mcnemar_compilation_results(self, mcnemar_results: Dict):
        """打印McNemar编译通过率测试结果"""
        print(f"\n{'='*60}")
        print(f"📊 McNemar编译通过率测试结果")
        print(f"{'='*60}")
        
        cm = mcnemar_results['confusion_matrix']
        print(f"🔍 混淆矩阵:")
        print(f"   n00 (都失败): {cm['n00']}")
        print(f"   n01 (前失败后成功): {cm['n01']} ← 关键指标")
        print(f"   n10 (前成功后失败): {cm['n10']} ← 关键指标")
        print(f"   n11 (都成功): {cm['n11']}")
        print(f"")
        
        print(f"📈 改进分析:")
        print(f"   n01 - n10 = {cm['n01']} - {cm['n10']} = {cm['n01'] - cm['n10']}")
        if cm['n01'] + cm['n10'] > 0:
            print(f"   改进率: {mcnemar_results['improvement_rate']:.4f}")
        print(f"")
        
        print(f"📊 统计测试:")
        print(f"   McNemar统计量: {mcnemar_results['mcnemar_statistic']:.4f}")
        print(f"   p值: {mcnemar_results['p_value']:.6f}")
        print(f"   显著性水平: α = {self.config.mcnemar_alpha}")
        print(f"   统计显著: {'✅ 是' if mcnemar_results['is_significant'] else '❌ 否'}")
        print(f"")
        
        print(f"💡 结果解释:")
        print(f"   {mcnemar_results['interpretation']}")
        print(f"{'='*60}")
        
    def evaluate_model(self, model_path: str, model_name: str) -> Dict:
        """评估单个模型"""
        print(f"\n{'='*60}")
        print(f"🎯 评估模型: {model_name}")
        print(f"📁 模型路径: {model_path}")
        print(f"{'='*60}")
        
        # 加载模型
        model, tokenizer = self.load_model_and_tokenizer(model_path)
        
        # 生成预测
        predictions = self.generate_predictions(model, tokenizer, self.examples)
        
        # 提取目标代码
        targets = [example.target for example in self.examples]
        
        # 评估编译通过率
        compilation_eval = self.evaluate_compilation(predictions)
        
        # 评估CodeBLEU
        codebleu_eval = self.evaluate_codebleu(predictions, targets)
        
        # 评估单独的AST和DFG
        ast_dfg_eval = self.evaluate_ast_dfg_individual(predictions, targets)
        
        # 保存预测结果
        if self.config.save_predictions:
            pred_file = self.output_dir / f"{model_name}_predictions.json"
            with open(pred_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'predictions': predictions,
                    'targets': targets,
                    'compilation_details': compilation_eval['details']
                }, f, indent=2, ensure_ascii=False)
            print(f"💾 预测结果保存到: {pred_file}")
        
        # 清理显存
        del model, tokenizer
        torch.cuda.empty_cache()
        
        return {
            'model_name': model_name,
            'model_path': model_path,
            'compilation': compilation_eval,
            'codebleu': codebleu_eval,
            'ast_dfg': ast_dfg_eval,
            'sample_count': len(predictions)
        }
        
    def run_evaluation(self) -> Dict:
        """运行完整评估"""
        print(f"\n🚀 开始模型对比评估")
        print(f"📊 数据集: {self.config.data_path}")
        print(f"🔄 样本数量: {len(self.examples)}")
        print(f"🌐 翻译方向: {self.source_lang} → {self.target_lang}")
        
        results = {}
        
        # 评估微调前模型
        results['before'] = self.evaluate_model(self.config.model_before, "微调前模型")
        
        # 评估微调后模型  
        results['after'] = self.evaluate_model(self.config.model_after, "微调后模型")
        
        # 执行McNemar编译通过率测试
        if self.config.enable_mcnemar:
            print("\n📊 执行McNemar编译通过率测试...")
            before_compilation = [detail['success'] for detail in results['before']['compilation']['details']]
            after_compilation = [detail['success'] for detail in results['after']['compilation']['details']]
            
            mcnemar_results = self.calculate_mcnemar_compilation(before_compilation, after_compilation)
            results['mcnemar_compilation'] = mcnemar_results
            
            # 打印McNemar测试结果
            self.print_mcnemar_compilation_results(mcnemar_results)
        
        # 生成对比报告
        self.generate_comparison_report(results)
        
        return results
        
    def generate_comparison_report(self, results: Dict):
        """生成对比评估报告"""
        print(f"\n📋 生成评估报告...")
        
        before = results['before']
        after = results['after']
        
        # 计算改进幅度
        def calc_improvement(before_val, after_val):
            if before_val == 0:
                return float('inf') if after_val > 0 else 0
            return ((after_val - before_val) / before_val) * 100
        
        # 准备报告数据
        report_data = {
            'evaluation_config': {
                'data_path': self.config.data_path,
                'source_lang': self.source_lang,
                'target_lang': self.target_lang,
                'sample_count': len(self.examples),
                'model_before': self.config.model_before,
                'model_after': self.config.model_after
            },
            'results': results,
            'comparison': {
                'compilation': {
                    'before': before['compilation']['compile_rate'],
                    'after': after['compilation']['compile_rate'],
                    'improvement': calc_improvement(
                        before['compilation']['compile_rate'],
                        after['compilation']['compile_rate']
                    )
                },
                'codebleu': {
                    'before': before['codebleu']['codebleu'],
                    'after': after['codebleu']['codebleu'],
                    'improvement': calc_improvement(
                        before['codebleu']['codebleu'],
                        after['codebleu']['codebleu']
                    )
                },
                'ast_match': {
                    'before': before['ast_dfg']['ast_mean'],
                    'after': after['ast_dfg']['ast_mean'],
                    'improvement': calc_improvement(
                        before['ast_dfg']['ast_mean'],
                        after['ast_dfg']['ast_mean']
                    )
                },
                'dfg_match': {
                    'before': before['ast_dfg']['dfg_mean'],
                    'after': after['ast_dfg']['dfg_mean'],
                    'improvement': calc_improvement(
                        before['ast_dfg']['dfg_mean'],
                        after['ast_dfg']['dfg_mean']
                    )
                }
            }
        }
        
        # 添加McNemar测试结果到报告
        if 'mcnemar_compilation' in results:
            report_data['mcnemar_test'] = results['mcnemar_compilation']
        
        # 保存详细报告
        report_file = self.output_dir / "evaluation_report.json"
        
        # 转换numpy类型为Python原生类型，确保JSON序列化
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # 转换报告数据
        serializable_report = convert_numpy_types(report_data)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, indent=2, ensure_ascii=False)
        
        # 生成简洁的文本报告
        self.print_summary_report(report_data)
        
        # 保存简洁报告
        summary_file = self.output_dir / "evaluation_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(self.format_summary_report(report_data))
        
        print(f"📊 详细报告: {report_file}")
        print(f"📋 简洁报告: {summary_file}")
        
    def print_summary_report(self, report_data: Dict):
        """打印简洁评估报告"""
        comp = report_data['comparison']
        
        print(f"\n{'='*80}")
        print(f"📊 模型对比评估报告")
        print(f"{'='*80}")
        print(f"📁 数据集: {self.config.data_path}")
        print(f"🔄 样本数: {len(self.examples)}")
        print(f"🌐 翻译: {self.source_lang} → {self.target_lang}")
        print(f"")
        
        print(f"🎯 评估指标对比:")
        print(f"{'指标':<15} {'微调前':<12} {'微调后':<12} {'改进幅度':<15}")
        print(f"{'-'*60}")
        
        metrics = [
            ('编译通过率', comp['compilation'], '%'),
            ('CodeBLEU', comp['codebleu'], ''),  
            ('AST匹配度', comp['ast_match'], ''),
            ('DFG匹配度', comp['dfg_match'], '')
        ]
        
        for name, data, unit in metrics:
            before = data['before']
            after = data['after']
            improvement = data['improvement']
            
            if unit == '%':
                before_str = f"{before*100:.2f}%"
                after_str = f"{after*100:.2f}%"
            else:
                before_str = f"{before:.4f}"
                after_str = f"{after:.4f}"
            
            if improvement == float('inf'):
                improve_str = "∞"
            else:
                improve_str = f"{improvement:+.2f}%"
                
            print(f"{name:<15} {before_str:<12} {after_str:<12} {improve_str:<15}")
        
        print(f"\n🏆 总体评价:")
        total_improvements = [
            comp['compilation']['improvement'],
            comp['codebleu']['improvement'], 
            comp['ast_match']['improvement'],
            comp['dfg_match']['improvement']
        ]
        
        positive_improvements = sum(1 for x in total_improvements if x > 0)
        avg_improvement = np.mean([x for x in total_improvements if x != float('inf')])
        
        print(f"   • 改进指标数: {positive_improvements}/4")
        print(f"   • 平均改进幅度: {avg_improvement:+.2f}%")
        
        # 显示McNemar测试结果
        if 'mcnemar_test' in report_data:
            mcnemar = report_data['mcnemar_test']
            print(f"")
            print(f"📊 McNemar编译通过率测试:")
            cm = mcnemar['confusion_matrix']
            print(f"   • n01 (前失败后成功): {cm['n01']}")
            print(f"   • n10 (前成功后失败): {cm['n10']}")
            print(f"   • 净改进: {cm['n01'] - cm['n10']}")
            print(f"   • 统计显著: {'✅ 是' if mcnemar['is_significant'] else '❌ 否'}")
            print(f"   • p值: {mcnemar['p_value']:.6f}")
        
        if positive_improvements >= 3:
            print(f"   • 🎉 微调效果显著，模型性能全面提升！")
        elif positive_improvements >= 2:
            print(f"   • ✅ 微调效果良好，多数指标有所改进")
        elif positive_improvements >= 1:
            print(f"   • ⚠️  微调效果一般，部分指标有改进")
        else:
            print(f"   • ❌ 微调效果不佳，建议检查训练配置")
            
        print(f"{'='*80}")
        
    def format_summary_report(self, report_data: Dict) -> str:
        """格式化简洁报告为文本"""
        lines = []
        comp = report_data['comparison']
        
        lines.append("=" * 80)
        lines.append("模型对比评估报告")
        lines.append("=" * 80)
        lines.append(f"数据集: {self.config.data_path}")
        lines.append(f"样本数: {len(self.examples)}")
        lines.append(f"翻译: {self.source_lang} → {self.target_lang}")
        lines.append("")
        
        lines.append("评估指标对比:")
        lines.append(f"{'指标':<15} {'微调前':<12} {'微调后':<12} {'改进幅度':<15}")
        lines.append("-" * 60)
        
        metrics = [
            ('编译通过率', comp['compilation'], '%'),
            ('CodeBLEU', comp['codebleu'], ''),
            ('AST匹配度', comp['ast_match'], ''),
            ('DFG匹配度', comp['dfg_match'], '')
        ]
        
        for name, data, unit in metrics:
            before = data['before']
            after = data['after']
            improvement = data['improvement']
            
            if unit == '%':
                before_str = f"{before*100:.2f}%"
                after_str = f"{after*100:.2f}%"
            else:
                before_str = f"{before:.4f}"
                after_str = f"{after:.4f}"
            
            if improvement == float('inf'):
                improve_str = "∞"
            else:
                improve_str = f"{improvement:+.2f}%"
                
            lines.append(f"{name:<15} {before_str:<12} {after_str:<12} {improve_str:<15}")
        
        return "\n".join(lines)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="模型对比评估脚本")
    
    # 必需参数
    parser.add_argument("--model_before", required=True, type=str,
                       help="微调前模型路径")
    parser.add_argument("--model_after", required=True, type=str,  
                       help="微调后模型路径")
    
    # 可选参数
    parser.add_argument("--data_path", default="data/qwen/Java-Python/val.jsonl", type=str,
                       help="验证数据集路径 (支持: Java-Python, Java-C++, C++-Python)")
    parser.add_argument("--source_lang", default="java", type=str,
                       help="源代码语言 (会自动从数据路径推断)")
    parser.add_argument("--target_lang", default="python", type=str,
                       help="目标代码语言 (会自动从数据路径推断)")
    parser.add_argument("--batch_size", default=8, type=int,
                       help="批次大小")
    parser.add_argument("--max_target_length", default=400, type=int,
                       help="目标代码最大长度")
    parser.add_argument("--output_dir", default="evaluation_results", type=str,
                       help="输出目录")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str,
                       help="计算设备")
    parser.add_argument("--no_save_predictions", action="store_true",
                       help="不保存预测结果")
    parser.add_argument("--disable_mcnemar", action="store_true",
                       help="禁用McNemar测试")
    parser.add_argument("--mcnemar_alpha", default=0.05, type=float,
                       help="McNemar测试显著性水平 (默认: 0.05)")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建评估配置
    config = EvaluationConfig(
        model_before=args.model_before,
        model_after=args.model_after,
        data_path=args.data_path,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        batch_size=args.batch_size,
        max_target_length=args.max_target_length,
        output_dir=args.output_dir,
        device=args.device,
        save_predictions=not args.no_save_predictions,
        enable_mcnemar=not args.disable_mcnemar,
        mcnemar_alpha=args.mcnemar_alpha
    )
    
    # 创建评估器
    evaluator = ModelEvaluator(config)
    
    # 运行评估
    try:
        results = evaluator.run_evaluation()
        print(f"\n✅ 评估完成！结果保存在: {config.output_dir}")
        
    except Exception as e:
        print(f"\n❌ 评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())