#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test SFT effect simplification script
"""

import os
import torch
import json
import random
from pathlib import Path
from transformers import AutoTokenizer
from model import QwenCoderHeadWithValueModelLocal, respond_to_batch

def quick_test_sft(model_path: str, data_path: str, n_samples: int = 10):
    """Quick test SFT effect"""
    
    print("🚀 Quick test SFT effect")
    print(f"📁 Model: {model_path}")
    print(f"📁 Data: {data_path}")
    print(f"📊 Sample number: {n_samples}")
    
    # Check path
    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        return
        
    if not os.path.exists(data_path):
        print(f"❌ Data path does not exist: {data_path}")
        return
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Use device: {device}")
    
    try:
        # Load model
        print("📥 Loading model...")
        model = QwenCoderHeadWithValueModelLocal(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device=device
        )
        model.to(device).eval()
        
        # Load tokenizer
        print("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            local_files_only=True, 
            trust_remote_code=True, 
            padding_side='right'
        )
        
        print("✅ Model and tokenizer loaded successfully")
        
        # Test language pairs
        language_pairs = [
            ("java", "python"),
            ("java", "cpp"), 
            ("cpp", "python")
        ]
        
        results = {}
        
        for source_lang, target_lang in language_pairs:
            print(f"\n🧪 Test {source_lang} → {target_lang}")
            
            # Find data file
            data_file = None
            possible_paths = [
                f"{data_path}/qwen/{source_lang.capitalize()}-{target_lang.capitalize()}/train.jsonl",
                f"{data_path}/qwen/{target_lang.capitalize()}-{source_lang.capitalize()}/train.jsonl"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    data_file = path
                    break
            
            if data_file is None:
                print(f"⚠️  Data file not found: {source_lang}-{target_lang}")
                continue
                
            # Read data
            print(f"📖 Reading data: {data_file}")
            examples = []
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            if 'messages' in data and len(data['messages']) >= 2:
                                # Extract user input and assistant response
                                user_msg = data['messages'][0]['content']
                                assistant_msg = data['messages'][1]['content']
                                examples.append({
                                    'source': user_msg,
                                    'target': assistant_msg
                                })
            except Exception as e:
                print(f"⚠️  Reading data failed: {e}")
                continue
                
            if len(examples) == 0:
                print(f"⚠️  No valid samples found")
                continue
                
            print(f"✅ Found {len(examples)} samples")
            
            # Random sampling
            if len(examples) > n_samples:
                sampled_examples = random.sample(examples, n_samples)
            else:
                sampled_examples = examples
                
            # Test generation
            success_count = 0
            total_reward = 0.0
            
            for i, example in enumerate(sampled_examples):
                try:
                    # Generate code
                    inputs = tokenizer(
                        example['source'], 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=400,
                        padding=True
                    )
                    
                    source_ids = inputs["input_ids"].to(device)
                    source_mask = inputs["attention_mask"].to(device)
                    
                    with torch.no_grad():
                        # Use respond_to_batch function to generate response
                        full_response = respond_to_batch(
                            model, 
                            source_ids, 
                            source_mask,
                            max_target_length=400,
                            top_k=5, 
                            top_p=0.9, 
                            tokenizer=tokenizer
                        )
                        
                        # Extract generated response
                        response_ids = full_response[:, source_ids.size(1):]
                        
                    # Decode
                    generated_text = tokenizer.decode(
                        response_ids[0], 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=False
                    )
                    
                    # Simple evaluation: check if response is generated
                    if len(generated_text.strip()) > 10:  # Simple length check
                        success_count += 1
                        total_reward += 1.0  # Simple reward
                        
                    print(f"  Sample {i+1}: {'✅' if len(generated_text.strip()) > 10 else '❌'} "
                          f"Length={len(generated_text.strip())}")
                          
                except Exception as e:
                    print(f"  Sample {i+1}: ❌ Error - {e}")
                    
            # Calculate statistics
            success_rate = success_count / len(sampled_examples) if sampled_examples else 0.0
            avg_reward = total_reward / len(sampled_examples) if sampled_examples else 0.0
            
            results[f"{source_lang}-{target_lang}"] = {
                'total_samples': len(sampled_examples),
                'success_count': success_count,
                'success_rate': success_rate,
                'avg_reward': avg_reward
            }
            
            print(f"📊 {source_lang}→{target_lang} Results:")
            print(f"    Success rate: {success_rate:.3f} ({success_count}/{len(sampled_examples)})")
            print(f"    Average reward: {avg_reward:.3f}")
            
        # Overall statistics
        print(f"\n{'='*60}")
        print("📊 Overall test results")
        print(f"{'='*60}")
        
        if results:
            total_samples = sum(r['total_samples'] for r in results.values())
            total_success = sum(r['success_count'] for r in results.values())
            overall_success_rate = total_success / total_samples if total_samples > 0 else 0.0
            
            print(f"✅ Number of language pairs tested: {len(results)}")
            print(f"📊 Total number of samples: {total_samples}")
            print(f"🎯 Overall success rate: {overall_success_rate:.3f}")
            
            # Sort by success rate
            sorted_results = sorted(results.items(), key=lambda x: x[1]['success_rate'], reverse=True)
            print(f"\n📈 Performance of each language pair:")
            for i, (lang_pair, result) in enumerate(sorted_results):
                print(f"   {i+1}. {lang_pair}: {result['success_rate']:.3f} "
                      f"({result['success_count']}/{result['total_samples']})")
        else:
            print("❌ No language pairs successfully completed the test")
            
        # Save results
        try:
            results_dir = Path("quick_sft_test_results")
            results_dir.mkdir(exist_ok=True)
            
            results_file = results_dir / "quick_test_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
            print(f"\n💾 Results saved to: {results_file}")
            
        except Exception as e:
            print(f"⚠️  Saving results failed: {e}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick test SFT effect")
    parser.add_argument("--model_path", required=True, type=str, help="Model path")
    parser.add_argument("--data_path", required=True, type=str, help="Data path")
    parser.add_argument("--n_samples", default=10, type=int, help="Number of samples per language pair tested")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(42)
    
    quick_test_sft(args.model_path, args.data_path, args.n_samples)
