from torch import nn
import torch.nn.functional as F
import torch
import sys
import os
from pathlib import Path

sys.path.append('../')
sys.path.append('../../')
from transformers import T5ForConditionalGeneration, T5Config


class CodeT5HeadWithValueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
        self.first_dropout = nn.Dropout(0.1)
        self.summary = nn.Linear(self.model.model_dim, 1)
        
    def load_base_model(self, load_model_path):
        self.model.load_state_dict(torch.load(load_model_path))

    def forward(self, input_ids, attention_mask=None, labels=None, decoder_attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, 
                                         decoder_attention_mask=decoder_attention_mask, output_hidden_states=True)
        hidden_states = outputs.decoder_hidden_states[-1]
        value = self.summary(self.first_dropout(hidden_states)).squeeze(-1)
        outputs = (outputs.logits, outputs, value)
        return outputs


class CodeT5HeadWithValueModelLocal(nn.Module):
    """
    CodeT5模型，只从本地加载，不自动下载预训练权重
    """
    def __init__(self, config_path=None):
        super().__init__()
        # 只从本地加载配置文件
        if config_path and os.path.exists(config_path):
            print(f"正在加载本地配置文件: {config_path}")
            config = T5Config.from_pretrained(config_path, local_files_only=True)
        else:
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        self.model = T5ForConditionalGeneration(config)
        self.first_dropout = nn.Dropout(0.1)
        self.summary = nn.Linear(self.model.model_dim, 1)
        
    def forward(self, input_ids, attention_mask=None, labels=None, decoder_attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, 
                                         decoder_attention_mask=decoder_attention_mask, output_hidden_states=True)
        hidden_states = outputs.decoder_hidden_states[-1]
        value = self.summary(self.first_dropout(hidden_states)).squeeze(-1)
        outputs = (outputs.logits, outputs, value)
        return outputs
        
    def load_model_weights(self, model_path, device='cpu'):
        """加载模型权重"""
        try:
            # 如果model_path是目录，尝试加载其中的pytorch_model.bin文件
            if os.path.isdir(model_path):
                weight_file = os.path.join(model_path, 'pytorch_model.bin')
                if os.path.exists(weight_file):
                    print(f"从目录加载模型权重: {weight_file}")
                    state_dict = torch.load(weight_file, map_location=device)
                else:
                    raise FileNotFoundError(f"在目录 {model_path} 中找不到 pytorch_model.bin 文件")
            else:
                # 如果model_path是文件，直接加载
                print(f"直接加载模型权重文件: {model_path}")
                state_dict = torch.load(model_path, map_location=device)
            
            print(f"成功加载模型文件，包含 {len(state_dict)} 个参数")
            
            # 检查state_dict的键名来确定加载方式
            state_dict_keys = list(state_dict.keys())
            print(f"模型键名示例: {state_dict_keys[:3]}...")
            
            if any(key.startswith('model.model.') for key in state_dict_keys):
                # 如果包含完整的CodeT5HeadWithValueModel结构
                print("检测到完整模型结构，直接加载...")
                self.load_state_dict(state_dict, strict=False)
            elif any(key.startswith('model.') for key in state_dict_keys):
                # 如果只包含T5模型部分
                print("检测到T5模型权重，加载到model.model中...")
                # 移除'model.'前缀，加载到self.model中
                t5_state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith('model.')}
                self.model.load_state_dict(t5_state_dict, strict=False)
            else:
                # 如果是纯T5权重（没有'model.'前缀）
                print("检测到纯T5权重，直接加载到model中...")
                self.model.load_state_dict(state_dict, strict=False)
                
        except Exception as e:
            raise RuntimeError(f"加载模型权重失败: {e}")

    
def respond_to_batch(model, source_ids, attention_mask, max_target_length=400, top_k=5, top_p=1.0):
    
    preds = model.model.generate(source_ids, attention_mask=attention_mask, do_sample=True, top_k=top_k, top_p=top_p,
                                 max_length=max_target_length)
    # preds = model.module.model.generate(source_ids, attention_mask=attention_mask, do_sample=True, top_k=top_k, top_p=top_p,
    #                              max_length=max_target_length)
    return preds
