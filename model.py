from torch import nn
import torch
import os
from transformers import AutoModelForCausalLM, AutoConfig

class QwenCoderHeadWithValueModelLocal(nn.Module):
    """
    Qwen2.5-Coder模型，只从本地加载，不自动下载预训练权重，只支持safetensors权重。
    """
    def __init__(self, config_path=None):
        super().__init__()
        if config_path and os.path.exists(config_path):
            config = AutoConfig.from_pretrained(config_path, local_files_only=True)
        else:
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        self.model = AutoModelForCausalLM.from_config(config)
        self.first_dropout = nn.Dropout(0.1)
        self.summary = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, labels=None, decoder_attention_mask=None):
        """
        Wrap HF Qwen causal LM forward.
        NOTE: 在RL阶段我们通常不需要监督loss；labels可以为None。
        如果labels为None，则不向HF传labels，避免不必要的cross_entropy计算。
        """
        if labels is None:
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 output_hidden_states=True)
        else:
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 labels=labels,
                                 output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        value = self.summary(self.first_dropout(hidden_states)).squeeze(-1)
        outputs = (outputs.logits, outputs, value)
        return outputs

    def load_model_weights(self, model_path, device='cpu'):
        """只支持加载safetensors权重，找不到直接报错。"""
        safetensors_file = os.path.join(model_path, 'model.safetensors')
        if not os.path.exists(safetensors_file):
            raise FileNotFoundError(f"在目录 {model_path} 中找不到 model.safetensors 权重文件")
        
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_file, device=device)
        
        # 检测权重格式并加载
        state_dict_keys = list(state_dict.keys())
        if any(key.startswith('model.') for key in state_dict_keys):
            # 包含'model.'前缀的权重，移除前缀后加载
            qwen_state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith('model.')}
            self.model.load_state_dict(qwen_state_dict, strict=False)
        else:
            # 纯Qwen权重，直接加载
            self.model.load_state_dict(state_dict, strict=False)


def respond_to_batch(model, source_ids, attention_mask, max_target_length=400, top_k=5, top_p=1.0, tokenizer=None):
    print(f"respond_to_batch输入: source_ids={source_ids.shape}, attention_mask={attention_mask.shape}")
    generation_config = {
        'do_sample': True,
        'top_k': top_k,
        'top_p': top_p,
        'max_new_tokens': max_target_length,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id
    }
    generation_config = {k: v for k, v in generation_config.items() if v is not None}
    preds = model.model.generate(
        input_ids=source_ids, 
        attention_mask=attention_mask, 
        **generation_config
    )
    print(f"respond_to_batch输出: {preds.shape}")
    return preds

# 向后兼容别名
CodeT5HeadWithValueModelLocal = QwenCoderHeadWithValueModelLocal
