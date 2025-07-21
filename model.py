from torch import nn
import torch
import os
from transformers import AutoModelForCausalLM, AutoConfig
from mem import log_mem, mem_guard


class QwenCoderHeadWithValueModelLocal(nn.Module):
    """
    Qwen2.5-Coder模型，只从本地加载，不自动下载预训练权重，只支持safetensors权重。
    """
    def __init__(self, model_path=None, torch_dtype=None, device='cpu'):
        """
        正确加载本地微调好的 Qwen 模型，并在其上加一个 value head。
        过去版本用 from_config() + 手动加载权重，容易因架构差异/权重键名不匹配导致模型退化。
        """
        super().__init__()
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")

        # 直接从本地 checkpoint 加载完整权重和自定义模块
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=None,  # 让调用方再 .to(device)
        )
        self.first_dropout = nn.Dropout(0.1)
        self.summary = nn.Linear(self.model.config.hidden_size, 1)
        self.hidden_size = self.model.config.hidden_size

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
        if hidden_states.dtype != self.summary.weight.dtype:
            hidden_states = hidden_states.to(self.summary.weight.dtype)
        value = self.summary(self.first_dropout(hidden_states)).squeeze(-1)
        outputs = (outputs.logits, outputs, value)
        return outputs

    def load_model_weights(self, *args, **kwargs):
        """
        兼容旧接口：现在无需单独加载；模型已在 __init__ 中加载完毕。
        留空以避免误覆盖。仅打印提示。
        """
        print("[QwenCoderHeadWithValueModelLocal] load_model_weights() 已废弃，忽略调用。")

def respond_to_batch(model, source_ids, attention_mask, max_target_length=400, top_k=5, top_p=1.0, tokenizer=None):
    """
    从批量 prompt 生成响应。支持 wrapper 或 HF 原始模型。
    """
    print(f"respond_to_batch输入: source_ids={source_ids.shape}, attention_mask={attention_mask.shape}")
    hf_model = model.model
    generation_config = {
        'do_sample': True,
        'top_k': top_k,
        'top_p': top_p,
        "max_new_tokens": max_target_length,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        with mem_guard("generate"):
            preds = hf_model.generate(input_ids=source_ids,
                                        attention_mask=attention_mask,
                                        **generation_config)
    torch.cuda.empty_cache()
    log_mem("after empty_cache")
    return preds

# 向后兼容别名
CodeT5HeadWithValueModelLocal = QwenCoderHeadWithValueModelLocal
