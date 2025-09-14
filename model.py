from torch import nn
import torch
import os
from transformers import AutoModelForCausalLM, AutoConfig


class QwenCoderHeadWithValueModelLocal(nn.Module):
    """
    Qwen2.5-Coder model, only loads from local, does not automatically download pre-trained weights, only supports safetensors weights.
    """
    def __init__(self, model_path=None, torch_dtype=None, device='cpu'):
        super().__init__()
        
        if model_path:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True
            )
        else:
            self.model = None
            
        self.hidden_size = self.model.config.hidden_size if self.model else 4096
        self.first_dropout = nn.Dropout(0.1)
        self.summary = nn.Linear(self.hidden_size, 1)
        
        # 🔧 Add config attribute to compatible with PPO trainer
        self.config = self.model.config if self.model else None

    def forward(self, input_ids, attention_mask=None, labels=None, decoder_attention_mask=None):
        """
        Wrap HF Qwen causal LM forward.
        """
        if labels is None:
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 output_hidden_states=True,
                                 use_cache=False,
                                 return_dict=True)
        else:
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 labels=labels,
                                 output_hidden_states=True,
                                 use_cache=False,
                                 return_dict=True)
        hidden_states = outputs.hidden_states[-1]
        if hidden_states.dtype != self.summary.weight.dtype:
            hidden_states = hidden_states.to(self.summary.weight.dtype)
        value = self.summary(self.first_dropout(hidden_states)).squeeze(-1)
        outputs = (outputs.logits, None, value)
        return outputs

    def load_model_weights(self, *args, **kwargs):
        """
        This function is deprecated, ignoring call.
        """
        print("[QwenCoderHeadWithValueModelLocal] load_model_weights() is deprecated, ignoring call.")
    
    def save_pretrained(self, save_directory, **kwargs):
        """
        Save the model and tokenizer to a directory.
        This method saves the underlying Hugging Face model and adds our custom components.
        """
        import os
        from pathlib import Path
        
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save the underlying Hugging Face model
        self.model.save_pretrained(save_directory, **kwargs)
        
        # Save our custom components (summary layer and dropout)
        custom_state_dict = {
            'summary.weight': self.summary.weight.data,
            'summary.bias': self.summary.bias.data,
            'first_dropout.p': 0.1,  # Save dropout rate
        }
        
        # Save custom components to a separate file
        custom_path = save_directory / "custom_components.bin"
        torch.save(custom_state_dict, custom_path)
        
        # Create a model info file
        model_info = {
            "model_type": "QwenCoderHeadWithValueModelLocal",
            "hidden_size": self.hidden_size,
            "custom_components": ["summary", "first_dropout"],
            "save_time": str(torch.cuda.Event() if torch.cuda.is_available() else "cpu")
        }
        
        import json
        info_path = save_directory / "model_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Model saved to: {save_directory}")
        print(f"   - Base model: {save_directory}")
        print(f"   - Custom components: {custom_path}")
        print(f"   - Model information: {info_path}")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load a model from a pretrained checkpoint.
        This method loads the underlying Hugging Face model and our custom components.
        """
        from pathlib import Path
        
        model_path = Path(pretrained_model_name_or_path)
        
        # Load the underlying Hugging Face model
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            **kwargs
        )
        
        # Create our custom model
        model = cls.__new__(cls)
        model.model = hf_model
        model.hidden_size = hf_model.config.hidden_size
        model.first_dropout = nn.Dropout(0.1)
        model.summary = nn.Linear(model.hidden_size, 1)
        
        # Load custom components if they exist
        custom_path = model_path / "custom_components.bin"
        if custom_path.exists():
            custom_state_dict = torch.load(custom_path, map_location='cpu')
            model.summary.weight.data = custom_state_dict['summary.weight']
            model.summary.bias.data = custom_state_dict['summary.bias']
            print(f"✅ Load custom components: {custom_path}")
        
        # Initialize the model properly
        model.__init__ = lambda *args, **kwargs: None  # Prevent re-initialization
        return model

def respond_to_batch(model, source_ids, attention_mask, max_target_length=400, top_k=5, top_p=1.0, tokenizer=None,temperature=0.7,repetition_penalty=1.1, do_sample=True):
    """
    Generate responses from batch prompts. Supports wrapper or HF original model.
    """
    #print(f"respond_to_batch input: source_ids={source_ids.shape}, attention_mask={attention_mask.shape}")
    hf_model = model.model
    generation_config = {
        'do_sample': do_sample,
        'top_k': top_k,
        'top_p': top_p,
        'temperature': temperature,  # 🔧 Add temperature to reduce repetition
        'repetition_penalty': repetition_penalty,  # 🔧 Add repetition_penalty to reduce repetition
        "max_new_tokens": max_target_length,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        preds = hf_model.generate(input_ids=source_ids,
                                    attention_mask=attention_mask,
                                    **generation_config)
        #print("Original complete output："+ tokenizer.decode(preds[0][source_ids.shape[1]:],skip_special_tokens=True))
    torch.cuda.empty_cache()
    return preds

# Backward compatibility alias
CodeT5HeadWithValueModelLocal = QwenCoderHeadWithValueModelLocal
