
__all__ = ['extract_structure','flatten_dict', 'stack_dicts', 'add_suffix', 'pad_to_size', 'logprobs_from_logits', 'whiten',
           'clip_by_value', 'entropy_from_logits', 'average_torch_dicts', 'stats_to_np', 'build_bert_batch_from_txt',
           'extract_code_from_qwen_response', 'read_qwen_examples', 'convert_qwen_examples_to_features', 'create_reward_wrapper']

# Cell
import torch
import torch.nn.functional as F
import collections
import numpy as np
from tqdm import tqdm
from code_parser import (tree_to_token_index,
                   tree_to_token_nodes,
                   index_to_code_token,
                   tree_to_variable_index, 
                   detokenize_code)
from code_parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript,DFG_csharp
from tree_sitter import Language, Parser
import pickle

dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'php':DFG_php,
    'javascript':DFG_javascript,
    'c_sharp':DFG_csharp,
    'c':DFG_csharp,
    'cpp':DFG_csharp,}
parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('code_parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser


class Example(object):
    def __init__(self,
                 idx,
                 source,
                 target,
                 source_orig,
                 target_orig
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.source_orig = source_orig
        self.target_orig = target_orig


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
                 target):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask   
        self.target = target
        

def read_examples(filename, args):
    examples=[]
    assert len(filename.split(','))==2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1,open(trg_filename) as f2:
            for line1,line2 in zip(f1,f2):
                line1=line1.strip().replace('▁', '_')
                line2=line2.strip().replace('▁', '_')
                if (args.l1=='php') and not(line1.startswith('<?php')):
                    line1 = '<?php '+line1
                if (args.l2 =='php') and not(line2.startswith('<?php')):
                    line2 = '<?php '+line2
                    
                orig_line1, orig_line2 = line1, line2
                
                if args.l1=='python':
                    line1 = detokenize_code(line1)
                else:
                    line1 = line1.replace('NEW_LINE', '\n')
                if args.l2=='python':
                    line2 = detokenize_code(line2)
                else:
                    line2 = line2.replace('NEW_LINE', '\n')

                examples.append(
                Example(idx = idx,
                        source=line1,
                        target=line2,
                        source_orig = orig_line1,
                        target_orig = orig_line2) )
                idx+=1
    return examples


def convert_examples_to_features(examples, tokenizer, args,stage=None):
    features = []
    for example_index, example in enumerate(examples):
        #source
        source_tokens = tokenizer.tokenize(example.source_orig)[:args.max_source_length-2]
        source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
        source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids+=[tokenizer.pad_token_id]*padding_length
        source_mask+=[0]*padding_length
 
        #target
        if stage=="test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target_orig)[:args.max_target_length-1]
        target_tokens = target_tokens+[tokenizer.sep_token]            
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] *len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        # target_ids+=[-100]*padding_length
        #MODIFIED
        target_ids+=[tokenizer.pad_token_id]*padding_length
        target_mask+=[0]*padding_length   

        features.append(InputFeatures(
                 example_index,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,
                 example.target_orig))
    # breakpoint()
    return features


def extract_structure(code, parser, lang):  
    try:
        # ast
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        ast_token_nodes = tree_to_token_nodes(root_node)
        tokens_index = [(node.start_point, node.end_point) for node in ast_token_nodes]
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index] 
        
        # dfg
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg,ast_token_nodes


def get_lr_path(leaf):
    if leaf==-1:
        return -1
    path = [leaf]
    while path[-1].parent is not None:
        path.append(path[-1].parent)
    return path


def get_node_types(node, l):
    l.append(node.type)
    for child in node.children:
        get_node_types(child, l)
        
        
def gather_node_types(examples, args):
    global node_types
    filename = args.output_dir+'/node_types.pkl'
    node_types = []
    for example in tqdm(examples):
        root = parsers[args.source_lang][0].parse(bytes(example.source,'utf8')).root_node 
        get_node_types(root, node_types)
        root = parsers[args.target_lang][0].parse(bytes(example.target,'utf8')).root_node 
        get_node_types(root, node_types)
    node_types = sorted(list(set(node_types)))
    pickle.dump(node_types, open(filename, 'wb'))
    node_types = {t:i for i,t in enumerate(node_types)}


def convert_path_to_idx(path, max_depth):
    if path==-1:
        return [-1]*max_depth
    path = [node_types.get(node.type, -1) for node in path][:max_depth]
    path = path + [-1]*(max_depth-len(path))
    return path


def convert_examples_to_ast_dfg(examples, tokenizer, args, stage=None):
    features = []
    match, nomatch = 1,1
    smatch, snomatch = 1,1
    bar = tqdm(enumerate(examples))
    for example_index, example in bar: 
        target_tokens = tokenizer.tokenize(example.target_orig)[:args.max_source_length-2]
        code_tokens,dfg,ast = extract_structure(example.target, parsers[args.target_lang], args.target_lang)
        for i in range(1, len(ast)):
            if (ast[i].start_point[0]<ast[i-1].start_point[0]) or \
                    ((ast[i].start_point[0]==ast[i-1].start_point[0]) and (ast[i].start_point[1]<ast[i-1].start_point[1])):
                raise Exception("Leaves not ordered by position in sequence.")      
    tcode = list(''.join(target_tokens).replace('Ġ', ' ').replace('ĉ', '\t'))
    scode = list(''.join(code_tokens))
    tcode_to_scode = []
    j = 0
    for i in range(len(tcode)):
        if j<len(scode):
            if tcode[i]==scode[j]:
                tcode_to_scode.append(j)
                j += 1
                match += 1
            else:
                tcode_to_scode.append(-1)
                if (tcode[i]!=' '):
                    if (tcode[i] not in [' ','N','E','W','_','L','I','N','E']):
                        nomatch += 1
        else:
            tcode_to_scode.append(-1)
            if (tcode[i]!=' '):
                if (tcode[i] not in [' ','N','E','W','_','L','I','N','E']):
                    nomatch += 1
    tcode_to_target = []
    for i in range(len(target_tokens)):
        tcode_to_target += [i]*len(target_tokens[i])
    scode_to_code = []
    for i in range(len(code_tokens)):
        scode_to_code += [i]*len(code_tokens[i])  
    target_to_code = [[] for i in range(len(target_tokens))]
    for i in range(len(tcode)):
        if tcode_to_scode[i]>=0:
            target_to_code[tcode_to_target[i]].append( scode_to_code[tcode_to_scode[i]] )
    code_to_target = [[] for i in range(len(code_tokens))]
    for i in range(len(target_to_code)):
        for c in set(target_to_code[i]):
            code_to_target[c].append(i) 

    target_tokens = target_tokens+[tokenizer.sep_token]            
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    
    target_len = len(target_ids)
    target_dfg = np.zeros((target_len, target_len))
    target_ast = -np.ones((target_len, args.max_ast_depth))
    target_ast_sim = -np.ones((target_len, target_len))
    tlr_paths = [get_lr_path(leaf) for leaf in ast]
    tlr_paths = [convert_path_to_idx(path, args.max_ast_depth) for path in tlr_paths]
    for i,ts in enumerate(code_to_target):
        target_ast[ts, :] = np.array(tlr_paths[i]).reshape((1,-1))
    for _,l,_,_,rs in dfg:
        for lt in code_to_target[l]:
            for r in rs:
                target_dfg[lt, code_to_target[r]] = 1
    target_dfg[-1,:] = -1
    target_dfg[:,-1] = -1
    
    return target_dfg, target_ast


def flatten_dict(nested, sep='/'):
    """Flatten dictionary and concatenate nested keys with separator."""
    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, collections.Mapping):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v
    flat = {}
    rec(nested, '', flat)
    return flat

def stack_dicts(stats_dicts):
    """Stack the values of a dict."""
    results = dict()
    for k in stats_dicts[0]:
        stats_list = [torch.flatten(d[k]) for d in stats_dicts]
        max_len = max([len(l) for l in stats_list])
        stats_list = [torch.cat((l.cpu(),torch.ones(max_len-len(l)))) for l in stats_list]
        results[k] = torch.stack(stats_list)
    return results

def add_suffix(input_dict, suffix):
    """Add suffix to dict keys."""
    return dict((k + suffix, v) for k,v in input_dict.items())

# Cell

def pad_to_size(tensor, size, dim=1, padding=50256):
    """Pad tensor to size."""
    t_size = tensor.size()[dim]
    if t_size==size:
        return tensor
    else:
        return torch.nn.functional.pad(tensor, (0,size-t_size), 'constant', padding)

def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy
    
    # logpy = torch.gather(logits, 2, labels.unsqueeze(2)).squeeze(-1)
    # logp = F.log_softmax(logpy, dim=-1)
    # return logp


def whiten(values, shift_mean=True, eps: float = 1e-8, min_std: float = 1e-3):
    """
    Standardise *along batch dim* (dim=0) 以保持序列内部原始方差。

    - 若 `values` 是二维 (B, T) ⇒ 先沿 dim=0 求均值/方差，再广播回去；
    - 若是一维向量 ⇒ 保持旧行为，只要给 std 加下界即可。
    """
    if values.dim() > 1:                       # e.g. (B, T)
        mean = values.mean(dim=0, keepdim=True)
        std  = values.std (dim=0, keepdim=True).clamp_min(min_std)
    else:                                      # 旧的一维用法
        mean = values.mean()
        std  = values.std().clamp_min(min_std)

    whitened = (values - mean) / (std + eps)
    if not shift_mean and values.dim() == 1:
        whitened += mean
    return whitened

def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped

def entropy_from_logits(logits):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, axis=-1) - torch.sum(pd*logits, axis=-1)
    return entropy


def average_torch_dicts(list_of_dicts):
    """Average values of a list of dicts wiht torch tensors."""
    average_dict = dict()
    for key in list_of_dicts[0].keys():
        average_dict[key] = torch.mean(torch.stack([d[key] for d in list_of_dicts]), axis=0)
    return average_dict

def stats_to_np(stats_dict):
    """Recursively detach / move to CPU / convert to numpy. 兼容 bf16."""
    new_dict = {}
    for k, v in stats_dict.items():
        if torch.is_tensor(v):
            if v.dtype == torch.bfloat16:
                v = v.to(torch.float32)  # numpy 不支持 bf16
            new_dict[k] = v.detach().cpu().numpy()
        elif isinstance(v, dict):
            new_dict[k] = stats_to_np(v)
        else:
            new_dict[k] = v
    return new_dict


# Cell

def build_bert_batch_from_txt(text_list, tokenizer, device):
    """Create token id and attention mask tensors from text list for BERT classification."""

    # tokenize
    tensors = [tokenizer.encode(txt, return_tensors="pt").to(device) for txt in text_list]

    # find max length to pad to
    max_len = max([t.size()[1] for t in tensors])

    # get padded tensors and attention masks
    # (attention masks make bert ignore padding)
    padded_tensors = []
    attention_masks = []
    for tensor in tensors:
        attention_mask = torch.ones(tensor.size(), device=device)
        padded_tensors.append(pad_to_size(tensor, max_len, padding=0))
        attention_masks.append(pad_to_size(attention_mask, max_len, padding=0))

    # stack all tensors
    padded_tensors = torch.cat(padded_tensors)
    attention_masks = torch.cat(attention_masks)

    return padded_tensors, attention_masks


# ========================================
# Qwen Data Processing Functions
# ========================================

def extract_code_from_qwen_response(response: str, target_lang: str = "cpp") -> str:
    """Extract code from Qwen response"""
    import re
    
    # Language pattern mapping
    lang_patterns = {
        'cpp': ['cpp', 'c++', 'cxx'], 'java': ['java'], 'python': ['python', 'py'], 
        'javascript': ['javascript', 'js'], 'c': ['c'], 'php': ['php'], 'c_sharp': ['csharp', 'c#', 'cs']
    }
    
    # Try matching specific language code block (full match)
    for pattern in lang_patterns.get(target_lang, [target_lang]):
        escaped_pattern = re.escape(pattern)
        code_match = re.search(rf'```{escaped_pattern}\s*(.*?)```', response, re.DOTALL | re.IGNORECASE)
        if code_match:
            return code_match.group(1).strip().lstrip('\n')
    
    # Try matching incomplete code block (only start marker)
    for pattern in lang_patterns.get(target_lang, [target_lang]):
        escaped_pattern = re.escape(pattern)
        # Match ```python start, but no end marker
        incomplete_match = re.search(rf'```{escaped_pattern}\s*(.*)', response, re.DOTALL | re.IGNORECASE)
        if incomplete_match:
            extracted = incomplete_match.group(1).strip().lstrip('\n')
            # Remove possible end markdown marker
            extracted = re.sub(r'```\s*$', '', extracted)
            return extracted
    
    # General code block matching (full)
    code_match = re.search(r'```\s*(.*?)```', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip().lstrip('\n')
    
    # General code block matching (incomplete)
    incomplete_match = re.search(r'```\s*(.*)', response, re.DOTALL)
    if incomplete_match:
        extracted = incomplete_match.group(1).strip().lstrip('\n')
        extracted = re.sub(r'```\s*$', '', extracted)
        return extracted
    
    # translation: marker matching
    translation_match = re.search(r'translation:\s*\n\n(.+)', response, re.DOTALL | re.IGNORECASE)
    if translation_match:
        return translation_match.group(1).strip()
    
    # Remove common prefixes
    prefixes = ["Here's the C++ translation:", "Here's the Java translation:", "Here's the translation:", 
               "Translation:", "```cpp", "```c++", "```java", "```python", "```"]
    
    for prefix in prefixes:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
    
    # Remove end marker and markdown line
    response = response.rstrip("```").strip()
    lines = [line for line in response.split('\n') 
             if line.strip() not in ['```cpp', '```c++', '```java', '```python', '```javascript', '```c', '```php', '```csharp', '```']]
    
    return '\n'.join(lines).strip()


def read_qwen_examples(filename: str, args) -> list:
    """Read Qwen format JSONL file"""
    import json
    
    examples = []
    with open(filename, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                messages = {msg.get('role'): msg.get('content', '') for msg in data.get('messages', [])}
                
                user_msg, assistant_msg = messages.get('user'), messages.get('assistant')
                if not user_msg or not assistant_msg:
                    continue
                
                source_code = extract_code_from_qwen_response(user_msg, args.source_lang)
                target_code = extract_code_from_qwen_response(assistant_msg, args.target_lang)
                
                if source_code and target_code:
                    e = Example(idx=idx, source=source_code, target=target_code, 
                               source_orig=user_msg, target_orig=assistant_msg)
                    setattr(e, "system_orig", messages.get('system', ''))
                    examples.append(e)
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"Skipping line {idx+1}, parsing error: {e}")
    return examples


def convert_qwen_examples_to_features(examples, tokenizer, args, stage=None):
    """Convert Qwen samples to model input features"""
    features = []
    default_system = "You are a helpful assistant for code translation. You specialize in translating Java code to C++ code while maintaining functionality and best practices."
        
    for example_index, example in enumerate(examples):
        # Apply chat template
        if hasattr(tokenizer, 'apply_chat_template'):
            try:
                system_content = getattr(example, "system_orig", "") or default_system
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": example.source_orig}
                ]
                source_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            except Exception as e:
                print(f"❌ apply_chat_template failed: {e}")
                source_text = example.source_orig
        else:
            source_text = example.source_orig
            
        # Encode source text and target text
        source_ids = tokenizer.encode(source_text, max_length=args.max_source_length, truncation=True, add_special_tokens=True)
        target_text = "None" if stage == "test" else example.target_orig
        target_ids = tokenizer.encode(target_text, max_length=args.max_target_length, truncation=True, add_special_tokens=True)
        
        # Left padding
        def pad_left(ids, max_length):
            mask = [1] * len(ids)
            padding = max_length - len(ids)
            return [tokenizer.pad_token_id] * padding + ids, [0] * padding + mask
        
        source_ids, source_mask = pad_left(source_ids, args.max_source_length)
        target_ids, target_mask = pad_left(target_ids, args.max_target_length)
        
        features.append(InputFeatures(example_index, source_ids, target_ids, source_mask, target_mask, example.target_orig))
    return features


def create_reward_wrapper(original_get_reward):
    """Wrap reward function, handle code extraction and re-encoding"""
    def get_reward_with_extraction(lang, code_ids=None, code_ref_ids=None, gold_ids=None, tokenizer=None):
        def _decode_rows(t):
            import torch
            arr = t.detach().cpu().numpy()
            eos_id = tokenizer.eos_token_id
            texts = []
            for row in arr:
                eos_pos = int((row == eos_id).argmax()) if eos_id in row else len(row)
                texts.append(tokenizer.decode(row[:eos_pos], skip_special_tokens=True, clean_up_tokenization_spaces=False))
            return texts

        # Decode original response and extract code
        raw_responses = _decode_rows(code_ids)
        raw_gold = _decode_rows(gold_ids)

        extracted_codes = [extract_code_from_qwen_response(txt, lang) for txt in raw_responses]
        extracted_codes_gold = [extract_code_from_qwen_response(txt, lang) for txt in raw_gold]

        # Handle reference code (may be None, e.g. in GRPO)
        if code_ref_ids is not None:
            raw_responses_ref = _decode_rows(code_ref_ids)
            extracted_codes_ref = [extract_code_from_qwen_response(txt, lang) for txt in raw_responses_ref]
        else:
            # If there is no reference code, use empty string as placeholder
            extracted_codes_ref = [""] * len(extracted_codes)

        # Re-encode and pad
        import torch
        eos_id, pad_id = tokenizer.eos_token_id, tokenizer.pad_token_id
        triplets = [(tokenizer.encode(c, add_special_tokens=False) + [eos_id],
                    tokenizer.encode(r, add_special_tokens=False) + [eos_id],
                    tokenizer.encode(g, add_special_tokens=False) + [eos_id])
                   for c, r, g in zip(extracted_codes, extracted_codes_ref, extracted_codes_gold)]

        max_len = max(len(x) for tri in triplets for x in tri) if triplets else 1
        _pad = lambda seq: seq + [pad_id] * (max_len - len(seq))

        policy_padded = [_pad(x[0]) for x in triplets]
        ref_padded = [_pad(x[1]) for x in triplets]
        gold_padded = [_pad(x[2]) for x in triplets]

        # Convert to tensor and call original reward function
        # Get device information (use code_ids device if code_ref_ids is None)
        device = code_ids.device if code_ids is not None else gold_ids.device
        
        return original_get_reward(
            lang=lang,
            code_ids=torch.tensor(policy_padded, dtype=torch.long, device=device),
            code_ref_ids=torch.tensor(ref_padded, dtype=torch.long, device=device),
            gold_ids=torch.tensor(gold_padded, dtype=torch.long, device=device),
            tokenizer=tokenizer,
        )
    return get_reward_with_extraction

