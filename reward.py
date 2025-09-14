import numpy as np
from tree_sitter import Language, Parser
import re
import torch
from code_prepro.lang_processors import *
from compiler.terminal_compiler import TerminalCompiler
import sys
from code_parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript,DFG_csharp
from termcolor import colored
import subprocess
import tempfile
import os

# Set environment variable to avoid tokenizer warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, '/home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/CodeBLEU/')
from codebleu.calc_code_bleu import calc_code_bleu


def format_code_with_clang_format(code, style='Google'):
    """Format C++ code with clang-format"""
    try:
        # Set environment variable to avoid tokenizer warning
        env = os.environ.copy()
        env['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        # Call clang-format
        result = subprocess.run(
            ['clang-format', f'--style={style}', temp_file],
            capture_output=True, text=True, check=True, env=env
        )
        
        # Clean up temporary file
        os.unlink(temp_file)
        
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If clang-format fails or is not installed, return original code
        return code

def calc_code_bl_with_format(reference_code, generated_code, lang, keywords_dir):
    """Calculate CodeBLEU score, preprocess C++ code with formatting"""
    if lang in ['cpp', 'c']:
        # Format C++ code
        formatted_reference = format_code_with_clang_format(reference_code)
        formatted_generated = format_code_with_clang_format(generated_code)
        result = calc_code_bleu([[formatted_reference]], [formatted_generated], lang, keywords_dir)
    else:
        # Other languages remain unchanged
        result = calc_code_bleu([[reference_code]], [generated_code], lang, keywords_dir)
    
    return result


code_tokenizers = {"java": java_tokenizer, "cpp": cpp_tokenizer, "c": c_tokenizer, "python": py_tokenizer,
                   "javascript": js_tokenizer, "php": php_tokenizer, "c_sharp": cs_tokenizer}
code_detokenizers = {"java": java_detokenizer, "cpp": cpp_detokenizer, "c": c_detokenizer, "python": py_detokenizer,
                   "javascript": js_detokenizer, "php": php_detokenizer, "c_sharp": cs_detokenizer}

lang2compiler = {
    "python": TerminalCompiler("Python"),
    "java": TerminalCompiler("Java"),
    "cpp": TerminalCompiler("C++"),
    "c_sharp": TerminalCompiler("C#"),
    "c": TerminalCompiler("C"),
    "php": TerminalCompiler("PHP"),
}

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
    parsers[lang]= parser
    
def remove_special_tokens(code_string):
    lines = code_string.split("NEW_LINE")
    lines = [item.strip() for item in lines]
    
    curr_indent = 0
    new_lines = []
    for line in lines:
        indent_count = line.count('INDENT')
        dedent_count = line.count('DEDENT')
        curr_indent += indent_count - dedent_count
        wo_indent = re.sub('INDENT\s?', '', line)
        wo_dedent = re.sub('DEDENT\s?', '', wo_indent)
        new_lines.append('\t'*curr_indent + wo_dedent)
    return ("\n").join(new_lines)

def dfs_parse_tree(node, level, count_list, verbose = False):
    if verbose:
        if node.type == 'ERROR':
            print (level, '-'*(level*2), colored(node.type, 'red'))
        else:
            print (level, '-'*(level*2), node.type)
    if node.type == 'ERROR':
        count_list[0]+=1
    else:
        count_list[1]+=1
    for child in node.children:
        dfs_parse_tree(child, level+1, count_list, verbose)
    return

def tree_sitter_full_compile(code, lang='python', verbose = False):
    root=parsers[lang].parse(bytes(code, 'utf-8')).root_node
    count_list = [0, 0]
    dfs_parse_tree(root, 0, count_list, verbose)
    return count_list


def get_reward(lang, code_ids=None,code_ref_ids=None,gold_ids=None, tokenizer=None):
    code_ids = np.array(code_ids.cpu())
    code_ref_ids = np.array(code_ref_ids.cpu())
    gold_ids = np.array(gold_ids.cpu())
    
    eos_positions = []
    max_len = code_ids.shape[1]
    for id in code_ids:
        if tokenizer.eos_token_id in id:
            eos_positions.append((id==tokenizer.eos_token_id).argmax())
        else:
            eos_positions.append(max_len)
    
    eos_positions_ref = []
    max_len_ref = code_ref_ids.shape[1]
    for id in code_ref_ids:
        if tokenizer.eos_token_id in id:
            eos_positions_ref.append((id==tokenizer.eos_token_id).argmax())
        else:
            eos_positions_ref.append(max_len_ref)
    
    eos_positions_gold = []
    max_len_gold = gold_ids.shape[1]
    for id in gold_ids:
        if tokenizer.eos_token_id in id:
            eos_positions_gold.append((id==tokenizer.eos_token_id).argmax())
        else:
            eos_positions_gold.append(max_len_gold)

    codes = [tokenizer.decode(id[:eos_pos], skip_special_tokens=True, clean_up_tokenization_spaces=False) \
             for id,eos_pos in zip(code_ids, eos_positions)]
    codes_ref = [tokenizer.decode(id[:eos_pos], skip_special_tokens=True, clean_up_tokenization_spaces=False) \
             for id,eos_pos in zip(code_ref_ids, eos_positions_ref)] 
    codes_gold = [tokenizer.decode(id[:eos_pos], skip_special_tokens=True, clean_up_tokenization_spaces=False) \
             for id,eos_pos in zip(gold_ids, eos_positions_gold)]
        
    codes = [code_detokenizers[lang](code) for code in codes]
    codes_ref = [code_detokenizers[lang](code) for code in codes_ref]
    codes_gold = [code_detokenizers[lang](code) for code in codes_gold]
    
    compilation = [lang2compiler[lang].compile_code_string(code) for code in codes]
    compilation_ref = [lang2compiler[lang].compile_code_string(code) for code in codes_ref]

    codes = [remove_special_tokens(code) for code in codes]
    codes_ref = [remove_special_tokens(code) for code in codes_ref]
    codes_gold = [remove_special_tokens(code) for code in codes_gold]
    error_node_counts = [tree_sitter_full_compile(code,lang) for code in codes]
    error_node_counts_ref = [tree_sitter_full_compile(code,lang) for code in codes_ref]
    error_node_counts_gold = [tree_sitter_full_compile(code,lang) for code in codes_gold]
    num_errors = [i[0] for i in error_node_counts]
    num_errors_ref = [i[0] for i in error_node_counts_ref]  
    num_errors_gold = [i[0] for i in error_node_counts_gold]  
    num_nodes = [i[1] for i in error_node_counts]
    num_nodes_ref = [i[1] for i in error_node_counts_ref]
    num_nodes_gold = [i[1] for i in error_node_counts_gold]
    
    keywords_dir = 'codebleu/keywords/'
    # ast_match = calc_code_bleu([codes_gold], codes, lang, keywords_dir)[2]
    # dfg_match = calc_code_bleu([codes_gold], codes, lang, keywords_dir)[3]
    
    rewards = np.zeros_like(code_ids, dtype=np.float64)
    rewards_ref = np.zeros_like(code_ref_ids, dtype=np.float64)
    ast_match_batch = 0
    dfg_match_batch = 0
    compile_batch = 0
    ast_match_batch_ref = 0
    dfg_match_batch_ref = 0
    compile_batch_ref = 0
    
    sample_compilation_success = []  # Each sample's compilation success status
    sample_ast_match = []           # Each sample's AST matching score
    sample_dfg_match = []           # Each sample's DFG matching score
    sample_compilation_success_ref = []  # Reference sample's compilation success status
    sample_ast_match_ref = []           # Reference sample's AST matching score
    sample_dfg_match_ref = []           # Reference sample's DFG matching score
    
    for i in range(len(rewards)):
        _, _, did_compile = compilation[i]
        reward = 1 if did_compile else -1
        
        ast_match = calc_code_bl_with_format(codes_gold[i], codes[i], lang, keywords_dir)[2]
        dfg_match = calc_code_bl_with_format(codes_gold[i], codes[i], lang, keywords_dir)[3]

        rewards[i, min(eos_positions[i],max_len-1)] = reward + ast_match + dfg_match
        # seq_len = eos_positions[i] + 1
        # per_token_r = reward / seq_len
        # rewards[i, :seq_len] += per_token_r 

        #total_reward = reward + ast_match + dfg_match
        #seq_len = min(eos_positions[i],max_len-1)+1
        #per_token_r = total_reward
        #rewards[i, :seq_len] += per_token_r 
        
        compile_batch += (1 if did_compile else 0) 
        ast_match_batch += ast_match
        dfg_match_batch += dfg_match
        
        # Record each sample's detailed information
        sample_compilation_success.append(did_compile)
        sample_ast_match.append(ast_match)
        sample_dfg_match.append(dfg_match)
        
        # Calculate ref rewards
        _, _, did_compile_ref = compilation_ref[i]
        reward_ref = 1 if did_compile_ref else -1
        
        ast_match_ref = calc_code_bl_with_format(codes_gold[i], codes_ref[i], lang, keywords_dir)[2]
        dfg_match_ref = calc_code_bl_with_format(codes_gold[i], codes_ref[i], lang, keywords_dir)[3]

        rewards_ref[i, min(eos_positions[i],max_len-1)] = reward_ref + ast_match_ref + dfg_match_ref
        # seq_len = eos_positions[i] + 1
        # per_token_r_ref = reward_ref / seq_len
        # rewards_ref[i, :seq_len] += per_token_r_ref 
        
        compile_batch_ref += (1 if did_compile_ref else 0)
        ast_match_batch_ref += ast_match_ref
        dfg_match_batch_ref += dfg_match_ref
        
        # Record reference sample's detailed information
        sample_compilation_success_ref.append(did_compile_ref)
        sample_ast_match_ref.append(ast_match_ref)
        sample_dfg_match_ref.append(dfg_match_ref)
     
    mean_rate = compile_batch/len(codes)
    mean_ast_match =  ast_match_batch/len(codes) 
    mean_dfg_match =  dfg_match_batch/len(codes)  
    mean_rate_ref = compile_batch_ref/len(codes_ref)
    mean_ast_match_ref =  ast_match_batch_ref/len(codes_ref) 
    mean_dfg_match_ref =  dfg_match_batch_ref/len(codes_ref)
    
    # Create detailed sample information dictionary
    sample_details = {
        'compilation_success': sample_compilation_success,
        'ast_match': sample_ast_match,
        'dfg_match': sample_dfg_match,
        'compilation_success_ref': sample_compilation_success_ref,
        'ast_match_ref': sample_ast_match_ref,
        'dfg_match_ref': sample_dfg_match_ref
    }
    
    return torch.Tensor(rewards), torch.Tensor(rewards_ref), mean_rate, mean_ast_match, mean_dfg_match, mean_rate_ref, mean_ast_match_ref, mean_dfg_match_ref, num_errors, num_errors_ref, num_nodes, num_nodes_ref, sample_details

