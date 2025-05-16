import json
import os
from typing import List, Dict
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Any
from tqdm import tqdm
import time
import requests
import tiktoken
import copy
import traceback

def prepare_prompt(tokenizer, task, model_type, model_name, left_cxt, right_cxt=None, crossfile_cxt=None, args=None):
    
    #print("args.max_seq_length, args.gen_length,  args.right_context_length, args.cfc_seq_length", args.max_seq_length, args.gen_length,  args.right_context_length, args.cfc_seq_length)
    if task == "function_completion":
        args.gen_length = 256


    adds_bos = False
    if hasattr(tokenizer, 'init_kwargs') and 'add_bos_token' in tokenizer.init_kwargs:
        adds_bos = tokenizer.init_kwargs['add_bos_token']
    
    if model_type == "codelm_leftright_context":
        left_tokens = tokenizer.encode(left_cxt)
        right_tokens = tokenizer.encode(right_cxt)
        
        if adds_bos:
            if len(left_tokens) > 0:
                left_tokens = left_tokens[1:]
            if len(right_tokens) > 0:
                right_tokens = right_tokens[1:]

        
        left_tokens_truncated = left_tokens[-(args.max_seq_length - args.gen_length - args.right_context_length):]
        right_tokens_truncated = right_tokens[:args.right_context_length]
        
        left_cxt_truncated = tokenizer.decode(left_tokens_truncated)
        right_cxt_truncated = tokenizer.decode(right_tokens_truncated)
        crossfile_cxt_truncated = None
    elif model_type == "codelm_right_cfc_left":
        assert crossfile_cxt is not None
        
        left_tokens = tokenizer.encode(left_cxt)
        right_tokens = tokenizer.encode(right_cxt)
        crossfile_tokens = tokenizer.encode('\n\n' + crossfile_cxt)
        
        if adds_bos:
            if len(left_tokens) > 0:
                left_tokens = left_tokens[1:]
            if len(right_tokens) > 0:
                right_tokens = right_tokens[1:]
            if len(crossfile_tokens) > 0:
                crossfile_tokens = crossfile_tokens[1:]

        
        left_tokens_truncated = left_tokens[-(args.max_seq_length - args.gen_length - args.right_context_length - args.cfc_seq_length):]
        right_tokens_truncated = right_tokens[:args.right_context_length]
        crossfile_tokens_truncated = crossfile_tokens[:args.cfc_seq_length]
        
        left_cxt_truncated = tokenizer.decode(left_tokens_truncated)
        right_cxt_truncated = tokenizer.decode(right_tokens_truncated)
        crossfile_cxt_truncated = tokenizer.decode(crossfile_tokens_truncated)
    else:
        raise NotImplementedError

    ## instruct fim
    if args.instruct_mode:
        
        if args.template_type == "qwen":
            COMPLETION_PROMPT = "You are a code completion assistant."
            REPO_COMPLETE_TEMPLATE = """Please complete the middle code, based on the given prefix/suffix code of the current file and context code of other files.
##Context Code##:
{}
##Prefix Code##:
{}
##Suffix Code##:
{}
##Middle Code##:
"""
            user_prompt = REPO_COMPLETE_TEMPLATE.format(crossfile_cxt_truncated, left_cxt_truncated, right_cxt_truncated)
            dialog_dict = [
                {"role": "system", "content": COMPLETION_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            prompt = tokenizer.apply_chat_template(dialog_dict, add_generation_prompt=True, tokenize=False)

        elif args.template_type == "special_token":
            COMPLETION_PROMPT = "You are a code completion assistant."
            REPO_COMPLETE_TEMPLATE = """<|fim_prefix|>{}<|fim_suffix|>{}{}<|fim_middle|>"""

            user_prompt = REPO_COMPLETE_TEMPLATE.format(left_cxt_truncated, right_cxt_truncated, crossfile_cxt_truncated)
            dialog_dict = [
                {"role": "system", "content": COMPLETION_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            prompt = tokenizer.apply_chat_template(dialog_dict, add_generation_prompt=True, tokenize=False)

        elif args.template_type == "system":
            COMPLETION_PROMPT = "You are a code completion assistant."
            REPO_COMPLETE_TEMPLATE = """Please complete the middle code, based on the given prefix/suffix code of the current file and context code of other files.
<|fim_prefix|>{}<|fim_suffix|>{}{}<|fim_middle|>"""

            user_prompt = REPO_COMPLETE_TEMPLATE.format(left_cxt_truncated, right_cxt_truncated, crossfile_cxt_truncated)
            dialog_dict = [
                {"role": "system", "content": COMPLETION_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            prompt = tokenizer.apply_chat_template(dialog_dict, add_generation_prompt=True, tokenize=False)


    ## base fim
    else:
       # 设置模型特定的tokens
        if "deepseek" in model_name.lower():
            prefix_token = '<｜fim▁begin｜>'
            middle_token = '<｜fim▁end｜>'
            suffix_token = '<｜fim▁hole｜>'
        elif "qwen1.5" in model_name.lower():
            prefix_token = '<fim_prefix>'
            middle_token = '<fim_middle>'
            suffix_token = '<fim_suffix>'
        elif "qwen" in model_name.lower():
            prefix_token = '<|fim_prefix|>'
            middle_token = '<|fim_middle|>'
            suffix_token = '<|fim_suffix|>'
        else:
            prefix_token = '<|fim_prefix|>'
            middle_token = '<|fim_middle|>'
            suffix_token = '<|fim_suffix|>'
        
        if model_type == "codelm_leftright_context":
            prompt = f'{prefix_token}{left_cxt_truncated}{suffix_token}{right_cxt_truncated}{middle_token}'
        elif model_type == "codelm_right_cfc_left":
            prompt = f'{prefix_token}{left_cxt_truncated}{suffix_token}{right_cxt_truncated}{crossfile_cxt_truncated}{middle_token}'
        else:
            raise NotImplementedError
    return prompt