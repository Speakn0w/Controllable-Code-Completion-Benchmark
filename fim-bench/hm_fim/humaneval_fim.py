#!/usr/bin/env python
# coding=utf-8

import argparse
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm import LLM, SamplingParams
from typing import Optional
from utils import prepare_prompt
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
from collections import defaultdict
from rich import print
from rich.table import Table
from fuzzywuzzy import fuzz
import numpy as np




def evaluate_results(results_file):
    language_results = defaultdict(lambda: {
        "count": 0,
        "exact_matches": 0,
        "edit_similarities": []
    })
    
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            model_gen = data.get("model_gen")
            canonical_solution = data.get("canonical_solution")
            language = data.get("language", "Unknown")
            
            # Process model generation
            if model_gen.startswith("\n"):
                response = model_gen.split("\n")[1]
            elif model_gen.startswith(" \n"):
                response = model_gen.split("\n")[1]
            else:
                response = model_gen.split("\n")[0]
            
            response = response.strip()
            canonical_solution = canonical_solution.strip()
            
            # Update statistics
            language_results[language]["count"] += 1
            if response == canonical_solution:
                language_results[language]["exact_matches"] += 1
            
            # Calculate edit similarity
            similarity = fuzz.ratio(response, canonical_solution)
            language_results[language]["edit_similarities"].append(similarity)
    
    # Calculate final metrics
    final_results = {}
    total_count = 0
    total_exact_matches = 0
    total_edit_similarites = 0
    all_similarities = []

    
    for lang, stats in language_results.items():
        exact_match_rate = (stats["exact_matches"] / stats["count"]) * 100
        avg_similarity = np.mean(stats["edit_similarities"])
        
        final_results[lang] = {
            "count": stats["count"],
            "exact_match_rate": exact_match_rate,
            "average_edit_similarity": avg_similarity
        }
        
        total_count += stats["count"]
        #total_exact_matches += stats["exact_matches"]
        total_exact_matches += exact_match_rate
        #all_similarities.extend(stats["edit_similarities"])
        total_edit_similarites += avg_similarity
    
    # Add overall results
    final_results["overall"] = {
        "count": total_count,
        "exact_match_rate": (total_exact_matches / len(language_results)),
        "average_edit_similarity": (total_edit_similarites / len(language_results))
    }
    
    return final_results

def print_results_table(results):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Language")
    table.add_column("Count")
    table.add_column("Exact Match Rate")
    table.add_column("Edit Similarity")
    
    # Add results for each language
    for lang, metrics in results.items():
        if lang != "overall":
            table.add_row(
                lang,
                str(metrics["count"]),
                f"{metrics['exact_match_rate']:.2f}%",
                f"{metrics['average_edit_similarity']:.2f}%"
            )
    
    # Add overall results
    table.add_row(
        "Overall",
        str(results["overall"]["count"]),
        f"{results['overall']['exact_match_rate']:.2f}%",
        f"{results['overall']['average_edit_similarity']:.2f}%",
        style="bold"
    )
    
    print(table)

def humaneval_fim(custom_args: Optional[argparse.Namespace] = None) -> None:
    """
    Args:
        custom_args: Optional pre-configured arguments
    """
    if custom_args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name_or_path", type=str, required=True)
        parser.add_argument(
            "--model_type",
            type=str,
            default="codelm",
            choices=["codelm", "codelm_cfc", "codelm_leftright_context", 'codelm_right_cfc_left']
        )
        parser.add_argument("--gen_length", type=int, default=256)
        parser.add_argument("--max_seq_length", type=int, default=2048)
        parser.add_argument("--right_context_length", type=int, default=512)
        parser.add_argument("--output_dir", type=str, default="output_dir")
        parser.add_argument('--input_file', type=str, default="benchmarks/fim-bench/hm_fim/data/fim_singleline.jsonl",
                      help='Input JSONL file path')
        parser.add_argument("--tp", type=int, default=8)
        parser.add_argument("--template_type", type=str, default="qwen")
        parser.add_argument("--instruct_mode", action="store_true")
        args = parser.parse_args()
    else:
        args = custom_args

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # Set sampling parameters
    sampling_params = SamplingParams(temperature=0, max_tokens=args.gen_length)

    # Initialize LLM
    llm = LLM(model=args.model_name_or_path, tensor_parallel_size=args.tp, trust_remote_code=True)

    # Process input file
    results_file = os.path.join(args.output_dir, "ans.jsonl")
    prompts = []
    datas = []
    with open(args.input_file, "r", encoding='utf-8') as test_file:
        for line in tqdm(test_file.readlines()):
            data = json.loads(line)
            datas.append(data)
            # print(data)
    with open(results_file, "w") as ret_file:
        for data in tqdm(datas):
            prefix = data["prompt"]
            suffix = data["suffix"]

            prompt = prepare_prompt(tokenizer, "line_completion", args.model_type, args.model_name_or_path, prefix, suffix, args=args)
            data["input"] = prompt
            prompts.append(prompt)
        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
        for output, data in zip(outputs, datas):
            output_text = output.outputs[0].text
            if args.instruct_mode:
                extracted_text = extract_block(output_text)
            else:
                extracted_text = output_text
            if extracted_text == "":
                pass
            else:
                output_text = extracted_text
            data["model_gen"] = output_text
            # print(output_text)
            ret_file.write(json.dumps(data) + "\n")

    # Evaluate results
    results = evaluate_results(results_file)
    
    # Print results table
    print_results_table(results)
    
    # Save results to JSON file
    results_json_path = os.path.join(args.output_dir, "results.json")
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nDetailed results saved to: {results_json_path}")


    return results

# def extract_block(text: str) -> str:
#     """提取文本中第一个代码块的内容,处理可能存在的语言标识。
    
#     Args:
#         text: 包含代码块的文本字符串
        
#     Returns:
#         str: 提取出的代码块内容(不含语言标识)
        
#     Example:
#         "```python\ndef foo():\n    pass\n```" -> "def foo():\n    pass"
#         "```\nsome code\n```" -> "some code"
#     """
#     start = text.find('```') + 3
#     end = text.find('```', start)
    
#     if start == -1 or end == -1:
#         return ""
        
#     # 获取第一行到第一个换行符之间的内容
#     content = text[start:end]
#     first_newline = content.find('\n')
    
#     # 如果没找到换行符,返回空字符串
#     if first_newline == -1:
#         return ""
    
#     # 检查第一行是否是语言标识(不含空格的纯文本)
#     first_line = content[:first_newline].strip()
#     if first_line and not first_line.isspace():
#         # 如果第一行是语言标识,从第一个换行符后开始取内容
#         content = content[first_newline + 1:]
    
#     return content

def extract_block(text: str) -> str:
    """提取文本中第一个代码块的内容,处理可能存在的语言标识。
    如果没有完整的代码块但有开头部分,则去掉开头符号并返回剩余内容。
    
    Args:
        text: 包含代码块的文本字符串
        
    Returns:
        str: 提取出的代码块内容(不含语言标识)
        
    Example:
        "```python\ndef foo():\n    pass\n```" -> "def foo():\n    pass"
        "```\nsome code\n```" -> "some code"
        "```python\nsome code" -> "some code"
    """
    start = text.find('```') + 3
    end = text.find('```', start)
    
    if start == 2:  # 没找到开始标记
        return ""
        
    if end == -1:  # 找到开始标记但没有结束标记
        content = text[start:]
    else:
        content = text[start:end]
    
    first_newline = content.find('\n')
    
    # 如果没找到换行符,返回空字符串
    if first_newline == -1:
        return ""
    
    # 检查第一行是否是语言标识(不含空格的纯文本)
    first_line = content[:first_newline].strip()
    if first_line and not first_line.isspace():
        # 如果第一行是语言标识,从第一个换行符后开始取内容
        content = content[first_newline + 1:]
    content = content.rstrip("\n")
    return content

if __name__ == "__main__":
    humaneval_fim()