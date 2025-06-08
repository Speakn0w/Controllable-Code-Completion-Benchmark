#!/usr/bin/env python
# coding=utf-8

import argparse
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vllm import LLM, SamplingParams
from typing import List, Dict, Optional
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
from collections import defaultdict
from rich import print
from rich.table import Table
from fuzzywuzzy import fuzz
import numpy as np
import contextlib
from contextlib import redirect_stdout
import signal
import time
import traceback
import asyncio
import aiohttp
import requests
import tempfile
import io
import re
import gc
from cc_clients import ConcurrentOpenAIClient

SYSTEM_PROMPT = """
You are a code completion assistant. Your task is to generate appropriate middle code that connects the given prefix and suffix code segments and follows the specific instructions provided.

Format:
- Code in Markdown block with language tag
"""

def cleanup_gpu():
    torch.cuda.empty_cache()
    gc.collect()

@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)

@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise "Timed out!"
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield

@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname

class redirect_stdin(contextlib._RedirectStream):
    _stream = "stdin"

class WriteOnlyStringIO(io.StringIO):
    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        return False

def collate_fn_generate(sample: Dict, **kwargs) -> List[Dict]:
    return sample["messages"]

def collate_fn_eval(sample: Dict, **kwargs) -> List[Dict]:
    prefix = sample["prefix"]
    instruction = sample["system"]
    middle = sample["model_gen"]
    middle_ground_truth = sample["middle"]
    suffix = sample["suffix"]
    system_prompt = """As a code evaluator, assess whether the provided implementation follows the instruction and matches the implementation approach of the ground truth.

Focus on:
1) Instruction adherence: Does the implementation use the specified method/approach?
2) Ground truth alignment: Does it follow similar implementation strategy?
3) Basic structure: Are the fundamental building blocks present?

Note: The evaluation should focus on whether the correct approaches/methods are attempted, even if the specific implementation contains logical errors or bugs.

Provide your evaluation in the following format:
[JUDGMENT]yes/no[/JUDGMENT]
[REASON]Brief explanation of your judgment (1-2 sentences)[/REASON]

Where:
- "yes": Implementation attempts to follow the instruction and use the ground truth approache
- "no": Implementation uses fundamentally different methods or approaches"""

    user_prompt = f"""[PREFIX CODE - For Context]
{prefix}

[INSTRUCTION TO EVALUATE AGAINST]
{instruction}

[MIDDLE CODE TO EVALUATE]
{middle}

[GROUND TRUTH IMPLEMENTATION]
{middle_ground_truth}

[SUFFIX CODE - For Context] 
{suffix}

Evaluate if the middle code follows the instruction and matches the ground truth approach. Provide both judgment and brief reason."""
    messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
    return messages

class CustomExit(Exception):
    def __init__(self, code=0):
        self.code = code
        super().__init__(f"Program exit with code {code}")

def raise_exit(code=0):
    raise CustomExit(code)

def raise_quit():
    return raise_exit(0)

def evaluate_results(results_file, output_path):
    language_results = defaultdict(lambda: {
        "count": 0,
        "exact_matches": 0,
        "pass": 0,
        "IF": 0,
        "pass_IF": 0,
        "follow_instruction": 0,
        "edit_similarities": []
    })
    
    cache_path = os.path.join(output_path, "eval_cache.jsonl")
    client_eval = ConcurrentOpenAIClient(
            cache_file=cache_path,
            collate_fn=collate_fn_eval,
            max_retries=None,
            base_url="YOUR_API_BASE_URL",
            api_key="YOUR_API_KEY",
        )
    
    temp_file = results_file.replace("ans.jsonl", "new_ans.jsonl")
    with open(results_file, "r", encoding="utf-8") as f, open(temp_file, "w", encoding="utf-8") as fw:
        all_data = []
        num = 1
        for line in f:
            print(f"num:{num}")
            num +=1
            try:
                data = json.loads(line)
                model_gen = data.get("model_gen")
                canonical_solution = data["middle"]
                prefix = data["prefix"]
                suffix = data["suffix"]
                instruction = data["system"]
                language = data.get("language", "Unknown")
                code = re.sub(r"'''.*?'''", '', prefix, flags=re.DOTALL) + "\n" + model_gen + "\n" + suffix
                is_pass = True

                output_io = io.StringIO()

                unit_tests = data['unit_tests']
                try:
                    import ast
                    unit_tests = ast.literal_eval(unit_tests)
                    for unit_test in unit_tests:
                        exec_globals = {
                        'exit': raise_exit,
                        'quit': raise_quit}
                        unit_input = unit_test['input']
                        data["exec_code"] = code
                        with time_limit(15):
                            input_io = io.StringIO(unit_input)
                            output_io = io.StringIO()
                                    
                            with redirect_stdin(input_io), redirect_stdout(output_io):
                                exec(code, exec_globals)
                            exec_output = output_io.getvalue()
                            output_lines = exec_output.strip()
                    unit_output = unit_test['output']
                    expected_output_lines = unit_output
                            
                    if output_lines != expected_output_lines[0]:
                        is_pass = False
                                
                except Exception as e:
                    is_pass = False
                    print("wrong alt")
                    print(f"Exception type: {type(e)}")
                    print(f"Exception message: {str(e)}")
                    print("Traceback:")
                    traceback.print_exc()
                    
                if is_pass:
                    language_results[language]["pass"] += 1

                data["is_pass"] = is_pass
                
                language_results[language]["count"] += 1
                if model_gen == canonical_solution:
                    language_results[language]["exact_matches"] += 1
                    data["EM"] = 1
                else:
                    data["EM"] = 0
                
                similarity = fuzz.ratio(model_gen, canonical_solution)
                language_results[language]["edit_similarities"].append(similarity)

                data["ES"] = similarity
                all_data.append(data)

            except Exception as e:
                print(f"Failed to process line: {line}")
                print(f"Error: {e}")
                continue

        try:
            client_eval.generate(
                samples=all_data,
                batch_size=1024,
                model="YOUR_MODEL_NAME",
                max_tokens=1024,
                temperature=0.0
            )
        except Exception as e:
            print(f"All attempts failed: {e}")
        responses = client_eval.collect_responses(all_data)
        for data, response in zip(all_data, responses):
            output = response.choices[0].message.content

            output_start = output.find("[JUDGMENT]")
            output_end = output.find("[/JUDGMENT]")

            reason_start = output.find("[REASON]")
            reason_end = output.find("[/REASON]")

            reason = output[reason_start + len("[REASON]"): reason_end].strip()

            output = output[output_start + len("[JUDGMENT]"):output_end].strip()
            if output.lower() == "yes":
                judge = True
            elif output.lower() == "no":
                judge = False
            
            data["IF"] = judge
            data["reason"] = reason
            fw.write(json.dumps(data) + "\n")
            if judge:
                language_results[language]["IF"] += 1
            data["pass_IF"] = data["is_pass"] and data["IF"]
            if data["pass_IF"]:
                language_results[language]["pass_IF"] += 1

    final_results = {}
    total_count = 0
    total_exact_matches = 0
    total_pass = 0
    if_rate = 0
    total_pass_if_rate = 0
    all_similarities = []

    for lang, stats in language_results.items():
        exact_match_rate = (stats["exact_matches"] / stats["count"]) * 100
        avg_similarity = np.mean(stats["edit_similarities"])
        pass_rate = (stats["pass"] / stats["count"]) * 100
        instruction_follow_rate = (stats["IF"] / stats["count"]) * 100
        pass_if_rate = (stats["pass_IF"] / stats["count"]) * 100
        
        final_results[lang] = {
            "count": stats["count"],
            "exact_match_rate": exact_match_rate,
            "pass_rate": pass_rate,
            "instruction_follow_rate": instruction_follow_rate,
            "average_edit_similarity": avg_similarity,
            "pass_if_rate": pass_if_rate
        }
        
        total_count += stats["count"]
        total_exact_matches += stats["exact_matches"]
        total_pass += stats["pass"]
        if_rate += stats["IF"]
        total_pass_if_rate += stats["pass_IF"]
        all_similarities.extend(stats["edit_similarities"])
    
    final_results["overall"] = {
        "count": total_count,
        "exact_match_rate": (total_exact_matches / total_count) * 100,
        "pass_rate": (total_pass / total_count) * 100,
        "instruction_follow_rate": (if_rate / total_count) * 100,
        "pass_if_rate": (total_pass_if_rate / total_count) * 100,
        "average_edit_similarity": np.mean(all_similarities)
    }
    
    return final_results

def print_results_table(results):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Language")
    table.add_column("Count")
    table.add_column("Exact Match Rate")
    table.add_column("Pass Rate")
    table.add_column("Instruction Follow Rate")
    table.add_column("Pass IF Rate")
    table.add_column("Edit Similarity")
    
    for lang, metrics in results.items():
        if lang != "overall":
            table.add_row(
                lang,
                str(metrics["count"]),
                f"{metrics['exact_match_rate']:.2f}%",
                f"{metrics['pass_rate']}",
                f"{metrics['instruction_follow_rate']:.2f}%",
                f"{metrics['pass_if_rate']:.2f}%",
                f"{metrics['average_edit_similarity']:.2f}%"
            )
    
    table.add_row(
        "Overall",
        str(results["overall"]["count"]),
        f"{results['overall']['exact_match_rate']:.2f}%",
        f"{results['overall']['pass_rate']:.2f}",
        f"{results['overall']['instruction_follow_rate']:.2f}",
        f"{results['overall']['pass_if_rate']:.2f}%",
        f"{results['overall']['average_edit_similarity']:.2f}%",
        style="bold"
    )
    
    print(table)

def extract_block(text: str) -> str:
    start = text.find('```')
    end = text.find('```', start+3)
    
    if start == -1 or end == -1:
        return text
    
    start = start + 3
    content = text[start:end]
    first_newline = content.find('\n')
    
    if first_newline == -1:
        return ""
    
    first_line = content[:first_newline].strip()
    if first_line and not first_line.isspace():
        content = content[first_newline + 1:]
    content = content.rstrip("\n")
    return content

def humaneval_fim(custom_args: Optional[argparse.Namespace] = None) -> None:
    if custom_args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name_or_path", type=str, required=True)
        parser.add_argument(
            "--model_type",
            type=str,
            default="codelm",
            choices=["codelm", "codelm_cfc", "codelm_leftright_context", 'codelm_right_cfc_left']
        )
        parser.add_argument("--gen_length", type=int, default=1024)
        parser.add_argument("--max_seq_length", type=int, default=2048)
        parser.add_argument("--right_context_length", type=int, default=512)
        parser.add_argument("--output_dir", type=str, default="output_dir")
        parser.add_argument('--input_file', type=str, default="PATH_TO_DATASET",
                      help='Input JSONL file path')
        parser.add_argument("--tp", type=int, default=8)
        parser.add_argument("--template_type", type=str, default="qwen")
        parser.add_argument("--instruct_mode", action="store_true")
        parser.add_argument("--api", action="store_true")
        args = parser.parse_args()
    else:
        args = custom_args

    os.makedirs(args.output_dir, exist_ok=True)
    if not args.api:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        sampling_params = SamplingParams(temperature=0, max_tokens=args.gen_length)
        llm = LLM(model=args.model_name_or_path, tensor_parallel_size=args.tp, trust_remote_code=True)

    results_file = os.path.join(args.output_dir, "ans.jsonl")
    prompts = []
    datas = []
    with open(args.input_file, "r") as test_file:
        for line in tqdm(test_file.readlines()):
            data = json.loads(line)
            datas.append(data)

    with open(results_file, "w") as ret_file:
        for data in tqdm(datas):
            prefix = data["prefix"]
            suffix = data["suffix"]
            instruction = data["system"]

            prefix_code_prompt = f"The prefix code is:\n"
            prefix_code_prompt += "```python\n"
            prefix_code_prompt += prefix
            prefix_code_prompt += "```\n\n"

            suffix_code_prompt = f"The suffix code in is:\n"
            suffix_code_prompt += "```python\n"
            suffix_code_prompt += suffix
            suffix_code_prompt += "```\n\n"

            instructions_prompt = "Instructions:\n"
            instructions_prompt += instruction

            user_prompt = instructions_prompt + "\n\n" + prefix_code_prompt + suffix_code_prompt

            if args.api:
                message = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
                
                data["input"] = message
                prompts.append({"messages": message})
            else:
                if "starcoder" in args.model_name_or_path or "gemma" in args.model_name_or_path.lower():
                    dialog_dict = [
                    {"role": "assistant", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
                else:
                    dialog_dict = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ]
                prompt = tokenizer.apply_chat_template(dialog_dict, add_generation_prompt=True, tokenize=False)
                data["input"] = prompt
                prompts.append(prompt)
            
        outputs = []
        if args.api:
            cache_path = os.path.join(args.output_dir, "gen_cache.jsonl")
            client_gen = ConcurrentOpenAIClient(
                cache_file=cache_path,
                collate_fn=collate_fn_generate,
                max_retries=None,
                base_url="YOUR_API_BASE_URL",
                api_key="YOUR_API_KEY",
            )
            
            try:
                client_gen.generate(
                samples=prompts,
                batch_size=1024,
                model=args.model_name_or_path,
                max_tokens=1024,
                temperature=0.0,
            )
            except Exception as e:  
                print(f"All attempts failed: {e}")
            
            responses = client_gen.collect_responses(prompts)
            for response in responses:
                outputs.append(response.choices[0].message.content)

        else:
            outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
     
        for output, data in zip(outputs, datas):
            if args.api:
                output_text = output
            else:
                output_text = output.outputs[0].text
            data["origin_gen"] = output_text
            if args.instruct_mode:
                output_text = extract_block(output_text)
            data["model_gen"] = output_text
            ret_file.write(json.dumps(data) + "\n")

    results = evaluate_results(results_file, args.output_dir)
    print_results_table(results)
    
    results_json_path = os.path.join(args.output_dir, "results.json")
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nDetailed results saved to: {results_json_path}")
    cleanup_gpu()
    torch.cuda.synchronize()

    return results

if __name__ == "__main__":
    humaneval_fim() 