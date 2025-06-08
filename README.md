## Requirements

```bash
pip install vllm transformers tqdm torch rich fuzzywuzzy numpy aiohttp requests
```

## Dataset

The dataset should be downloaded from HuggingFace: [DATASET_LINK_HERE]

The dataset should be in JSONL format with each line containing:
- prefix: Code before the target completion area
- suffix: Code after the target completion area
- system: Instructions for the completion
- middle: Ground truth implementation
- unit_tests: Test cases to verify the implementation
- language: Programming language of the code

## Usage

### Local Model Evaluation

To evaluate a local model:

```bash
python run_ccc_fim_eval.py \
    --model_name_or_path /path/to/your/model \
    --output_dir ./output \
    --input_file /path/to/dataset.jsonl
```

### API-based Evaluation

To evaluate using an API:

```bash
python run_ccc_fim_eval.py \
    --model_name_or_path YOUR_MODEL_NAME \
    --output_dir ./output \
    --input_file /path/to/dataset.jsonl \
    --api
```

Before running API-based evaluation, make sure to:
1. Set your API base URL in the script (replace "YOUR_API_BASE_URL")
2. Set your API key in the script (replace "YOUR_API_KEY")

## Arguments

- `--model_name_or_path`: Path to local model or model name for API
- `--model_type`: Model type (default: "codelm")
- `--gen_length`: Maximum generation length (default: 1024)
- `--max_seq_length`: Maximum sequence length (default: 2048)
- `--right_context_length`: Right context length (default: 512)
- `--output_dir`: Output directory (default: "output_dir")
- `--input_file`: Input JSONL file path
- `--tp`: Tensor parallel size (default: 8)
- `--template_type`: Template type (default: "qwen")
- `--instruct_mode`: Enable instruction mode
- `--api`: Use API instead of local model

## Output

The script generates:
1. `ans.jsonl`: Raw evaluation results
2. `new_ans.jsonl`: Processed evaluation results with metrics
3. `results.json`: Final evaluation metrics including:
   - Exact match rate
   - Pass rate
   - Instruction follow rate
   - Pass IF rate
   - Edit similarity

Results are also displayed in a formatted table during execution.

## Metrics Explanation

- **Exact Match Rate**: Percentage of generations matching ground truth exactly
- **Pass Rate**: Percentage of generations passing unit tests
- **Instruction Follow Rate**: Percentage of generations following instructions
- **Pass IF Rate**: Percentage of generations both passing tests and following instructions
- **Edit Similarity**: Average similarity between generations and ground truth 
