"""
Evaluation script for function calling models.

Metrics:
- JSON Valid Rate: Can the model produce valid JSON?
- Function Name Accuracy: Is the correct function selected?
- Argument Names Accuracy: Are all argument keys correct?
- Argument Values Accuracy: Are all argument values correct?
- Exact Match Rate: Is the entire function call perfectly correct?

Usage:
    python src/evaluate.py --model_id Qwen/Qwen2.5-3B-Instruct --adapter_path ./output/qwen25_3b
"""

import argparse
import json
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from data_utils import load_and_parse_dataset


def extract_json_from_text(text):
    """Extract the first complete JSON object from text.
    
    Handles nested braces correctly. This is critical for accurate
    evaluation — a naive json.loads() fails when the model generates
    multiple JSON objects or wraps JSON in additional text.
    """
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    start = None
    return None


def generate_function_call(model, tokenizer, system_prompt, user_query, 
                           max_seq_length=512, max_new_tokens=256):
    """Generate a function call from the model."""
    prompt = f"""### System:
{system_prompt}

When you need to call a function, respond ONLY with a JSON object in this exact format:
{{"name": "function_name", "arguments": {{"arg1": "value1"}}}}
Do not include any other text before or after the JSON.

### User:
{user_query}

### Assistant:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, 
                       max_length=max_seq_length).to(model.device)

    # Build stop token ids
    eos_ids = [tokenizer.eos_token_id]
    for special_token in ["<|im_end|>", "<|endoftext|>"]:
        token_id = tokenizer.convert_tokens_to_ids(special_token)
        if token_id != tokenizer.unk_token_id and token_id not in eos_ids:
            eos_ids.append(token_id)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
            use_cache=False,  # Compatibility with Phi-3.5
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def evaluate_function_call(predicted_json, ground_truth_json):
    """Evaluate a single function call prediction against ground truth."""
    result = {
        "json_valid": predicted_json is not None,
        "name_correct": False,
        "args_name_correct": False,
        "args_value_correct": False,
        "exact_match": False,
    }
    if predicted_json is None:
        return result

    gt = ground_truth_json
    result["name_correct"] = (predicted_json.get("name", "") == gt.get("name", ""))

    pred_args = predicted_json.get("arguments", {})
    gt_args = gt.get("arguments", {})

    if isinstance(pred_args, str):
        try: pred_args = json.loads(pred_args)
        except: pred_args = {}
    if isinstance(gt_args, str):
        try: gt_args = json.loads(gt_args)
        except: gt_args = {}

    pred_keys = set(pred_args.keys()) if isinstance(pred_args, dict) else set()
    gt_keys = set(gt_args.keys()) if isinstance(gt_args, dict) else set()
    result["args_name_correct"] = (pred_keys == gt_keys)

    if result["args_name_correct"] and isinstance(pred_args, dict) and isinstance(gt_args, dict):
        all_match = True
        for key in gt_keys:
            if str(pred_args.get(key, "")).strip().lower() != str(gt_args.get(key, "")).strip().lower():
                all_match = False
                break
        result["args_value_correct"] = all_match

    result["exact_match"] = (result["name_correct"] and result["args_value_correct"])
    return result


def run_evaluation(model, tokenizer, test_samples, max_seq_length=512):
    """Run evaluation on test samples and return metrics."""
    results = []
    for i, sample in enumerate(test_samples):
        pred_text = generate_function_call(
            model, tokenizer, sample["system"], sample["user"],
            max_seq_length=max_seq_length,
        )
        pred_json = extract_json_from_text(pred_text)
        gt_json = json.loads(sample["function_call"])

        eval_result = evaluate_function_call(pred_json, gt_json)
        eval_result["predicted_text"] = pred_text
        eval_result["ground_truth"] = sample["function_call"]
        eval_result["function_name"] = sample["function_name"]
        results.append(eval_result)

        if (i + 1) % 50 == 0:
            acc = sum(r["exact_match"] for r in results) / len(results)
            print(f"  [{i+1}/{len(test_samples)}] Exact match: {acc:.1%}")

    n = len(results)
    metrics = {
        "JSON Valid Rate": sum(r["json_valid"] for r in results) / n,
        "Function Name Acc": sum(r["name_correct"] for r in results) / n,
        "Arg Names Acc": sum(r["args_name_correct"] for r in results) / n,
        "Arg Values Acc": sum(r["args_value_correct"] for r in results) / n,
        "Exact Match Rate": sum(r["exact_match"] for r in results) / n,
    }
    return metrics, results


def print_results(metrics, title="Results"):
    """Pretty-print evaluation metrics."""
    print("=" * 55)
    print(title)
    print("=" * 55)
    for metric, value in metrics.items():
        bar = "█" * int(value * 30) + "░" * (30 - int(value * 30))
        print(f"  {metric:<20} {bar} {value:.1%}")
    print("=" * 55)


def print_comparison(base_metrics, ft_metrics, model_id):
    """Print zero-shot vs fine-tuned comparison table."""
    print("=" * 65)
    print(f"COMPARISON: Zero-Shot vs Fine-Tuned ({model_id})")
    print("=" * 65)
    print(f"  {'Metric':<20} {'Zero-Shot':>12} {'Fine-Tuned':>12} {'Improve':>12}")
    print("-" * 65)
    for metric in ft_metrics:
        base_val = base_metrics[metric]
        ft_val = ft_metrics[metric]
        delta = ft_val - base_val
        arrow = "+" if delta > 0 else ""
        print(f"  {metric:<20} {base_val:>11.1%} {ft_val:>11.1%} {arrow}{delta:>10.1%}")
    print("=" * 65)


def main(args):
    use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8 if torch.cuda.is_available() else False

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    parsed_data = load_and_parse_dataset()
    test_samples = parsed_data[args.train_samples:args.train_samples + args.test_samples]

    if args.adapter_path:
        # Evaluate fine-tuned model
        print(f"\nLoading fine-tuned model from {args.adapter_path}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_id, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        )
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
        model.eval()

        print(f"Evaluating fine-tuned model on {len(test_samples)} samples...")
        metrics, results = run_evaluation(model, tokenizer, test_samples)
        print_results(metrics, f"Fine-Tuned {args.model_id}")
    else:
        # Evaluate base model (zero-shot)
        print(f"\nLoading base model {args.model_id}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        )
        print(f"Evaluating zero-shot on {len(test_samples)} samples...")
        metrics, results = run_evaluation(model, tokenizer, test_samples)
        print_results(metrics, f"Zero-Shot {args.model_id}")

    # Save results
    if args.save_path:
        with open(args.save_path, "w") as f:
            json.dump({"metrics": metrics, "detailed_results": results[:20]}, f, indent=2)
        print(f"Results saved to: {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--train_samples", type=int, default=10000)
    parser.add_argument("--test_samples", type=int, default=500)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
