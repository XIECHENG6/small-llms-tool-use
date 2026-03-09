"""
Data utilities for parsing and formatting the Glaive Function Calling v2 dataset.
"""

import json
import re
import random
from datasets import load_dataset, Dataset


def parse_glaive_sample(sample):
    """Parse a single sample from Glaive Function Calling v2 dataset.
    
    Extracts the user query, system prompt, and function call from the raw
    conversation format. Handles various JSON formatting edge cases.
    
    Args:
        sample: Dict with 'system' and 'chat' fields from the dataset.
        
    Returns:
        Dict with parsed fields, or None if parsing fails.
    """
    system_prompt = sample.get("system", "")
    chat = sample.get("chat", "")

    if "<functioncall>" not in chat:
        return None

    parts = chat.split("ASSISTANT:")
    if len(parts) < 2:
        return None

    user_part = parts[0]
    user_match = re.search(r'USER:\s*(.*)', user_part, re.DOTALL)
    if not user_match:
        return None
    user_query = user_match.group(1).strip()

    fc_text = None
    for part in parts[1:]:
        if "<functioncall>" in part:
            fc_text = part
            break
    if fc_text is None:
        return None

    fc_match = re.search(r'<functioncall>\s*(.*?)\s*<\|endoftext\|>', fc_text, re.DOTALL)
    if not fc_match:
        fc_match = re.search(r'<functioncall>\s*(.*)', fc_text, re.DOTALL)
    if not fc_match:
        return None

    raw_fc = fc_match.group(1).strip()

    # Try multiple JSON parsing strategies
    fc_json = None
    try:
        fc_json = json.loads(raw_fc)
    except json.JSONDecodeError:
        pass

    if fc_json is None:
        try:
            fixed = re.sub(r"'(\{.*?\})'", r'\1', raw_fc)
            fc_json = json.loads(fixed)
        except Exception:
            pass

    if fc_json is None:
        try:
            fixed = raw_fc.replace("'", '"')
            fc_json = json.loads(fixed)
        except Exception:
            return None

    if not isinstance(fc_json, dict) or "name" not in fc_json:
        return None

    args = fc_json.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    if args is None:
        args = {}

    clean_fc = json.dumps({"name": fc_json["name"], "arguments": args}, ensure_ascii=False)

    return {
        "system": system_prompt.strip(),
        "user": user_query,
        "function_call": clean_fc,
        "function_name": fc_json["name"],
    }


def load_and_parse_dataset(dataset_name="glaiveai/glaive-function-calling-v2"):
    """Load and parse the Glaive Function Calling v2 dataset.
    
    Returns:
        List of parsed samples with valid function calls.
    """
    print(f"Loading {dataset_name}...")
    raw_dataset = load_dataset(dataset_name, split="train")
    print(f"Total raw samples: {len(raw_dataset)}")

    parsed_data = []
    failed = 0
    for sample in raw_dataset:
        result = parse_glaive_sample(sample)
        if result:
            parsed_data.append(result)
        else:
            failed += 1

    print(f"Successfully parsed: {len(parsed_data)}")
    print(f"Skipped: {failed}")
    return parsed_data


def format_training_text(sample, eos_token=""):
    """Format a parsed sample into training text with EOS token.
    
    The EOS token after the function call teaches the model to stop
    generating after producing the JSON output, preventing the
    "repetition generation" problem.
    """
    text = f"""### System:
{sample['system']}

When you need to call a function, respond ONLY with a JSON object in this exact format:
{{"name": "function_name", "arguments": {{"arg1": "value1"}}}}
Do not include any other text before or after the JSON.

### User:
{sample['user']}

### Assistant:
{sample['function_call']}{eos_token}"""
    return {"text": text}


def prepare_datasets(parsed_data, tokenizer, max_train=10000, max_test=500, seed=42):
    """Prepare train/test splits from parsed data.
    
    Args:
        parsed_data: List of parsed samples.
        tokenizer: Tokenizer (used for EOS token).
        max_train: Number of training samples.
        max_test: Number of test samples.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_dataset, test_dataset, test_samples_raw).
    """
    random.seed(seed)
    
    formatted = [format_training_text(s, eos_token=tokenizer.eos_token) for s in parsed_data]
    random.shuffle(formatted)

    train_data = formatted[:max_train]
    test_data = formatted[max_train:max_train + max_test]
    test_samples_raw = parsed_data[max_train:max_train + max_test]

    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    return train_dataset, test_dataset, test_samples_raw


def tokenize_dataset(dataset, tokenizer, max_length=512):
    """Tokenize and truncate a dataset for training."""
    tokenized = []
    for sample in dataset:
        encoded = tokenizer(
            sample["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        encoded["labels"] = encoded["input_ids"].copy()
        tokenized.append(encoded)
    return Dataset.from_list(tokenized)
