"""
Interactive inference demo for function calling.

Usage:
    python demo/inference.py --model_id Qwen/Qwen2.5-3B-Instruct --adapter_path ./output/qwen25_3b
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


EXAMPLE_FUNCTIONS = """Available functions:
1. get_weather(city: str, unit: str = "celsius") - Get current weather
2. search_restaurant(location: str, cuisine: str, price_range: str) - Search restaurants
3. calculate_tip(bill_amount: float, tip_percentage: float) - Calculate tip
4. send_email(to: str, subject: str, body: str) - Send an email
5. get_stock_price(symbol: str) - Get current stock price
6. translate_text(text: str, source_lang: str, target_lang: str) - Translate text
7. set_alarm(time: str, label: str) - Set an alarm
8. calculate_bmi(weight: float, height: float) - Calculate BMI"""


def load_model(model_id, adapter_path=None):
    """Load model with optional LoRA adapter."""
    use_bf16 = torch.cuda.get_device_capability(0)[0] >= 8 if torch.cuda.is_available() else False

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"Loaded LoRA adapter from {adapter_path}")
    
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, system_prompt, user_query):
    """Generate a function call."""
    prompt = f"""### System:
{system_prompt}

When you need to call a function, respond ONLY with a JSON object in this exact format:
{{"name": "function_name", "arguments": {{"arg1": "value1"}}}}
Do not include any other text before or after the JSON.

### User:
{user_query}

### Assistant:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    eos_ids = [tokenizer.eos_token_id]
    for token in ["<|im_end|>", "<|endoftext|>"]:
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid != tokenizer.unk_token_id and tid not in eos_ids:
            eos_ids.append(tid)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
            use_cache=False,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    
    # Extract first JSON object
    try:
        return json.loads(text), text
    except json.JSONDecodeError:
        depth = 0
        start = None
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0: start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        return json.loads(text[start:i+1]), text
                    except:
                        start = None
    return None, text


def main(args):
    print(f"Loading model: {args.model_id}")
    if args.adapter_path:
        print(f"With adapter: {args.adapter_path}")
    model, tokenizer = load_model(args.model_id, args.adapter_path)

    print("\n" + "=" * 60)
    print("Function Calling Demo")
    print("=" * 60)
    print(f"\n{EXAMPLE_FUNCTIONS}\n")
    print("Type a request (or 'quit' to exit):")
    print("-" * 60)

    system_prompt = f"You are a helpful assistant with access to the following functions. Use them if required.\n{EXAMPLE_FUNCTIONS}"

    while True:
        user_input = input("\n🧑 You: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        if not user_input:
            continue

        result, raw = generate(model, tokenizer, system_prompt, user_input)
        
        if result:
            print(f"\n🤖 Function Call:")
            print(f"   Name: {result.get('name', '?')}")
            args_dict = result.get('arguments', {})
            if args_dict:
                print(f"   Arguments:")
                for k, v in args_dict.items():
                    print(f"     {k}: {v}")
        else:
            print(f"\n🤖 Raw output: {raw[:200]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
