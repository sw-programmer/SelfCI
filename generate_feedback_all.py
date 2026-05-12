import argparse
import json
import os
import gc
import re
import torch
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams

from loguru import logger
from pathlib import Path
import sys
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.hf_hub import build_hf_repo_id, resolve_hf_token

ENV_PATH = REPO_ROOT / ".env"
if load_dotenv(ENV_PATH):
    logger.info("Dotenv loaded from {}", ENV_PATH)
else:
    logger.warning("Dotenv not found at {}", ENV_PATH)

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser(description="Generate CI feedback for keywords.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--dataset_path", type=str, default="huseyinatahaninan/ContextualIntegritySyntheticDataset")
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token for uploading/downloading. Defaults to HF_TOKEN from .env if omitted.",
    )
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens for generation")
    parser.add_argument("--target_keywords", type=str, choices=["disallowed", "allowed", "all"], default="all", help="Target keywords to generate feedback for: 'disallowed', 'allowed', or 'all'")
    return parser.parse_args()

def construct_prompt(user_task, recipient, keyword, keyword_value, rubric, context_info, is_allowed):
    """
    Constructs a prompt that forces the model to explain why a keyword is appropriate or inappropriate
    based on the specific CI rubric, including the actual content of the information.
    """
    
    # Rubric definitions based on Section 3.2 of the paper
    rubric_definitions = {
        "Confidentiality": "Information unrelated to the context (task, sender, recipient relationship) should not be shared.",
        "Proportionality": "Shared information should be proportionate to the task and not excessive.",
        "Consent": "Information sharing depends on the awareness and consent of the data subject."
    }
    
    definition = rubric_definitions.get(rubric, "Information should only be shared if appropriate for the context.")

    if is_allowed:
        status = "APPROPRIATE (Allowed)"
        goal = f"Provide feedback explaining strictly why sharing '{keyword}' is appropriate and aligns with Contextual Integrity in this specific context."
    else:
        status = "INAPPROPRIATE (Disallowed)"
        goal = f"Provide feedback explaining strictly why sharing '{keyword}' is a violation of Contextual Integrity in this specific context."

    prompt_content = f"""You are a Contextual Integrity (CI) expert.
    
Context:
- User Task: "{user_task}"
- Recipient: "{recipient}"
- Data Subject: "{context_info.get('data_subject', 'User')}"

The user is attempting to share the following information which has been flagged as {status}:
- Information Type: "{keyword}"
- Specific Content: "{keyword_value}"

Your Goal:
{goal}
You must base your explanation on the principle of "{rubric}".

Rubric Definition ({rubric}):
{definition}

Generate a concise feedback statement addressing the user:"""
    return prompt_content

def process_keywords(keywords_data, is_allowed, information_content, user_task, recipient, rubric, data_subject, idx):
    """Helper function to process either allowed or disallowed keywords."""
    inputs = []
    
    # Check if keywords_data is a dict (from JSON) or list
    if isinstance(keywords_data, dict):
        keywords = keywords_data.keys()
    elif isinstance(keywords_data, list):
        keywords = keywords_data
    else:
        keywords = []

    for keyword in keywords:
        keyword_value = information_content.get(keyword, "N/A")

        prompt_text = construct_prompt(
            user_task=user_task,
            recipient=recipient,
            keyword=keyword,
            keyword_value=keyword_value,
            rubric=rubric,
            context_info={'data_subject': data_subject},
            is_allowed=is_allowed
        )
        
        messages = [{"role": "user", "content": prompt_text}]
        
        inputs.append({
            "row_idx": idx,
            "keyword": keyword,
            "is_allowed": is_allowed,
            "messages": messages
        })
    return inputs

def strip_deepseek_thinking(model_path: str, generated_text: str) -> str:
    """For DeepSeek reasoning models, keep only the final response after </think>."""
    if "instruct" in model_path.lower():
        return generated_text

    parts = re.split(r"</think\s*>", generated_text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) == 2:
        return parts[1].lstrip()
    return generated_text

def main():
    args = parse_args()
    hf_token = resolve_hf_token(args.hf_token)

    # 1. Load Dataset
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_dataset(args.dataset_path, split="train", token=hf_token)
    
    # 2. Prepare Prompts
    print("Preprocessing dataset and constructing prompts...")
    
    flattened_inputs = []
    
    for idx, row in enumerate(dataset):
        try:
            # Parse JSON columns
            seed_data = json.loads(row['seed'])
            item_data = json.loads(row['dataset_item'])
            
            # Extract Contextual Info
            rubric = seed_data.get('transmission_principle', 'Confidentiality')
            recipient = seed_data.get('recipient', 'unknown recipient')
            data_subject = seed_data.get('data_subject', 'user')
            
            user_task = item_data.get('user_task', '')
            
            annotation = item_data.get('annotation', {})
            information_content = item_data.get('information', {})
            
            disallowed_data = annotation.get('disallowed', {})
            allowed_data = annotation.get('allowed', {})

            if args.target_keywords in ['disallowed', 'all']:
                flattened_inputs.extend(process_keywords(
                    disallowed_data, False, information_content, user_task, recipient, rubric, data_subject, idx
                ))

            if args.target_keywords in ['allowed', 'all']:
                flattened_inputs.extend(process_keywords(
                    allowed_data, True, information_content, user_task, recipient, rubric, data_subject, idx
                ))
                
        except json.JSONDecodeError:
            print(f"Skipping row {idx} due to JSON parse error.")
            continue

    print(f"Generated {len(flattened_inputs)} prompts for feedback generation.")
    
    if len(flattened_inputs) > 0:
        print("Example Prompt:")
        print(flattened_inputs[0])

    # 3. Load Model (vllm)
    print(f"Loading model: {args.model_path}")
    num_gpus = torch.cuda.device_count()
    llm = LLM(args.model_path, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.9)
    
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=args.max_tokens
    )

    # 4. Generate
    print("Generating feedbacks...")
    messages_batch = [item['messages'] for item in flattened_inputs]
    outputs = llm.chat(messages_batch, sampling_params)

    # 5. Post-process and Merge
    print("Merging results back to dataset...")
    
    feedback_map = {}
    truncated_count = 0
    
    for i, output in enumerate(outputs):
        row_idx = flattened_inputs[i]['row_idx']
        keyword = flattened_inputs[i]['keyword']
        is_allowed = flattened_inputs[i]['is_allowed']
        
        # --- QUALITY CONTROL START ---
        finish_reason = output.outputs[0].finish_reason
        
        if finish_reason == "length":
            status_str = "Allowed" if is_allowed else "Disallowed"
            print(f"[QC Warning] Dropping feedback for Row {row_idx} ({status_str} Keyword: '{keyword}') due to length truncation.")
            truncated_count += 1
            continue

        generated_text = strip_deepseek_thinking(args.model_path, output.outputs[0].text)
        
        if row_idx not in feedback_map:
            feedback_map[row_idx] = {"allowed_feedbacks": {}, "disallowed_feedbacks": {}}
        
        if is_allowed:
            feedback_map[row_idx]["allowed_feedbacks"][keyword] = generated_text
        else:
            feedback_map[row_idx]["disallowed_feedbacks"][keyword] = generated_text

    print(f"Quality Control Summary: {truncated_count} out of {len(outputs)} generated feedbacks were dropped.")

    # 6. Add new columns to dataset
    def add_feedback_columns(example, idx):
        if idx in feedback_map:
            return {
                "allowed_feedbacks": json.dumps(feedback_map[idx]["allowed_feedbacks"], ensure_ascii=False),
                "disallowed_feedbacks": json.dumps(feedback_map[idx]["disallowed_feedbacks"], ensure_ascii=False)
            }
        else:
            return {
                "allowed_feedbacks": "{}",
                "disallowed_feedbacks": "{}"
            }

    new_dataset = dataset.map(add_feedback_columns, with_indices=True)

    # Clean up
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # 7. Upload/Save
    model_name_clean = args.model_path.split("/")[-1]
    try:
        new_dataset_name = build_hf_repo_id(
            f"ContextualIntegritySyntheticDataset_{model_name_clean}_{args.target_keywords}",
            hf_token,
        )
        print(f"Pushing new dataset to Hub: {new_dataset_name}")
        new_dataset.push_to_hub(new_dataset_name, token=hf_token)
        print("Upload successful!")
    except Exception as e:
        print(f"Upload failed: {e}")
        print(f"Saving locally to ./output_{model_name_clean}_{args.target_keywords}")
        new_dataset.save_to_disk(f"./output_{model_name_clean}_{args.target_keywords}")

if __name__ == "__main__":
    main()
