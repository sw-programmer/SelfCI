from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import Dataset, DatasetDict, load_from_disk
from dotenv import load_dotenv
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from distill.training_arg_defaults import (
    add_base_training_cli_args,
    add_distillation_cli_args,
    attach_distillation_training_args,
    build_hf_training_args,
)
from utils.onpolicy_trainer import MultiGPUOnPolicyTrainer
from utils.eval_metrics import build_compute_eval_metrics
from utils.preprocess import prepare_grpo_dataset_splits_from_hf

ENV_PATH = REPO_ROOT / ".env"
if load_dotenv(ENV_PATH):
    logger.info("Dotenv loaded from {}", ENV_PATH)
else:
    logger.warning("Dotenv not found at {}", ENV_PATH)

DATASET_NAME_OR_PATH = "huseyinatahaninan/ContextualIntegritySyntheticDataset"
DEFAULT_STUDENT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TEACHER_MODEL = "Qwen/Qwen2.5-32B-Instruct"
TRAIN_SPLIT = "train"
EVAL_SPLIT = "eval"
TEST_SPLIT = "test"
LORA_RANK = 32
LORA_DROPOUT = 0.05
LORA_ALPHA = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="On-policy distillation training entrypoint.")
    parser.add_argument(
        "--student-model-name-or-path",
        type=str,
        default=DEFAULT_STUDENT_MODEL,
    )
    parser.add_argument(
        "--teacher-model-name-or-path",
        type=str,
        default=DEFAULT_TEACHER_MODEL,
    )
    parser.add_argument(
        "--dataset-name-or-path",
        type=str,
        default=DATASET_NAME_OR_PATH,
    )
    parser.add_argument("--output-dir", type=str, default="output/onpolicy")
    parser.add_argument("--run-name", type=str, default="onpolicy_distill")
    parser.add_argument("--seed", type=int, default=42)
    add_base_training_cli_args(parser)
    add_distillation_cli_args(parser)

    parser.add_argument("--wandb-project", type=str, default=os.getenv("WANDB_PROJECT", "CI"))
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def load_training_and_eval_datasets(
    dataset_name_or_path: str,
    student_model_name_or_path: str,
) -> tuple[Dataset, Dataset | None]:
    dataset_path = Path(dataset_name_or_path)
    if dataset_path.exists():
        loaded = load_from_disk(str(dataset_path))
        if isinstance(loaded, Dataset):
            return loaded, None
        if isinstance(loaded, DatasetDict):
            if TRAIN_SPLIT not in loaded:
                raise ValueError(
                    f"{dataset_name_or_path} does not include '{TRAIN_SPLIT}'. "
                    f"Available splits: {list(loaded.keys())}."
                )
            eval_dataset = loaded[EVAL_SPLIT] if EVAL_SPLIT in loaded else None
            return loaded[TRAIN_SPLIT], eval_dataset
        raise TypeError(f"Unsupported dataset object: {type(loaded)}")

    splits = prepare_grpo_dataset_splits_from_hf(
        dataset_path=dataset_name_or_path,
        model_id=student_model_name_or_path,
        source_split=TRAIN_SPLIT,
    )
    if TEST_SPLIT not in splits:
        raise ValueError(
            f"Preprocessing did not produce '{TEST_SPLIT}' split. Available: {list(splits.keys())}."
        )
    return splits[TRAIN_SPLIT], splits[EVAL_SPLIT]


def validate_dataset_columns(dataset: Dataset, split_name: str) -> None:
    if "prompt" not in dataset.column_names:
        raise ValueError(
            f"Dataset split '{split_name}' is missing required 'prompt' column. "
            f"Available columns: {dataset.column_names}"
        )


def build_training_args(
    args: argparse.Namespace,
    has_eval_dataset: bool,
    enable_wandb: bool,
) -> TrainingArguments:
    training_args = build_hf_training_args(
        args=args,
        output_dir=args.output_dir,
        has_eval_dataset=has_eval_dataset,
        enable_wandb=enable_wandb,
        config_class=TrainingArguments,
        extra_config_kwargs={
            "bf16": torch.cuda.is_available(),
        },
    )
    return attach_distillation_training_args(training_args, args)


def init_wandb_run(args: argparse.Namespace):
    if wandb is None:
        return None
    return wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=vars(args),
    )


def _is_olmo_model(model: torch.nn.Module) -> bool:
    config = getattr(model, "config", None)
    if config is None:
        return False
    model_type = getattr(config, "model_type", "")
    return "olmo" in model_type.lower()


def apply_lora(model: torch.nn.Module) -> torch.nn.Module:
    lora_kwargs: dict[str, Any] = {
        "r": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "bias": "none",
        "task_type": TaskType.CAUSAL_LM,
    }
    if _is_olmo_model(model):
        lora_kwargs["target_modules"] = ["q_proj", "v_proj"]
    peft_config = LoraConfig(**lora_kwargs)
    return get_peft_model(model, peft_config)


def _keyword_aliases(example: dict[str, Any]) -> tuple[Any, Any]:
    required = example.get("required_keywords")
    if required is None:
        required = example.get("allowed_keywords", [])
    restricted = example.get("restricted_keywords")
    if restricted is None:
        restricted = example.get("disallowed_keywords", [])
    return required, restricted


def prompt_only_collator(examples: list[dict[str, Any]]) -> dict[str, Any]:
    prompts: list[str] = []
    required_keywords: list[Any] = []
    restricted_keywords: list[Any] = []
    use_post_think_response: list[Any] = []
    for example in examples:
        required, restricted = _keyword_aliases(example)
        prompts.append(example["prompt"])
        required_keywords.append(required if required is not None else [])
        restricted_keywords.append(restricted if restricted is not None else [])
        use_post_think_response.append(example.get("use_post_think_response"))
    return {
        "prompt": prompts,
        "required_keywords": required_keywords,
        "restricted_keywords": restricted_keywords,
        "use_post_think_response": use_post_think_response,
    }


def main() -> None:
    args = parse_args()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    set_seed(args.seed)

    if accelerator.is_main_process:
        logger.info("Training args: {}", vars(args))

    enable_wandb = (
        accelerator.is_main_process
        and not args.disable_wandb
        and wandb is not None
        and bool(os.getenv("WANDB_API_KEY"))
    )
    wandb_run = init_wandb_run(args) if enable_wandb else None

    try:
        train_dataset, eval_dataset = load_training_and_eval_datasets(
            dataset_name_or_path=args.dataset_name_or_path,
            student_model_name_or_path=args.student_model_name_or_path,
        )
        validate_dataset_columns(train_dataset, TRAIN_SPLIT)
        if eval_dataset is not None:
            validate_dataset_columns(eval_dataset, EVAL_SPLIT)

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.student_model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.student_model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = apply_lora(model)
        if accelerator.is_main_process and hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

        training_args = build_training_args(
            args=args,
            has_eval_dataset=eval_dataset is not None and args.eval_strategy != "no",
            enable_wandb=enable_wandb,
        )

        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "teacher_model_path": args.teacher_model_name_or_path,
            "args": training_args,
            "train_dataset": train_dataset,
            "processing_class": tokenizer,
            "loss_type": args.loss_type,
            "beta": args.beta,
            "data_collator": prompt_only_collator,
            "compute_metrics": build_compute_eval_metrics(args.eval_num_generations),
        }
        if eval_dataset is not None and args.eval_strategy != "no":
            trainer_kwargs["eval_dataset"] = eval_dataset

        trainer = MultiGPUOnPolicyTrainer(**trainer_kwargs)
        trainer.train()
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
