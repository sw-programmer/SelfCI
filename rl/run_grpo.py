from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
import sys
from typing import Any

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import Dataset, DatasetDict, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer

import wandb
from loguru import logger
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENV_PATH = REPO_ROOT / ".env"
if load_dotenv(ENV_PATH):
    logger.info("Dotenv loaded from {}", ENV_PATH)
else:
    logger.warning("Dotenv not found at {}", ENV_PATH)

from rl.rewards import contextual_integrity_reward, contextual_integrity_reward_metrics
from utils.preprocess import prepare_grpo_dataset_splits_from_hf

DATASET_NAME_OR_PATH = "huseyinatahaninan/ContextualIntegritySyntheticDataset"
TRAIN_SPLIT = "train"
EVAL_SPLIT = "eval"
TEST_SPLIT = "test"
LORA_RANK = 32
LORA_DROPOUT = 0.05
LORA_ALPHA = 64
CI_METRIC_NAMES = ("utility", "integrity", "complete")


def _repair_completion_text_for_eval(completion: Any) -> str:
    if isinstance(completion, str):
        text = completion
    elif isinstance(completion, dict):
        content = completion.get("content", "")
        text = content if isinstance(content, str) else str(content)
    elif isinstance(completion, list) and completion:
        last_item = completion[-1]
        if isinstance(last_item, dict):
            content = last_item.get("content", "")
            text = content if isinstance(content, str) else str(content)
        elif isinstance(last_item, str):
            text = last_item
        else:
            text = str(last_item)
    else:
        text = str(completion)

    text_lower = text.lower()
    if "<answer>" in text_lower and "</answer>" not in text_lower:
        return text.rstrip() + "\n</answer>"
    return text


def build_contextual_integrity_metric_tensor(
    completions: list[Any],
    examples: list[dict[str, Any]],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if not completions or not examples:
        return torch.empty((0, len(CI_METRIC_NAMES)), dtype=torch.float32, device=device)

    required_keywords: list[Any] = []
    restricted_keywords: list[Any] = []
    use_post_think_response_flags: list[bool] = []
    for example in examples:
        required = example.get("required_keywords", example.get("allowed_keywords"))
        restricted = example.get("restricted_keywords", example.get("disallowed_keywords"))
        if required is None or restricted is None:
            return torch.empty((0, len(CI_METRIC_NAMES)), dtype=torch.float32, device=device)
        required_keywords.append(required)
        restricted_keywords.append(restricted)
        use_post_think_response_flags.append(bool(example.get("use_post_think_response")))

    per_completion_metrics = contextual_integrity_reward_metrics(
        completions=completions,
        required_keywords=required_keywords,
        restricted_keywords=restricted_keywords,
        use_post_think_response=use_post_think_response_flags,
    )
    metric_rows = [
        [row["utility"], row["integrity"], row["complete"]]
        for row in per_completion_metrics
    ]
    if not metric_rows:
        return torch.empty((0, len(CI_METRIC_NAMES)), dtype=torch.float32, device=device)

    return torch.tensor(metric_rows, dtype=torch.float32, device=device)


class ContextualIntegrityGRPOTrainer(GRPOTrainer):
    def _ci_metric_suffix(self) -> str:
        num_generations = self.num_generations if self.model.training else self.num_generations_eval
        return f"_avg@{max(1, int(num_generations))}"

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        effective_completions = completions
        if not self.model.training:
            effective_completions = [
                _repair_completion_text_for_eval(completion)
                for completion in completions
            ]

        rewards_per_func = super()._calculate_rewards(
            inputs,
            prompts,
            effective_completions,
            completion_ids_list,
        )

        metric_tensor = build_contextual_integrity_metric_tensor(
            completions=effective_completions,
            examples=inputs,
            device=self.accelerator.device,
        )
        if metric_tensor.numel() == 0:
            return rewards_per_func

        gathered_metric_tensor = self.accelerator.gather(metric_tensor)
        metric_means = gathered_metric_tensor.mean(dim=0).tolist()
        mode = "train" if self.model.training else "eval"
        metric_suffix = self._ci_metric_suffix()
        for metric_name, metric_value in zip(CI_METRIC_NAMES, metric_means, strict=True):
            metric_value = float(metric_value)
            self._metrics[mode][metric_name].append(metric_value)
            self._metrics[mode][f"{metric_name}{metric_suffix}"].append(metric_value)

        return rewards_per_func


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal GRPO training entrypoint.")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/grpo")
    parser.add_argument("--run-name", type=str, default="grpo")
    parser.add_argument("--seed", type=int, default=42)

    # Defaults from the provided hyperparameter settings.
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=15)
    parser.add_argument("--num-generations", type=int, default=16)
    parser.add_argument("--eval-num-generations", type=int, default=5)
    parser.add_argument("--mini-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-penalty", type=float, default=0.0)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=30)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--wandb-project", type=str, default=os.getenv("WANDB_PROJECT", "CI"))
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    if args.eval_batch_size < 1:
        parser.error("--eval-batch-size must be at least 1.")
    if args.eval_num_generations < 1:
        parser.error("--eval-num-generations must be at least 1.")
    return args


def load_training_and_eval_datasets(
    dataset_name_or_path: str,
    model_name_or_path: str,
) -> tuple[Dataset, Dataset | None]:
    dataset_path = Path(dataset_name_or_path)
    if dataset_path.exists():
        loaded = load_from_disk(str(dataset_path))
        if isinstance(loaded, Dataset):
            raise ValueError(
                f"{dataset_name_or_path} is a single Dataset. "
                "Expected a DatasetDict with 'train', 'eval', and 'test' splits "
                "from utils/preprocess.py."
            )
        if isinstance(loaded, DatasetDict):
            required_splits = {TRAIN_SPLIT, EVAL_SPLIT, TEST_SPLIT}
            missing = required_splits.difference(set(loaded.keys()))
            if missing:
                raise ValueError(
                    f"{dataset_name_or_path} is missing required splits {sorted(missing)}. "
                    f"Available splits: {list(loaded.keys())}."
                )
            return loaded[TRAIN_SPLIT], loaded[EVAL_SPLIT]
        raise TypeError(f"Unsupported dataset object: {type(loaded)}")

    # Raw HF dataset has only "train". Build train/eval/test via preprocessing.
    splits = prepare_grpo_dataset_splits_from_hf(
        dataset_path=dataset_name_or_path,
        model_id=model_name_or_path,
        source_split=TRAIN_SPLIT,
    )
    return splits[TRAIN_SPLIT], splits[EVAL_SPLIT]


def _ensure_keyword_aliases(dataset: Dataset) -> Dataset:
    columns = set(dataset.column_names)
    needs_required_alias = "required_keywords" not in columns and "allowed_keywords" in columns
    needs_restricted_alias = (
        "restricted_keywords" not in columns and "disallowed_keywords" in columns
    )

    if not needs_required_alias and not needs_restricted_alias:
        return dataset

    def with_aliases(example: dict[str, Any]) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        if needs_required_alias:
            updates["required_keywords"] = example["allowed_keywords"]
        if needs_restricted_alias:
            updates["restricted_keywords"] = example["disallowed_keywords"]
        return updates

    return dataset.map(with_aliases)


def validate_dataset_columns(dataset: Dataset) -> None:
    required_columns = {"prompt"}
    allowed_candidates = {"allowed_keywords", "required_keywords"}
    disallowed_candidates = {"disallowed_keywords", "restricted_keywords"}

    missing = required_columns.difference(dataset.column_names)
    if missing or not allowed_candidates.intersection(dataset.column_names) or not disallowed_candidates.intersection(dataset.column_names):
        keyword_missing: list[str] = []
        if not allowed_candidates.intersection(dataset.column_names):
            keyword_missing.append("allowed_keywords/required_keywords")
        if not disallowed_candidates.intersection(dataset.column_names):
            keyword_missing.append("disallowed_keywords/restricted_keywords")
        raise ValueError(
            "Preprocessed GRPO dataset is missing required columns. "
            f"Missing prompt columns: {sorted(missing)}. "
            f"Missing keyword columns: {keyword_missing}."
        )


def _supported_kwargs(config_class: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop unsupported fields to tolerate minor TRL version differences."""
    try:
        parameter_names = set(inspect.signature(config_class.__init__).parameters.keys())
    except (TypeError, ValueError):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameter_names}


def build_grpo_config(
    args: argparse.Namespace,
    has_eval_dataset: bool,
    enable_wandb: bool,
) -> GRPOConfig:
    total_samples = args.batch_size * args.num_generations
    if total_samples % args.mini_batch_size != 0:
        raise ValueError(
            "mini-batch-size must divide (batch-size * num-generations). "
            f"Got {args.mini_batch_size} for {args.batch_size} * {args.num_generations}."
        )
    gradient_accumulation_steps = total_samples // args.mini_batch_size

    config_kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "seed": args.seed,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "num_generations": args.num_generations,
        "num_generations_eval": args.eval_num_generations,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": 0.1,
        "beta": args.beta,
        "epsilon": args.clip_ratio,
        "temperature": args.temperature,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_response_length,
        "num_train_epochs": args.epochs,
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "save_total_limit": args.save_total_limit,
        "report_to": ["wandb"] if enable_wandb else [],
        "use_vllm": True,
        "vllm_mode": "colocate",
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vllm_max_model_length": 16384
    }

    if has_eval_dataset:
        config_kwargs["eval_strategy"] = "epoch"
        config_kwargs["eval_on_start"] = True
    else:
        config_kwargs["eval_strategy"] = "no"

    # Different TRL versions use different names for entropy control.
    for entropy_key in (
        "entropy_beta",
        "entropy_coef",
        "entropy_coeff",
        "entropy_coefficient",
    ):
        config_kwargs[entropy_key] = args.entropy_penalty

    filtered_kwargs = _supported_kwargs(GRPOConfig, config_kwargs)
    return GRPOConfig(**filtered_kwargs)


def init_wandb_run(args: argparse.Namespace):
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


def main() -> None:
    args = parse_args()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    set_seed(args.seed)
    if accelerator.is_main_process:
        logger.info("Training args: {}", vars(args))

    wandb_run = init_wandb_run(args) if accelerator.is_main_process else None

    try:
        train_dataset, eval_dataset = load_training_and_eval_datasets(
            DATASET_NAME_OR_PATH,
            model_name_or_path=args.model_name_or_path,
        )
        train_dataset = _ensure_keyword_aliases(train_dataset)
        validate_dataset_columns(train_dataset)
        if eval_dataset is not None:
            eval_dataset = _ensure_keyword_aliases(eval_dataset)
            validate_dataset_columns(eval_dataset)

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = apply_lora(model)
        if accelerator.is_main_process and hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

        training_args = build_grpo_config(
            args,
            has_eval_dataset=eval_dataset is not None,
            enable_wandb=accelerator.is_main_process,
        )

        trainer_kwargs: dict[str, Any] = {
            "model": model,
            "processing_class": tokenizer,
            "reward_funcs": [contextual_integrity_reward],
            "args": training_args,
            "train_dataset": train_dataset,
        }
        if eval_dataset is not None:
            trainer_kwargs["eval_dataset"] = eval_dataset

        trainer = ContextualIntegrityGRPOTrainer(**trainer_kwargs)
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
