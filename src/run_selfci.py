from __future__ import annotations

import argparse
import copy
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

from src.utils.training_arg_defaults import (
    add_base_training_cli_args,
    add_generation_cli_args,
    attach_generation_training_args,
    build_hf_training_args,
)
from src.utils.eval_metrics import build_compute_eval_metrics
from src.utils.selfci_trainer import (
    FEEDBACK_WEIGHT_TYPE_CHOICES,
    KL_SUPPORT_CHOICES,
    SelfCITrainer,
)
from src.utils.preprocess import (
    dataset_has_raw_ci_source_columns,
    prepare_dual_feedback_onpolicy_dataset_splits_from_hf,
    render_dual_feedback_onpolicy_dataset,
)

ENV_PATH = REPO_ROOT / ".env"
if load_dotenv(ENV_PATH):
    logger.info("Dotenv loaded from {}", ENV_PATH)
else:
    logger.warning("Dotenv not found at {}", ENV_PATH)

DATASET_NAME_OR_PATH = "Sangsang/ContextualIntegritySyntheticDataset_Qwen2.5-7B-Instruct_merged"
DEFAULT_STUDENT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TEACHER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TRAIN_SPLIT = "train"
EVAL_SPLIT = "eval"
TEST_SPLIT = "test"
LORA_RANK = 32
LORA_DROPOUT = 0.05
LORA_ALPHA = 64

TEACHER_UPDATE_CHOICES = ("none", "ema", "interp", "ema+interp")

def _has_explicit_feedback_weight_arg(argv: list[str]) -> bool:
    weight_flags = ("--backward-weight", "--forward-weight", "--lambda-allow", "--lambda-disallow")
    return any(
        arg == flag or arg.startswith(f"{flag}=")
        for arg in argv
        for flag in weight_flags
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Weighted asymmetric bidirectional KL on-policy distillation training entrypoint."
    )
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
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/feedback_weighted_asymmetric_bidirectional_kl_onpolicy",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="feedback_weighted_asymmetric_bidirectional_kl_onpolicy_distill",
    )
    parser.add_argument("--seed", type=int, default=42)
    add_base_training_cli_args(parser)
    add_generation_cli_args(parser)
    parser.add_argument("--backward-weight", type=float, default=1.0)
    parser.add_argument("--forward-weight", type=float, default=1.0)
    parser.add_argument(
        "--feedback-weight-type",
        type=str,
        default="fixed",
        choices=FEEDBACK_WEIGHT_TYPE_CHOICES,
        help="Use fixed branch weights or adaptive CI-failure-driven weights.",
    )
    parser.add_argument(
        "--kl-support",
        type=str,
        default="full",
        choices=KL_SUPPORT_CHOICES,
        help=(
            "KL support used per generated token: full vocabulary, student top-k support, "
            "or the sampled student token only."
        ),
    )
    parser.add_argument(
        "--kl-topk",
        type=int,
        default=50,
        help="Top-k value used when --kl-support=topk.",
    )
    parser.add_argument("--lambda-allow", dest="lambda_allow", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--lambda-disallow",
        dest="lambda_disallow",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--teacher-update",
        type=str,
        default="none",
        choices=TEACHER_UPDATE_CHOICES,
        help="Teacher target source: frozen teacher, EMA teacher, interpolation, or EMA+interpolation.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="EMA decay used when --teacher-update is ema or ema+interp.",
    )
    parser.add_argument(
        "--student-target-weight",
        type=float,
        default=0.3,
        help="Detached-student interpolation weight used when --teacher-update is interp or ema+interp.",
    )

    parser.add_argument("--wandb-project", type=str, default=os.getenv("WANDB_PROJECT", "CI"))
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to a Trainer checkpoint directory, or 'last' to resume from the latest checkpoint in --output-dir.",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=os.getenv("WANDB_RUN_ID"),
        help="Existing W&B run id to continue logging into.",
    )
    parser.add_argument(
        "--wandb-resume",
        type=str,
        default=os.getenv("WANDB_RESUME"),
        choices=("allow", "must", "never"),
        help="W&B resume policy. Defaults to 'must' when --wandb-run-id is set.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.feedback_weight_type == "adaptive" and _has_explicit_feedback_weight_arg(sys.argv[1:]):
        parser.error(
            "--feedback-weight-type=adaptive cannot be combined with --backward-weight, "
            "--forward-weight, --lambda-allow, or --lambda-disallow."
        )
    if args.lambda_allow is not None:
        args.forward_weight = args.lambda_allow
    if args.lambda_disallow is not None:
        args.backward_weight = args.lambda_disallow
    if args.kl_topk <= 0:
        parser.error("--kl-topk must be a positive integer.")
    if not 0.0 <= args.ema_decay <= 1.0:
        parser.error("--ema-decay must be in the interval [0, 1].")
    if not 0.0 <= args.student_target_weight <= 1.0:
        parser.error("--student-target-weight must be in the interval [0, 1].")
    if args.wandb_resume is not None and args.wandb_run_id is None:
        parser.error("--wandb-resume requires --wandb-run-id.")
    return args


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

    splits = prepare_dual_feedback_onpolicy_dataset_splits_from_hf(
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
    required_columns = {"prompt", "allowed_feedbacks", "disallowed_feedbacks"}
    missing = required_columns.difference(dataset.column_names)
    if missing:
        raise ValueError(
            f"Dataset split '{split_name}' is missing required columns {sorted(missing)}. "
            f"Available columns: {dataset.column_names}."
        )

    if "required_keywords" not in dataset.column_names and "allowed_keywords" not in dataset.column_names:
        raise ValueError(
            f"Dataset split '{split_name}' must include either 'required_keywords' or 'allowed_keywords'. "
            f"Available columns: {dataset.column_names}."
        )
    if "restricted_keywords" not in dataset.column_names and "disallowed_keywords" not in dataset.column_names:
        raise ValueError(
            f"Dataset split '{split_name}' must include either 'restricted_keywords' or 'disallowed_keywords'. "
            f"Available columns: {dataset.column_names}."
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
    return attach_generation_training_args(training_args, args)


def init_wandb_run(args: argparse.Namespace):
    if wandb is None:
        return None

    init_kwargs: dict[str, Any] = {
        "project": args.wandb_project,
        "name": args.run_name,
        "config": vars(args),
    }
    if args.wandb_run_id is not None:
        init_kwargs["id"] = args.wandb_run_id
        init_kwargs["resume"] = args.wandb_resume or "must"

    return wandb.init(**init_kwargs)


def resolve_resume_from_checkpoint(resume_from_checkpoint: str | None) -> str | bool | None:
    if resume_from_checkpoint is None:
        return None

    normalized = resume_from_checkpoint.strip()
    if not normalized:
        return None
    if normalized.lower() in {"1", "true", "yes", "last", "latest", "auto"}:
        return True
    return normalized


def apply_lora(model: torch.nn.Module) -> torch.nn.Module:
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, peft_config)


def _keyword_aliases(example: dict[str, Any]) -> tuple[Any, Any]:
    required = example.get("required_keywords")
    if required is None:
        required = example.get("allowed_keywords", [])
    restricted = example.get("restricted_keywords")
    if restricted is None:
        restricted = example.get("disallowed_keywords", [])
    return required, restricted


def prompt_and_weighted_asymmetric_bidirectional_kl_collator(examples: list[dict[str, Any]]) -> dict[str, Any]:
    required_keywords: list[Any] = []
    restricted_keywords: list[Any] = []
    for example in examples:
        required, restricted = _keyword_aliases(example)
        required_keywords.append(required if required is not None else [])
        restricted_keywords.append(restricted if restricted is not None else [])

    return {
        "prompt": [example["prompt"] for example in examples],
        "allowed_feedbacks": [example.get("allowed_feedbacks", []) for example in examples],
        "disallowed_feedbacks": [example.get("disallowed_feedbacks", []) for example in examples],
        "required_keywords": required_keywords,
        "restricted_keywords": restricted_keywords,
        "use_post_think_response": [example.get("use_post_think_response") for example in examples],
    }


def main() -> None:
    args = parse_args()
    resume_from_checkpoint = resolve_resume_from_checkpoint(args.resume_from_checkpoint)
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
            "args": training_args,
            "train_dataset": train_dataset,
            "processing_class": tokenizer,
            "backward_weight": args.backward_weight,
            "forward_weight": args.forward_weight,
            "feedback_weight_type": args.feedback_weight_type,
            "kl_support": args.kl_support,
            "kl_topk": args.kl_topk,
            "teacher_update": args.teacher_update,
            "ema_decay": args.ema_decay,
            "student_target_weight": args.student_target_weight,
            "data_collator": prompt_and_weighted_asymmetric_bidirectional_kl_collator,
            "compute_metrics": build_compute_eval_metrics(args.eval_num_generations),
        }
        if eval_dataset is not None and args.eval_strategy != "no":
            trainer_kwargs["eval_dataset"] = eval_dataset

        if args.teacher_update in {"ema", "ema+interp"}:
            if accelerator.is_main_process and args.teacher_model_name_or_path:
                logger.info(
                    "Ignoring --teacher-model-name-or-path={} because --teacher-update={} uses a student-backed teacher.",
                    args.teacher_model_name_or_path,
                    args.teacher_update,
                )
            trainer_kwargs["teacher_model"] = copy.deepcopy(model)
        else:
            trainer_kwargs["teacher_model_path"] = args.teacher_model_name_or_path

        trainer = SelfCITrainer(**trainer_kwargs)
        if accelerator.is_main_process and resume_from_checkpoint is not None:
            logger.info("Resuming training from checkpoint target: {}", resume_from_checkpoint)
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
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
