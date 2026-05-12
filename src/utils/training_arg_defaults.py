from __future__ import annotations

import argparse
import inspect
from typing import Any


TRAINING_STRATEGY_CHOICES = ("no", "steps", "epoch")
DISTILLATION_LOSS_CHOICES = ("reverse_kl", "forward_kl", "jsd")

BASE_TRAINING_CLI_DEFAULTS: dict[str, Any] = {
    "batch_size": 1,
    "eval_batch_size": 2,
    "gradient_accumulation_steps": 2,
    "learning_rate": 1e-6,
    "weight_decay": 0.0,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "epochs": 30,
    "logging_steps": 10,
    "save_total_limit": 30,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "save_steps": 1,
    "eval_steps": 1,
}

GENERATION_CLI_DEFAULTS: dict[str, Any] = {
    "temperature": 0.7,
    "max_completion_length": 2048,
    "eval_num_generations": 5,
    "vllm_gpu_memory_utilization": 0.3,
    "vllm_tensor_parallel_size": 1,
    "vllm_seed": 42,
    "max_model_len": None,
}

DISTILLATION_OBJECTIVE_CLI_DEFAULTS: dict[str, Any] = {
    "loss_type": "reverse_kl",
    "beta": 0.5,
}

DISTILLATION_CLI_DEFAULTS: dict[str, Any] = {
    **DISTILLATION_OBJECTIVE_CLI_DEFAULTS,
    **GENERATION_CLI_DEFAULTS,
}


def _merge_defaults(defaults: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = defaults.copy()
    if overrides:
        merged.update(overrides)
    return merged


def add_base_training_cli_args(
    parser: argparse.ArgumentParser,
    overrides: dict[str, Any] | None = None,
) -> None:
    defaults = _merge_defaults(BASE_TRAINING_CLI_DEFAULTS, overrides)
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--eval-batch-size", type=int, default=defaults["eval_batch_size"])
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=defaults["gradient_accumulation_steps"],
    )
    parser.add_argument("--learning-rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--warmup-ratio", type=float, default=defaults["warmup_ratio"])
    parser.add_argument("--max-grad-norm", type=float, default=defaults["max_grad_norm"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--logging-steps", type=int, default=defaults["logging_steps"])
    parser.add_argument("--save-total-limit", type=int, default=defaults["save_total_limit"])
    parser.add_argument(
        "--save-strategy",
        type=str,
        default=defaults["save_strategy"],
        choices=TRAINING_STRATEGY_CHOICES,
    )
    parser.add_argument(
        "--eval-strategy",
        type=str,
        default=defaults["eval_strategy"],
        choices=TRAINING_STRATEGY_CHOICES,
    )
    parser.add_argument("--save-steps", type=int, default=defaults["save_steps"])
    parser.add_argument("--eval-steps", type=int, default=defaults["eval_steps"])


def add_generation_cli_args(
    parser: argparse.ArgumentParser,
    overrides: dict[str, Any] | None = None,
) -> None:
    defaults = _merge_defaults(GENERATION_CLI_DEFAULTS, overrides)
    parser.add_argument("--temperature", type=float, default=defaults["temperature"])
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=defaults["max_completion_length"],
    )
    parser.add_argument(
        "--eval-num-generations",
        type=int,
        default=defaults["eval_num_generations"],
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=defaults["vllm_gpu_memory_utilization"],
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=defaults["vllm_tensor_parallel_size"],
    )
    parser.add_argument("--vllm-seed", type=int, default=defaults["vllm_seed"])
    parser.add_argument(
        "--max-model-len",
        "--vllm-max-model-len",
        type=int,
        dest="max_model_len",
        default=defaults["max_model_len"],
    )


def add_kl_objective_cli_args(
    parser: argparse.ArgumentParser,
    overrides: dict[str, Any] | None = None,
) -> None:
    defaults = _merge_defaults(DISTILLATION_OBJECTIVE_CLI_DEFAULTS, overrides)
    parser.add_argument(
        "--loss-type",
        type=str,
        default=defaults["loss_type"],
        choices=DISTILLATION_LOSS_CHOICES,
    )
    parser.add_argument("--beta", type=float, default=defaults["beta"])


def add_distillation_cli_args(
    parser: argparse.ArgumentParser,
    overrides: dict[str, Any] | None = None,
) -> None:
    defaults = _merge_defaults(DISTILLATION_CLI_DEFAULTS, overrides)
    add_kl_objective_cli_args(parser, defaults)
    add_generation_cli_args(parser, defaults)


def supported_kwargs(config_class: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        parameter_names = set(inspect.signature(config_class.__init__).parameters.keys())
    except (TypeError, ValueError):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameter_names}


def build_training_config_kwargs(
    *,
    args: argparse.Namespace,
    output_dir: str,
    has_eval_dataset: bool,
    enable_wandb: bool,
    extra_config_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eval_strategy = args.eval_strategy if has_eval_dataset else "no"
    config_kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "run_name": args.run_name,
        "seed": args.seed,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "num_train_epochs": args.epochs,
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "save_strategy": args.save_strategy,
        "report_to": ["wandb"] if enable_wandb else [],
        "remove_unused_columns": False,
        "logging_first_step": True,
        "eval_strategy": eval_strategy,
        "eval_on_start": True,
    }

    if extra_config_kwargs:
        config_kwargs.update(extra_config_kwargs)

    if eval_strategy == "steps":
        config_kwargs["eval_steps"] = args.eval_steps or args.logging_steps
    if args.save_strategy == "steps":
        config_kwargs["save_steps"] = args.save_steps or args.logging_steps

    return config_kwargs


def build_hf_training_args(
    *,
    args: argparse.Namespace,
    output_dir: str,
    has_eval_dataset: bool,
    enable_wandb: bool,
    config_class: type,
    extra_config_kwargs: dict[str, Any] | None = None,
):
    config_kwargs = build_training_config_kwargs(
        args=args,
        output_dir=output_dir,
        has_eval_dataset=has_eval_dataset,
        enable_wandb=enable_wandb,
        extra_config_kwargs=extra_config_kwargs,
    )
    filtered_kwargs = supported_kwargs(config_class, config_kwargs)
    return config_class(**filtered_kwargs)


def attach_generation_training_args(training_args: Any, args: argparse.Namespace) -> Any:
    setattr(training_args, "temperature", args.temperature)
    setattr(training_args, "max_completion_length", args.max_completion_length)
    setattr(training_args, "eval_num_generations", max(1, args.eval_num_generations))
    setattr(
        training_args,
        "vllm_gpu_memory_utilization",
        args.vllm_gpu_memory_utilization,
    )
    setattr(
        training_args,
        "vllm_tensor_parallel_size",
        args.vllm_tensor_parallel_size,
    )
    setattr(training_args, "vllm_seed", args.vllm_seed)
    setattr(training_args, "max_model_len", args.max_model_len)
    return training_args


def attach_distillation_training_args(training_args: Any, args: argparse.Namespace) -> Any:
    return attach_generation_training_args(training_args, args)
