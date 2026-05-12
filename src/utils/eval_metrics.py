from __future__ import annotations

from typing import Any

import torch

from rl.rewards import EVAL_METRIC_NAMES, EVAL_PREDICTION_METRIC_NAMES


def build_compute_eval_metrics(eval_num_generations: int):
    num_generations = max(1, int(eval_num_generations))
    suffix = f"_avg@{num_generations}"

    def compute_eval_metrics(eval_pred: Any) -> dict[str, float]:
        predictions = getattr(eval_pred, "predictions", None)
        if predictions is None:
            return {}

        values = torch.as_tensor(predictions, dtype=torch.float32)
        if values.numel() == 0:
            return {}

        if values.ndim == 1:
            mean_reward = values.mean().item()
            return {
                "reward": mean_reward,
                f"reward{suffix}": mean_reward,
            }

        if values.ndim == 2 and values.shape[-1] in {
            len(EVAL_METRIC_NAMES),
            len(EVAL_PREDICTION_METRIC_NAMES),
        }:
            metric_names = (
                EVAL_PREDICTION_METRIC_NAMES
                if values.shape[-1] == len(EVAL_PREDICTION_METRIC_NAMES)
                else EVAL_METRIC_NAMES
            )
            means = values.mean(dim=0)
            summary: dict[str, float] = {}
            for index, name in enumerate(metric_names):
                metric_value = means[index].item()
                summary[name] = metric_value
                summary[f"{name}{suffix}"] = metric_value
            return summary

        flattened = values.reshape(-1)
        mean_reward = flattened.mean().item()
        return {
            "reward": mean_reward,
            f"reward{suffix}": mean_reward,
        }

    return compute_eval_metrics
