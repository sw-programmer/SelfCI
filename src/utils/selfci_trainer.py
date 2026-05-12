from __future__ import annotations

from typing import Any

import torch
from torch.nn.functional import kl_div, log_softmax, pad
from transformers import TrainerCallback

from rl.rewards import contextual_integrity_reward_metrics
from src.utils.onpolicy_trainer import MultiGPUOnPolicyTrainer


FEEDBACK_WEIGHT_TYPE_CHOICES = ("fixed", "adaptive")
KL_SUPPORT_CHOICES = ("full", "topk", "sampled")
TEACHER_UPDATE_CHOICES = ("none", "ema", "interp", "ema+interp")


class _WeightedAsymmetricBidirectionalKLEMATeacherCallback(TrainerCallback):
    def __init__(self, trainer: "SelfCITrainer"):
        self.trainer = trainer

    def on_optimizer_step(self, args, state, control, **kwargs):
        del args, state, kwargs  # Unused callback inputs.
        self.trainer._update_teacher_model_ema(self.trainer.ema_decay)
        return control


class SelfCITrainer(MultiGPUOnPolicyTrainer):
    """
    On-policy trainer with asymmetric teacher prompts and weighted bidirectional KL.

    Student generation uses the base prompt. For each sampled completion, the
    reverse-KL branch conditions the teacher on disallowed feedback only, while
    allowed-feedback branch conditions the teacher on allowed feedback only.
    """

    def __init__(
        self,
        *args,
        backward_weight: float = 1.0,
        forward_weight: float = 1.0,
        feedback_weight_type: str = "fixed",
        kl_support: str = "full",
        kl_topk: int = 50,
        teacher_update: str = "none",
        ema_decay: float = 0.999,
        student_target_weight: float = 0.3,
        **kwargs,
    ):
        lambda_allow = kwargs.pop("lambda_allow", None)
        lambda_disallow = kwargs.pop("lambda_disallow", None)
        if lambda_allow is not None:
            forward_weight = lambda_allow
        if lambda_disallow is not None:
            backward_weight = lambda_disallow

        super().__init__(*args, **kwargs)
        if backward_weight < 0 or forward_weight < 0:
            raise ValueError("backward_weight and forward_weight must be non-negative.")
        if feedback_weight_type not in FEEDBACK_WEIGHT_TYPE_CHOICES:
            raise ValueError(f"Unsupported feedback_weight_type: {feedback_weight_type}")
        if kl_support not in KL_SUPPORT_CHOICES:
            raise ValueError(f"Unsupported kl_support: {kl_support}")
        if kl_topk <= 0:
            raise ValueError("kl_topk must be a positive integer.")
        if teacher_update not in TEACHER_UPDATE_CHOICES:
            raise ValueError(f"Unsupported teacher_update: {teacher_update}")
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError("ema_decay must be in the interval [0, 1].")
        if not 0.0 <= student_target_weight <= 1.0:
            raise ValueError("student_target_weight must be in the interval [0, 1].")
        total_weight = backward_weight + forward_weight
        if total_weight <= 0:
            raise ValueError("At least one of backward_weight or forward_weight must be positive.")

        self.backward_weight = backward_weight
        self.forward_weight = forward_weight
        self.total_weight = total_weight
        self.feedback_weight_type = feedback_weight_type
        self.kl_support = kl_support
        self.kl_topk = int(kl_topk)
        self.teacher_update = teacher_update
        self.ema_decay = float(ema_decay)
        self.student_target_weight = float(student_target_weight)
        self._single_user_prompt_wrapper: tuple[str, str] | None = None

        if self.teacher_update in {"ema", "ema+interp"}:
            self.add_callback(_WeightedAsymmetricBidirectionalKLEMATeacherCallback(self))

    def save_model(self, output_dir=None, _internal_call: bool = False):
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        if self.teacher_update not in {"ema", "ema+interp"} or not self.args.should_save:
            return

        resolved_output_dir = output_dir or self.args.output_dir
        self._save_teacher_adapter(resolved_output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        if self.teacher_update in {"ema", "ema+interp"}:
            self._load_teacher_adapter(resume_from_checkpoint, strict=True)

    def _get_single_user_prompt_wrapper(self) -> tuple[str, str]:
        if self._single_user_prompt_wrapper is not None:
            return self._single_user_prompt_wrapper

        sentinel = "__CODEX_USER_CONTENT__"
        rendered = self.processing_class.apply_chat_template(
            [{"role": "user", "content": sentinel}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if sentinel not in rendered:
            raise ValueError("Unable to infer chat template wrapper for single-user prompts.")

        self._single_user_prompt_wrapper = rendered.split(sentinel, 1)
        return self._single_user_prompt_wrapper

    def _append_feedback_to_prompt(
        self,
        base_prompt: str,
        feedback_text: str,
        *,
        allowed_only: bool = False,
    ) -> str:
        prefix, suffix = self._get_single_user_prompt_wrapper()
        if not base_prompt.startswith(prefix) or not base_prompt.endswith(suffix):
            raise ValueError("Prompt does not match the expected single-user chat template.")

        content_end = len(base_prompt) - len(suffix) if suffix else len(base_prompt)
        prompt_content = base_prompt[len(prefix) : content_end]
        feedback_prefix = (
            "[NOTE] Only the following keywords are appropriate to share in this specific context:"
            if allowed_only
            else "[NOTE] Following keywords are inappropriate to share in this specific context:"
        )
        prompt_content = f"{prompt_content.rstrip()}\n\n{feedback_prefix}\n{feedback_text}"
        return f"{prefix}{prompt_content}{suffix}"

    def _flatten_feedback_values(self, raw_feedbacks: Any) -> list[str]:
        if raw_feedbacks is None:
            return []
        if isinstance(raw_feedbacks, str):
            text = raw_feedbacks.strip()
            return [text] if text else []
        if isinstance(raw_feedbacks, dict):
            flattened: list[str] = []
            for value in raw_feedbacks.values():
                flattened.extend(self._flatten_feedback_values(value))
            return flattened
        if isinstance(raw_feedbacks, (list, tuple, set)):
            flattened: list[str] = []
            for value in raw_feedbacks:
                flattened.extend(self._flatten_feedback_values(value))
            return flattened
        text = str(raw_feedbacks).strip()
        return [text] if text else []

    def _normalize_feedbacks(self, raw_feedbacks: Any) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for feedback in self._flatten_feedback_values(raw_feedbacks):
            normalized = feedback.strip()
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
        return deduped

    def _build_teacher_prompts_from_feedback_groups(
        self,
        prompts: list[str],
        feedback_groups: list[list[Any]],
        *,
        allowed_only: bool = False,
    ) -> list[str]:
        teacher_prompts: list[str] = []
        for sample_idx, base_prompt in enumerate(prompts):
            sample_feedbacks = [
                batch_feedbacks[sample_idx] if sample_idx < len(batch_feedbacks) else []
                for batch_feedbacks in feedback_groups
            ]
            merged_feedbacks = self._normalize_feedbacks(sample_feedbacks)
            if not merged_feedbacks:
                teacher_prompts.append(base_prompt)
                continue

            teacher_prompts.append(
                self._append_feedback_to_prompt(
                    base_prompt,
                    "\n\n".join(merged_feedbacks),
                    allowed_only=allowed_only,
                )
            )

        return teacher_prompts

    def _build_forward_teacher_prompts(self, generation_batch: dict[str, Any]) -> list[str]:
        prompts = generation_batch["prompt"]
        batch_allowed_feedbacks = generation_batch.get("allowed_feedbacks", [])
        return self._build_teacher_prompts_from_feedback_groups(
            prompts,
            [batch_allowed_feedbacks],
            allowed_only=True,
        )

    def _build_reverse_teacher_prompts(self, generation_batch: dict[str, Any]) -> list[str]:
        prompts = generation_batch["prompt"]
        batch_disallowed_feedbacks = generation_batch.get("disallowed_feedbacks", [])
        return self._build_teacher_prompts_from_feedback_groups(prompts, [batch_disallowed_feedbacks])

    def _tokenize_teacher_prompts(self, teacher_prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        tokenized_teacher = self.processing_class(
            teacher_prompts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        return self._pad_sequences(tokenized_teacher["input_ids"], padding_side="left")

    def _tokenize_teacher_prompt_pair(
        self,
        forward_teacher_prompts: list[str],
        reverse_teacher_prompts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if reverse_teacher_prompts == forward_teacher_prompts:
            forward_teacher_prompt_ids, forward_teacher_prompt_mask = self._tokenize_teacher_prompts(
                forward_teacher_prompts
            )
            return (
                forward_teacher_prompt_ids,
                forward_teacher_prompt_mask,
                forward_teacher_prompt_ids,
                forward_teacher_prompt_mask,
            )

        unique_teacher_prompts: list[str] = []
        prompt_to_index: dict[str, int] = {}
        for prompt in forward_teacher_prompts + reverse_teacher_prompts:
            if prompt in prompt_to_index:
                continue
            prompt_to_index[prompt] = len(unique_teacher_prompts)
            unique_teacher_prompts.append(prompt)

        tokenized_teacher = self.processing_class(
            unique_teacher_prompts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        unique_teacher_input_ids = tokenized_teacher["input_ids"]
        forward_teacher_input_ids = [unique_teacher_input_ids[prompt_to_index[prompt]] for prompt in forward_teacher_prompts]
        reverse_teacher_input_ids = [unique_teacher_input_ids[prompt_to_index[prompt]] for prompt in reverse_teacher_prompts]
        forward_teacher_prompt_ids, forward_teacher_prompt_mask = self._pad_sequences(
            forward_teacher_input_ids,
            padding_side="left",
        )
        reverse_teacher_prompt_ids, reverse_teacher_prompt_mask = self._pad_sequences(
            reverse_teacher_input_ids,
            padding_side="left",
        )
        return (
            forward_teacher_prompt_ids,
            forward_teacher_prompt_mask,
            reverse_teacher_prompt_ids,
            reverse_teacher_prompt_mask,
        )

    def _prepare_inputs(self, generation_batch: dict[str, Any]) -> dict[str, Any]:
        prepared_inputs = super()._prepare_inputs(generation_batch)
        forward_teacher_prompts = self._build_forward_teacher_prompts(generation_batch)
        reverse_teacher_prompts = self._build_reverse_teacher_prompts(generation_batch)
        (
            forward_teacher_prompt_ids,
            forward_teacher_prompt_mask,
            reverse_teacher_prompt_ids,
            reverse_teacher_prompt_mask,
        ) = self._tokenize_teacher_prompt_pair(
            forward_teacher_prompts,
            reverse_teacher_prompts,
        )

        batch_size = len(generation_batch["prompt"])
        required_keywords = generation_batch.get("required_keywords", generation_batch.get("allowed_keywords"))
        restricted_keywords = generation_batch.get("restricted_keywords", generation_batch.get("disallowed_keywords"))
        if required_keywords is None:
            required_keywords = [[] for _ in range(batch_size)]
        if restricted_keywords is None:
            restricted_keywords = [[] for _ in range(batch_size)]

        prepared_inputs["teacher_prompt_ids"] = forward_teacher_prompt_ids
        prepared_inputs["teacher_prompt_mask"] = forward_teacher_prompt_mask
        prepared_inputs["forward_teacher_prompt_ids"] = forward_teacher_prompt_ids
        prepared_inputs["forward_teacher_prompt_mask"] = forward_teacher_prompt_mask
        prepared_inputs["reverse_teacher_prompt_ids"] = reverse_teacher_prompt_ids
        prepared_inputs["reverse_teacher_prompt_mask"] = reverse_teacher_prompt_mask
        prepared_inputs["required_keywords"] = required_keywords
        prepared_inputs["restricted_keywords"] = restricted_keywords
        return prepared_inputs

    def _compute_teacher_logps(
        self,
        student_logits: torch.Tensor | None,
        teacher_prompt_ids: torch.Tensor,
        teacher_prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            teacher_logits, teacher_logps = self._compute_completion_logits_and_logps(
                self.teacher_model,
                teacher_prompt_ids,
                teacher_prompt_mask,
                completion_ids,
                completion_mask,
            )

            if self.teacher_update in {"interp", "ema+interp"} and self.student_target_weight > 0.0:
                if student_logits is None:
                    raise ValueError("student_logits are required when teacher interpolation is enabled.")
                blended_teacher_logits = self._blend_logits(
                    teacher_logits,
                    student_logits.detach(),
                    self.student_target_weight,
                )
                teacher_logps = log_softmax(blended_teacher_logits, dim=-1)

        return teacher_logps

    def _left_pad_branch_tensor(self, tensor: torch.Tensor, target_length: int, pad_value: int) -> torch.Tensor:
        current_length = tensor.size(1)
        if current_length >= target_length:
            return tensor
        return pad(tensor, (target_length - current_length, 0), value=pad_value)

    def _compute_teacher_logps_for_active_branches(
        self,
        student_logits: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        reverse_teacher_prompt_ids: torch.Tensor,
        reverse_teacher_prompt_mask: torch.Tensor,
        forward_teacher_prompt_ids: torch.Tensor,
        forward_teacher_prompt_mask: torch.Tensor,
        *,
        backward_active: bool,
        forward_active: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not backward_active and not forward_active:
            return None, None

        if (
            backward_active
            and forward_active
            and forward_teacher_prompt_ids is reverse_teacher_prompt_ids
            and forward_teacher_prompt_mask is reverse_teacher_prompt_mask
        ):
            shared_teacher_logps = self._compute_teacher_logps(
                student_logits,
                reverse_teacher_prompt_ids,
                reverse_teacher_prompt_mask,
                completion_ids,
                completion_mask,
            )
            return shared_teacher_logps, shared_teacher_logps

        branch_specs: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        if backward_active:
            branch_specs.append(("reverse", reverse_teacher_prompt_ids, reverse_teacher_prompt_mask))
        if forward_active:
            branch_specs.append(("forward", forward_teacher_prompt_ids, forward_teacher_prompt_mask))

        if len(branch_specs) == 1:
            branch_name, teacher_prompt_ids, teacher_prompt_mask = branch_specs[0]
            teacher_logps = self._compute_teacher_logps(
                student_logits,
                teacher_prompt_ids,
                teacher_prompt_mask,
                completion_ids,
                completion_mask,
            )
            if branch_name == "reverse":
                return teacher_logps, None
            return None, teacher_logps

        pad_token_id = self.processing_class.pad_token_id or self.processing_class.eos_token_id
        max_prompt_length = max(teacher_prompt_ids.size(1) for _, teacher_prompt_ids, _ in branch_specs)
        batched_teacher_prompt_ids = torch.cat(
            [
                self._left_pad_branch_tensor(teacher_prompt_ids, max_prompt_length, pad_token_id)
                for _, teacher_prompt_ids, _ in branch_specs
            ],
            dim=0,
        )
        batched_teacher_prompt_mask = torch.cat(
            [
                self._left_pad_branch_tensor(teacher_prompt_mask, max_prompt_length, 0)
                for _, _, teacher_prompt_mask in branch_specs
            ],
            dim=0,
        )
        batched_completion_ids = torch.cat([completion_ids] * len(branch_specs), dim=0)
        batched_completion_mask = torch.cat([completion_mask] * len(branch_specs), dim=0)
        student_logits_for_teacher = (
            student_logits.repeat(len(branch_specs), 1, 1)
            if self.teacher_update in {"interp", "ema+interp"} and self.student_target_weight > 0.0
            else None
        )

        # Evaluate both teacher branches together to avoid duplicate model launches.
        batched_teacher_logps = self._compute_teacher_logps(
            student_logits_for_teacher,
            batched_teacher_prompt_ids,
            batched_teacher_prompt_mask,
            batched_completion_ids,
            batched_completion_mask,
        )

        branch_batch_size = completion_ids.size(0)
        reverse_teacher_logps: torch.Tensor | None = None
        forward_teacher_logps: torch.Tensor | None = None
        for (branch_name, _, _), branch_teacher_logps in zip(
            branch_specs,
            batched_teacher_logps.split(branch_batch_size, dim=0),
        ):
            if branch_name == "reverse":
                reverse_teacher_logps = branch_teacher_logps
            else:
                forward_teacher_logps = branch_teacher_logps

        return reverse_teacher_logps, forward_teacher_logps

    def _compute_feedback_weight_tensors(
        self,
        inputs: dict[str, Any],
        student_logps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = student_logps.size(0)
        device = student_logps.device
        dtype = student_logps.dtype

        if self.feedback_weight_type == "fixed":
            backward_weights = torch.full(
                (batch_size, 1, 1),
                self.backward_weight,
                device=device,
                dtype=dtype,
            )
            forward_weights = torch.full(
                (batch_size, 1, 1),
                self.forward_weight,
                device=device,
                dtype=dtype,
            )
            return backward_weights, forward_weights

        completion_texts = inputs.get("completion_texts")
        required_keywords = inputs.get("required_keywords", inputs.get("allowed_keywords"))
        restricted_keywords = inputs.get("restricted_keywords", inputs.get("disallowed_keywords"))
        if completion_texts is None or required_keywords is None or restricted_keywords is None:
            raise ValueError(
                "Adaptive feedback weights require completion_texts and required/restricted keyword metadata."
            )

        sample_metrics = contextual_integrity_reward_metrics(
            completions=completion_texts,
            required_keywords=required_keywords,
            restricted_keywords=restricted_keywords,
        )
        backward_weights = torch.tensor(
            [1.0 - float(metric["integrity"]) for metric in sample_metrics],
            device=device,
            dtype=dtype,
        ).view(batch_size, 1, 1)
        forward_weights = torch.tensor(
            [1.0 - float(metric["utility"]) for metric in sample_metrics],
            device=device,
            dtype=dtype,
        ).view(batch_size, 1, 1)
        return backward_weights, forward_weights

    def _compute_student_teacher_kl_per_token(
        self,
        student_logps: torch.Tensor,
        teacher_logps: torch.Tensor,
        completion_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.kl_support == "full":
            return kl_div(teacher_logps, student_logps, reduction="none", log_target=True).sum(dim=-1)

        if self.kl_support == "topk":
            topk = min(self.kl_topk, student_logps.size(-1))
            topk_indices = torch.topk(student_logps, k=topk, dim=-1).indices
            student_topk_logps = torch.gather(student_logps, dim=-1, index=topk_indices)
            teacher_topk_logps = torch.gather(teacher_logps, dim=-1, index=topk_indices)

            student_topk_logps = student_topk_logps - torch.logsumexp(
                student_topk_logps,
                dim=-1,
                keepdim=True,
            )
            teacher_topk_logps = teacher_topk_logps - torch.logsumexp(
                teacher_topk_logps,
                dim=-1,
                keepdim=True,
            )
            return kl_div(teacher_topk_logps, student_topk_logps, reduction="none", log_target=True).sum(dim=-1)

        sampled_token_indices = completion_ids.unsqueeze(-1)
        student_token_logps = torch.gather(student_logps, dim=-1, index=sampled_token_indices).squeeze(-1)
        teacher_token_logps = torch.gather(teacher_logps, dim=-1, index=sampled_token_indices).squeeze(-1)
        return student_token_logps - teacher_token_logps

    def _compute_weighted_bidirectional_kl(
        self,
        student_logps: torch.Tensor,
        completion_ids: torch.Tensor,
        backward_teacher_logps: torch.Tensor | None,
        forward_teacher_logps: torch.Tensor | None,
        backward_token_weights: torch.Tensor,
        forward_token_weights: torch.Tensor,
    ) -> torch.Tensor:
        branch_teacher_logps: list[torch.Tensor] = []
        branch_token_weights: list[torch.Tensor] = []

        if backward_teacher_logps is not None:
            branch_teacher_logps.append(backward_teacher_logps)
            branch_token_weights.append(backward_token_weights)
        if forward_teacher_logps is not None:
            branch_teacher_logps.append(forward_teacher_logps)
            branch_token_weights.append(forward_token_weights)
        if not branch_teacher_logps:
            return torch.zeros_like(student_logps[..., 0])

        if len(branch_teacher_logps) == 1:
            branch_kls = self._compute_student_teacher_kl_per_token(
                student_logps,
                branch_teacher_logps[0],
                completion_ids,
            ).unsqueeze(0)
        else:
            branch_kls = self._compute_student_teacher_kl_per_token(
                student_logps.unsqueeze(0).expand(len(branch_teacher_logps), -1, -1, -1),
                torch.stack(branch_teacher_logps, dim=0),
                completion_ids.unsqueeze(0).expand(len(branch_teacher_logps), -1, -1),
            )

        stacked_token_weights = torch.stack(branch_token_weights, dim=0)
        kl_loss = (stacked_token_weights * branch_kls).sum(dim=0)
        combined_weights = stacked_token_weights.sum(dim=0)
        return torch.where(
            combined_weights > 0,
            kl_loss / combined_weights.clamp(min=1.0),
            torch.zeros_like(kl_loss),
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        del kwargs  # Unused, maintained for HF Trainer signature compatibility.

        prompt_ids = inputs["prompt_ids"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        forward_teacher_prompt_ids = inputs.get("forward_teacher_prompt_ids", inputs["teacher_prompt_ids"])
        forward_teacher_prompt_mask = inputs.get("forward_teacher_prompt_mask", inputs["teacher_prompt_mask"])
        reverse_teacher_prompt_ids = inputs.get("reverse_teacher_prompt_ids", inputs["teacher_prompt_ids"])
        reverse_teacher_prompt_mask = inputs.get("reverse_teacher_prompt_mask", inputs["teacher_prompt_mask"])

        student_input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        student_attention_mask = torch.cat([inputs["prompt_mask"], completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        student_outputs = model(input_ids=student_input_ids, attention_mask=student_attention_mask)
        student_logits = student_outputs.logits[:, -logits_to_keep - 1 : -1, :]
        student_logps = log_softmax(student_logits, dim=-1)
        backward_weights, forward_weights = self._compute_feedback_weight_tensors(inputs, student_logps)
        backward_active = bool(torch.any(backward_weights > 0).item())
        forward_active = bool(torch.any(forward_weights > 0).item())

        reverse_teacher_logps, forward_teacher_logps = self._compute_teacher_logps_for_active_branches(
            student_logits,
            completion_ids,
            completion_mask,
            reverse_teacher_prompt_ids,
            reverse_teacher_prompt_mask,
            forward_teacher_prompt_ids,
            forward_teacher_prompt_mask,
            backward_active=backward_active,
            forward_active=forward_active,
        )

        backward_token_weights = backward_weights.squeeze(-1)
        forward_token_weights = forward_weights.squeeze(-1)
        if backward_active:
            if reverse_teacher_logps is None:
                raise RuntimeError("Reverse-KL teacher logits were not computed.")
        if forward_active:
            if forward_teacher_logps is None:
                raise RuntimeError("Allowed-feedback teacher logits were not computed.")
        kl_loss = self._compute_weighted_bidirectional_kl(
            student_logps,
            completion_ids,
            reverse_teacher_logps,
            forward_teacher_logps,
            backward_token_weights,
            forward_token_weights,
        )

        mask = completion_mask.to(dtype=kl_loss.dtype)
        valid_tokens = mask.sum(-1).clamp(min=1.0)
        loss = ((kl_loss * mask).sum(-1) / valid_tokens).mean()

        if return_outputs:
            return loss, {
                "student_logits": student_logits,
                "backward_weights": backward_weights.squeeze(-1).squeeze(-1),
                "forward_weights": forward_weights.squeeze(-1).squeeze(-1),
            }
        return loss


MultiGPUDualFeedbackTrainerV2 = SelfCITrainer
