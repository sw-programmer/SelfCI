from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from peft import load_peft_weights, set_peft_model_state_dict
from torch.nn.functional import kl_div, log_softmax
from transformers import AutoModelForCausalLM, PreTrainedModel
from trl.trainer.base_trainer import BaseTrainer

from rl.rewards import EVAL_METRIC_NAMES, contextual_integrity_reward_metrics

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

class MultiGPUOnPolicyTrainer(BaseTrainer):
    def __init__(
        self,
        model: PreTrainedModel,
        args,
        train_dataset,
        processing_class,
        loss_type: str = "reverse_kl",
        beta: float = 0.5,
        teacher_model_path: str | None = None,
        teacher_model: PreTrainedModel | None = None,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            processing_class=processing_class,
            **kwargs
        )
        self.loss_type = loss_type
        self.beta = beta
        self.teacher_model_is_local = False

        if teacher_model is None:
            if not teacher_model_path:
                raise ValueError("Either teacher_model_path or teacher_model must be provided.")
            teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_model_path,
                torch_dtype=model.dtype,
                trust_remote_code=getattr(args, "trust_remote_code", True),
            )

        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

        # Prepare teacher model for distributed environments (DeepSpeed/FSDP wrapping)
        self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

        import os

        dist_env_vars = ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_PORT", "MASTER_ADDR"]
        backup_env = {k: os.environ[k] for k in dist_env_vars if k in os.environ}
        backup_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")

        for k in dist_env_vars:
            os.environ.pop(k, None)

        local_rank_str = str(self.accelerator.local_process_index)
        if backup_cvd is not None:
            cvd_list = backup_cvd.split(",")
            if int(local_rank_str) < len(cvd_list):
                os.environ["CUDA_VISIBLE_DEVICES"] = cvd_list[int(local_rank_str)]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = local_rank_str

        try:
            from vllm import LLM

            # Initialize vLLM engine in colocate mode on the training GPUs.
            self.vllm_tensor_parallel_size = getattr(args, "vllm_tensor_parallel_size", 1)
            llm_kwargs = {
                "model": model.config._name_or_path,
                "tensor_parallel_size": self.vllm_tensor_parallel_size,
                "gpu_memory_utilization": getattr(args, "vllm_gpu_memory_utilization", 0.4),
                "max_model_len": 16384,
            }
            max_model_len = getattr(args, "max_model_len", None)
            if max_model_len is not None:
                llm_kwargs["max_model_len"] = max_model_len
            self.llm = LLM(
                **llm_kwargs,
            )
        finally:
            for k, v in backup_env.items():
                os.environ[k] = v
            if backup_cvd is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = backup_cvd
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        self._last_loaded_step = -1

    def _unwrap_student_model(self) -> PreTrainedModel:
        return self.accelerator.unwrap_model(self.model)

    def _unwrap_teacher_model(self) -> PreTrainedModel:
        if self.teacher_model_is_local:
            return self.teacher_model
        return self.accelerator.unwrap_model(self.teacher_model)

    def _get_active_adapter_name(self, model: PreTrainedModel) -> str:
        if hasattr(model, "active_adapters"):
            active_adapters = getattr(model, "active_adapters")
            if active_adapters:
                return active_adapters[0]
        if hasattr(model, "active_adapter"):
            active_adapter = getattr(model, "active_adapter")
            if isinstance(active_adapter, str) and active_adapter:
                return active_adapter
        return "default"

    def _save_peft_adapter(self, model: PreTrainedModel, output_dir: str, adapter_subdir: str) -> str:
        if not hasattr(model, "save_pretrained") or not hasattr(model, "peft_config"):
            raise TypeError("Adapter persistence requires a PEFT model.")

        adapter_dir = Path(output_dir) / adapter_subdir
        adapter_dir.mkdir(parents=True, exist_ok=True)
        adapter_name = self._get_active_adapter_name(model)
        model.save_pretrained(
            str(adapter_dir),
            safe_serialization=getattr(self.args, "save_safetensors", True),
            selected_adapters=[adapter_name],
            save_embedding_layers="auto",
        )
        return str(adapter_dir)

    def _load_peft_adapter(
        self,
        model: PreTrainedModel,
        checkpoint_dir: str,
        *,
        adapter_subdir: str,
        strict: bool = False,
    ) -> str | None:
        if not hasattr(model, "peft_config"):
            raise TypeError("Adapter restore requires a PEFT model.")

        adapter_dir = Path(checkpoint_dir) / adapter_subdir
        if not adapter_dir.is_dir():
            if strict:
                raise ValueError(f"Expected adapter checkpoint at '{adapter_dir}', but it does not exist.")
            return None

        adapter_name = self._get_active_adapter_name(model)
        adapter_state_dict = load_peft_weights(str(adapter_dir), device="cpu")
        set_peft_model_state_dict(model, adapter_state_dict, adapter_name=adapter_name)
        model.eval()
        return str(adapter_dir)

    def _iter_trainable_student_teacher_parameter_pairs(self):
        student_model = self._unwrap_student_model()
        teacher_model = self._unwrap_teacher_model()
        teacher_params = dict(teacher_model.named_parameters())

        for name, student_param in student_model.named_parameters():
            if not student_param.requires_grad:
                continue
            teacher_param = teacher_params.get(name)
            if teacher_param is None:
                raise KeyError(f"Teacher parameter '{name}' is missing from the EMA teacher model.")
            yield name, student_param, teacher_param

    def _update_teacher_model_ema(self, decay: float) -> bool:
        if not 0.0 <= decay <= 1.0:
            raise ValueError("EMA decay must be in the interval [0, 1].")
        if getattr(self.accelerator, "optimizer_step_was_skipped", False):
            return False

        with torch.no_grad():
            for _, student_param, teacher_param in self._iter_trainable_student_teacher_parameter_pairs():
                teacher_param.lerp_(
                    student_param.detach().to(device=teacher_param.device, dtype=teacher_param.dtype),
                    1.0 - decay,
                )

        return True

    def _save_teacher_adapter(self, output_dir: str) -> str:
        teacher_model = self._unwrap_teacher_model()
        return self._save_peft_adapter(teacher_model, output_dir, "ema_teacher")

    def _load_teacher_adapter(self, checkpoint_dir: str, *, strict: bool = False) -> str | None:
        teacher_model = self._unwrap_teacher_model()
        return self._load_peft_adapter(
            teacher_model,
            checkpoint_dir,
            adapter_subdir="ema_teacher",
            strict=strict,
        )

    def _sync_student_weights_to_vllm(self):
        """
        Memory-efficient weight synchronization (latency-optimized).
        Updates vLLM weights without OOM in DeepSpeed ZeRO-3 or FSDP setups.
        """
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

        if zero_stage_3:
            import deepspeed
            gather_context = deepspeed.zero.GatheredParameters
        else:
            gather_context = nullcontext

        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model

        # Check whether this is a PEFT (LoRA) model
        is_peft = hasattr(self.model, "peft_config")

        if is_peft:
            from peft.tuners.lora.layer import LoraLayer

            # 1. Iterate through modules to find LoRA layers
            for name, module in self.model.named_modules():
                if isinstance(module, LoraLayer):
                    # Restore to the original HF parameter name that vLLM can recognize
                    # e.g., base_model.model.model.layers.0... -> model.layers.0...weight
                    vllm_name = name.replace("base_model.model.", "") + ".weight"

                    adapter_name = module.active_adapters[0] if hasattr(module, "active_adapters") else "default"

                    base_param = module.base_layer.weight
                    lora_a_param = module.lora_A[adapter_name].weight
                    lora_b_param = module.lora_B[adapter_name].weight
                    scaling = module.scaling[adapter_name]

                    # 2. Gather parameters (e.g., for DeepSpeed ZeRO-3) and merge in memory
                    with gather_context([base_param, lora_a_param, lora_b_param]):
                        # W_new = W_base + (B @ A) * scaling
                        merged_weight = base_param.data + (lora_b_param.data @ lora_a_param.data) * scaling

                        # 3. Inject the merged tensor into vLLM (immediately in-loop to avoid memory spikes)
                        llm_model.load_weights([(vllm_name, merged_weight)])
        else:
            # Existing base model logic
            for name, param in self.model.named_parameters():
                vllm_name = name.replace("_checkpoint_wrapped_module.", "")
                with gather_context([param]):
                    llm_model.load_weights([(vllm_name, param.data)])

        # Invalidate previous KV cache because weights have changed
        self.llm.reset_prefix_cache()

    def _sample_output_to_text(self, sample_output: Any) -> str:
        if hasattr(sample_output, "text") and isinstance(sample_output.text, str):
            return sample_output.text

        token_ids = getattr(sample_output, "token_ids", None)
        if token_ids is not None:
            return self.processing_class.decode(token_ids, skip_special_tokens=False)

        return str(sample_output)

    @staticmethod
    def _repair_completion_text_for_eval(completion_text: str) -> str:
        text = str(completion_text or "")
        text_lower = text.lower()
        if "<answer>" in text_lower and "</answer>" not in text_lower:
            return text.rstrip() + "\n</answer>"
        return text

    def _sample_output_token_count(self, sample_output: Any) -> float:
        token_ids = getattr(sample_output, "token_ids", None)
        if token_ids is not None:
            return float(len(token_ids))

        text = self._sample_output_to_text(sample_output)
        if not text:
            return 0.0

        encoded = self.processing_class(
            text,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        input_ids = encoded.get("input_ids", [])
        if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
            return float(len(input_ids[0]))
        return float(len(input_ids))

    def _sample_model_outputs(self, prompts: list[str], num_generations: int = 1):
        """
        Sample completions from the current student with vLLM.
        """
        if self.state.global_step != self._last_loaded_step:
            self._sync_student_weights_to_vllm()
            self._last_loaded_step = self.state.global_step

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=getattr(self.args, "temperature", 0.7),
            max_tokens=getattr(self.args, "max_completion_length", 2048),
            n=max(1, int(num_generations)),
            seed=getattr(self.args, "vllm_seed", 42),
        )

        return self.llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

    def _prepare_inputs(self, generation_batch: dict[str, Any]) -> dict[str, Any]:
        """
        Run on-policy rollouts with vLLM from a dataloader batch.
        """
        prompts = generation_batch["prompt"]
        outputs = self._sample_model_outputs(prompts, num_generations=1)

        completion_ids: list[list[int]] = []
        completion_texts: list[str] = []
        for out in outputs:
            first_output = out.outputs[0] if out.outputs else None
            if first_output is None:
                completion_ids.append([])
                completion_texts.append("")
                continue
            completion_ids.append(first_output.token_ids)
            completion_texts.append(self._sample_output_to_text(first_output))

        # Pad sequences (convert to Hugging Face Trainer-compatible format)
        prompt_ids, prompt_mask = self._pad_sequences([out.prompt_token_ids for out in outputs], padding_side="left")
        comp_ids, comp_mask = self._pad_sequences(completion_ids, padding_side="right")

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": comp_ids,
            "completion_mask": comp_mask,
            "completion_texts": completion_texts,
        }

    def _compute_avg_eval_metric_tensor(
        self,
        prompts: list[str],
        required_keywords: list[Any],
        restricted_keywords: list[Any],
        use_post_think_response: list[Any] | None = None,
    ) -> torch.Tensor:
        outputs = self._sample_model_outputs(
            prompts,
            num_generations=getattr(self.args, "eval_num_generations", 1),
        )
        batch_metrics: list[list[float]] = []
        if use_post_think_response is None:
            use_post_think_response = [None] * len(outputs)

        for output, required, restricted, post_think_flag in zip(
            outputs,
            required_keywords,
            restricted_keywords,
            use_post_think_response,
        ):
            sample_outputs = output.outputs if output.outputs else []
            completion_texts = [
                self._repair_completion_text_for_eval(self._sample_output_to_text(sample))
                for sample in sample_outputs
            ]
            avg_resp_token = (
                sum(self._sample_output_token_count(sample) for sample in sample_outputs) / len(sample_outputs)
                if sample_outputs
                else 0.0
            )
            if not completion_texts:
                completion_texts = [""]

            sample_metrics = contextual_integrity_reward_metrics(
                completions=completion_texts,
                required_keywords=[required] * len(completion_texts),
                restricted_keywords=[restricted] * len(completion_texts),
                use_post_think_response=[post_think_flag] * len(completion_texts),
            )
            averaged_metrics = [
                sum(metric_row[name] for metric_row in sample_metrics) / len(sample_metrics)
                for name in EVAL_METRIC_NAMES
            ]
            batch_metrics.append(averaged_metrics + [avg_resp_token])

        return torch.tensor(batch_metrics, device=self.accelerator.device, dtype=torch.float32)

    def _compute_completion_logits_and_logps(
        self,
        model: PreTrainedModel,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, -logits_to_keep - 1 : -1, :]
        return logits, log_softmax(logits, dim=-1)

    def _compute_distillation_tensor(
        self,
        student_logps: torch.Tensor,
        target_logps: torch.Tensor,
    ) -> torch.Tensor:
        if self.loss_type == "reverse_kl":
            return kl_div(target_logps, student_logps, reduction="none", log_target=True)
        if self.loss_type == "forward_kl":
            return kl_div(student_logps, target_logps, reduction="none", log_target=True)
        if self.loss_type == "jsd":
            beta_tensor = torch.tensor(self.beta, dtype=student_logps.dtype, device=student_logps.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack(
                    [
                        target_logps + torch.log(beta_tensor),
                        student_logps + torch.log(1 - beta_tensor),
                    ]
                ),
                dim=0,
            )
            kl_target = kl_div(mixture_log_probs, target_logps, reduction="none", log_target=True)
            kl_student = kl_div(mixture_log_probs, student_logps, reduction="none", log_target=True)
            return self.beta * kl_target + (1 - self.beta) * kl_student
        raise ValueError(f"Unsupported loss_type: {self.loss_type}")

    def _reduce_masked_token_loss(
        self,
        tokenwise_loss: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        per_token_loss = tokenwise_loss.sum(-1)
        mask = completion_mask.to(dtype=per_token_loss.dtype)
        valid_tokens = mask.sum(-1).clamp(min=1.0)
        return ((per_token_loss * mask).sum(-1) / valid_tokens).mean()

    @staticmethod
    def _blend_logits(
        primary_logits: torch.Tensor,
        secondary_logits: torch.Tensor,
        secondary_weight: float,
    ) -> torch.Tensor:
        secondary_weight = float(secondary_weight)
        if not 0.0 <= secondary_weight <= 1.0:
            raise ValueError("Blend weights must be in the interval [0, 1].")
        if secondary_weight <= 0.0:
            return primary_logits
        if secondary_weight >= 1.0:
            return secondary_logits
        primary_weight = 1.0 - secondary_weight
        return (primary_logits * primary_weight) + (secondary_logits * secondary_weight)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute token-level KL loss between teacher and student.
        """
        prompt_ids = inputs["prompt_ids"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Compute student log probabilities
        student_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits[:, -logits_to_keep-1:-1, :]
        student_logps = log_softmax(student_logits, dim=-1)

        # Compute teacher log probabilities (no gradients needed)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits[:, -logits_to_keep-1:-1, :]
            teacher_logps = log_softmax(teacher_logits, dim=-1)

        # Branch by loss type
        if self.loss_type == "reverse_kl":
            kl_loss = kl_div(teacher_logps, student_logps, reduction="none", log_target=True)
        elif self.loss_type == "forward_kl":
            kl_loss = kl_div(student_logps, teacher_logps, reduction="none", log_target=True)
        elif self.loss_type == "jsd":
            beta_tensor = torch.tensor(self.beta, dtype=student_logps.dtype, device=student_logps.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([
                    teacher_logps + torch.log(beta_tensor),
                    student_logps + torch.log(1 - beta_tensor)
                ]), dim=0
            )
            kl_teacher = kl_div(mixture_log_probs, teacher_logps, reduction="none", log_target=True)
            kl_student = kl_div(mixture_log_probs, student_logps, reduction="none", log_target=True)
            kl_loss = self.beta * kl_teacher + (1 - self.beta) * kl_student
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        # Apply mask to ignore padded tokens, then average
        per_token_loss = kl_loss.sum(-1)
        valid_tokens = completion_mask.sum(-1).clamp(min=1.0)
        loss = ((per_token_loss * completion_mask).sum(-1) / valid_tokens).mean()

        if return_outputs:
            return loss, {"student_logits": student_logits}

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Evaluation for on-policy batches must go through the custom KL loss path.
        """
        del ignore_keys  # Unused in custom prediction flow.
        prepared_inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, prepared_inputs)
                if prediction_loss_only:
                    return (loss.detach().mean(), None, None)

        batch_size = len(prepared_inputs["completion_texts"])
        required_keywords = inputs.get("required_keywords", inputs.get("allowed_keywords"))
        restricted_keywords = inputs.get("restricted_keywords", inputs.get("disallowed_keywords"))
        use_post_think_response = inputs.get("use_post_think_response")
        if required_keywords is None:
            required_keywords = [[] for _ in range(batch_size)]
        if restricted_keywords is None:
            restricted_keywords = [[] for _ in range(batch_size)]
        if use_post_think_response is None:
            use_post_think_response = [None] * batch_size

        metric_tensor = self._compute_avg_eval_metric_tensor(
            prompts=inputs["prompt"],
            required_keywords=required_keywords,
            restricted_keywords=restricted_keywords,
            use_post_think_response=use_post_think_response,
        )
        # HF Trainer may skip compute_metrics when labels are None.
        # We pass a dummy label tensor to ensure metric hooks run.
        return (loss.detach().mean(), metric_tensor, metric_tensor)

    def _pad_sequences(self, sequences, padding_side="right"):
        """
        Take a list of sequences and return padded tensors with attention masks.
        """
        device = self.accelerator.device
        pad_token_id = self.processing_class.pad_token_id or self.processing_class.eos_token_id

        max_len = max(len(seq) for seq in sequences)
        padded_seqs = []
        masks = []

        for seq in sequences:
            pad_len = max_len - len(seq)
            if padding_side == "right":
                padded_seqs.append(seq + [pad_token_id] * pad_len)
                masks.append([1] * len(seq) + [0] * pad_len)
            else:
                padded_seqs.append([pad_token_id] * pad_len + seq)
                masks.append([0] * pad_len + [1] * len(seq))

        return (
            torch.tensor(padded_seqs, device=device, dtype=torch.long),
            torch.tensor(masks, device=device, dtype=torch.long),
        )
