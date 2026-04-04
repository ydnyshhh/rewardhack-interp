from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rewardhack_interp.config import ModelConfig, SamplingConfig
from rewardhack_interp.types import FinishReason


@dataclass(slots=True)
class GeneratedSample:
    text: str
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    decoded_completion_tokens: list[str]
    token_logprobs: list[float] | None
    finish_reason: FinishReason
    rendered_prompt: str


@dataclass(slots=True)
class LoadedQwenModel:
    tokenizer: Any
    model: Any
    config: ModelConfig

    def render_prompt(self, user_prompt: str) -> str:
        if self.config.use_chat_template and getattr(self.tokenizer, "chat_template", None):
            messages: list[dict[str, str]] = []
            if self.config.system_prompt:
                messages.append({"role": "system", "content": self.config.system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=self.config.add_generation_prompt,
            )
        return user_prompt

    def generate(self, user_prompt: str, sampling: SamplingConfig) -> list[GeneratedSample]:
        rendered_prompt = self.render_prompt(user_prompt)
        encoded = self.tokenizer(
            rendered_prompt,
            add_special_tokens=not self.config.use_chat_template,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        prompt_token_ids = encoded["input_ids"][0].tolist()

        generate_kwargs: dict[str, Any] = {
            **encoded,
            "do_sample": sampling.do_sample,
            "max_new_tokens": sampling.max_new_tokens,
            "num_return_sequences": sampling.num_completions,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": sampling.repetition_penalty,
            "return_dict_in_generate": True,
            "output_scores": True,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
        }
        if sampling.top_k > 0:
            generate_kwargs["top_k"] = sampling.top_k

        generation_output = self.model.generate(
            **generate_kwargs,
        )

        scores = list(generation_output.scores)
        prompt_length = len(prompt_token_ids)
        samples: list[GeneratedSample] = []
        for sample_index, sequence in enumerate(generation_output.sequences):
            raw_completion = sequence[prompt_length:].tolist()
            trimmed_completion, finish_reason = trim_generated_ids(
                raw_completion,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            logprobs = extract_token_logprobs(scores, sample_index, trimmed_completion)
            decoded_tokens = [
                self.tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                for token_id in trimmed_completion
            ]
            samples.append(
                GeneratedSample(
                    text=self.tokenizer.decode(trimmed_completion, skip_special_tokens=True),
                    prompt_token_ids=prompt_token_ids,
                    completion_token_ids=trimmed_completion,
                    decoded_completion_tokens=decoded_tokens,
                    token_logprobs=logprobs,
                    finish_reason=finish_reason,
                    rendered_prompt=rendered_prompt,
                )
            )
        return samples


def load_qwen_model(config: ModelConfig) -> LoadedQwenModel:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model_kwargs: dict[str, Any] = {
        "revision": config.revision,
        "trust_remote_code": config.trust_remote_code,
        "torch_dtype": dtype_map[config.torch_dtype],
    }
    if config.device_map is not None:
        model_kwargs["device_map"] = config.device_map
    if config.attention_implementation is not None:
        model_kwargs["attn_implementation"] = config.attention_implementation

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **model_kwargs)
    if config.adapter_name_or_path:
        model = PeftModel.from_pretrained(model, config.adapter_name_or_path)
    model.eval()
    return LoadedQwenModel(tokenizer=tokenizer, model=model, config=config)


def trim_generated_ids(
    token_ids: list[int],
    *,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> tuple[list[int], FinishReason]:
    trimmed: list[int] = []
    finish_reason = FinishReason.length
    for token_id in token_ids:
        if eos_token_id is not None and token_id == eos_token_id:
            finish_reason = FinishReason.eos
            break
        if pad_token_id is not None and token_id == pad_token_id:
            finish_reason = FinishReason.stop
            break
        trimmed.append(token_id)
    return trimmed, finish_reason


def extract_token_logprobs(
    score_tensors: list[Any],
    sample_index: int,
    completion_token_ids: list[int],
) -> list[float] | None:
    if not score_tensors or not completion_token_ids:
        return None
    logprobs: list[float] = []
    for step_index, token_id in enumerate(completion_token_ids):
        if step_index >= len(score_tensors):
            break
        step_scores = score_tensors[step_index][sample_index]
        step_logprobs = step_scores.log_softmax(dim=-1)
        logprobs.append(float(step_logprobs[token_id].detach().cpu().item()))
    return logprobs
