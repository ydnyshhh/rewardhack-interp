from __future__ import annotations

from pathlib import Path
from typing import Any

from rewardhack_interp.config import GRPORunConfig
from rewardhack_interp.gym_integration import build_environment
from rewardhack_interp.io import write_json
from rewardhack_interp.rl.rewards import make_reward_function
from rewardhack_interp.utils.paths import ensure_dir, to_path_string
from rewardhack_interp.utils.seeding import set_global_seed


def run_grpo(config: GRPORunConfig) -> dict[str, Any]:
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig as HFGRPOConfig
    from trl import GRPOTrainer

    set_global_seed(config.random_state)
    output_dir = ensure_dir(config.output_dir)
    environment = build_environment(config.environment)

    rows: list[dict[str, Any]] = []
    seeds = _dataset_task_seeds(config)
    for seed in seeds:
        task = environment.sample_task(seed=seed)
        rows.append(
            {
                "prompt": task.prompt,
                "task_seed": seed,
                "task_id": task.task_id,
                "family": task.family,
            }
        )
    dataset = Dataset.from_list(rows)

    processing_class = AutoTokenizer.from_pretrained(
        config.model.model_name_or_path,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
    )
    if processing_class.pad_token_id is None and processing_class.eos_token_id is not None:
        processing_class.pad_token = processing_class.eos_token

    trainer_args = HFGRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        num_generations=config.num_generations,
        beta=config.beta,
        report_to=config.report_to,
        **config.trainer_kwargs,
    )

    peft_config = None
    if config.lora is not None:
        peft_config = LoraConfig(
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=list(config.lora.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )

    trainer = GRPOTrainer(
        model=config.model.model_name_or_path,
        reward_funcs=make_reward_function(config),
        args=trainer_args,
        train_dataset=dataset,
        processing_class=processing_class,
        peft_config=peft_config,
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir))

    summary = {
        "run_name": config.run_name,
        "output_dir": to_path_string(output_dir),
        "reward_signal": config.reward_signal.value,
        "dataset_size": len(rows),
        "task_seed_start": seeds[0] if seeds else None,
        "task_seed_end": seeds[-1] if seeds else None,
        "train_metrics": dict(train_result.metrics),
    }
    write_json(Path(output_dir) / "training_summary.json", summary)
    return summary


def _dataset_task_seeds(config: GRPORunConfig) -> list[int]:
    resolved = config.environment.resolved_task_seeds()
    if len(resolved) >= config.dataset_size:
        return resolved[: config.dataset_size]
    extra_needed = config.dataset_size - len(resolved)
    next_seed = (max(resolved) + 1) if resolved else config.environment.start_seed
    return resolved + list(range(next_seed, next_seed + extra_needed))
