from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rewardhack_interp.checkpoints import build_checkpoint_summary
from rewardhack_interp.config import GRPORunConfig, ModelConfig, RolloutConfig
from rewardhack_interp.gym_integration import build_environment
from rewardhack_interp.io import load_models, write_json
from rewardhack_interp.rl.rewards import make_reward_function
from rewardhack_interp.rollouts import collect_rollouts
from rewardhack_interp.tracking import ensure_wandb_report_to, start_wandb_run
from rewardhack_interp.types import (
    GRPOHeldoutEvaluationArtifact,
    GRPOTrainingArtifact,
    TrajectoryArtifact,
)
from rewardhack_interp.utils.paths import ensure_dir, to_path_string
from rewardhack_interp.utils.seeding import set_global_seed


@dataclass(frozen=True, slots=True)
class GRPOEvaluationTarget:
    model_config: ModelConfig
    phase_label: str
    global_step: int | None
    checkpoint_path: Path | None
    rollout_path: Path


def run_grpo(config: GRPORunConfig) -> GRPOTrainingArtifact:
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig as HFGRPOConfig
    from trl import GRPOTrainer

    set_global_seed(config.random_state)
    output_dir = ensure_dir(config.output_dir)
    environment = build_environment(config.environment)
    if config.reward_trace_output is not None:
        reward_trace_path = Path(config.reward_trace_output)
        if reward_trace_path.exists():
            reward_trace_path.unlink()

    rows: list[dict[str, Any]] = []
    seeds = dataset_task_seeds(config)
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

    report_to = ensure_wandb_report_to(config.report_to, config.wandb)
    with start_wandb_run(
        wandb_config=config.wandb,
        run_name=config.run_name,
        job_type="grpo_training",
        config_payload=config.model_dump(mode="json"),
    ) as tracker:
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
            report_to=report_to,
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

        heldout_evaluation = (
            evaluate_grpo_saved_policies(config) if config.evaluation is not None else None
        )
        artifact = GRPOTrainingArtifact(
            run_name=config.run_name,
            output_dir=to_path_string(output_dir),
            reward_signal=config.reward_signal,
            dataset_size=len(rows),
            training_environment_name=config.environment.name,
            training_environment_profile=config.environment.profile,
            training_task_seeds=seeds,
            train_metrics=dict(train_result.metrics),
            reward_trace_output=(
                to_path_string(config.reward_trace_output)
                if config.reward_trace_output is not None
                else None
            ),
            heldout_evaluation_path=(
                to_path_string(config.evaluation.output_path)
                if config.evaluation is not None
                else None
            ),
            heldout_evaluation=heldout_evaluation,
        )
        summary_path = Path(output_dir) / "training_summary.json"
        write_json(summary_path, artifact.model_dump(mode="json"))
        summary_payload = {
            "run_name": artifact.run_name,
            "output_dir": artifact.output_dir,
            "reward_signal": artifact.reward_signal.value,
            "dataset_size": artifact.dataset_size,
            "task_seed_start": seeds[0] if seeds else None,
            "task_seed_end": seeds[-1] if seeds else None,
            "train_metrics": artifact.train_metrics,
        }
        if heldout_evaluation is not None:
            summary_payload["heldout_num_checkpoints"] = len(heldout_evaluation.summaries)
            final_summary = next(
                (
                    summary
                    for summary in heldout_evaluation.summaries
                    if summary.phase_label == "post_training_final"
                ),
                None,
            )
            if final_summary is not None:
                summary_payload["heldout_final_official_reward"] = (
                    final_summary.mean_official_reward
                )
                summary_payload["heldout_final_oracle_reward"] = final_summary.mean_oracle_reward
                summary_payload["heldout_final_verifier_gap"] = final_summary.mean_verifier_gap
                summary_payload["heldout_final_false_pass_rate"] = final_summary.false_pass_rate
        tracker.log_summary(summary_payload)
        tracker.log_path(summary_path, artifact_type="training-summary")
        if config.evaluation is not None:
            tracker.log_path(
                config.evaluation.output_path,
                artifact_type="grpo-heldout-evaluation",
            )
        return artifact


def dataset_task_seeds(config: GRPORunConfig) -> list[int]:
    return config.resolved_training_task_seeds()


def evaluate_grpo_saved_policies(config: GRPORunConfig) -> GRPOHeldoutEvaluationArtifact:
    if config.evaluation is None:
        raise ValueError("GRPO held-out evaluation requested without an evaluation config.")

    evaluation_targets = build_grpo_evaluation_targets(config)
    summaries = []
    for target in evaluation_targets:
        collect_rollouts(
            config=build_grpo_evaluation_rollout_config(config, target),
        )
        records = load_models(target.rollout_path, TrajectoryArtifact)
        summaries.append(
            build_checkpoint_summary(
                records=records,
                model_config=target.model_config,
                rollout_path=target.rollout_path,
                checkpoint_path=target.checkpoint_path,
                phase_label=target.phase_label,
                split_name=config.evaluation.split_name,
                global_step=target.global_step,
            )
        )

    artifact = GRPOHeldoutEvaluationArtifact(
        run_name=config.run_name,
        reward_signal=config.reward_signal,
        evaluation_environment_name=config.evaluation.environment.name,
        evaluation_environment_profile=config.evaluation.environment.profile,
        evaluation_task_seeds=config.evaluation.environment.resolved_task_seeds(),
        split_name=config.evaluation.split_name,
        summaries=summaries,
    )
    write_json(config.evaluation.output_path, artifact.model_dump(mode="json"))
    return artifact


def build_grpo_evaluation_targets(config: GRPORunConfig) -> list[GRPOEvaluationTarget]:
    if config.evaluation is None:
        return []

    rollout_dir = ensure_dir(config.evaluation.per_checkpoint_rollout_dir)
    targets: list[GRPOEvaluationTarget] = []
    if config.evaluation.include_base_model:
        targets.append(
            GRPOEvaluationTarget(
                model_config=config.model,
                phase_label="pre_training",
                global_step=0,
                checkpoint_path=None,
                rollout_path=rollout_dir / "pre_training.jsonl",
            )
        )

    checkpoint_dirs = sorted(
        (
            checkpoint_dir
            for checkpoint_dir in Path(config.output_dir).iterdir()
            if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith("checkpoint-")
        ),
        key=checkpoint_global_step,
    )
    if config.evaluation.include_intermediate_checkpoints:
        for checkpoint_dir in checkpoint_dirs:
            step = checkpoint_global_step(checkpoint_dir)
            phase_label = (
                f"checkpoint_{step}" if step is not None else checkpoint_dir.name.replace("-", "_")
            )
            targets.append(
                GRPOEvaluationTarget(
                    model_config=build_checkpoint_model_config(config, checkpoint_dir),
                    phase_label=phase_label,
                    global_step=step,
                    checkpoint_path=checkpoint_dir,
                    rollout_path=rollout_dir / f"{phase_label}.jsonl",
                )
            )

    if config.evaluation.include_final_model:
        targets.append(
            GRPOEvaluationTarget(
                model_config=build_checkpoint_model_config(config, Path(config.output_dir)),
                phase_label="post_training_final",
                global_step=None,
                checkpoint_path=Path(config.output_dir),
                rollout_path=rollout_dir / "post_training_final.jsonl",
            )
        )
    return targets


def build_checkpoint_model_config(config: GRPORunConfig, checkpoint_path: Path) -> ModelConfig:
    if config.lora is not None:
        return config.model.model_copy(
            update={"adapter_name_or_path": str(checkpoint_path.resolve())}
        )
    return config.model.model_copy(
        update={
            "model_name_or_path": str(checkpoint_path.resolve()),
            "adapter_name_or_path": None,
        }
    )


def build_grpo_evaluation_rollout_config(
    config: GRPORunConfig,
    target: GRPOEvaluationTarget,
) -> RolloutConfig:
    if config.evaluation is None:
        raise ValueError("GRPO held-out evaluation requested without an evaluation config.")
    return RolloutConfig(
        run_name=f"{config.run_name}-{config.evaluation.split_name}-{target.phase_label}",
        environment=config.evaluation.environment,
        model=target.model_config,
        sampling=config.evaluation.sampling,
        output_path=target.rollout_path,
        wandb=None,
    )


def checkpoint_global_step(checkpoint_path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", checkpoint_path.name)
    if match is None:
        return 10_000_000
    return int(match.group(1))
