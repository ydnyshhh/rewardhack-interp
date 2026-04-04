from __future__ import annotations

from pathlib import Path
from typing import Any

from rewardhack_interp.activations import load_activation_artifact, load_activation_tensors
from rewardhack_interp.analysis.features import pool_tensor
from rewardhack_interp.config import LogitLensConfig
from rewardhack_interp.io import write_json
from rewardhack_interp.modeling import load_qwen_model
from rewardhack_interp.tracking import start_wandb_run
from rewardhack_interp.types import LogitLensArtifact, LogitLensCandidate, LogitLensLayerResult


def run_logit_lens(config: LogitLensConfig) -> LogitLensArtifact:
    import torch

    with start_wandb_run(
        wandb_config=config.wandb,
        run_name=Path(config.output_path).stem,
        job_type="logit_lens",
        config_payload=config.model_dump(mode="json"),
    ) as tracker:
        activation_artifact = load_activation_artifact(config.activation_artifact_path)
        tensor_bundle = load_activation_tensors(activation_artifact)
        model_bundle = load_qwen_model(config.model)
        model = model_bundle.model
        tokenizer = model_bundle.tokenizer
        lm_head = model.lm_head
        final_norm = resolve_final_norm(model)
        selected_tensor_names = config.tensor_names or [
            name for name in activation_artifact.tensor_names if name.startswith("hidden_state.")
        ]

        layer_results: list[LogitLensLayerResult] = []
        with torch.no_grad():
            for tensor_name in selected_tensor_names:
                vector = pool_tensor(
                    tensor_bundle[tensor_name],
                    prompt_token_count=activation_artifact.prompt_token_count,
                    completion_token_count=activation_artifact.completion_token_count,
                    strategy=config.position_strategy,
                )
                hidden = torch.as_tensor(vector, device=next(model.parameters()).device)
                hidden = hidden.unsqueeze(0)
                if final_norm is not None:
                    hidden = final_norm(hidden)
                logits = lm_head(hidden)[0]
                probabilities = torch.softmax(logits, dim=-1)
                top_probabilities, top_indices = torch.topk(probabilities, k=config.top_k)
                candidates = [
                    LogitLensCandidate(
                        token=tokenizer.decode([int(token_id)], skip_special_tokens=False),
                        token_id=int(token_id),
                        probability=float(probability.item()),
                    )
                    for probability, token_id in zip(
                        top_probabilities,
                        top_indices,
                        strict=True,
                    )
                ]
                layer_results.append(
                    LogitLensLayerResult(
                        tensor_name=tensor_name,
                        position=describe_position(
                            activation_artifact.prompt_token_count,
                            activation_artifact.completion_token_count,
                            config.position_strategy,
                        ),
                        top_tokens=candidates,
                    )
                )

        artifact = LogitLensArtifact(
            trace_id=activation_artifact.trace_id,
            position_strategy=config.position_strategy,
            layers=layer_results,
        )
        write_json(config.output_path, artifact.model_dump(mode="json"))
        tracker.log_summary(
            {
                "trace_id": artifact.trace_id,
                "num_layers": len(artifact.layers),
                "position_strategy": artifact.position_strategy,
            }
        )
        tracker.log_path(config.output_path, artifact_type="logit-lens-report")
        return artifact


def resolve_final_norm(model: Any) -> Any | None:
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    return None


def describe_position(
    prompt_token_count: int,
    completion_token_count: int,
    strategy: Any,
) -> int:
    from rewardhack_interp.types import PoolingStrategy

    if strategy == PoolingStrategy.last_completion_token:
        return prompt_token_count + max(completion_token_count - 1, 0)
    if strategy == PoolingStrategy.last_prompt_token:
        return max(prompt_token_count - 1, 0)
    return -1
