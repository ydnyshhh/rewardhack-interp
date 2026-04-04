from __future__ import annotations

from pathlib import Path

import typer

from rewardhack_interp.activations import ActivationCaptureRunner
from rewardhack_interp.analysis import (
    run_clustering,
    run_logit_lens,
    run_probe,
    run_representation_comparison,
)
from rewardhack_interp.causal import run_patch_experiment
from rewardhack_interp.checkpoints import compare_checkpoints
from rewardhack_interp.config import (
    CheckpointComparisonConfig,
    ClusteringConfig,
    ExperimentOneConfig,
    GRPORunConfig,
    LogitLensConfig,
    PatchConfig,
    ProbeConfig,
    RepresentationConfig,
    RolloutConfig,
    load_config,
)
from rewardhack_interp.experiments import run_experiment_one
from rewardhack_interp.io import append_jsonl, load_models
from rewardhack_interp.modeling import load_qwen_model
from rewardhack_interp.rl.grpo import run_grpo
from rewardhack_interp.rollouts import collect_rollouts
from rewardhack_interp.tracking import start_wandb_run
from rewardhack_interp.types import TrajectoryArtifact
from rewardhack_interp.utils.paths import ensure_parent_dir

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command("collect-rollouts")
def collect_rollouts_command(config: Path) -> None:
    """Sample rewardhack-gym tasks, run Qwen, and save labeled trajectories."""

    result = collect_rollouts(load_config(config, RolloutConfig))
    typer.echo(f"Wrote {result['rows_written']} rollout rows to {result['output_path']}")


@app.command("capture-activations")
def capture_activations_command(config: Path, rollouts: Path) -> None:
    """Replay saved rollouts and capture hidden states/module outputs."""

    rollout_config = load_config(config, RolloutConfig)
    records = load_models(rollouts, TrajectoryArtifact)
    if not rollout_config.activation_capture.enabled:
        raise typer.BadParameter("activation_capture.enabled must be true in the rollout config.")
    model_bundle = load_qwen_model(rollout_config.model)
    runner = ActivationCaptureRunner(model_bundle, rollout_config.activation_capture)
    manifest_path = rollout_config.activation_manifest_path or rollouts.with_name(
        f"{rollouts.stem}.activations.jsonl"
    )
    with start_wandb_run(
        wandb_config=rollout_config.wandb,
        run_name=f"{rollout_config.run_name}-capture-activations",
        job_type="activation_capture",
        config_payload=rollout_config.model_dump(mode="json"),
    ) as tracker:
        ensure_parent_dir(manifest_path).write_text("", encoding="utf-8")
        for record in records:
            artifact = runner.capture_from_rollout_record(record)
            append_jsonl(manifest_path, [artifact.model_dump(mode="json")])
        tracker.log_summary(
            {
                "manifest_path": str(manifest_path.resolve()),
                "num_records": len(records),
            }
        )
        tracker.log_path(manifest_path, artifact_type="activation-manifest")
    typer.echo(f"Wrote activation manifest to {manifest_path.resolve()}")


@app.command("train-probe")
def train_probe_command(config: Path) -> None:
    artifact = run_probe(load_config(config, ProbeConfig))
    typer.echo(f"Probe accuracy: {artifact.metrics.accuracy:.3f}")


@app.command("compare-representations")
def compare_representations_command(config: Path) -> None:
    artifact = run_representation_comparison(load_config(config, RepresentationConfig))
    typer.echo(f"Computed representation statistics for {artifact.target_name}")


@app.command("cluster-activations")
def cluster_activations_command(config: Path) -> None:
    artifact = run_clustering(load_config(config, ClusteringConfig))
    typer.echo(f"Clustered {artifact.target_name} into {artifact.n_clusters} clusters")


@app.command("logit-lens")
def logit_lens_command(config: Path) -> None:
    artifact = run_logit_lens(load_config(config, LogitLensConfig))
    typer.echo(f"Wrote logit-lens report for trace {artifact.trace_id}")


@app.command("patch-activations")
def patch_activations_command(config: Path) -> None:
    artifact = run_patch_experiment(load_config(config, PatchConfig))
    typer.echo(
        "Patched completion "
        f"official/oracle: "
        f"{artifact.patched_official_reward:.3f}/{artifact.patched_oracle_reward:.3f}"
    )


@app.command("train-grpo")
def train_grpo_command(config: Path) -> None:
    result = run_grpo(load_config(config, GRPORunConfig))
    typer.echo(f"Saved GRPO outputs to {result['output_dir']}")


@app.command("compare-checkpoints")
def compare_checkpoints_command(config: Path) -> None:
    artifact = compare_checkpoints(load_config(config, CheckpointComparisonConfig))
    typer.echo(f"Compared {len(artifact.summaries)} checkpoints")


@app.command("run-experiment-1")
def run_experiment_one_command(config: Path) -> None:
    artifact = run_experiment_one(load_config(config, ExperimentOneConfig))
    typer.echo(
        "Wrote Experiment 1 report to "
        f"{artifact.report_path}, plot to {artifact.plot_path}, "
        f"and matched pairs to {artifact.matched_pairs_path}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
