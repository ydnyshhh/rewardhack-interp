# Workflows

## 1. Collect Rollouts

Create a TOML config like [`configs/examples/rollout_qwen3.toml`](/D:/rewardhack-interp/configs/examples/rollout_qwen3.toml), then run:

```bash
uv run rewardhack-interp collect-rollouts --config configs/examples/rollout_qwen3.toml
```

This produces:

- rollout JSONL with one row per sampled completion
- a summary JSON file
- optional activation tensor bundles plus an activation manifest

To send the run to W&B, add a `[wandb]` block to the rollout config, for example:

```toml
[wandb]
enabled = true
project = "rewardhack-interp"
entity = "your-entity"
group = "qwen3-spec-overfit"
mode = "online"
```

## 2. Train a Probe

After capturing activations, run:

```bash
uv run rewardhack-interp train-probe --config configs/examples/probe_false_pass_vs_true_pass.toml
```

This trains a logistic probe on a chosen activation target and pooling strategy.

## 3. Compare Representations

Create a representation config pointing at an activation manifest and target names, then run:

```bash
uv run rewardhack-interp compare-representations --config path/to/representations.toml
uv run rewardhack-interp cluster-activations --config path/to/clustering.toml
```

These outputs help answer where reward-hack vs true-success cohorts become separable.

## 4. Logit Lens

Point a config at one activation metadata JSON file:

```bash
uv run rewardhack-interp logit-lens --config path/to/logit_lens.toml
```

This projects saved hidden states back through the LM head to inspect token preferences layer by layer.

## 5. Causal Patching

Choose matched source/donor trajectories plus their activation files:

```bash
uv run rewardhack-interp patch-activations --config path/to/patch.toml
```

The current intervention path patches prompt-boundary activations for selected layers or modules and then regenerates from the source prompt.

## 6. GRPO Training

Use a config like [`configs/examples/grpo_weak_reward.toml`](/D:/rewardhack-interp/configs/examples/grpo_weak_reward.toml):

```bash
uv run rewardhack-interp train-grpo --config configs/examples/grpo_weak_reward.toml
```

Reward modes:

- `official`
- `oracle`
- `gap_aware`
- `anti_hack`

GRPO runs also support W&B through the same `[wandb]` config block.

## 7. Compare Checkpoints

Run the same environment seeds across multiple checkpoints:

```bash
uv run rewardhack-interp compare-checkpoints --config path/to/checkpoints.toml
```

The output summarizes official reward, oracle reward, verifier gap, false-pass rate, and cohort counts for each checkpoint.

## 8. Experiment 1

This is the first full separation experiment:

- environment: `code/patch-verification`
- model: `Qwen/Qwen3-4B-Instruct`
- profile: `high`
- question: are false passes internally separable from true passes?

Step 1, collect rollouts and activations:

```bash
uv run rewardhack-interp collect-rollouts --config configs/examples/experiment1_rollout_patch_verification.toml
```

Step 2, run the layerwise analysis:

```bash
uv run rewardhack-interp run-experiment-1 --config configs/examples/experiment1_patch_verification.toml
```

The experiment runner writes:

- a report JSON with layerwise probe, centroid-distance, CKA, and clustering results
- a probe-accuracy-by-layer plot
- a matched-pairs JSONL for true-pass vs false-pass comparisons
- cohort counts, scenario slices, semantic-failure summaries, and false-pass exploit label summaries

The default comparisons are:

- `true_pass_vs_false_pass`
- `true_pass_vs_clean_failure`
- `false_pass_vs_clean_failure`
