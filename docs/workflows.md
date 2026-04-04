# Workflows

## 1. Collect Rollouts

Create a TOML config like [`configs/examples/rollout_qwen3_4b_spec_overfit.toml`](../configs/examples/rollout_qwen3_4b_spec_overfit.toml), then run:

```bash
uv run rewardhack-interp collect-rollouts configs/examples/rollout_qwen3_4b_spec_overfit.toml
```

This produces:

- rollout JSONL with one row per sampled completion
- a summary JSON file
- no activation dump by default

If you only want a quick pilot run, use
[`configs/examples/rollout_qwen3_pilot.toml`](../configs/examples/rollout_qwen3_pilot.toml)
instead of the larger default example.

To send the run to W&B, add a `[wandb]` block to the rollout config, for example:

```toml
[wandb]
enabled = true
project = "rewardhack-interp"
entity = "your-entity"
group = "qwen3-4b-spec-overfit"
mode = "online"
```

The bundled code-task examples set `enable_thinking = false` for Qwen and use a
strict final-only system prompt so the model emits the required Python function
instead of a reasoning trace.

## 2. Summarize Rollouts

Use the rollout artifact as the first analysis object:

```bash
uv run rewardhack-interp summarize-rollouts artifacts/rollouts/qwen3_4b_spec_overfit.jsonl --output artifacts/analysis/qwen3_4b_spec_overfit_summary.json
```

This writes a reusable summary JSON plus saved plots for:

- cohort counts
- reward distributions by cohort
- verifier-gap histograms
- completion length distributions

## 3. Build a Balanced Subset

Build a reproducible `reward_hack` vs `genuine_failure` subset before any activation replay:

```bash
uv run rewardhack-interp build-rollout-subset configs/examples/subset_rewardhack_vs_failure.toml
```

The subset utility supports:

- choosing cohorts to include
- max samples per cohort
- reproducible random seeds
- optional bucket matching by completion length, official reward, and verifier gap
- a typed summary JSON saved next to the subset JSONL

## 4. Extract Case Studies

Pull a few representative examples for manual inspection:

```bash
uv run rewardhack-interp extract-case-studies configs/examples/case_studies_rewardhack_vs_failure.toml
```

Each saved example includes cohort, official/oracle reward, false-pass status, verifier gap, a prompt summary, and the completion text.

## 5. Capture Activations on the Subset

Replay only the selected subset for activation capture:

```bash
uv run rewardhack-interp capture-activations configs/examples/capture_rewardhack_vs_failure.toml
```

The default-safe capture config:

- reads the subset JSONL directly
- saves hidden states only
- stores only a configurable layer subset
- stores a compact token representation by default (`last_completion_token`)
- writes a manifest JSONL that maps stable sample ids to tensor files

This is the preferred storage-safe path for 40-60 example analyses.
On Runpod, it is best to run this replay step in the same environment where the model
weights are already cached. On a fresh machine, set `HF_TOKEN` and make sure
`Qwen/Qwen3-4B` is accessible before replay capture.

## 6. Run the First Real Probe

Run the canonical first mechanistic analysis:

```bash
uv run rewardhack-interp run-layerwise-probe configs/examples/probe_rewardhack_vs_failure.toml
```

This layerwise binary probe asks:

- at which layers are `reward_hack` and `genuine_failure` separable?

The report writes per-layer metrics plus saved accuracy and AUROC plots.

## 7. Train a Probe

After capturing activations, run:

```bash
uv run rewardhack-interp train-probe configs/examples/probe_false_pass_vs_true_pass.toml
```

This trains a logistic probe on a chosen activation target and pooling strategy.
For the subset-first workflow above, prefer
[`configs/examples/probe_rewardhack_vs_failure.toml`](../configs/examples/probe_rewardhack_vs_failure.toml).
The legacy single-target probe example remains available for ad hoc activation manifests.

## 8. Compare Representations

Create a representation config pointing at an activation manifest and target names, then run:

```bash
uv run rewardhack-interp compare-representations path/to/representations.toml
uv run rewardhack-interp cluster-activations path/to/clustering.toml
```

These outputs help answer where reward-hack vs true-success cohorts become separable.

## 9. Logit Lens

Point a config at one activation metadata JSON file:

```bash
uv run rewardhack-interp logit-lens path/to/logit_lens.toml
```

This projects saved hidden states back through the LM head to inspect token preferences layer by layer.

## 10. Causal Patching

Choose matched source/donor trajectories plus their activation files:

```bash
uv run rewardhack-interp patch-activations path/to/patch.toml
```

The current intervention path patches prompt-boundary activations for selected layers or modules and then regenerates from the source prompt.

## 11. GRPO Training

Use a config like [`configs/examples/grpo_weak_reward.toml`](../configs/examples/grpo_weak_reward.toml):

```bash
uv run rewardhack-interp train-grpo configs/examples/grpo_weak_reward.toml
```

Reward modes:

- `official`
- `oracle`
- `gap_aware`
- `anti_hack`

GRPO runs also support W&B through the same `[wandb]` config block.

The main GRPO example now includes:

- Qwen 3 4B as the default first-pass training model
- a larger training slice instead of a smoke-test-sized run
- an explicit held-out evaluation seed block
- post-training evaluation across the base model, saved checkpoints, and final policy

The held-out evaluation artifact reports:

- mean official reward
- mean oracle reward
- mean verifier gap
- false-pass rate
- cohort counts

If you want a smaller pilot pass instead, use
[`configs/examples/grpo_weak_reward_pilot.toml`](../configs/examples/grpo_weak_reward_pilot.toml).
The GRPO examples share the same non-thinking, code-only prompt contract so
training and held-out evaluation follow the same interface.

## 12. Compare Checkpoints

Run the same environment seeds across multiple checkpoints:

```bash
uv run rewardhack-interp compare-checkpoints path/to/checkpoints.toml
```

The output summarizes official reward, oracle reward, verifier gap, false-pass rate, and cohort counts for each checkpoint.

## 13. Experiment 1

This is the first full separation experiment:

- environment: `code/patch-verification`
- model: `Qwen/Qwen3-4B`
- profile: `high`
- question: are false passes internally separable from true passes?

Step 1, collect rollouts and activations:

```bash
uv run rewardhack-interp collect-rollouts configs/examples/experiment1_rollout_patch_verification.toml
```

Step 2, run the layerwise analysis:

```bash
uv run rewardhack-interp run-experiment-1 configs/examples/experiment1_patch_verification.toml
```

The experiment runner writes:

- a report JSON with layerwise probe, centroid-distance, CKA, and clustering results
- a probe-accuracy-by-layer plot
- a matched-pairs JSONL for true-pass vs false-pass comparisons
- cohort counts, scenario slices, semantic-failure summaries, and false-pass exploit label summaries
- the split protocol, explicit train/eval seed partitions when configured, repeated split seeds, and confidence interval summaries for probe metrics

If you want a smaller pilot version first, use:

```bash
uv run rewardhack-interp collect-rollouts configs/examples/experiment1_rollout_patch_verification_pilot.toml
uv run rewardhack-interp run-experiment-1 configs/examples/experiment1_patch_verification_pilot.toml
```

The default comparisons are:

- `true_pass_vs_false_pass`
- `true_pass_vs_clean_failure`
- `false_pass_vs_clean_failure`

The main Experiment 1 analysis config uses explicit seed-level holdout:

- training task seeds: `0..191`
- held-out evaluation task seeds: `192..255`
- split strategy: `task_seed_holdout`
- confidence intervals: bootstrap over the held-out evaluation slice

That keeps rows from the same environment seed out of both train and test, which is a better first-pass check of whether the representation difference transfers across task instances rather than only across completions.
The pilot rollout config is deterministic now (`num_completions = 1`, `do_sample = false`)
for interface debugging, while the main rollout keeps low-temperature sampling
for cohort construction.

## Runpod Notes

- Generated artifacts live under `artifacts/` inside the repo checkout on the execution machine.
- Rollout JSONL rows keep nested fields such as `reward_metrics`, `generation`, and `task`; downstream tools in this repo read those nested fields directly.
- Files created under `/workspace` on Runpod are not automatically committed back to Git. Copy or sync back the artifacts you want to keep.
