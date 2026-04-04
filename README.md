# rewardhack-interp

`rewardhack-interp` is a uv-only Python research project for studying reward hacking in language models through mechanistic interpretability on top of [`rewardhack-gym`](https://github.com/ydnyshhh/rewardhack-gym).

This repository does not re-implement environments. It treats `rewardhack-gym` as the substrate for:

- task sampling
- official verifier and oracle scoring
- exploit surfaces and annotations
- RL-facing reward records
- trace-oriented evaluation

The integration layer in [`src/rewardhack_interp/gym_integration.py`](src/rewardhack_interp/gym_integration.py)
is the main adapter boundary. It owns the normalized rollout record used across the repo:
`prompt`, `completion`, `env_id`, `family_id`, `task_id`, `official_reward`, `oracle_reward`,
`verifier_gap`, `false_pass`, and `exploit_labels`.

The purpose of this repo is to let you run Qwen 3 models on those tasks, collect and label trajectories, capture activations, run internal-state analyses, intervene causally, and compare checkpoints before and after GRPO-style optimization.

## Scientific Focus

The project is built around questions like:

- What internal signatures separate genuine solving from high-official / low-oracle reward hacking?
- At which layers do false passes become separable from true passes?
- How do internal representations change after training on weak verifier reward instead of oracle reward?
- Can activation patching move a reward-hacking trajectory toward a genuine solution?

## What It Supports

- Rollout collection against `rewardhack-gym` tasks with saved trajectories and direct access to `official_reward`, `oracle_reward`, `verifier_gap`, and `false_pass`
- Cohort construction for `genuine_success`, `genuine_failure`, `reward_hack`, and `oracle_only_pass`
- Hidden-state and module-output capture for Qwen 3 replayed rollouts
- Probe training, clustering, centroid-distance comparison, and linear CKA-style representation analysis
- Logit-lens inspection across saved hidden-state layers
- Prompt-boundary activation patching experiments on matched examples
- GRPO training under `official`, `oracle`, `gap_aware`, and `anti_hack` reward definitions
- Held-out checkpoint evaluation for GRPO runs across base, intermediate, and final policies
- Checkpoint comparison across base and post-training variants
- Optional Weights & Biases tracking for rollouts, activation capture, analysis jobs, checkpoint comparison, and GRPO

## Quick Start

```bash
uv sync
uv run rewardhack-interp --help
```

If you want W&B logging, configure a `[wandb]` block in the relevant TOML file and set `WANDB_API_KEY`.

Example workflows:

```bash
uv run rewardhack-interp collect-rollouts --config configs/examples/rollout_qwen3.toml
uv run rewardhack-interp train-probe --config configs/examples/probe_false_pass_vs_true_pass.toml
uv run rewardhack-interp train-grpo --config configs/examples/grpo_weak_reward.toml
```

Pilot-scale configs are available separately at
[`configs/examples/rollout_qwen3_pilot.toml`](configs/examples/rollout_qwen3_pilot.toml),
[`configs/examples/grpo_weak_reward_pilot.toml`](configs/examples/grpo_weak_reward_pilot.toml),
and
[`configs/examples/experiment1_rollout_patch_verification_pilot.toml`](configs/examples/experiment1_rollout_patch_verification_pilot.toml).
The unsuffixed example files are now the main 4B experiment paths.

First serious experiment:

```bash
uv run rewardhack-interp collect-rollouts --config configs/examples/experiment1_rollout_patch_verification.toml
uv run rewardhack-interp run-experiment-1 --config configs/examples/experiment1_patch_verification.toml
```

The Experiment 1 runner writes a JSON report, a layerwise probe plot, and a
matched-pairs JSONL derived from `rewardhack-gym`'s true-pass/false-pass pairing logic.
The report also preserves scenario-level cohort counts, semantic-failure summaries,
false-pass exploit metadata, split metadata, repeated-split seeds, and confidence
interval summaries so later slice analyses do not need to rebuild them.
The main analysis config now declares an explicit seed-level protocol with
`train_task_seeds = 0..191` and `eval_task_seeds = 192..255`; the probe stage
holds out entire task seeds instead of randomly mixing rows from the same seed
across train and test.
The main Experiment 1 example now uses `Qwen/Qwen3-4B`; the smaller
pilot pair lives in
[`configs/examples/experiment1_rollout_patch_verification_pilot.toml`](configs/examples/experiment1_rollout_patch_verification_pilot.toml)
and
[`configs/examples/experiment1_patch_verification_pilot.toml`](configs/examples/experiment1_patch_verification_pilot.toml).

## Command Surface

- `uv run rewardhack-interp collect-rollouts --config <path>`
- `uv run rewardhack-interp capture-activations --config <rollout-config> --rollouts <jsonl>`
- `uv run rewardhack-interp train-probe --config <path>`
- `uv run rewardhack-interp compare-representations --config <path>`
- `uv run rewardhack-interp cluster-activations --config <path>`
- `uv run rewardhack-interp logit-lens --config <path>`
- `uv run rewardhack-interp patch-activations --config <path>`
- `uv run rewardhack-interp train-grpo --config <path>`
- `uv run rewardhack-interp compare-checkpoints --config <path>`
- `uv run rewardhack-interp run-experiment-1 --config <path>`

## Project Layout

- [`src/rewardhack_interp`](src/rewardhack_interp): library code
- [`configs/examples`](configs/examples): example TOML configs
- [`docs/architecture.md`](docs/architecture.md): module layout and design decisions
- [`docs/workflows.md`](docs/workflows.md): end-to-end workflow guide
- [`docs/artifacts.md`](docs/artifacts.md): saved artifact formats
- [`artifacts/README.md`](artifacts/README.md): output directory guide
- [`tests`](tests): unit tests for configs, reward logic, and analysis helpers

## Notes

- The rollout pipeline is intentionally config-driven so experiments stay reproducible through explicit environment seeds.
- Activation capture is implemented as replay over the exact prompt/completion token sequence saved with each rollout.
- The causal patching workflow currently patches prompt-boundary activations for selected layers or modules, which keeps the intervention path inspectable and easy to extend.
- Experiment 1 now defaults to seed-level holdout rather than row-level random splitting, and the report stores bootstrap confidence intervals for the layerwise probe metrics.
- GRPO example configs now enforce explicit train/held-out seed separation, and the runner writes held-out official reward, oracle reward, verifier gap, false-pass rate, and cohort counts for base, intermediate, and final checkpoints.
- `rewardhack-gym` is installed directly from GitHub through the project dependency set so the integration stays pinned to the real substrate rather than a local copy.
- W&B support uses `wandb.init()` context-managed runs and optional artifact logging, following the official W&B SDK guidance.
