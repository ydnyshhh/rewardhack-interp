# Artifact Formats

## Rollouts

Rollouts are stored as JSONL. Each row is a `TrajectoryArtifact` with:

- nested `task` metadata
- nested `generation` metadata
- nested `reward_metrics`
- rendered prompt
- completion text and token ids
- normalized rollout adapter record with:
  - `prompt`
  - `completion`
  - `env_id`
  - `family_id`
  - `task_id`
  - `official_reward`
  - `oracle_reward`
  - `verifier_gap`
  - `false_pass`
  - `exploit_labels`
- raw `rewardhack-gym` trajectory payload
- cohort label
- mech-interp row keyed by `trace_id`

For downstream replay and subset workflows, the analysis utilities derive a stable
per-row `sample_id` from `trace_id` plus `completion_index`. This avoids collisions
when one task seed has multiple sampled completions.

The rollout utilities in this repo read these nested fields directly rather than assuming flat top-level reward columns.

## Activations

Activation capture creates two files per sampled row:

- `<sample_id>.json`
  Metadata describing tensor names, shapes, cohort, token counts, and source model info
- `<sample_id>.pt`
  `torch.save` bundle containing:
  - selected hidden states
  - selected module outputs
  - token ids
  - attention mask

The manifest JSONL stores both `trace_id` and the derived `sample_id`, and tensor
paths are written with a filesystem-safe stem so replay capture works for real
Runpod artifacts whose trace ids contain characters such as `:`.

For subset-first workflows, the default-safe capture mode stores:

- hidden states only
- a small layer subset
- pooled token representations by default (`last_completion_token`)

Full all-layer all-token dumps are intentionally not the default because they become storage-heavy quickly.

## Analysis Outputs

- probe reports: JSON plus `.npy` coefficients
- representation comparison: JSON
- clustering: JSON
- logit lens: JSON
- patch trials: JSON
- checkpoint comparison: JSON
- Experiment 1: JSON report, probe plot, and matched-pairs JSONL
- GRPO: training summary JSON plus optional held-out checkpoint evaluation JSON
- rollout summary: JSON plus saved cohort/reward plots
- rollout subset: JSONL plus selection summary JSON
- case studies: JSON with representative examples
- layerwise probe: JSON report plus saved accuracy/AUROC plots

Every downstream report is designed to stay small and human-inspectable while still pointing back to the originating rollout rows through `sample_id` and `trace_id`.

Experiment 1 reports additionally preserve:

- scenario-level cohort counts
- semantic-failure counts by cohort
- false-pass exploit label and exploit-class summaries
- matched-pair counts by match level
- split strategy and any explicit train/eval task-seed partitions
- repeated split seeds
- confidence level and bootstrap repeat count
- per-comparison standard deviations and confidence intervals for probe metrics

GRPO training summaries additionally preserve:

- the exact training task seeds used after dataset-size expansion
- the reward-trace output path when enabled
- held-out checkpoint evaluation summaries for base, intermediate, and final policies

## W&B Logging

When a workflow config includes an enabled `[wandb]` block, the repo logs:

- run config
- summary metrics
- output JSON and JSONL artifacts
- activation manifests
- probe coefficient files

Full activation tensor directories and full checkpoint directories are not uploaded automatically by default.

## Runpod Storage

- Generated artifacts live under `artifacts/` inside the repo checkout on the execution machine.
- If you run the repo on Runpod, artifacts created under `/workspace` are not automatically committed back to Git.
- Replay capture requires access to the original base model weights again. Run it where the Hugging Face cache is present, or set `HF_TOKEN` so the model can be downloaded.
