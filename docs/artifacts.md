# Artifact Formats

## Rollouts

Rollouts are stored as JSONL. Each row is a `TrajectoryArtifact` with:

- environment/task reference
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
- direct reward fields:
  - `official_reward`
  - `oracle_reward`
  - `verifier_gap`
  - `false_pass`
- cohort label
- mech-interp row keyed by `trace_id`

## Activations

Activation capture creates two files per trace:

- `<trace_id>.json`
  Metadata describing tensor names, shapes, cohort, token counts, and source model info
- `<trace_id>.pt`
  `torch.save` bundle containing:
  - selected hidden states
  - selected module outputs
  - token ids
  - attention mask

## Analysis Outputs

- probe reports: JSON plus `.npy` coefficients
- representation comparison: JSON
- clustering: JSON
- logit lens: JSON
- patch trials: JSON
- checkpoint comparison: JSON
- Experiment 1: JSON report, probe plot, and matched-pairs JSONL
- GRPO: training summary JSON plus optional held-out checkpoint evaluation JSON

Every downstream report is designed to stay small and human-inspectable while still pointing back to the originating trace ids.

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
