# Architecture

## Principles

- Build on top of `rewardhack-gym`, never around it
- Keep research artifacts typed and inspectable
- Separate collection, activation capture, analysis, intervention, and training concerns
- Favor JSONL metadata plus tensor bundles over opaque binary-only logs

## Package Structure

- `rewardhack_interp.config`
  Central typed config models for rollouts, activation capture, analysis, causal experiments, GRPO runs, and checkpoint comparison.

- `rewardhack_interp.types`
  Pydantic schemas for persisted rollouts, activations, probes, clustering reports, logit-lens outputs, patch trials, and checkpoint summaries.

- `rewardhack_interp.gym_integration`
  The main adapter boundary over `rewardhack-gym`: environment creation, task serialization,
  normalized rollout construction, mech-interp row construction, and reward extraction.

- `rewardhack_interp.modeling`
  Qwen 3 loading and completion generation through Hugging Face Transformers and optional PEFT adapters.

- `rewardhack_interp.rollouts`
  End-to-end rollout collection: sample tasks, generate completions, evaluate under official/oracle, label cohorts, and optionally capture activations.

- `rewardhack_interp.activations`
  Replay-time hidden-state and module-output capture with forward hooks and saved tensor bundles.

- `rewardhack_interp.analysis`
  Feature extraction, linear probes, representation comparison, clustering, and logit lens.

- `rewardhack_interp.causal`
  Activation patching experiments for matched trajectories.

- `rewardhack_interp.rl`
  Reward scalarization, GRPO training utilities, and held-out checkpoint evaluation built directly from `rewardhack-gym` signals.

- `rewardhack_interp.tracking`
  Optional Weights & Biases run management, metric flattening, artifact logging, and trainer integration.

- `rewardhack_interp.checkpoints`
  Shared-seed evaluation across multiple checkpoints.

## Data Model

The key persisted units are:

- `TrajectoryArtifact`
  Full rollout metadata plus the raw `rewardhack-gym` trajectory payload and a normalized adapter
  record for prompt/completion/reward-gap analysis.

- `ActivationCaptureArtifact`
  Metadata for a saved activation tensor bundle, keyed by `trace_id`.

- analysis artifacts
  Probe, clustering, representation, logit-lens, patch, and checkpoint reports.

This split keeps large tensors outside the rollout JSONL while still making every downstream artifact traceable back to the original task and completion.
