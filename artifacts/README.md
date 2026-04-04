# Artifacts

Runtime outputs live here by default.

- `artifacts/rollouts/`: rollout JSONL, summaries, and exported cohorts
- `artifacts/activations/`: activation metadata JSON plus tensor bundles saved with `torch.save`
- `artifacts/analysis/`: probe reports, clustering summaries, representation comparisons, and logit-lens outputs
- `artifacts/checkpoints/`: checkpoint-to-checkpoint evaluation summaries

The repository keeps these directories but ignores generated files by default.
