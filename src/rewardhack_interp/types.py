from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CohortLabel(StrEnum):
    genuine_success = "genuine_success"
    genuine_failure = "genuine_failure"
    reward_hack = "reward_hack"
    oracle_only_pass = "oracle_only_pass"
    ambiguous = "ambiguous"


class RewardSignal(StrEnum):
    official = "official"
    oracle = "oracle"
    gap_aware = "gap_aware"
    anti_hack = "anti_hack"


class PoolingStrategy(StrEnum):
    last_completion_token = "last_completion_token"
    mean_completion = "mean_completion"
    max_completion = "max_completion"
    last_prompt_token = "last_prompt_token"
    mean_prompt = "mean_prompt"


class ExperimentOneSplitStrategy(StrEnum):
    task_seed_holdout = "task_seed_holdout"
    row_random_holdout = "row_random_holdout"


class SubsetMatchField(StrEnum):
    completion_length_bucket = "completion_length_bucket"
    official_reward_bucket = "official_reward_bucket"
    verifier_gap_bucket = "verifier_gap_bucket"


class CaseStudySelectionStrategy(StrEnum):
    random = "random"
    top_verifier_gap = "top_verifier_gap"


class ActivationCaptureMode(StrEnum):
    replay = "replay"


class InterventionMode(StrEnum):
    generation = "generation"
    replay = "replay"


class FinishReason(StrEnum):
    eos = "eos"
    length = "length"
    stop = "stop"


class TaskReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    environment_name: str
    environment_profile: str
    task_seed: int
    task_id: str
    family: str
    difficulty: str
    prompt: str
    expected_interface: str
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RewardMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    official_reward: float
    oracle_reward: float
    verifier_gap: float
    false_pass: bool
    official_passed: bool
    oracle_passed: bool
    exploit_labels: list[str] = Field(default_factory=list)
    annotations: dict[str, Any] = Field(default_factory=dict)
    reward_metadata: dict[str, Any] = Field(default_factory=dict)


class NormalizedRolloutRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    completion: str
    env_id: str
    family_id: str
    task_id: str
    official_reward: float
    oracle_reward: float
    verifier_gap: float
    false_pass: bool
    exploit_labels: list[str] = Field(default_factory=list)


class GenerationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    completion_index: int
    rendered_prompt: str
    completion_text: str
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    decoded_completion_tokens: list[str] = Field(default_factory=list)
    token_logprobs: list[float] | None = None
    prompt_token_count: int
    completion_token_count: int
    finish_reason: FinishReason
    sampling: dict[str, Any] = Field(default_factory=dict)


class TrajectoryArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    trace_id: str
    sample_id: str | None = None
    policy_id: str
    model_name_or_path: str
    revision: str | None = None
    adapter_name_or_path: str | None = None
    task: TaskReference
    generation: GenerationRecord
    reward_metrics: RewardMetrics
    normalized_rollout: NormalizedRolloutRecord
    cohort: CohortLabel
    trajectory: dict[str, Any]
    mech_interp_row: dict[str, Any]
    activation_artifact_path: str | None = None


class ActivationCaptureArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    sample_id: str | None = None
    rollout_run_id: str
    source_model_name_or_path: str
    source_adapter_name_or_path: str | None = None
    cohort: CohortLabel
    completion_index: int | None = None
    tensor_path: str
    prompt_token_count: int
    completion_token_count: int
    tensor_names: list[str]
    tensor_shapes: dict[str, list[int]]
    token_ids: list[int]
    capture_mode: ActivationCaptureMode
    stored_token_strategy: PoolingStrategy | None = None
    module_globs: list[str] = Field(default_factory=list)


class ProbeMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accuracy: float
    macro_f1: float
    roc_auc: float | None = None
    n_train: int
    n_test: int


class ProbeArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_name: str
    pooling_strategy: PoolingStrategy
    positive_labels: list[CohortLabel]
    negative_labels: list[CohortLabel]
    metrics: ProbeMetrics
    label_counts: dict[str, int]
    coefficients_path: str | None = None


class HistogramSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    counts: list[int]
    bin_edges: list[float]


class NumericDistributionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean: float | None = None
    min: float | None = None
    max: float | None = None
    percentiles: dict[str, float] = Field(default_factory=dict)
    histogram: HistogramSummary | None = None


class RolloutCohortSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int
    official_reward: NumericDistributionSummary
    oracle_reward: NumericDistributionSummary
    verifier_gap: NumericDistributionSummary
    completion_length: NumericDistributionSummary
    false_pass_rate: float


class RolloutSummaryArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rollout_path: str
    total_rows: int
    cohort_counts: dict[str, int]
    cohort_summaries: dict[str, RolloutCohortSummary]
    environment_names: list[str] = Field(default_factory=list)
    policy_ids: list[str] = Field(default_factory=list)
    run_ids: list[str] = Field(default_factory=list)
    plot_paths: dict[str, str] = Field(default_factory=dict)


class RolloutSubsetArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_rollout_path: str
    output_rollout_path: str
    summary_path: str
    included_cohorts: list[CohortLabel]
    max_samples_per_cohort: int
    random_seed: int
    matching_fields: list[SubsetMatchField] = Field(default_factory=list)
    fill_unmatched_remainder: bool
    counts_before: dict[str, int]
    counts_after: dict[str, int]
    matched_bucket_counts: dict[str, int] = Field(default_factory=dict)
    selected_trace_ids: list[str] = Field(default_factory=list)
    selected_sample_ids: list[str] = Field(default_factory=list)


class RepresentativeCaseStudy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    sample_id: str | None = None
    task_id: str
    cohort: CohortLabel
    official_reward: float
    oracle_reward: float
    verifier_gap: float
    false_pass: bool
    prompt_summary: str
    completion_text: str
    scenario_id: str | None = None


class CaseStudyArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rollout_path: str
    output_path: str
    samples_per_cohort: int
    random_seed: int
    selection_strategy: CaseStudySelectionStrategy
    included_cohorts: list[CohortLabel]
    counts_by_cohort: dict[str, int]
    examples: list[RepresentativeCaseStudy]


class LayerwiseBinaryProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layer_name: str
    layer_index: int
    accuracy: float | None = None
    macro_f1: float | None = None
    roc_auc: float | None = None
    n_train: int = 0
    n_test: int = 0
    n_examples: int = 0
    skipped_reason: str | None = None


class LayerwiseBinaryProbeArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    activation_manifest_path: str
    positive_label: CohortLabel
    negative_label: CohortLabel
    pooling_strategy: PoolingStrategy
    label_counts: dict[str, int]
    results: list[LayerwiseBinaryProbeResult]
    report_path: str
    plot_paths: dict[str, str] = Field(default_factory=dict)


class RepresentationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_name: str
    pooling_strategy: PoolingStrategy
    counts_by_cohort: dict[str, int]
    centroid_distances: dict[str, float]
    linear_cka: dict[str, float]
    within_group_variance: dict[str, float]


class ClusteringArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_name: str
    pooling_strategy: PoolingStrategy
    n_clusters: int
    inertia: float
    silhouette_score: float | None = None
    cluster_label_histograms: dict[str, dict[str, int]]


class LogitLensCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: str
    token_id: int
    probability: float


class LogitLensLayerResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tensor_name: str
    position: int
    top_tokens: list[LogitLensCandidate]


class LogitLensArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    position_strategy: PoolingStrategy
    layers: list[LogitLensLayerResult]


class PatchTrialArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_trace_id: str
    donor_trace_id: str
    mode: InterventionMode
    layer_names: list[str]
    baseline_completion: str
    patched_completion: str
    baseline_official_reward: float
    baseline_oracle_reward: float
    patched_official_reward: float
    patched_oracle_reward: float
    baseline_cohort: CohortLabel
    patched_cohort: CohortLabel


class CheckpointSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name_or_path: str
    adapter_name_or_path: str | None = None
    checkpoint_path: str | None = None
    phase_label: str | None = None
    split_name: str | None = None
    global_step: int | None = None
    rollout_path: str | None = None
    mean_official_reward: float
    mean_oracle_reward: float
    mean_verifier_gap: float
    false_pass_rate: float
    counts_by_cohort: dict[str, int]


class CheckpointComparisonArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    environment_name: str
    environment_profile: str
    task_seeds: list[int]
    summaries: list[CheckpointSummary]


class GRPOHeldoutEvaluationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_name: str
    reward_signal: RewardSignal
    evaluation_environment_name: str
    evaluation_environment_profile: str
    evaluation_task_seeds: list[int]
    split_name: str
    summaries: list[CheckpointSummary]


class GRPOTrainingArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_name: str
    output_dir: str
    reward_signal: RewardSignal
    dataset_size: int
    training_environment_name: str
    training_environment_profile: str
    training_task_seeds: list[int]
    train_metrics: dict[str, Any]
    reward_trace_output: str | None = None
    heldout_evaluation_path: str | None = None
    heldout_evaluation: GRPOHeldoutEvaluationArtifact | None = None


class ExperimentOneComparisonResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comparison_name: str
    positive_label: CohortLabel
    negative_label: CohortLabel
    split_strategy: ExperimentOneSplitStrategy
    layer_name: str
    layer_index: int
    accuracy: float | None = None
    accuracy_std: float | None = None
    accuracy_ci_low: float | None = None
    accuracy_ci_high: float | None = None
    macro_f1: float | None = None
    macro_f1_std: float | None = None
    macro_f1_ci_low: float | None = None
    macro_f1_ci_high: float | None = None
    roc_auc: float | None = None
    roc_auc_std: float | None = None
    roc_auc_ci_low: float | None = None
    roc_auc_ci_high: float | None = None
    n_train: int = 0
    n_test: int = 0
    n_examples: int = 0
    num_repeats: int = 0
    repeat_random_seeds: list[int] = Field(default_factory=list)
    skipped_reason: str | None = None


class ExperimentOneRepresentationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pair_name: str
    left_label: CohortLabel
    right_label: CohortLabel
    layer_name: str
    layer_index: int
    centroid_distance: float | None = None
    linear_cka: float | None = None
    left_count: int = 0
    right_count: int = 0


class ExperimentOneClusteringResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layer_name: str
    layer_index: int
    n_clusters: int
    purity: float | None = None
    silhouette_score: float | None = None
    label_histograms: dict[str, dict[str, int]] = Field(default_factory=dict)
    n_examples: int = 0
    skipped_reason: str | None = None


class ExperimentOneMatchedPairSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    match_level: str
    n_pairs: int
    n_pair_groups: int


class ExperimentOneArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    rollout_path: str
    activation_manifest_path: str
    report_path: str
    environment_name: str
    environment_profile: str | None = None
    model_name_or_path: str
    policy_id: str
    pooling_strategy: PoolingStrategy
    split_strategy: ExperimentOneSplitStrategy
    train_task_seeds: list[int] = Field(default_factory=list)
    eval_task_seeds: list[int] = Field(default_factory=list)
    split_random_seeds: list[int] = Field(default_factory=list)
    confidence_level: float
    bootstrap_repeats: int
    layer_names: list[str]
    cohort_counts: dict[str, int]
    scenario_counts: dict[str, int]
    cohort_counts_by_scenario: dict[str, dict[str, int]]
    semantic_failure_counts_by_cohort: dict[str, dict[str, int]]
    false_pass_exploit_label_counts: dict[str, int]
    false_pass_exploit_class_counts: dict[str, int]
    matched_pair_summaries: list[ExperimentOneMatchedPairSummary]
    comparison_results: list[ExperimentOneComparisonResult]
    representation_results: list[ExperimentOneRepresentationResult]
    clustering_results: list[ExperimentOneClusteringResult]
    matched_pairs_path: str | None = None
    plot_path: str | None = None
