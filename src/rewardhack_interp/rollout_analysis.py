from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path

from rewardhack_interp.config import CaseStudyConfig, RolloutSubsetConfig
from rewardhack_interp.io import load_models, write_json, write_jsonl
from rewardhack_interp.types import (
    CaseStudyArtifact,
    CaseStudySelectionStrategy,
    CohortLabel,
    HistogramSummary,
    NumericDistributionSummary,
    RepresentativeCaseStudy,
    RolloutCohortSummary,
    RolloutSubsetArtifact,
    RolloutSummaryArtifact,
    SubsetMatchField,
    TrajectoryArtifact,
)
from rewardhack_interp.utils.identifiers import trajectory_sample_id
from rewardhack_interp.utils.paths import ensure_dir, ensure_parent_dir, to_path_string


def summarize_rollouts(
    rollout_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> RolloutSummaryArtifact:
    records = load_models(rollout_path, TrajectoryArtifact)
    grouped = group_records_by_cohort(records)
    artifact = RolloutSummaryArtifact(
        rollout_path=to_path_string(rollout_path),
        total_rows=len(records),
        cohort_counts={cohort: len(group_records) for cohort, group_records in grouped.items()},
        cohort_summaries={
            cohort: summarize_cohort_records(group_records)
            for cohort, group_records in grouped.items()
        },
        environment_names=sorted({record.task.environment_name for record in records}),
        policy_ids=sorted({record.policy_id for record in records}),
        run_ids=sorted({record.run_id for record in records}),
    )
    if output_path is not None:
        output = ensure_parent_dir(output_path)
        plot_dir = ensure_dir(output.with_name(f"{output.stem}_plots"))
        plot_paths = build_rollout_summary_plots(records, output_dir=plot_dir)
        artifact = artifact.model_copy(
            update={key: value for key, value in {"plot_paths": plot_paths}.items()}
        )
        write_json(output, artifact.model_dump(mode="json"))
    return artifact


def format_rollout_summary(artifact: RolloutSummaryArtifact) -> str:
    lines = [
        f"rollout: {artifact.rollout_path}",
        f"total rows: {artifact.total_rows}",
        "cohorts: "
        + ", ".join(
            f"{name}={count}" for name, count in sorted(artifact.cohort_counts.items())
        ),
    ]
    for cohort_name, cohort_summary in sorted(artifact.cohort_summaries.items()):
        lines.extend(
            [
                f"[{cohort_name}]",
                f"  count: {cohort_summary.count}",
                f"  official mean: {format_optional_float(cohort_summary.official_reward.mean)}",
                f"  oracle mean: {format_optional_float(cohort_summary.oracle_reward.mean)}",
                f"  gap mean: {format_optional_float(cohort_summary.verifier_gap.mean)}",
                f"  false-pass rate: {cohort_summary.false_pass_rate:.3f}",
                "  completion length mean: "
                + format_optional_float(cohort_summary.completion_length.mean),
            ]
        )
    return "\n".join(lines)


def build_rollout_subset(config: RolloutSubsetConfig) -> RolloutSubsetArtifact:
    records = load_models(config.rollout_path, TrajectoryArtifact)
    filtered = [
        record for record in records if record.cohort in set(config.included_cohorts)
    ]
    grouped = {
        cohort: [record for record in filtered if record.cohort == cohort]
        for cohort in config.included_cohorts
    }
    rng = random.Random(config.random_seed)
    selected_by_cohort, matched_bucket_counts = select_balanced_subset(grouped, config, rng)
    selected_records = [
        record
        for cohort in config.included_cohorts
        for record in selected_by_cohort.get(cohort, [])
    ]
    selected_records.sort(
        key=lambda record: (
            record.cohort.value,
            record.task.task_seed,
            record.generation.completion_index,
        )
    )

    write_jsonl(config.output_path, [record.model_dump(mode="json") for record in selected_records])
    summary_path = config.summary_output_path or config.output_path.with_name(
        f"{config.output_path.stem}.summary.json"
    )
    artifact = RolloutSubsetArtifact(
        input_rollout_path=to_path_string(config.rollout_path),
        output_rollout_path=to_path_string(config.output_path),
        summary_path=to_path_string(summary_path),
        included_cohorts=list(config.included_cohorts),
        max_samples_per_cohort=config.max_samples_per_cohort,
        random_seed=config.random_seed,
        matching_fields=list(config.matching_fields),
        fill_unmatched_remainder=config.fill_unmatched_remainder,
        counts_before={cohort.value: len(grouped[cohort]) for cohort in config.included_cohorts},
        counts_after={
            cohort.value: len(selected_by_cohort.get(cohort, []))
            for cohort in config.included_cohorts
        },
        matched_bucket_counts=matched_bucket_counts,
        selected_trace_ids=[record.trace_id for record in selected_records],
        selected_sample_ids=[trajectory_sample_id(record) for record in selected_records],
    )
    write_json(summary_path, artifact.model_dump(mode="json"))
    return artifact


def extract_case_studies(config: CaseStudyConfig) -> CaseStudyArtifact:
    records = load_models(config.rollout_path, TrajectoryArtifact)
    rng = random.Random(config.random_seed)
    grouped = {
        cohort: [record for record in records if record.cohort == cohort]
        for cohort in config.included_cohorts
    }
    examples: list[RepresentativeCaseStudy] = []
    counts_by_cohort: dict[str, int] = {}

    for cohort in config.included_cohorts:
        cohort_records = grouped.get(cohort, [])
        selected = select_case_study_records(
            cohort_records=cohort_records,
            samples_per_cohort=config.samples_per_cohort,
            selection_strategy=config.selection_strategy,
            rng=rng,
        )
        counts_by_cohort[cohort.value] = len(selected)
        for record in selected:
            examples.append(
                RepresentativeCaseStudy(
                    trace_id=record.trace_id,
                    sample_id=trajectory_sample_id(record),
                    task_id=record.task.task_id,
                    cohort=record.cohort,
                    official_reward=record.reward_metrics.official_reward,
                    oracle_reward=record.reward_metrics.oracle_reward,
                    verifier_gap=record.reward_metrics.verifier_gap,
                    false_pass=record.reward_metrics.false_pass,
                    prompt_summary=summarize_prompt(record.task.prompt),
                    completion_text=record.generation.completion_text,
                    scenario_id=string_or_none(record.mech_interp_row.get("scenario_id")),
                )
            )

    artifact = CaseStudyArtifact(
        rollout_path=to_path_string(config.rollout_path),
        output_path=to_path_string(config.output_path),
        samples_per_cohort=config.samples_per_cohort,
        random_seed=config.random_seed,
        selection_strategy=config.selection_strategy,
        included_cohorts=list(config.included_cohorts),
        counts_by_cohort=counts_by_cohort,
        examples=examples,
    )
    write_json(config.output_path, artifact.model_dump(mode="json"))
    return artifact


def group_records_by_cohort(
    records: list[TrajectoryArtifact],
) -> dict[str, list[TrajectoryArtifact]]:
    grouped: dict[str, list[TrajectoryArtifact]] = defaultdict(list)
    for record in records:
        grouped[record.cohort.value].append(record)
    return dict(sorted(grouped.items()))


def summarize_cohort_records(records: list[TrajectoryArtifact]) -> RolloutCohortSummary:
    official_rewards = [record.reward_metrics.official_reward for record in records]
    oracle_rewards = [record.reward_metrics.oracle_reward for record in records]
    verifier_gaps = [record.reward_metrics.verifier_gap for record in records]
    completion_lengths = [record.generation.completion_token_count for record in records]
    false_pass_rate = (
        sum(int(record.reward_metrics.false_pass) for record in records) / len(records)
        if records
        else 0.0
    )
    return RolloutCohortSummary(
        count=len(records),
        official_reward=summarize_numeric_distribution(official_rewards, include_histogram=False),
        oracle_reward=summarize_numeric_distribution(oracle_rewards, include_histogram=False),
        verifier_gap=summarize_numeric_distribution(verifier_gaps, include_histogram=True),
        completion_length=summarize_numeric_distribution(
            completion_lengths,
            include_histogram=False,
        ),
        false_pass_rate=false_pass_rate,
    )


def summarize_numeric_distribution(
    values: list[float] | list[int],
    *,
    include_histogram: bool,
) -> NumericDistributionSummary:
    import numpy as np

    if not values:
        return NumericDistributionSummary()
    array = np.asarray(values, dtype=np.float64)
    percentile_points = [0, 5, 25, 50, 75, 95, 100]
    percentiles = {
        str(point): float(np.percentile(array, point)) for point in percentile_points
    }
    histogram = build_histogram(array.tolist()) if include_histogram else None
    return NumericDistributionSummary(
        mean=float(array.mean()),
        min=float(array.min()),
        max=float(array.max()),
        percentiles=percentiles,
        histogram=histogram,
    )


def build_histogram(values: list[float], *, bins: int = 12) -> HistogramSummary:
    import numpy as np

    if not values:
        return HistogramSummary(counts=[], bin_edges=[])
    counts, bin_edges = np.histogram(np.asarray(values, dtype=np.float64), bins=bins)
    return HistogramSummary(
        counts=[int(value) for value in counts.tolist()],
        bin_edges=[float(value) for value in bin_edges.tolist()],
    )


def build_rollout_summary_plots(
    records: list[TrajectoryArtifact],
    *,
    output_dir: str | Path,
) -> dict[str, str]:
    import pandas as pd
    import seaborn as sns

    plot_dir = ensure_dir(output_dir)
    plot_frame = pd.DataFrame(
        [
            {
                "cohort": record.cohort.value,
                "official_reward": record.reward_metrics.official_reward,
                "oracle_reward": record.reward_metrics.oracle_reward,
                "verifier_gap": record.reward_metrics.verifier_gap,
                "completion_length": record.generation.completion_token_count,
            }
            for record in records
        ]
    )
    sns.set_theme(style="whitegrid", context="talk")

    count_path = plot_dir / "cohort_counts.png"
    reward_path = plot_dir / "reward_distributions.png"
    gap_path = plot_dir / "verifier_gap_histogram.png"
    length_path = plot_dir / "completion_lengths.png"

    axis = sns.countplot(data=plot_frame, x="cohort", order=sorted(plot_frame["cohort"].unique()))
    axis.set(xlabel="Cohort", ylabel="Count", title="Rollout Cohort Counts")
    axis.figure.tight_layout()
    axis.figure.savefig(count_path, dpi=200)
    axis.figure.clf()

    reward_frame = plot_frame.melt(
        id_vars="cohort",
        value_vars=["official_reward", "oracle_reward", "verifier_gap"],
        var_name="metric",
        value_name="value",
    )
    axis = sns.boxplot(data=reward_frame, x="metric", y="value", hue="cohort")
    axis.set(xlabel="Metric", ylabel="Value", title="Reward Statistics by Cohort")
    axis.figure.tight_layout()
    axis.figure.savefig(reward_path, dpi=200)
    axis.figure.clf()

    axis = sns.histplot(
        data=plot_frame,
        x="verifier_gap",
        hue="cohort",
        bins=12,
        element="step",
        common_norm=False,
    )
    axis.set(xlabel="Verifier Gap", ylabel="Count", title="Verifier Gap Distribution by Cohort")
    axis.figure.tight_layout()
    axis.figure.savefig(gap_path, dpi=200)
    axis.figure.clf()

    axis = sns.boxplot(data=plot_frame, x="cohort", y="completion_length")
    axis.set(
        xlabel="Cohort",
        ylabel="Completion Tokens",
        title="Completion Length by Cohort",
    )
    axis.figure.tight_layout()
    axis.figure.savefig(length_path, dpi=200)
    axis.figure.clf()

    return {
        "cohort_counts": to_path_string(count_path),
        "reward_distributions": to_path_string(reward_path),
        "verifier_gap_histogram": to_path_string(gap_path),
        "completion_lengths": to_path_string(length_path),
    }


def select_balanced_subset(
    grouped: dict[CohortLabel, list[TrajectoryArtifact]],
    config: RolloutSubsetConfig,
    rng: random.Random,
) -> tuple[dict[CohortLabel, list[TrajectoryArtifact]], dict[str, int]]:
    if not config.matching_fields:
        return (
            {
                cohort: sample_records(records, config.max_samples_per_cohort, rng)
                for cohort, records in grouped.items()
            },
            {},
        )

    bucketed = {
        cohort: bucket_records(records, config)
        for cohort, records in grouped.items()
    }
    bucket_key_sets = [set(buckets.keys()) for buckets in bucketed.values()]
    common_bucket_keys = set.intersection(*bucket_key_sets) if bucket_key_sets else set()

    matched_candidates: dict[CohortLabel, list[TrajectoryArtifact]] = {
        cohort: [] for cohort in grouped
    }
    matched_bucket_counts: dict[str, int] = {}
    for bucket_key in sorted(common_bucket_keys):
        common_count = min(len(bucketed[cohort][bucket_key]) for cohort in grouped)
        if common_count <= 0:
            continue
        matched_bucket_counts[render_bucket_key(bucket_key)] = common_count
        for cohort in grouped:
            matched_candidates[cohort].extend(
                sample_records(bucketed[cohort][bucket_key], common_count, rng)
            )

    selected_by_cohort: dict[CohortLabel, list[TrajectoryArtifact]] = {
        cohort: sample_records(records, config.max_samples_per_cohort, rng)
        for cohort, records in matched_candidates.items()
    }
    if not config.fill_unmatched_remainder:
        return selected_by_cohort, matched_bucket_counts

    for cohort, records in grouped.items():
        selected = selected_by_cohort.get(cohort, [])
        if len(selected) >= config.max_samples_per_cohort:
            continue
        selected_ids = {trajectory_sample_id(record) for record in selected}
        remaining = [
            record for record in records if trajectory_sample_id(record) not in selected_ids
        ]
        needed = config.max_samples_per_cohort - len(selected)
        selected.extend(sample_records(remaining, needed, rng))
        selected_by_cohort[cohort] = selected
    return selected_by_cohort, matched_bucket_counts


def bucket_records(
    records: list[TrajectoryArtifact],
    config: RolloutSubsetConfig,
) -> dict[tuple[str, ...], list[TrajectoryArtifact]]:
    grouped: dict[tuple[str, ...], list[TrajectoryArtifact]] = defaultdict(list)
    for record in records:
        bucket_key = tuple(
            bucket_label_for_field(record, match_field, config)
            for match_field in config.matching_fields
        )
        grouped[bucket_key].append(record)
    return grouped


def bucket_label_for_field(
    record: TrajectoryArtifact,
    match_field: SubsetMatchField,
    config: RolloutSubsetConfig,
) -> str:
    if match_field == SubsetMatchField.completion_length_bucket:
        return bucketize_numeric_value(
            float(record.generation.completion_token_count),
            bucket_size=float(config.completion_length_bucket_size),
            prefix="len",
        )
    if match_field == SubsetMatchField.official_reward_bucket:
        return bucketize_numeric_value(
            record.reward_metrics.official_reward,
            bucket_size=config.official_reward_bucket_size,
            prefix="official",
        )
    if match_field == SubsetMatchField.verifier_gap_bucket:
        return bucketize_numeric_value(
            record.reward_metrics.verifier_gap,
            bucket_size=config.verifier_gap_bucket_size,
            prefix="gap",
        )
    raise ValueError(f"Unsupported matching field {match_field!r}.")


def bucketize_numeric_value(value: float, *, bucket_size: float, prefix: str) -> str:
    lower = math.floor(value / bucket_size) * bucket_size
    upper = lower + bucket_size
    return f"{prefix}:{lower:.3f}-{upper:.3f}"


def render_bucket_key(bucket_key: tuple[str, ...]) -> str:
    return "|".join(bucket_key)


def sample_records(
    records: list[TrajectoryArtifact],
    count: int,
    rng: random.Random,
) -> list[TrajectoryArtifact]:
    if count <= 0 or not records:
        return []
    if len(records) <= count:
        sampled = list(records)
        rng.shuffle(sampled)
        return sampled
    sampled = list(records)
    rng.shuffle(sampled)
    return sampled[:count]


def select_case_study_records(
    *,
    cohort_records: list[TrajectoryArtifact],
    samples_per_cohort: int,
    selection_strategy: CaseStudySelectionStrategy,
    rng: random.Random,
) -> list[TrajectoryArtifact]:
    if selection_strategy == CaseStudySelectionStrategy.top_verifier_gap:
        return sorted(
            cohort_records,
            key=lambda record: (
                record.reward_metrics.verifier_gap,
                record.reward_metrics.official_reward,
            ),
            reverse=True,
        )[:samples_per_cohort]
    return sample_records(cohort_records, samples_per_cohort, rng)


def summarize_prompt(prompt: str, *, max_length: int = 180) -> str:
    first_line = prompt.strip().splitlines()[0] if prompt.strip() else ""
    if len(first_line) <= max_length:
        return first_line
    return f"{first_line[: max_length - 3]}..."


def string_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def format_optional_float(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"
