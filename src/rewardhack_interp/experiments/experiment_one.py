from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from rewardhack_interp.activations import load_activation_tensors
from rewardhack_interp.analysis.features import pool_tensor
from rewardhack_interp.analysis.representations import linear_cka
from rewardhack_interp.config import ExperimentOneComparisonConfig, ExperimentOneConfig
from rewardhack_interp.io import load_models, write_json, write_jsonl
from rewardhack_interp.tracking import start_wandb_run
from rewardhack_interp.types import (
    ActivationCaptureArtifact,
    CohortLabel,
    ExperimentOneArtifact,
    ExperimentOneClusteringResult,
    ExperimentOneComparisonResult,
    ExperimentOneMatchedPairSummary,
    ExperimentOneRepresentationResult,
    PoolingStrategy,
    TrajectoryArtifact,
)
from rewardhack_interp.utils.paths import ensure_dir, ensure_parent_dir, to_path_string


@dataclass(frozen=True, slots=True)
class JoinedActivationRecord:
    activation: ActivationCaptureArtifact
    rollout: TrajectoryArtifact


def run_experiment_one(config: ExperimentOneConfig) -> ExperimentOneArtifact:
    output_dir = ensure_dir(config.output_dir)
    report_path = config.report_path or output_dir / "experiment_one_report.json"
    plot_path = config.plot_path or output_dir / "probe_accuracy_by_layer.png"

    with start_wandb_run(
        wandb_config=config.wandb,
        run_name=config.experiment_name,
        job_type="experiment_one",
        config_payload=config.model_dump(mode="json"),
    ) as tracker:
        joined_records = load_joined_activation_records(
            rollout_path=config.rollout_path,
            activation_manifest_path=config.activation_manifest_path,
            include_cohorts=config.include_cohorts,
        )
        if not joined_records:
            raise ValueError("No joined rollout/activation records matched the requested cohorts.")

        layer_names = resolve_experiment_layer_names(joined_records, config)
        layer_feature_bank = build_layer_feature_bank(
            joined_records=joined_records,
            layer_names=layer_names,
            pooling_strategy=config.pooling_strategy,
        )
        (
            scenario_counts,
            cohort_counts_by_scenario,
            semantic_failure_counts_by_cohort,
        ) = summarize_slice_metadata(joined_records)
        matched_pairs_path = output_dir / "matched_pairs.jsonl"
        matched_pair_rows, matched_pair_summaries = build_experiment_matched_pairs(
            joined_records
        )
        write_jsonl(matched_pairs_path, matched_pair_rows)

        cohort_counts = Counter(record.rollout.cohort.value for record in joined_records)
        false_pass_exploit_label_counts = Counter()
        false_pass_exploit_class_counts = Counter()
        for record in joined_records:
            if record.rollout.cohort != CohortLabel.reward_hack:
                continue
            false_pass_exploit_label_counts.update(record.rollout.reward_metrics.exploit_labels)
            exploit_class = record.rollout.mech_interp_row.get("exploit_class")
            if isinstance(exploit_class, str) and exploit_class:
                false_pass_exploit_class_counts[exploit_class] += 1

        comparison_results: list[ExperimentOneComparisonResult] = []
        representation_results: list[ExperimentOneRepresentationResult] = []
        clustering_results: list[ExperimentOneClusteringResult] = []

        for layer_name in layer_names:
            layer_rows = layer_feature_bank[layer_name]
            layer_index = parse_layer_index(layer_name)
            comparison_results.extend(
                run_layerwise_probe_suite(
                    layer_name=layer_name,
                    layer_index=layer_index,
                    config=config,
                    comparisons=config.comparisons,
                    features=layer_rows["features"],
                    labels=layer_rows["labels"],
                )
            )
            representation_results.extend(
                run_layerwise_representation_suite(
                    layer_name=layer_name,
                    layer_index=layer_index,
                    comparisons=config.comparisons,
                    features=layer_rows["features"],
                    labels=layer_rows["labels"],
                )
            )
            clustering_results.append(
                run_layerwise_clustering(
                    layer_name=layer_name,
                    layer_index=layer_index,
                    features=layer_rows["features"],
                    labels=layer_rows["labels"],
                    n_clusters=config.n_clusters,
                )
            )

        figure_path = plot_probe_accuracy_by_layer(
            comparison_results=comparison_results,
            output_path=plot_path,
        )
        first_rollout = joined_records[0].rollout
        artifact = ExperimentOneArtifact(
            experiment_name=config.experiment_name,
            rollout_path=to_path_string(config.rollout_path),
            activation_manifest_path=to_path_string(config.activation_manifest_path),
            report_path=to_path_string(report_path),
            environment_name=first_rollout.task.environment_name,
            environment_profile=first_rollout.task.environment_profile,
            model_name_or_path=first_rollout.model_name_or_path,
            policy_id=first_rollout.policy_id,
            pooling_strategy=config.pooling_strategy,
            layer_names=list(layer_names),
            cohort_counts=dict(sorted(cohort_counts.items())),
            scenario_counts=dict(sorted(scenario_counts.items())),
            cohort_counts_by_scenario={
                scenario_id: dict(sorted(counts.items()))
                for scenario_id, counts in sorted(cohort_counts_by_scenario.items())
            },
            semantic_failure_counts_by_cohort={
                cohort_name: dict(sorted(counts.items()))
                for cohort_name, counts in sorted(semantic_failure_counts_by_cohort.items())
            },
            false_pass_exploit_label_counts=dict(
                sorted(false_pass_exploit_label_counts.items())
            ),
            false_pass_exploit_class_counts=dict(
                sorted(false_pass_exploit_class_counts.items())
            ),
            matched_pair_summaries=matched_pair_summaries,
            comparison_results=comparison_results,
            representation_results=representation_results,
            clustering_results=clustering_results,
            matched_pairs_path=to_path_string(matched_pairs_path),
            plot_path=to_path_string(figure_path),
        )
        write_json(report_path, artifact.model_dump(mode="json"))
        tracker.log_summary(
            {
                "environment_name": artifact.environment_name,
                "environment_profile": artifact.environment_profile,
                "num_layers": len(artifact.layer_names),
                "cohort_counts": artifact.cohort_counts,
                "matched_pair_counts": {
                    summary.match_level: summary.n_pairs
                    for summary in artifact.matched_pair_summaries
                },
            }
        )
        tracker.log_path(report_path, artifact_type="experiment-one-report")
        tracker.log_path(figure_path, artifact_type="experiment-one-plot")
        tracker.log_path(matched_pairs_path, artifact_type="experiment-one-matched-pairs")
        return artifact


def load_joined_activation_records(
    *,
    rollout_path: str | Path,
    activation_manifest_path: str | Path,
    include_cohorts: list[CohortLabel],
) -> list[JoinedActivationRecord]:
    rollouts = load_models(rollout_path, TrajectoryArtifact)
    activations = load_models(activation_manifest_path, ActivationCaptureArtifact)
    rollout_by_trace = {rollout.trace_id: rollout for rollout in rollouts}
    allowed = set(include_cohorts)
    joined_records: list[JoinedActivationRecord] = []
    for activation in activations:
        rollout = rollout_by_trace.get(activation.trace_id)
        if rollout is None:
            continue
        if rollout.cohort not in allowed:
            continue
        joined_records.append(JoinedActivationRecord(activation=activation, rollout=rollout))
    return joined_records


def resolve_experiment_layer_names(
    joined_records: list[JoinedActivationRecord],
    config: ExperimentOneConfig,
) -> list[str]:
    if config.layer_names is not None:
        return sorted(config.layer_names, key=parse_layer_index)
    layer_names = {
        tensor_name
        for record in joined_records
        for tensor_name in record.activation.tensor_names
        if fnmatchcase(tensor_name, config.layer_name_pattern)
    }
    if not layer_names:
        raise ValueError(
            "No layer names matched the experiment config. "
            "Capture hidden states or provide explicit layer_names."
        )
    return sorted(layer_names, key=parse_layer_index)


def build_layer_feature_bank(
    *,
    joined_records: list[JoinedActivationRecord],
    layer_names: list[str],
    pooling_strategy: PoolingStrategy,
) -> dict[str, dict[str, Any]]:
    import numpy as np

    raw_bank: dict[str, dict[str, list[Any]]] = {
        layer_name: {"features": [], "labels": [], "trace_ids": []} for layer_name in layer_names
    }
    for record in joined_records:
        tensor_bundle = load_activation_tensors(record.activation)
        for layer_name in layer_names:
            tensor = tensor_bundle.get(layer_name)
            if tensor is None:
                continue
            pooled = pool_tensor(
                tensor=tensor,
                prompt_token_count=record.activation.prompt_token_count,
                completion_token_count=record.activation.completion_token_count,
                strategy=pooling_strategy,
            )
            raw_bank[layer_name]["features"].append(pooled.reshape(-1))
            raw_bank[layer_name]["labels"].append(record.rollout.cohort)
            raw_bank[layer_name]["trace_ids"].append(record.rollout.trace_id)

    layer_feature_bank: dict[str, dict[str, Any]] = {}
    for layer_name, rows in raw_bank.items():
        if not rows["features"]:
            continue
        layer_feature_bank[layer_name] = {
            "features": np.stack(rows["features"]),
            "labels": list(rows["labels"]),
            "trace_ids": list(rows["trace_ids"]),
        }
    return layer_feature_bank


def run_layerwise_probe_suite(
    *,
    layer_name: str,
    layer_index: int,
    config: ExperimentOneConfig,
    comparisons: list[ExperimentOneComparisonConfig],
    features: Any,
    labels: list[CohortLabel],
) -> list[ExperimentOneComparisonResult]:
    results: list[ExperimentOneComparisonResult] = []
    for comparison in comparisons:
        result = fit_binary_probe(
            comparison=comparison,
            layer_name=layer_name,
            layer_index=layer_index,
            features=features,
            labels=labels,
            min_examples_per_label=config.min_examples_per_label,
            test_size=config.test_size,
            random_state=config.random_state,
            regularization_strength=config.regularization_strength,
            max_iter=config.max_iter,
        )
        results.append(result)
    return results


def fit_binary_probe(
    *,
    comparison: ExperimentOneComparisonConfig,
    layer_name: str,
    layer_index: int,
    features: Any,
    labels: list[CohortLabel],
    min_examples_per_label: int,
    test_size: float,
    random_state: int,
    regularization_strength: float,
    max_iter: int,
) -> ExperimentOneComparisonResult:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    selected_indices = [
        index
        for index, label in enumerate(labels)
        if label in {comparison.positive_label, comparison.negative_label}
    ]
    n_positive = sum(labels[index] == comparison.positive_label for index in selected_indices)
    n_negative = sum(labels[index] == comparison.negative_label for index in selected_indices)
    if n_positive < min_examples_per_label or n_negative < min_examples_per_label:
        return ExperimentOneComparisonResult(
            comparison_name=comparison.name,
            positive_label=comparison.positive_label,
            negative_label=comparison.negative_label,
            layer_name=layer_name,
            layer_index=layer_index,
            n_examples=len(selected_indices),
            skipped_reason="insufficient_examples",
        )

    x = features[selected_indices]
    y = np.asarray(
        [1 if labels[index] == comparison.positive_label else 0 for index in selected_indices],
        dtype=np.int64,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    probe = LogisticRegression(
        C=regularization_strength,
        max_iter=max_iter,
        random_state=random_state,
    )
    probe.fit(x_train_scaled, y_train)
    y_pred = probe.predict(x_test_scaled)
    y_score = probe.predict_proba(x_test_scaled)[:, 1]
    roc_auc = float(roc_auc_score(y_test, y_score)) if len(set(y_test.tolist())) > 1 else None
    return ExperimentOneComparisonResult(
        comparison_name=comparison.name,
        positive_label=comparison.positive_label,
        negative_label=comparison.negative_label,
        layer_name=layer_name,
        layer_index=layer_index,
        accuracy=float(accuracy_score(y_test, y_pred)),
        macro_f1=float(f1_score(y_test, y_pred, average="macro")),
        roc_auc=roc_auc,
        n_train=len(y_train),
        n_test=len(y_test),
        n_examples=len(selected_indices),
    )


def run_layerwise_representation_suite(
    *,
    layer_name: str,
    layer_index: int,
    comparisons: list[ExperimentOneComparisonConfig],
    features: Any,
    labels: list[CohortLabel],
) -> list[ExperimentOneRepresentationResult]:
    import numpy as np

    results: list[ExperimentOneRepresentationResult] = []
    for comparison in comparisons:
        left_rows = features[
            [index for index, label in enumerate(labels) if label == comparison.positive_label]
        ]
        right_rows = features[
            [index for index, label in enumerate(labels) if label == comparison.negative_label]
        ]
        centroid_distance = None
        cka_value = None
        if len(left_rows) > 0 and len(right_rows) > 0:
            centroid_distance = float(
                np.linalg.norm(left_rows.mean(axis=0) - right_rows.mean(axis=0))
            )
            cka_value = float(linear_cka(left_rows, right_rows))
        results.append(
            ExperimentOneRepresentationResult(
                pair_name=comparison.name,
                left_label=comparison.positive_label,
                right_label=comparison.negative_label,
                layer_name=layer_name,
                layer_index=layer_index,
                centroid_distance=centroid_distance,
                linear_cka=cka_value,
                left_count=len(left_rows),
                right_count=len(right_rows),
            )
        )
    return results


def run_layerwise_clustering(
    *,
    layer_name: str,
    layer_index: int,
    features: Any,
    labels: list[CohortLabel],
    n_clusters: int,
) -> ExperimentOneClusteringResult:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if len(features) < n_clusters or len(set(label.value for label in labels)) < 2:
        return ExperimentOneClusteringResult(
            layer_name=layer_name,
            layer_index=layer_index,
            n_clusters=n_clusters,
            n_examples=len(features),
            skipped_reason="insufficient_examples",
        )
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    assignments = kmeans.fit_predict(features)
    label_histograms: dict[str, dict[str, int]] = defaultdict(dict)
    for cluster_index, label in zip(assignments, labels, strict=True):
        cluster_key = str(cluster_index)
        histogram = label_histograms.setdefault(cluster_key, {})
        histogram[label.value] = histogram.get(label.value, 0) + 1
    purity = clustering_purity(label_histograms)
    silhouette = (
        float(silhouette_score(features, assignments)) if len(features) > n_clusters else None
    )
    return ExperimentOneClusteringResult(
        layer_name=layer_name,
        layer_index=layer_index,
        n_clusters=n_clusters,
        purity=purity,
        silhouette_score=silhouette,
        label_histograms=dict(label_histograms),
        n_examples=len(features),
    )


def summarize_slice_metadata(
    joined_records: list[JoinedActivationRecord],
) -> tuple[Counter[str], dict[str, Counter[str]], dict[str, Counter[str]]]:
    scenario_counts: Counter[str] = Counter()
    cohort_counts_by_scenario: dict[str, Counter[str]] = defaultdict(Counter)
    semantic_failure_counts_by_cohort: dict[str, Counter[str]] = defaultdict(Counter)

    for record in joined_records:
        mech_interp_row = record.rollout.mech_interp_row
        scenario_id = str(mech_interp_row.get("scenario_id") or "unknown")
        cohort_name = record.rollout.cohort.value
        scenario_counts[scenario_id] += 1
        cohort_counts_by_scenario[scenario_id][cohort_name] += 1
        semantic_failure_counts_by_cohort[cohort_name].update(
            normalize_string_labels(mech_interp_row.get("semantic_failures"))
        )

    return scenario_counts, cohort_counts_by_scenario, semantic_failure_counts_by_cohort


def build_experiment_matched_pairs(
    joined_records: list[JoinedActivationRecord],
) -> tuple[list[dict[str, Any]], list[ExperimentOneMatchedPairSummary]]:
    from rewardhack_gym import build_matched_pairs

    pair_rows = build_matched_pairs([record.rollout.mech_interp_row for record in joined_records])
    return pair_rows, summarize_matched_pair_rows(pair_rows)


def summarize_matched_pair_rows(
    pair_rows: list[dict[str, Any]],
) -> list[ExperimentOneMatchedPairSummary]:
    pair_counts_by_level: Counter[str] = Counter()
    pair_groups_by_level: dict[str, set[str]] = defaultdict(set)
    for pair_row in pair_rows:
        match_level = str(pair_row.get("match_level") or "unknown")
        pair_group_id = str(pair_row.get("pair_group_id") or "unknown")
        pair_counts_by_level[match_level] += 1
        pair_groups_by_level[match_level].add(pair_group_id)

    return [
        ExperimentOneMatchedPairSummary(
            match_level=match_level,
            n_pairs=pair_counts_by_level[match_level],
            n_pair_groups=len(pair_groups_by_level[match_level]),
        )
        for match_level in sorted(pair_counts_by_level)
    ]


def clustering_purity(label_histograms: dict[str, dict[str, int]]) -> float:
    total = 0
    dominant = 0
    for histogram in label_histograms.values():
        cluster_total = sum(histogram.values())
        if cluster_total == 0:
            continue
        total += cluster_total
        dominant += max(histogram.values())
    return float(dominant / total) if total > 0 else 0.0


def plot_probe_accuracy_by_layer(
    *,
    comparison_results: list[ExperimentOneComparisonResult],
    output_path: str | Path,
) -> Path:
    import pandas as pd
    import seaborn as sns

    figure_path = ensure_parent_dir(output_path)
    plot_rows = [
        {
            "comparison_name": result.comparison_name,
            "layer_index": result.layer_index,
            "accuracy": result.accuracy,
        }
        for result in comparison_results
        if result.accuracy is not None
    ]
    if not plot_rows:
        raise ValueError("No probe accuracy values were available to plot for Experiment 1.")

    plot_frame = pd.DataFrame(plot_rows).sort_values(
        by=["comparison_name", "layer_index"],
        kind="stable",
    )
    sns.set_theme(style="whitegrid", context="talk")
    axis = sns.lineplot(
        data=plot_frame,
        x="layer_index",
        y="accuracy",
        hue="comparison_name",
        marker="o",
        linewidth=2.5,
    )
    axis.set(
        xlabel="Layer",
        ylabel="Probe Accuracy",
        title="Experiment 1: Probe Accuracy by Layer",
        ylim=(0.0, 1.0),
    )
    axis.figure.tight_layout()
    axis.figure.savefig(figure_path, dpi=200)
    axis.figure.clf()
    return figure_path


def parse_layer_index(layer_name: str) -> int:
    if layer_name.startswith("hidden_state.layer_"):
        return int(layer_name.rsplit("_", maxsplit=1)[-1])
    if layer_name == "hidden_state.embed":
        return -1
    return 10_000


def normalize_string_labels(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []

    normalized_labels: list[str] = []
    for item in value:
        label = str(item)
        if label and label not in normalized_labels:
            normalized_labels.append(label)
    return normalized_labels
