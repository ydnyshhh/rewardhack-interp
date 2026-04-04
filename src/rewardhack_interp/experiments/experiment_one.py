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
    ExperimentOneSplitStrategy,
    PoolingStrategy,
    TrajectoryArtifact,
)
from rewardhack_interp.utils.identifiers import activation_sample_id, trajectory_sample_id
from rewardhack_interp.utils.paths import ensure_dir, ensure_parent_dir, to_path_string


@dataclass(frozen=True, slots=True)
class JoinedActivationRecord:
    activation: ActivationCaptureArtifact
    rollout: TrajectoryArtifact


@dataclass(frozen=True, slots=True)
class ProbeSplit:
    split_random_seed: int
    train_indices: list[int]
    test_indices: list[int]


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
                    task_seeds=layer_rows["task_seeds"],
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
            split_strategy=config.split_strategy,
            train_task_seeds=sorted(config.train_task_seeds or []),
            eval_task_seeds=sorted(config.eval_task_seeds or []),
            split_random_seeds=list(config.split_random_seeds),
            confidence_level=config.confidence_level,
            bootstrap_repeats=config.bootstrap_repeats,
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
                "split_strategy": artifact.split_strategy.value,
                "confidence_level": artifact.confidence_level,
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
    rollout_by_sample_id = {
        trajectory_sample_id(rollout): rollout for rollout in rollouts
    }
    allowed = set(include_cohorts)
    joined_records: list[JoinedActivationRecord] = []
    for activation in activations:
        rollout = rollout_by_sample_id.get(activation_sample_id(activation))
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
        layer_name: {"features": [], "labels": [], "trace_ids": [], "task_seeds": []}
        for layer_name in layer_names
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
            raw_bank[layer_name]["trace_ids"].append(trajectory_sample_id(record.rollout))
            raw_bank[layer_name]["task_seeds"].append(record.rollout.task.task_seed)

    layer_feature_bank: dict[str, dict[str, Any]] = {}
    for layer_name, rows in raw_bank.items():
        if not rows["features"]:
            continue
        layer_feature_bank[layer_name] = {
            "features": np.stack(rows["features"]),
            "labels": list(rows["labels"]),
            "trace_ids": list(rows["trace_ids"]),
            "task_seeds": list(rows["task_seeds"]),
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
    task_seeds: list[int],
) -> list[ExperimentOneComparisonResult]:
    results: list[ExperimentOneComparisonResult] = []
    for comparison in comparisons:
        result = fit_binary_probe(
            comparison=comparison,
            layer_name=layer_name,
            layer_index=layer_index,
            features=features,
            labels=labels,
            task_seeds=task_seeds,
            min_examples_per_label=config.min_examples_per_label,
            test_size=config.test_size,
            regularization_strength=config.regularization_strength,
            max_iter=config.max_iter,
            split_strategy=config.split_strategy,
            split_random_seeds=config.split_random_seeds,
            train_task_seeds=config.train_task_seeds,
            eval_task_seeds=config.eval_task_seeds,
            confidence_level=config.confidence_level,
            bootstrap_repeats=config.bootstrap_repeats,
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
    task_seeds: list[int],
    min_examples_per_label: int,
    test_size: float,
    regularization_strength: float,
    max_iter: int,
    split_strategy: ExperimentOneSplitStrategy,
    split_random_seeds: list[int],
    train_task_seeds: list[int] | None,
    eval_task_seeds: list[int] | None,
    confidence_level: float,
    bootstrap_repeats: int,
) -> ExperimentOneComparisonResult:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
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
            split_strategy=split_strategy,
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
    selected_task_seeds = [task_seeds[index] for index in selected_indices]
    probe_splits = build_probe_splits(
        y=y,
        task_seeds=selected_task_seeds,
        test_size=test_size,
        split_strategy=split_strategy,
        split_random_seeds=split_random_seeds,
        train_task_seeds=train_task_seeds,
        eval_task_seeds=eval_task_seeds,
    )
    if not probe_splits:
        return ExperimentOneComparisonResult(
            comparison_name=comparison.name,
            positive_label=comparison.positive_label,
            negative_label=comparison.negative_label,
            split_strategy=split_strategy,
            layer_name=layer_name,
            layer_index=layer_index,
            n_examples=len(selected_indices),
            skipped_reason="invalid_holdout_split",
        )

    accuracy_values: list[float] = []
    macro_f1_values: list[float] = []
    roc_auc_values: list[float] = []
    split_train_sizes: list[int] = []
    split_test_sizes: list[int] = []
    repeat_random_seeds: list[int] = []

    bootstrap_accuracy_values: list[float] = []
    bootstrap_macro_f1_values: list[float] = []
    bootstrap_roc_auc_values: list[float] = []

    for probe_split in probe_splits:
        train_indices = np.asarray(probe_split.train_indices, dtype=np.int64)
        test_indices = np.asarray(probe_split.test_indices, dtype=np.int64)
        x_train = x[train_indices]
        x_test = x[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]

        if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
            continue

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        probe = LogisticRegression(
            C=regularization_strength,
            max_iter=max_iter,
            random_state=probe_split.split_random_seed,
        )
        probe.fit(x_train_scaled, y_train)
        y_pred = probe.predict(x_test_scaled)
        y_score = probe.predict_proba(x_test_scaled)[:, 1]

        accuracy_values.append(float(accuracy_score(y_test, y_pred)))
        macro_f1_values.append(float(f1_score(y_test, y_pred, average="macro")))
        if len(set(y_test.tolist())) > 1:
            roc_auc_values.append(float(roc_auc_score(y_test, y_score)))
        split_train_sizes.append(len(y_train))
        split_test_sizes.append(len(y_test))
        repeat_random_seeds.append(probe_split.split_random_seed)

        if len(probe_splits) == 1 and bootstrap_repeats > 0:
            bootstrap_metrics = bootstrap_probe_metrics(
                y_test=y_test,
                y_pred=y_pred,
                y_score=y_score,
                n_repeats=bootstrap_repeats,
                random_state=probe_split.split_random_seed,
            )
            bootstrap_accuracy_values = bootstrap_metrics["accuracy"]
            bootstrap_macro_f1_values = bootstrap_metrics["macro_f1"]
            bootstrap_roc_auc_values = bootstrap_metrics["roc_auc"]

    if not accuracy_values:
        return ExperimentOneComparisonResult(
            comparison_name=comparison.name,
            positive_label=comparison.positive_label,
            negative_label=comparison.negative_label,
            split_strategy=split_strategy,
            layer_name=layer_name,
            layer_index=layer_index,
            n_examples=len(selected_indices),
            skipped_reason="degenerate_split",
        )

    accuracy_summary = summarize_metric_distribution(
        values=bootstrap_accuracy_values or accuracy_values,
        confidence_level=confidence_level,
        point_estimate=float(np.mean(accuracy_values)),
    )
    macro_f1_summary = summarize_metric_distribution(
        values=bootstrap_macro_f1_values or macro_f1_values,
        confidence_level=confidence_level,
        point_estimate=float(np.mean(macro_f1_values)),
    )
    roc_auc_summary = summarize_metric_distribution(
        values=bootstrap_roc_auc_values or roc_auc_values,
        confidence_level=confidence_level,
        point_estimate=float(np.mean(roc_auc_values)) if roc_auc_values else None,
    )

    return ExperimentOneComparisonResult(
        comparison_name=comparison.name,
        positive_label=comparison.positive_label,
        negative_label=comparison.negative_label,
        split_strategy=split_strategy,
        layer_name=layer_name,
        layer_index=layer_index,
        accuracy=accuracy_summary["mean"],
        accuracy_std=accuracy_summary["std"],
        accuracy_ci_low=accuracy_summary["ci_low"],
        accuracy_ci_high=accuracy_summary["ci_high"],
        macro_f1=macro_f1_summary["mean"],
        macro_f1_std=macro_f1_summary["std"],
        macro_f1_ci_low=macro_f1_summary["ci_low"],
        macro_f1_ci_high=macro_f1_summary["ci_high"],
        roc_auc=roc_auc_summary["mean"],
        roc_auc_std=roc_auc_summary["std"],
        roc_auc_ci_low=roc_auc_summary["ci_low"],
        roc_auc_ci_high=roc_auc_summary["ci_high"],
        n_train=round(sum(split_train_sizes) / len(split_train_sizes)),
        n_test=round(sum(split_test_sizes) / len(split_test_sizes)),
        n_examples=len(selected_indices),
        num_repeats=len(accuracy_values),
        repeat_random_seeds=repeat_random_seeds,
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


def build_probe_splits(
    *,
    y: Any,
    task_seeds: list[int],
    test_size: float,
    split_strategy: ExperimentOneSplitStrategy,
    split_random_seeds: list[int],
    train_task_seeds: list[int] | None,
    eval_task_seeds: list[int] | None,
) -> list[ProbeSplit]:
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    explicit_train = set(train_task_seeds or [])
    explicit_eval = set(eval_task_seeds or [])
    row_indices = np.arange(len(y), dtype=np.int64)

    if explicit_train or explicit_eval:
        train_indices: list[int] = []
        test_indices: list[int] = []
        for row_index, task_seed in enumerate(task_seeds):
            in_train = task_seed in explicit_train
            in_eval = task_seed in explicit_eval
            if explicit_train and explicit_eval:
                if in_train:
                    train_indices.append(row_index)
                elif in_eval:
                    test_indices.append(row_index)
            elif explicit_eval:
                if in_eval:
                    test_indices.append(row_index)
                else:
                    train_indices.append(row_index)
            else:
                if in_train:
                    train_indices.append(row_index)
                else:
                    test_indices.append(row_index)

        return validate_probe_splits(
            y=y,
            splits=[
                ProbeSplit(
                    split_random_seed=-1,
                    train_indices=train_indices,
                    test_indices=test_indices,
                )
            ],
        )

    if split_strategy == ExperimentOneSplitStrategy.task_seed_holdout:
        splits = []
        groups = np.asarray(task_seeds, dtype=np.int64)
        for split_seed in split_random_seeds:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=split_seed,
            )
            train_indices, test_indices = next(splitter.split(row_indices, y=y, groups=groups))
            splits.append(
                ProbeSplit(
                    split_random_seed=split_seed,
                    train_indices=train_indices.tolist(),
                    test_indices=test_indices.tolist(),
                )
            )
        return validate_probe_splits(y=y, splits=splits)

    splits = []
    for split_seed in split_random_seeds:
        train_indices, test_indices = train_test_split(
            row_indices,
            test_size=test_size,
            random_state=split_seed,
            stratify=y,
        )
        splits.append(
            ProbeSplit(
                split_random_seed=split_seed,
                train_indices=train_indices.tolist(),
                test_indices=test_indices.tolist(),
            )
        )
    return validate_probe_splits(y=y, splits=splits)


def validate_probe_splits(*, y: Any, splits: list[ProbeSplit]) -> list[ProbeSplit]:
    import numpy as np

    valid_splits: list[ProbeSplit] = []
    for probe_split in splits:
        train_indices = np.asarray(probe_split.train_indices, dtype=np.int64)
        test_indices = np.asarray(probe_split.test_indices, dtype=np.int64)
        if len(train_indices) == 0 or len(test_indices) == 0:
            continue
        y_train = y[train_indices]
        y_test = y[test_indices]
        if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
            continue
        valid_splits.append(probe_split)
    return valid_splits


def bootstrap_probe_metrics(
    *,
    y_test: Any,
    y_pred: Any,
    y_score: Any,
    n_repeats: int,
    random_state: int,
) -> dict[str, list[float]]:
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    bootstrap = {"accuracy": [], "macro_f1": [], "roc_auc": []}
    if n_repeats <= 0 or len(y_test) == 0:
        return bootstrap

    rng = np.random.default_rng(random_state)
    test_size = len(y_test)
    for _ in range(n_repeats):
        sample_indices = rng.integers(0, test_size, size=test_size)
        sampled_y_test = y_test[sample_indices]
        sampled_y_pred = y_pred[sample_indices]
        sampled_y_score = y_score[sample_indices]
        bootstrap["accuracy"].append(float(accuracy_score(sampled_y_test, sampled_y_pred)))
        bootstrap["macro_f1"].append(
            float(f1_score(sampled_y_test, sampled_y_pred, average="macro"))
        )
        if len(set(sampled_y_test.tolist())) > 1:
            bootstrap["roc_auc"].append(float(roc_auc_score(sampled_y_test, sampled_y_score)))
    return bootstrap


def summarize_metric_distribution(
    *,
    values: list[float],
    confidence_level: float,
    point_estimate: float | None,
) -> dict[str, float | None]:
    import numpy as np

    if point_estimate is None or not values:
        return {"mean": point_estimate, "std": None, "ci_low": None, "ci_high": None}

    distribution = np.asarray(values, dtype=np.float64)
    alpha = (1.0 - confidence_level) / 2.0
    ci_low, ci_high = np.quantile(distribution, [alpha, 1.0 - alpha]).tolist()
    std_value = float(distribution.std(ddof=1)) if len(distribution) > 1 else 0.0
    return {
        "mean": float(point_estimate),
        "std": std_value,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


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
            "accuracy_ci_low": result.accuracy_ci_low,
            "accuracy_ci_high": result.accuracy_ci_high,
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
    for comparison_name, grouped_frame in plot_frame.groupby("comparison_name", sort=False):
        if grouped_frame["accuracy_ci_low"].notna().all() and grouped_frame[
            "accuracy_ci_high"
        ].notna().all():
            line = next(
                (
                    line_candidate
                    for line_candidate in axis.lines
                    if line_candidate.get_label() == comparison_name
                ),
                None,
            )
            fill_color = line.get_color() if line is not None else None
            axis.fill_between(
                grouped_frame["layer_index"],
                grouped_frame["accuracy_ci_low"],
                grouped_frame["accuracy_ci_high"],
                alpha=0.15,
                color=fill_color,
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
