from __future__ import annotations

from collections import Counter
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from rewardhack_interp.activations import load_activation_tensors
from rewardhack_interp.config import LayerwiseProbeConfig
from rewardhack_interp.io import load_models, write_json
from rewardhack_interp.tracking import start_wandb_run
from rewardhack_interp.types import (
    ActivationCaptureArtifact,
    CohortLabel,
    LayerwiseBinaryProbeArtifact,
    LayerwiseBinaryProbeResult,
)
from rewardhack_interp.utils.paths import ensure_parent_dir, to_path_string


def run_layerwise_probe(config: LayerwiseProbeConfig) -> LayerwiseBinaryProbeArtifact:
    with start_wandb_run(
        wandb_config=config.wandb,
        run_name=Path(config.output_path).stem,
        job_type="layerwise_probe",
        config_payload=config.model_dump(mode="json"),
    ) as tracker:
        artifacts = load_models(config.activation_manifest_path, ActivationCaptureArtifact)
        selected_artifacts = [
            artifact
            for artifact in artifacts
            if artifact.cohort in {config.positive_label, config.negative_label}
        ]
        if len(selected_artifacts) < 4:
            raise ValueError("Need at least four activation artifacts for layerwise probing.")

        layer_names = resolve_layer_names(selected_artifacts, config)
        layer_rows = build_layer_feature_rows(selected_artifacts, layer_names, config)
        results = [
            fit_layerwise_binary_probe(
                layer_name=layer_name,
                layer_index=parse_layer_index(layer_name),
                features=layer_rows[layer_name]["features"],
                labels=layer_rows[layer_name]["labels"],
                positive_label=config.positive_label,
                negative_label=config.negative_label,
                test_size=config.test_size,
                random_state=config.random_state,
                regularization_strength=config.regularization_strength,
                max_iter=config.max_iter,
            )
            for layer_name in layer_names
            if layer_name in layer_rows
        ]

        report_path = ensure_parent_dir(config.output_path)
        accuracy_plot_path = report_path.with_suffix(".accuracy.png")
        roc_auc_plot_path = report_path.with_suffix(".roc_auc.png")
        plot_paths = {
            "accuracy": to_path_string(
                plot_layerwise_metric(
                    results=results,
                    metric_name="accuracy",
                    output_path=accuracy_plot_path,
                    title="Reward Hack vs Genuine Failure Accuracy by Layer",
                    ylabel="Accuracy",
                )
            ),
            "roc_auc": to_path_string(
                plot_layerwise_metric(
                    results=results,
                    metric_name="roc_auc",
                    output_path=roc_auc_plot_path,
                    title="Reward Hack vs Genuine Failure AUROC by Layer",
                    ylabel="AUROC",
                )
            ),
        }
        artifact = LayerwiseBinaryProbeArtifact(
            activation_manifest_path=to_path_string(config.activation_manifest_path),
            positive_label=config.positive_label,
            negative_label=config.negative_label,
            pooling_strategy=config.pooling_strategy,
            label_counts=dict(
                Counter(artifact.cohort.value for artifact in selected_artifacts)
            ),
            results=results,
            report_path=to_path_string(report_path),
            plot_paths=plot_paths,
        )
        write_json(report_path, artifact.model_dump(mode="json"))
        tracker.log_summary(
            {
                "label_counts": artifact.label_counts,
                "num_layers": len(artifact.results),
                "best_accuracy": best_metric_value(artifact.results, "accuracy"),
                "best_roc_auc": best_metric_value(artifact.results, "roc_auc"),
            }
        )
        tracker.log_path(report_path, artifact_type="layerwise-probe-report")
        tracker.log_path(accuracy_plot_path, artifact_type="layerwise-probe-plot")
        tracker.log_path(roc_auc_plot_path, artifact_type="layerwise-probe-plot")
        return artifact


def resolve_layer_names(
    artifacts: list[ActivationCaptureArtifact],
    config: LayerwiseProbeConfig,
) -> list[str]:
    if config.layer_names is not None:
        return sorted(config.layer_names, key=parse_layer_index)
    layer_names = {
        tensor_name
        for artifact in artifacts
        for tensor_name in artifact.tensor_names
        if fnmatchcase(tensor_name, config.layer_name_pattern)
    }
    if not layer_names:
        raise ValueError("No layer names matched the layerwise probe config.")
    return sorted(layer_names, key=parse_layer_index)


def build_layer_feature_rows(
    artifacts: list[ActivationCaptureArtifact],
    layer_names: list[str],
    config: LayerwiseProbeConfig,
) -> dict[str, dict[str, Any]]:
    import numpy as np

    from rewardhack_interp.analysis.features import pool_tensor

    feature_rows: dict[str, dict[str, list[Any]]] = {
        layer_name: {"features": [], "labels": []} for layer_name in layer_names
    }
    for artifact in artifacts:
        tensor_bundle = load_activation_tensors(artifact)
        for layer_name in layer_names:
            tensor = tensor_bundle.get(layer_name)
            if tensor is None:
                continue
            pooled = pool_tensor(
                tensor=tensor,
                prompt_token_count=artifact.prompt_token_count,
                completion_token_count=artifact.completion_token_count,
                strategy=config.pooling_strategy,
            )
            feature_rows[layer_name]["features"].append(pooled.reshape(-1))
            feature_rows[layer_name]["labels"].append(artifact.cohort)

    return {
        layer_name: {
            "features": np.stack(rows["features"]),
            "labels": list(rows["labels"]),
        }
        for layer_name, rows in feature_rows.items()
        if rows["features"]
    }


def fit_layerwise_binary_probe(
    *,
    layer_name: str,
    layer_index: int,
    features: Any,
    labels: list[CohortLabel],
    positive_label: CohortLabel,
    negative_label: CohortLabel,
    test_size: float,
    random_state: int,
    regularization_strength: float,
    max_iter: int,
) -> LayerwiseBinaryProbeResult:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    selected_indices = [
        index for index, label in enumerate(labels) if label in {positive_label, negative_label}
    ]
    if len(selected_indices) < 4:
        return LayerwiseBinaryProbeResult(
            layer_name=layer_name,
            layer_index=layer_index,
            n_examples=len(selected_indices),
            skipped_reason="insufficient_examples",
        )

    x = features[selected_indices]
    y = np.asarray(
        [1 if labels[index] == positive_label else 0 for index in selected_indices],
        dtype=np.int64,
    )
    if len(set(y.tolist())) < 2:
        return LayerwiseBinaryProbeResult(
            layer_name=layer_name,
            layer_index=layer_index,
            n_examples=len(selected_indices),
            skipped_reason="single_class",
        )

    test_count = max(round(len(y) * test_size), 2)
    if len(y) - test_count < 2:
        return LayerwiseBinaryProbeResult(
            layer_name=layer_name,
            layer_index=layer_index,
            n_examples=len(selected_indices),
            skipped_reason="insufficient_examples",
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_count,
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

    return LayerwiseBinaryProbeResult(
        layer_name=layer_name,
        layer_index=layer_index,
        accuracy=float(accuracy_score(y_test, y_pred)),
        macro_f1=float(f1_score(y_test, y_pred, average="macro")),
        roc_auc=float(roc_auc_score(y_test, y_score))
        if len(set(y_test.tolist())) > 1
        else None,
        n_train=len(y_train),
        n_test=len(y_test),
        n_examples=len(selected_indices),
    )


def plot_layerwise_metric(
    *,
    results: list[LayerwiseBinaryProbeResult],
    metric_name: str,
    output_path: str | Path,
    title: str,
    ylabel: str,
) -> Path:
    import pandas as pd
    import seaborn as sns

    rows = [
        {
            "layer_index": result.layer_index,
            "value": getattr(result, metric_name),
        }
        for result in results
        if getattr(result, metric_name) is not None
    ]
    output = ensure_parent_dir(output_path)
    if not rows:
        return output

    frame = pd.DataFrame(rows).sort_values("layer_index")
    sns.set_theme(style="whitegrid", context="talk")
    axis = sns.lineplot(data=frame, x="layer_index", y="value", marker="o", linewidth=2.5)
    axis.set(xlabel="Layer", ylabel=ylabel, title=title, ylim=(0.0, 1.0))
    axis.figure.tight_layout()
    axis.figure.savefig(output, dpi=200)
    axis.figure.clf()
    return output


def parse_layer_index(layer_name: str) -> int:
    if layer_name.startswith("hidden_state.layer_"):
        return int(layer_name.rsplit("_", maxsplit=1)[-1])
    if layer_name == "hidden_state.embed":
        return -1
    return 10_000


def best_metric_value(results: list[LayerwiseBinaryProbeResult], metric_name: str) -> float | None:
    values = [
        float(metric_value)
        for result in results
        if (metric_value := getattr(result, metric_name)) is not None
    ]
    return max(values) if values else None
