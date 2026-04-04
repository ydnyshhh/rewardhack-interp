from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

from rewardhack_interp.analysis.features import load_feature_table
from rewardhack_interp.config import RepresentationConfig
from rewardhack_interp.io import write_json
from rewardhack_interp.tracking import start_wandb_run
from rewardhack_interp.types import RepresentationArtifact


def run_representation_comparison(config: RepresentationConfig) -> RepresentationArtifact:
    import numpy as np

    with start_wandb_run(
        wandb_config=config.wandb,
        run_name=Path(config.output_path).stem,
        job_type="representation_comparison",
        config_payload=config.model_dump(mode="json"),
    ) as tracker:
        features, labels, _trace_ids = load_feature_table(config.features)
        target_name = config.features.target_names[0]

        grouped: dict[str, Any] = {}
        for cohort in sorted({label.value for label in labels}):
            cohort_rows = features[
                [index for index, label in enumerate(labels) if label.value == cohort]
            ]
            if len(cohort_rows) > 0:
                grouped[cohort] = cohort_rows

        centroids = {cohort: rows.mean(axis=0) for cohort, rows in grouped.items()}
        centroid_distances = {
            f"{left}__{right}": float(np.linalg.norm(centroids[left] - centroids[right]))
            for left, right in combinations(sorted(centroids), 2)
        }
        linear_cka_scores = {
            f"{left}__{right}": float(linear_cka(grouped[left], grouped[right]))
            for left, right in combinations(sorted(grouped), 2)
        }
        within_group_variance = {
            cohort: float(np.var(rows, axis=0).mean()) for cohort, rows in grouped.items()
        }
        artifact = RepresentationArtifact(
            target_name=target_name,
            pooling_strategy=config.features.pooling_strategy,
            counts_by_cohort={cohort: len(rows) for cohort, rows in grouped.items()},
            centroid_distances=centroid_distances,
            linear_cka=linear_cka_scores,
            within_group_variance=within_group_variance,
        )
        write_json(config.output_path, artifact.model_dump(mode="json"))
        tracker.log_summary(artifact.model_dump(mode="json"))
        tracker.log_path(config.output_path, artifact_type="representation-report")
        return artifact


def linear_cka(x: Any, y: Any) -> float:
    import numpy as np

    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean(axis=0, keepdims=True)
    cross = x_centered.T @ y_centered
    hsic_xy = float(np.linalg.norm(cross, ord="fro") ** 2)
    hsic_xx = float(np.linalg.norm(x_centered.T @ x_centered, ord="fro") ** 2)
    hsic_yy = float(np.linalg.norm(y_centered.T @ y_centered, ord="fro") ** 2)
    denominator = (hsic_xx ** 0.5) * (hsic_yy ** 0.5)
    if denominator == 0.0:
        return 0.0
    return hsic_xy / denominator
