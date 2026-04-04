from __future__ import annotations

from collections import Counter
from pathlib import Path

from rewardhack_interp.analysis.features import load_feature_table
from rewardhack_interp.config import ProbeConfig
from rewardhack_interp.io import write_json
from rewardhack_interp.types import ProbeArtifact, ProbeMetrics
from rewardhack_interp.utils.paths import ensure_parent_dir, to_path_string


def run_probe(config: ProbeConfig) -> ProbeArtifact:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    features, labels, trace_ids = load_feature_table(config.features)
    target_name = config.features.target_names[0]

    positive = set(config.positive_labels)
    negative = set(config.negative_labels)
    filtered_indices = [
        index for index, label in enumerate(labels) if label in positive or label in negative
    ]
    if len(filtered_indices) < 4:
        raise ValueError("Need at least four labeled examples to train a probe.")

    x = features[filtered_indices]
    y = np.asarray(
        [1 if labels[index] in positive else 0 for index in filtered_indices],
        dtype=np.int64,
    )
    selected_trace_ids = [trace_ids[index] for index in filtered_indices]
    _ = selected_trace_ids

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    probe = LogisticRegression(
        C=config.regularization_strength,
        max_iter=config.max_iter,
        random_state=config.random_state,
    )
    probe.fit(x_train_scaled, y_train)
    y_pred = probe.predict(x_test_scaled)
    y_score = probe.predict_proba(x_test_scaled)[:, 1]

    metrics = ProbeMetrics(
        accuracy=float(accuracy_score(y_test, y_pred)),
        macro_f1=float(f1_score(y_test, y_pred, average="macro")),
        roc_auc=float(roc_auc_score(y_test, y_score)) if len(set(y_test.tolist())) > 1 else None,
        n_train=len(y_train),
        n_test=len(y_test),
    )

    coefficients_path = ensure_parent_dir(Path(config.output_path).with_suffix(".coefficients.npy"))
    np.save(coefficients_path, probe.coef_)

    artifact = ProbeArtifact(
        target_name=target_name,
        pooling_strategy=config.features.pooling_strategy,
        positive_labels=list(config.positive_labels),
        negative_labels=list(config.negative_labels),
        metrics=metrics,
        label_counts=dict(Counter(label.value for label in labels)),
        coefficients_path=to_path_string(coefficients_path),
    )
    write_json(config.output_path, artifact.model_dump(mode="json"))
    return artifact
