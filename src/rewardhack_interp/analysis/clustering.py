from __future__ import annotations

from collections import defaultdict

from rewardhack_interp.analysis.features import load_feature_table
from rewardhack_interp.config import ClusteringConfig
from rewardhack_interp.io import write_json
from rewardhack_interp.types import ClusteringArtifact


def run_clustering(config: ClusteringConfig) -> ClusteringArtifact:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    features, labels, _trace_ids = load_feature_table(config.features)
    target_name = config.features.target_names[0]

    kmeans = KMeans(n_clusters=config.n_clusters, random_state=config.random_state, n_init="auto")
    assignments = kmeans.fit_predict(features)
    label_histograms: dict[str, dict[str, int]] = defaultdict(dict)
    for cluster_index, label in zip(assignments, labels, strict=True):
        cluster_key = str(cluster_index)
        histogram = label_histograms.setdefault(cluster_key, {})
        histogram[label.value] = histogram.get(label.value, 0) + 1

    silhouette = None
    if len(features) > config.n_clusters:
        silhouette = float(silhouette_score(features, assignments))

    artifact = ClusteringArtifact(
        target_name=target_name,
        pooling_strategy=config.features.pooling_strategy,
        n_clusters=config.n_clusters,
        inertia=float(kmeans.inertia_),
        silhouette_score=silhouette,
        cluster_label_histograms=dict(label_histograms),
    )
    write_json(config.output_path, artifact.model_dump(mode="json"))
    return artifact
