import os
from typing import Literal

import numpy as np

from src.persistence.spectral_clustering import _get_spectral_clustering_folder


def _get_knn_clustering_file_name(
    burst_extraction_params: str or dict,
    spectral_clustering_params: str or dict,
):
    return os.path.join(
        _get_spectral_clustering_folder(
            params_spectral_clustering=spectral_clustering_params,
            params_burst_extraction=burst_extraction_params,
        ),
        "knn_clustering_results.npz",
    )


def exist_knn_clustering_results(
    burst_extraction_params: str or dict,
    spectral_clustering_params: str or dict,
    feature_set_name: Literal["shape", "traditional", "combined"],
):
    return os.path.exists(
        _get_knn_clustering_file_name(
            burst_extraction_params, spectral_clustering_params, feature_set_name
        )
    )


def save_knn_clustering_results(
    burst_extraction_params: str or dict,
    spectral_clustering_params: str or dict,
    score,
    true_labels,
    predicted_labels,
    class_labels,
    relative_votes,
):
    np.savez(
        _get_knn_clustering_file_name(
            burst_extraction_params, spectral_clustering_params
        ),
        score=score,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        class_labels=class_labels,
        relative_votes=relative_votes,
    )


def load_knn_clustering_results(
    burst_extraction_params: str or dict,
    spectral_clustering_params: str or dict,
):
    data = np.load(
        _get_knn_clustering_file_name(
            burst_extraction_params, spectral_clustering_params
        ),
        allow_pickle=True,
    )
    score = data["score"]
    true_labels = data["true_labels"]
    predicted_labels = data["predicted_labels"]
    class_labels = data["class_labels"]
    relative_votes = data["relative_votes"]
    return score, true_labels, predicted_labels, class_labels, relative_votes
