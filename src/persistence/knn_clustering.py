import os
from typing import Literal

import numpy as np

from src.persistence.spectral_clustering import _get_spectral_clustering_folder


def _get_knn_clustering_file_name(
    burst_extraction_params: str or dict,
    spectral_clustering_params: str or dict,
    cv_type: Literal["RepeatedStratifiedKFold", "StratifiedShuffleSplit"] = None,
    special_target: bool = False,
):
    return os.path.join(
        _get_spectral_clustering_folder(
            params_spectral_clustering=spectral_clustering_params,
            params_burst_extraction=burst_extraction_params,
        ),
        f"knn_clustering_results{'_' + cv_type if cv_type else '_'}{'_special_target' if special_target is True else ''}.npz",
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
            burst_extraction_params,
            spectral_clustering_params,
        ),
        allow_pickle=True,
    )
    score = data["score"]
    true_labels = data["true_labels"]
    predicted_labels = data["predicted_labels"]
    class_labels = data["class_labels"]
    relative_votes = data["relative_votes"]
    return score, true_labels, predicted_labels, class_labels, relative_votes


def save_knn_clustering_results_cv(
    burst_extraction_params: str or dict,
    spectral_clustering_params: str or dict,
    cv_type: Literal["RepeatedStratifiedKFold", "StratifiedShuffleSplit"],
    nested_scores,
    all_y_test,
    all_y_pred,
    special_target: bool = False,
):
    np.savez(
        _get_knn_clustering_file_name(
            burst_extraction_params,
            spectral_clustering_params,
            cv_type,
            special_target=special_target,
        ),
        nested_scores=nested_scores,
        all_y_test=all_y_test,
        all_y_pred=all_y_pred,
    )


def load_knn_clustering_results_cv(
    burst_extraction_params: str or dict,
    spectral_clustering_params: str or dict,
    cv_type: Literal["RepeatedStratifiedKFold", "StratifiedShuffleSplit"],
    special_target: bool = False,
):
    data = np.load(
        _get_knn_clustering_file_name(
            burst_extraction_params,
            spectral_clustering_params,
            cv_type,
            special_target=special_target,
        ),
        allow_pickle=True,
    )
    nested_scores = data["nested_scores"]
    all_y_test = data["all_y_test"]
    all_y_pred = data["all_y_pred"]
    return nested_scores, all_y_test, all_y_pred


def exist_knn_clustering_results_cv(
    burst_extraction_params: str or dict,
    spectral_clustering_params: str or dict,
    cv_type: Literal["RepeatedStratifiedKFold", "StratifiedShuffleSplit"],
    special_target: bool = False,
):
    return os.path.exists(
        _get_knn_clustering_file_name(
            burst_extraction_params,
            spectral_clustering_params,
            cv_type,
            special_target=special_target,
        )
    )
