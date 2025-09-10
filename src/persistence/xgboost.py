import os
from typing import Literal

import numpy as np

from src.persistence.spectral_clustering import _get_spectral_clustering_folder


def _get_xgboost_file_name(
    burst_extraction_params: str or dict,
    spectral_clustering_params: str or dict,
    feature_set_name: Literal["shape", "traditional", "combined"],
    cv_type: Literal[
        "RepeatedStratifiedKFold", "StratifiedShuffleSplit"
    ] = "StratifiedShuffleSplit",
):
    return os.path.join(
        _get_spectral_clustering_folder(
            params_spectral_clustering=spectral_clustering_params,
            params_burst_extraction=burst_extraction_params,
        ),
        f"xgboost_results_{feature_set_name}_{cv_type}.npz",
    )


def exist_xgboost_results(
    burst_extraction_params: str or dict,
    spectral_clustering_params: str or dict,
    feature_set_name: Literal["shape", "traditional", "combined"],
    cv_type: Literal[
        "RepeatedStratifiedKFold", "StratifiedShuffleSplit"
    ] = "StratifiedShuffleSplit",
):
    return os.path.exists(
        _get_xgboost_file_name(
            burst_extraction_params,
            spectral_clustering_params,
            feature_set_name,
            cv_type,
        )
    )


def save_xgboost_results(
    burst_extraction_params: str or dict,
    spectral_clustering_params: str or dict,
    feature_set_name: Literal["shape", "traditional", "combined"],
    cv_type: Literal["RepeatedStratifiedKFold", "StratifiedShuffleSplit"],
    features,
    nested_scores,
    all_shap_values,
    all_y_pred,
    all_y_test,
):
    np.savez(
        _get_xgboost_file_name(
            burst_extraction_params,
            spectral_clustering_params,
            feature_set_name,
            cv_type,
        ),
        features=features,
        nested_scores=nested_scores,
        all_shap_values=all_shap_values,
        all_y_pred=all_y_pred,
        all_y_test=all_y_test,
    )


def load_xgboost_results(
    burst_extraction_params: str or dict,
    spectral_clustering_params: str or dict,
    feature_set_name: Literal["shape", "traditional", "combined"],
    cv_type: Literal[
        "RepeatedStratifiedKFold", "StratifiedShuffleSplit"
    ] = "StratifiedShuffleSplit",
):
    data = np.load(
        _get_xgboost_file_name(
            burst_extraction_params,
            spectral_clustering_params,
            feature_set_name,
            cv_type,
        )
    )
    features = data["features"]
    nested_scores = data["nested_scores"]
    all_shap_values = data["all_shap_values"]
    all_y_pred = data["all_y_pred"]
    all_y_test = data["all_y_test"]
    return features, nested_scores, all_shap_values, all_y_pred, all_y_test
