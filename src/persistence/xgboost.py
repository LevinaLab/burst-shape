import os
from typing import Literal

import numpy as np

from src.persistence.spectral_clustering import _get_spectral_clustering_folder


def _get_xgboost_file_name(
    burst_extraction_params: str or dict,
    spectral_clustering_params: str or dict,
    feature_set_name: Literal["shape", "traditional", "combined"],
):
    return os.path.join(
        _get_spectral_clustering_folder(
            params_spectral_clustering=spectral_clustering_params,
            params_burst_extraction=burst_extraction_params,
        ),
        f"xgboost_results_{feature_set_name}.npz",
    )


def exist_xgboost_results(
    burst_extraction_params: str or dict,
    spectral_clustering_params: str or dict,
    feature_set_name: Literal["shape", "traditional", "combined"],
):
    return os.path.exists(
        _get_xgboost_file_name(
            burst_extraction_params, spectral_clustering_params, feature_set_name
        )
    )


def save_xgboost_results(
    burst_extraction_params: str or dict,
    spectral_clustering_params: str or dict,
    feature_set_name: Literal["shape", "traditional", "combined"],
    features,
    nested_scores,
    all_shap_values,
    all_y_pred,
    all_y_test,
):
    np.savez(
        _get_xgboost_file_name(
            burst_extraction_params, spectral_clustering_params, feature_set_name
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
):
    data = np.load(
        _get_xgboost_file_name(
            burst_extraction_params, spectral_clustering_params, feature_set_name
        )
    )
    features = data["features"]
    nested_scores = data["nested_scores"]
    all_shap_values = data["all_shap_values"]
    all_y_pred = data["all_y_pred"]
    all_y_test = data["all_y_test"]
    return features, nested_scores, all_shap_values, all_y_pred, all_y_test
