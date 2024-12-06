import os
from typing import Literal

import numpy as np
from scipy.cluster.hierarchy import fcluster

from src.persistence.burst_extraction import _get_burst_folder
from src.persistence.cross_validation_string import _cv_params_to_string

_agglomerative_clustering_keys = [
    "linkage",
]


def _agglomerative_clustering_params_to_str(agglomerative_clustering_params):
    if isinstance(agglomerative_clustering_params, dict):
        name = "agglomerating_clustering"
        for key, value in agglomerative_clustering_params.items():
            assert (
                key in _agglomerative_clustering_keys
            ), f"key {key} not in {_agglomerative_clustering_keys}"
            name += f"_{key}_{value}"
        return name
    elif isinstance(agglomerative_clustering_params, str):
        return agglomerative_clustering_params
    else:
        raise ValueError(
            "agglomerative_clustering_params should be a dict or a string."
        )


def _get_agglomerative_clustering_folder(
    burst_extraction_params,
    agglomerative_clustering_params,
) -> Literal["str"] | str:
    folder_name = os.path.join(
        _get_burst_folder(burst_extraction_params),
        _agglomerative_clustering_params_to_str(agglomerative_clustering_params),
    )
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def _get_linkage_filename(
    burst_extraction_params,
    agglomerative_clustering_params,
    params_cross_validation=None,
    i_split=None,
):
    if params_cross_validation is not None and i_split is not None:
        if isinstance(params_cross_validation, dict):
            params_cross_validation = _cv_params_to_string(params_cross_validation)
        name = f"linkage_{params_cross_validation}_{i_split}.npy"
    else:
        name = "linkage.npy"
    return os.path.join(
        _get_agglomerative_clustering_folder(
            burst_extraction_params,
            agglomerative_clustering_params,
        ),
        name,
    )


def load_linkage(
    burst_extraction_params,
    agglomerative_clustering_params,
    params_cross_validation=None,
    i_split=None,
):
    """Load linkage."""
    return np.load(
        _get_linkage_filename(
            burst_extraction_params,
            agglomerative_clustering_params,
            params_cross_validation,
            i_split,
        )
    )


def save_linkage(
    linkage,
    burst_extraction_params,
    agglomerative_clustering_params,
    params_cross_validation=None,
    i_split=None,
):
    """Save linkage."""
    np.save(
        _get_linkage_filename(
            burst_extraction_params,
            agglomerative_clustering_params,
            params_cross_validation,
            i_split,
        ),
        linkage,
    )


def linkage_exists(
    burst_extraction_params,
    agglomerative_clustering_params,
    params_cross_validation=None,
    i_split=None,
):
    """Check if linkage exists."""
    return os.path.exists(
        _get_linkage_filename(
            burst_extraction_params,
            agglomerative_clustering_params,
            params_cross_validation,
            i_split,
        )
    )


def get_agglomerative_labels(
    n_clusters,
    burst_extraction_params,
    agglomerative_clustering_params,
    params_cross_validation=None,
    i_split=None,
):
    linkage = load_linkage(
        burst_extraction_params,
        agglomerative_clustering_params,
        params_cross_validation,
        i_split,
    )
    return fcluster(linkage, n_clusters, criterion="maxclust")
