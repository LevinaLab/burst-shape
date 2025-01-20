import json
import os
import pickle

import numpy as np
import scipy.sparse

from .burst_extraction import _get_burst_folder
from .cross_validation_string import _cv_params_to_string

_spectral_clustering_defaults: dict = {
    "n_components_max": 30,
    "affinity": "nearest_neighbors",
    "metric": None,
    "n_neighbors": 10,
    "random_state": 0,
}

_labels_defaults: dict = {
    "n_clusters_min": 2,
    "n_clusters_max": 30,
    "assign_labels": "cluster_qr",
    "random_state": 0,
}


def _spectral_clustering_params_to_str(params):
    name = "spectral"
    for key, value in params.items():
        if (
            key in _spectral_clustering_defaults
            and value != _spectral_clustering_defaults[key]
        ):
            name += f"_{key}_{value}"
    return name


def _get_spectral_clustering_folder(
    params_spectral_clustering: str or dict, params_burst_extraction: str or dict
):
    if isinstance(params_spectral_clustering, dict):
        params_spectral_clustering = _spectral_clustering_params_to_str(
            params_spectral_clustering
        )
    path_burst_extraction = _get_burst_folder(params_burst_extraction)
    return os.path.join(path_burst_extraction, params_spectral_clustering)


def save_clustering_params(params, params_burst_extraction):
    save_folder = _get_spectral_clustering_folder(params, params_burst_extraction)
    os.makedirs(save_folder, exist_ok=True)
    with open(
        os.path.join(save_folder, "clustering_params.json"),
        "w",
    ) as f:
        json.dump(params, f, indent=4)


def save_clustering_maps(
    clustering,
    params_spectral_clustering,
    params_burst_extraction,
    params_cross_validation=None,
    i_split=None,
):
    save_folder = _get_spectral_clustering_folder(
        params_spectral_clustering, params_burst_extraction
    )
    os.makedirs(save_folder, exist_ok=True)
    if params_cross_validation is not None and i_split is not None:
        cv_string = _cv_params_to_string(params_cross_validation, i_split)
        name = f"clustering_maps_{cv_string}.pkl"
    else:
        name = "clustering_maps.pkl"
    with open(os.path.join(save_folder, name), "wb") as f:
        pickle.dump(clustering, f)


def load_clustering_maps(
    params_spectral_clustering,
    params_burst_extraction,
    params_cross_validation=None,
    i_split=None,
):
    save_folder = _get_spectral_clustering_folder(
        params_spectral_clustering, params_burst_extraction
    )
    if params_cross_validation is not None and i_split is not None:
        cv_string = _cv_params_to_string(params_cross_validation, i_split)
        name = f"clustering_maps_{cv_string}.pkl"
    else:
        name = "clustering_maps.pkl"
    with open(os.path.join(save_folder, name), "rb") as f:
        clustering = pickle.load(f)
    return clustering


def _labels_params_to_str(params: str or dict):
    if isinstance(params, str):
        return params
    else:
        name = "labels"
        for key, value in params.items():
            if key in _labels_defaults and value != _labels_defaults[key]:
                name += f"_{key}_{value}"
        return name


def _get_labels_params_file(params_labels):
    name = _labels_params_to_str(params_labels)
    name += "_params.json"
    return name


def save_labels_params(
    params_labels, params_spectral_clustering, params_burst_extraction
):
    save_folder = _get_spectral_clustering_folder(
        params_spectral_clustering, params_burst_extraction
    )
    with open(
        os.path.join(save_folder, _get_labels_params_file(params_labels)), "w"
    ) as f:
        json.dump(params_labels, f, indent=4)


def save_clustering_labels(
    clustering,
    params_spectral_clustering,
    params_burst_extraction,
    params_labels,
    params_cross_validation=None,
    i_split=None,
):
    save_folder = _get_spectral_clustering_folder(
        params_spectral_clustering, params_burst_extraction
    )
    os.makedirs(save_folder, exist_ok=True)
    labels_string = _labels_params_to_str(params_labels)
    if params_cross_validation is not None and i_split is not None:
        cv_string = _cv_params_to_string(params_cross_validation, i_split)
        name = f"clustering_{labels_string}_{cv_string}.pkl"
    else:
        name = f"clustering_{labels_string}.pkl"
    with open(os.path.join(save_folder, name), "wb") as f:
        pickle.dump(clustering, f)


def load_clustering_labels(
    params_spectral_clustering,
    params_burst_extraction,
    params_labels,
    params_cross_validation=None,
    i_split=None,
):
    save_folder = _get_spectral_clustering_folder(
        params_spectral_clustering, params_burst_extraction
    )
    labels_string = _labels_params_to_str(params_labels)
    if params_cross_validation is not None and i_split is not None:
        cv_string = _cv_params_to_string(params_cross_validation, i_split)
        name = f"clustering_{labels_string}_{cv_string}.pkl"
    else:
        name = f"clustering_{labels_string}.pkl"
    with open(os.path.join(save_folder, name), "rb") as f:
        clustering = pickle.load(f)
    return clustering


def _get_path_affinity_matrix(
    params_burst_extraction,
    params_spectral_clustering,
    params_cross_validation=None,
    i_split=None,
):
    save_folder = _get_spectral_clustering_folder(
        params_spectral_clustering, params_burst_extraction
    )
    name = "affinity_matrix"
    assert isinstance(params_spectral_clustering, dict)
    keys = ["metric", "n_neighbors"]
    for key in keys:
        if _spectral_clustering_defaults[key] != params_spectral_clustering[key]:
            name += f"_{key}_{params_spectral_clustering[key]}"
    if params_cross_validation is not None and i_split is not None:
        cv_string = _cv_params_to_string(params_cross_validation, i_split)
        name += f"_{cv_string}"
    name += ".npz"
    return os.path.join(save_folder, name)


def save_affinity_matrix(
    affinity_matrix,
    params_spectral_clustering,
    params_burst_extraction,
    params_cross_validation=None,
    i_split=None,
):
    path = _get_path_affinity_matrix(
        params_burst_extraction,
        params_spectral_clustering,
        params_cross_validation,
        i_split,
    )
    with open(path, "wb") as f:
        scipy.sparse.save_npz(f, affinity_matrix)


def load_affinity_matrix(
    params_spectral_clustering,
    params_burst_extraction,
    params_cross_validation=None,
    i_split=None,
):
    path = _get_path_affinity_matrix(
        params_burst_extraction,
        params_spectral_clustering,
        params_cross_validation,
        i_split,
    )
    with open(path, "rb") as f:
        affinity_matrix = scipy.sparse.load_npz(f)
    return affinity_matrix


def _get_spectral_embedding_file(burst_extraction_params, params_spectral_clustering):
    return os.path.join(
        _get_spectral_clustering_folder(
            params_spectral_clustering, burst_extraction_params
        ),
        "spectral_embedding.npy",
    )


def load_spectral_embedding(
    burst_extraction_params,
    params_spectral_clustering,
):
    """Load spectral embedding."""
    return np.load(
        _get_spectral_embedding_file(
            burst_extraction_params,
            params_spectral_clustering,
        )
    )


def save_spectral_embedding(
    spectral_embedding,
    burst_extraction_params,
    params_spectral_clustering,
):
    """Save spectral embedding."""
    np.save(
        _get_spectral_embedding_file(
            burst_extraction_params,
            params_spectral_clustering,
        ),
        spectral_embedding,
    )


def spectral_embedding_exists(
    burst_extraction_params,
    params_spectral_clustering,
):
    """Check if spectral embedding exists."""
    return os.path.exists(
        _get_spectral_embedding_file(
            burst_extraction_params,
            params_spectral_clustering,
        )
    )
