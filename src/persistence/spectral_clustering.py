import os

from src.persistence import get_burst_folder

_spectral_clustering_defaults: dict = {
    "n_components_max": 30,
    "affinity": "nearest_neighbors",
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


def get_spectral_clustering_folder(
    params_spectral_clustering: str or dict, params_burst_extraction: str or dict
):
    if isinstance(params_spectral_clustering, dict):
        params_spectral_clustering = _spectral_clustering_params_to_str(
            params_spectral_clustering
        )
    path_burst_extraction = get_burst_folder(params_burst_extraction)
    return os.path.join(path_burst_extraction, params_spectral_clustering)


def _labels_params_to_str(params):
    name = "004_clustering_labels"
    for key, value in params.items():
        if key in _labels_defaults and value != _labels_defaults[key]:
            name += f"_{key}_{value}"
    name += ".pkl"
    return name


def get_labels_params_file(params):
    name = "labels_params"
    for key, value in params.items():
        if key in _labels_defaults and value != _labels_defaults[key]:
            name += f"_{key}_{value}"
    name += ".json"
    return name


def get_labels_file(
    params_labels: str or dict,
    params_spectral_clustering: str or dict,
    params_burst_extraction: str or dict,
):
    if isinstance(params_labels, dict):
        params_labels = _labels_params_to_str(params_labels)
    return os.path.join(
        get_spectral_clustering_folder(
            params_spectral_clustering, params_burst_extraction
        ),
        params_labels,
    )
