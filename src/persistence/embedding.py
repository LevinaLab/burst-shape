import os

import numpy as np

from src.persistence.burst_extraction import _get_burst_folder

_tsne_params_defaults = {
    "initialization": "pca",
}


def _get_pca_file(
    burst_extraction_params,
):
    return os.path.join(
        _get_burst_folder(burst_extraction_params),
        "pca.npy",
    )


def _get_tsne_file(
    burst_extraction_params,
    tsne_params,
):
    name = "tsne"
    if tsne_params is not None:
        assert isinstance(tsne_params, dict)
        for k, v in tsne_params.items():
            if _tsne_params_defaults[k] == v:
                continue
            else:
                name += f"_{k}__{v}"
    name += ".npy"
    return os.path.join(
        _get_burst_folder(burst_extraction_params),
        name,
    )


def load_pca(
    burst_extraction_params,
):
    """Load PCA."""
    return np.load(
        _get_pca_file(
            burst_extraction_params,
        )
    )


def save_pca(
    pca,
    burst_extraction_params,
):
    """Save PCA."""
    np.save(
        _get_pca_file(
            burst_extraction_params,
        ),
        pca,
    )


def pca_exists(
    burst_extraction_params,
):
    """Check if PCA exists."""
    return os.path.exists(
        _get_pca_file(
            burst_extraction_params,
        )
    )


def load_tsne(
    burst_extraction_params,
    tsne_params=None,
):
    """Load t-SNE."""
    return np.load(
        _get_tsne_file(
            burst_extraction_params,
            tsne_params,
        )
    )


def save_tsne(
    tsne,
    burst_extraction_params,
    tsne_params=None,
):
    """Save t-SNE."""
    np.save(
        _get_tsne_file(
            burst_extraction_params,
            tsne_params,
        ),
        tsne,
    )


def tsne_exists(
    burst_extraction_params,
    tsne_params=None,
):
    """Check if t-SNE exists."""
    return os.path.exists(
        _get_tsne_file(
            burst_extraction_params,
            tsne_params,
        )
    )
