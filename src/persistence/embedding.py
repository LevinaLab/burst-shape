import os

import numpy as np

from src.persistence.burst_extraction import _get_burst_folder


def _get_pca_file(
    burst_extraction_params,
):
    return os.path.join(
        _get_burst_folder(burst_extraction_params),
        "pca.npy",
    )


def _get_tsne_file(
    burst_extraction_params,
):
    return os.path.join(
        _get_burst_folder(burst_extraction_params),
        "tsne.npy",
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
):
    """Load t-SNE."""
    return np.load(
        _get_tsne_file(
            burst_extraction_params,
        )
    )


def save_tsne(
    tsne,
    burst_extraction_params,
):
    """Save t-SNE."""
    np.save(
        _get_tsne_file(
            burst_extraction_params,
        ),
        tsne,
    )


def tsne_exists(
    burst_extraction_params,
):
    """Check if t-SNE exists."""
    return os.path.exists(
        _get_tsne_file(
            burst_extraction_params,
        )
    )
