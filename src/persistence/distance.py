import os
from typing import Literal

import numpy as np
from scipy.spatial.distance import squareform

from src.persistence.burst_extraction import _get_burst_folder


def _get_distance_matrix_filename(
    burst_extraction_params,
    params_cross_validation=None,
    i_split=None,
):
    if params_cross_validation is not None and i_split is not None:
        name = f"distance_matrix_{params_cross_validation}_{i_split}.npy"
    else:
        name = "distance_matrix.npy"
    return os.path.join(
        _get_burst_folder(burst_extraction_params),
        name,
    )


def load_distance_matrix(
    burst_extraction_params,
    params_cross_validation=None,
    i_split=None,
    form: Literal["matrix", "vector"] = "vector",
):
    """Load distance matrix."""
    filename = _get_distance_matrix_filename(
        burst_extraction_params, params_cross_validation, i_split
    )
    match form:
        case "matrix":
            return squareform(np.load(filename), force="tomatrix")
        case "vector":
            return np.load(filename)


def save_distance_matrix(
    distance_matrix,
    burst_extraction_params,
    params_cross_validation=None,
    i_split=None,
):
    """Save distance matrix."""
    np.save(
        _get_distance_matrix_filename(
            burst_extraction_params, params_cross_validation, i_split
        ),
        distance_matrix,
    )


def distance_matrix_exists(
    burst_extraction_params,
    params_cross_validation=None,
    i_split=None,
):
    """Check if distance matrix exists."""
    return os.path.exists(
        _get_distance_matrix_filename(
            burst_extraction_params, params_cross_validation, i_split
        )
    )
