import json
import os

import numpy as np
import pandas as pd

from src.folders import get_results_folder

from .cross_validation_string import _cv_params_to_string

_burst_extraction_defaults: dict = {
    "dataset": "wagenaar",
    "maxISIstart": 5,
    "maxISIb": 5,
    "minBdur": 40,
    "minIBI": 40,
    "minSburst": 50,
    "bin_size": None,
    "n_bins": None,
    "extend_left": 0,
    "extend_right": 0,
    "burst_length_threshold": None,
    "pad_right": False,
    "normalization": None,
    "min_length": None,
    "min_firing_rate": None,
    "smoothing_kernel": None,
}


def _burst_params_to_str(burst_params):
    name = "burst"
    for key, value in burst_params.items():
        if (
            key in _burst_extraction_defaults
            and value != _burst_extraction_defaults[key]
        ):
            name += f"_{key}_{value}"
    return name


def _get_burst_folder(burst_params: str or dict):
    if isinstance(burst_params, dict):
        burst_params = _burst_params_to_str(burst_params)
    return os.path.join(get_results_folder(), burst_params)


def save_df_cultures(df_cultures, burst_params):
    save_folder = _get_burst_folder(burst_params)
    df_save_path = os.path.join(save_folder, "df_cultures.pkl")
    os.makedirs(save_folder, exist_ok=True)
    df_cultures.to_pickle(df_save_path)


def load_df_cultures(burst_params):
    save_folder = _get_burst_folder(burst_params)
    df_save_path = os.path.join(save_folder, "df_cultures.pkl")
    return pd.read_pickle(df_save_path)


def save_df_bursts(df_bursts, burst_params, cv_params=None):
    save_folder = _get_burst_folder(burst_params)
    if cv_params is None:
        df_save_path = os.path.join(save_folder, "df_bursts.pkl")
    else:
        cv_string = _cv_params_to_string(cv_params)
        df_save_path = os.path.join(save_folder, f"df_bursts_{cv_string}.pkl")
    os.makedirs(save_folder, exist_ok=True)
    df_bursts.to_pickle(df_save_path)


def load_df_bursts(burst_params, cv_params=None):
    save_folder = _get_burst_folder(burst_params)
    if cv_params is None:
        df_save_path = os.path.join(save_folder, "df_bursts.pkl")
    else:
        cv_string = _cv_params_to_string(cv_params)
        df_save_path = os.path.join(save_folder, f"df_bursts_{cv_string}.pkl")
    return pd.read_pickle(df_save_path)


def save_burst_matrix(burst_matrix, burst_params):
    save_folder = _get_burst_folder(burst_params)
    df_save_path = os.path.join(save_folder, "burst_matrix.npy")
    os.makedirs(save_folder, exist_ok=True)
    np.save(df_save_path, burst_matrix)


def load_burst_matrix(burst_params):
    save_folder = _get_burst_folder(burst_params)
    df_save_path = os.path.join(save_folder, "burst_matrix.npy")
    return np.load(df_save_path)


def save_burst_extraction_params(burst_params):
    save_folder = _get_burst_folder(burst_params)
    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, "params.json"), "w") as f:
        json.dump(burst_params, f, indent=4)
