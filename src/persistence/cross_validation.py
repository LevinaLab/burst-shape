import json
import os

from .burst_extraction import _get_burst_folder
from .cross_validation_string import _cv_params_to_string


def save_cv_params(cv_params, burst_params):
    save_folder = _get_burst_folder(burst_params)
    cv_string = _cv_params_to_string(cv_params)
    with open(
        os.path.join(save_folder, f"{cv_string}_params.json"),
        "w",
    ) as f:
        json.dump(cv_params, f, indent=4)


def load_cv_params(burst_params, cv_params: str or dict):
    if isinstance(cv_params, dict):
        cv_string = _cv_params_to_string(cv_params)
    else:
        cv_string = cv_params
    with open(
        os.path.join(_get_burst_folder(burst_params), f"{cv_string}_params.json"),
        "r",
    ) as f:
        cv_params = json.load(f)
    return cv_params
