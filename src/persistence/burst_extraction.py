import os

from src.folders import get_results_folder

_burst_extraction_defaults: dict = {
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


def get_burst_folder(burst_params: str or dict):
    if isinstance(burst_params, dict):
        burst_params = _burst_params_to_str(burst_params)
    return os.path.join(get_results_folder(), burst_params)
