import pytest

from src.persistence.burst_extraction import (
    _burst_extraction_defaults,
    _burst_params_to_str,
    burst_params_str_to_dict,
)
from src.persistence.params_conversion_helper import params_dict_to_string
from src.persistence.spectral_clustering import (
    _spectral_clustering_defaults,
    _spectral_clustering_params_to_str,
    spectral_clustering_params_str_to_dict,
)

RESULTS_BURST_FOLDER_NAMES = [
    "burst_dataset_hommersom_binary_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30",
    "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30",
    "burst_dataset_hommersom_test_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30",
    "burst_dataset_hommersom_test_minIBI_50_n_bins_50_normalization_integral_min_length_30",
    "burst_dataset_hommersom_test_minIBI_50_n_bins_50_normalization_integral_min_length_30_min_firing_rate_1585",
    "burst_dataset_human_slice_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30",
    "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30",
    "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4",
    "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4",
    "burst_dataset_mossink_KS",
    "burst_dataset_mossink_MELAS",
    "burst_dataset_mossink_maxISIstart_100_maxISIb_100_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30",
    "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30",
    "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30",
    "burst_dataset_mossink_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30",
    "burst_dataset_mossink_maxISIstart_50_maxISIb_50_minBdur_100_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30",
    "burst_dataset_wagenaar_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4",
]


SCRIPT_BURST_PARAMS_TO_EXPECTED_STRING = [
    (
        {
            "dataset": "wagenaar",
            "maxISIstart": 5,
            "maxISIb": 5,
            "minBdur": 40,
            "minIBI": 40,
            "minSburst": 50,
            "bin_size": None,
            "n_bins": 50,
            "extend_left": 0,
            "extend_right": 0,
            "burst_length_threshold": None,
            "pad_right": False,
            "normalization": "integral",
            "min_length": 30,
            "min_firing_rate": 3162,
            "smoothing_kernel": 4,
        },
        "burst_dataset_wagenaar_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4",
    ),
    (
        {
            "dataset": "kapucu",
            "maxISIstart": 20,
            "maxISIb": 20,
            "minBdur": 50,
            "minIBI": 500,
            "minSburst": 100,
            "bin_size": None,
            "n_bins": 50,
            "extend_left": 0,
            "extend_right": 0,
            "burst_length_threshold": None,
            "pad_right": False,
            "normalization": "integral",
            "min_length": 30,
            "min_firing_rate": 316,
            "smoothing_kernel": 4,
        },
        "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4",
    ),
    (
        {
            "dataset": "hommersom_test",
            "maxISIstart": 20,
            "maxISIb": 20,
            "minBdur": 50,
            "minIBI": 100,
            "minSburst": 100,
            "bin_size": None,
            "n_bins": 50,
            "extend_left": 0,
            "extend_right": 0,
            "burst_length_threshold": None,
            "pad_right": False,
            "normalization": "integral",
            "min_length": 30,
        },
        "burst_dataset_hommersom_test_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30",
    ),
    (
        {
            "dataset": "inhibblock",
            "maxISIstart": 20,
            "maxISIb": 20,
            "minBdur": 50,
            "minIBI": 100,
            "minSburst": 100,
            "bin_size": None,
            "n_bins": 50,
            "extend_left": 0,
            "extend_right": 0,
            "burst_length_threshold": None,
            "pad_right": False,
            "normalization": "integral",
            "min_length": 30,
        },
        "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30",
    ),
    (
        {
            "dataset": "mossink",
            "maxISIstart": 100,
            "maxISIb": 50,
            "minBdur": 100,
            "minIBI": 500,
            "minSburst": 50,
            "bin_size": None,
            "n_bins": 50,
            "extend_left": 0,
            "extend_right": 0,
            "burst_length_threshold": None,
            "pad_right": False,
            "normalization": "integral",
            "min_length": 30,
        },
        "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30",
    ),
    (
        {
            "dataset": "hommersom",
            "maxISIstart": 20,
            "maxISIb": 20,
            "minBdur": 50,
            "minIBI": 100,
            "minSburst": 100,
            "bin_size": None,
            "n_bins": 50,
            "extend_left": 0,
            "extend_right": 0,
            "burst_length_threshold": None,
            "pad_right": False,
            "normalization": "integral",
            "min_length": 30,
        },
        "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30",
    ),
    (
        {
            "dataset": "hommersom_binary",
            "maxISIstart": 20,
            "maxISIb": 20,
            "minBdur": 50,
            "minIBI": 100,
            "minSburst": 100,
            "bin_size": None,
            "n_bins": 50,
            "extend_left": 0,
            "extend_right": 0,
            "burst_length_threshold": None,
            "pad_right": False,
            "normalization": "integral",
            "min_length": 30,
        },
        "burst_dataset_hommersom_binary_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30",
    ),
]


def test_burst_params_round_trip_dict_to_str_to_dict():
    params = dict(_burst_extraction_defaults)
    params.update(
        {
            "dataset": "hommersom_binary",
            "maxISIb": 7,
            "pad_right": True,
            "min_length": 12,
        }
    )

    params_str = _burst_params_to_str(params)

    assert params_str.startswith("burst")
    assert burst_params_str_to_dict(params_str) == params


def test_spectral_params_round_trip_dict_to_str_to_dict():
    params = dict(_spectral_clustering_defaults)
    params.update(
        {
            "affinity": "precomputed",
            "metric": "wasserstein",
            "n_neighbors": 21,
        }
    )

    params_str = _spectral_clustering_params_to_str(params)

    assert (
        params_str == "spectral_affinity_precomputed_metric_wasserstein_n_neighbors_21"
    )
    assert spectral_clustering_params_str_to_dict(params_str) == params


def test_params_dict_to_string_skips_default_values():
    params = {"alpha": 1, "beta": 3}
    defaults = {"alpha": 1, "beta": 2}

    assert (
        params_dict_to_string(params=params, defaults=defaults, startswith="example")
        == "example_beta_3"
    )


def test_burst_params_str_to_dict_rejects_wrong_prefix():
    with pytest.raises(ValueError, match="must start with"):
        burst_params_str_to_dict("not_burst_dataset_hommersom_binary")


def test_spectral_params_str_to_dict_rejects_unknown_key():
    with pytest.raises(ValueError, match="Unknown parameter key"):
        spectral_clustering_params_str_to_dict("spectral_non_existing_key_1")


@pytest.mark.parametrize("folder_name", RESULTS_BURST_FOLDER_NAMES)
def test_real_results_folder_names_round_trip(folder_name):
    parsed = burst_params_str_to_dict(folder_name)
    assert _burst_params_to_str(parsed) == folder_name


@pytest.mark.parametrize(
    ("params_burst_extraction", "expected_folder_name"),
    SCRIPT_BURST_PARAMS_TO_EXPECTED_STRING,
)
def test_script_burst_dict_matches_expected_folder_name(
    params_burst_extraction, expected_folder_name
):
    assert _burst_params_to_str(params_burst_extraction) == expected_folder_name
