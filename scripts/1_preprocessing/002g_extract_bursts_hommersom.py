import os

import pandas as pd

from src.folders import get_data_hommersom_binary_folder
from src.persistence import (
    save_burst_extraction_params,
    save_burst_matrix,
    save_df_bursts,
    save_df_cultures,
)
from src.preprocess import burst_extraction

# parameters
params_burst_extraction = {
    "dataset": "hommersom_binary",
    "maxISIstart": 20,  # 5,
    "maxISIb": 20,  # 5,
    "minBdur": 50,  # 40,
    "minIBI": 100,
    "minSburst": 100,  # 50,
    "bin_size": None,
    "n_bins": 50,
    "extend_left": 0,
    "extend_right": 0,
    "burst_length_threshold": None,
    "pad_right": False,
    "normalization": "integral",
    "min_length": 30,
    # "min_firing_rate": 1585,  #  10 ** 3.2
    # "smoothing_kernel": 4,
    # TODO consider a minimum starting time
}


def _construct_df_cultures():
    return pd.read_pickle(
        os.path.join(get_data_hommersom_binary_folder(), "df_hommersom_binary.pkl")
    )


# extract bursts
df_cultures, df_bursts, burst_matrix = burst_extraction.extract_bursts(
    construct_df_cultures=_construct_df_cultures,
    **params_burst_extraction,
)

# save
save_burst_extraction_params(params_burst_extraction)

save_df_cultures(df_cultures, params_burst_extraction)
save_df_bursts(df_bursts, params_burst_extraction)
save_burst_matrix(burst_matrix, params_burst_extraction)
