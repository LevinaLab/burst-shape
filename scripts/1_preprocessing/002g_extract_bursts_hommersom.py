import os

import numpy as np
import pandas as pd

from burst_shape.folders import get_data_hommersom_binary_folder
from burst_shape.persistence import (
    save_burst_extraction_params,
    save_burst_matrix,
    save_df_bursts,
    save_df_cultures,
)
from burst_shape.preprocess import burst_extraction

# Burst detector selection.
#   supplementary = False -> MAIN detector (MI_bursts on pooled spikes; main text)
#   supplementary = True  -> supplementary detector ("ISI + 20% threshold"):
#       per-electrode burstlets with fixed ISI = MAIN_ISI * sqrt(n_units),
#       combined via the simultaneity rule that keeps a network burst while
#       >= 20% of electrodes are simultaneously bursting (unit_threshold=0.2).
supplementary = False
n_units_total = 16  # electrodes per recording (used by the supplementary detector)

if not supplementary:
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
else:
    params_burst_extraction = {
        "dataset": "hommersom_binary",
        "algorithm": "overlap",
        "maxISIstart": int(round(20 * np.sqrt(n_units_total))),  # 80
        "maxISIb": int(round(20 * np.sqrt(n_units_total))),  # 80
        "minBdur": 50,
        "minIBI": 100,
        "minSburst": int(round(100 / n_units_total)),  # 6
        "network_rule": "simultaneity",
        "unit_threshold": 0.2,  # keep burst while >= 20% of electrodes active
        "n_units_total": n_units_total,
        "n_bins": 50,
        "normalization": "integral",
        "min_length": 30,
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
