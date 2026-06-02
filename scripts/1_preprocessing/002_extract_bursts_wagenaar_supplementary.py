"""
Alternative burst detection for revision (simultaneously active electrodes).
"""

import os

import numpy as np
import pandas as pd

from burst_shape.folders import get_data_folder
from burst_shape.persistence import (
    save_burst_extraction_params,
    save_burst_matrix,
    save_df_bursts,
    save_df_cultures,
)
from burst_shape.preprocess import burst_extraction

# parameters
n_units_total = 59
params_burst_extraction = {
    "dataset": "wagenaar",
    # values rounded vs 5*sqrt(59)=38.41 and 50/59=0.847 to keep the
    # folder name under the 255-byte OS limit once the pinned old-behavior
    # kwargs below are appended. Algorithmic drift is sub-percent.
    "maxISIstart": round(5 * np.sqrt(n_units_total)),  # 38
    "maxISIb": round(5 * np.sqrt(n_units_total)),  # 38
    "minBdur": 40,
    "minIBI": 40,  # / n_units_total,
    "minSburst": round(50 / n_units_total, 2),  # 0.85
    "bin_size": None,
    "n_bins": 50,
    "extend_left": 0,
    "extend_right": 0,
    "burst_length_threshold": None,
    "pad_right": False,
    "normalization": "integral",
    "min_length": 30,
    "min_firing_rate": 3162,  # 10 ** 3.5,
    "smoothing_kernel": 4,
    "algorithm": "overlap",
    "unit_threshold": 0.2,
    "n_units_total": n_units_total,
    # Explicitly pin the simultaneity rule and disable entourage extension
    # (both differ from the package defaults).
    "network_rule": "simultaneity",
    "entourage_maxISI": None,
}


def _get_data_from_file():
    data_folder = os.path.join(get_data_folder(), "extracted")
    print(f"Build dataframe from files in {data_folder}")
    file_list = os.listdir(data_folder)
    # file_list = [file for file in file_list if ".txt" in file]
    df = pd.DataFrame(file_list, columns=["file_name"])
    # create columns batch-culture-day
    df["batch"] = df["file_name"].apply(lambda x: x.split("-")[0])
    df["culture"] = df["file_name"].apply(lambda x: x.split("-")[1])
    df["day"] = df["file_name"].apply(lambda x: x.split("-")[2].split(".")[0])
    for col in ["batch", "culture", "day"]:
        df[col] = df[col].astype(int)
    # set index
    df.set_index(["batch", "culture", "day"], inplace=True)
    print("Done")
    return df


# extract bursts
df_cultures, df_bursts, burst_matrix = burst_extraction.extract_bursts(
    construct_df_cultures=_get_data_from_file,
    **params_burst_extraction,
)

# save
save_burst_extraction_params(params_burst_extraction)

save_df_cultures(df_cultures, params_burst_extraction)
save_df_bursts(df_bursts, params_burst_extraction)
save_burst_matrix(burst_matrix, params_burst_extraction)
