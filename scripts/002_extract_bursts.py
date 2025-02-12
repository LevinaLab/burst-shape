import os

import numpy as np

from src.folders import get_data_folder
from src.persistence import (
    save_burst_extraction_params,
    save_burst_matrix,
    save_df_bursts,
    save_df_cultures,
)
from src.preprocess import burst_extraction

# parameters
params_burst_extraction = {
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
    "min_firing_rate": 3162,  # 10 ** 3.5,
    "smoothing_kernel": 4,
}


# extract bursts
df_cultures, df_bursts, burst_matrix = burst_extraction.extract_bursts(
    data_folder=os.path.join(get_data_folder(), "extracted"),
    **params_burst_extraction,
)

# manually remove outlier
print("Manually removing outlier...")
"""
start_orig                                               10530.52
end_orig                                                 10661.52
start_extend                                             10530.52
end_extend                                               10661.52
time_orig                                                   131.0
time_extend                                                 131.0
burst           [0.0022913033070004443, 0.005188409620542584, ...
peak_height                                          22160.305344
integral                                            195312.022901
Name: (3, 5, 13, 4), dtype: object
"""
try:
    # identify index of outlier by start_orig being close (within 1) to 10530.52
    index_to_remove = df_bursts[
        df_bursts["start_orig"].between(10529.52, 10531.52)
    ].index
    assert len(index_to_remove) == 1
    index_to_remove = index_to_remove[0]
    # get iloc index
    index_to_remove = df_bursts.index.get_loc(index_to_remove)
    print(f"Index to remove: {index_to_remove}")
    print(df_bursts.iloc[index_to_remove])
    # assert that the burst is the same in the burst_matrix
    assert np.allclose(
        burst_matrix[index_to_remove], df_bursts.iloc[index_to_remove]["burst"]
    )
    df_bursts = df_bursts.drop(df_bursts.iloc[index_to_remove].name)
    burst_matrix = np.delete(burst_matrix, index_to_remove, axis=0)

    index_to_remove_culture = df_bursts.iloc[index_to_remove].name[:3]
    df_cultures.at[index_to_remove_culture, "n_bursts"] -= 1
except AssertionError:
    print("Outlier not found. Continuing without removing outlier.")

# save
save_burst_extraction_params(params_burst_extraction)

save_df_cultures(df_cultures, params_burst_extraction)
save_df_bursts(df_bursts, params_burst_extraction)
save_burst_matrix(burst_matrix, params_burst_extraction)
