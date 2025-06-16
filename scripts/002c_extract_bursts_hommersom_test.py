import os

import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm

from src.folders import get_data_hommersom_test_folder
from src.persistence import (
    save_burst_extraction_params,
    save_burst_matrix,
    save_df_bursts,
    save_df_cultures,
)
from src.preprocess import burst_extraction

# parameters
params_burst_extraction = {
    "dataset": "hommersom_test",
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


def _get_hommersom_test_data_from_file(
    fs=12500,  # samples per second
):
    data_folder = get_data_hommersom_test_folder()
    print(f"Build dataframe from files in {data_folder}")
    data = []
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".mat"):
                relative_path = os.path.relpath(os.path.join(root, file), data_folder)
                parts = relative_path.split(os.sep)

                if len(parts) < 3:
                    continue  # Skip files that don't fit the hierarchy

                batch, clone, filename = parts[-3], parts[-2], parts[-1]
                clone = clone.split("_")[-1]
                if clone.islower():
                    clone = clone.capitalize()
                well_id = filename.split("_")[-1][:2]
                data.append(
                    {
                        "batch": batch,
                        "clone": clone,
                        "well_id": well_id,
                        "path": relative_path,
                    }
                )
    df_cultures = pd.DataFrame(data)
    del data
    # transform well_id to well_idx
    df_cultures["well_idx"] = pd.Series(0, dtype=int)
    for row in df_cultures[["batch", "clone"]].drop_duplicates().itertuples():
        batch, clone = row.batch, row.clone
        n_wells = len(
            df_cultures.loc[
                (df_cultures["batch"] == batch) & (df_cultures["clone"] == clone)
            ]
        )
        df_cultures.loc[
            (df_cultures["batch"] == batch) & (df_cultures["clone"] == clone),
            "well_idx",
        ] = list(range(n_wells))
    df_cultures["well_idx"] = df_cultures["well_idx"].astype(int)
    df_cultures.set_index(["batch", "clone", "well_idx"], inplace=True)

    df_cultures["times"] = pd.Series(dtype=object)
    df_cultures["gid"] = pd.Series(dtype=object)
    for index in tqdm(df_cultures.index, desc="Loading files"):
        data = sio.loadmat(os.path.join(data_folder, df_cultures.at[index, "path"]))
        data = data["Ts_AP"]
        times_order = np.argsort(data[:, 1])
        data = data[times_order]
        gid = data[:, 0]
        st = data[:, 1]
        st = st / fs
        df_cultures.at[index, "times"] = st
        df_cultures.at[index, "gid"] = gid
    print("Done")
    return df_cultures


# extract bursts
df_cultures, df_bursts, burst_matrix = burst_extraction.extract_bursts(
    construct_df_cultures=_get_hommersom_test_data_from_file,
    **params_burst_extraction,
)

# save
save_burst_extraction_params(params_burst_extraction)

save_df_cultures(df_cultures, params_burst_extraction)
save_df_bursts(df_bursts, params_burst_extraction)
save_burst_matrix(burst_matrix, params_burst_extraction)
