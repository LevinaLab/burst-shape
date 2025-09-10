import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.folders import get_data_kapucu_folder
from src.persistence import (
    save_burst_extraction_params,
    save_burst_matrix,
    save_df_bursts,
    save_df_cultures,
)
from src.preprocess import burst_extraction

na = np.array

# parameters
params_burst_extraction = {
    "dataset": "kapucu",
    "maxISIstart": 20,  # 5
    "maxISIb": 20,  # 5
    "minBdur": 50,  # 40
    "minIBI": 500,  # 50
    "minSburst": 100,
    "bin_size": None,
    "n_bins": 50,
    "extend_left": 0,
    "extend_right": 0,
    "burst_length_threshold": None,
    "pad_right": False,
    "normalization": "integral",
    "min_length": 30,
    "min_firing_rate": 316,  # 10**2.5,
    "smoothing_kernel": 4,
}


def _gid_to_numbers(gid):
    for i, u_id in enumerate(np.unique(gid)):
        gid[gid == u_id] = i
    return gid


def _get_kapucu_data_from_file():
    data_folder = get_data_kapucu_folder()
    print(f"Build dataframe from files in {data_folder}")
    res = list(os.walk(data_folder, topdown=True))
    files = res[0][2]  # all file names
    div_days = [f.split("_")[3] for f in files if "DIV" in f]
    types = [f.split("_")[0] for f in files if "DIV" in f]
    mea_n = [f.split("_")[2] for f in files if "DIV" in f]

    div_days = [re.findall(r"\d+", div) for div in div_days]
    div_days = na(div_days, dtype=int).flatten()
    indis = np.argsort(div_days)
    div_days = div_days[indis]
    types = na(types)[indis]
    mea_n = na(mea_n)[indis]
    files = na(files)[indis]

    divs = []
    # summaries = []
    well_id = []
    culture_type = []
    mea_number = []
    spks = []
    for i, file_ in tqdm(enumerate(files), desc="Loading files"):
        div = div_days[i]
        type_ = types[i]
        mea_ = mea_n[i]
        spikes = pd.read_csv(os.path.join(data_folder, file_))
        channels = spikes["Channel"]
        wells = [ch.split("_")[0] for ch in channels]
        ch_n = [ch.split("_")[1] for ch in channels]
        spikes["well"] = wells
        spikes["ch_n"] = ch_n
        # Extract spikes for different wells
        # well_spikes= []
        for well in np.unique(wells):
            st = na(spikes["Time"][spikes["well"] == well])
            gid = na(spikes["ch_n"][spikes["well"] == well])
            spks.append([st, gid])
            # summaries.append(get_summary([st,gid],type_))
            divs.append(div)
            well_id.append(well)
            culture_type.append(type_)
            mea_number.append(mea_)

    # Cut the noise at the beginning of a recording
    mask = spks[247][0] > 125
    spks[247][0] = spks[247][0][mask]
    spks[247][1] = spks[247][1][mask]
    df = pd.DataFrame(
        {
            "spikes": spks,
            "DIV": divs,
            "well_id": well_id,
            "culture_type": culture_type,
            "mea_number": mea_number,
        }
    )
    df["times"] = df["spikes"].apply(lambda x: x[0])
    df["gid"] = df["spikes"].apply(lambda x: x[1])
    df["gid"] = df["gid"].apply(_gid_to_numbers)
    # delete "spikes" column
    df.drop(columns=["spikes"], inplace=True)
    # set index
    df.set_index(["culture_type", "mea_number", "well_id", "DIV"], inplace=True)
    print("Done")
    return df


# extract bursts
df_cultures, df_bursts, burst_matrix = burst_extraction.extract_bursts(
    construct_df_cultures=_get_kapucu_data_from_file,
    **params_burst_extraction,
)

# save
save_burst_extraction_params(params_burst_extraction)

save_df_cultures(df_cultures, params_burst_extraction)
save_df_bursts(df_bursts, params_burst_extraction)
save_burst_matrix(burst_matrix, params_burst_extraction)
