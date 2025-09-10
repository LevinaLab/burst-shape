import os
import warnings

import numpy as np

from src.folders import get_data_folder


def get_spike_times_in_seconds(df_cultures, idx, dataset):
    match dataset:
        case "wagenaar":
            st, gid = np.loadtxt(
                os.path.join(
                    get_data_folder(), "extracted", df_cultures.loc[idx].file_name
                )
            ).T
            gid = gid.astype(int)
        case "kapucu" | "hommersom" | "hommersom_binary" | "hommersom_test" | "inhibblock" | "mossink":
            st = df_cultures.at[idx, "times"]
            gid = df_cultures.at[idx, "gid"]
        case _:
            warnings.warn(
                f"get_spike_times_in_seconds() not implemented for dataset={dataset}. "
                "Trying to load with default behaviour: Loading from df_cultures columns 'times' and 'gid'. "
                "If this doesn't work or to suppress this warning implement your dataset explicitly here."
            )
            st = df_cultures.at[idx, "times"]
            gid = df_cultures.at[idx, "gid"]
    return st, gid


def get_spike_times_in_milliseconds(df_cultures, idx, dataset):
    st, gid = get_spike_times_in_seconds(df_cultures, idx, dataset)
    return st * 1000, gid
