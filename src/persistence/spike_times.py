import os

import numpy as np

from src.folders import get_data_folder


def get_wagenaar_spike_times(df_cultures, idx):
    st, gid = np.loadtxt(
        os.path.join(get_data_folder(), "extracted", df_cultures.loc[idx].file_name)
    ).T
    st *= 1000
    gid = gid.astype(int)
    return st, gid


def get_kapucu_spike_times(df_cultures, idx):
    return (
        df_cultures.at[idx, "times"] * 1000,
        df_cultures.at[idx, "gid"],
    )


def get_hommersom_spike_times(df_cultures, idx):
    return (
        df_cultures.at[idx, "times"] * 1000,
        df_cultures.at[idx, "gid"],
    )


def get_inhibblock_spike_times(df_cultures, idx):
    return (
        df_cultures.at[idx, "times"] * 1000,
        df_cultures.at[idx, "gid"],
    )


def get_mossink_spike_times(df_cultures, idx):
    return (
        df_cultures.at[idx, "times"] * 1000,
        df_cultures.at[idx, "gid"],
    )
