import numpy as np
import pandas as pd
from tqdm import tqdm

from src.persistence.spike_times import get_spike_times_in_milliseconds


def get_classical_features(df_cultures, df_bursts, dataset):
    """Get classical features.

    Computes as in van Hugte et al., 2023, but skipping those aspects that we don't analyze.

    Args:
        df_cultures: pandas DataFrame containing culture data.
        df_bursts: pandas DataFrame containing burst data.

    Returns:
        df_cultures: pandas DataFrame containing classical features.
        features: pandas DataFrame containing classical features.
    """
    assert all(
        df_cultures["n_bursts"] > 0
    ), "df_cultures must have at least one burst per culture! Before calling this function, remove unnecessary rows with df_cultures = df_cultures[df_cultures['n_bursts'] > 0]"

    features = [
        "MFR",
        "BSR",
        # "MBR",
        # "BD",
        "NBR",
        "NBD",
        "PRS",
        # "HFB",
    ]
    for feature in features + ["n_spikes"]:
        df_cultures[feature] = pd.Series(dtype=float)

    # sort index for performance
    df_bursts = df_bursts.sort_index()
    df_cultures = df_cultures.sort_index()

    if dataset == "wagenaar":
        for index in tqdm(df_cultures.index, desc="Compute classical features"):
            st, gid = get_spike_times_in_milliseconds(df_cultures, index, dataset)

            # MFR
            # MFR was calculated for each well individually by averaging the firing rate of each separate channel by the total number of active channels of the well
            df_cultures.at[index, "n_spikes"] = len(st)
            df_cultures.at[index, "MFR"] = (
                df_cultures.at[index, "n_spikes"] / max(st) / len(np.unique(gid))
            )

            # BSR burst spike rate
            # for index in tqdm(df_cultures.index, desc="BSR"):
            df_bursts_select = df_bursts.loc[index]
            df_cultures.at[index, "BSR"] = df_bursts_select["firing_rate"].mean()

            # MBR
            # The mean burst rate (MBR) was calculated by averaging the burst rate across all electrodes.
            # TODO not applicable because no single channel bursts

            # BD
            # burst duration TODO not applicable because no single channel bursts

            # NBR
            # The NB rate (NBR) was derived by dividing all NBs in the recording by the length of the recording in minutes.
            df_cultures.at[index, "NBR"] = df_cultures.at[index, "n_bursts"] / max(
                st
            )  # * 60  # convert to minutes

            # NBD
            # for index in tqdm(df_cultures.index, desc="NBD"):
            df_bursts_select = df_bursts.loc[index]
            df_cultures.at[index, "NBD"] = df_bursts_select["time_orig"].mean()

            # PRS
            # The percentage of random spikes (PRS) was defined by calculating the percentage of isolated spikes that did not belong to a burst.
            # for index in tqdm(df_cultures.index, desc="PRS"):
            # times = df_cultures.at[index, "times"] * 1000
            df_cultures.at[index, "n_spikes_in_bursts"] = np.sum(
                [
                    np.sum((st >= start) & (st <= end))
                    for start, end in df_cultures.at[index, "burst_start_end"]
                ]
            )
            df_cultures.at[index, "PRS"] = (
                1
                - df_cultures.at[index, "n_spikes_in_bursts"]
                / df_cultures.at[index, "n_spikes"]
            )

        # The IBI and NB IBI (NIBI) were calculated by subtraction of the time stamp at the beginning from the time stamp at the end of each burst or NB

        # HFB
        # High frequency bursts (HFBs) within an NB period were detected by decreasing the burst detection ISI to 5 ms, with a maximum IBI of 10 ms, to individually detect each HFB.
        # TODO we don't detect HFBs

    else:
        # MFR
        # MFR was calculated for each well individually by averaging the firing rate of each separate channel by the total number of active channels of the well
        df_cultures["n_spikes"] = df_cultures["times"].apply(len)
        df_cultures["MFR"] = (
            df_cultures["n_spikes"]
            / (df_cultures["times"].apply(max))
            / df_cultures["gid"].apply(np.unique).apply(len)
        )

        # BSR burst spike rate
        for index in tqdm(df_cultures.index, desc="BSR"):
            df_bursts_select = df_bursts.loc[index]
            df_cultures.at[index, "BSR"] = df_bursts_select["firing_rate"].mean()

        # MBR
        # The mean burst rate (MBR) was calculated by averaging the burst rate across all electrodes.
        # TODO not applicable because no single channel bursts

        # BD
        # burst duration TODO not applicable because no single channel bursts

        # NBR
        # The NB rate (NBR) was derived by dividing all NBs in the recording by the length of the recording in minutes.
        df_cultures["NBR"] = df_cultures["n_bursts"] / df_cultures["times"].apply(
            max
        )  # * 60  # convert to minutes

        # NBD
        for index in tqdm(df_cultures.index, desc="NBD"):
            df_bursts_select = df_bursts.loc[index]
            df_cultures.at[index, "NBD"] = df_bursts_select["time_orig"].mean()

        # PRS
        # The percentage of random spikes (PRS) was defined by calculating the percentage of isolated spikes that did not belong to a burst.
        for index in tqdm(df_cultures.index, desc="PRS"):
            times = df_cultures.at[index, "times"] * 1000
            df_cultures.at[index, "n_spikes_in_bursts"] = np.sum(
                [
                    np.sum((times >= start) & (times <= end))
                    for start, end in df_cultures.at[index, "burst_start_end"]
                ]
            )
        df_cultures["PRS"] = (
            1 - df_cultures["n_spikes_in_bursts"] / df_cultures["n_spikes"]
        )

        # The IBI and NB IBI (NIBI) were calculated by subtraction of the time stamp at the beginning from the time stamp at the end of each burst or NB

        # HFB
        # High frequency bursts (HFBs) within an NB period were detected by decreasing the burst detection ISI to 5 ms, with a maximum IBI of 10 ms, to individually detect each HFB.
        # TODO we don't detect HFBs

    return df_cultures, features
