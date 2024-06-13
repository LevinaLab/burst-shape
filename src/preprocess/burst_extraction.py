import itertools
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.folders import get_data_folder
from src.preprocess import burst_detection


def extract_bursts(
    data_folder=None,
    maxISIstart=5,
    maxISIb=5,
    minBdur=40,
    minIBI=40,
    minSburst=50,
    bin_size=None,
    n_bins=None,
    extend_left=0.0,
    extend_right=0.0,
    burst_length_threshold=None,
    pad_right=False,
    normalization=None,
):
    """Extract bursts from data files.

    All times in milliseconds.

    Args:
        data_folder (str, optional): Path to folder containing data files. Defaults to None.
        maxISIstart (int, optional): Maximum inter-spike interval (ISI) for start of burst.
            Defaults to 5.
        maxISIb (int, optional): Maximum ISI for burst. Defaults to 5.
        minBdur (int, optional): Minimum burst duration. Defaults to 40.
        minIBI (int, optional): Minimum inter-burst interval (IBI). Defaults to 40.
        minSburst (int, optional): Minimum number of spikes in a burst. Defaults to 50.
        bin_size (float, optional): Size of bins for binning spike times.
            Either bin_size or n_bins must be specified. Defaults to None.
        n_bins (int, optional): Number of bins for binning spike times.
            Either bin_size or n_bins must be specified. Defaults to None.
        extend_left (float, optional): Extend burst start time by this amount. Defaults to 0.0.
        extend_right (float, optional): Extend burst end time by this amount. Defaults to 0.0.
        burst_length_threshold (int, optional): Threshold for maximum burst length. Defaults to None.
        pad_right (bool, optional): Pad bursts to the right with zeros. Defaults to False.
        normalization (str, optional): Normalization to apply to bursts.
            Can be None, 'zscore', 'peak', 'integral'. Defaults to None.

    Returns:
        df_cultures (pd.DataFrame): Dataframe with columns 'file_name', 'n_bursts', 'burst_start_end'.
            Index is ('batch', 'culture', 'day').
        df_bursts (pd.DataFrame): Dataframe with columns 'start_orig', 'end_orig', 'start_extend', 'end_extend',
            'time_orig', 'time_extend', 'burst', 'peak_height', 'integral'.
            Index is ('batch', 'culture', 'day', 'i_burst').
        burst_matrix (np.ndarray): Matrix of bursts. Shape (n_bursts, n_bins).
    """
    if data_folder is None:
        data_folder = os.path.join(get_data_folder(), "extracted")
    df_cultures = _get_data_from_file(data_folder)
    df_cultures = _bursts_from_culture(
        df_cultures,
        data_folder,
        maxISIstart,
        maxISIb,
        minBdur,
        minIBI,
        minSburst,
    )
    df_bursts = _build_bursts_df(
        df_cultures,
        data_folder,
        bin_size,
        n_bins,
        extend_left,
        extend_right,
    )
    df_bursts = _filter_bursts(
        df_bursts,
        burst_length_threshold,
        pad_right,
        bin_size,
    )
    df_bursts = _normalize_bursts(df_bursts, normalization)
    burst_matrix = np.stack(df_bursts["burst"].values)
    return df_cultures, df_bursts, burst_matrix


def _get_data_from_file(data_folder):
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


def _bursts_from_culture(
    df,
    data_folder,
    maxISIstart,
    maxISIb,
    minBdur,
    minIBI,
    minSburst,
):
    df["n_bursts"] = pd.Series(dtype=object)
    df["burst_start_end"] = pd.Series(dtype=object)
    for index in tqdm(df.index, desc="Compute burst times for each culture"):
        bursts_start_end = burst_detection.MI_bursts(
            np.loadtxt(os.path.join(data_folder, df.at[index, "file_name"]))[:, 0]
            * 1000,
            maxISIstart=maxISIstart,
            maxISIb=maxISIb,
            minBdur=minBdur,
            minIBI=minIBI,
            minSburst=minSburst,
        )
        df.at[index, "n_bursts"] = len(bursts_start_end)
        df.at[index, "burst_start_end"] = bursts_start_end
    return df


def _build_bursts_df(
    df_cultures,
    data_folder,
    bin_size,
    n_bins,
    extend_left,
    extend_right,
):
    df_bursts = pd.DataFrame(
        columns=[
            "start_orig",
            "end_orig",
            "start_extend",
            "end_extend",
            "time_orig",
            "time_extend",
            "burst",
            "peak_height",
            "integral",
        ],
        index=pd.MultiIndex.from_tuples(
            itertools.chain.from_iterable(
                [
                    [
                        (batch, culture, day, i_burst)
                        for i_burst in range(
                            df_cultures.at[(batch, culture, day), "n_bursts"]
                        )
                    ]
                    for batch, culture, day in df_cultures.index
                ]
            ),
            names=["batch", "culture", "day", "i_burst"],
        ),
    )
    for index in tqdm(df_cultures.index, desc="Build dataframe of bursts with times."):
        batch, culture, day = index
        bursts_start_end = df_cultures.at[index, "burst_start_end"]
        for i_burst, (start, end) in enumerate(bursts_start_end):
            # df_bursts.at[(batch, culture, day, i_burst), "i_burst"] = i_burst
            df_bursts.at[(batch, culture, day, i_burst), "start_orig"] = start
            df_bursts.at[(batch, culture, day, i_burst), "end_orig"] = end
            df_bursts.at[(batch, culture, day, i_burst), "start_extend"] = (
                start - extend_left
            )
            df_bursts.at[(batch, culture, day, i_burst), "end_extend"] = (
                end + extend_right
            )
            df_bursts.at[(batch, culture, day, i_burst), "time_orig"] = end - start
            df_bursts.at[(batch, culture, day, i_burst), "time_extend"] = (
                end - start + extend_left + extend_right
            )

    assert (
        bin_size is not None or n_bins is not None
    ), "Either bin_size or n_bins must be specified"
    assert (
        bin_size is None or n_bins is None
    ), "Only one of bin_size and n_bins can be specified"
    for index in tqdm(df_cultures.index, desc="Extract bursts and bin them."):
        batch, culture, day = index
        bursts_start_end = df_cultures.at[index, "burst_start_end"]
        if len(bursts_start_end) == 0:
            continue
        file_name = df_cultures.at[index, "file_name"]
        file_path = os.path.join(data_folder, file_name)
        data = np.loadtxt(file_path)[:, 0] * 1000
        if bin_size is not None:
            time_max = np.max(data)
            bins = np.arange(0, time_max + bin_size, bin_size)
            counts, _ = np.histogram(data, bins=bins)
            for i_burst in range(len(bursts_start_end)):
                index = (batch, culture, day, i_burst)
                df_bursts.at[index, "burst"] = counts[
                    int(np.floor(df_bursts.at[index, "start_extend"] / bin_size)) : int(
                        np.ceil(df_bursts.at[index, "end_extend"] / bin_size)
                    )
                ]
        if n_bins is not None:
            if np.all(
                [
                    df_bursts.at[(batch, culture, day, i_burst), "start_extend"]
                    > df_bursts.at[(batch, culture, day, i_burst - 1), "end_extend"]
                    for i_burst in range(1, len(bursts_start_end))
                ]
            ):
                # if all starts are bigger than previous ends, then we can bin simultaneously
                bins = np.concatenate(
                    [
                        np.linspace(
                            df_bursts.at[
                                (batch, culture, day, i_burst), "start_extend"
                            ],
                            df_bursts.at[(batch, culture, day, i_burst), "end_extend"],
                            n_bins + 1,
                            endpoint=True,
                        )
                        for i_burst in range(len(bursts_start_end))
                    ]
                )
                counts, _ = np.histogram(data, bins=bins)
                for i_burst in range(len(bursts_start_end)):
                    index = (batch, culture, day, i_burst)
                    df_bursts.at[index, "burst"] = counts[
                        (n_bins + 1) * i_burst : (n_bins + 1) * (i_burst + 1) - 1
                    ]
            else:
                # otherwise we have to do it one by one
                print(
                    f"Warning: some bursts overlap for {file_name}, binning one by one"
                )
                for i_burst in range(len(bursts_start_end)):
                    index = (batch, culture, day, i_burst)
                    bins = np.linspace(
                        df_bursts.at[index, "start_extend"],
                        df_bursts.at[index, "end_extend"],
                        n_bins + 1,
                        endpoint=True,
                    )
                    counts, _ = np.histogram(data, bins=bins)
                    df_bursts.at[index, "burst"] = counts

    # convert bursts to firing rate (Hz)
    if n_bins is not None:
        df_bursts["burst"] = (
            df_bursts["burst"] / df_bursts["time_extend"] * 1000 * n_bins
        )
    if bin_size is not None:
        df_bursts["burst"] = df_bursts["burst"] / bin_size * 1000

    # compute peak height and integral
    for index in tqdm(df_bursts.index, desc="Compute peak height and integral"):
        burst = df_bursts.at[index, "burst"]
        df_bursts.at[index, "peak_height"] = np.max(burst)
        df_bursts.at[index, "integral"] = np.sum(burst)
    return df_bursts


def _filter_bursts(
    df_bursts,
    burst_length_threshold,
    pad_right,
    bin_size,
):
    print("Filter bursts (burst length, pad right)")
    if burst_length_threshold is not None:
        len_before = len(df_bursts)
        df_bursts = df_bursts[df_bursts["time_extend"] <= burst_length_threshold]
        len_after = len(df_bursts)
        print(
            f"Removed {len_before - len_after} bursts above threshold ({burst_length_threshold} ms)"
        )
    else:
        burst_length_threshold = np.max(df_bursts["time_extend"])
    if pad_right:
        assert bin_size is not None, "bin_size must be specified if pad_right is True"
        print(f"Pad bursts to {burst_length_threshold} ms")
        n_bins = int(np.ceil(burst_length_threshold / bin_size))
        df_bursts["burst"] = df_bursts["burst"].apply(
            lambda burst: np.pad(burst, (0, n_bins - len(burst)))
        )
    print("Done")
    return df_bursts


def _normalize_bursts(df_bursts, normalization):
    if normalization is not None:
        print(f"Normalize bursts with {normalization}")
        match normalization:
            case None:
                pass
            case "zscore":
                df_bursts["burst"] = df_bursts["burst"].apply(
                    lambda burst: (burst - np.mean(burst)) / np.std(burst)
                )
            case "peak":
                df_bursts["burst"] = df_bursts["burst"] / df_bursts["peak_height"]
            case "integral":
                df_bursts["burst"] = df_bursts["burst"] / df_bursts["integral"]
            case _:
                raise ValueError(f"Normalization {normalization} not recognized")
    return df_bursts
