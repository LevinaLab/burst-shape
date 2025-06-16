import itertools
import warnings
from typing import Callable, Literal

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.persistence.spike_times import get_spike_times_in_milliseconds
from src.preprocess import burst_detection


def extract_bursts(
    dataset: Literal[
        "kapucu", "wagenaar", "hommersom", "hommersom_test", "inhibblock", "mossink"
    ],
    construct_df_cultures: Callable[[], pd.DataFrame],
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
    min_length=None,
    min_firing_rate=None,
    smoothing_kernel=None,
):
    """Extract bursts from data files.

    All times in milliseconds.

    Args:
        dataset (Literal["kapucu", "wagenaar", "hommersom", "hommersom_test", "inhibblock", "mossink"]):
            Dataset to extract bursts from.
        construct_df_cultures (Callable[[], pd.DataFrame]): A function that constructs the initial df_cultures.
            It must contain the index, likely a multi-index. This index will be used both for df_cultures and df_bursts.
            It is recommended that it has a columns 'times' (in seconds) and 'gid' where the spike times can be loaded from.
            If you choose a different format for storing the spike times,
            you must modify the function get_spike_times_in_seconds() elsewhere.
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
        min_length (float, optional): Minimum length of burst. Defaults to None.
        min_firing_rate (float, optional): Minimum firing rate of burst. Defaults to None.
        smoothing_kernel (int, optional): Kernel size for smoothing burst. Defaults to None.

    Returns:
        df_cultures (pd.DataFrame): Dataframe with index as constructed, but sorted.
            It will contain additional columns 'n_bursts' and 'burst_start_end'.
        df_bursts (pd.DataFrame):
            Dataframe with index same as df_cultures but one additional index level 'i_burst'.
            It contains columns 'start_orig', 'end_orig', 'start_extend', 'end_extend',
            'time_orig', 'time_extend', 'burst', 'peak_height', 'integral'.
        burst_matrix (np.ndarray):
            Matrix of bursts. Shape (n_bursts, n_bins).
            It has the same order as df_bursts.
    """

    df_cultures = construct_df_cultures()
    df_cultures = _bursts_from_df_culture(
        df_cultures,
        dataset,
        maxISIstart,
        maxISIb,
        minBdur,
        minIBI,
        minSburst,
    )
    df_bursts = _build_bursts_df(
        df_cultures,
        bin_size,
        n_bins,
        extend_left,
        extend_right,
        smoothing_kernel,
        dataset,
    )
    df_bursts = _filter_bursts(
        df_cultures,
        df_bursts,
        burst_length_threshold,
        pad_right,
        bin_size,
        min_length,
        min_firing_rate,
    )
    df_bursts = _normalize_bursts(df_bursts, normalization)

    # sort index for efficient access
    df_bursts.sort_index(inplace=True)
    df_cultures.sort_index(inplace=True)

    burst_matrix = np.stack(df_bursts["burst"].values)
    return df_cultures, df_bursts, burst_matrix


def _bursts_from_df_culture(
    df,
    dataset,
    maxISIstart,
    maxISIb,
    minBdur,
    minIBI,
    minSburst,
):
    """Detect bursts in df_culture.

    df_cultures must have column "times" containing spike times in seconds.
    Creates new columns "n_bursts", "burst_start_end"
    """
    df["n_bursts"] = pd.Series(dtype=object)
    df["burst_start_end"] = pd.Series(dtype=object)
    for index in tqdm(df.index, desc="Compute burst times for each culture"):
        st, _ = get_spike_times_in_milliseconds(df, index, dataset)
        if isinstance(st, np.ndarray):
            bursts_start_end = burst_detection.MI_bursts(
                st,
                maxISIstart=maxISIstart,
                maxISIb=maxISIb,
                minBdur=minBdur,
                minIBI=minIBI,
                minSburst=minSburst,
            )
            df.at[index, "n_bursts"] = len(bursts_start_end)
            df.at[index, "burst_start_end"] = bursts_start_end
        else:
            warnings.warn(
                f"The spike times of index {index} has type {type(st)}, but type ndarry was expected. "
                "Possible reason are either wrong format or that there are 0 (NaN) or only 1 spike (float)."
                "Continuing by setting n_bursts to 0."
            )
            df.at[index, "n_bursts"] = 0
            df.at[index, "burst_start_end"] = []
    assert df["n_bursts"].sum() > 0, "No bursts found"
    return df


def _build_bursts_df(
    df_cultures,
    bin_size,
    n_bins,
    extend_left,
    extend_right,
    smoothing_kernel,
    dataset,
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
            "firing_rate",
        ],
        index=pd.MultiIndex.from_tuples(
            itertools.chain.from_iterable(
                [
                    [
                        (*index, i_burst)
                        for i_burst in range(df_cultures.at[index, "n_bursts"])
                    ]
                    for index in df_cultures.index
                ]
            ),
            names=[*df_cultures.index.names, "i_burst"],
        ),
    )
    for index in tqdm(df_cultures.index, desc="Build dataframe of bursts with times."):
        bursts_start_end = df_cultures.at[index, "burst_start_end"]
        for i_burst, (start, end) in enumerate(bursts_start_end):
            # df_bursts.at[(batch, culture, day, i_burst), "i_burst"] = i_burst
            df_bursts.at[(*index, i_burst), "start_orig"] = start
            df_bursts.at[(*index, i_burst), "end_orig"] = end
            df_bursts.at[(*index, i_burst), "start_extend"] = start - extend_left
            df_bursts.at[(*index, i_burst), "end_extend"] = end + extend_right
            df_bursts.at[(*index, i_burst), "time_orig"] = end - start
            df_bursts.at[(*index, i_burst), "time_extend"] = (
                end - start + extend_left + extend_right
            )

    assert (
        bin_size is not None or n_bins is not None
    ), "Either bin_size or n_bins must be specified"
    assert (
        bin_size is None or n_bins is None
    ), "Only one of bin_size and n_bins can be specified"
    for index in tqdm(df_cultures.index, desc="Extract bursts and bin them."):
        bursts_start_end = df_cultures.at[index, "burst_start_end"]
        if len(bursts_start_end) == 0:
            continue
        st, _ = get_spike_times_in_milliseconds(df_cultures, index, dataset)
        if bin_size is not None:
            time_max = np.max(st)
            bins = np.arange(0, time_max + bin_size, bin_size)
            counts, _ = np.histogram(st, bins=bins)
            for i_burst in range(len(bursts_start_end)):
                index_burst = (*index, i_burst)
                df_bursts.at[index_burst, "burst"] = counts[
                    int(
                        np.floor(df_bursts.at[index_burst, "start_extend"] / bin_size)
                    ) : int(np.ceil(df_bursts.at[index_burst, "end_extend"] / bin_size))
                ]
        if n_bins is not None:
            if np.all(
                [
                    df_bursts.at[(*index, i_burst), "start_extend"]
                    > df_bursts.at[(*index, i_burst - 1), "end_extend"]
                    for i_burst in range(1, len(bursts_start_end))
                ]
            ):
                # if all starts are bigger than previous ends, then we can bin simultaneously
                bins = np.concatenate(
                    [
                        np.linspace(
                            df_bursts.at[(*index, i_burst), "start_extend"],
                            df_bursts.at[(*index, i_burst), "end_extend"],
                            n_bins + 1,
                            endpoint=True,
                        )
                        for i_burst in range(len(bursts_start_end))
                    ]
                )
                counts, _ = np.histogram(st, bins=bins)
                for i_burst in range(len(bursts_start_end)):
                    df_bursts.at[(*index, i_burst), "burst"] = counts[
                        (n_bins + 1) * i_burst : (n_bins + 1) * (i_burst + 1) - 1
                    ]
            else:
                # otherwise we have to do it one by one
                print(
                    f"Warning: some bursts overlap for index={index}, "
                    "continuing by binning one by one. "
                    "This can happen when padding burst duration, "
                    "otherwise this should not happen."
                )
                for i_burst in range(len(bursts_start_end)):
                    index_burst = (*index, i_burst)
                    bins = np.linspace(
                        df_bursts.at[index_burst, "start_extend"],
                        df_bursts.at[index_burst, "end_extend"],
                        n_bins + 1,
                        endpoint=True,
                    )
                    counts, _ = np.histogram(st, bins=bins)
                    df_bursts.at[index_burst, "burst"] = counts

    # convert bursts to firing rate (Hz)
    if n_bins is not None:
        df_bursts["burst"] = (
            df_bursts["burst"] / df_bursts["time_extend"] * 1000 * n_bins
        )
    if bin_size is not None:
        df_bursts["burst"] = df_bursts["burst"] / bin_size * 1000

    # smooth bursts
    if smoothing_kernel is not None:
        assert (
            n_bins is not None
        ), "n_bins must be specified if smoothing_kernel is not None"
        print(f"Smooth bursts with kernel size {smoothing_kernel}")
        for index in tqdm(df_bursts.index, desc="Smooth bursts"):
            kernel_size_float = (
                smoothing_kernel / df_bursts.at[index, "time_extend"] * n_bins
            )
            if kernel_size_float <= 1:
                continue
            kernel_size = np.ceil(kernel_size_float)
            if kernel_size % 2 == 0:
                kernel_size += 1
            assert kernel_size >= 3, "Kernel size must be at least 3"
            kernel = np.ones(int(kernel_size)) / kernel_size_float
            kernel[[0, -1]] = (1 - (kernel_size - 2) / kernel_size_float) / 2
            assert np.isclose(np.sum(kernel), 1), "Kernel must sum to 1"
            df_bursts.at[index, "burst"] = np.convolve(
                df_bursts.at[index, "burst"], kernel, mode="same"
            )

    # compute peak height and integral
    for index in tqdm(df_bursts.index, desc="Compute peak height and integral"):
        burst = df_bursts.at[index, "burst"]
        df_bursts.at[index, "peak_height"] = np.max(burst)
        df_bursts.at[index, "integral"] = np.sum(burst)
        df_bursts.at[index, "firing_rate"] = np.mean(burst)
    return df_bursts


def _remove_bursts(df_cultures, df_bursts, index_to_remove):
    for index_burst in df_bursts.index[index_to_remove]:
        index_culture = index_burst[:-1]
        df_cultures.at[index_culture, "n_bursts"] = (
            df_cultures.at[index_culture, "n_bursts"] - 1
        )
        start_orig = df_bursts.at[index_burst, "start_orig"]
        df_cultures.at[index_culture, "burst_start_end"] = [
            start_end
            for start_end in df_cultures.at[index_culture, "burst_start_end"]
            if not np.isclose(start_orig, start_end[0])
        ]
        assert df_cultures.at[index_culture, "n_bursts"] == len(
            df_cultures.at[index_culture, "burst_start_end"]
        )
    df_bursts = df_bursts[~index_to_remove]
    assert df_cultures["n_bursts"].sum() == len(df_bursts)
    return df_cultures, df_bursts


def _filter_bursts(
    df_cultures,
    df_bursts,
    burst_length_threshold,
    pad_right,
    bin_size,
    min_length,
    min_firing_rate,
):
    print("Filter bursts (burst length, min_firing_rate, pad right)")
    if burst_length_threshold is not None:
        len_before = len(df_bursts)
        index_to_remove = df_bursts["time_extend"] > burst_length_threshold
        df_cultures, df_bursts = _remove_bursts(df_cultures, df_bursts, index_to_remove)
        len_after = len(df_bursts)
        print(
            f"Removed {len_before - len_after} bursts above threshold ({burst_length_threshold} ms)"
        )
    else:
        burst_length_threshold = np.max(df_bursts["time_extend"])
    if min_length is not None:
        len_before = len(df_bursts)
        index_to_remove = df_bursts["time_extend"] < min_length
        df_cultures, df_bursts = _remove_bursts(df_cultures, df_bursts, index_to_remove)
        len_after = len(df_bursts)
        print(
            f"Removed {len_before - len_after} bursts below threshold ({min_length} ms)"
        )
    if min_firing_rate is not None:
        len_before = len(df_bursts)
        index_to_remove = df_bursts["firing_rate"] < min_firing_rate
        df_cultures, df_bursts = _remove_bursts(df_cultures, df_bursts, index_to_remove)
        len_after = len(df_bursts)
        print(
            f"Removed {len_before - len_after} bursts below threshold ({min_firing_rate} Hz)"
        )
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
