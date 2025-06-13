import itertools
import os
import re
from typing import Literal

import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm

from src.folders import (
    get_data_folder,
    get_data_hommersom_test_folder,
    get_data_inhibblock_folder,
    get_data_kapucu_folder,
    get_data_mossink_folder,
)
from src.preprocess import burst_detection

na = np.array


def extract_bursts(
    dataset: Literal[
        "kapucu", "wagenaar", "hommersom_test", "inhibblock", "mossink"
    ] = "wagenaar",
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
    min_length=None,
    min_firing_rate=None,
    smoothing_kernel=None,
):
    """Extract bursts from data files.

    All times in milliseconds.

    Args:
        dataset (Literal["kapucu", "wagenaar", "hommersom_test", "inhibblock", "mossink"], optional): Dataset to extract bursts from.
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
        min_length (float, optional): Minimum length of burst. Defaults to None.
        min_firing_rate (float, optional): Minimum firing rate of burst. Defaults to None.
        smoothing_kernel (int, optional): Kernel size for smoothing burst. Defaults to None.

    Returns:
        df_cultures (pd.DataFrame): Dataframe with columns 'file_name', 'n_bursts', 'burst_start_end'.
            Index is ('batch', 'culture', 'day').
        df_bursts (pd.DataFrame): Dataframe with columns 'start_orig', 'end_orig', 'start_extend', 'end_extend',
            'time_orig', 'time_extend', 'burst', 'peak_height', 'integral'.
            Index is ('batch', 'culture', 'day', 'i_burst').
        burst_matrix (np.ndarray): Matrix of bursts. Shape (n_bursts, n_bins).
    """
    if data_folder is None:
        match dataset:
            case "wagenaar":
                data_folder = os.path.join(get_data_folder(), "extracted")
            case "kapucu":
                data_folder = get_data_kapucu_folder()
            case "hommersom_test":
                data_folder = get_data_hommersom_test_folder()
            case "inhibblock":
                data_folder = get_data_inhibblock_folder()
            case "mossink":
                data_folder = get_data_mossink_folder()
            case _:
                raise ValueError(f"Unknown dataset {dataset}.")
    match dataset:
        case "wagenaar":
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
        case "kapucu":
            df_cultures = _get_kapucu_data_from_file(data_folder)
            df_cultures = _bursts_from_df_culture(
                df_cultures,
                data_folder,
                maxISIstart,
                maxISIb,
                minBdur,
                minIBI,
                minSburst,
            )
            # set index
            df_cultures.set_index(
                ["culture_type", "mea_number", "well_id", "DIV"], inplace=True
            )
        case "hommersom_test":
            df_cultures = _get_hommersom_test_data_from_file(data_folder)
            df_cultures = _bursts_from_df_culture(
                df_cultures,
                data_folder,
                maxISIstart,
                maxISIb,
                minBdur,
                minIBI,
                minSburst,
            )
        case "inhibblock":
            df_cultures = pd.read_pickle(
                os.path.join(get_data_inhibblock_folder(), "df_inhibblock.pkl")
            )
            df_cultures = _bursts_from_df_culture(
                df_cultures,
                data_folder,
                maxISIstart,
                maxISIb,
                minBdur,
                minIBI,
                minSburst,
            )
        case "mossink":
            df_cultures = pd.read_pickle(
                os.path.join(get_data_mossink_folder(), "df_mossink.pkl")
            )
            df_cultures = _bursts_from_df_culture(
                df_cultures,
                data_folder,
                maxISIstart,
                maxISIb,
                minBdur,
                minIBI,
                minSburst,
            )
        case _:
            raise ValueError(f"Unknown dataset {dataset}.")
    df_bursts = _build_bursts_df(
        df_cultures,
        data_folder,
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


def _gid_to_numbers(gid):
    for i, u_id in enumerate(np.unique(gid)):
        gid[gid == u_id] = i
    return gid


def _get_kapucu_data_from_file(data_folder):
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
    print("Done")
    return df


def _bursts_from_df_culture(
    df,
    data_folder,
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
        bursts_start_end = burst_detection.MI_bursts(
            df.at[index, "times"] * 1000,
            maxISIstart=maxISIstart,
            maxISIb=maxISIb,
            minBdur=minBdur,
            minIBI=minIBI,
            minSburst=minSburst,
        )
        df.at[index, "n_bursts"] = len(bursts_start_end)
        df.at[index, "burst_start_end"] = bursts_start_end
    assert df["n_bursts"].sum() > 0, "No bursts found"
    return df


def _get_hommersom_test_data_from_file(
    data_folder,
    fs=12500,  # samples per second
):
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


def _build_bursts_df(
    df_cultures,
    data_folder,
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
        match dataset:
            case "wagenaar":
                file_name = df_cultures.at[index, "file_name"]
                file_path = os.path.join(data_folder, file_name)
                data = np.loadtxt(file_path)[:, 0] * 1000
            case "kapucu" | "hommersom_test" | "inhibblock" | "mossink":
                data = df_cultures.at[index, "times"] * 1000
            case _:
                raise NotImplementedError(f"Dataset {dataset} not implemented")
        if bin_size is not None:
            time_max = np.max(data)
            bins = np.arange(0, time_max + bin_size, bin_size)
            counts, _ = np.histogram(data, bins=bins)
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
                counts, _ = np.histogram(data, bins=bins)
                for i_burst in range(len(bursts_start_end)):
                    df_bursts.at[(*index, i_burst), "burst"] = counts[
                        (n_bins + 1) * i_burst : (n_bins + 1) * (i_burst + 1) - 1
                    ]
            else:
                # otherwise we have to do it one by one
                print(
                    f"Warning: some bursts overlap for {file_name}, binning one by one"
                )
                for i_burst in range(len(bursts_start_end)):
                    index_burst = (*index, i_burst)
                    bins = np.linspace(
                        df_bursts.at[index_burst, "start_extend"],
                        df_bursts.at[index_burst, "end_extend"],
                        n_bins + 1,
                        endpoint=True,
                    )
                    counts, _ = np.histogram(data, bins=bins)
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
