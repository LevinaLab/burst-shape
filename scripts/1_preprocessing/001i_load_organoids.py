import os

import numpy as np
import pynwb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scipy.signal

from src.folders import get_data_organoid_folder

path = "/media/tim/DATA_EXT/Protosequences/001603"
file = os.path.join(
    path,
    "sub-HO1/sub-HO1_ses-20250924T002125.nwb",  # curated_binning
    # "sub-HO1/sub-HO1_ses-20250924T011900_ecephys.nwb",  # electrical recordings
)

# %%
with pynwb.NWBHDF5IO(file, "r") as io:
    nwbfile = io.read()
    print("Age", nwbfile.subject.age)
    print("Available fields in NWBFile:")
    print("Acquisition:", list(nwbfile.acquisition.keys()))
    print("Processing modules:", list(nwbfile.processing.keys()))
    print("Intervals:", list(nwbfile.intervals.keys()))
    print("Stimulus:", list(nwbfile.stimulus.keys()))
    print("Units columns:", list(nwbfile.units.colnames))
    # For each processing module, list its data interfaces
    for name, module in nwbfile.processing.items():
        print(f"Processing module '{name}':", list(module.data_interfaces.keys()))

# %%
with pynwb.NWBHDF5IO(file, "r") as io:
    nwbfile = io.read()
    curated_binning = nwbfile.processing['curated_binning']
    t_spk_mat = curated_binning.data_interfaces['t_spk_mat']
    print(type(t_spk_mat))
    print("Attributes:", dir(t_spk_mat))
    # If it's a TimeSeries or similar, access its data
    if hasattr(t_spk_mat, 'data'):
        data = t_spk_mat.data[:]
        print("Shape of data:", data.shape)
        print("First row:", data[0])

# %%
time_steps = np.arange(data.shape[0]) / 1000 # ms to s
bin_size = 100 # ms

fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 4))
sns.despine()
ax.plot(
    time_steps[::bin_size],
    data.mean(axis=1).reshape(-1, bin_size).mean(axis=1) * 1000,  # to Hz
    color="k",
    linewidth=1,
)
ax.set(
    xlabel="Time (s)",
    ylabel="Mean firing rate (Hz)",
    title="Mean firing rate over time"
)
fig.show()

# %% for each file in sub-HO2 print subject
path_ho2 = "/media/tim/DATA_EXT/Protosequences/001603/sub-HO2"
path_mo10 = "/media/tim/DATA_EXT/Protosequences/001603/sub-MO10"

search_path = path_ho2
for file in os.listdir(search_path):
    if file.endswith(".nwb") and "ecephys" not in file:
        with pynwb.NWBHDF5IO(os.path.join(search_path, file), "r") as io:
            nwbfile = io.read()
            print(file)
            print(nwbfile)#.subject)

# %%
def _binned_data_to_spike_times(binned_data, bin_size_ms=1):
    assert np.all(np.isin(binned_data, [0, 1]))
    spike_times, spike_units = np.where(binned_data)
    spike_times = spike_times * bin_size_ms
    return spike_times, spike_units

def _spike_times_to_binned_data(spike_times, spike_units, num_units, total_time_ms, bin_size_ms=1):
    num_bins = total_time_ms // bin_size_ms
    binned_data = np.zeros((num_bins, num_units), dtype=np.int8)
    binned_indices = spike_times // bin_size_ms
    binned_data[binned_indices, spike_units] = 1
    return binned_data

# %% list all files in a dataframe
base_path = "/media/tim/DATA_EXT/Protosequences/001603"
list_of_dicts = []
for sub in tqdm(os.listdir(base_path)):
    sub_path = os.path.join(base_path, sub)
    if os.path.isdir(sub_path):
        for file in os.listdir(sub_path):
            if file.endswith(".nwb") and "ecephys" not in file:
                file_path = os.path.join(sub_path, file)
                with pynwb.NWBHDF5IO(file_path, "r") as io:
                    nwbfile = io.read()
                    curated_binning = nwbfile.processing['curated_binning']
                    t_spk_mat = curated_binning.data_interfaces['t_spk_mat']
                    binned_data = t_spk_mat.data[:]
                    spike_times, spike_units = _binned_data_to_spike_times(binned_data, bin_size_ms=1)
                    list_of_dicts.append({
                        "file": file,
                        "subject_id": nwbfile.subject.subject_id,
                        "age": nwbfile.subject.age,
                        "species": nwbfile.subject.species,
                        "total_time": binned_data.shape[0] / 1000,  # in seconds
                        "num_units": binned_data.shape[1],
                        "n_spikes": int(binned_data.sum()),
                        "spike_times": spike_times,
                        "spike_units": spike_units,
                    })
df = pd.DataFrame(list_of_dicts)
print(df)

# %% save dataframe
df.to_pickle(
    os.path.join(
        get_data_organoid_folder(),
        "df_organoid_spike_data.pkl",
    )
)
# %% load dataframe
df = pd.read_pickle(
    os.path.join(
        get_data_organoid_folder(),
        "df_organoid_spike_data.pkl",
    )
)
# %% filter: "file" should contain "20250924" and "species" should be "Homo sapiens"
df_human = df[
    (df["file"].apply(lambda x: "20250924" in x)) & (df["species"] == "Homo sapiens")
]
print(df_human)

# %% plot human timeseries with bin size 100 ms
def _smmothed_firing_rate_from_spikes(
        spike_times,
        spike_units,
        num_units,
        total_time_ms,
        gaussian_kernel=100,
        square_kernel=10,
):
    binned_data = _spike_times_to_binned_data(spike_times,
        spike_units,
        num_units,
        total_time_ms)
    firing_rate = binned_data.mean(axis=1) * 1000  # to Hz
    # firing rate: 20 ms sliding square window
    firing_rate = (
        np.convolve(
            firing_rate,
            np.ones(square_kernel) / square_kernel,
            mode="same",
        )
    )
    # firing rate: convolve with 100ms gaussian kernel
    firing_rate = (
            np.convolve(
                firing_rate,
                np.exp(-0.5 * ((np.arange(-3 * gaussian_kernel, 3 * gaussian_kernel + 1) / gaussian_kernel) ** 2)),
                mode='same',
            )
            / (gaussian_kernel * np.sqrt(2 * np.pi))
    )
    return firing_rate


fig_all_burst_together, ax_all_burst_together = plt.subplots(constrained_layout=True, figsize=(8, 4))
fig_normalized_burst, ax_normalized_burst = plt.subplots(constrained_layout=True, figsize=(8, 4))

df_plot = df_human#.head(2) # df  # df_human
use_my_binning = False
bin_size = 100
for idx in df_plot.index:
    if use_my_binning is True:
        spikes = df_plot.at[idx, "spike_times"]
        # less than 200s
        # spikes = spikes[spikes <= 180e3]
        firing_rate, time_steps = np.histogram(spikes, bins=np.arange(0, spikes.max() + bin_size, bin_size))
        time_steps = time_steps / 1000
        firing_rate = firing_rate * 1000 / bin_size / df_plot.at[idx, "num_units"]
    else:
        firing_rate = _smmothed_firing_rate_from_spikes(
            df_plot.at[idx, "spike_times"],
            df_plot.at[idx, "spike_units"],
            df_plot.at[idx, "num_units"],
            (df_plot.at[idx, "total_time"] * 1000).round().astype(int),
        )
        firing_rate_fast = _smmothed_firing_rate_from_spikes(
            df_plot.at[idx, "spike_times"],
            df_plot.at[idx, "spike_units"],
            df_plot.at[idx, "num_units"],
            (df_plot.at[idx, "total_time"] * 1000).round().astype(int),
            gaussian_kernel=5,
            square_kernel=5,
        )
        time_steps = np.arange(firing_rate.shape[0]) / 1000
    root_mean_square = np.sqrt((firing_rate ** 2).mean())
    burst_threshold = 4 * root_mean_square

    # detect peaks
    # [peak_amp, peaks] = findpeaks(pop_rate, 'MinPeakHeight', pop_rms * THR_BURST, 'MinPeakDistance', MIN_BURST_DIFF);
    peaks, peak_properties = scipy.signal.find_peaks(
        firing_rate,
        height=burst_threshold,
        distance=700,
    )

    # find burst start and end (10% of peak value)
    peaks_list = []
    for peak in peaks:
        peak_height = firing_rate[peak]
        # find start
        outside_peak = firing_rate < (.1 * peak_height)
        peak_start = np.where(outside_peak[:peak])[0][-1]
        peak_end = np.where(outside_peak[peak:])[0][0] + peak
        peaks_list.append({
            "start": peak_start,
            "end": peak_end,
            "height": peak_height,
            "times": time_steps[peak_start:peak_end],
            "times_aligned": time_steps[peak_start:peak_end] - time_steps[peak],
            "firing_rate": firing_rate[peak_start:peak_end],
            "firing_rate_fast": firing_rate_fast[peak_start:peak_end],
        })
    df_bursts = pd.DataFrame(peaks_list)
    avg_start = df_bursts["times_aligned"].apply(lambda x: x[0]).mean()
    avg_end = df_bursts["times_aligned"].apply(lambda x: x[-1]).mean()
    time_bins = np.linspace(avg_start, avg_end, num=50)
    def _normalize_burst(time_points, values):
        bin_idx = np.digitize(time_points, time_bins) - 1
        valid = (bin_idx >= 0) & (bin_idx < len(time_bins) - 1)

        bin_sum = np.bincount(bin_idx[valid], weights=values[valid],
                              minlength=len(time_bins) - 1)
        bin_count = np.bincount(bin_idx[valid],
                                minlength=len(time_bins) - 1)

        bin_mean = np.divide(
            bin_sum, bin_count,
            out=np.zeros_like(bin_sum),
            where=bin_count > 0
        )
        return bin_mean
    df_bursts["firing_rate_binned"] = pd.Series()
    for idx_burst in df_bursts.index:
        df_bursts.at[idx_burst, "firing_rate_binned"] = _normalize_burst(
            df_bursts.at[idx_burst, "times_aligned"],
            df_bursts.at[idx_burst, "firing_rate_fast"],
        )
    # avg_burst = df_bursts["firing_rate_binned"].mean()


    fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 4))
    sns.despine()
    ax.plot(
        time_steps[:-1] if use_my_binning else time_steps,
        firing_rate,
        color="k",
        linewidth=1,
    )
    ax.axhline(burst_threshold, color="orange", linestyle="--")
    # highlight peaks
    ax.scatter(
        time_steps[peaks],
        firing_rate[peaks],
    )
    ax.set(
        xlabel="Time (s)",
        ylabel="Mean firing rate (Hz)",
        title=f"subject = {df_plot.at[idx, 'subject_id']} ({len(peaks)} peaks)",
    )
    fig.show()

    fig, ax = plt.subplots(constrained_layout=True, figsize=(8, 4))
    sns.despine()
    for idx_burst in df_bursts.index:
        ax.plot(
            df_bursts.at[idx_burst, "times_aligned"],
            df_bursts.at[idx_burst, "firing_rate_fast"],
            color="k",
            alpha=0.3,
        )
    ax.set(
        xlabel="Time relative to peak (s)",
        ylabel="Mean firing rate (Hz)",
        title=f"subject = {df_plot.at[idx, 'subject_id']} ({len(peaks)} peaks)",
    )
    fig.show()

    colormap = {
        "HO1" : "C0",
        "HO2" : "C1",
        "HO3" : "C2",
        "HO4" : "C3",
        # "HO5" : "C4",
        # "HO6" : "C5",
        # "HO7" : "C6",
        # "HO8" : "C7",
    }
    if df_plot.at[idx, 'subject_id'] in colormap.keys():
        for idx_burst in df_bursts.index:
            ax_all_burst_together.plot(
                df_bursts.at[idx_burst, "times_aligned"],
                df_bursts.at[idx_burst, "firing_rate_fast"],
                color=colormap[df_plot.at[idx, 'subject_id']],
                alpha=0.3,
            )

        avg_burst = df_bursts["firing_rate_binned"].mean()
        avg_burst = avg_burst / avg_burst.mean()
        ax_normalized_burst.plot(
            # np.linspace(0, 1, num=49),
            time_bins[:-1],
            avg_burst,
            color=colormap[df_plot.at[idx, 'subject_id']],
        )


fig_all_burst_together.show()
fig_normalized_burst.show()