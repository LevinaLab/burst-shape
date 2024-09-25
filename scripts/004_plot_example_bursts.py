import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.folders import get_data_folder
from src.persistence import load_burst_matrix, load_df_bursts, load_df_cultures

# actual parameters copied from params.json or string
burst_extraction_params = "burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"


# load data
df_cultures = load_df_cultures(burst_extraction_params)
df_bursts = load_df_bursts(burst_extraction_params)
bursts_mat = load_burst_matrix(burst_extraction_params)

# %% Plot examples of burst extraction
idx = 110
bin_size = 10  # ms
data = (
    np.loadtxt(
        os.path.join(get_data_folder(), "extracted", df_cultures.iloc[idx].file_name)
    )[:, 0]
    * 1000
)
bins = np.arange(0, data.max() + bin_size, bin_size)
# histogram and convert to Hz
data = np.histogram(data, bins=bins)[0] / (bin_size / 1000)
batch, culture, day = df_cultures.iloc[idx].name
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
fig.suptitle(
    f"Burst extraction, batch {batch}, culture {culture}, day {day}"
    f"\n{df_cultures.iloc[idx].n_bursts} bursts"
)
sns.despine()
ax.plot(bins[:-1], data, color="k")
# highlight burst from df_bursts
for idx in df_bursts.loc[(batch, culture, day, slice(None))].index:
    ax.axvspan(
        df_bursts.loc[(batch, culture, day, idx)].start_extend,
        df_bursts.loc[(batch, culture, day, idx)].end_extend,
        color="r",
        alpha=0.5,
    )
ax.set_xlabel(f"Time [ms], {bin_size} ms bins")
ax.set_ylabel("Rate [Hz]")
fig.show()

# %% Plot number of bursts per culture
print("Number of cultures: ", len(df_cultures))
print(
    f"Number of cultures with no bursts: {len(df_cultures[df_cultures.n_bursts == 0])}"
)
print(f"Number of cultures with bursts: {len(df_cultures[df_cultures.n_bursts > 0])}")
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.despine()
sns.histplot(
    data=df_cultures,
    x="n_bursts",
    log_scale=True,
    stat="count",
    bins=100,
    ax=ax,
)
ax.set_xlabel("Number of bursts per culture")
fig.show()

# %% Plot burst length distribution
for extended in [True, False]:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.despine()
    sns.histplot(
        data=df_bursts,
        x="time_extend" if extended else "time_orig",
        log_scale=True,
        stat="probability",
        bins=100,
        ax=ax,
    )
    ax.set_xlabel(
        f"Burst length [ms] ({'including extension' if extended else 'unextended'})"
    )
    fig.show()

# %% Plot example bursts
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(bursts_mat[950:1000, :].T, color="k")
ax.set_xlabel("Time [50 bins, arbitrary units]")
ax.set_ylabel("Rate [Hz]")
ax.set_title("Example bursts")
sns.despine()
fig.tight_layout()
fig.show()

# %% same with real time
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
sns.despine()
for idx in range(950, 1000):
    ax.plot(
        np.linspace(-50, df_bursts.iloc[idx].time_orig + 50, 50),
        bursts_mat[idx, :],
        color="k",
        alpha=0.5,
    )
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Rate [Hz]")
fig.show()
