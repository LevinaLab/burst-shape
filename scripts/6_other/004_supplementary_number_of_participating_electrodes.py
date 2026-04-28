"""
Check the portion of participating electrodes in each burst.

Report useful statistics:
- histogram of participating electrodes
- averages
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from persistence import load_df_bursts, load_df_cultures
from persistence.spike_times import get_spike_times_in_milliseconds
from plot import prepare_plotting, savefig
from settings import get_dataset_from_burst_extraction_params

cm = prepare_plotting()

burst_extraction_params = (
    # "burst_dataset_wagenaar_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"  # noqa: E501
    # "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"  # noqa: E501
    # "burst_dataset_hommersom_binary_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"  # noqa: E501
    "burst_dataset_mossink_KS"
    # "burst_dataset_mossink_maxISIstart_50_maxISIb_50_minBdur_100_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30"  # noqa: E501
)
dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
df_cultures = load_df_cultures(burst_extraction_params)
df_bursts = load_df_bursts(burst_extraction_params)
df_cultures.sort_index(inplace=True)
df_bursts.sort_index(inplace=True)


# %% Compute number of participating electrodes per burst
# Iterate existing df_bursts rows per culture (use df_bursts' i_burst labels)
# This uses the authoritative index stored in df_bursts (loaded from persistence)
# and avoids creating any new MultiIndex labels during assignment.
# Ensure the target column exists with nullable integer dtype
df_bursts["n_participating_electrodes"] = pd.Series(
    index=df_bursts.index, dtype="Int64"
)
for index_culture in tqdm(df_cultures.index):
    # skip cultures that don't have any bursts recorded in df_bursts
    try:
        bursts_df = df_bursts.loc[index_culture]
    except KeyError:
        # no bursts for this culture in df_bursts
        continue
    st, gid = get_spike_times_in_milliseconds(df_cultures, index_culture, dataset)
    # bursts_df.index contains the i_burst labels (may be non-contiguous)
    for i_burst in bursts_df.index.tolist():
        start = bursts_df.at[i_burst, "start_orig"]
        end = bursts_df.at[i_burst, "end_orig"]
        burst_filter = (st >= start) & (st <= end)
        gid_filter = gid[burst_filter]
        df_bursts.at[(*index_culture, i_burst), "n_participating_electrodes"] = len(
            np.unique(gid_filter)
        )

# assert no NaNs
assert df_bursts["n_participating_electrodes"].hasnans is False


# %% Plot histogram
values = df_bursts["n_participating_electrodes"].dropna().astype(int)
if values.empty:
    raise ValueError("No participating-electrodes values available for plotting.")

mean_value = values.mean()
median_value = values.median()
percentile_10 = values.quantile(0.10)
bin_min = values.min() - 0.5
bin_max = values.max() + 0.5
bins = np.arange(bin_min, bin_max + 1, 1)

fig, ax = plt.subplots(figsize=(8 * cm, 4 * cm), constrained_layout=True)
sns.despine()
sns.histplot(
    x=values,
    bins=bins,
    stat="percent",
    element="step",
    color="black",
    ax=ax,
)
ax.axvline(
    mean_value,
    color="tab:red",
    linestyle="--",
    linewidth=1.5,
    label=f"Mean = {mean_value:.2f}",
)
ax.axvline(
    median_value,
    color="tab:blue",
    linestyle="-.",
    linewidth=1.5,
    label=f"Median = {median_value:.2f}",
)
ax.axvline(
    percentile_10,
    color="tab:green",
    linestyle="--",
    linewidth=1.5,
    label=f"10th Percentile = {percentile_10:.2f}",
)
ax.set_xlabel("Number of participating electrodes")
ax.set_ylabel("Percent of bursts")
ax.legend(frameon=False)
fig.show()
savefig(
    fig,
    f"supplementary_participating_electrodes_histogram_{dataset}",
    file_format=["svg"],
)
