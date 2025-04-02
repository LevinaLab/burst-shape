import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.persistence import load_df_bursts, load_df_cultures
from src.utils.classical_features import get_classical_features

###############################################################################
#                           Parameters                                        #
###############################################################################
burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4_outlier_removed"
    # "dataset_kapucu_burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
)

if "kapucu" in burst_extraction_params:
    dataset = "kapucu"
elif "hommersom" in burst_extraction_params:
    dataset = "hommersom"
elif "inhibblock" in burst_extraction_params:
    dataset = "inhibblock"
    group_label = "drug_label"
    batch_label = "div"
else:
    dataset = "wagenaar"
print(f"Detected dataset: {dataset}")


# get df_bursts and labels
df_cultures = load_df_cultures(burst_extraction_params)
df_cultures_reset = df_cultures.reset_index()
df_bursts = load_df_bursts(burst_extraction_params)


def _plot(df_cultures, y_labels):
    df_cultures_reset = df_cultures.reset_index()
    for y_label in y_labels:
        fig, ax = plt.subplots(constrained_layout=True)
        sns.despine()
        sns.violinplot(
            data=df_cultures.reset_index(),
            x=batch_label,
            y=y_label,
            hue=group_label,
            inner="point",
        )
        fig.show()

        print(f"\nTest {y_label}:")
        for x in df_cultures_reset[batch_label].unique():
            samples = []
            for hue in df_cultures_reset[df_cultures_reset[batch_label] == x][
                group_label
            ].unique():
                sample = df_cultures_reset[
                    (df_cultures_reset[batch_label] == x)
                    & (df_cultures_reset[group_label] == hue)
                ][y_label]
                samples.append(sample.values.astype(float))
            ttest_result = scipy.stats.ttest_ind(*samples)
            stars = "*" if ttest_result.pvalue < 0.05 else ""
            print(f"{stars}Batch {x}:")
            print(f"{stars}{ttest_result}")


# %% plot culture differences in number of bursts and spikes and inter-burst-intervals (IBI)
df_cultures["n_spikes"] = df_cultures["times"].apply(len)
df_cultures["spikes_per_burst"] = df_cultures["n_spikes"] / df_cultures["n_bursts"]
# df_cultures["burst_starts"] = df_cultures["burst_start_end"].apply(lambda start_end_list: [start_end[0] for start_end in start_end_list])
df_cultures["IBIs"] = df_cultures["burst_start_end"].apply(
    lambda start_end_list: [
        start_end_next[0] - start_end_previous[1]
        for start_end_previous, start_end_next in zip(
            start_end_list[:-1], start_end_list[1:]
        )
    ]
)
df_cultures["IBI_mean"] = df_cultures["IBIs"].apply(lambda x: np.mean(x))
df_cultures["IBI_std"] = df_cultures["IBIs"].apply(lambda x: np.std(x))
df_cultures["IBI_CV"] = df_cultures["IBI_std"] / df_cultures["IBI_mean"]

_plot(
    df_cultures,
    ["n_spikes", "n_bursts", "spikes_per_burst", "IBI_mean", "IBI_std", "IBI_CV"],
)

# %% plot burst characteristic differences
for y_label in ["avg_burst_duration", "avg_firing_rate"]:
    df_cultures[y_label] = pd.Series(dtype=float)
for index in tqdm(df_cultures.index):
    df_bursts_select = df_bursts.loc[index]
    df_cultures.at[index, "avg_burst_duration"] = df_bursts_select["time_orig"].mean()
    df_cultures.at[index, "avg_firing_rate"] = df_bursts_select["firing_rate"].mean()

_plot(df_cultures, ["avg_burst_duration", "avg_firing_rate"])


# %% classical features as in van Hugte et al.
df_cultures, features = get_classical_features(df_cultures, df_bursts, dataset)

_plot(df_cultures, features)
