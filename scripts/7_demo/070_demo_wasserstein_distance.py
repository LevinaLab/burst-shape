import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.folders import get_fig_folder
from src.persistence import load_df_bursts, load_df_cultures
from src.plot import prepare_plotting
from src.settings import get_dataset_from_burst_extraction_params

cm = prepare_plotting()

burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_test_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30"
)
dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
df_cultures = load_df_cultures(burst_extraction_params)
df_bursts = load_df_bursts(burst_extraction_params, cv_params=None)
print(f"Detected dataset: {dataset}")

# %% inhibblock examples
plot_only_cumulative = True
if dataset == "inhibblock":
    examples_indices = [
        ("control", 17, 12, 124),
        # ("bic", 17, 2, 115),
        ("control", 17, 4, 77),
        # ("bic", 17, 7, 98),
    ]

    fig, axs = plt.subplots(
        ncols=1 if plot_only_cumulative else 2,
        constrained_layout=True,
        figsize=(6 * cm, 3 * cm),
        sharex="col",
    )
    if plot_only_cumulative:
        ax = axs
    sns.despine()
    # sns.despine()
    for i, index in enumerate(examples_indices):
        if not plot_only_cumulative:
            ax = axs[0]
            ax.plot(df_bursts.at[index, "burst"])

            ax = axs[1]
        ax.plot(np.cumsum(df_bursts.at[index, "burst"]))

    if not plot_only_cumulative:
        ax = axs[0]
        ax.set_xlabel("Time [a.u.]")
        ax.set_ylabel("Density")

        ax = axs[1]
    ax.set_xlabel("Time [a.u.]")
    ax.set_ylabel("Cumulative")
    ax.fill_between(
        np.arange(50),
        np.cumsum(df_bursts.at[examples_indices[0], "burst"]),
        np.cumsum(df_bursts.at[examples_indices[1], "burst"]),
        color="C2",
        alpha=0.5,
        edgecolor=None,
    )

    fig.show()
    fig.savefig(
        os.path.join(get_fig_folder(), f"demo_Wasserstein.svg"),
        transparent=True,
    )
