import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.persistence import load_df_bursts, load_df_cultures
from src.plot import get_group_colors, prepare_plotting, savefig
from src.prediction.define_target import make_target_label
from src.settings import get_dataset_from_burst_extraction_params

cm = prepare_plotting()

# parameters which clustering to plot
burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_test_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_hommersom_binary_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
)
dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
print(f"Detected dataset: {dataset}")

# load bursts
df_bursts = load_df_bursts(burst_extraction_params)
df_cultures = load_df_cultures(burst_extraction_params)

# make target label
df_cultures, df_bursts, target_label = make_target_label(
    dataset, df_cultures, df_bursts
)

# %%
seed = 0
np.random.seed(seed)

n_labels = len(df_cultures["target_label"].unique())
fig, axs = plt.subplots(
    1,
    n_labels,
    figsize=(n_labels * 1.5 * cm, 1.5 * cm),
    sharey=True,
    constrained_layout=True,
)
sns.despine(bottom=True, left=True)
for ax, label in zip(axs, df_cultures["target_label"].unique()):
    # select 10 random bursts from df_bursts where df_cultures.target_label == label
    sample_bursts = df_bursts[df_bursts["target_label"] == label].sample(
        3, random_state=seed
    )
    print(f"Label {label}: {len(sample_bursts)} bursts")

    sample_bursts = sample_bursts["burst"].values
    x_values = np.arange(50) + 0.5
    for burst in sample_bursts:
        ax.plot(x_values, burst, color=get_group_colors(dataset)[label], alpha=1, lw=1)

    # remove all labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    fig.show()
    savefig(fig, f"example_bursts_{dataset}", file_format=["svg", "png"])
