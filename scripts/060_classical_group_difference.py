import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

from src.persistence import load_df_bursts, load_df_cultures
from src.plot import (
    get_group_colors,
    get_group_labels,
    label_sig_diff,
    prepare_plotting,
    savefig,
)
from src.prediction.define_target import make_target_label
from src.settings import get_dataset_from_burst_extraction_params
from src.utils.classical_features import get_classical_features

cm = prepare_plotting()
###############################################################################
#                           Parameters                                        #
###############################################################################
burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4_outlier_removed"
    # "dataset_kapucu_burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_test_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    "burst_dataset_hommersom_binary_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
)

dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
print(f"Detected dataset: {dataset}")


# get df_bursts and labels
df_cultures = load_df_cultures(burst_extraction_params)
df_cultures_reset = df_cultures.reset_index()
df_bursts = load_df_bursts(burst_extraction_params)

# %% classical features as in van Hugte et al.
df_cultures, features = get_classical_features(df_cultures, df_bursts, dataset)

# %%
df_cultures, target_label = make_target_label(dataset, df_cultures)
print("Target label", target_label)

if dataset == "inhibblock":
    df_cultures = df_cultures.sort_values(by="target_label", ascending=False)


def _run_group_comparison(df_cultures, compare_column, plot_significance=True):
    grouped = df_cultures.groupby("target_label", observed=True)
    groups = list(grouped.groups.keys())

    if len(groups) == 2:
        group1 = grouped.get_group(groups[0])[compare_column]
        group2 = grouped.get_group(groups[1])[compare_column]

        # Independent two-sample t-test
        t_stat, p_val = stats.ttest_ind(
            group1, group2, equal_var=False
        )  # Welch's t-test
        print("Run ttest")
        return_values = (t_stat, p_val)
    else:
        # For more than 2 groups, use ANOVA or Kruskal-Wallis
        values = [grouped.get_group(g)[compare_column] for g in groups]
        f_stat, p_val = stats.f_oneway(*values)
        print("Run f_oneway")
        return_values = (f_stat, p_val)
    fig, ax = plt.subplots(figsize=(4 * cm, 5 * cm), constrained_layout=True)
    ax.set_position([0.5, 0.2, 0.4, 0.7])
    sns.despine()
    sns.boxplot(
        data=df_cultures.reset_index(),
        x="target_label",
        hue="target_label",
        y=compare_column,
        palette=get_group_colors(dataset),
        flierprops=dict(
            marker="o", markerfacecolor="black", markersize=4, linestyle="none"
        ),
    )
    ax.set_xlabel("")
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(
        [get_group_labels(dataset, lbl.get_text()) for lbl in ax.get_xticklabels()]
    )
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_y(label.get_position()[1] - 0.1 * (i % 2))
    if plot_significance is True:
        label_sig_diff(
            ax,
            [0, 1],
            df_cultures[compare_column].max(),
            p_val,
            0.1
            * (df_cultures[compare_column].max() - df_cultures[compare_column].min()),
            0.05
            * (df_cultures[compare_column].max() - df_cultures[compare_column].min()),
            "k",
            10,
            1,
        )
    fig.show()

    print(return_values)
    return *return_values, fig, ax


for column in features:
    print(f"\n{column}:")
    mean_std = df_cultures.groupby("target_label", observed=True)[column].agg(
        ["mean", "std"]
    )
    print(mean_std)
    _, _, fig, ax = _run_group_comparison(df_cultures, column)
    savefig(
        fig,
        f"{dataset}_classical_feature_{column}_comparison",
        file_format=["pdf", "svg"],
    )
