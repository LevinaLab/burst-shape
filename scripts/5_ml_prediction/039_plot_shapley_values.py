import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from src.folders import get_fig_folder
from src.persistence import load_df_cultures
from src.persistence.xgboost import load_xgboost_results
from src.plot import get_group_colors, prepare_plotting
from src.prediction.define_target import make_target_label
from src.settings import (
    get_chosen_spectral_embedding_params,
    get_dataset_from_burst_extraction_params,
)

cm = prepare_plotting()

burst_extraction_params_list = [
    "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30",
    "burst_dataset_hommersom_binary_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30",
    "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4",
    "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4",
    "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30",
    "burst_dataset_mossink_KS",
    # "burst_dataset_mossink_MELAS",
]
cv_type = (
    # "RepeatedStratifiedKFold"
    "StratifiedShuffleSplit"
)

share_shape_importance = []

n_classes = {}
df_accuracies = []
for burst_extraction_params in burst_extraction_params_list:
    dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
    spectral_clustering_params = get_chosen_spectral_embedding_params(dataset)

    for feature_set_name in ["combined"]:  # , "shape", "traditional"]:
        (
            features,
            nested_scores,
            all_shap_values,
            all_y_pred,
            all_y_test,
        ) = load_xgboost_results(
            burst_extraction_params,
            spectral_clustering_params,
            feature_set_name,
            cv_type,
        )
        # ----------------------------------------------------------------------------
        # Plotting
        average_score = np.mean(nested_scores)
        print(f"Average balanced accuracy:\t{average_score:.3f}")

        # Aggregate SHAP values across folds
        mean_shap_values = np.mean(all_shap_values, axis=0)
        mean_abs_shap_values = np.mean(np.abs(mean_shap_values), axis=0)
        # sorted_importances = pd.Series(mean_abs_shap_values, index=X.columns).sort_values(ascending=False)
        # extract_corr_and_impact(X, list(X.columns), mean_shap_values, mean_abs_shap_values

        # Plot aggregated SHAP values
        if mean_shap_values.ndim == 2:
            fig, ax = plt.subplots(figsize=(4 * cm, 6 * cm), constrained_layout=True)
            sns.despine()
            df_plot = pd.DataFrame(
                {
                    "Feature": features,
                    "SHAP": mean_abs_shap_values,
                }
            )
            order = features[np.argsort(mean_abs_shap_values)[::-1]]
            sns.barplot(
                data=df_plot,
                x="Feature",
                y="SHAP",
                order=order,
            )
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_ylabel("mean([SHAP value])")
            fig.show()
            for file_type in ["pdf", "svg"]:
                fig.savefig(
                    os.path.join(
                        get_fig_folder(),
                        f"{dataset}_xgboost_{feature_set_name}_shapley_values_bar_vertical.{file_type}",
                    ),
                    transparent=True,
                )
        else:
            df_cultures = load_df_cultures(burst_extraction_params)
            df_cultures = df_cultures[df_cultures["n_bursts"] > 0]
            df_cultures, target_label = make_target_label(
                dataset, df_cultures, None, special_target=False
            )

            label_encoder = LabelEncoder()
            df_cultures["encoded_label"] = label_encoder.fit_transform(
                df_cultures["target_label"]
            )

            fig, ax = plt.subplots(figsize=(4 * cm, 6 * cm), constrained_layout=True)
            sns.despine()
            categories = list(label_encoder.classes_)
            df_plot = (
                pd.DataFrame(mean_abs_shap_values, index=features, columns=categories)
                .reset_index()
                .melt(id_vars="index", var_name="Category", value_name="SHAP")
            )
            df_plot = df_plot.rename(columns={"index": "Feature"})

            # Pivot to get a Feature × Category DataFrame for stacking
            df_pivot = df_plot.pivot(
                index="Feature", columns="Category", values="SHAP"
            ).fillna(0)

            # Optional: reorder features by total SHAP sum
            order = df_pivot.sum(axis=1).sort_values(ascending=False).index
            df_pivot = df_pivot.loc[order]

            # Stacked bar plot using matplotlib
            bottom = np.zeros(len(df_pivot))
            x = np.arange(len(df_pivot))

            for cat in df_pivot.columns:
                ax.bar(
                    x,
                    df_pivot[cat],
                    bottom=bottom,
                    label=cat,
                    color=get_group_colors(dataset)[cat],
                )
                bottom += df_pivot[cat]

            ax.set_xticks(x)
            ax.set_xticklabels(df_pivot.index, rotation=90)
            ax.set_ylabel("mean([SHAP value])")
            ax.set_xlabel("Feature")

            fig.show()
            for file_type in ["pdf", "svg"]:
                fig.savefig(
                    os.path.join(
                        get_fig_folder(),
                        f"{dataset}_xgboost_{feature_set_name}_shapley_values_bar_vertical.{file_type}",
                    ),
                    transparent=True,
                )

        # share of shape importance
        shape_indices = [i for i, s in enumerate(features) if "shape" in s]
        share_shape_importance.append(
            {
                "dataset": dataset,
                "share_shape": mean_abs_shap_values[shape_indices].sum()
                / mean_abs_shap_values.sum(),
            }
        )

df_shape_share = pd.DataFrame(share_shape_importance)

# %%
dataset_map = {
    "inhibblock": "Blocked inhib.",
    "hommersom_binary": "CACNA1A",
    "wagenaar": "Wagenaar",
    "kapucu": "Rat vs hPSC",
    "mossink": "hPSC disease",
    "mossink_KS": "KS iPSC",
}
df_shape_share["dataset_label"] = df_shape_share["dataset"].map(dataset_map)

fig, ax = plt.subplots(figsize=(4 * cm, 6 * cm), constrained_layout=True)
sns.despine()
sns.barplot(
    data=df_shape_share,
    x="dataset_label",
    y="share_shape",
)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_ylim(0, 1)
ax.axhline(y=1, color="k", linestyle="--", linewidth=1)
ax.axhline(y=0.5, color="k", linestyle="--", linewidth=1)
ax.set_ylabel("Share of shape\nimportance")
fig.show()
for file_type in ["pdf", "svg"]:
    fig.savefig(
        os.path.join(
            get_fig_folder(),
            f"xgboost_shape_importance.{file_type}",
        ),
        transparent=True,
    )

# %% same but vertical
dataset_map = {
    "inhibblock": "Blocked inhib.",
    "hommersom_binary": "CACNA1A",
    "wagenaar": "Wagenaar",
    "kapucu": "Rat vs hPSC",
    "mossink": "hPSC disease",
    "mossink_KS": "KS iPSC",
}
df_shape_share["dataset_label"] = df_shape_share["dataset"].map(dataset_map)

datasets_order = [
    dataset_map[key]
    for key in [
        "inhibblock",
        "hommersom_binary",
        "mossink_KS",
        "wagenaar",
        "kapucu",
    ]
]

fig, ax = plt.subplots(figsize=(5.5 * cm, 3.5 * cm), constrained_layout=True)
ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")
sns.despine(bottom=True, top=False)
sns.barplot(
    data=df_shape_share,
    y="dataset_label",
    x="share_shape",
    order=datasets_order,
    # fill=False,
    color="grey",
    edgecolor="k",
    width=0.5,
    linewidth=1.5,
)
# ax.set_yticks(ax.get_xticks())
# ax.set_yticklabels(ax.get_yticklabels(), rotation=90)
ax.set_xlim(0, 1)
ax.axvline(x=1, color="k", linestyle="--", linewidth=1)
ax.axvline(x=0.5, color="k", linestyle="--", linewidth=1)
ax.set_xticks([0, 0.5, 1])
ax.set_xticklabels([0.0, 0.5, 1.0])
ax.set_xlabel("Share of shape\nimportance")
ax.set_ylabel("")
fig.show()
for file_type in ["pdf", "svg"]:
    fig.savefig(
        os.path.join(
            get_fig_folder(),
            f"xgboost_shape_importance_horizontal.{file_type}",
        ),
        transparent=True,
    )
