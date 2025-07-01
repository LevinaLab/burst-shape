import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.preprocessing import LabelEncoder

from src.folders import get_fig_folder
from src.persistence import load_df_cultures
from src.persistence.knn_clustering import load_knn_clustering_results
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
]

n_classes = {}
df_accuracies = []
for burst_extraction_params in burst_extraction_params_list:
    dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
    spectral_clustering_params = get_chosen_spectral_embedding_params(dataset)

    for feature_set_name in ["combined", "shape", "traditional"]:
        (
            features,
            nested_scores,
            all_shap_values,
            all_y_pred,
            all_y_test,
        ) = load_xgboost_results(
            burst_extraction_params, spectral_clustering_params, feature_set_name
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
            shap.summary_plot(
                mean_shap_values,
                # features=X,
                plot_type="bar",
                feature_names=features,
                show=False,
            )
            fig = plt.gcf()
            fig.show()
            fig.savefig(
                os.path.join(
                    get_fig_folder(),
                    f"{dataset}_xgboost_{feature_set_name}_shapley_values_bar.svg",
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

            shap.summary_plot(
                mean_shap_values,
                # features=X,
                plot_type="bar",
                feature_names=features,
                class_inds="original",
                class_names=list(label_encoder.classes_),
                color=lambda i: [
                    get_group_colors(dataset)[j] for j in list(label_encoder.classes_)
                ][i],
                show=False,
            )
            fig = plt.gcf()
            fig.show()
            fig.savefig(
                os.path.join(
                    get_fig_folder(),
                    f"{dataset}_xgboost_{feature_set_name}_shapley_values.svg",
                ),
                transparent=True,
            )
