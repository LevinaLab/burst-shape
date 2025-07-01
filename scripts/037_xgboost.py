"""
Train XGBoost on burst shape and traditional feature sets.

For each of the feature sets (only shape, only traditional, combined) fit XGBoost.
XGBoost hyperparameters are selected on a 5 fold cross-validation.
Evaluation (test) is done with a 5 fold cross-validation.
Class imbalance is handled in fit function.
Shapley values are evaluated on the full dataset (yes, this is confusing but correct).

Plots:
- average embedding locations for illustration
- confusion matrix
- Shapley values (feature importance)
"""
import os

import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib import pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, make_scorer
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
from xgboost import XGBClassifier

from src.folders import get_fig_folder
from src.persistence import load_df_bursts, load_df_cultures, load_spectral_embedding
from src.persistence.xgboost import (
    exist_xgboost_results,
    load_xgboost_results,
    save_xgboost_results,
)
from src.plot import get_group_colors, prepare_plotting
from src.prediction.define_target import make_target_label
from src.prediction.knn_clustering import get_recording_mask
from src.settings import (
    get_chosen_spectral_embedding_params,
    get_dataset_from_burst_extraction_params,
)
from src.utils.classical_features import get_classical_features

cm = prepare_plotting()
special_target = False  # changes target in mossink from disease label to subject label

burst_extraction_params = (
    "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_test_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_hommersom_binary_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
)
dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
df_cultures = load_df_cultures(burst_extraction_params)
df_cultures = df_cultures[df_cultures["n_bursts"] > 0]
df_bursts = load_df_bursts(burst_extraction_params, cv_params=None)

df_cultures, target_label = make_target_label(
    dataset, df_cultures, None, special_target=special_target
)

print(f"Dataset:\t\t{dataset}\nTarget label:\t{target_label}")

# %% get embedding coordinates
spectral_clustering_params = get_chosen_spectral_embedding_params(dataset)
spectral_embedding = load_spectral_embedding(
    burst_extraction_params, spectral_clustering_params
)
n_spectral_dims = spectral_embedding.shape[1]
for i_dim in range(n_spectral_dims):
    name_dim = f"shape_{i_dim + 1}"
    df_bursts[name_dim] = spectral_embedding[:, i_dim]
    df_cultures[name_dim] = pd.Series(dtype=float)
    for index in df_cultures.index:
        mask_recording = get_recording_mask(df_bursts, index)
        df_cultures.at[index, name_dim] = (
            df_bursts[name_dim].values[mask_recording].mean()
        )

fig, ax = plt.subplots(constrained_layout=True)
ax.set_title("Spectral Embedding (Shape features) for each recording")
sns.despine()
sns.scatterplot(
    data=df_cultures.reset_index(),
    x="shape_1",
    y="shape_2",
    hue="target_label",
    palette=get_group_colors(dataset),
)
ax.legend(
    frameon=False,
    title="Recordings\naverage location\nin embedding",
    bbox_to_anchor=(1.1, 0.8),
    loc="upper left",
)
fig.show()
fig.savefig(
    os.path.join(
        get_fig_folder(),
        f"{dataset}_xgboost_recording_embedding_shape.svg",
    ),
    transparent=True,
)
# %% get classical features
df_cultures, classical_features = get_classical_features(
    df_cultures, df_bursts, dataset
)

# %% define feature sets
shape_features = [f"shape_{i_dim + 1}" for i_dim in range(n_spectral_dims)]
# classical features
all_features = shape_features + classical_features

feature_dict = {
    "combined": all_features,
    "shape": shape_features,
    "traditional": classical_features,
}
# %% prediction with xgboost
for feature_set_name, features in feature_dict.items():
    # Encode target labels
    label_encoder = LabelEncoder()
    df_cultures["encoded_label"] = label_encoder.fit_transform(
        df_cultures["target_label"]
    )
    num_classes = len(label_encoder.classes_)

    X = df_cultures[features].astype(float)  # .values
    y = df_cultures["encoded_label"]  # .values

    if not exist_xgboost_results(
        burst_extraction_params, spectral_clustering_params, feature_set_name
    ):
        print(
            f"XBoost results for feature set {feature_set_name} don't exists. Computing them."
        )

        objective = "multi:softprob" if num_classes > 2 else "binary:logistic"
        eval_metric = "mlogloss" if num_classes > 2 else "logloss"

        random_state = 1234567890
        n_splits = 100
        outer_cv = StratifiedShuffleSplit(
            n_splits=n_splits, test_size=0.2, random_state=random_state
        )
        # outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        # XGBoost hyperparameters grid
        param_grid_xgb = {
            "n_estimators": [100, 150, 200, 250, 300],
            "learning_rate": [0.01, 0.04, 0.1, 0.2],
            "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "subsample": [0.6, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.1, 1],
            "reg_lambda": [0.5, 1, 2],
        }

        xgb = XGBClassifier(
            objective=objective,
            eval_metric=eval_metric,
            random_state=random_state,
            # do not weight classes here, weighting is done in fit
        )
        scorer = make_scorer(balanced_accuracy_score)

        grid_search = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid_xgb,
            scoring=scorer,
            cv=inner_cv,
            n_jobs=-1,
        )
        random_search = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=param_grid_xgb,
            n_iter=100,
            scoring=scorer,
            cv=inner_cv,
            n_jobs=-1,
        )

        nested_scores = []
        all_y_test = []
        all_y_pred = []
        all_shap_values = []
        all_best_models = []

        for train_idx, test_idx in tqdm(
            outer_cv.split(X, y), total=n_splits, desc="Outer loop of cv"
        ):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # grid_search.fit(X_train, y_train)
            # best_model = grid_search.best_estimator_

            random_search.fit(
                X_train,
                y_train,
                sample_weight=compute_sample_weight(class_weight="balanced", y=y_train),
            )
            best_model = random_search.best_estimator_

            y_pred = best_model.predict(X_test)

            all_best_models.append(best_model)
            all_y_test.extend(y_test)
            all_y_pred.extend(y_pred)

            score = balanced_accuracy_score(y_test, y_pred)
            nested_scores.append(score)

            # Compute SHAP values for the entire dataset using the best model
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X)
            all_shap_values.append(shap_values)

        # save results
        features = np.array(features)
        nested_scores = np.array(nested_scores)
        all_shap_values = np.array(all_shap_values)
        all_y_pred = np.array(all_y_pred)
        all_y_test = np.array(all_y_test)
        save_xgboost_results(
            burst_extraction_params,
            spectral_clustering_params,
            feature_set_name,
            features,
            nested_scores,
            all_shap_values,
            all_y_pred,
            all_y_test,
        )
        print(f"Saved the results for feature set {feature_set_name}.")
    else:
        print(
            f"XGBoost results for feature set {feature_set_name} already exists. Loading..."
        )
        (
            features,
            nested_scores,
            all_shap_values,
            all_y_pred,
            all_y_test,
        ) = load_xgboost_results(
            burst_extraction_params, spectral_clustering_params, feature_set_name
        )
        print(f"Loaded the results for feature set {feature_set_name}.")

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
            features=X,
            plot_type="bar",
            feature_names=X.columns,
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
        shap.summary_plot(
            mean_shap_values, features=X, feature_names=X.columns, show=False
        )
        fig = plt.gcf()
        fig.show()
        fig.savefig(
            os.path.join(
                get_fig_folder(),
                f"{dataset}_xgboost_{feature_set_name}_shapley_values_bee.svg",
            ),
            transparent=True,
        )
    else:
        shap.summary_plot(
            mean_shap_values,
            features=X,
            plot_type="bar",
            feature_names=X.columns,
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

    # %%
    match dataset:
        case "inhibblock" | "kapucu" | "wagenaar":
            figsize = (7 * cm, 6 * cm)
        case "mossink":
            if special_target is True:
                figsize = (10 * cm, 10 * cm)
            else:
                figsize = (7 * cm, 6 * cm)
        case _:
            figsize = (7 * cm, 6 * cm)

    fig, ax = plt.subplots(constrained_layout=True, figsize=figsize)
    sns.heatmap(
        confusion_matrix(all_y_pred, all_y_test, normalize="true"),
        vmin=0,
        vmax=1,
        annot=False,
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        cbar_kws={"label": "Accuracy"},
    )
    if get_group_colors(dataset) is not None:
        for label_x, label_y, color in zip(
            ax.get_xticklabels(),
            ax.get_yticklabels(),
            [get_group_colors(dataset)[j] for j in list(label_encoder.classes_)],
        ):
            label_x.set_color(color)
            label_y.set_color(color)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.show()
    fig.savefig(
        os.path.join(
            get_fig_folder(),
            f"{dataset}_xgboost_{feature_set_name}_confusion_matrix.svg",
        ),
        transparent=True,
    )

    # %% feature importance for shape vs traditional
    if feature_set_name == "combined":
        print("\nFeature importance")
        feature_importance = mean_abs_shap_values
        feature_importance = feature_importance / feature_importance.sum()
        print(f"Shape:\n{feature_importance[:2].sum():.3f}")
        print(f"Traditional:\n{feature_importance[2:].sum():.3f}")
