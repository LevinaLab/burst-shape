from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.persistence import load_df_bursts, load_df_cultures, load_spectral_embedding
from src.plot import get_group_colors, prepare_plotting
from src.prediction.define_target import make_target_label
from src.prediction.knn_clustering import get_recording_mask
from src.settings import (
    get_chosen_spectral_embedding_params,
    get_dataset_from_burst_extraction_params,
)
from src.utils.classical_features import get_classical_features

cm = prepare_plotting()
special_target = True  # changes target in mossink from disease label to subject label

burst_extraction_params = (
    "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_test_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30"
)
dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
df_cultures = load_df_cultures(burst_extraction_params)
df_cultures = df_cultures[df_cultures["n_bursts"] > 0]
df_bursts = load_df_bursts(burst_extraction_params, cv_params=None)

df_cultures, df_bursts, target_label = make_target_label(
    dataset, df_cultures, df_bursts, special_target=special_target
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
# %% get classical features
df_cultures, classical_features = get_classical_features(
    df_cultures, df_bursts, dataset
)

# %% prediction with xgboost

shape_features = [f"shape_{i_dim + 1}" for i_dim in range(n_spectral_dims)]
# classical features
all_features = shape_features + classical_features

# Encode target labels
label_encoder = LabelEncoder()
df_cultures["encoded_label"] = label_encoder.fit_transform(df_cultures["target_label"])

num_classes = len(label_encoder.classes_)
objective = "multi:softprob" if num_classes > 2 else "binary:logistic"
eval_metric = "mlogloss" if num_classes > 2 else "logloss"

# Handle class balancing
sample_weight = None
xgb_params = {
    "n_estimators": 100,
    # 'max_depth': 4,
    # 'learning_rate': 0.1,
    "objective": objective,
    # 'use_label_encoder': False,
    "eval_metric": eval_metric,
}

for features in [shape_features, classical_features, all_features]:
    X = df_cultures[features].astype(float)  # .values
    y = df_cultures["encoded_label"]  # .values
    kf = StratifiedKFold(n_splits=5, shuffle=True)  # , random_state=42)
    if num_classes == 2:
        shap_values_all = np.zeros_like(X, dtype=float)
    else:
        shap_values_all = np.zeros((*X.shape, num_classes), dtype=float)
    preds_all = np.zeros(len(y))
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, # random_state=42
    # )
    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if num_classes == 2:
            # Binary: use scale_pos_weight
            class_counts = Counter(y_train)
            neg, pos = class_counts[0], class_counts[1]
            xgb_params["scale_pos_weight"] = neg / pos
        else:
            # Multiclass: use sample_weight
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

        bst = XGBClassifier(**xgb_params)
        bst.fit(X_train, y_train, sample_weight=sample_weight)

        preds_all[test_idx] = bst.predict(X_test)

        explainer = shap.TreeExplainer(bst)
        shap_values = explainer(X_test)
        shap_values_all[test_idx, :] = shap_values.values

    acc = accuracy_score(y, preds_all)
    bal_acc = balanced_accuracy_score(y, preds_all)

    print(f"Features: {features}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}\n")

    match dataset:
        case "inhibblock" | "kapucu" | "wagenaar":
            figsize = (6 * cm, 6 * cm)
        case "mossink":
            if special_target is True:
                figsize = (10 * cm, 10 * cm)
            else:
                figsize = (6 * cm, 6 * cm)
        case _:
            raise NotImplementedError

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # Compute confusion matrix
    class_labels = range(len(label_encoder.classes_))
    # class_labels = list(label_encoder.classes_)
    matrix_confusion = confusion_matrix(
        y, preds_all, labels=class_labels, normalize="true"
    )
    sns.heatmap(
        matrix_confusion,
        annot=False,
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.show()

    if isinstance(shap_values_all, np.ndarray) and shap_values_all.ndim == 2:
        # Binary classification: shape (n_samples, n_features)
        print("Plotting SHAP values for binary classification")
        shap.summary_plot(shap_values_all, X, show=True)

    else:  # multi-class
        # Choose class to visualize (e.g. class 0 or majority class)
        dominant_class = np.bincount(y).argmax()
        print(
            f"Plotting SHAP values for class: {label_encoder.classes_[dominant_class]}"
        )

        # Plot SHAP values for that class
        shap.summary_plot(shap_values_all[:, :, dominant_class], X, show=True)

        # for i in range(shap_values_all.shape[2]):
        #     print(f"Class {i}: {label_encoder.classes_[i]}")
        #     shap.summary_plot(shap_values_all[:, :, i], X_test, show=True)

        # Plot mean # TODO: does this even make sense? Probably, we have to find a good way to average avoiding extinction.
        shap.summary_plot(np.abs(shap_values_all).mean(axis=2), X, show=True)
