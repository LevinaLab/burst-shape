import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from src.persistence import load_df_bursts, load_df_cultures
from src.plot import prepare_plotting
from src.prediction.define_target import make_target_label
from src.settings import get_dataset_from_burst_extraction_params
from src.utils.classical_features import get_classical_features

cm = prepare_plotting()
special_target = True

burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30"
)
dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
df_cultures = load_df_cultures(burst_extraction_params)
df_bursts = load_df_bursts(burst_extraction_params, cv_params=None)

df_cultures, df_bursts, target_label = make_target_label(
    dataset, df_cultures, df_bursts, special_target=special_target
)

print(f"Dataset:\t\t{dataset}\nTarget label:\t{target_label}")

# %%
df_cultures = df_cultures[df_cultures["n_bursts"] > 0]
df_cultures, classical_features = get_classical_features(
    df_cultures, df_bursts, dataset
)

# %%
split = ["direct", "leave-one-out"][1]
model = ["RandomForestClassifier", "KNeighborsClassifier"][0]
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

X = df_cultures[classical_features].values
y = df_cultures["target_label"].values
class_labels = np.sort(np.unique(y))
# TODO max_depth
match model:
    case "RandomForestClassifier":
        clf = RandomForestClassifier(
            max_depth=4,
            class_weight="balanced",
            random_state=0,
        )
    case "KNeighborsClassifier":
        clf = KNeighborsClassifier(
            n_neighbors=5,
            # class_weight="balanced",
            # random_state=0,
        )
if split == "direct":
    clf.fit(X, y)
    y_pred = clf.predict(X)
elif split == "leave-one-out":
    y_pred = np.zeros_like(y)
    n_samples = len(y)
    for i in tqdm(range(n_samples)):
        mask = np.arange(n_samples) != i
        y_pred[i] = clf.fit(X[mask], y[mask]).predict(X[i].reshape(1, -1))[0]
else:
    raise NotImplementedError

matrix_confusion = confusion_matrix(y, y_pred, labels=class_labels)

fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
matrix_confusion = (
    matrix_confusion.astype("float") / matrix_confusion.sum(axis=1)[:, np.newaxis]
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

print(f"Accuracy:\t{np.mean(y_pred == y)}")
print(f"Balanced accuracy:\t{matrix_confusion.diagonal().mean():.2f}")
