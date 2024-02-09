import pandas as pd

from src.persistence import load_clustering_labels, load_df_bursts, load_cv_params

# parameters which clustering to evaluate
burst_extraction_params = "burst_n_bins_50_extend_left_50_extend_right_50"
clustering_params = "spectral"
labels_params = "labels"
cv_params = "cv"

# load cross-validation parameters
cv_params = load_cv_params(burst_extraction_params, cv_params)
n_splits = cv_params["n_splits"]

# load data
df_bursts = load_df_bursts(burst_extraction_params, cv_params=cv_params)
clustering = load_clustering_labels(
    clustering_params, burst_extraction_params, labels_params, cv_params, i_split=None
)
for n_clusters_ in clustering.n_clusters:
    df_bursts[f"cluster_{n_clusters_}"] = clustering.labels_[n_clusters_]
for i in range(n_splits):
    idx_train = df_bursts.index[df_bursts[f"cv_{i}_train"]]
    clustering = load_clustering_labels(
        clustering_params, burst_extraction_params, labels_params, cv_params, i_split=i
    )
    for n_clusters_ in clustering.n_clusters:
        df_bursts[f"cluster_{n_clusters_}_cv_{i}"] = pd.Series(dtype=int)
        df_bursts.loc[idx_train, f"cluster_{n_clusters_}_cv_{i}"] = clustering.labels_[
            n_clusters_
        ]


# labels of n_clusters is saved in "cluster_{n_clusters}"
# labels of n_clusters in cross-validation split i is saved in "cluster_{n_clusters}_cv_{i}"
# index of training data in cross-validation split i is saved in "cv_{i}_train"
# index of training data can be accessed as df_bursts.index[df_bursts[f"cv_{i}_train"]]

# reference for sklearn methods
# https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
