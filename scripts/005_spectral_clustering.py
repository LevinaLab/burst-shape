import numpy as np
import time

from src.spectral_clustering import SpectralClusteringModified
from src.persistence import (
    load_burst_matrix,
    load_df_bursts,
    save_clustering_params,
    save_clustering_maps,
    load_clustering_maps,
    save_labels_params,
    save_clustering_labels,
    load_cv_params,
)

# choose parameters for clustering
n_jobs = 12
burst_extraction_params = "burst_n_bins_50_extend_left_50_extend_right_50"
cv_params = "cv"  # set to None if not using cross-validation or to specific cv_params
all_data = True  # set to False if you want to compute only cross-validation
clustering_params = {
    "n_components_max": 30,
    "affinity": "nearest_neighbors",
    "n_neighbors": 10,
    "random_state": 0,
}

# load cross-validation parameters
if cv_params is not None:
    cv_params = load_cv_params(burst_extraction_params, cv_params)
    n_splits = cv_params["n_splits"]
else:
    n_splits = 0

# define which clustering tasks to perform
tasks = []
if all_data:
    tasks.append(None)
if n_splits > 0:
    tasks.extend(list(range(n_splits)))

# save params as json
save_clustering_params(clustering_params, burst_extraction_params)

# %% compute eigenvectors
df_bursts = load_df_bursts(burst_extraction_params, cv_params=cv_params)
bursts = load_burst_matrix(burst_extraction_params)
for i_split in tasks:
    if i_split is None:
        print("Compute eigenvectors for all data")
    else:
        print(f"Compute eigenvectors for split {i_split}")
    start_time = time.time()
    bursts_split = (
        bursts if i_split is None else bursts[df_bursts[f"cv_{i_split}_train"]]
    )
    clustering = SpectralClusteringModified(
        n_jobs=n_jobs,
        verbose=True,
        **clustering_params,
    ).compute_maps(bursts_split)
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")
    save_clustering_maps(
        clustering,
        clustering_params,
        burst_extraction_params,
        params_cross_validation=cv_params,
        i_split=i_split,
    )

# %% compute labels
# choose parameters for labels
label_params = {
    "n_clusters_min": 2,
    "n_clusters_max": 30,
    "assign_labels": "cluster_qr",
    "random_state": 0,
}

# save params as json
save_labels_params(label_params, clustering_params, burst_extraction_params)

for i_split in tasks:
    if i_split is None:
        print("Compute labels for all data")
    else:
        print(f"Compute labels for split {i_split}")
    clustering = load_clustering_maps(
        clustering_params,
        burst_extraction_params,
        params_cross_validation=cv_params,
        i_split=i_split,
    )
    clustering.n_jobs = n_jobs
    clustering.n_clusters = np.arange(
        label_params["n_clusters_min"], label_params["n_clusters_max"] + 1
    )
    clustering.assign_labels = label_params["assign_labels"]
    clustering.random_state = label_params["random_state"]
    clustering.verbose = False
    clustering = clustering.compute_labels()
    save_clustering_labels(
        clustering,
        clustering_params,
        burst_extraction_params,
        label_params,
        params_cross_validation=cv_params,
        i_split=i_split,
    )
