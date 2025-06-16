import time

import numpy as np

from src.persistence import (
    load_affinity_matrix,
    load_burst_matrix,
    load_clustering_maps,
    load_cv_params,
    load_df_bursts,
    save_affinity_matrix,
    save_clustering_labels,
    save_clustering_maps,
    save_clustering_params,
    save_labels_params,
)
from src.spectral_clustering import SpectralClusteringModified, compute_affinity_matrix

# choose parameters for clustering
n_jobs = 12
burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_n_bins_50_normalization_integral"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_test_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_maxISIstart_50_maxISIb_50_minBdur_100_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30"
    "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
)
cv_params = (
    None  # "cv"  # set to None if not using cross-validation or to specific cv_params
)
all_data = True  # set to False if you want to compute only cross-validation
clustering_params = {
    "n_components_max": 30,
    "affinity": "precomputed",
    "metric": "wasserstein",
    "n_neighbors": 55,  # 85,  # 6,  # 60,  # 150,
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

# %% compute affinity matrix and eigenvectors
for i_split in tasks:
    if i_split is None:
        print("Compute spectral clustering for all data")
    else:
        print(f"Compute spectral clustering for split {i_split}")
    try:
        clustering = load_clustering_maps(
            clustering_params,
            burst_extraction_params,
            params_cross_validation=cv_params,
            i_split=i_split,
        )
        print("Eigenvectors already computed. Loading them.")
    except FileNotFoundError:
        print("Eigenvectors not found, computing them...")
        start_time = time.time()

        # load data
        if clustering_params["affinity"] == "precomputed":
            print("Requires precomputed affinity matrix, trying to load it...")
            try:
                X_split = load_affinity_matrix(
                    clustering_params,
                    burst_extraction_params,
                    params_cross_validation=cv_params,
                    i_split=i_split,
                )
                print("Precomputed affinity matrix found. Loading it.")
            except FileNotFoundError:
                print("Precomputed affinity matrix not found, computing it...")
                burst_matrix = load_burst_matrix(burst_extraction_params)
                if i_split is not None:
                    burst_matrix = burst_matrix[
                        load_df_bursts(burst_extraction_params, cv_params=cv_params)[
                            f"cv_{i_split}_train"
                        ]
                    ]
                X_split = compute_affinity_matrix(
                    burst_matrix,
                    n_jobs=n_jobs,
                    metric=clustering_params["metric"],
                    n_neighbors=clustering_params["n_neighbors"],
                )
                save_affinity_matrix(
                    X_split,
                    clustering_params,
                    burst_extraction_params,
                    params_cross_validation=cv_params,
                    i_split=i_split,
                )
                print("Finished computing affinity matrix.")
            print("Now computing eigenvectors...")
        else:
            bursts = load_burst_matrix(burst_extraction_params)
            df_bursts = load_df_bursts(burst_extraction_params, cv_params=cv_params)
            X_split = bursts[df_bursts[f"cv_{i_split}_train"]]

        # compute eigenvectors
        clustering_params_copy = clustering_params.copy()
        clustering_params_copy.pop("metric")
        clustering = SpectralClusteringModified(
            n_jobs=n_jobs,
            verbose=True,
            **clustering_params_copy,
        ).compute_maps(X_split)
        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time}")

        # save
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
