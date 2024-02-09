import os
import numpy as np
import time
import pickle
import json

import pandas as pd

from src.spectral_clustering import SpectralClusteringModified
from src.persistence import (
    get_burst_folder,
    get_spectral_clustering_folder,
    get_labels_file,
    get_labels_params_file,
    cv_params_to_string,
)

# choose parameters for clustering
n_jobs = 12
burst_extraction_params = "burst_n_bins_50_extend_left_50_extend_right_50"
cv_params = "cv"  # set to None if not using cross-validation or to specific cv_params
all_data = False  # set to False if you want to compute only cross-validation
clustering_params = {
    "n_components_max": 30,
    "affinity": "nearest_neighbors",
    "n_neighbors": 10,
    "random_state": 0,
}

if cv_params is not None:
    if isinstance(cv_params, dict):
        cv_string = cv_params_to_string(cv_params)
    else:
        cv_string = cv_params
    with open(
        os.path.join(
            get_burst_folder(burst_extraction_params), f"{cv_string}_params.json"
        ),
        "r",
    ) as f:
        cv_params = json.load(f)
        n_splits = cv_params["n_splits"]
else:
    n_splits = 0
tasks = []
if all_data:
    tasks.append(None)
if n_splits > 0:
    tasks.extend(list(range(n_splits)))

# create folder
save_folder = get_spectral_clustering_folder(clustering_params, burst_extraction_params)
os.makedirs(save_folder, exist_ok=True)
# save params as json
with open(os.path.join(save_folder, "clustering_params.json"), "w") as f:
    json.dump(clustering_params, f, indent=4)

# %% compute eigenvectors
df_bursts = pd.read_pickle(
    os.path.join(
        get_burst_folder(burst_extraction_params),
        f"002_wagenaar_bursts_df_{cv_string}.pkl",
    )
)
bursts = np.load(
    os.path.join(
        get_burst_folder(burst_extraction_params),
        "002_wagenaar_bursts_mat.npy",
    )
)  # n_burst x time
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
    maps_file = os.path.join(
        save_folder,
        f"004_clustering_maps.pkl"
        if i_split is None
        else f"004_clustering_maps_{cv_string}_{i_split}.pkl",
    )
    with open(maps_file, "wb") as f:
        pickle.dump(clustering, f)

# %% compute labels
# choose parameters for labels
label_params = {
    "n_clusters_min": 2,
    "n_clusters_max": 30,
    "assign_labels": "cluster_qr",
    "random_state": 0,
}

# save params as json
with open(os.path.join(save_folder, get_labels_params_file(label_params)), "w") as f:
    json.dump(label_params, f, indent=4)

for i_split in tasks:
    if i_split is None:
        print("Compute labels for all data")
    else:
        print(f"Compute labels for split {i_split}")
    maps_file = os.path.join(
        save_folder,
        f"004_clustering_maps.pkl"
        if i_split is None
        else f"004_clustering_maps_{cv_string}_{i_split}.pkl",
    )
    with open(maps_file, "rb") as f:
        clustering = pickle.load(f)
    clustering.n_jobs = n_jobs
    clustering.n_clusters = np.arange(
        label_params["n_clusters_min"], label_params["n_clusters_max"] + 1
    )
    clustering.assign_labels = label_params["assign_labels"]
    clustering.random_state = label_params["random_state"]
    clustering.verbose = False
    clustering = clustering.compute_labels()
    with open(
        get_labels_file(
            label_params, clustering_params, burst_extraction_params, i_split=i_split
        ),
        "wb",
    ) as f:
        pickle.dump(clustering, f)
