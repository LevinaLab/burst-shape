import os
import numpy as np
import time
import pickle
import json

from src.spectral_clustering import SpectralClusteringModified
from src.persistence import (
    get_burst_folder,
    get_spectral_clustering_folder,
    get_labels_file,
    get_labels_params_file,
)

# choose parameters for clustering
burst_extraction_params = "burst_n_bins_50_extend_left_50_extend_right_50"
clustering_params = {
    "n_components_max": 30,
    "affinity": "nearest_neighbors",
    "n_neighbors": 10,
    "random_state": 0,
}

# create folder
save_folder = get_spectral_clustering_folder(clustering_params, burst_extraction_params)
os.makedirs(save_folder, exist_ok=True)
# save params as json
with open(os.path.join(save_folder, "clustering_params.json"), "w") as f:
    json.dump(clustering_params, f, indent=4)

# %% compute eigenvectors
bursts = np.load(
    os.path.join(
        get_burst_folder(burst_extraction_params),
        "002_wagenaar_bursts_mat.npy",
    )
)  # n_burst x time

print("Compute eigenvectors")
start_time = time.time()
clustering = SpectralClusteringModified(
    n_jobs=12,
    verbose=True,
    **clustering_params,
).compute_maps(bursts)
end_time = time.time()
print("Elapsed time: {}".format(end_time - start_time))

with open(os.path.join(save_folder, "004_clustering_maps.pkl"), "wb") as f:
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

print("Compute labels...")
with open(os.path.join(save_folder, "004_clustering_maps.pkl"), "rb") as f:
    clustering = pickle.load(f)
clustering.n_clusters = np.arange(
    label_params["n_clusters_min"], label_params["n_clusters_max"] + 1
)
clustering.assign_labels = label_params["assign_labels"]
clustering.random_state = label_params["random_state"]
clustering.verbose = False
clustering = clustering.compute_labels()

with open(
    get_labels_file(label_params, clustering_params, burst_extraction_params),
    "wb",
) as f:
    pickle.dump(clustering, f)
print("Done.")
