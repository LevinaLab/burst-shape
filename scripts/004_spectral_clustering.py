import os
import numpy as np
import time
import pickle

from tqdm import tqdm

from src.folders import get_results_folder
from src.spectral_clustering import SpectralClusteringModified

n_clusters_list = np.arange(2, 31, 1)
save_folder = os.path.join(get_results_folder(), "004_spectral_clustering")
os.makedirs(save_folder, exist_ok=True)

# %% compute eigenvectors
bursts = np.load(
    os.path.join(get_results_folder(), "002_wagenaar_bursts_mat.npy")
)  # n_burst x time

print("Compute eigenvectors")
start_time = time.time()
clustering = SpectralClusteringModified(
    n_clusters=n_clusters_list[0],
    n_components_max=n_clusters_list[-1],
    affinity="nearest_neighbors",
    n_neighbors=10,
    assign_labels="cluster_qr",
    n_jobs=12,
    random_state=0,
    verbose=True,
).compute_maps(bursts)
end_time = time.time()
print("Elapsed time: {}".format(end_time - start_time))

with open(os.path.join(save_folder, "004_clustering_maps.pkl"), "wb") as f:
    pickle.dump(clustering, f)

# %% compute labels
with open(os.path.join(save_folder, "004_clustering_maps.pkl"), "rb") as f:
    clustering = pickle.load(f)
clustering.verbose = False
for n_clusters in tqdm(n_clusters_list, desc="Assign labels"):
    clustering.n_clusters = n_clusters
    clustering = clustering.compute_labels()

    with open(
        os.path.join(save_folder, f"004_clustering_labels_{n_clusters}.pkl"),
        "wb",
    ) as f:
        pickle.dump(clustering, f)

# %% compute labels all at once
print("Compute labels all at once...")
with open(os.path.join(save_folder, "004_clustering_maps.pkl"), "rb") as f:
    clustering = pickle.load(f)
clustering.verbose = False
clustering.n_clusters = n_clusters_list
clustering = clustering.compute_labels()

with open(
    os.path.join(save_folder, f"004_clustering_labels.pkl"),
    "wb",
) as f:
    pickle.dump(clustering, f)
print("Done.")
