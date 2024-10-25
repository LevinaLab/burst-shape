import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform

from src import folders
from src.persistence import save_df_bursts, save_burst_matrix
from src.persistence.burst_extraction import _get_burst_folder, load_burst_matrix, load_df_bursts

# %% load settings
burst_extraction_params = (
    "burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
)
n_bursts = None  # if None uses all bursts
linkage_method = "complete"
np.random.seed(0)

# plot settings
n_clusters = 9  # 3  # if None chooses the number of clusters with Davies-Bouldin index

# plotting
cm = 1 / 2.54  # centimeters in inches
fig_path = folders.get_fig_folder()

folder_agglomerating_clustering = os.path.join(
    _get_burst_folder(burst_extraction_params),
    f"agglomerating_clustering_linkage_{linkage_method}_n_bursts_{n_bursts}",
)
file_distance_matrix = os.path.join(
    folder_agglomerating_clustering, "distance_matrix.npy"
)
file_linkage = os.path.join(folder_agglomerating_clustering, "linkage.npy")

# %% load bursts
burst_matrix = load_burst_matrix(burst_extraction_params)
df_bursts = load_df_bursts(burst_extraction_params)

print(f"Loading linkage from {file_linkage}")
linkage = np.load(file_linkage)
print(f"Loading distance matrix from {file_distance_matrix}")
distance_matrix = np.load(file_distance_matrix)  # vector-form

# %% identify outlier
labels = fcluster(linkage, t=n_clusters, criterion="maxclust")
# identify the cluster with only one burst
unique, index, counts = np.unique(labels, return_index=True, return_counts=True)
print(f"Number of clusters of size 1: {np.sum(counts == 1)}")
print(f"Indices from these clusters: {index[counts == 1]}")

index_to_remove = index[counts == 1][0]

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(range(50), burst_matrix[index_to_remove], label="from burst_matrix")
ax.plot(range(50), df_bursts.iloc[index_to_remove]["burst"], linestyle="--", label="from df_bursts")
ax.legend(title="outlier")
fig.show()

# %% remove outlier from all data
# df_bursts
df_bursts_new = df_bursts.drop(df_bursts.iloc[index_to_remove].name)
# burst_matrix
burst_matrix_new = np.delete(burst_matrix, index_to_remove, axis=0)
# distance_matrix
distance_matrix = squareform(distance_matrix, force="tomatrix")
distance_matrix_new = np.delete(
    np.delete(distance_matrix, index_to_remove, axis=0),
    index_to_remove,
    axis=1,
)
distance_matrix_new = squareform(distance_matrix_new, force="tovector")

# %% create new folder for new data
burst_extraction_params_new = burst_extraction_params + "_outlier_removed"
os.makedirs(_get_burst_folder(burst_extraction_params_new), exist_ok=True)
folder_agglomerating_clustering_new = os.path.join(
    _get_burst_folder(burst_extraction_params_new),
    f"agglomerating_clustering_linkage_{linkage_method}_n_bursts_{n_bursts}",
)
os.makedirs(folder_agglomerating_clustering_new, exist_ok=True)
# %% save new data
save_df_bursts(df_bursts_new, burst_extraction_params_new)
save_burst_matrix(burst_matrix_new, burst_extraction_params_new)
np.save(
    os.path.join(
        folder_agglomerating_clustering_new,
        "distance_matrix.npy",
    ),
    distance_matrix_new,
)
