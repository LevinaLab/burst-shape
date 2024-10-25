import multiprocessing
import os
from time import time

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from src.persistence.burst_extraction import (
    _get_burst_folder,
    load_burst_matrix,
    load_df_bursts,
)

burst_extraction_params = (
    "burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4_outlier_removed"
)
n_bursts = None  # if None uses all bursts
compute_parallel = True  # if True uses double the memory but is faster
recompute_distance_matrix = False  # if False and available loads the data from disk
recompute_linkage = False  # if False and available loads the data from disk
linkage_method = "complete"
np.random.seed(0)

# %% define paths
folder_agglomerating_clustering = os.path.join(
    _get_burst_folder(burst_extraction_params),
    f"agglomerating_clustering_linkage_{linkage_method}_n_bursts_{n_bursts}",
)
file_distance_matrix = os.path.join(
    folder_agglomerating_clustering, "distance_matrix.npy"
)
file_linkage = os.path.join(folder_agglomerating_clustering, "linkage.npy")

if (
    not recompute_distance_matrix
    and not recompute_linkage
    and os.path.exists(file_distance_matrix)
    and os.path.exists(file_linkage)
):
    print(
        f"Linkage and distance matrix already computed:"
        f"\nFile distance matrix: {file_distance_matrix}"
        f"\nFile linkage: {file_linkage}"
    )
    exit()

# %% load bursts
burst_matrix = load_burst_matrix(burst_extraction_params)
df_bursts = load_df_bursts(burst_extraction_params)
# select randomly bursts to reduce computation time
np.random.seed(0)
if n_bursts is not None:
    idx = np.random.choice(burst_matrix.shape[0], n_bursts, replace=False)
    np.save(os.path.join(folder_agglomerating_clustering, "idx.npy"), idx)
    burst_matrix = burst_matrix[idx]
    df_bursts = df_bursts.iloc[idx]
print(burst_matrix.shape)

# %%


def _wasserstein_distance(a, b):
    a_cumsum = np.cumsum(a)
    b_cumsum = np.cumsum(b)
    a_cumsum /= a_cumsum[-1]
    b_cumsum /= b_cumsum[-1]
    return np.sum(np.abs(a_cumsum - b_cumsum))


# %% compute distance matrix
if not recompute_distance_matrix and os.path.exists(file_distance_matrix):
    print(f"Loading distance matrix from disk: {file_distance_matrix}")
    distance_matrix = np.load(file_distance_matrix)  # vector-form
else:
    print("Computing distance matrix and linkage")
    t0 = time()
    if compute_parallel:
        n = burst_matrix.shape[0]
        size = n * (n - 1) // 2
        # initialize shared memory array
        distance_matrix = multiprocessing.Array("d", n * (n - 1) // 2, lock=False)
        # convert the burst matrix to shared memory with lock=False
        burst_matrix_parallel = np.frombuffer(
            multiprocessing.Array("d", burst_matrix.size, lock=False)
        ).reshape(burst_matrix.shape)
        burst_matrix_parallel[:] = burst_matrix
        # create a shared lookup table for the index of the distance matrix
        lookup_table = np.zeros((size, 2), dtype=int)
        k = 0
        for i in tqdm(range(n - 1), desc="Computing lookup table"):
            lookup_table[k : k + n - i - 1, 0] = i
            lookup_table[k : k + n - i - 1, 1] = np.arange(i + 1, n)
            k += n - i - 1

        def _metric_from_index(k):
            i, j = lookup_table[k]
            distance_matrix[k] = _wasserstein_distance(
                burst_matrix_parallel[i], burst_matrix_parallel[j]
            )

        # parallel computation of the distance matrix
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            pool.map(
                _metric_from_index,
                range(size),
            )
    else:
        distance_matrix = pdist(burst_matrix, metric=_wasserstein_distance)
    t1 = time()
    print(f"Distance matrix: {t1 - t0:.2f} s")
    print(f"Saving distance matrix to disk: {file_distance_matrix}")
    os.makedirs(folder_agglomerating_clustering, exist_ok=True)
    np.save(file_distance_matrix, distance_matrix)

# %% compute linkage
if not recompute_linkage and os.path.exists(file_linkage):
    print(f"Loading linkage from disk: {file_distance_matrix}")
    Z = np.load(file_linkage)
else:
    print("Computing linkage")
    t1 = time()
    Z = linkage(distance_matrix, method=linkage_method)
    t2 = time()
    print(f"Linkage: {t2 - t1:.2f} s")
    print(f"Saving linkage to disk: {file_linkage}")
    os.makedirs(folder_agglomerating_clustering, exist_ok=True)
    np.save(file_linkage, Z)
