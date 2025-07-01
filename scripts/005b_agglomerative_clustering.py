import multiprocessing
from time import time

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from src.persistence import (
    distance_matrix_exists,
    linkage_exists,
    load_cv_params,
    load_distance_matrix,
    save_distance_matrix,
    save_linkage,
)
from src.persistence.burst_extraction import load_burst_matrix, load_df_bursts

burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "dataset_kapucu_burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_test_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_maxISIstart_50_maxISIb_50_minBdur_100_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_maxISIstart_100_maxISIb_50_minBdur_100_minIBI_500_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_hommersom_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    "burst_dataset_hommersom_binary_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
)
compute_parallel = True  # if True uses double the memory but is faster
recompute_distance_matrix = False  # if False and available loads the data from disk

# clustering params
recompute_linkage = False  # if False and available loads the data from disk
agglomerative_clustering_params = {
    "linkage": "ward",
}
cv_params = (
    None  # "cv"  # set to None if not using cross-validation or to specific cv_params
)
all_data = True  # set to False if you want to compute only cross-validation
np.random.seed(0)

# load cross-validation parameters
if cv_params is not None:
    cv_params_dict = load_cv_params(burst_extraction_params, cv_params)
    n_splits = cv_params_dict["n_splits"]
else:
    n_splits = 0

# define which clustering tasks to perform
tasks = []
if all_data:
    tasks.append(None)
if n_splits > 0:
    tasks.extend(list(range(n_splits)))

# %% load bursts
burst_matrix = load_burst_matrix(burst_extraction_params)

# %%


def _wasserstein_distance(a, b):
    a_cumsum = np.cumsum(a)
    b_cumsum = np.cumsum(b)
    a_cumsum /= a_cumsum[-1]
    b_cumsum /= b_cumsum[-1]
    return np.sum(np.abs(a_cumsum - b_cumsum))


# %% compute distance matrix
for i_split in tasks:
    split_str = f"{'all data' if i_split is None else f'split {i_split}'}"
    if not recompute_distance_matrix and distance_matrix_exists(
        burst_extraction_params,
        cv_params,
        i_split,
    ):
        print(f"Distance matrix for {split_str} already exists.")
        continue
    elif i_split is None:
        print(f"Computing distance matrix for {split_str}.")
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

    else:
        print(f"Subsampling distance_matrix for {split_str}.")
        # if local variable distance_matrix_full is not defined, load it from disk
        if "distance_matrix_full" not in locals():
            print("Loading full distance matrix from disk.")
            distance_matrix_full = load_distance_matrix(
                burst_extraction_params, form="matrix"
            )
        else:
            print("Using already loaded distance matrix from previous iteration.")
        df_bursts = load_df_bursts(burst_extraction_params, cv_params)
        index_split = df_bursts[f"cv_{i_split}_train"]
        print("Do subsampling.")
        distance_matrix = distance_matrix_full[np.ix_(index_split, index_split)]
        distance_matrix = squareform(distance_matrix, force="tovector")
    print(f"Saving distance matrix to disk.")  # : {file_distance_matrix}")
    save_distance_matrix(distance_matrix, burst_extraction_params, cv_params, i_split)

# %% compute linkage
for i_split in tasks:
    print(f"Linkage for {'all data' if i_split is None else f'split {i_split}'}")
    if not recompute_linkage and linkage_exists(
        burst_extraction_params, agglomerative_clustering_params, cv_params, i_split
    ):
        print(f"Linkage already exists.")
        continue
    else:
        print(f"Loading distance matrix from file")  # {file_distance_matrix_}")
        try:
            distance_matrix = load_distance_matrix(
                burst_extraction_params,
                params_cross_validation=cv_params,
                i_split=i_split,
            )
        except FileNotFoundError as e:
            if i_split is None:
                raise e
            else:
                raise FileNotFoundError(
                    "Distance matrix doesn't exist yet for cross-validation."
                    "Potentially, you have to first run 'split_cross_validation' to split up the distance_matrix."
                )
        print("Computing linkage")
        t1 = time()
        Z = linkage(distance_matrix, method=agglomerative_clustering_params["linkage"])
        t2 = time()
        print(f"Linkage: {t2 - t1:.2f} s")
        print(f"Saving linkage to disk.")  # : {file_linkage_}")
        save_linkage(
            Z,
            burst_extraction_params,
            agglomerative_clustering_params,
            cv_params,
            i_split,
        )
