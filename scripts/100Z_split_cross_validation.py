import os

import numpy as np
from scipy.spatial.distance import squareform
from tqdm import tqdm

from src.cross_validation.split import split_training_and_validation_data
from src.persistence import load_df_bursts, save_cv_params, save_df_bursts
from src.persistence.burst_extraction import _get_burst_folder

# parameters
burst_extraction_params = "burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
agglomerating_clustering_params = "agglomerating_clustering_linkage_complete_n_bursts_None"
cv_params = {
    "type": "kfold",
    "n_splits": 5,
    "shuffle": True,
    "random_state": 0,
}

# load data
print("Loading data (df_bursts)...")
df_bursts = load_df_bursts(burst_extraction_params)

# split data
print("Splitting data...")
df_bursts = split_training_and_validation_data(df_bursts, **cv_params)

# split distance matrix and save
print("Loading distance matrix...")
folder_agglomerating_clustering = os.path.join(
    _get_burst_folder(burst_extraction_params),
    agglomerating_clustering_params,
)
file_distance_matrix = os.path.join(
    folder_agglomerating_clustering, "distance_matrix.npy"
)
distance_matrix = np.load(file_distance_matrix)
distance_matrix = squareform(distance_matrix, force="tomatrix")
# %%
for i_split in tqdm(range(cv_params["n_splits"]), desc="Splitting distance matrix"):
    index_split = df_bursts[f"cv_{i_split}_train"]
    distance_matrix_split = distance_matrix[np.ix_(index_split, index_split)]
    distance_matrix_split = squareform(distance_matrix_split, force="tovector")
    np.save(
        os.path.join(
            folder_agglomerating_clustering, f"distance_matrix_cv_{i_split}"
        ),
        distance_matrix_split,
    )

# save cv_params
print("Saving cv_params...")
save_cv_params(cv_params, burst_extraction_params)

# save
print("Saving df_bursts_cv")
save_df_bursts(df_bursts, burst_extraction_params, cv_params=cv_params)

print("Finished.")
