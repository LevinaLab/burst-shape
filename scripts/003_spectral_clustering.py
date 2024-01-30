import os
import numpy as np

from src.folders import get_results_folder
from src.spectral_clustering import spectral_clustering

bursts = np.load(
    os.path.join(get_results_folder(), "002_wagenaar_bursts_mat.npy")
)  # n_burst x time


labels_per_n_clusters, eigvec, eigval = spectral_clustering.spectral_clustering(
    data=bursts[:, :],
    metric="local_scaled_affinity",
    metric_info="similarity",
    n_clusters=[5],
    precomputed_matrix=None,
    k=5,
    mutual=False,
    weighting=False,
    normalize=True,
    normalization_type="symmetric",
    use_lambda_heuristic=False,
    reg_lambda=0.1,
    saving_lambda_file="data",
    save_laplacian=False,
    save_eigenvalues_and_vectors=False,
)

np.save(
    os.path.join(get_results_folder(), "003_wagenaar_labels_per_n_clusters.npy"),
    labels_per_n_clusters,
)
np.save(
    os.path.join(get_results_folder(), "003_wagenaar_eigvec.npy"),
    eigvec,
)
np.save(
    os.path.join(get_results_folder(), "003_wagenaar_eigval.npy"),
    eigval,
)
