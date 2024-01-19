import os
import numpy as np

from src.folders import get_results_folder
from src.spectral_clustering import spectral_clustering

bursts = np.load(
    os.path.join(get_results_folder(), "002_wagenaar_bursts.npy")
)  # n_burst x time


labels_per_n_clusters, eigvec, eigval = spectral_clustering(data = bursts[:2000, :15] , 
                                                            metric = 'local_scaled_affinity',
                                                            metric_info = 'similarity',
                                                            n_clusters = [5], 
                                                            precomputed_matrix=None, 
                                                            k=5, 
                                                            mutual = False,
                                                            weighting = False, 
                                                            normalize = True, 
                                                            normalization_type = "symmetric",
                                                            use_lambda_heuristic = False, 
                                                            reg_lambda = 0.1, 
                                                            saving_lambda_file= "data",
                                                            save_laplacian = False, 
                                                            save_eigenvalues_and_vectors = False)