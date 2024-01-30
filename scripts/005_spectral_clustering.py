from sklearn.cluster import SpectralClustering
import os
import numpy as np
import time
import pickle

from src.folders import get_results_folder

bursts = np.load(
    os.path.join(get_results_folder(), "002_wagenaar_bursts_mat.npy")
)  # n_burst x time

start_time = time.time()
clustering = SpectralClustering(
    n_clusters=6,
    affinity="nearest_neighbors",
    n_neighbors=10,
    assign_labels="cluster_qr",
    n_jobs=12,
    random_state=0,
    verbose=True,
).fit(bursts)
end_time = time.time()
print("Elapsed time: {}".format(end_time - start_time))

with open(os.path.join(get_results_folder(), "005_clustering_test.pkl"), "wb") as f:
    pickle.dump(clustering, f)
