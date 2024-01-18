import os
import numpy as np

from src.folders import get_results_folder

bursts = np.load(
    os.path.join(get_results_folder(), "002_wagenaar_bursts.npy")
)  # n_burst x time

