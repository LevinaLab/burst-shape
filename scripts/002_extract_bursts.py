import json
import os
import numpy as np

from src.preprocess import burst_extraction

from src.folders import get_data_folder
from src.persistence import get_burst_folder

# parameters
params_burst_extraction = {
    "maxISIstart": 5,
    "maxISIb": 5,
    "minBdur": 40,
    "minIBI": 40,
    "minSburst": 50,
    "bin_size": None,
    "n_bins": 50,
    "extend_left": 50,
    "extend_right": 50,
    "burst_length_threshold": None,
    "pad_right": False,
}


save_folder = get_burst_folder(params_burst_extraction)
data_folder = os.path.join(get_data_folder(), "extracted")

burst_length_threshold = 5  # in seconds

df_save_path = os.path.join(save_folder, "002_wagenaar_cultures_df.pkl")
df_burst_save_path = os.path.join(save_folder, "002_wagenaar_bursts_df.pkl")
df_mat_save_path = os.path.join(save_folder, "002_wagenaar_bursts_mat.npy")

df_cultures, df_bursts, burst_matrix = burst_extraction.extract_bursts(
    data_folder=os.path.join(get_data_folder(), "extracted"),
    **params_burst_extraction,
)

# save
os.makedirs(save_folder, exist_ok=True)
# dump parameters, new line between each parameter
with open(os.path.join(save_folder, "params.json"), "w") as f:
    json.dump(params_burst_extraction, f, indent=4)
df_cultures.to_pickle(df_save_path)
df_bursts.to_pickle(df_burst_save_path)
np.save(df_mat_save_path, burst_matrix)
