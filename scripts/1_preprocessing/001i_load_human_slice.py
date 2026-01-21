import os

import numpy as np
import pandas as pd
from src.folders import get_data_human_slice_folder

path_data = get_data_human_slice_folder()
path_package = os.path.join(path_data, "Data_package1")
label_list = os.listdir(path_package)

df_list = []

for label in label_list:
    # label = "Spont1"
    file_name = os.path.join(
        path_data,
        path_package,
        label,
        "Spike_Data_for_LFP/All_Spike_Times.csv",
    )
    print(f"Loading human slice data from: {file_name}")

    data = pd.read_csv(file_name)


    st = data["Time_s"]
    gid = data["Channel"]

    df_list.append({
        "label": label,
        "dummy": 0,
        "times": np.array(st),
        "gid": np.array(gid),
    })
df = pd.DataFrame(df_list)
df.set_index(["label", "dummy"], inplace=True)

df.to_pickle(os.path.join(path_data, "df_package1.pkl"))

