import os

import numpy as np
import pandas as pd

from src.folders import get_data_human_slice_folder

path_data = get_data_human_slice_folder()
path_raw = os.path.join(path_data, "2026_04_14_Package")
subject_list = os.listdir(path_raw)

df_list = []

for subject_id in subject_list:
    path_subject = os.path.join(path_raw, subject_id)
    cut_list = [
        cut_name_
        for cut_name_ in os.listdir(path_subject)
        if cut_name_.startswith(subject_id)
    ]
    for cut_name in cut_list:
        path_cut = os.path.join(path_subject, cut_name)
        condition_list = [
            condition_name_
            for condition_name_ in os.listdir(path_cut)
            if os.path.isdir(os.path.join(path_cut, condition_name_))
        ]
        for condition_name in condition_list:
            file_name = os.path.join(
                path_raw,
                subject_id,
                cut_name,
                condition_name,
                "All_Spike_Times.csv",
            )
            print(f"Loading human slice data from: {file_name}")

            data = pd.read_csv(file_name)

            st = data["Time_s"]
            gid = data["Channel"]

            df_list.append(
                {
                    "subject_id": subject_id,
                    "cut": cut_name,
                    "condition": condition_name,
                    "times": np.array(st),
                    "gid": np.array(gid),
                }
            )
df = pd.DataFrame(df_list)
df.set_index(["subject_id", "cut", "condition"], inplace=True)

# print statistics of "condition" column
print("Condition value counts:")
print(df.reset_index()["condition"].value_counts())

# cleanup
# 1) remove condition = 'test'
df = df[df.index.get_level_values("condition") != "test"]

# 2) C60_1ÊM_Spont1 -> C60_Spont1  (and same for Spont2)
df = df.rename(index=lambda x: x.replace("_1ÊM", ""), level="condition")

# 3) remove all conditions with <=4 samples
value_counts = df.reset_index()["condition"].value_counts()
df = df[
    df.index.get_level_values("condition").isin(value_counts[value_counts > 4].index)
]

# print statistics of "condition" column
print("\nAfter cleanup")
print("Condition value counts:")
print(df.reset_index()["condition"].value_counts())

# save
df.to_pickle(os.path.join(path_data, "df_human_slice_half_data.pkl"))
