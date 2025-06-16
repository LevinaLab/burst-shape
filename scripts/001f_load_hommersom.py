import os
import random

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.folders import get_data_folder, get_data_hommersom_folder

df_batch_list = []
for batch in tqdm([1, 2, 3, 4], desc="Load Batch data"):
    file = os.path.join(
        get_data_folder(), "data_hommersom", f"Batch{batch}_Axion_PT_all.csv"
    )
    df_batch = pd.read_csv(file)
    df_batch["batch"] = batch
    df_batch_list.append(df_batch)
df_spikes = pd.concat(df_batch_list, ignore_index=True)
del df_batch_list, df_batch

# %% rast plot one well
unique_batch_culture = df_spikes[["batch", "Well_Label"]].drop_duplicates()
batch, well = unique_batch_culture[["batch", "Well_Label"]].values[
    random.randint(0, len(unique_batch_culture) - 1)
]
print(f"plot random example batch {batch} well {well}")

df_select = df_spikes[(df_spikes["Well_Label"] == well) & (df_spikes["batch"] == batch)]
# %%
st, gid = df_select["Timestamp"], df_select["Channel_Label"]
fig, ax = plt.subplots(constrained_layout=True)
sns.despine()
ax.plot(st, gid, "|", ms=5, color="k", alpha=0.5)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Channel")
fig.show()

# %%
df_cultures = pd.DataFrame(unique_batch_culture.values, columns=["batch", "well"])
df_cultures.set_index(["batch", "well"], inplace=True)
df_cultures.sort_index(inplace=True)
df_cultures["times"] = pd.Series(dtype="object")
df_cultures["gid"] = pd.Series(dtype="object")
for index in tqdm(df_cultures.index, desc="Load spikes into df_cultures"):
    batch, well = index
    selection = (df_spikes["batch"] == batch) & (df_spikes["Well_Label"] == well)
    st = df_spikes.loc[selection]["Timestamp"].values
    gid = df_spikes.loc[selection]["Channel_Label"].values
    argsort = np.argsort(st)
    st, gid = st[argsort], gid[argsort]
    df_cultures.at[index, "times"] = st
    df_cultures.at[index, "gid"] = gid

df_cultures.to_pickle(os.path.join(get_data_hommersom_folder(), "df_hommersom.pkl"))
