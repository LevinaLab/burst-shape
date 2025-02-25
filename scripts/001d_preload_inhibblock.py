import os
from typing import Literal

import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns
from matplotlib import pyplot as plt

from src.folders import get_data_inhibblock_folder

bic_label = ["A2", "A3", "A4", "A5", "A6", "B2", "B3", "B4", "B5", "B6"]

control_label = ["A1", "B1", "C1", "D1"]
standard_label = ["C2", "C3", "C4", "C5", "C6", "D2", "D3", "D4", "D5", "D6"]


def _get_inhibblock_data(day, conc="4.2"):
    file_postfix = ""
    match day:
        case 18:
            day_folder = "ctx_03.04.22_Hertie"
            file_postfix = "2"
        case 17:
            day_folder = "ctx_14.03.22_Hertie"
        case _:
            raise NotImplementedError(f"day {day} not implemented")

    file = os.path.join(
        get_data_inhibblock_folder(),
        day_folder,
        "extracted_data",
        f"day{day}_potassium{conc}_spikes{file_postfix}.csv",
    )

    data = pd.read_csv(file)

    times = []
    gids = []
    div = []
    wells = []
    drug_label = []
    for label in np.hstack(
        [control_label, standard_label, bic_label]
    ):  # standard_label,bic_label
        st = (data.filter(items=["Timestamp [µs]"])[data["Well Label"] == label]) * 1e-6
        gid = data.filter(items=["Channel ID"])[data["Well Label"] == label]
        st = np.array(st).T[0]
        gid = np.array(gid).T[0]
        # sort st
        times_order = np.argsort(st)

        # well_spikes.append([np.array(st).T[0], np.array(gid).T[0]])
        times.append(st[times_order])
        gids.append(gid[times_order])
        if label in bic_label:
            drug_label.append("bic")
        else:
            assert label in control_label or label in standard_label
            drug_label.append("control")
        div.append(day)

        wells.append(label)
    return pd.DataFrame(
        {
            "well": wells,
            "div": div,
            "drug_label": drug_label,
            # "well_spikes": well_spikes,
            "times": times,
            "gid": gids,
        }
    )


df1 = _get_inhibblock_data(18)
df2 = _get_inhibblock_data(17)
df = pd.concat([df1, df2])
df.reset_index(inplace=True, drop=True)
del df1, df2

# transform well_id to well_idx
df["well_idx"] = pd.Series(0, dtype=int)
for row in df[["drug_label", "div"]].drop_duplicates().itertuples():
    drug_label, div = row.drug_label, row.div
    n_wells = len(df.loc[(df["drug_label"] == drug_label) & (df["div"] == div)])
    df.loc[(df["drug_label"] == drug_label) & (df["div"] == div), "well_idx"] = list(
        range(n_wells)
    )
df["well_idx"] = df["well_idx"].astype(int)
df.set_index(["drug_label", "div", "well_idx"], inplace=True)

df.to_pickle(os.path.join(get_data_inhibblock_folder(), "df_inhibblock.pkl"))
# %% example
index = 2
st, gid = df.iloc[index][["times", "gid"]]
print(f"Selected example, index {index}, drug label = {df.iloc[index].name[0]}")
# %% raster plot
fig, ax = plt.subplots(constrained_layout=True)
sns.despine()
ax.plot(st, gid, "|", ms=20, color="k", alpha=0.5)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Channel")
fig.show()

# %% trace of firing rate
bin_size = 0.1  # s
times_all = np.arange(0, st.max() + bin_size, bin_size)
firing_rate = np.histogram(st, bins=times_all)[0] / (bin_size)  #  / 1000)
times_all = 0.5 * (times_all[1:] + times_all[:-1])

fig, ax = plt.subplots(constrained_layout=True)
sns.despine()
ax.plot(times_all, firing_rate, "-", color="k")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Frequency")
fig.show()
