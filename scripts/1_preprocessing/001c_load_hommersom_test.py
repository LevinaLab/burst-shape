import os
from typing import Literal

import numpy as np
import scipy.io as sio
import seaborn as sns
from matplotlib import pyplot as plt

from src.folders import get_data_hommersom_test_folder

# file = os.path.join(get_data_hommersom_folder(), "Batch1", "CACN_clone1", "APS_CACN_Clone1_C1.mat")


def _get_hommersom_st(batch, clone: Literal["Clone1", "Clone2", "WTC"], well_id):
    fs = 12500

    file = os.path.join(
        get_data_hommersom_test_folder(),
        batch,
        "WTC" if clone == "WTC" else f"CACN_{clone.lower()}",
        f"APS_CACN_{clone}_{well_id}.mat",
    )

    data = sio.loadmat(file)["Ts_AP"]
    gid = data[:, 0]
    st = data[:, 1]
    st = st / fs
    return gid, st


# gid, st = _get_hommersom_st("Batch1", "Clone1", "C1")
gid, st = _get_hommersom_st("Batch1", "WTC", "A3")
# %% stats

# print("channels", np.unique(data[:, 0]))
# print("Time (min, max)", data[:, 1].min(), data[:, 1].max())

print("channels", np.unique(gid))
print("Time (min, max)", st.min(), st.max())

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
