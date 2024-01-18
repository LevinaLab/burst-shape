import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocess import burst_detection
from src.folders import get_data_folder, get_results_folder

data_folder = os.path.join(get_data_folder(), "extracted")
plot = False
bin_size = 0.1  # in seconds
maxISIstart = 5
maxISIb = 5
minBdur = 40
minIBI = 40
minSburst = 50

burst_length_threshold = 5  # in seconds

df_save_path = os.path.join(get_results_folder(), "002_wagenaar_bursts.csv")
burst_save_path = os.path.join(get_results_folder(), "002_wagenaar_bursts.npy")

# %% build pandas dataframe
file_list = os.listdir(data_folder)
# file_list = [file for file in file_list if ".txt" in file]
df = pd.DataFrame(file_list, columns=["file_name"])
# create columns batch-culture-day
df["batch"] = df["file_name"].apply(lambda x: x.split("-")[0])
df["culture"] = df["file_name"].apply(lambda x: x.split("-")[1])
df["day"] = df["file_name"].apply(lambda x: x.split("-")[2].split(".")[0])
# set index
df.set_index(["batch", "culture", "day"], inplace=True)

# %%
df["n_bursts"] = pd.Series(dtype=object)
df["burst_list"] = pd.Series(dtype=object)
for index in tqdm(df.index):
    batch, culture, day = index
    file_name = df.at[index, "file_name"]
    file_path = os.path.join(data_folder, file_name)
    data = np.loadtxt(file_path)[:, 0]  # only spike times
    time_max = np.max(data)
    bursts_start_end = burst_detection.MI_bursts(
        data * 1000,
        maxISIstart=maxISIstart,
        maxISIb=maxISIb,
        minBdur=minBdur,
        minIBI=minIBI,
        minSburst=minSburst,
    )
    bursts_start_end = np.array(bursts_start_end) / 1000  # convert to seconds

    # bin data
    bins = np.arange(0, time_max + bin_size, bin_size)
    counts, _ = np.histogram(data, bins=bins)

    # cut bursts from binned data
    bursts_start_end_index = [
        (
            int(np.floor(burst[0] / bin_size)),
            int(np.ceil(burst[1] / bin_size)),
        )
        for burst in bursts_start_end
    ]
    bursts = [counts[burst[0] : burst[1]] for burst in bursts_start_end_index]

    # save
    df.at[index, "n_bursts"] = len(bursts_start_end)
    df.at[index, "burst_list"] = bursts

    if plot is True:
        # plot activity
        fig, ax = plt.subplots()
        sns.despine()
        ax.plot(bins[:-1], counts)
        # highlight bursts
        for burst in bursts_start_end:
            ax.axvspan(burst[0], burst[1], color="red", alpha=0.5)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("spike count")
        ax.set_title(f"b={batch}, c={culture}, d={day}: bursts={len(bursts_start_end)}")
        fig.show()
# save
df.to_pickle(df_save_path)

# %%
# threshold burst
burst_list = []
n_deleted_bursts = 0
for bursts in df["burst_list"]:
    additional_burst = [
        burst for burst in bursts if len(burst) * bin_size <= burst_length_threshold
    ]
    burst_list.extend(additional_burst)
    n_deleted_bursts += len(bursts) - len(additional_burst)
print(
    f"Deleted {n_deleted_bursts} bursts "
    f"because they were longer than {burst_length_threshold} seconds."
)

# pad with zeros at the end
max_length = int(np.max([len(burst) for burst in burst_list]))
burst_list = [
    np.pad(burst, (0, max_length - len(burst)), "constant") for burst in burst_list
]
burst_matrix = np.vstack(burst_list)
# save
np.save(burst_save_path, burst_matrix)
