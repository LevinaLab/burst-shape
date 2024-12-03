from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

from src.persistence import load_df_bursts

threshold = 10 ** 3.5
burst_extraction_params = (
    "burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
)

df_bursts = load_df_bursts(burst_extraction_params)
df_bursts["firing_rate"] = df_bursts["integral"] / 50
df_bursts["log_firing"] = df_bursts["firing_rate"].apply(lambda x: np.log10(x))
fig, ax = plt.subplots(figsize=(6, 4))
sns.despine()
# df_bursts.plot.hist(column=["log_firing"], ax=ax)
ax.hist(df_bursts["log_firing"], bins=100)
ax.axvline(np.log10(threshold), color="black", linestyle="--")
ax.set_xlabel("log10(firing rate)")
ax.set_ylabel("count")
# ax.set_xscale("log")
fig.show()
