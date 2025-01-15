import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from src.persistence import load_df_bursts

burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
    "burst_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4"
)
dataset = "kapucu" if "kapucu" in burst_extraction_params else "wagenaar"

df_bursts = load_df_bursts(burst_extraction_params)
df_bursts.reset_index(drop=False, inplace=True)
if "firing_rate" not in df_bursts.columns:
    df_bursts["firing_rate"] = df_bursts["integral"] / 50
df_bursts["log_firing"] = df_bursts["firing_rate"].apply(lambda x: np.log10(x))
fig, ax = plt.subplots(figsize=(6, 4))
sns.despine()
# df_bursts.plot.hist(column=["log_firing"], ax=ax)
match dataset:
    case "wagenaar":
        sns.histplot(data=df_bursts, x="log_firing", bins=100, ax=ax)
        threshold = 10**3.5
        ax.axvline(np.log10(threshold), color="black", linestyle="--")
    case "kapucu":
        sns.histplot(
            data=df_bursts, x="log_firing", bins=100, ax=ax, hue="culture_type"
        )
        threshold = 10**2.5
        ax.axvline(np.log10(threshold), color="black", linestyle="--")
ax.set_xlabel("log10(firing rate)")
ax.set_ylabel("count")
# ax.set_xscale("log")
fig.show()
