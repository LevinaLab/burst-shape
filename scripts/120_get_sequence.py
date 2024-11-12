"""Exctract sequence of cluster labels.

They are saved in df_cultures with index (batch, culture, day) and columns:
- n_bursts: number of bursts
- sequence: list of integers containing the sequence
"""
import os

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import  fcluster
from tqdm import tqdm

from src.persistence import load_df_bursts
from src.persistence.burst_extraction import _get_burst_folder

burst_extraction_params = (
    "burst_n_bins_50_normalization_integral_min_length_30_smoothing_kernel_4_outlier_removed"
)
agglomerating_clustering_params = "agglomerating_clustering_linkage_complete_n_bursts_None"
np.random.seed(0)

# plot settings
n_clusters = 5  # CHOOSE NUMBER OF CLUSTERS HERE

folder_agglomerating_clustering = os.path.join(
    _get_burst_folder(burst_extraction_params),
    agglomerating_clustering_params,
)
file_linkage = os.path.join(folder_agglomerating_clustering, "linkage.npy")

# load bursts
df_bursts = load_df_bursts(burst_extraction_params)
df_bursts = df_bursts.sort_index()

if not os.path.exists(file_linkage):
    raise FileNotFoundError(f"Linkage file not found: {file_linkage}")
else:
    print(f"Loading linkage from {file_linkage}")
    Z = np.load(file_linkage)

# %% get clusters from linkage
print("Getting clusters from linkage...")
labels = fcluster(Z, t=n_clusters, criterion="maxclust")
df_bursts["cluster"] = labels

# Define a color palette for the clusters
palette = sns.color_palette(n_colors=n_clusters)  # "Set1", n_clusters)
cluster_colors = [palette[i - 1] for i in range(1, n_clusters + 1)]
# convert colors to string (hex format)
cluster_colors = [
    f"#{int(c[0] * 255):02x}{int(c[1] * 255):02x}{int(c[2] * 255):02x}"
    for c in cluster_colors
]

# %% build new dataframe df_cultures with index ('batch', 'culture', 'day') and columns ('n_bursts', 'cluster_abs', 'cluster_rel')
print("Building df_cultures...")
df_bursts_reset = df_bursts.reset_index(
    drop=False
)  # reset index to access columns in groupby()
df_cultures = df_bursts_reset.groupby(["batch", "culture", "day"]).agg(
    n_bursts=pd.NamedAgg(column="i_burst", aggfunc="count")
)

# for all unique combinations of batch and culture
unique_batch_culture = df_cultures.reset_index()[["batch", "culture"]].drop_duplicates()
# sort by batch and culture
unique_batch_culture.sort_values(["batch", "culture"], inplace=True)
# assign an index to each unique combination
unique_batch_culture["i_culture"] = np.arange(len(unique_batch_culture))  # [::-1]
unique_batch_culture.set_index(["batch", "culture"], inplace=True)

df_cultures["i_culture"] = pd.Series(
    data=(
        df_cultures.reset_index().apply(
            lambda x: unique_batch_culture.loc[(x["batch"], x["culture"]), "i_culture"],
            axis=1,
        )
    ).values,
    index=df_cultures.index,
    dtype=int,
)
# dd cluster information
df_cultures["sequence"] = pd.Series(dtype="object")
for index in tqdm(df_cultures.index):
    batch, culture, day = index
    df_select = df_bursts.loc[index]
    assert len(df_select) == df_cultures.at[index, "n_bursts"]
    assert df_select.index.is_monotonic_increasing
    assert df_select["start_orig"].is_monotonic_increasing
    df_cultures.at[index, "sequence"] = df_select["cluster"].to_list()

