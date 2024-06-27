"""Visualize the burst cluster similar to the Figure 3 in Wagenaar et al. 2006."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.persistence import load_clustering_labels, load_df_bursts

# which clustering to plot
n_clusters = 8
col_cluster = f"cluster_{n_clusters}"

# parameters which clustering to plot
burst_extraction_params = "burst_n_bins_50_normalization_integral"
clustering_params = "spectral_affinity_precomputed_metric_wasserstein"
labels_params = "labels"
cv_params = "cv"  # if cv_split is not None, chooses the cross-validation split
cv_split = (
    None  # set to None for plotting the whole clustering, set to int for specific split
)

# load data
clustering = load_clustering_labels(
    clustering_params, burst_extraction_params, labels_params, cv_params, cv_split
)
df_bursts = load_df_bursts(burst_extraction_params, cv_params=cv_params)
if cv_split is not None:
    df_bursts = df_bursts[df_bursts[f"cv_{cv_split}_train"]]
for n_clusters_ in clustering.n_clusters:
    df_bursts[f"cluster_{n_clusters_}"] = clustering.labels_[n_clusters_]

# index of df_bursts is ('batch', 'culture', 'day', 'i_burst')
df_bursts.reset_index(inplace=True)

# %% build new dataframe df_cultures with index ('batch', 'culture', 'day') and columns ('n_bursts', 'cluster_abs', 'cluster_rel')
df_cultures = df_bursts.groupby(["batch", "culture", "day"]).agg(
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

# %% add cluster information
for i_cluster in range(n_clusters):
    col_cluster = f"cluster_{n_clusters}"
    df_cultures[f"cluster_abs_{i_cluster}"] = df_bursts.groupby(
        ["batch", "culture", "day"]
    )[col_cluster].agg(lambda x: np.sum(x == i_cluster))
    df_cultures[f"cluster_rel_{i_cluster}"] = (
        df_cultures[f"cluster_abs_{i_cluster}"] / df_cultures["n_bursts"]
    )

# %% plot a pie chart for each entry in df_cultures
# position of the pie chart in the grid is determined by the day and i_culture
colors = sns.color_palette("Set1", n_colors=n_clusters)
nrows = df_cultures["i_culture"].max() + 1
ncols = df_cultures.index.get_level_values("day").max() + 1
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
sns.despine(fig=fig, top=True, right=True, left=True, bottom=True)
# set all axes to invisible
for ax in axs.flatten():
    ax.axis("off")
for index in df_cultures.index:
    i_day = index[2]
    i_culture = df_cultures.loc[index, "i_culture"]
    ax = axs[i_culture, i_day]
    ax.axis("on")
    # ax.set_title(f"Day {i_day} - Culture {i_culture}")
    cluster_rel = [
        df_cultures.loc[index, f"cluster_rel_{i_cluster}"]
        for i_cluster in range(n_clusters)
    ]
    ax.pie(cluster_rel, colors=colors, startangle=90)

# Add a shared x-axis for days
for i_day in np.arange(ncols)[::2]:
    ax = axs[-1, i_day]
    ax.axis("on")
    ax.set_xlabel(f"Day {i_day}", fontsize=12)
    ax.xaxis.set_label_position("bottom")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)

# write batches on the left
batch_label_pos = np.linspace(0.04, 0.97, nrows, endpoint=True)[::-1]
for (batch, culture), row in unique_batch_culture.iterrows():
    i_culture = row["i_culture"]
    fig.text(
        0.05, batch_label_pos[i_culture], fontsize=12, va="center", s=f"Batch {batch}"
    )

fig.legend(
    [f"Cluster {i_cluster}" for i_cluster in range(n_clusters)],
    loc="center right",
    frameon=False,
)
fig.tight_layout()
fig.subplots_adjust(wspace=-0.15, hspace=-0.15)
fig.show()
