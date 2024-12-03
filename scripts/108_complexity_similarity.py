import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import  fcluster

from src import folders
from src.persistence import load_burst_matrix, load_df_bursts
from src.persistence.burst_extraction import _get_burst_folder

burst_extraction_params = (
    "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
)
agglomerating_clustering_params = "agglomerating_clustering_linkage_complete_n_bursts_None"
np.random.seed(0)

# plot settings
n_clusters = 5  # 3  # if None chooses the number of clusters with Davies-Bouldin index

# plotting
cm = 1 / 2.54  # centimeters in inches
fig_path = folders.get_fig_folder()

folder_agglomerating_clustering = os.path.join(
    _get_burst_folder(burst_extraction_params),
    agglomerating_clustering_params,
)
file_distance_matrix = os.path.join(
    folder_agglomerating_clustering, "distance_matrix.npy"
)
file_linkage = os.path.join(folder_agglomerating_clustering, "linkage.npy")

# load bursts
burst_matrix = load_burst_matrix(burst_extraction_params)
df_bursts = load_df_bursts(burst_extraction_params)
np.random.seed(0)

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
    f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
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
for i_cluster in range(1, n_clusters + 1):
    col_cluster = "cluster"
    df_cultures[f"cluster_abs_{i_cluster}"] = df_bursts.groupby(
        ["batch", "culture", "day"]
    )[col_cluster].agg(lambda x: np.sum(x == i_cluster))
    df_cultures[f"cluster_rel_{i_cluster}"] = (
        df_cultures[f"cluster_abs_{i_cluster}"] / df_cultures["n_bursts"]
    )

# %% build df_cultures_weeks where
# First, reset the index to access the 'day' column, if needed
df_cultures_reset = df_cultures.reset_index()

# Add a new 'week' column representing the week of each 'day'
df_cultures_reset['week'] = df_cultures_reset['day'] // 7

# Group by batch, culture, and week, then average the values within each week
df_cultures_weeks = df_cultures_reset.groupby(['batch', 'culture', 'week']).agg(
    n_bursts=('n_bursts', 'mean'),
    i_culture=('i_culture', 'first')  # assuming i_culture remains constant within a culture
)

# For cluster-related columns, average them for each week as well
for i_cluster in range(1, n_clusters + 1):
    df_cultures_weeks[f'cluster_abs_{i_cluster}'] = df_cultures_reset.groupby(['batch', 'culture', 'week'])[f'cluster_abs_{i_cluster}'].mean()
    df_cultures_weeks[f'cluster_rel_{i_cluster}'] = df_cultures_reset.groupby(['batch', 'culture', 'week'])[f'cluster_rel_{i_cluster}'].mean()

# Reset index if needed, otherwise keep it as is for hierarchical indexing
# df_cultures_weeks.reset_index(inplace=True)

del df_cultures_reset

# %% development plot based
print("Plotting development...")
fig, ax = plt.subplots(figsize=(4.6 * cm, 3.5 * cm))
sns.despine()
days = df_bursts.index.get_level_values("day").unique().sort_values()
fraction = np.zeros((len(days), n_clusters))
for i, day in enumerate(days):
    for cluster in range(1, n_clusters + 1):
        fraction[i, cluster - 1] = np.mean(
            df_bursts[df_bursts.index.get_level_values("day") == day]["cluster"]
            == cluster
        )
# for cluster in range(1, n_clusters + 1):
# ax.plot(days, fraction[:, cluster - 1], color=palette[cluster - 1], label=f"Cluster {cluster}")
# smooth fraction by moving average over 5 days
for cluster in range(1, n_clusters + 1):
    fraction[:, cluster - 1] = np.convolve(
        fraction[:, cluster - 1], np.ones(5) / 5, mode="same"
    )
days = days[2:-2]
fraction = fraction[2:-2]
# fraction /= fraction.sum(axis=1)[:, None]
for cluster in range(1, n_clusters + 1):
    ax.plot(
        days, fraction[:, cluster - 1], color=palette[cluster - 1]
    )  # , linestyle="--")
ax.set_xlabel("Day")
ax.set_ylabel("Fraction")
# ax.legend()
fig.tight_layout()
fig.show()
fig.savefig(os.path.join(fig_path, "fraction_clusters.svg"))
fig.savefig(os.path.join(fig_path, "fraction_clusters.pdf"))


# %% development plot based on df_cultures
print("Plotting development based on df_cultures...")
for df_, column in zip([df_cultures, df_cultures_weeks], ["day", "week"]):
    fig, ax = plt.subplots(figsize=(4.6 * cm, 3.5 * cm), constrained_layout=True)
    sns.despine()
    days = df_.index.get_level_values(column).unique().sort_values()
    fraction = np.zeros((len(days), n_clusters))
    for i, day in enumerate(days):
        for cluster in range(1, n_clusters + 1):
            fraction[i, cluster - 1] = np.mean(
                df_[df_.index.get_level_values(column) == day][
                    f"cluster_rel_{cluster}"
                ]
            )
    if column == "day":
        # smooth fraction by moving average over 5 days
        for cluster in range(1, n_clusters + 1):
            fraction[:, cluster - 1] = np.convolve(
                fraction[:, cluster - 1], np.ones(5) / 5, mode="same"
            )
        days = days[2:-2]
        fraction = fraction[2:-2]
    # fraction /= fraction.sum(axis=1)[:, None]
    for cluster in range(1, n_clusters + 1):
        ax.plot(
            days,
            fraction[:, cluster - 1],
            color=palette[cluster - 1],
            label=f"Cluster {cluster}",
        )
    ax.set_xlabel(column)
    ax.set_ylabel("Fraction")
    fig.show()
    fig.savefig(os.path.join(fig_path, f"fraction_clusters_df_cultures_{column}.svg"))
    fig.savefig(os.path.join(fig_path, f"fraction_clusters_df_cultures_{column}.pdf"))

# %% complexity plot (information)
print("Plotting information...")
for df_, column in zip([df_cultures, df_cultures_weeks], ["day", "week"]):
    # compute information based on cluster_rel columns
    columns = [f"cluster_rel_{i_cluster}" for i_cluster in range(1, n_clusters + 1)]
    df_["information"] = df_.apply(
        lambda x: -np.sum([(p * np.log2(p) if p > 0 else 0) for p in x[columns]]), axis=1
    )
    # plot average information per day
    fig, ax = plt.subplots(figsize=(4.6 * cm, 3.5 * cm), constrained_layout=True)
    sns.despine()
    days = df_.index.get_level_values(column).unique().sort_values()
    information = np.zeros(len(days))
    for i, day in enumerate(days):
        information[i] = df_[df_.index.get_level_values(column) == day][
            "information"
        ].mean()
    if column == "day":
        # smooth information by moving average over 5 days
        information = np.convolve(information, np.ones(5) / 5, mode="valid")
        days = days[2:-2]
    ax.plot(days, information, color="black")
    ax.set_xlabel("column")
    ax.set_ylabel("Info [bits]")
    fig.show()
    fig.savefig(os.path.join(fig_path, f"information_{column}.svg"))
    fig.savefig(os.path.join(fig_path, f"information_{column}.pdf"))


# %% similarity between batches (vector product of cluster_rel)
print("Computing similarity (attempt 1)...")


def _similarity(x, y):
    return np.sum(
        [
            x[f"cluster_rel_{i_cluster}"] * y[f"cluster_rel_{i_cluster}"]
            for i_cluster in range(1, n_clusters + 1)
        ]
    )


def _similarity(x, y):
    # MSE
    return np.sum(
        [
            (x[f"cluster_rel_{i_cluster}"] - y[f"cluster_rel_{i_cluster}"]) ** 2
            for i_cluster in range(1, n_clusters + 1)
        ]
    )


def _similarity_df(df):
    # average between all combinations
    if len(df) <= 1:
        return np.array([np.nan])
    similarities = np.zeros((len(df), len(df)))
    for i, x in enumerate(df.iterrows()):
        for j, y in enumerate(df.iterrows()):
            similarities[i, j] = _similarity(x[1], y[1])
    # average over upper triangle
    similarities = similarities[np.triu_indices(len(df), k=1)].flatten()
    return similarities


similarity_random = _similarity_df(df_cultures.sample(frac=1))
similarity_batch_random = np.array(
    [
        _similarity_df(
            df_cultures[df_cultures.index.get_level_values("batch") == batch]
        ).mean()
        for batch in df_cultures.index.get_level_values("batch").unique()
    ]
).flatten()
similarity_same_day = np.array(
    [
        _similarity_df(
            df_cultures[df_cultures.index.get_level_values("day") == day]
        ).mean()
        for day in df_cultures.index.get_level_values("day").unique()
    ]
).flatten()
similarity_batch_day = np.array(
    [
        _similarity_df(
            df_cultures[
                (df_cultures.index.get_level_values("day") == day)
                & (df_cultures.index.get_level_values("batch") == batch)
            ]
        ).mean()
        for day in df_cultures.index.get_level_values("day").unique()
        for batch in df_cultures.index.get_level_values("batch").unique()
    ]
).flatten()

fig, ax = plt.subplots(figsize=(4.6 * cm, 3.5 * cm), constrained_layout=True)
sns.despine()
sns.violinplot(
    data=[
        similarity_random,
        similarity_batch_random,
        similarity_same_day,
        similarity_batch_day,
    ],
    ax=ax,
)
ax.set_ylabel("Similarity")
fig.show()


# %% plot a pie chart for each entry in df_cultures
# position of the pie chart in the grid is determined by the day and i_culture
print("Plotting pie charts...")
for df_, column in zip([df_cultures, df_cultures_weeks], ["day", "week"]):
    colors = palette  #  sns.color_palette("Set1", n_colors=n_clusters)
    nrows = df_["i_culture"].max() + 1
    ncols = df_.index.get_level_values(column).max() + 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
    sns.despine(fig=fig, top=True, right=True, left=True, bottom=True)
    # set all axes to invisible
    for ax in axs.flatten():
        ax.axis("off")
    for index in df_.index:
        i_day = index[2]
        i_culture = df_.loc[index, "i_culture"]
        ax = axs[i_culture, i_day]
        ax.axis("on")
        # ax.set_title(f"Day {i_day} - Culture {i_culture}")
        cluster_rel = [
            df_.loc[index, f"cluster_rel_{i_cluster}"]
            for i_cluster in range(1, n_clusters + 1)
        ]
        ax.pie(cluster_rel, colors=colors, startangle=90)

    # Add a shared x-axis for days
    for i_day in np.arange(ncols)[::2]:
        ax = axs[-1, i_day]
        ax.axis("on")
        ax.set_xlabel(f"{column} {i_day}", fontsize=12)
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
        [f"Cluster {i_cluster}" for i_cluster in range(1, n_clusters + 1)],
        loc="center right",
        frameon=False,
    )
    fig.tight_layout()
    fig.subplots_adjust(wspace=-0.15, hspace=-0.15)
    fig.show()
    fig.savefig(os.path.join(fig_path, f"pie_charts_{column}.svg"))
    fig.savefig(os.path.join(fig_path, f"pie_charts_{column}.pdf"))

# %% second attempt at similarity
print("Computing similarity (attempt 2)...")
def _cosine_similarity(x, y):
    return np.dot(x ,y) / (np.linalg.norm(x) * np.linalg.norm(y))

columns = [f"cluster_rel_{i_cluster}" for i_cluster in range(1, n_clusters + 1)]

def _cosine_similarity_df(df_selection, df_selection2=None):
    # average between all combinations
    is_comparison = df_selection2 is not None
    n_samples = len(df_selection)
    fraction_array = df_selection[columns].to_numpy()
    if is_comparison:
        n_samples2 = len(df_selection2)
        fraction_array2 = df_selection2[columns].to_numpy()
        if n_samples == 0 or n_samples2 == 0:
            return np.array([np.nan])
    else:
        if n_samples <= 1:
            return np.array([np.nan])
        n_samples2 = n_samples
        fraction_array2 = fraction_array
    similarities = np.zeros((n_samples, n_samples2))
    for i_sample in range(n_samples):
        for j_sample in range(n_samples2):
            similarities[i_sample, j_sample] = _cosine_similarity(
                fraction_array[i_sample],
                fraction_array2[j_sample],
            )
    if is_comparison:
        similarities = similarities.flatten()
    else:
        # average over upper triangle
        similarities = similarities[np.triu_indices(n_samples, k=1)].flatten()
    return similarities

for df_, column in zip([df_cultures, df_cultures_weeks], ["day", "week"]):
    # print("Random across whole dataframe", _cosine_similarity_df(df_cultures).mean())
    similarity_random = _cosine_similarity_df(df_.sample(frac=1))
    similarity_batch_random = np.array(
        [
            _cosine_similarity_df(
                df_[df_.index.get_level_values("batch") == batch]
            ).mean()
            for batch in df_.index.get_level_values("batch").unique()
        ]
    ).flatten()
    similarity_same_day = np.array(
        [
            _cosine_similarity_df(
                df_[df_.index.get_level_values(column) == day]
            ).mean()
            for day in df_.index.get_level_values(column).unique()
        ]
    ).flatten()
    similarity_batch_day = np.array(
        [
            [
                _cosine_similarity_df(
                    df_[
                        (df_.index.get_level_values(column) == day)
                        & (df_.index.get_level_values("batch") == batch)
                    ]
                ).mean()
                for batch in df_.index.get_level_values("batch").unique()
            ]
            for day in df_.index.get_level_values(column).unique().sort_values()
        ]
    )
    with warnings.catch_warnings(action="ignore"):
        similarity_notbatch_day = np.array(
            [
                [
                    np.nanmean(np.array([
                        _cosine_similarity_df(
                            df_[
                                (df_.index.get_level_values(column) == day)
                                & (df_.index.get_level_values("batch") == batch)
                            ],
                            df_[
                                (df_.index.get_level_values(column) == day)
                                & (df_.index.get_level_values("batch") == batch2)
                            ],
                        ).mean()
                        for batch2 in df_.index.get_level_values("batch").unique() if batch2 != batch
                    ]))
                    for batch in df_.index.get_level_values("batch").unique()
                ]
                for day in df_.index.get_level_values(column).unique().sort_values()
            ]
        )
    consecutive_day_similarity = np.array([
        _cosine_similarity_df(
            df_[df_.index.get_level_values(column) == day],
            df_[df_.index.get_level_values(column) == next_day]
        ).mean()
        for day, next_day in zip(
            sorted(df_.index.get_level_values(column).unique())[:-1],
            sorted(df_.index.get_level_values(column).unique())[1:]
        )
    ])

    unique_days = sorted(df_.index.get_level_values(column).unique())
    n_days = len(unique_days)
    batch_culture_pairs = df_.groupby(['batch', 'culture']).groups.keys()

    # Initialize an array filled with NaNs to ensure correct shape
    next_existing_day_similarity = np.full((n_days, len(batch_culture_pairs)), np.nan)

    # Populate similarity values
    for idx, (batch, culture) in enumerate(batch_culture_pairs):
        # Select data for the current batch and culture
        group = df_[(df_.index.get_level_values("batch") == batch) &
                    (df_.index.get_level_values("culture") == culture)]
        days = sorted(group.index.get_level_values(column).unique())

        # Compute similarity to the next existing day within this batch and culture
        for i, (day, next_day) in enumerate(zip(days[:-1], days[1:])):
            day_index = unique_days.index(day)
            similarity = _cosine_similarity_df(
                group[group.index.get_level_values(column) == day],
                group[group.index.get_level_values(column) == next_day]
            ).mean()
            next_existing_day_similarity[day_index, idx] = similarity

    fig, ax = plt.subplots(figsize=(7 * cm, 5 * cm), constrained_layout=True)
    sns.despine()
    sns.violinplot(
        data=[
            similarity_random,
            similarity_batch_random,
            similarity_same_day,
            similarity_batch_day.flatten(),
            similarity_notbatch_day.flatten(),
            consecutive_day_similarity,
            next_existing_day_similarity.flatten(),
        ],
        ax=ax,
    )
    ax.set_ylabel("Similarity")
    fig.show()

    match column:
        case "day":
            time_range = np.array(df_.index.get_level_values(column).unique().sort_values())  #  np.arange(7, 35)
        case "week":
            time_range = np.arange(1, 5)
        case _:
            raise NotImplementedError(f"Unknown column: {column}")
    fig, ax = plt.subplots(figsize=(10 * cm, 7 * cm), constrained_layout=True)
    sns.despine()
    ax.plot(
        time_range,
        similarity_batch_day,
        color="black",
        alpha=0.5,
        linewidth=1,
    )
    ax.plot(
        time_range,
        np.nanmean(similarity_batch_day, axis=1),
        color="black",
        linewidth=3,
        label="within batches"
    )
    ax.axhline(
        similarity_random.mean(),
        label="random",
    )
    ax.plot(
        time_range,
        similarity_notbatch_day,
        color="red",
        alpha=0.5,
        linewidth=1,
        linestyle=":"
    )
    ax.plot(
        time_range,
        np.nanmean(similarity_notbatch_day, axis=1),
        color="red",
        linewidth=3,
        linestyle=":",
        label="between batches",
    )
    ax.plot(
        (time_range[1:] + time_range[:-1]) / 2,
        np.nanmean(next_existing_day_similarity[:-1, :], axis=1),
        color="green",
        linewidth=3,
        linestyle="--",
        label=f"{column}-to-{column}",
    )
    ax.legend(frameon=False)
    ax.set_ylabel("Similarity")
    ax.set_xlabel(column)
    fig.show()
    fig.savefig(os.path.join(fig_path, f"similarity_{column}.svg"))
    fig.savefig(os.path.join(fig_path, f"similarity_{column}.pdf"))

print("Finished.")
