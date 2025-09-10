import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.persistence import load_df_cultures
from src.persistence.spike_times import get_spike_times_in_seconds
from src.plot import prepare_plotting, savefig
from src.settings import get_dataset_from_burst_extraction_params

cm = prepare_plotting()

# parameters which clustering to plot
burst_extraction_params = (
    # "burst_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_dataset_kapucu_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30_min_firing_rate_316_smoothing_kernel_4"
    # "burst_dataset_hommersom_test_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
)
dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
print(f"Detected dataset: {dataset}")

# load bursts
# df_bursts = load_df_bursts(burst_extraction_params)
df_cultures = load_df_cultures(burst_extraction_params)

# %% plot one timeseries example
if dataset == "inhibblock":
    index = ("control", 18, 0)
    xlim = (412, 427)
    bin = 0.01

    index = ("control", 17, 0)
    xlim = (10, 16.5)
    bin = 0.03

# xlim=None
# number_of_spikes_to_plot = 1000
number_of_spikes_to_plot = None

st, gid = get_spike_times_in_seconds(df_cultures, index, dataset)
if xlim is not None:
    selection = (st >= xlim[0]) & (st <= xlim[1])
    st = st[selection]
    gid = gid[selection]

bins = np.arange(st.min(), st.max() + bin, bin)
firing_rate = np.histogram(st, bins=bins)[0] / bin  # in Hz
t_values = (bins[1:] + bins[:-1]) / 2

burst_start_end = df_cultures.at[index, "burst_start_end"]
burst_start_end = [(start / 1000, end / 1000) for start, end in burst_start_end]
if xlim is not None:
    burst_start_end = [
        (start, end)
        for start, end in burst_start_end
        if (start >= xlim[0] and start <= xlim[1])
        or (end >= xlim[0] and end <= xlim[1])
    ]

# traditional features
mean_firing_rate = np.mean(firing_rate)
burst_spike_rate = np.mean(
    [
        np.sum((st >= start) & (st <= end)) / (end - start)
        for start, end in burst_start_end
    ]
)

fig, axs = plt.subplots(
    nrows=2,
    sharex="all",
    constrained_layout=True,
    figsize=(6 * cm, 4 * cm),
    # relative height
    gridspec_kw={"height_ratios": [1, 2]},
)
sns.despine()

ax = axs[0]
sns.despine(ax=ax, left=True, bottom=True)
lineoffset = 0.1
linelength = 0.8
linewidths = 0.2
alpha = 1

if False:
    ax.eventplot(
        st
        if number_of_spikes_to_plot is None
        else np.random.choice(st, size=number_of_spikes_to_plot, replace=False),
        lineoffsets=1,
        linelengths=0.9,
        colors="black",
        linewidths=0.4,
        alpha=1,
    )
else:
    # Build vertices and codes for one Path
    vertices = []
    for t, y_pos in zip(st, gid):
        # Move to bottom of spike
        vertices.append((t, lineoffset + y_pos))
        # Draw to top of spike
        vertices.append((t, lineoffset + y_pos + linelength))

    # Create a single Path and add as a patch
    path = matplotlib.path.Path(
        vertices,
        [
            matplotlib.path.Path.MOVETO if i % 2 == 0 else matplotlib.path.Path.LINETO
            for i in range(len(vertices))
        ],
    )
    patch = matplotlib.patches.PathPatch(
        path, facecolor="none", edgecolor="black", linewidth=linewidths, alpha=alpha
    )
    ax.add_patch(patch)
    ax.set_ylim(gid.min() - 1, gid.max() + 2)

ax.set_xticks([])
ax.set_yticks([])
# ax.set_ylabel("Spikes", rotation=0, labelpad=20)

ax = axs[1]
sns.despine(ax=ax, left=True, bottom=True)
ax.plot(t_values, firing_rate, color="black")
for start, end in burst_start_end:
    selection = (t_values >= start) & (t_values <= end)
    ax.plot(t_values[selection], firing_rate[selection], color="#008000")
ax.axhline(mean_firing_rate, color="#AA0000", linestyle="--", label="mean")
ax.axhline(burst_spike_rate, color="#AA0000", linestyle="--", label="burst spike rate")
# ax.set_xlabel("Time [s]")
# ax.set_ylabel("Firing rate [Hz]")
ax.set_yticks([])

# draw burst duration underneath the plot indicated by brown bars with whiskers
ax_burst = axs[1].inset_axes([0, -0.15, 1, 0.1], sharex=axs[1])
sns.despine(ax=ax_burst, left=True, bottom=True)
ax_burst.set_xticks([])
ax_burst.set_yticks([])
for start, end in burst_start_end[-1:]:
    # Draw burst line
    ax_burst.plot([start, end], [0.5, 0.5], color="#AA0000", lw=1)
    # Draw whiskers
    ax_burst.plot([start, start], [0.3, 0.7], color="#AA0000", lw=1)
    ax_burst.plot([end, end], [0.3, 0.7], color="#AA0000", lw=1)

for start, end in burst_start_end:
    for ax in axs:
        ax.axvspan(start, end, color="grey", alpha=0.3)
fig.show()
savefig(
    fig,
    f"{dataset}_demo_timeseries_burst_features",
    file_format=["pdf", "svg"],
)

# %%
fig, ax = plt.subplots(
    constrained_layout=True,
    figsize=(3.5 * cm, 4 * cm),
)
sns.despine()
matplotlib.rcParams["hatch.linewidth"] = 4
bars = ax.bar(
    [0, 1, 2],
    [0.6, 0.8, 1],
    color=["#AA0000", "#008000", "#008000"],
    fill=True,
    alpha=0.5,
    edgecolor=["#AA0000", "#008000", "#AA0000"],
    hatch=["", "", "//"],
    linewidth=2,
)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Traditional", "Burst shape", "Combined"], rotation=0, ha="center")
label_trad, label_shape, label_combined = ax.get_xticklabels()
label_trad.set_color("#AA0000")
label_shape.set_color("#008000")
label_shape.set_y(-0.14)
ax.set_yticks([])
ax.set_ylabel("Accuracy")
fig.show()
savefig(
    fig,
    f"{dataset}_demo_accuracy_improvement",
    file_format=["pdf", "svg"],
)
