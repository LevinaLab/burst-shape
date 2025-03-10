import os

import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from src.folders import get_fig_folder
from src.plot import get_group_colors, prepare_plotting

# %% wagenaar
cm = prepare_plotting()
# Get the color mapping
group_colors = get_group_colors("wagenaar")

# Create legend handles
legend_handles = [
    mpatches.Patch(color=color, label=group) for group, color in group_colors.items()
]
legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=color,
        markersize=8,
        label=group,
    )
    for group, color in group_colors.items()
]

# Create figure
fig, ax = plt.subplots(constrained_layout=True, figsize=(4 * cm, 4 * cm))
ax.legend(
    handles=legend_handles,
    loc="center",
    frameon=False,
    ncols=2,
    handletextpad=-0.2,
    columnspacing=0.0,
    title="Batch",
)
ax.axis("off")  # Hide axes

fig.show()
fig.savefig(
    os.path.join(get_fig_folder(), "wagenaar_legend_batch.svg"), transparent=True
)


# %% Kapucu
cm = prepare_plotting()
# Get the color mapping
group_colors = get_group_colors("kapucu")

# Create legend handles
legend_handles = [
    mpatches.Patch(color=color, label=group) for group, color in group_colors.items()
]
legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor=color,
        markersize=8,
        label=group[0],
    )
    for group, color in group_colors.items()
]

# Create figure
fig, ax = plt.subplots(constrained_layout=True, figsize=(4 * cm, 4 * cm))
ax.legend(
    handles=legend_handles,
    loc="center",
    frameon=False,
    ncols=1,
    handletextpad=-0.2,
    columnspacing=0.0,
    title="Group",
)
ax.axis("off")  # Hide axes

fig.show()
fig.savefig(os.path.join(get_fig_folder(), "kapucu_legend_batch.svg"), transparent=True)
