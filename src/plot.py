import met_brewer
from matplotlib import pyplot as plt


def get_cluster_colors(n_clusters):
    if n_clusters <= 7:
        austria_custom = met_brewer.met_brew("Austria", 7, brew_type="discrete")
        # austria_custom = [austria_custom[i] for i in [1, 4, 6, 0, 2, 3, 5][:n_clusters]]
        return austria_custom
    else:
        return met_brewer.met_brew("Austria", n_clusters, brew_type="continuous")


def make_cluster_legend(n_clusters, n_cols, symbol):
    fig, ax = plt.subplots(constrained_layout=True)
    cluster_colors = get_cluster_colors(n_clusters)

    handles = []
    labels = [f"{i+1}" for i in range(n_clusters)]

    for i in range(n_clusters):
        if symbol == "line":
            (handle,) = ax.plot(
                [], [], color=cluster_colors[i], label=labels[i], linewidth=2
            )
        elif symbol == "dot":
            handle = ax.scatter([], [], color=cluster_colors[i], label=labels[i], s=50)
        handles.append(handle)

    ax.legend(
        handles=handles,
        labels=labels,
        ncol=n_cols,
        loc="center",
        frameon=False,
        handletextpad=0.1,
        columnspacing=0.3,
        title="Cluster",
    )
    ax.axis("off")  # Hide axes since it's just a legend

    return fig, ax
