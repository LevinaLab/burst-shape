import met_brewer
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def prepare_plotting():
    plt.rcParams["axes.edgecolor"] = "k"
    plt.rcParams["axes.facecolor"] = "w"
    plt.rcParams["axes.linewidth"] = "0.8"
    plt.rcParams.update({"font.size": 10})
    plt.rcParams["savefig.dpi"] = 300

    plt.rcParams["pdf.fonttype"] = 42  # prepare as vector graphic
    plt.rcParams["ps.fonttype"] = 42

    # plt.rcParams["font.family"] = "Helvetica"

    cm = 1 / 2.54
    return cm


def get_cluster_colors(n_clusters):
    if n_clusters <= 7:
        austria_custom = met_brewer.met_brew("Austria", 7, brew_type="discrete")
        # austria_custom = [austria_custom[i] for i in [1, 4, 6, 0, 2, 3, 5][:n_clusters]]
        return austria_custom[:n_clusters]
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


def get_group_colors(dataset):
    match dataset:
        case "inhibblock":
            return get_inhibblock_colors()
        case "wagenaar":
            return get_wagenaar_colors()
        case "kapucu":
            return get_kapucu_colors()
        case _:
            return None


def get_inhibblock_colors():
    return {
        "bic": "#e377c2",  #  "#ff7f0e",
        "control": "#17becf",  # "#1f77b4",
    }


def get_wagenaar_colors():
    set1_colors = sns.color_palette("Set1", 8).as_hex()
    colors = {i + 1: set1_colors[i] for i in range(8)}
    return colors


def get_kapucu_colors():
    return {
        ("Rat", "MEA1"): "#03fcdf",
        ("hPSC", "MEA1"): "#fc0303",
        ("hPSC", "MEA2"): "#fc9d03",
    }


def label_sig_diff(
    ax, inds, max_data, text_sig, y_sig, length_sig, col_sig, ft_sig, lw_sig
):
    if isinstance(text_sig, float):
        if text_sig < 0.001:
            text_sig = r"***"
        elif text_sig < 0.01:
            text_sig = r"**"
        elif text_sig < 0.05:
            text_sig = r"*"
        else:
            text_sig = "n.s."
    # Custom function to draw the diff bars
    x1, x2 = inds
    y = np.max(max_data) + y_sig
    h = length_sig
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=lw_sig, c=col_sig)
    ax.text(
        (x1 + x2) * 0.5,
        y + h,
        text_sig,
        ha="center",
        va="bottom",
        color=col_sig,
        fontsize=ft_sig,
    )
