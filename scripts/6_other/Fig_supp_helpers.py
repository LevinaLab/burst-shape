import os

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.stats import kurtosis, skew, spearmanr

# Loaders / path helpers / savefig from the project package so the notebooks
# resolve files relative to the repo root regardless of the working directory.
# load_df_cultures and savefig are re-exported for (imported by name in) the
# notebooks.
from burst_shape.folders import get_results_folder
from burst_shape.persistence.burst_extraction import (
    load_burst_matrix,
    load_df_bursts,
    load_df_cultures,  # noqa: F401
)
from burst_shape.persistence.spike_times import get_spike_times_in_seconds
from burst_shape.plot import prepare_plotting, savefig  # noqa: F401

na = np.array

cmap = sns.color_palette("viridis", as_cmap=True)
norm = mcolors.Normalize(vmin=1, vmax=50)

# Create a dummy ScalarMappable for the colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Required for colorbar


def gid_to_numbers(gid):
    """Convert the gid to numbers"""
    for i, u_id in enumerate(np.unique(gid)):
        gid[gid == u_id] = i
    return gid


def build_palette(cmap, uniq):
    """
    Map sorted unique values to colors. `cmap` may be a matplotlib
    colormap/callable (sampled evenly across the unique values) or a dict
    mapping specific values directly to colors (e.g. {'Control': '#4289bf',
    'MELAS': '#469f5b'}); values missing from the dict fall back to gray.
    """
    sorted_uniq = sorted(uniq)
    if isinstance(cmap, dict):
        palette = [cmap.get(a, "0.7") for a in sorted_uniq]
    else:
        palette = [
            cmap(float(i) / max(len(sorted_uniq) - 1, 1))
            for i in range(len(sorted_uniq))
        ]
    arg_to_idx = {a: i for i, a in enumerate(sorted_uniq)}
    return palette, arg_to_idx


def _spikes_and_window(dff, cult_orig, n, keys_in, dataset):
    """Locate the culture row matching row `n` of `dff` and return
    (spike_times_s, gid, xmin, xmax) for that burst.

    Spike times are loaded via the project loader, which transparently handles
    raw on-disk spike files (e.g. wagenaar) and in-memory columns (others).
    The burst window is taken from the burst table's own ``start_orig`` /
    ``end_orig`` (robust across datasets, unlike indexing the per-culture
    ``burst_start_end`` by ``i_burst``).
    """
    if len(keys_in) > 0:
        values = [dff.iloc[n][k] for k in keys_in]
        masks = [
            np.array(cult_orig[k] == v) for k, v in zip(keys_in, values, strict=False)
        ]
        mask = np.prod(np.vstack(masks), axis=0).astype(bool)
    else:
        mask = np.ones(len(cult_orig), dtype=bool)
    idx = int(np.flatnonzero(mask)[0])
    if dataset is None:
        # in-memory spike columns (mossink / hommersom / inhibblock)
        st = cult_orig.iloc[idx]["times"]
        gid = cult_orig.iloc[idx]["gid"]
    else:
        # dataset-aware loader handles raw on-disk spikes (e.g. wagenaar)
        st, gid = get_spike_times_in_seconds(cult_orig, idx, dataset)
    xmin, xmax = dff.iloc[n]["start_orig"], dff.iloc[n]["end_orig"]
    return na(st), na(gid), xmin, xmax


def plot_embeddings(
    dff,
    cult_orig,
    ns,
    extra_ns,
    inset_on=False,
    keys=None,
    x_window=None,
    sc_scale=2,
    inset_offset=(0.1, 0.4, 0.7),  # example positions along x for 3 insets
    cmap=None,
    color_by="argmax",
    dataset=None,
):
    """
    Plot neuronal embedding coordinates with associated burst raster/histogram insets.

    This function creates a composite figure showing:
    (1) raster/histogram plots of neuronal bursts for a selection of indices (`ns`);
    (2) a 2D scatterplot of embedding coordinates (x0, x1) colored by cluster
        label (`argmax`);
    (3) optional inset plots for additional indices (`extra_ns`) embedded
        inside the main scatter.

    Parameters
    ----------
    dff : pandas.DataFrame
        DataFrame containing embedding and burst information.
        Must include at least the columns:
        - 'x0', 'x1' : float
            embedding coordinates
        - 'argmax' : int
            cluster assignment (used for color)
        - 'i_burst' : int
            burst index within the culture
        Additional columns can be used for matching keys.
    cult_orig : pandas.DataFrame or dict-like
        Structure containing per-culture burst data, expected keys:
        - 'times' : array-like or object column with spike times
        - 'gid' : array-like with neuron identifiers
        - 'burst_start_end' : array-like of (start, end) times for bursts
        and any matching columns listed in `keys`.
    ns : list of int
        Indices into `dff` to plot as left-side burst rasters/histograms
        and to highlight on the embedding scatter.
    extra_ns : list of int
        Additional indices into `dff` to highlight on the embedding scatter
        and (if `inset_on=True`) to display as inset burst rasters/histograms.
    inset_on : bool, default=False
        If True, create inset axes on the embedding plot for each index in `extra_ns`.
    keys : list of str, optional
        List of metadata keys used to match rows of `dff` to rows of `cult_orig`.
        If None, no metadata matching is performed (all rows are considered).
    x_window : dict, default={'left': 1000, 'extra': 2000}
        Time windows (in ms) for plotting histograms:
        - 'left' : width for left-side panels
        - 'extra' : width for inset panels
    sc_scale : float, default=2
        Scaling factor applied to histogram counts before plotting.
    inset_offset : float or sequence, default=(0.1, 0.4, 0.7)
        Position offsets for inset axes inside the main embedding plot.
        If scalar, reused for all insets; if sequence, each entry applies
        to the corresponding element of `extra_ns`.
    cmap : matplotlib colormap or callable, optional
        Colormap used to map cluster indices (`argmax`) to colors.
        If None, defaults to 'viridis'.
    color_by : str, default='argmax'
        Column used to color the embedding scatter and the burst raster
        lines. Looked up in `dff` first; if not found there, falls back
        to a per-culture column in `cult_orig` (e.g. 'group'), matched
        onto `dff` via `keys`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure instance.

    Notes
    -----
    - Requires seaborn, matplotlib, and numpy.
    - The `cult_orig` structure must support boolean masking and `.item()` access
      for nested objects, as produced by pandas with object columns.
    - Colors are assigned consistently across subplots using cluster labels.

    Examples
    --------
    >>> fig = plot_embeddings(dff, cult_orig, ns=[0,1,2], extra_ns=[10,11],
    ...                       inset_on=True, keys=['drug_label','div'])
    >>> fig.show()
    """

    # --- safety / defaults ---
    if keys is None:
        keys = []
    if x_window is None:
        x_window = {"left": 1000, "extra": 2000}
    # limit to keys that truly exist in cult_orig
    keys_in = [k for k in keys if k in getattr(cult_orig, "keys", lambda: [])()]

    # color_by may be a per-culture column (e.g. 'group') that only lives in
    # cult_orig rather than dff -- pull it in via the matching keys so the
    # rest of the function can treat it like any other dff column
    if (
        color_by not in dff.columns
        and keys_in
        and color_by in getattr(cult_orig, "keys", lambda: [])()
    ):
        lut = cult_orig[keys_in + [color_by]].drop_duplicates(subset=keys_in)
        dff = dff.merge(lut, on=keys_in, how="left").reset_index(drop=True)

    # consistent colormap: callable for lines AND list for seaborn palette
    if cmap is None:
        cmap = plt.get_cmap("viridis")
    # build a palette for seaborn based on number of unique hues in dff['argmax']
    if color_by in dff:
        uniq = np.unique(dff[color_by])
        palette, arg_to_idx = build_palette(cmap, uniq)
    else:
        palette = None
        arg_to_idx = {}

    # --- layout ---
    fig = plt.figure(figsize=(8, 5), dpi=100)
    gs = gridspec.GridSpec(nrows=4, ncols=15, figure=fig, wspace=0.3, hspace=0.1)

    # --- left panels: signals ---
    last_ax = None
    for i, n in enumerate(ns):
        ax = fig.add_subplot(gs[i, :5])
        last_ax = ax

        # locate the matching culture row and pull its spikes / burst window
        st, gid, xmin, xmax = _spikes_and_window(dff, cult_orig, n, keys_in, dataset)

        sc, bins = np.histogram(
            na(st) * 1000, np.arange(xmin - 200, xmin + x_window["left"], 10)
        )

        color_idx = arg_to_idx.get(dff[color_by].iloc[n], 0)
        ax.plot(
            na(st) * 1000,
            gid_to_numbers(na(gid)),
            "|",
            color=palette[color_idx] if palette is not None else "C0",
            alpha=0.4,
        )

        ax.plot(bins[1:], sc / sc_scale, "-", color="k")
        ax.set_xlim([xmin - 200, xmin + x_window["left"]])
        ax.plot([xmin - 200, xmin], [-1, -1], "k")
        ax.axis("off")

    # --- right: embedding scatter ---
    ax_main = fig.add_subplot(gs[:, 5:])
    sns.scatterplot(
        x="x0",
        y="x1",
        data=dff,
        hue=color_by if color_by in dff else None,
        s=3,
        palette=palette,
        legend=False,
        ax=ax_main,
    )

    # overlay selected points
    ax_main.plot(dff.loc[ns, "x0"], dff.loc[ns, "x1"], "o", mec="none", mfc="r", ms=4)
    ax_main.plot(
        dff.loc[extra_ns, "x0"], dff.loc[extra_ns, "x1"], "^", mec="none", mfc="r", ms=4
    )
    ax_main.axis("off")
    # --- insets on the embedding ---
    if inset_on and len(extra_ns) > 0:
        if last_ax is None:
            raise ValueError("No left subplots were created; cannot size insets.")
        fig.canvas.draw()  # ensure correct pixel sizes
        bbox = last_ax.get_window_extent()
        width_in = bbox.width / fig.dpi
        height_in = bbox.height / fig.dpi

        # normalize inset_offset to a sequence of x positions
        if not hasattr(inset_offset, "__len__"):
            inset_offset = [inset_offset] * len(extra_ns)

        for i, n in enumerate(extra_ns):
            # place each inset at a different x (and descending y) inside ax_main
            ax_in = inset_axes(
                ax_main,
                width=width_in,
                height=height_in,
                loc="upper right",
                bbox_to_anchor=(inset_offset[i], 0.05 - 0.3 * i, 1, 1),
                bbox_transform=ax_main.transAxes,
                borderpad=0,
            )

            st, gid, xmin, xmax = _spikes_and_window(
                dff, cult_orig, n, keys_in, dataset
            )

            sc, bins = np.histogram(
                na(st) * 1000, np.arange(xmin - 200, xmin + x_window["extra"], 10)
            )
            color_idx = arg_to_idx.get(dff[color_by].iloc[n], 0)
            ax_in.plot(
                na(st) * 1000,
                gid_to_numbers(na(gid)),
                "|",
                color=palette[color_idx] if palette is not None else "C0",
                alpha=0.4,
            )

            ax_in.plot(bins[1:], sc / sc_scale, "-", color="k")
            ax_in.set_xlim([xmin - 200, xmin + x_window["extra"]])
            ax_in.plot([xmin - 200, xmin], [-1, -1], "k")
            sns.despine(ax=ax_in)
            ax_in.axis("off")

    return fig


def plot_burst_examples(
    dff,
    cult_orig,
    ns,
    keys=None,
    t_pre=200,
    t_post=1000,
    figsize=None,
    cmap=None,
    color_by="argmax",
    dataset=None,
):
    """
    Plot a raster and spike-count histogram on twin y-axes of the same
    panel, for each index in `ns` -- same data lookup and coloring as
    plot_embeddings, but without the embedding scatter or insets.
    `t_pre`/`t_post` control how much time (ms) is shown before/after the
    start of the burst. See plot_embeddings for the meaning of the other
    parameters.
    """
    if keys is None:
        keys = []
    keys_in = [k for k in keys if k in getattr(cult_orig, "keys", lambda: [])()]

    if (
        color_by not in dff.columns
        and keys_in
        and color_by in getattr(cult_orig, "keys", lambda: [])()
    ):
        lut = cult_orig[keys_in + [color_by]].drop_duplicates(subset=keys_in)
        dff = dff.merge(lut, on=keys_in, how="left").reset_index(drop=True)

    # consistent figure styling (font, line widths, dpi, ...)
    cm = prepare_plotting()

    if cmap is None:
        cmap = plt.get_cmap("viridis")
    if color_by in dff:
        uniq = np.unique(dff[color_by])
        palette, arg_to_idx = build_palette(cmap, uniq)
    else:
        palette = None
        arg_to_idx = {}

    if figsize is None:
        # 5.5 cm wide (17.4 cm / 3), height preserves the previous 5:1.6*n aspect
        width_cm = 5.5
        height_cm = width_cm * (1.6 * len(ns)) / 5
        figsize = (width_cm * cm, height_cm * cm)
    # raster markers were sized for the previous ~5-inch-wide figure; scale them
    # by the figure rescale factor so they keep the same relative size. For the
    # "|" marker, markersize sets the height and markeredgewidth the width, so
    # both must be scaled to shrink the marker proportionally.
    raster_scale = figsize[0] / 5.0
    raster_ms = plt.rcParams["lines.markersize"] * raster_scale
    raster_mew = plt.rcParams["lines.markeredgewidth"] * raster_scale
    line_lw = plt.rcParams["lines.linewidth"] * raster_scale
    fig, axes = plt.subplots(
        len(ns),
        1,
        figsize=figsize,
        squeeze=False,
        sharex=True,
        constrained_layout=True,
    )

    for i, n in enumerate(ns):
        ax = axes[i, 0]
        ax2 = ax.twinx()

        st, gid, xmin, xmax = _spikes_and_window(dff, cult_orig, n, keys_in, dataset)

        st_rel = na(st) * 1000 - xmin
        sc, bins = np.histogram(st_rel, np.arange(-t_pre, t_post, 10))

        color_idx = arg_to_idx.get(dff[color_by].iloc[n], 0)
        color = palette[color_idx] if palette is not None else "C0"

        ax.plot(
            st_rel,
            gid_to_numbers(na(gid)),
            "|",
            color=color,
            alpha=0.4,
            markersize=raster_ms,
            markeredgewidth=raster_mew,
        )
        ax.set_xlim([-t_pre, t_post])
        # shared x axes: only label/number the bottom panel
        if i == len(ns) - 1:
            ax.set_xlabel("Time from burst start (ms)")
        ax.set_ylabel("Electrodes")
        sns.despine(ax=ax)

        ax2.plot(bins[1:], sc, "-", color="k", linewidth=line_lw)
        # ax2.plot([-t_pre, 0], [0, 0], color="k", linewidth=line_lw)
        ax2.set_ylabel("Spike count")
        sns.despine(ax=ax2, left=True, right=False)
        # despine drops the right-side ticks on the twin axis; restore them
        ax2.tick_params(axis="y", which="both", right=True, labelright=True)

    return fig


def spearman_ignore_nan(x, y):
    """Calculate Spearman correlation using only finite pairs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x_valid = x[valid]
    y_valid = y[valid]

    # Correlation is undefined with too few values or a constant variable
    if len(x_valid) < 3 or np.unique(x_valid).size < 2 or np.unique(y_valid).size < 2:
        return np.nan, np.nan

    return spearmanr(x_valid, y_valid)


def get_group_colors(dataset):
    match dataset:
        case "inhibblock":
            return get_inhibblock_colors()
        case "wagenaar":
            return get_wagenaar_colors()
        case "kapucu":
            return get_kapucu_colors()
        case "mossink" | "mossink_KS" | "mossink_MELAS":
            return get_mossink_colors()
        case "hommersom":
            return get_hommersom_colors()
        case "hommersom_binary":
            return get_hommersom_binary_colors()
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
        "Rat": "#03fcdf",
        "hPSC": "#fc9d03",
    }


def get_mossink_colors():
    control_colors = plt.cm.Blues(np.linspace(0.4, 0.9, 10))
    melas_colors = plt.cm.Greens(np.linspace(0.4, 0.9, 3))
    ks_colors = plt.cm.Reds(np.linspace(0.4, 0.9, 4))
    label_color_dict = {}

    def _rgba_to_hex(color):
        return "#{:02x}{:02x}{:02x}".format(
            int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
        )

    for i, color in enumerate(control_colors, start=1):
        label = f"Control {i}"
        label_color_dict[label] = _rgba_to_hex(color)
    for i, color in enumerate(melas_colors, start=1):
        label = f"MELAS {i}"
        label_color_dict[label] = _rgba_to_hex(color)
    for i, color in enumerate(ks_colors, start=1):
        label = f"KS {i}"
        label_color_dict[label] = _rgba_to_hex(color)

    def _average_color(colors):
        avg = np.mean(colors[:, :3], axis=0)  # exclude alpha channel if present
        return _rgba_to_hex((*avg, 1.0))

    label_color_dict["Control"] = _average_color(control_colors)
    label_color_dict["MELAS"] = _average_color(melas_colors)
    label_color_dict["KS"] = _average_color(ks_colors)
    return label_color_dict


def get_hommersom_colors():
    # label_color_dict = {
    #     "Control": "#0fff00",  # "#d62728",  # "#7f7f7f",
    #     "CACNA1A": "#1A75A1",
    #     "Other": "#DAA520",  # "#7f7f7f",
    # }
    label_color_dict = get_hommersom_binary_colors()
    label_color_dict["Other"] = "#0fff00"
    return label_color_dict


def get_hommersom_binary_colors():
    label_color_dict = {
        "Control": "#7f7f7f",  # "#0fff00",  # "#d62728",
        "CACNA1A": "#C87533",  # "#DAA520",  # "#1A75A1",
    }
    return label_color_dict


def fwhm(x, y):
    peak_height = y.max()
    half_height = peak_height / 2
    above = x[y >= half_height]
    return above[-1] - above[0]


def bim_coeff(x):
    return (skew(x, bias=False) ** 2 + 1) / kurtosis(x, fisher=False, bias=False)


def rise_decay_model(t, baseline, amplitude, t0, tau_rise, tau_decay):
    t = np.asarray(t, dtype=float)
    dt = np.maximum(t - t0, 0.0)

    return baseline + (
        amplitude * (1.0 - np.exp(-dt / tau_rise)) * np.exp(-dt / tau_decay)
    )


def decay_model(t, baseline, amplitude, tau_decay):
    t = np.asarray(t, dtype=float)
    dt = t - t[0]
    return baseline + amplitude * np.exp(-dt / tau_decay)


def fit_rise_decay(t, y, min_rise_points=3):
    """
    Fit a rise-decay curve. If the peak occurs too close to the beginning,
    fit a decay-only model and return tau_rise = NaN.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(t) & np.isfinite(y)
    t = t[valid]
    y = y[valid]

    if len(t) < 6:
        raise ValueError("At least 6 finite observations are required.")

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    unique_t, unique_indices = np.unique(t, return_index=True)
    t = unique_t
    y = y[unique_indices]

    if len(t) < 6:
        raise ValueError("At least 6 unique time points are required.")

    time_span = t[-1] - t[0]
    if time_span <= 0:
        raise ValueError("Time values must span a non-zero interval.")

    dt_values = np.diff(t)
    minimum_tau = max(np.min(dt_values) * 0.1, np.finfo(float).eps)

    peak_index = int(np.argmax(y))
    peak_time = t[peak_index]

    n_baseline = max(2, min(len(y) // 10, peak_index))

    if n_baseline >= 2:
        baseline_guess = np.median(y[:n_baseline])
    else:
        # No reliable pre-peak baseline
        baseline_guess = np.median(y[-max(3, len(y) // 10) :])

    peak_height = y[peak_index] - baseline_guess

    if peak_height <= 0:
        peak_height = np.ptp(y)

    if peak_height <= 0:
        raise ValueError("The signal is constant or has no positive peak.")

    # Case 1: insufficient rising phase
    if peak_index < min_rise_points:
        t_decay = t[peak_index:]
        y_decay = y[peak_index:]

        if len(t_decay) < 4:
            raise ValueError("Too few samples after the peak to fit decay.")

        decay_span = t_decay[-1] - t_decay[0]

        p0 = [
            np.median(y_decay[-max(2, len(y_decay) // 10) :]),
            max(y_decay[0] - y_decay[-1], np.ptp(y_decay)),
            max(decay_span / 3, minimum_tau),
        ]

        lower_bounds = [
            -np.inf,
            0.0,
            minimum_tau,
        ]

        upper_bounds = [
            np.inf,
            np.inf,
            max(decay_span * 100, minimum_tau * 10),
        ]

        parameters, covariance = curve_fit(
            decay_model,
            t_decay,
            y_decay,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=50_000,
        )

        baseline, amplitude, tau_decay = parameters
        fitted = decay_model(t_decay, *parameters)
        residuals = y_decay - fitted

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_decay - np.mean(y_decay)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return {
            "parameters": {
                "baseline": baseline,
                "amplitude": amplitude,
                "t0": peak_time,
                "tau_rise": np.nan,
                "tau_decay": tau_decay,
            },
            "fit_type": "decay_only",
            "t": t_decay,
            "fitted": fitted,
            "residuals": residuals,
            "covariance": covariance,
            "r_squared": r_squared,
            "peak_index": peak_index,
        }

    # Case 2: full rise-decay fit
    threshold = baseline_guess + 0.1 * peak_height
    crossings = np.flatnonzero(y[: peak_index + 1] >= threshold)

    if len(crossings):
        t0_guess = t[crossings[0]]
    else:
        t0_guess = t[0]

    # t0 must be strictly before the peak
    t0_upper = peak_time - minimum_tau

    if t0_upper <= t[0]:
        t0_upper = np.nextafter(peak_time, -np.inf)

    t0_guess = np.clip(
        t0_guess,
        t[0] + minimum_tau * 0.01,
        t0_upper - minimum_tau * 0.01,
    )

    tau_rise_guess = max((peak_time - t0_guess) / 2, minimum_tau)

    tau_decay_guess = max(
        (t[-1] - peak_time) / 3,
        tau_rise_guess * 2,
        minimum_tau,
    )

    p0 = [
        baseline_guess,
        max(peak_height * 2, np.ptp(y)),
        t0_guess,
        tau_rise_guess,
        tau_decay_guess,
    ]

    lower_bounds = [
        -np.inf,
        0.0,
        t[0],
        minimum_tau,
        minimum_tau,
    ]

    upper_bounds = [
        np.inf,
        np.inf,
        t0_upper,
        time_span * 100,
        time_span * 100,
    ]

    parameters, covariance = curve_fit(
        rise_decay_model,
        t,
        y,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=50_000,
    )

    fitted = rise_decay_model(t, *parameters)
    residuals = y - fitted

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    names = [
        "baseline",
        "amplitude",
        "t0",
        "tau_rise",
        "tau_decay",
    ]

    return {
        "parameters": dict(zip(names, parameters, strict=False)),
        "standard_errors": dict(zip(names, np.sqrt(np.diag(covariance)), strict=False)),
        "fit_type": "rise_decay",
        "t": t,
        "fitted": fitted,
        "residuals": residuals,
        "covariance": covariance,
        "r_squared": r_squared,
        "peak_index": peak_index,
    }


def weighted_skewness(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(t) & np.isfinite(y) & (y >= 0)
    t = t[valid]
    y = y[valid]
    if y.sum() == 0:
        return np.nan
    w = y / y.sum()
    mean = np.sum(w * t)
    variance = np.sum(w * (t - mean) ** 2)
    if variance == 0:
        return np.nan
    return np.sum(w * (t - mean) ** 3) / variance**1.5


def extract_features(df):
    """Add fwhm, bc (bimodality coefficient), tau_rise, tau_decay derived from
    each burst waveform."""
    xs = np.arange(len(df.iloc[0]["burst"]))
    widths, bims, rises, decays, act80perc, skewnesses = [], [], [], [], [], []
    for burst in df["burst"]:
        widths.append(fwhm(xs, burst))
        bims.append(bim_coeff(burst))
        result = fit_rise_decay(xs, burst)
        rises.append(result["parameters"]["tau_rise"])
        decays.append(result["parameters"]["tau_decay"])
        act80perc.append(burst[40])
        skewnesses.append(weighted_skewness(np.arange(len(burst)), burst))
    df["fwhm"] = widths
    df["bc"] = bims
    df["tau_rise"] = na(rises)
    df["tau_decay"] = na(decays)
    df["skewnesses"] = na(skewnesses)
    df["act80perc"] = na(act80perc)
    return df


def load_spectral_embedding_npy(burst_params, embedding_subdir):
    """Load the raw spectral_embedding.npy for a given burst-extraction folder.

    `burst_params` is the result folder name (e.g.
    'burst_dataset_mossink_...'); the absolute path is resolved via the
    project's results folder so it works regardless of working directory.
    """
    return np.load(
        os.path.join(
            get_results_folder(),
            burst_params,
            embedding_subdir,
            "spectral_embedding.npy",
        )
    )


def load_dataset(burst_params, embedding_subdir):
    """Load burst matrix + embedding + per-burst dataframe for one dataset.

    Mirrors the loading done in Fig3.ipynb.

    `burst_params` is the result folder name (not a full path); files are
    resolved through the project loaders / results folder.
    """
    dd = load_burst_matrix(burst_params)
    embedding_coordinates = load_spectral_embedding_npy(burst_params, embedding_subdir)
    df = load_df_bursts(burst_params)

    labels = np.array([df.iloc[i].name[0] for i in range(len(df))])
    df = df.reset_index()
    # Find out where Wells are lost!!
    # print(df.columns)
    filt_dd = na([gaussian_filter1d(d, sigma=1) for d in dd])

    df["batch"] = labels
    df["x0"] = embedding_coordinates[:, 0]
    df["x1"] = embedding_coordinates[:, 1]
    df["argmax"] = np.argmax(filt_dd, 1)
    df = extract_features(df)
    return df
