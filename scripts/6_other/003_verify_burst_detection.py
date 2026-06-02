"""
Check: directly detecting network bursts == traditional approach with bursts per unit.

The "direct" reference is whatever burst_extraction_params points to
(typically the default-algorithm MI_bursts on flattened spike trains).

The script runs:
(1) Time-overlap agreement (2x2 + Cohen's kappa + Jaccard + Dice) of the
    direct bursts against per-unit burst detection + simultaneity threshold.
    Runs for ALL datasets.

(2) Same comparison against the canonical SIMMUX algorithm
    (Wagenaar/DeMarse/Potter 2005: per-electrode cores + entourage +
    chain-overlap rule), plus a burst-existence analysis and a
    width-dilution diagnostic. Only runs for the Wagenaar dataset.

Switch dataset by uncommenting one of the burst_extraction_params strings
below. Output is printed to stdout.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from burst_shape.persistence import burst_params_str_to_dict, load_df_cultures
from burst_shape.persistence.spike_times import get_spike_times_in_milliseconds
from burst_shape.preprocess.burst_detection_alternative import (
    _network_bursts_chain,
    _network_bursts_simultaneity,
    network_bursts_from_unit_overlap,
)
from burst_shape.settings import get_dataset_from_burst_extraction_params

# ---------- pick a dataset by uncommenting exactly one line ---------------
burst_extraction_params = (
    "burst_dataset_wagenaar_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"  # noqa: E501
    # "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"  # noqa: E501
    # "burst_dataset_hommersom_binary_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"  # noqa: E501
    # "burst_dataset_mossink_KS"
    # "burst_dataset_mossink_maxISIstart_50_maxISIb_50_minBdur_100_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30"  # noqa: E501
)

DIRECT_BURST_COLUMN = "burst_start_end"

# SIMMUX-specific (only used when dataset == "wagenaar"). Match the post-
# filters of the direct reference (min_firing_rate=3162 Hz, min_length=30 ms)
# so the burst counts here align with what 005a/037 train on.
SIMMUX_UNIT_THRESHOLD = 4
SIMMUX_MIN_FIRING_RATE_HZ = 3162
SIMMUX_MIN_LENGTH_MS = 30

# Per-dataset electrode count (used to scale the simultaneity-rule parameters
# the same way the original analysis did).
N_UNITS_BY_DATASET = {
    "inhibblock": 12,
    "wagenaar": 59,
    "hommersom_binary": 16,
    "mossink_KS": 12,
    "mossink": 12,
}


# ==========================================================================
# helpers
# ==========================================================================


def _to_intervals(intervals):
    """Convert interval-like input to valid (start, end) tuples with end > start."""
    if intervals is None:
        return []
    values = (
        intervals.to_numpy()
        if isinstance(intervals, pd.DataFrame)
        else np.asarray(intervals)
    )
    if values.size == 0:
        return []
    out = []
    if values.ndim == 1:
        if values.shape[0] >= 2:
            s, e = values[0], values[1]
            if np.isfinite(s) and np.isfinite(e) and e > s:
                out.append((float(s), float(e)))
        return out
    for interval in values:
        if len(interval) < 2:
            continue
        s, e = interval[0], interval[1]
        if np.isfinite(s) and np.isfinite(e) and e > s:
            out.append((float(s), float(e)))
    return out


def _merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def _duration_states_2x2(intervals_a, intervals_b, t_start, t_end):
    """Time spent in each (A active/inactive, B active/inactive) state, ms."""
    matrix = np.zeros((2, 2), dtype=float)
    if t_end <= t_start:
        return matrix

    events = []
    for s, e in intervals_a:
        s, e = max(s, t_start), min(e, t_end)
        if e > s:
            events += [(s, "a", 1), (e, "a", -1)]
    for s, e in intervals_b:
        s, e = max(s, t_start), min(e, t_end)
        if e > s:
            events += [(s, "b", 1), (e, "b", -1)]
    # half-open: end events before start events at same time
    events.sort(key=lambda item: (item[0], 0 if item[2] == -1 else 1))

    active_a = active_b = 0
    prev = t_start
    i = 0
    while i < len(events):
        t = events[i][0]
        if t > prev:
            matrix[int(active_a > 0), int(active_b > 0)] += t - prev
            prev = t
        while i < len(events) and events[i][0] == t:
            _, metric, delta = events[i]
            if metric == "a":
                active_a += delta
            else:
                active_b += delta
            i += 1
    if t_end > prev:
        matrix[int(active_a > 0), int(active_b > 0)] += t_end - prev
    return matrix


def _print_agreement_metrics(label, total_matrix):
    total = total_matrix.sum()
    tn_, fp_ = total_matrix[0, 0], total_matrix[0, 1]
    fn_, tp_ = total_matrix[1, 0], total_matrix[1, 1]
    obs = (tn_ + tp_) / total if total > 0 else np.nan
    exp = (
        ((tn_ + fp_) / total) * ((tn_ + fn_) / total)
        + ((fn_ + tp_) / total) * ((fp_ + tp_) / total)
        if total > 0
        else np.nan
    )
    kappa = (
        (obs - exp) / (1.0 - exp) if np.isfinite(exp) and (1.0 - exp) != 0 else np.nan
    )
    jacc = tp_ / (fp_ + fn_ + tp_) if (fp_ + fn_ + tp_) > 0 else np.nan
    dice = (2 * tp_) / (2 * tp_ + fp_ + fn_) if (2 * tp_ + fp_ + fn_) > 0 else np.nan

    print(f"\n=== {label} ===")
    print("2x2 time-overlap matrix (ms):")
    print("rows: direct burst inactive/active, cols: reconstructed inactive/active")
    print(total_matrix)
    if total > 0:
        print(f"observed agreement:   {obs:.4f}")
        print(f"expected agreement:   {exp:.4f}")
        print(f"Cohen's kappa:        {kappa:.4f}")
        print(f"active-state Jaccard: {jacc:.4f}")
        print(f"active-state Dice:    {dice:.4f}")


def _burst_pop_firing_rate_hz(start_ms, end_ms, st_ms):
    """Pooled population firing rate (Hz) inside a burst window."""
    dur_s = (end_ms - start_ms) / 1000.0
    if dur_s <= 0:
        return 0.0
    return int(np.sum((st_ms >= start_ms) & (st_ms < end_ms))) / dur_s


def _count_matched(intervals_a, intervals_b):
    """Count intervals in A that have nonzero overlap with at least one in B."""
    matched = 0
    for a_s, a_e in intervals_a:
        for b_s, b_e in intervals_b:
            if b_s < a_e and b_e > a_s:
                matched += 1
                break
    return matched


def _agreement_per_recording(df, get_reference_intervals):
    """Compute the per-recording 2x2 matrices vs the direct bursts.

    `get_reference_intervals(index, spike_arr) -> list[(start, end)]`
    returns the comparison intervals for that recording.
    """
    total = np.zeros((2, 2), dtype=float)
    for index in tqdm(df.index, "  Compute 2x2 agreement matrix"):
        st_ms, _ = get_spike_times_in_milliseconds(df, index, dataset)
        spike_arr = st_ms if isinstance(st_ms, np.ndarray) and st_ms.size > 0 else None
        intervals_direct = _merge_intervals(
            _to_intervals(df.at[index, DIRECT_BURST_COLUMN])
        )
        intervals_ref = _merge_intervals(
            _to_intervals(get_reference_intervals(index, spike_arr))
        )
        if spike_arr is not None:
            t_end_eval = float(np.max(spike_arr))
        else:
            ends = [e for _, e in intervals_direct + intervals_ref]
            t_end_eval = max(ends) if ends else 0.0
        total += _duration_states_2x2(intervals_direct, intervals_ref, 0.0, t_end_eval)
    return total


# ==========================================================================
# load + dispatch
# ==========================================================================

dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
print(f"Dataset: {dataset}")
print(f"Folder:  {burst_extraction_params}")

df_cultures = load_df_cultures(burst_extraction_params)
if DIRECT_BURST_COLUMN not in df_cultures.columns:
    raise KeyError(
        f"Expected direct burst column '{DIRECT_BURST_COLUMN}' in df_cultures."
    )

if dataset not in N_UNITS_BY_DATASET:
    raise KeyError(
        f"Add {dataset!r} to N_UNITS_BY_DATASET (electrode count for this dataset)."
    )
n_units = N_UNITS_BY_DATASET[dataset]

burst_params = burst_params_str_to_dict(burst_extraction_params)


# ==========================================================================
# (1) direct vs per-unit overlap + simultaneity threshold
# ==========================================================================

# scale the burst-detection parameters per electrode (the original analysis)
maxISIstart_unit = burst_params["maxISIstart"] * np.sqrt(n_units)
maxISIb_unit = burst_params["maxISIb"] * np.sqrt(n_units)
minBdur_unit = burst_params["minBdur"]
minIBI_unit = burst_params["minIBI"]  # / n_units
minSburst_unit = burst_params["minSburst"] / n_units
PERCENT_THRESHOLD = 0.2

df_cultures["burst_start_end_network"] = pd.Series(dtype=object)
for index in tqdm(df_cultures.index, "Detect simultaneity network bursts"):
    st, gid = get_spike_times_in_milliseconds(df_cultures, index, dataset)
    if not (isinstance(st, np.ndarray) and st.size > 0):
        df_cultures.at[index, "burst_start_end_network"] = []
        continue
    df_cultures.at[index, "burst_start_end_network"] = network_bursts_from_unit_overlap(
        st,
        gid,
        maxISIstart=maxISIstart_unit,
        maxISIb=maxISIb_unit,
        minBdur=minBdur_unit,
        minIBI=minIBI_unit,
        minSburst=minSburst_unit,
        threshold=PERCENT_THRESHOLD,
        n_units=n_units,
        # Pin the pre-SIMMUX-defaults behavior; the package defaults later
        # switched to chain + entourage. Pinning keeps this stable.
        network_rule="simultaneity",
        entourage_maxISI=None,
    )

matrix_sim = _agreement_per_recording(
    df_cultures,
    get_reference_intervals=lambda idx, _spk: df_cultures.at[
        idx, "burst_start_end_network"
    ],
)
_print_agreement_metrics(
    "direct (flattened MI_bursts) vs per-unit overlap + simultaneity",
    matrix_sim,
)


# Apply the same post-detection filters (min_length, min_firing_rate) that the
# direct extraction applied. The direct bursts in df_cultures already have them
# applied, so the unfiltered reference above gives an unfair comparison.
_min_length_direct = burst_params.get("min_length")
_min_firing_rate_direct = burst_params.get("min_firing_rate")


def _apply_direct_filters(intervals, spike_arr):
    """Filter intervals by `min_length` / `min_firing_rate` from burst_params."""
    if not intervals:
        return intervals
    if _min_length_direct is not None:
        intervals = [(s, e) for (s, e) in intervals if (e - s) >= _min_length_direct]
    if _min_firing_rate_direct is not None and spike_arr is not None and intervals:
        intervals = [
            (s, e)
            for (s, e) in intervals
            if _burst_pop_firing_rate_hz(s, e, spike_arr) >= _min_firing_rate_direct
        ]
    return intervals


if _min_length_direct is not None or _min_firing_rate_direct is not None:
    matrix_sim_filt = _agreement_per_recording(
        df_cultures,
        get_reference_intervals=lambda idx, spike_arr: _apply_direct_filters(
            df_cultures.at[idx, "burst_start_end_network"], spike_arr
        ),
    )
    _print_agreement_metrics(
        f"direct vs per-unit overlap + simultaneity, apples-to-apples "
        f"(reference filtered to match direct: min_length={_min_length_direct}, "
        f"min_firing_rate={_min_firing_rate_direct} Hz)",
        matrix_sim_filt,
    )


# ==========================================================================
# burst-existence overlap helper (used for all datasets)
# ==========================================================================


def _burst_existence_overlap(get_ref_intervals, label, ref_name="reconstructed"):
    """Report total burst counts and per-burst matching vs the direct reference.

    `get_ref_intervals(index, spike_arr) -> list[(start, end)]`
    returns the comparison intervals for that recording.
    """
    rows = []
    for index in df_cultures.index:
        st_ms, _ = get_spike_times_in_milliseconds(df_cultures, index, dataset)
        spike_arr = st_ms if isinstance(st_ms, np.ndarray) and st_ms.size > 0 else None
        intervals_direct = _merge_intervals(
            _to_intervals(df_cultures.at[index, DIRECT_BURST_COLUMN])
        )
        intervals_ref = _merge_intervals(
            _to_intervals(get_ref_intervals(index, spike_arr))
        )
        rows.append(
            {
                "n_direct": len(intervals_direct),
                "n_ref": len(intervals_ref),
                "matched_direct": _count_matched(intervals_direct, intervals_ref),
                "matched_ref": _count_matched(intervals_ref, intervals_direct),
            }
        )
    df = pd.DataFrame(rows)

    n_d = int(df["n_direct"].sum())
    n_s = int(df["n_ref"].sum())
    m_d = int(df["matched_direct"].sum())
    m_s = int(df["matched_ref"].sum())
    recall = m_d / n_d if n_d > 0 else np.nan
    precision = m_s / n_s if n_s > 0 else np.nan
    f1 = (
        2 * recall * precision / (recall + precision)
        if (recall + precision) > 0
        else np.nan
    )

    both = df[(df["n_direct"] > 0) & (df["n_ref"] > 0)].copy()
    both["recall"] = both["matched_direct"] / both["n_direct"]
    both["precision"] = both["matched_ref"] / both["n_ref"]

    print(f"\n=== Burst-existence overlap: {label} ===")
    print(
        f"  direct bursts total : {n_d:>6}   "
        f"matched by >=1 {ref_name}: {m_d:>6}  ({recall:.3f})  <- recall"
    )
    print(
        f"  {ref_name} bursts total : {n_s:>6}   "
        f"matched by >=1 direct: {m_s:>6}  ({precision:.3f})  <- precision"
    )
    print(f"  burst-existence F1  : {f1:.4f}")
    print(f"  direct bursts MISSED by {ref_name}: {n_d - m_d:>6}")
    print(f"  {ref_name} bursts NOT confirmed   : {n_s - m_s:>6}")
    print(f"  per-recording (both sides nonzero, n={len(both)}):")
    print(
        f"    recall    median {both['recall'].median():.3f}   "
        f"mean {both['recall'].mean():.3f}"
    )
    print(
        f"    precision median {both['precision'].median():.3f}   "
        f"mean {both['precision'].mean():.3f}"
    )
    n_direct_only = int(((df["n_direct"] > 0) & (df["n_ref"] == 0)).sum())
    n_ref_only = int(((df["n_ref"] > 0) & (df["n_direct"] == 0)).sum())
    print(
        f"  recordings where direct has bursts but {ref_name} has none: {n_direct_only}"
    )
    print(f"  recordings where {ref_name} has bursts but direct has none: {n_ref_only}")


_burst_existence_overlap(
    get_ref_intervals=lambda idx, _spk: df_cultures.at[idx, "burst_start_end_network"],
    label="direct vs per-unit overlap + simultaneity",
    ref_name="simultaneity",
)
if _min_length_direct is not None or _min_firing_rate_direct is not None:
    _burst_existence_overlap(
        get_ref_intervals=lambda idx, spike_arr: _apply_direct_filters(
            df_cultures.at[idx, "burst_start_end_network"], spike_arr
        ),
        label=f"direct vs per-unit overlap + simultaneity, apples-to-apples "
        f"(reference filtered to match direct: min_length={_min_length_direct}, "
        f"min_firing_rate={_min_firing_rate_direct} Hz)",
        ref_name="simultaneity",
    )


# ==========================================================================
# (2) canonical SIMMUX comparison (Wagenaar dataset only)
# ==========================================================================

if dataset != "wagenaar":
    print(
        f"\nSIMMUX comparison skipped (only runs for dataset='wagenaar', got {dataset!r})."
    )
    raise SystemExit(0)

# Cache per-electrode SIMMUX burstlets once (the expensive step) for two
# variants: canonical SIMMUX (with entourage extension) and a no-entourage
# variant. The actual saved Wagenaar SIMMUX dataset was extracted with
# entourage_maxISI=None, so reporting both lets us see the discrepancy.
SIMMUX_VARIANTS = {
    # label                            (entourage_maxISI, entourage_cap_ms, network_rule)
    # "chain_with_entourage":             (1 / 3, 200,  "chain"),  # canonical SIMMUX
    # "chain_no_entourage":               (None,  None, "chain"),
    # matches the algorithm actually used to extract the burst dataset that
    # 005a/037 train on (per-unit overlap with simultaneity rule, no entourage):
    "simultaneity_no_entourage": (None, None, "simultaneity"),
}
for variant, (
    entourage_maxISI,
    entourage_cap_ms,
    network_rule,
) in SIMMUX_VARIANTS.items():
    col_name = f"simmux_unit_bursts__{variant}"
    df_cultures[col_name] = pd.Series(dtype=object)
    for index in tqdm(
        df_cultures.index, f"Detect SIMMUX per-electrode burstlets ({variant})"
    ):
        st, gid = get_spike_times_in_milliseconds(df_cultures, index, dataset)
        if not (isinstance(st, np.ndarray) and st.size > 0):
            df_cultures.at[index, col_name] = {}
            continue
        # threshold=0 keeps every chain/simultaneity component; we filter post-hoc.
        _, unit_bursts = network_bursts_from_unit_overlap(
            st,
            gid,
            # SIMMUX core: adaptive per-electrode ISI = min(1/(4 f_c), 100 ms).
            maxISIstart=0.25,
            maxISIb=0.25,
            isi_cap_ms=100,
            minBdur=0,
            minIBI=0,
            minSburst=3,  # >=4 spikes per core
            threshold=0,
            n_units=n_units,
            return_unit_bursts=True,
            entourage_maxISI=entourage_maxISI,
            entourage_cap_ms=entourage_cap_ms,
            network_rule=network_rule,
        )
        df_cultures.at[index, col_name] = unit_bursts


def _simmux_bursts(index, spike_arr, min_firing_rate_hz, variant):
    """Re-derive SIMMUX network bursts from cached per-electrode burstlets.

    Also applies the `SIMMUX_MIN_LENGTH_MS` filter to match the post-detection
    filtering that the saved burst dataset (used by 005a/037) has applied.
    """
    unit_bursts = df_cultures.at[index, f"simmux_unit_bursts__{variant}"]
    if not (isinstance(unit_bursts, dict) and unit_bursts):
        return []
    network_rule = SIMMUX_VARIANTS[variant][2]
    if network_rule == "chain":
        bursts = _network_bursts_chain(
            unit_bursts, n_units_threshold=SIMMUX_UNIT_THRESHOLD
        )
    elif network_rule == "simultaneity":
        bursts = _network_bursts_simultaneity(
            unit_bursts, n_units_threshold=SIMMUX_UNIT_THRESHOLD
        )
    else:
        raise ValueError(f"Unknown network_rule: {network_rule!r}")
    if SIMMUX_MIN_LENGTH_MS is not None and bursts:
        bursts = [(s, e) for (s, e) in bursts if (e - s) >= SIMMUX_MIN_LENGTH_MS]
    if min_firing_rate_hz is not None and spike_arr is not None and bursts:
        bursts = [
            (s, e)
            for (s, e) in bursts
            if _burst_pop_firing_rate_hz(s, e, spike_arr) >= min_firing_rate_hz
        ]
    return bursts


# ---------- (2a) Time-overlap Dice vs direct -----------------------------

for variant in SIMMUX_VARIANTS:
    matrix_simmux = _agreement_per_recording(
        df_cultures,
        get_reference_intervals=lambda idx, spike_arr, v=variant: _simmux_bursts(
            idx, spike_arr, SIMMUX_MIN_FIRING_RATE_HZ, v
        ),
    )
    _print_agreement_metrics(
        f"direct vs SIMMUX ({variant}, unit_threshold={SIMMUX_UNIT_THRESHOLD}, "
        f"min_firing_rate={SIMMUX_MIN_FIRING_RATE_HZ} Hz)",
        matrix_simmux,
    )


# ---------- (2b) Burst-existence overlap (with and without rate filter) --

for variant in SIMMUX_VARIANTS:
    _burst_existence_overlap(
        get_ref_intervals=lambda idx, spk, v=variant: _simmux_bursts(
            idx, spk, SIMMUX_MIN_FIRING_RATE_HZ, v
        ),
        label=f"direct vs SIMMUX ({variant}, "
        f"unit_threshold={SIMMUX_UNIT_THRESHOLD}, "
        f"min_firing_rate={SIMMUX_MIN_FIRING_RATE_HZ} Hz)",
        ref_name="SIMMUX",
    )
    _burst_existence_overlap(
        get_ref_intervals=lambda idx, spk, v=variant: _simmux_bursts(idx, spk, None, v),
        label=f"direct vs SIMMUX ({variant}, "
        f"unit_threshold={SIMMUX_UNIT_THRESHOLD}, NO firing-rate filter)",
        ref_name="SIMMUX",
    )


# ---------- (2c) Width-dilution diagnostic --------------------------------
# Hypothesis: SIMMUX bursts include lower-rate edges/gaps that direct bursts
# don't, so the population firing rate over the SIMMUX window is lower
# than over the direct window, even when both cover the same event.

widths_direct = []
widths_simmux_all = []
pair_records = []

for index in df_cultures.index:
    st_ms, _ = get_spike_times_in_milliseconds(df_cultures, index, dataset)
    spike_arr = st_ms if isinstance(st_ms, np.ndarray) and st_ms.size > 0 else None
    direct = _merge_intervals(_to_intervals(df_cultures.at[index, DIRECT_BURST_COLUMN]))
    simmux = _merge_intervals(
        _to_intervals(
            _simmux_bursts(index, spike_arr, None, "simultaneity_no_entourage")
        )
    )
    widths_direct.extend(e - s for s, e in direct)
    widths_simmux_all.extend(e - s for s, e in simmux)
    if spike_arr is None:
        continue
    for s_s, s_e in simmux:
        overlapping = [(d_s, d_e) for d_s, d_e in direct if d_s < s_e and d_e > s_s]
        if not overlapping:
            continue
        d_s_best, d_e_best = max(overlapping, key=lambda x: x[1] - x[0])
        s_width = s_e - s_s
        d_width = d_e_best - d_s_best
        s_rate = _burst_pop_firing_rate_hz(s_s, s_e, spike_arr)
        d_rate = _burst_pop_firing_rate_hz(d_s_best, d_e_best, spike_arr)
        pair_records.append(
            {
                "simmux_width_ms": s_width,
                "direct_width_ms": d_width,
                "width_ratio": s_width / d_width if d_width > 0 else np.nan,
                "simmux_rate_hz": s_rate,
                "direct_rate_hz": d_rate,
                "rate_ratio": s_rate / d_rate if d_rate > 0 else np.nan,
                "simmux_passes": s_rate >= SIMMUX_MIN_FIRING_RATE_HZ,
                "direct_passes": d_rate >= SIMMUX_MIN_FIRING_RATE_HZ,
            }
        )

df_pair = pd.DataFrame(pair_records)

print("\n=== Width-dilution diagnostic ===")
print(
    f"  direct burst widths (ms)            n={len(widths_direct):>5}  "
    f"median {np.median(widths_direct):>7.0f}   mean {np.mean(widths_direct):>7.0f}"
)
print(
    f"  SIMMUX burst widths (all, ms)       n={len(widths_simmux_all):>5}  "
    f"median {np.median(widths_simmux_all):>7.0f}   "
    f"mean {np.mean(widths_simmux_all):>7.0f}"
)
print()
print(f"For SIMMUX bursts that overlap a direct burst (n={len(df_pair)}):")
print(
    f"  width ratio (SIMMUX/direct)  median {df_pair['width_ratio'].median():.2f}   "
    f"mean {df_pair['width_ratio'].mean():.2f}"
)
print(
    f"  rate  ratio (SIMMUX/direct)  median {df_pair['rate_ratio'].median():.2f}   "
    f"mean {df_pair['rate_ratio'].mean():.2f}"
)

n_both = int((df_pair["simmux_passes"] & df_pair["direct_passes"]).sum())
n_dir_only = int((~df_pair["simmux_passes"] & df_pair["direct_passes"]).sum())
n_sim_only = int((df_pair["simmux_passes"] & ~df_pair["direct_passes"]).sum())
n_neither = int((~df_pair["simmux_passes"] & ~df_pair["direct_passes"]).sum())
print()
print(
    f"  {SIMMUX_MIN_FIRING_RATE_HZ} Hz filter outcome for matched pairs "
    f"(n={len(df_pair)}):"
)
print(f"    both pass        : {n_both:>6}")
print(f"    only direct pass : {n_dir_only:>6}   <- SIMMUX dropped here")
print(f"    only SIMMUX pass : {n_sim_only:>6}")
print(f"    neither pass     : {n_neither:>6}")

df_dropped = df_pair[~df_pair["simmux_passes"] & df_pair["direct_passes"]]
if len(df_dropped) > 0:
    print()
    print(
        f"For the {len(df_dropped)} pairs where SIMMUX was dropped but direct passed:"
    )
    print(
        f"  width ratio   median {df_dropped['width_ratio'].median():.2f}   "
        f"mean {df_dropped['width_ratio'].mean():.2f}   "
        f"max {df_dropped['width_ratio'].max():.2f}"
    )
    print(
        f"  rate ratio    median {df_dropped['rate_ratio'].median():.2f}   "
        f"mean {df_dropped['rate_ratio'].mean():.2f}"
    )
    print(
        f"  SIMMUX rate Hz median {df_dropped['simmux_rate_hz'].median():.0f}   "
        f"direct rate Hz median {df_dropped['direct_rate_hz'].median():.0f}"
    )
