"""
Check: directly detecting network bursts == traditional approach with bursts per unit

In this work, we detect network bursts by directly
analyzing the flattened network spikes.
Here, we test whether this definition agrees with the traditional approach which
- first determine bursts per unit
- determine network bursts by thresholding the number of units bursting in a time bin
We measure by determining the overlapping times.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from persistence import burst_params_from_str
from persistence.spike_times import get_spike_times_in_milliseconds
from preprocess.burst_detection import MI_bursts
from settings import get_dataset_from_burst_extraction_params

from persistence import load_df_cultures

burst_extraction_params = (
    "burst_dataset_wagenaar_n_bins_50_normalization_integral_min_length_30_min_firing_rate_3162_smoothing_kernel_4"
    # "burst_dataset_inhibblock_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_hommersom_binary_maxISIstart_20_maxISIb_20_minBdur_50_minIBI_100_minSburst_100_n_bins_50_normalization_integral_min_length_30"
    # "burst_dataset_mossink_KS"
    # "burst_dataset_mossink_maxISIstart_50_maxISIb_50_minBdur_100_minIBI_500_minSburst_100_n_bins_50_normalization_integral_min_length_30"
)
dataset = get_dataset_from_burst_extraction_params(burst_extraction_params)
df_cultures = load_df_cultures(burst_extraction_params)

# %% individual units
burst_extraction_params_dict = burst_params_from_str(burst_extraction_params)
n_units = {
    "inhibblock": 12,
    "wagenaar": 59,
    "hommersom_binary": 16,
    "mossink_KS": 12,
    "mossink": 12,
}[dataset]
maxISIstart_unit = burst_extraction_params_dict["maxISIstart"] * np.sqrt(n_units)
maxISIb_unit = burst_extraction_params_dict["maxISIb"] * np.sqrt(n_units)
minBdur_unit = burst_extraction_params_dict["minBdur"]
minIBI_unit = burst_extraction_params_dict["minIBI"] / n_units
minSburst_unit = burst_extraction_params_dict["minSburst"] / n_units

df_cultures["burst_start_end_individual"] = pd.Series(dtype=object)
for index in tqdm(df_cultures.index, "Detect unit bursts"):
    st, gid = get_spike_times_in_milliseconds(df_cultures, index, dataset)
    burst_start_end_individual = {}
    for _unit in np.unique(gid):
        _unit_filter = gid == _unit
        st_unit = st[_unit_filter]
        # detect bursts
        bursts_unit = MI_bursts(
            st_unit,
            maxISIstart=maxISIstart_unit,
            maxISIb=maxISIb_unit,
            minBdur=minBdur_unit,
            minIBI=minIBI_unit,
            minSburst=minSburst_unit,
        )
        burst_start_end_individual[_unit] = bursts_unit
    df_cultures.at[index, "burst_start_end_individual"] = burst_start_end_individual

# %% determine network bursts from overlapping unit bursts
percent_threshold = 0.2
n_units_threshold = n_units * percent_threshold
df_cultures["burst_start_end_network"] = pd.Series(dtype=object)


def _to_intervals(intervals):
    """Convert interval-like input to valid (start, end) tuples with end > start."""
    if intervals is None:
        return []

    if isinstance(intervals, pd.DataFrame):
        values = intervals.to_numpy()
    else:
        values = np.asarray(intervals)

    if values.size == 0:
        return []

    normalized = []

    if values.ndim == 1:
        if values.shape[0] >= 2:
            start, end = values[0], values[1]
            if np.isfinite(start) and np.isfinite(end) and end > start:
                normalized.append((float(start), float(end)))
        return normalized

    for interval in values:
        if len(interval) < 2:
            continue
        start, end = interval[0], interval[1]
        if np.isfinite(start) and np.isfinite(end) and end > start:
            normalized.append((float(start), float(end)))
    return normalized


# Build reconstructed network bursts from unit-level overlap.
for index in tqdm(df_cultures.index, "Detect network bursts from unit overlap"):
    unit_bursts = df_cultures.at[index, "burst_start_end_individual"]
    events = []

    if isinstance(unit_bursts, dict):
        for bursts in unit_bursts.values():
            for start, end in _to_intervals(bursts):
                # Use half-open intervals [start, end): ending units stop being active at end.
                events.append((start, 1))
                events.append((end, -1))

    if not events:
        df_cultures.at[index, "burst_start_end_network"] = np.empty((0, 2), dtype=float)
        continue

    # Sort end events before start events at identical timestamps for [start, end).
    events.sort(key=lambda item: (item[0], 0 if item[1] == -1 else 1))

    active_units = 0
    network_start = None
    network_bursts = []

    for time, delta in events:
        previous_active = active_units
        active_units += delta

        if previous_active <= n_units_threshold < active_units:
            network_start = time
        elif (
            previous_active > n_units_threshold >= active_units
            and network_start is not None
        ):
            network_bursts.append((network_start, time))
            network_start = None

    df_cultures.at[index, "burst_start_end_network"] = np.asarray(
        network_bursts, dtype=float
    )


# %% 2x2 agreement matrix (time spent in each active/inactive combination)
def _merge_intervals(intervals):
    if len(intervals) == 0:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _duration_states_2x2(intervals_a, intervals_b, t_start, t_end):
    """
    Rows correspond to A inactive/active (0/1), columns to B inactive/active (0/1).
    Values are durations in ms.
    """
    matrix = np.zeros((2, 2), dtype=float)
    if t_end <= t_start:
        return matrix

    events = []
    for start, end in intervals_a:
        s = max(start, t_start)
        e = min(end, t_end)
        if e > s:
            events.append((s, "a", 1))
            events.append((e, "a", -1))
    for start, end in intervals_b:
        s = max(start, t_start)
        e = min(end, t_end)
        if e > s:
            events.append((s, "b", 1))
            events.append((e, "b", -1))

    # End events before start events for half-open semantics [start, end).
    events.sort(key=lambda item: (item[0], 0 if item[2] == -1 else 1))

    active_a = 0
    active_b = 0
    prev_time = t_start
    i = 0
    while i < len(events):
        time = events[i][0]
        if time > prev_time:
            matrix[int(active_a > 0), int(active_b > 0)] += time - prev_time
            prev_time = time

        while i < len(events) and events[i][0] == time:
            _, metric, delta = events[i]
            if metric == "a":
                active_a += delta
            else:
                active_b += delta
            i += 1

    if t_end > prev_time:
        matrix[int(active_a > 0), int(active_b > 0)] += t_end - prev_time

    return matrix


# Compare direct bursts with reconstructed network bursts via a time-based 2x2 matrix.
direct_burst_column = "burst_start_end"
if direct_burst_column not in df_cultures.columns:
    raise KeyError(
        f"Expected direct burst column '{direct_burst_column}' in df_cultures."
    )

df_cultures["agreement_time_2x2"] = pd.Series(dtype=object)

for index in tqdm(df_cultures.index, "Compute 2x2 agreement matrix"):
    intervals_direct = _merge_intervals(
        _to_intervals(df_cultures.at[index, direct_burst_column])
    )
    intervals_overlap = _merge_intervals(
        _to_intervals(df_cultures.at[index, "burst_start_end_network"])
    )

    st, _ = get_spike_times_in_milliseconds(df_cultures, index, dataset)
    if isinstance(st, np.ndarray) and st.size > 0:
        t_start = 0.0
        t_end = float(np.max(st))
    else:
        # Fallback: if spike times are unavailable, restrict to interval support only.
        all_ends = [end for _, end in intervals_direct + intervals_overlap]
        t_start = 0.0
        t_end = max(all_ends) if len(all_ends) else 0.0

    agreement_matrix = _duration_states_2x2(
        intervals_direct,
        intervals_overlap,
        t_start,
        t_end,
    )
    df_cultures.at[index, "agreement_time_2x2"] = agreement_matrix

total_agreement_time_2x2 = np.sum(df_cultures["agreement_time_2x2"].to_list(), axis=0)

total_time = total_agreement_time_2x2.sum()
tn, fp = total_agreement_time_2x2[0, 0], total_agreement_time_2x2[0, 1]
fn, tp = total_agreement_time_2x2[1, 0], total_agreement_time_2x2[1, 1]

observed_agreement = (tn + tp) / total_time if total_time > 0 else np.nan
expected_agreement = (
    ((tn + fp) / total_time) * ((tn + fn) / total_time)
    + ((fn + tp) / total_time) * ((fp + tp) / total_time)
    if total_time > 0
    else np.nan
)
kappa_denom = 1.0 - expected_agreement if np.isfinite(expected_agreement) else np.nan
cohen_kappa = (
    (observed_agreement - expected_agreement) / kappa_denom
    if np.isfinite(kappa_denom) and kappa_denom != 0
    else np.nan
)

active_jaccard = tp / (fp + fn + tp) if (fp + fn + tp) > 0 else np.nan
active_dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan

print("\n2x2 time-overlap matrix (ms):")
print("rows: direct burst inactive/active, cols: reconstructed inactive/active")
print(total_agreement_time_2x2)
if total_time > 0:
    print(f"observed agreement: {observed_agreement:.4f}")
    print(f"expected agreement: {expected_agreement:.4f}")
    print(f"Cohen's kappa: {cohen_kappa:.4f}")
    print(f"active-state Jaccard: {active_jaccard:.4f}")
    print(f"active-state Dice: {active_dice:.4f}")
