"""Alternative burst detection utilities.

This module implements a network burst detection function that
reconstructs network bursts from unit-level bursts by thresholding
the number of simultaneously active units.

The main function `network_bursts_from_unit_overlap` accepts spike
times and gids (unit ids) and uses the existing `MI_bursts` function
for per-unit burst detection. It returns an array of (start, end)
intervals for network bursts.
"""

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .burst_detection import MI_bursts


def _to_intervals(intervals: Optional[Iterable]) -> List[Tuple[float, float]]:
    """Normalize various interval-like inputs to a list of (start, end).

    Returns an empty list if input is None or empty. Filters out invalid
    intervals where end <= start or values are non-finite.
    """
    if intervals is None:
        return []

    values = np.asarray(intervals)
    if values.size == 0:
        return []

    normalized: List[Tuple[float, float]] = []
    if values.ndim == 1:
        if values.shape[0] >= 2:
            start, end = float(values[0]), float(values[1])
            if np.isfinite(start) and np.isfinite(end) and end > start:
                normalized.append((start, end))
        return normalized

    for row in values:
        if len(row) < 2:
            continue
        start, end = float(row[0]), float(row[1])
        if np.isfinite(start) and np.isfinite(end) and end > start:
            normalized.append((start, end))
    return normalized


def network_bursts_from_unit_overlap(
    st: Sequence[float],
    gid: Sequence,
    maxISIstart: float = 4.5,
    maxISIb: float = 4.5,
    minBdur: float = 40,
    minIBI: float = 40,
    minSburst: float = 50,
    threshold: float = 0.2,
    n_units: Optional[int] = None,
) -> np.ndarray:
    """Detect network bursts by overlapping unit-level bursts.

    For each unique unit id in `gid`, we detect bursts using `MI_bursts`
    (with the provided interval parameters). Network bursts are produced
    by thresholding the number (or fraction) of simultaneously active
    units.

    Args:
        st: sequence of spike times in milliseconds (same length as gid).
        gid: sequence of unit ids corresponding to st.
        maxISIstart, maxISIb, minBdur, minIBI, minSburst: passed to
            `MI_bursts` for per-unit burst detection.
        threshold: if in (0,1], treated as fraction of units (percent)
            required to consider the network active; if >1, treated as
            an absolute number of units.
        n_units: optional override for number of units; if None we use
            the number of unique ids in `gid`.

    Returns:
        numpy array of shape (k,2) with dtype float for k detected
        network bursts (start, end). If none detected, returns an
        empty array with shape (0,2).
    """
    st_arr = np.asarray(st)
    gid_arr = np.asarray(gid)

    if st_arr.size == 0:
        return np.empty((0, 2), dtype=float)

    unique_units = np.unique(gid_arr)
    n_units_detected = int(n_units) if n_units is not None else len(unique_units)

    # Interpret threshold: fraction -> absolute count
    if 0.0 < threshold <= 1.0:
        n_units_threshold = n_units_detected * float(threshold)
    else:
        # threshold > 1 interpreted as absolute units
        n_units_threshold = float(threshold)

    # Collect per-unit bursts
    events: List[Tuple[float, int]] = []
    for unit in unique_units:
        mask = gid_arr == unit
        st_unit = st_arr[mask]
        if st_unit.size == 0:
            continue
        bursts_unit = MI_bursts(
            st_unit,
            maxISIstart=maxISIstart,
            maxISIb=maxISIb,
            minBdur=minBdur,
            minIBI=minIBI,
            minSburst=minSburst,
        )
        for start, end in _to_intervals(bursts_unit):
            # half-open interval [start, end): start adds, end removes
            events.append((start, 1))
            events.append((end, -1))

    if not events:
        return np.empty((0, 2), dtype=float)

    # Sort events; end (-1) should be processed before start (+1) at same time
    events.sort(key=lambda item: (item[0], 0 if item[1] == -1 else 1))

    active_units = 0
    prev_active = 0
    network_start: Optional[float] = None
    network_bursts: List[Tuple[float, float]] = []

    for time, delta in events:
        prev_active = active_units
        active_units += delta

        # start when we cross above threshold
        if prev_active <= n_units_threshold < active_units:
            network_start = float(time)
        # end when we drop to threshold or below
        elif (
            prev_active > n_units_threshold >= active_units
            and network_start is not None
        ):
            network_bursts.append((network_start, float(time)))
            network_start = None

    return np.asarray(network_bursts, dtype=float)
