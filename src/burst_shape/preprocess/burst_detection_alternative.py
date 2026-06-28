"""Alternative burst detection utilities.

This module implements per-electrode burstlet detection plus two
network-level burst combination rules:

- "simultaneity" (the original `"overlap"` rule): emit a network burst
  while the count of simultaneously active per-unit burstlets exceeds
  a threshold.
- "chain": follow the SIMMUX rule from Wagenaar, DeMarse & Potter
  (2005, MeaBench paper): a network burst is a maximal set of
  per-unit burstlets connected by pairwise temporal overlap. Tiny
  bursts (fewer than `threshold` distinct units) are filtered out
  post-hoc.

When `maxISIstart`/`maxISIb` are passed as values < 1 they are
interpreted as a fraction of each unit's inverse mean firing rate
(fraction * mean ISI on that unit), optionally capped at
`isi_cap_ms`. The SIMMUX paper's "core" burstlet uses
`maxISIstart=maxISIb=0.25` with `isi_cap_ms=100`.

When `entourage_maxISI` is given (also < 1 = fraction, >= 1 = ms),
each core burstlet is extended in both temporal directions by spikes
whose ISI is below `min(entourage_maxISI * mean_isi, entourage_cap_ms)`.
Adjacent burstlets whose extended regions touch are merged. The SIMMUX
paper uses `entourage_maxISI=1/3` with `entourage_cap_ms=200`.

Together with `network_rule="chain"`, `minSburst=3`, `minBdur=0`,
`minIBI=0`, and `threshold=5` (i.e. ">=5 electrodes"), this reproduces
the SIMMUX algorithm from Wagenaar et al. 2005/2006.

Note: the unit `threshold` uses inclusive ">=" semantics (a burst is
kept/active when the number of units is >= threshold). Earlier versions
used strict ">"; to reproduce old behaviour add 1 to the threshold
(old ">4" == new ">=5").
"""

from itertools import groupby
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


def _resolve_unit_isi_threshold(
    value: float,
    st_unit: np.ndarray,
    recording_duration_ms: float,
    isi_cap_ms: Optional[float],
) -> float:
    """Resolve an ISI parameter for a single unit.

    Values >= 1 are interpreted as a fixed ms threshold and returned as-is.
    Values < 1 are interpreted as a fraction of that unit's inverse mean
    firing rate (i.e. fraction * mean ISI on that unit). If `isi_cap_ms`
    is provided, the adaptive value is capped at it.
    """
    if value >= 1:
        return float(value)

    n_spikes = int(st_unit.size)
    if recording_duration_ms <= 0 or n_spikes < 2:
        # No usable rate estimate; fall back to the cap if given, else the
        # raw value (which is a tiny ms threshold and will detect nothing).
        return float(isi_cap_ms) if isi_cap_ms is not None else float(value)

    # mean ISI on this unit, derived from average firing rate over the
    # whole recording duration: 1 / rate = recording_duration / n_spikes.
    mean_isi_ms = recording_duration_ms / n_spikes
    threshold_ms = float(value) * mean_isi_ms
    if isi_cap_ms is not None and threshold_ms > isi_cap_ms:
        threshold_ms = float(isi_cap_ms)
    return threshold_ms


def _extend_burstlets_with_entourage(
    cores: List[Tuple[float, float]],
    spikes: np.ndarray,
    extend_isi_threshold_ms: float,
) -> List[Tuple[float, float]]:
    """Extend each core burstlet by an "entourage" of nearby spikes.

    For each core (start, end) interval, the burstlet is extended backward
    while the ISI to the next spike inside the burstlet is below
    `extend_isi_threshold_ms`, and analogously forward. Burstlets whose
    extended regions touch or overlap are merged into one.

    `spikes` must be sorted in ascending order.
    """
    if not cores or extend_isi_threshold_ms <= 0 or spikes.size < 2:
        return list(cores)

    extended: List[Tuple[float, float]] = []
    for start, end in cores:
        # walk backward from the first core spike
        i_start = int(np.searchsorted(spikes, start, side="left"))
        new_start = start
        while i_start > 0:
            isi = spikes[i_start] - spikes[i_start - 1]
            if isi < extend_isi_threshold_ms:
                i_start -= 1
                new_start = float(spikes[i_start])
            else:
                break

        # walk forward from the last core spike
        i_end = int(np.searchsorted(spikes, end, side="right")) - 1
        # safety: in degenerate cases searchsorted may yield -1
        if i_end < 0:
            i_end = 0
        new_end = end
        while i_end < spikes.size - 1:
            isi = spikes[i_end + 1] - spikes[i_end]
            if isi < extend_isi_threshold_ms:
                i_end += 1
                new_end = float(spikes[i_end])
            else:
                break

        extended.append((new_start, new_end))

    # merge adjacent extended burstlets whose intervals touch/overlap
    extended.sort(key=lambda x: x[0])
    merged: List[Tuple[float, float]] = []
    for start, end in extended:
        if merged and start <= merged[-1][1]:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _detect_unit_burstlets(
    st_arr: np.ndarray,
    gid_arr: np.ndarray,
    maxISIstart: float,
    maxISIb: float,
    minBdur: float,
    minIBI: float,
    minSburst: float,
    isi_cap_ms: Optional[float],
    entourage_maxISI: Optional[float],
    entourage_cap_ms: Optional[float],
    recording_duration_ms: float,
) -> dict:
    """Detect per-unit burstlets, optionally extended by an entourage.

    Returns a dict mapping unit id to a list of (start, end) tuples.
    Units with no burstlets are omitted.
    """
    unit_bursts: dict = {}
    for unit in np.unique(gid_arr):
        mask = gid_arr == unit
        st_unit = np.sort(st_arr[mask])
        if st_unit.size == 0:
            continue
        unit_maxISIstart = _resolve_unit_isi_threshold(
            maxISIstart, st_unit, recording_duration_ms, isi_cap_ms
        )
        unit_maxISIb = _resolve_unit_isi_threshold(
            maxISIb, st_unit, recording_duration_ms, isi_cap_ms
        )
        cores = _to_intervals(
            MI_bursts(
                st_unit,
                maxISIstart=unit_maxISIstart,
                maxISIb=unit_maxISIb,
                minBdur=minBdur,
                minIBI=minIBI,
                minSburst=minSburst,
            )
        )
        if not cores:
            continue

        if entourage_maxISI is not None:
            extend_threshold_ms = _resolve_unit_isi_threshold(
                entourage_maxISI, st_unit, recording_duration_ms, entourage_cap_ms
            )
            cores = _extend_burstlets_with_entourage(
                cores, st_unit, extend_threshold_ms
            )

        if cores:
            unit_bursts[unit] = cores
    return unit_bursts


def _network_bursts_simultaneity(
    unit_bursts: dict,
    n_units_threshold: float,
) -> List[Tuple[float, float]]:
    """Emit a network burst while > n_units_threshold units are active."""
    events: List[Tuple[float, int]] = []
    for intervals in unit_bursts.values():
        for start, end in intervals:
            events.append((start, 1))
            events.append((end, -1))
    if not events:
        return []

    # end (-1) processed before start (+1) at same time
    events.sort(key=lambda item: (item[0], 0 if item[1] == -1 else 1))

    merged_events = [
        (time, sum(delta for _, delta in group))
        for time, group in groupby(events, key=lambda x: x[0])
    ]

    active_units = 0
    network_start: Optional[float] = None
    network_bursts: List[Tuple[float, float]] = []
    for time, delta in merged_events:
        prev_active = active_units
        active_units += delta
        # A network burst is "on" while active_units >= n_units_threshold
        # (inclusive ">=" semantics).
        if prev_active < n_units_threshold <= active_units:
            network_start = float(time)
        elif (
            prev_active >= n_units_threshold > active_units
            and network_start is not None
        ):
            network_bursts.append((network_start, float(time)))
            network_start = None
    return network_bursts


def _network_bursts_chain(
    unit_bursts: dict,
    n_units_threshold: float,
) -> List[Tuple[float, float]]:
    """Chain-overlap rule (SIMMUX): connected components on the overlap graph.

    All per-unit burstlets are sorted by start time; any two burstlets
    with non-zero temporal overlap belong to the same component. A
    component becomes a network burst spanning
    `[min(starts), max(ends)]`. Components with fewer than
    `n_units_threshold` distinct units are dropped (matching the
    "tiny burst" exclusion: with `n_units_threshold=5`, the surviving
    bursts span >= 5 units). Inclusive ">=" semantics.
    """
    if not unit_bursts:
        return []

    intervals: List[Tuple[float, float, object]] = []
    for unit, ivs in unit_bursts.items():
        for start, end in ivs:
            intervals.append((float(start), float(end), unit))
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])

    network_bursts: List[Tuple[float, float]] = []
    comp_start = intervals[0][0]
    comp_end = intervals[0][1]
    comp_units = {intervals[0][2]}
    for start, end, unit in intervals[1:]:
        if start < comp_end:
            # overlaps current component -> extend
            if end > comp_end:
                comp_end = end
            comp_units.add(unit)
        else:
            # close component
            if len(comp_units) >= n_units_threshold:
                network_bursts.append((comp_start, comp_end))
            comp_start = start
            comp_end = end
            comp_units = {unit}
    if len(comp_units) >= n_units_threshold:
        network_bursts.append((comp_start, comp_end))
    return network_bursts


def _network_bursts_mcs(
    unit_bursts: dict,
    min_simultaneous: float,
    min_participating: float,
) -> List[Tuple[float, float]]:
    """Multi Channel Systems (MCS) network burst rule (manual 5.13.3.2).

    Network burst = chain component of temporally overlapping per-channel
    burstlets, spanning [min(start), max(end)] (same window as the chain rule).
    A component is kept only if (POST-HOC gates that keep/drop the whole burst;
    they do NOT trim the window):
      - distinct participating channels >= `min_participating`
        ("Min. Channels Participating"), and
      - the peak number of simultaneously active burstlets >= `min_simultaneous`
        ("Min. Simultaneous Channels").
    Inclusive ">=" semantics, consistent with the other rules.
    """
    intervals = sorted(
        (float(s), float(e), u) for u, ivs in unit_bursts.items() for s, e in ivs
    )
    if not intervals:
        return []
    network_bursts: List[Tuple[float, float]] = []

    def _emit(members):
        # peak simultaneously active burstlets (ends before starts at ties)
        events = sorted(
            [ev for s, e, _u in members for ev in ((s, 1), (e, -1))],
            key=lambda x: (x[0], 0 if x[1] == -1 else 1),
        )
        active = peak = 0
        for _t, d in events:
            active += d
            peak = max(peak, active)
        if (
            len({u for _s, _e, u in members}) >= min_participating
            and peak >= min_simultaneous
        ):
            network_bursts.append((members[0][0], max(e for _s, e, _u in members)))

    comp = [intervals[0]]
    comp_end = intervals[0][1]
    for s, e, u in intervals[1:]:
        if s < comp_end:
            comp.append((s, e, u))
            comp_end = max(comp_end, e)
        else:
            _emit(comp)
            comp = [(s, e, u)]
            comp_end = e
    _emit(comp)
    return network_bursts


def _gate_simultaneity_windows(
    windows: List[Tuple[float, float]],
    unit_bursts: dict,
    min_peak: float,
    min_participating: float,
) -> List[Tuple[float, float]]:
    """Filter simultaneity windows by MCS-style gates (no reshaping).

    Each window is a maximal interval over which the number of simultaneously
    active burstlets is >= the simultaneity trim level. The window is kept iff,
    restricted to that window:
      - the peak number of simultaneously active burstlets is >= `min_peak`
        ("Min. Simultaneous Channels"), and
      - the number of distinct participating units (with a burstlet overlapping
        the window) is >= `min_participating` ("Min. Channels Participating").
    Gates keep/drop whole windows; they do NOT trim. This is the validated
    "Route 2" rule: simultaneity-trim first, then MCS gate each fragment.
    Inclusive ">=" semantics.
    """
    kept: List[Tuple[float, float]] = []
    for ws, we in windows:
        events: List[Tuple[float, int]] = []
        participating = 0
        for ivs in unit_bursts.values():
            overlaps = False
            for s, e in ivs:
                if s < we and e > ws:  # burstlet overlaps the window
                    overlaps = True
                    events.append((max(s, ws), 1))
                    events.append((min(e, we), -1))
            if overlaps:
                participating += 1
        if participating < min_participating:
            continue
        events.sort(key=lambda item: (item[0], 0 if item[1] == -1 else 1))
        active = peak = 0
        for _t, delta in events:
            active += delta
            if active > peak:
                peak = active
        if peak >= min_peak:
            kept.append((ws, we))
    return kept


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
    isi_cap_ms: Optional[float] = None,
    recording_duration_ms: Optional[float] = None,
    entourage_maxISI: Optional[float] = None,
    entourage_cap_ms: Optional[float] = 200,
    network_rule: str = "chain",
    mcs_min_participating: Optional[float] = None,
    mcs_min_simultaneous: Optional[float] = None,
    return_unit_bursts: bool = False,
):
    """Detect network bursts from per-unit burstlets.

    Per-unit burstlets are detected with `MI_bursts`. With
    `entourage_maxISI` set, each core is extended on both sides by
    spikes with ISI below `min(entourage_maxISI * mean_isi,
    entourage_cap_ms)`, and overlapping extended burstlets are merged.

    The `network_rule` selects how per-unit burstlets are combined
    (both use inclusive ">=" semantics on `threshold`):
      - "simultaneity" (default): emit a network burst while the
        number of simultaneously active per-unit burstlets is >=
        the absolute or fractional `threshold`.
      - "chain": connected-component on the overlap graph (Wagenaar
        SIMMUX). Components with fewer than `threshold` distinct units
        (interpreted as an absolute count) are dropped.

    Args:
        st: spike times in milliseconds (same length as gid).
        gid: unit ids corresponding to st.
        maxISIstart, maxISIb: passed to `MI_bursts`. If >= 1, absolute
            ms. If < 1, fraction of each unit's inverse mean rate,
            capped at `isi_cap_ms`. SIMMUX uses 0.25 + cap=100.
        minBdur, minIBI, minSburst: passed to `MI_bursts`. SIMMUX uses
            0, 0, 3.
        threshold: for both rules, values in (0,1] are interpreted as
            a fraction of `n_units` and values > 1 as an absolute count.
            Comparison is inclusive ">=", so e.g. threshold=5 keeps
            bursts spanning >= 5 distinct units, and threshold=0.5 with
            n_units=10 keeps bursts spanning >= 5. Required (None
            raises ValueError).
        n_units: override for total units (defaults to len(unique gid)).
            Used to resolve fractional `threshold` for both rules.
        isi_cap_ms: cap on the per-unit core ISI threshold when
            `maxISIstart`/`maxISIb` are fractions. SIMMUX uses 100.
        recording_duration_ms: optional override for the per-unit
            mean-rate estimate. Falls back to `max(st) - min(st)`.
        entourage_maxISI: enables entourage extension when not None.
            Fraction (<1) or ms (>=1). SIMMUX uses 1/3.
        entourage_cap_ms: cap on entourage ISI when fractional.
            SIMMUX uses 200.
        network_rule: "simultaneity" or "chain".
        return_unit_bursts: also return per-unit burstlet intervals
            as a dict.

    Returns:
        list of (start, end) network burst tuples; if
        `return_unit_bursts=True`, returns
        `(network_bursts, unit_bursts_dict)`.
    """
    st_arr = np.asarray(st)
    gid_arr = np.asarray(gid)

    if st_arr.size == 0:
        return ([], {}) if return_unit_bursts else []

    duration_ms: float = (
        float(recording_duration_ms)
        if recording_duration_ms is not None
        else float(np.max(st_arr)) - float(np.min(st_arr))
    )

    unit_bursts = _detect_unit_burstlets(
        st_arr=st_arr,
        gid_arr=gid_arr,
        maxISIstart=maxISIstart,
        maxISIb=maxISIb,
        minBdur=minBdur,
        minIBI=minIBI,
        minSburst=minSburst,
        isi_cap_ms=isi_cap_ms,
        entourage_maxISI=entourage_maxISI,
        entourage_cap_ms=entourage_cap_ms,
        recording_duration_ms=duration_ms,
    )

    if threshold is None:
        raise ValueError(
            "threshold is required (got None). Pass unit_threshold>1 for an "
            "absolute count, or unit_threshold in (0,1] for a fraction of "
            "n_units. With inclusive '>=', e.g. unit_threshold=5 keeps bursts "
            "spanning >=5 distinct units."
        )

    unique_units = np.unique(gid_arr)
    n_units_detected = int(n_units) if n_units is not None else len(unique_units)
    if 0.0 < threshold <= 1.0:
        n_units_threshold = n_units_detected * float(threshold)
    else:
        n_units_threshold = float(threshold)

    if network_rule == "simultaneity":
        network_bursts = _network_bursts_simultaneity(unit_bursts, n_units_threshold)
    elif network_rule == "chain":
        network_bursts = _network_bursts_chain(unit_bursts, n_units_threshold)
    elif network_rule == "mcs":
        if mcs_min_participating is None:
            raise ValueError("network_rule='mcs' requires mcs_min_participating")
        if 0.0 < mcs_min_participating <= 1.0:
            min_participating = n_units_detected * float(mcs_min_participating)
        else:
            min_participating = float(mcs_min_participating)
        # threshold -> Min. Simultaneous Channels; mcs_min_participating ->
        # Min. Channels Participating.
        network_bursts = _network_bursts_mcs(
            unit_bursts, n_units_threshold, min_participating
        )
    elif network_rule == "sim_mcs":
        # Route 2: simultaneity trim (threshold = trim level k) followed by MCS
        # gates that keep/drop each fragment. `mcs_min_simultaneous` -> peak
        # co-activity gate; `mcs_min_participating` -> distinct-channels gate.
        if mcs_min_simultaneous is None or mcs_min_participating is None:
            raise ValueError(
                "network_rule='sim_mcs' requires both "
                "mcs_min_simultaneous and mcs_min_participating"
            )
        windows = _network_bursts_simultaneity(unit_bursts, n_units_threshold)
        if 0.0 < mcs_min_simultaneous <= 1.0:
            min_peak = n_units_detected * float(mcs_min_simultaneous)
        else:
            min_peak = float(mcs_min_simultaneous)
        if 0.0 < mcs_min_participating <= 1.0:
            min_participating = n_units_detected * float(mcs_min_participating)
        else:
            min_participating = float(mcs_min_participating)
        network_bursts = _gate_simultaneity_windows(
            windows, unit_bursts, min_peak, min_participating
        )
    else:
        raise ValueError(
            f"Unknown network_rule {network_rule!r}; "
            "expected 'simultaneity', 'chain', 'mcs' or 'sim_mcs'."
        )

    if return_unit_bursts:
        return network_bursts, unit_bursts
    return network_bursts
