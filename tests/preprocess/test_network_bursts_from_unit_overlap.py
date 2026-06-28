"""Regression tests for network_bursts_from_unit_overlap.

This file covers both modes of the `"overlap"` algorithm exposed via
`extract_bursts(algorithm="overlap", ...)`:

1. The pre-existing fixed-ms ISI mode (`maxISIstart`/`maxISIb` >= 1).
2. The Wagenaar 2006 / SIMMUX adaptive mode (`maxISIstart`/`maxISIb`
   < 1 plus `isi_cap_ms`).

The expected outputs encode the *current* behavior of the simultaneity
rule (`network_rule="simultaneity"`, no entourage). The module-level
`_SIM_KWARGS` dict pins those two knobs so the tests stay decoupled
from the package-level defaults (which align with canonical SIMMUX:
chain rule + entourage).

Notes on quirks captured here so they are not silently regressed away:
- network_bursts_from_unit_overlap uses strict `>` against the
  threshold (so absolute `threshold=N` requires N+1 simultaneously
  active units to open a network burst).
- A `threshold` of exactly 1 is treated as the fraction 100% (i.e.
  all units), not as the absolute count "1".
- End events (-1) are processed before start events (+1) at the same
  timestamp, so a burst that ends at time T and another that starts at
  T are not considered overlapping.

The Wagenaar-mode helper _resolve_unit_isi_threshold is tested
separately in test_resolve_unit_isi_threshold.py.
"""

import numpy as np

from burst_shape.preprocess.burst_detection_alternative import (
    network_bursts_from_unit_overlap,
)

# pin the per-test rule + entourage off so these tests stay decoupled
# from the package-level SIMMUX defaults.
_SIM_KWARGS = {"network_rule": "simultaneity", "entourage_maxISI": None}


# -------- network_bursts_from_unit_overlap (fixed-ms ISI) -------------
# This is the pre-existing "overlap" algorithm: per-unit MI_bursts with
# absolute ms thresholds, then aggregate by number of simultaneously
# bursting units.


def _three_unit_overlap_spikes():
    """3 units with one tight burst each, offset by 2 ms so they overlap.

    Each unit also has a few isolated spikes far away from the burst so
    a long ISI terminates each burstlet (required by find_burstlets).
    """
    st_per_unit = {
        0: np.array([0.0, *np.arange(1000, 1010, 1.0), 50_000.0, 99_000.0]),
        1: np.array([0.0, *np.arange(1002, 1012, 1.0), 50_000.0, 99_000.0]),
        2: np.array([0.0, *np.arange(1004, 1014, 1.0), 50_000.0, 99_000.0]),
    }
    st_list, gid_list = [], []
    for u, s in st_per_unit.items():
        st_list.append(s)
        gid_list.append(np.full(s.size, u))
    st = np.concatenate(st_list)
    gid = np.concatenate(gid_list)
    order = np.argsort(st)
    return st[order], gid[order]


def test_overlap_empty_input_returns_empty():
    assert (
        network_bursts_from_unit_overlap(np.array([]), np.array([]), **_SIM_KWARGS)
        == []
    )


def test_overlap_empty_input_with_return_unit_bursts():
    bursts, unit_bursts = network_bursts_from_unit_overlap(
        np.array([]), np.array([]), return_unit_bursts=True, **_SIM_KWARGS
    )
    assert bursts == []
    assert unit_bursts == {}


def test_overlap_no_burstlet_when_isi_above_threshold():
    # Single unit with spikes spaced 100 ms apart; threshold = 50 ms.
    st = np.arange(0, 1000, 100.0)
    gid = np.zeros(st.size, dtype=int)
    assert (
        network_bursts_from_unit_overlap(
            st,
            gid,
            maxISIstart=50,
            maxISIb=50,
            minBdur=0,
            minIBI=0,
            minSburst=2,
            threshold=1,  # treated as 100% of units -> all units required
            n_units=1,
            **_SIM_KWARGS,
        )
        == []
    )


def test_overlap_detects_simultaneous_burstlets_absolute_threshold():
    st, gid = _three_unit_overlap_spikes()
    # threshold=3 -> at least 3 active units (inclusive ">="), i.e. all 3.
    # All 3 units are simultaneously bursting in the window [1004, 1009].
    bursts = network_bursts_from_unit_overlap(
        st,
        gid,
        maxISIstart=25,
        maxISIb=25,
        minBdur=0,
        minIBI=0,
        minSburst=3,
        threshold=3,
        n_units=3,
        **_SIM_KWARGS,
    )
    assert bursts == [(1004.0, 1009.0)]


def test_overlap_threshold_above_unit_count_never_triggers():
    # With inclusive ">=" semantics, absolute threshold=4 with n_units=3
    # never triggers, because at most 3 units are ever active at once.
    st, gid = _three_unit_overlap_spikes()
    bursts = network_bursts_from_unit_overlap(
        st,
        gid,
        maxISIstart=25,
        maxISIb=25,
        minBdur=0,
        minIBI=0,
        minSburst=3,
        threshold=4,
        n_units=3,
        **_SIM_KWARGS,
    )
    assert bursts == []


def test_overlap_no_network_burst_when_units_do_not_overlap():
    # 3 units, each bursts at a different time without overlap.
    st_per_unit = {
        0: np.array([0.0, *np.arange(1000, 1010, 1.0), 99_000.0]),
        1: np.array([0.0, *np.arange(5000, 5010, 1.0), 99_000.0]),
        2: np.array([0.0, *np.arange(9000, 9010, 1.0), 99_000.0]),
    }
    st_list, gid_list = [], []
    for u, s in st_per_unit.items():
        st_list.append(s)
        gid_list.append(np.full(s.size, u))
    st = np.concatenate(st_list)
    gid = np.concatenate(gid_list)
    order = np.argsort(st)
    st, gid = st[order], gid[order]

    bursts = network_bursts_from_unit_overlap(
        st,
        gid,
        maxISIstart=25,
        maxISIb=25,
        minBdur=0,
        minIBI=0,
        minSburst=3,
        threshold=1.0001,  # any > 1 active -> network burst
        n_units=3,
        **_SIM_KWARGS,
    )
    assert bursts == []


def test_overlap_fractional_threshold_equivalent_to_absolute():
    # n_units=3 with threshold=0.5 -> n_units_threshold=1.5 -> active >= 1.5
    # i.e. >= 2 active units. The same as absolute threshold=1.5.
    st, gid = _three_unit_overlap_spikes()
    bursts = network_bursts_from_unit_overlap(
        st,
        gid,
        maxISIstart=25,
        maxISIb=25,
        minBdur=0,
        minIBI=0,
        minSburst=3,
        threshold=0.5,
        n_units=3,
        **_SIM_KWARGS,
    )
    # First two units overlap in window [1002, 1009]; the third joins at
    # 1004 (still active) and leaves at 1013 (unit 1 already gone by 1011).
    assert bursts == [(1002.0, 1011.0)]


def test_overlap_returns_per_unit_bursts_when_requested():
    st, gid = _three_unit_overlap_spikes()
    bursts, unit_bursts = network_bursts_from_unit_overlap(
        st,
        gid,
        maxISIstart=25,
        maxISIb=25,
        minBdur=0,
        minIBI=0,
        minSburst=3,
        threshold=3,
        n_units=3,
        return_unit_bursts=True,
        **_SIM_KWARGS,
    )
    assert bursts == [(1004.0, 1009.0)]
    # keys may be numpy ints; cast for comparison
    unit_bursts_py = {int(k): v for k, v in unit_bursts.items()}
    assert unit_bursts_py == {
        0: [(1000.0, 1009.0)],
        1: [(1002.0, 1011.0)],
        2: [(1004.0, 1013.0)],
    }


# ============ network_bursts_from_unit_overlap (Wagenaar/SIMMUX) ======
# Adaptive per-unit ISI threshold mode (maxISIstart/maxISIb < 1).


def test_overlap_wagenaar_mode_uses_adaptive_per_unit_threshold():
    # 3 units, each with one tight burst. With maxISIstart=0.25 and
    # isi_cap_ms=100, the per-unit threshold is min(0.25 * mean_isi, 100).
    # The recording is ~99 seconds with ~13 spikes per unit, giving
    # mean ISI ~ 7615 ms and adaptive threshold ~ 1903 ms, which is
    # capped to 100 ms -- well above the 1 ms intra-burst ISIs.
    st, gid = _three_unit_overlap_spikes()
    bursts = network_bursts_from_unit_overlap(
        st,
        gid,
        maxISIstart=0.25,
        maxISIb=0.25,
        minBdur=0,
        minIBI=0,
        minSburst=3,  # >= 4 spikes per burstlet
        threshold=3,
        n_units=3,
        isi_cap_ms=100,
        **_SIM_KWARGS,
    )
    assert bursts == [(1004.0, 1009.0)]


def test_overlap_wagenaar_mode_no_cap_yields_no_bursts_on_sparse_unit():
    # Without a cap, a sparse unit's adaptive threshold collapses to a
    # fraction of its mean ISI -- which is huge in absolute terms BUT
    # the strict-`<` start condition makes the per-unit detection still
    # work. To exercise the no-cap fallback, use a unit with only 1
    # spike (n_spikes < 2 -> falls back to the raw 0.25 ms threshold,
    # which detects nothing).
    st = np.array([0.0, 1000.0])  # one spike per unit
    gid = np.array([0, 1])
    bursts = network_bursts_from_unit_overlap(
        st,
        gid,
        maxISIstart=0.25,
        maxISIb=0.25,
        minBdur=0,
        minIBI=0,
        minSburst=2,
        threshold=0.5,
        n_units=2,
        isi_cap_ms=None,
        **_SIM_KWARGS,
    )
    assert bursts == []


def test_overlap_wagenaar_mode_isi_cap_actually_limits_detection():
    # With a tight cap (1 ms), even adjacent-in-time spikes fail the
    # strict-`<` start condition (1 ms ISI is NOT < 1 ms threshold).
    st, gid = _three_unit_overlap_spikes()
    bursts = network_bursts_from_unit_overlap(
        st,
        gid,
        maxISIstart=0.25,
        maxISIb=0.25,
        minBdur=0,
        minIBI=0,
        minSburst=3,
        threshold=2,
        n_units=3,
        isi_cap_ms=1.0,
        **_SIM_KWARGS,
    )
    assert bursts == []


def test_overlap_recording_duration_override_changes_adaptive_threshold():
    # Pass an explicit recording duration so each unit's mean rate is
    # high enough that the adaptive threshold falls below 100 ms (cap)
    # and below the intra-burst ISI -- killing detection.
    st_per_unit = {
        0: np.array([*np.arange(1000, 1010, 1.0), 99_000.0]),
        1: np.array([*np.arange(1002, 1012, 1.0), 99_000.0]),
        2: np.array([*np.arange(1004, 1014, 1.0), 99_000.0]),
    }
    st_list, gid_list = [], []
    for u, s in st_per_unit.items():
        st_list.append(s)
        gid_list.append(np.full(s.size, u))
    st = np.concatenate(st_list)
    gid = np.concatenate(gid_list)
    order = np.argsort(st)
    st, gid = st[order], gid[order]

    # With 11 spikes per unit and recording_duration_ms=11 -> mean ISI=1
    # -> 0.25 * 1 = 0.25 ms threshold (well below the 1 ms intra-burst
    # ISIs, so no burstlets are detected).
    bursts = network_bursts_from_unit_overlap(
        st,
        gid,
        maxISIstart=0.25,
        maxISIb=0.25,
        minBdur=0,
        minIBI=0,
        minSburst=3,
        threshold=2,
        n_units=3,
        isi_cap_ms=None,
        recording_duration_ms=11.0,
        **_SIM_KWARGS,
    )
    assert bursts == []
