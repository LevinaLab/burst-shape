"""Regression tests for the SIMMUX-style entourage extension.

The entourage rule (Wagenaar/DeMarse/Potter 2005 SIMMUX paper) extends
each core burstlet in both directions by adding spikes whose ISI is
below `min(entourage_maxISI * mean_isi, entourage_cap_ms)`. Burstlets
whose extended regions touch or overlap are merged into one.

These tests cover the helper `_extend_burstlets_with_entourage` plus
the integration through `network_bursts_from_unit_overlap` via the
`entourage_maxISI`/`entourage_cap_ms` kwargs.
"""

import numpy as np

from burst_shape.preprocess.burst_detection_alternative import (
    _extend_burstlets_with_entourage,
    network_bursts_from_unit_overlap,
)


def test_entourage_extends_single_core_both_sides():
    # core covers [10, 13] (4 tight spikes at ms 10, 11, 12, 13).
    # neighbors at 6, 8 (gaps 2 and 2 -- both < 5 ms cutoff) and
    # 15, 17 (gaps 2 and 2 -- both < 5 ms cutoff) should be absorbed.
    spikes = np.array([0.0, 6.0, 8.0, 10.0, 11.0, 12.0, 13.0, 15.0, 17.0, 100.0])
    cores = [(10.0, 13.0)]
    out = _extend_burstlets_with_entourage(cores, spikes, extend_isi_threshold_ms=5.0)
    assert out == [(6.0, 17.0)]


def test_entourage_stops_at_first_gap_above_threshold():
    # extending backward: gaps 6->10 is 4 ms (< 5), 0->6 is 6 ms (>= 5).
    # so backward stops at 6.0.
    spikes = np.array([0.0, 6.0, 10.0, 11.0, 12.0, 13.0])
    cores = [(10.0, 13.0)]
    out = _extend_burstlets_with_entourage(cores, spikes, extend_isi_threshold_ms=5.0)
    assert out == [(6.0, 13.0)]


def test_entourage_strict_less_than_threshold():
    # ISI exactly equal to the threshold should NOT extend (strict <).
    spikes = np.array([5.0, 10.0, 11.0, 12.0, 13.0])
    cores = [(10.0, 13.0)]
    out = _extend_burstlets_with_entourage(cores, spikes, extend_isi_threshold_ms=5.0)
    assert out == [(10.0, 13.0)]


def test_entourage_merges_adjacent_extended_burstlets():
    # two cores [10,13] and [25,28] with a chain of close-ISI spikes
    # bridging them: 14, 16, 18, 20, 22, 24 -- all gaps == 2 ms (< 5).
    # the first core's forward extension and the second core's
    # backward extension should meet, merging both into one burstlet.
    spikes = np.array(
        [
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            16.0,
            18.0,
            20.0,
            22.0,
            24.0,
            25.0,
            26.0,
            27.0,
            28.0,
        ]
    )
    cores = [(10.0, 13.0), (25.0, 28.0)]
    out = _extend_burstlets_with_entourage(cores, spikes, extend_isi_threshold_ms=5.0)
    assert out == [(10.0, 28.0)]


def test_entourage_disabled_when_threshold_nonpositive():
    spikes = np.array([5.0, 10.0, 11.0, 12.0, 13.0, 18.0])
    cores = [(10.0, 13.0)]
    out = _extend_burstlets_with_entourage(cores, spikes, extend_isi_threshold_ms=0.0)
    assert out == [(10.0, 13.0)]


def test_entourage_disabled_when_no_cores():
    spikes = np.array([0.0, 1.0, 2.0])
    out = _extend_burstlets_with_entourage([], spikes, extend_isi_threshold_ms=10.0)
    assert out == []


def test_entourage_integration_via_network_bursts():
    # one unit with a core burstlet at 1000..1003 (4 spikes spaced 1 ms),
    # plus a nearby spike at 1010 that the core ISI (1 ms) would reject
    # but the entourage ISI (20 ms) accepts. Then a long gap to 99 000.
    st = np.concatenate(
        [
            np.array([0.0]),
            np.arange(1000, 1004, 1.0),
            np.array([1010.0, 50_000.0, 99_000.0]),
        ]
    )
    gid = np.zeros(st.size, dtype=int)

    # without entourage: core only spans [1000, 1003]
    bursts_no_ent, unit_bursts_no_ent = network_bursts_from_unit_overlap(
        st,
        gid,
        maxISIstart=5,
        maxISIb=5,
        minBdur=0,
        minIBI=0,
        minSburst=3,
        threshold=0.5,
        n_units=1,
        entourage_maxISI=None,
        network_rule="simultaneity",
        return_unit_bursts=True,
    )
    assert unit_bursts_no_ent[0] == [(1000.0, 1003.0)]

    # with entourage_maxISI=20 ms (absolute): extend to include 1010
    bursts_ent, unit_bursts_ent = network_bursts_from_unit_overlap(
        st,
        gid,
        maxISIstart=5,
        maxISIb=5,
        minBdur=0,
        minIBI=0,
        minSburst=3,
        threshold=0.5,
        n_units=1,
        entourage_maxISI=20.0,  # absolute ms
        network_rule="simultaneity",
        return_unit_bursts=True,
    )
    assert unit_bursts_ent[0] == [(1000.0, 1010.0)]
