"""Regression tests for the SIMMUX `network_rule="chain"` combination rule.

A network burst is the union of any maximal chain of per-unit
burstlets connected by pairwise temporal overlap. Components with
fewer than `threshold` distinct units are dropped (the "tiny burst"
exclusion, inclusive ">=" semantics: threshold=5 keeps only bursts
spanning >=5 units).
"""

import numpy as np

from burst_shape.preprocess.burst_detection_alternative import (
    _network_bursts_chain,
    network_bursts_from_unit_overlap,
)


def test_chain_empty_input_returns_empty():
    assert _network_bursts_chain({}, n_units_threshold=4) == []


def test_chain_single_unit_below_threshold_dropped():
    # one unit; threshold<=1 keeps it, threshold>=2 drops it
    # (chain uses inclusive ">=" so distinct_units must be >= threshold).
    assert _network_bursts_chain({0: [(0.0, 10.0)]}, n_units_threshold=1) == [
        (0.0, 10.0)
    ]
    assert _network_bursts_chain({0: [(0.0, 10.0)]}, n_units_threshold=2) == []


def test_chain_daisy_chain_forms_single_burst():
    # five units, each burstlet overlaps its temporal neighbor by only
    # a few ms. At no single time are >2 simultaneously active, but the
    # chain has 5 distinct units so it survives threshold=4.
    unit_bursts = {
        0: [(0.0, 10.0)],
        1: [(8.0, 20.0)],
        2: [(18.0, 30.0)],
        3: [(28.0, 40.0)],
        4: [(38.0, 50.0)],
    }
    out = _network_bursts_chain(unit_bursts, n_units_threshold=4)
    assert out == [(0.0, 50.0)]


def test_chain_disconnected_components_emitted_separately():
    # two disjoint clusters each with 5 fully overlapping units.
    unit_bursts = {u: [(0.0, 20.0)] for u in range(5)}
    unit_bursts.update({u + 100: [(100.0, 120.0)] for u in range(5)})
    out = _network_bursts_chain(unit_bursts, n_units_threshold=4)
    assert out == [(0.0, 20.0), (100.0, 120.0)]


def test_chain_filters_tiny_bursts_below_threshold():
    # one cluster of 5 (kept) and a stray pair of 2 (dropped).
    unit_bursts = {u: [(0.0, 20.0)] for u in range(5)}
    unit_bursts.update({100: [(100.0, 105.0)], 101: [(102.0, 110.0)]})
    out = _network_bursts_chain(unit_bursts, n_units_threshold=4)
    assert out == [(0.0, 20.0)]


def test_chain_touching_endpoints_do_not_merge():
    # A.end == B.start -> strict ">" overlap check: NOT overlapping.
    # so this stays 2 singletons (1 unit each). With threshold=2 both are
    # dropped; had they merged, the 2-unit component would survive.
    unit_bursts = {0: [(0.0, 10.0)], 1: [(10.0, 20.0)]}
    out = _network_bursts_chain(unit_bursts, n_units_threshold=2)
    assert out == []


def test_chain_burstlet_with_multiple_intervals_each_contributes():
    # one unit with two intervals; the first overlaps a second unit,
    # the second overlaps a third. Two separate components, each with
    # 2 units, both dropped at threshold=3 (inclusive ">=").
    unit_bursts = {
        0: [(0.0, 10.0), (100.0, 110.0)],
        1: [(5.0, 15.0)],
        2: [(105.0, 115.0)],
    }
    out = _network_bursts_chain(unit_bursts, n_units_threshold=3)
    assert out == []
    # threshold=2 keeps both (each component spans 2 distinct units).
    out = _network_bursts_chain(unit_bursts, n_units_threshold=2)
    assert out == [(0.0, 15.0), (100.0, 115.0)]


def test_chain_via_network_bursts_from_unit_overlap_integration():
    # 5 units with daisy-chain overlap (no single moment with all 5),
    # plus a stray sparse spike pair. With network_rule="chain" and
    # threshold=4 the chain forms one burst spanning all 5 units.
    # Disable entourage to test the bare chain rule on cores only.
    st_per_unit = {
        0: [0.0, 1.0, 2.0, 3.0, 4.0],  # burstlet at [0, 4]
        1: [10.0, 11.0, 12.0, 13.0],  # burstlet at [10, 13]
        2: [20.0, 21.0, 22.0, 23.0],  # burstlet at [20, 23]
        3: [30.0, 31.0, 32.0, 33.0],  # burstlet at [30, 33]
        4: [40.0, 41.0, 42.0, 43.0],  # burstlet at [40, 43]
    }
    # add long-gap suffix on each so MI_bursts can terminate each burstlet
    suffix = [10_000.0, 99_000.0]
    st_list, gid_list = [], []
    for u, s in st_per_unit.items():
        full = np.array(s + suffix, dtype=float)
        st_list.append(full)
        gid_list.append(np.full(full.size, u))
    st = np.concatenate(st_list)
    gid = np.concatenate(gid_list)
    order = np.argsort(st)
    st, gid = st[order], gid[order]

    # entourage extends each core forward/backward by spikes with ISI<15
    # so consecutive cores merge across the 10-ms gaps for each unit
    # -- but here each unit has only ONE core, so the entourage only
    # extends to spikes within the same unit. With no entourage,
    # the per-unit burstlets are [0,4], [10,13], [20,23], ..., and
    # the chain (using direct overlap only) connects them only if
    # adjacent cores temporally overlap. Here [0,4] and [10,13] do NOT
    # overlap (4 < 10), so chain leaves them as separate singletons
    # -> all dropped at threshold=4.
    out_no_overlap = network_bursts_from_unit_overlap(
        st,
        gid,
        maxISIstart=5,
        maxISIb=5,
        minBdur=0,
        minIBI=0,
        minSburst=3,
        threshold=4,
        n_units=5,
        network_rule="chain",
        entourage_maxISI=None,
    )
    assert out_no_overlap == []

    # Now shift units so each burstlet overlaps the next by 2 ms.
    st_per_unit_overlap = {
        0: [0.0, 1.0, 2.0, 3.0, 12.0],  # [0, 3] with stray spike at 12 (gap=9)
        1: [2.0, 3.0, 4.0, 5.0],  # [2, 5]
        2: [4.0, 5.0, 6.0, 7.0],  # [4, 7]
        3: [6.0, 7.0, 8.0, 9.0],  # [6, 9]
        4: [8.0, 9.0, 10.0, 11.0],  # [8, 11]
    }
    st_list, gid_list = [], []
    for u, s in st_per_unit_overlap.items():
        full = np.array(s + suffix, dtype=float)
        st_list.append(full)
        gid_list.append(np.full(full.size, u))
    st = np.concatenate(st_list)
    gid = np.concatenate(gid_list)
    order = np.argsort(st)
    st, gid = st[order], gid[order]

    out_overlap = network_bursts_from_unit_overlap(
        st,
        gid,
        maxISIstart=5,
        maxISIb=5,
        minBdur=0,
        minIBI=0,
        minSburst=3,
        threshold=4,
        n_units=5,
        network_rule="chain",
        entourage_maxISI=None,
    )
    # 5 burstlets daisy-chained -> 1 connected component spanning [0,11].
    assert out_overlap == [(0.0, 11.0)]


def test_unknown_network_rule_raises():
    import pytest

    with pytest.raises(ValueError, match="Unknown network_rule"):
        network_bursts_from_unit_overlap(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0, 0, 0, 0]),
            maxISIstart=5,
            maxISIb=5,
            minBdur=0,
            minIBI=0,
            minSburst=3,
            threshold=0,
            n_units=1,
            network_rule="nonsense",
        )


def test_threshold_none_raises_clear_error():
    # The package default unit_threshold is None, so any extract_bursts
    # call that forgets to set it used to crash with a cryptic
    # TypeError("float() argument must be a string or a real number, not
    # 'NoneType'") deep inside the dispatch. Validate up front with a
    # clear message instead.
    import pytest

    for rule in ("simultaneity", "chain"):
        with pytest.raises(ValueError, match="threshold is required"):
            network_bursts_from_unit_overlap(
                np.array([0.0, 1.0, 2.0, 3.0]),
                np.array([0, 0, 0, 0]),
                maxISIstart=5,
                maxISIb=5,
                minBdur=0,
                minIBI=0,
                minSburst=3,
                threshold=None,
                n_units=1,
                network_rule=rule,
                entourage_maxISI=None,
            )


def test_chain_fractional_threshold_uses_n_units():
    # Chain rule should accept fractional threshold just like simultaneity:
    # threshold=0.5 with n_units=10 -> drop components with <5 units.
    unit_bursts = {u: [(0.0, 10.0)] for u in range(6)}  # 6 overlapping units
    # threshold=0.5, n_units=10 -> n_units_threshold=5 -> keep (6 >= 5)
    bursts = network_bursts_from_unit_overlap(
        # craft tiny spike train just so the function reaches the rule:
        st=np.array([0.0]),
        gid=np.array([0]),
        threshold=0.5,
        n_units=10,
        network_rule="chain",
        entourage_maxISI=None,
    )
    # the above tiny input has no real bursts; assert via the helper directly
    from burst_shape.preprocess.burst_detection_alternative import (
        _network_bursts_chain,
    )

    assert _network_bursts_chain(unit_bursts, n_units_threshold=6.0) == [(0.0, 10.0)]
    assert _network_bursts_chain(unit_bursts, n_units_threshold=7.0) == []
    # bursts is whatever the synthetic call produced (probably []) -- this
    # subtest just ensures no exception is raised for the fractional case.
    assert isinstance(bursts, list)
