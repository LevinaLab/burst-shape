"""Regression tests for MI_bursts and its inner numba-jitted helper
find_burstlets. MI_bursts is the function exposed via
`extract_bursts(algorithm="default", ...)`, run on the flattened
population spike train.

The expected outputs encode the *current* behavior of the algorithm
exactly. They are intended to catch unintentional behavior changes:
if any of these tests start failing, the algorithm has changed and
the test (and the change) should be reviewed.

Notes on quirks captured here so they are not silently regressed away:
- find_burstlets uses strict `<` for the burst-start ISI check and
  `<=` for the in-burst ISI check.
- find_burstlets emits the burst when the *next* large ISI is seen, so
  a burst that runs all the way to the last spike is never emitted.
- MI_bursts unconditionally appends the first burstlet without
  applying the minBdur filter; minBdur only affects subsequent
  burstlets.
"""

import numpy as np
import pytest

from burst_shape.preprocess.burst_detection import MI_bursts, find_burstlets


# -------------------- find_burstlets ----------------------------------


def _burstlets(spikes, **kwargs):
    spikes = np.asarray(spikes, dtype=float)
    spikes = np.sort(spikes)
    return find_burstlets(
        spikes,
        np.round(spikes, -1),
        np.diff(spikes),
        **kwargs,
    )


def test_find_burstlets_empty_returns_no_bursts():
    assert _burstlets([], maxISIstart=10, maxISIb=10, minSburst=2) == []


def test_find_burstlets_single_spike_returns_no_bursts():
    assert _burstlets([100.0], maxISIstart=10, maxISIb=10, minSburst=2) == []


def test_find_burstlets_emits_burst_when_long_isi_follows():
    # Spikes: 4 close + 1 isolated. ISIs: [10, 10, 10, 970].
    bursts = _burstlets([0, 10, 20, 30, 1000], maxISIstart=50, maxISIb=50, minSburst=2)
    # burst spans spikes[0]..spikes[3]; spike[4] terminates it.
    assert bursts == [(0.0, 30.0)]


def test_find_burstlets_drops_burst_extending_to_last_spike():
    # No long ISI to terminate the burst -> never emitted (documented quirk).
    bursts = _burstlets([0, 10, 20, 30], maxISIstart=50, maxISIb=50, minSburst=2)
    assert bursts == []


def test_find_burstlets_start_uses_strict_less_than():
    # ISI exactly equal to maxISIstart must NOT start a burst.
    bursts = _burstlets([0, 10, 20, 30, 1000], maxISIstart=10, maxISIb=10, minSburst=2)
    assert bursts == []


def test_find_burstlets_continue_uses_less_or_equal():
    # First ISI < maxISIstart so we start. Subsequent ISIs == maxISIb keep
    # the burst alive.
    bursts = _burstlets([0, 9, 19, 29, 1000], maxISIstart=10, maxISIb=10, minSburst=2)
    assert bursts == [(0.0, 29.0)]


def test_find_burstlets_below_minSburst_is_discarded():
    # b_start counts small ISIs; for ≥ N spikes we need minSburst = N-1.
    bursts = _burstlets([0, 10, 20, 1000], maxISIstart=50, maxISIb=50, minSburst=3)
    assert bursts == []  # only 2 small ISIs (b_start=2) < minSburst=3


def test_find_burstlets_multiple_bursts():
    bursts = _burstlets(
        [0, 10, 20, 1000, 1010, 1020, 2000],
        maxISIstart=50,
        maxISIb=50,
        minSburst=2,
    )
    assert bursts == [(0.0, 20.0), (1000.0, 1020.0)]


# -------------------- MI_bursts ---------------------------------------


def test_MI_bursts_empty_returns_empty():
    assert MI_bursts(np.array([])) == []


def test_MI_bursts_no_burst_returns_empty():
    # ISIs all above threshold.
    assert (
        MI_bursts(
            np.array([0, 100, 200, 300]),
            maxISIstart=10,
            maxISIb=10,
            minBdur=0,
            minIBI=0,
            minSburst=2,
        )
        == []
    )


def test_MI_bursts_single_burst():
    assert MI_bursts(
        np.array([0, 10, 20, 30, 1000]),
        maxISIstart=50,
        maxISIb=50,
        minBdur=0,
        minIBI=0,
        minSburst=2,
    ) == [(0.0, 30.0)]


def test_MI_bursts_two_well_separated_bursts():
    assert MI_bursts(
        np.array([0, 10, 20, 1000, 1010, 1020, 2000]),
        maxISIstart=50,
        maxISIb=50,
        minBdur=0,
        minIBI=0,
        minSburst=2,
    ) == [(0.0, 20.0), (1000.0, 1020.0)]


def test_MI_bursts_merges_burstlets_within_minIBI():
    # Two burstlets separated by a 30 ms gap; with minIBI=100 they merge.
    bursts = MI_bursts(
        np.array([0, 10, 20, 50, 60, 70, 1000]),
        maxISIstart=20,
        maxISIb=20,
        minBdur=0,
        minIBI=100,
        minSburst=2,
    )
    assert bursts == [(0.0, 70.0)]


def test_MI_bursts_minBdur_drops_short_subsequent_burstlet():
    # Two burstlets, second is shorter than minBdur and farther than minIBI
    # from the first -> the second is discarded.
    bursts = MI_bursts(
        np.array([0, 10, 20, 1000, 1010, 1020, 2000]),
        maxISIstart=50,
        maxISIb=50,
        minBdur=30,
        minIBI=100,
        minSburst=2,
    )
    assert bursts == [(0.0, 20.0)]


def test_MI_bursts_first_burstlet_bypasses_minBdur_filter():
    # Documented quirk: the very first burstlet is always emitted
    # regardless of minBdur (here burst is only 20 ms long but minBdur=30).
    bursts = MI_bursts(
        np.array([0, 10, 20, 2000]),
        maxISIstart=50,
        maxISIb=50,
        minBdur=30,
        minIBI=100,
        minSburst=2,
    )
    assert bursts == [(0.0, 20.0)]


def test_MI_bursts_handles_unsorted_input():
    # Input is sorted internally.
    sorted_result = MI_bursts(
        np.array([0, 10, 20, 30, 1000]),
        maxISIstart=50,
        maxISIb=50,
        minBdur=0,
        minIBI=0,
        minSburst=2,
    )
    unsorted_result = MI_bursts(
        np.array([1000, 10, 30, 0, 20]),
        maxISIstart=50,
        maxISIb=50,
        minBdur=0,
        minIBI=0,
        minSburst=2,
    )
    assert sorted_result == unsorted_result == [(0.0, 30.0)]


@pytest.mark.parametrize(
    ("minSburst", "expected"),
    [
        (2, [(0.0, 40.0)]),  # 4 small ISIs (b_start=4) >= 2 -> emit
        (4, [(0.0, 40.0)]),  # b_start=4 >= 4 -> emit
        (5, []),  # b_start=4 < 5 -> no emit
    ],
)
def test_MI_bursts_minSburst_threshold(minSburst, expected):
    assert (
        MI_bursts(
            np.array([0, 10, 20, 30, 40, 1000]),
            maxISIstart=50,
            maxISIb=50,
            minBdur=0,
            minIBI=0,
            minSburst=minSburst,
        )
        == expected
    )
