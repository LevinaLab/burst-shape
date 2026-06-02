"""Regression tests for _resolve_unit_isi_threshold, the helper that
computes the per-unit ISI threshold used by network_bursts_from_unit_overlap
in Wagenaar 2006 / SIMMUX adaptive mode.

Behavior under test:
- value >= 1 is treated as an absolute ms threshold and returned as-is.
- value < 1 is treated as a fraction of that unit's inverse mean firing
  rate (= fraction * mean ISI on that unit). The result is capped at
  isi_cap_ms when provided.
- With fewer than 2 spikes or zero recording duration, the rate is
  undefined and the function falls back to isi_cap_ms (or the raw
  fractional value if no cap was given).
"""

import numpy as np
import pytest

from burst_shape.preprocess.burst_detection_alternative import (
    _resolve_unit_isi_threshold,
)


def test_resolve_isi_threshold_fixed_mode_returned_as_is():
    # value >= 1 is interpreted as absolute ms and returned unchanged.
    out = _resolve_unit_isi_threshold(
        value=5.0,
        st_unit=np.array([0.0, 10.0, 20.0]),
        recording_duration_ms=1000.0,
        isi_cap_ms=100.0,
    )
    assert out == 5.0


def test_resolve_isi_threshold_adaptive_no_cap():
    # 100 spikes over 1000 ms -> mean ISI = 10 ms -> 0.25 * 10 = 2.5 ms.
    out = _resolve_unit_isi_threshold(
        value=0.25,
        st_unit=np.arange(100, dtype=float),
        recording_duration_ms=1000.0,
        isi_cap_ms=None,
    )
    assert out == pytest.approx(2.5)


def test_resolve_isi_threshold_adaptive_cap_applied():
    # 10 spikes over 1000 ms -> mean ISI = 100 ms -> 0.25 * 100 = 25 ms.
    # With isi_cap_ms=20, the adaptive value is capped at 20.
    out = _resolve_unit_isi_threshold(
        value=0.25,
        st_unit=np.arange(10, dtype=float),
        recording_duration_ms=1000.0,
        isi_cap_ms=20.0,
    )
    assert out == pytest.approx(20.0)


def test_resolve_isi_threshold_adaptive_cap_not_triggered():
    # mean ISI = 10 ms, threshold = 2.5 ms, cap = 100 ms -> not triggered.
    out = _resolve_unit_isi_threshold(
        value=0.25,
        st_unit=np.arange(100, dtype=float),
        recording_duration_ms=1000.0,
        isi_cap_ms=100.0,
    )
    assert out == pytest.approx(2.5)


@pytest.mark.parametrize(
    "st_unit",
    [np.array([]), np.array([5.0])],
)
def test_resolve_isi_threshold_too_few_spikes_falls_back_to_cap(st_unit):
    # With < 2 spikes the rate is undefined; fall back to the cap.
    out = _resolve_unit_isi_threshold(
        value=0.25,
        st_unit=st_unit,
        recording_duration_ms=1000.0,
        isi_cap_ms=100.0,
    )
    assert out == 100.0


def test_resolve_isi_threshold_too_few_spikes_no_cap_returns_raw():
    # No cap: return the raw fractional value (a tiny ms threshold that
    # will detect nothing -- documented fallback).
    out = _resolve_unit_isi_threshold(
        value=0.25,
        st_unit=np.array([5.0]),
        recording_duration_ms=1000.0,
        isi_cap_ms=None,
    )
    assert out == 0.25


def test_resolve_isi_threshold_zero_duration_falls_back_to_cap():
    out = _resolve_unit_isi_threshold(
        value=0.25,
        st_unit=np.array([0.0, 0.0, 0.0]),
        recording_duration_ms=0.0,
        isi_cap_ms=100.0,
    )
    assert out == 100.0
