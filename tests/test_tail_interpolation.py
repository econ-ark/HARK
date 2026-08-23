"""Tests for HARK.tail_interpolation.

Strategy: build knot data from an EXACT power-law-gap consumption function
c(x) = kappa*(x+h) - A*((x+h)/(x_top+h))**(-q), so every tail claim has a
closed form to compare against, and the MoM log-gap chart of the data is
exactly linear in chart coordinates (log gap vs log(x+h)) -- interpolation
must reproduce it to floating-point accuracy everywhere, including beyond the
knot range.
"""

import copy
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from HARK.interpolation import CubicHermiteInterp, LinearInterp
from HARK.tail_interpolation import (
    MoMLogGapChartInterp,
    PowerLawDecayCubicHermiteInterp,
    PowerLawDecayLinearInterp,
    chartify_in_place,
    retrofit_powerlaw,
)

KAPPA = 0.05
H = 10.0
Q_TRUE = 1.2
A_TOP = 0.3


def make_data(n=40, x_lo=0.5, x_hi=40.0):
    x = np.geomspace(x_lo, x_hi, n)
    pivot = x_hi + H
    gap = A_TOP * ((x + H) / pivot) ** (-Q_TRUE)
    c = KAPPA * (x + H) - gap
    dc = KAPPA + Q_TRUE * gap / (x + H)
    return x, c, dc


def closed_form(x_above, x_top):
    pivot = x_top + H
    gap = A_TOP * ((x_above + H) / pivot) ** (-Q_TRUE)
    c = KAPPA * (x_above + H) - gap
    dc = KAPPA + Q_TRUE * gap / (x_above + H)
    return c, dc


X_ABOVE = np.geomspace(41.0, 4000.0, 50)


def test_linear_powerlaw_matches_closed_form():
    x, c, _ = make_data()
    f = PowerLawDecayLinearInterp(
        x, c, intercept_limit=KAPPA * H, slope_limit=KAPPA, decay_extrap_Q=Q_TRUE
    )
    assert f.decay_extrap_form == "powerlaw"
    c_ref, dc_ref = closed_form(X_ABOVE, x[-1])
    assert_allclose(f(X_ABOVE), c_ref, rtol=1e-12)
    assert_allclose(f.derivative(X_ABOVE), dc_ref, rtol=1e-12)


def test_cubic_powerlaw_matches_closed_form():
    x, c, dc = make_data()
    f = PowerLawDecayCubicHermiteInterp(
        x, c, dc, intercept_limit=KAPPA * H, slope_limit=KAPPA, decay_extrap_Q=Q_TRUE
    )
    assert f.decay_extrap_form == "powerlaw"
    c_ref, dc_ref = closed_form(X_ABOVE, x[-1])
    assert_allclose(f(X_ABOVE), c_ref, rtol=1e-12)
    assert_allclose(f.derivative(X_ABOVE), dc_ref, rtol=1e-12)


def test_cubic_slope_derived_q_recovers_true_exponent():
    # With EXACT knot slopes, Q = B*(x_top+h) from the top-knot level+slope
    # pair equals the true exponent identically.
    x, c, dc = make_data()
    f = PowerLawDecayCubicHermiteInterp(
        x, c, dc, intercept_limit=KAPPA * H, slope_limit=KAPPA
    )
    assert f.decay_extrap_form == "powerlaw"
    assert_allclose(f.decay_extrap_Q, Q_TRUE, rtol=1e-9)


def test_linear_slope_derived_q_close():
    # LinearInterp's top slope is a secant approximation, so the slope-derived
    # exponent is close to (not exactly) the true one.
    x, c, _ = make_data(n=200)
    f = PowerLawDecayLinearInterp(x, c, intercept_limit=KAPPA * H, slope_limit=KAPPA)
    assert f.decay_extrap_form == "powerlaw"
    assert_allclose(f.decay_extrap_Q, Q_TRUE, rtol=0.1)


def test_guards_refuse_bad_geometry():
    # Top knot ON the limiting line: A = 0 -> refuse, keep a safe form.
    x = np.linspace(1.0, 20.0, 10)
    c = KAPPA * (x + H)  # exactly on the line
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = PowerLawDecayLinearInterp(
            x, c, intercept_limit=KAPPA * H, slope_limit=KAPPA, decay_extrap_Q=Q_TRUE
        )
    assert f.decay_extrap_form != "powerlaw"
    assert np.all(np.isfinite(f(X_ABOVE)))


def test_retrofit_linear_preserves_interior_bitwise():
    x, c, _ = make_data()
    f = LinearInterp(x, c, intercept_limit=KAPPA * H, slope_limit=KAPPA)
    x_interior = np.linspace(x[0], x[-1], 137)
    before = f(x_interior).copy()
    ok = retrofit_powerlaw(f, Q_TRUE)
    assert ok
    assert isinstance(f, PowerLawDecayLinearInterp)
    assert np.array_equal(f(x_interior), before)
    c_ref, _ = closed_form(X_ABOVE, x[-1])
    assert_allclose(f(X_ABOVE), c_ref, rtol=1e-12)


def test_retrofit_cubichermite_preserves_interior_bitwise():
    x, c, dc = make_data()
    f = CubicHermiteInterp(x, c, dc, intercept_limit=KAPPA * H, slope_limit=KAPPA)
    x_interior = np.linspace(x[0], x[-1], 137)
    before = f(x_interior).copy()
    ok = retrofit_powerlaw(f, Q_TRUE)
    assert ok
    assert isinstance(f, PowerLawDecayCubicHermiteInterp)
    assert np.array_equal(f(x_interior), before)
    c_ref, dc_ref = closed_form(X_ABOVE, x[-1])
    assert_allclose(f(X_ABOVE), c_ref, rtol=1e-12)
    assert_allclose(f.derivative(X_ABOVE), dc_ref, rtol=1e-12)


def test_chart_reproduces_exact_powerlaw_everywhere():
    # log(gap) is exactly linear in log(x+h) for this data, so the hermite
    # chart must reproduce c to floating point everywhere -- interior, above
    # the top knot, and below the bottom knot.
    x, c, dc = make_data()
    f = MoMLogGapChartInterp(x, c, dc, MPCmin=KAPPA, hNrm=H)
    x_all = np.geomspace(0.05, 4000.0, 300)
    pivot = x[-1] + H
    gap = A_TOP * ((x_all + H) / pivot) ** (-Q_TRUE)
    c_ref = KAPPA * (x_all + H) - gap
    dc_ref = KAPPA + Q_TRUE * gap / (x_all + H)
    assert_allclose(f(x_all), c_ref, rtol=1e-10)
    assert_allclose(f.derivative(x_all), dc_ref, rtol=1e-8)
    assert f.decay_extrap_form == "mom_chart"
    assert_allclose(f.decay_extrap_Q, Q_TRUE, rtol=1e-9)


def test_chart_pchip_variant():
    x, c, dc = make_data()
    f = MoMLogGapChartInterp(x, c, dc, MPCmin=KAPPA, hNrm=H, interp_kind="pchip")
    x_all = np.geomspace(0.5, 400.0, 100)
    pivot = x[-1] + H
    gap = A_TOP * ((x_all + H) / pivot) ** (-Q_TRUE)
    assert_allclose(f(x_all), KAPPA * (x_all + H) - gap, rtol=1e-7)


def test_chartify_in_place_matches_constructed_chart():
    x, c, dc = make_data()
    host = CubicHermiteInterp(x, c, dc)
    built = MoMLogGapChartInterp(x, c, dc, MPCmin=KAPPA, hNrm=H)
    ok, why = chartify_in_place(host, MPCmin=KAPPA, hNrm=H)
    assert ok, why
    assert isinstance(host, MoMLogGapChartInterp)
    x_all = np.geomspace(0.5, 400.0, 100)
    assert_allclose(host(x_all), built(x_all), rtol=1e-12)
    # idempotent
    ok2, why2 = chartify_in_place(host, MPCmin=KAPPA, hNrm=H)
    assert ok2 and why2 == "already chart"


def test_chartify_refuses_and_leaves_host_untouched():
    x = np.linspace(1.0, 20.0, 10)
    c = KAPPA * (x + H) + 0.01  # ABOVE the bound: gap < 0
    dc = np.full_like(x, KAPPA)
    host = CubicHermiteInterp(x, c, dc)
    ok, why = chartify_in_place(host, MPCmin=KAPPA, hNrm=H)
    assert not ok
    assert type(host) is CubicHermiteInterp
    assert "gap" in why


def test_deepcopy_survives():
    # The CubicHermiteInterp base carries scipy-deepcopy immunity; the chart
    # and retrofit subclasses must inherit it (including the chart's own
    # scipy splines).
    x, c, dc = make_data()
    chart = MoMLogGapChartInterp(x, c, dc, MPCmin=KAPPA, hNrm=H)
    chs = CubicHermiteInterp(x, c, dc, intercept_limit=KAPPA * H, slope_limit=KAPPA)
    retrofit_powerlaw(chs, Q_TRUE)
    probe = np.geomspace(0.5, 400.0, 50)
    for f in (chart, chs):
        g = copy.deepcopy(f)
        assert_allclose(g(probe), f(probe), rtol=0, atol=0)


def test_retrofit_rejects_nonpositive_q():
    x, c, _ = make_data()
    f = LinearInterp(x, c, intercept_limit=KAPPA * H, slope_limit=KAPPA)
    assert not retrofit_powerlaw(f, 0.0)
    assert not retrofit_powerlaw(f, None)
    assert type(f) is LinearInterp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# measure_local_q (the estimation companion: PR amendment 2026-08-23)
# ---------------------------------------------------------------------------
from HARK.tail_interpolation import measure_local_q


def test_measure_local_q_recovers_exact_exponent():
    # Exact power-law gap in the pivot coordinate xi = m + H (make_data's
    # form): both secants equal Q_TRUE to fp precision, drift ~ 0.
    m = np.geomspace(2.0, 40.0, 12)
    xi = m + H
    pivot = m[-1] + H
    c = KAPPA * xi - A_TOP * (xi / pivot) ** (-Q_TRUE)
    out = measure_local_q(m, c, KAPPA, H)
    assert out["ok"], out["reason"]
    np.testing.assert_allclose(out["Q"], Q_TRUE, rtol=1e-12)
    assert abs(out["drift"]) < 1e-9


def test_measure_local_q_rejects_shallow_span():
    m = np.linspace(10.0, 10.2, 5)   # ln-span far below min_span
    xi = m + H
    c = KAPPA * xi - A_TOP * (xi / (m[-1] + H)) ** (-Q_TRUE)
    out = measure_local_q(m, c, KAPPA, H)
    assert not out["ok"]
    assert "span" in out["reason"]


def test_measure_then_retrofit_end_to_end():
    # The full loop the PR previously could not close alone: measure Q from a
    # solved-knot set (make_data's exact power-law tail), retrofit the
    # interpolant with the MEASURED exponent, and match the closed form
    # beyond the grid top — no externally supplied Q anywhere.
    x, c, _ = make_data()
    out = measure_local_q(x, c, KAPPA, H)
    assert out["ok"], out["reason"]
    np.testing.assert_allclose(out["Q"], Q_TRUE, rtol=1e-12)
    assert abs(out["drift"]) < 1e-9
    f = LinearInterp(x, c, intercept_limit=KAPPA * H, slope_limit=KAPPA)
    ok = retrofit_powerlaw(f, out["Q"])
    assert ok
    c_ref, _ = closed_form(X_ABOVE, x[-1])
    assert_allclose(f(X_ABOVE), c_ref, rtol=1e-10)
