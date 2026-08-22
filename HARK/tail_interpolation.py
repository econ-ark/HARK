"""Tail representations for buffer-stock consumption functions.

This module collects three related tools for the "above the grid" problem: how a
solved consumption function c(m) should be represented beyond the top gridpoint,
where theory pins the asymptote but the solver has no knots.

Background. For a standard buffer-stock consumer satisfying the relevant
impatience conditions, consumption approaches the perfect-foresight policy
cbar(m) = MPCmin * (m + hNrm) from below (Carroll-Kimball concavity), and the
gap cbar - c decays asymptotically like a POWER LAW in normalized total wealth
X = m + hNrm, not like an exponential in m - m_top. HARK's stock interpolants
(``LinearInterp``/``CubicHermiteInterp`` built with ``intercept_limit`` and
``slope_limit``) attach an exponential-gap extrapolation; over a short span
above the grid the two forms are indistinguishable (the power law's local
expansion IS the exponential), but over the ranges that wealth distributions
and simulation grids actually visit the tails differ materially.

Contents:

* ``PowerLawDecayLinearInterp`` / ``PowerLawDecayCubicHermiteInterp`` -- drop-in
  subclasses whose above-grid gap is ``A * ((x + h)/(x_top + h))**(-Q)`` with
  ``h = intercept_limit/slope_limit``. By default ``Q = B * (x_top + h)`` with
  ``A``/``B`` exactly as the exponential path computes them, so the power law
  matches the LEVEL and SLOPE of the interpolant at the top knot -- the same
  two conditions the exponential matches, with no extra parameters. An explicit
  ``decay_extrap_Q`` override is accepted for externally estimated exponents.

* ``retrofit_powerlaw`` -- IN-PLACE upgrade of an already-constructed stock
  ``LinearInterp``/``CubicHermiteInterp`` to the power-law tail, preserving
  every interior coefficient bit-for-bit (a class swap plus the attach
  attributes). Useful when the interpolant is built deep inside a solver you
  do not wish to modify: every captured reference (for example a
  ``MargValueFuncCRRA`` holding the same object) sees the upgraded tail.

* ``MoMLogGapChartInterp`` / ``chartify_in_place`` -- a Method-of-Moderation
  style re-chart of a solved consumption function: cubic interpolation of
  y = log(cbar - c) against xi = log(m + hNrm), with exact transformed knot
  slopes, evaluated back through the chart. Linear continuation in the chart
  beyond the knot range IS a power-law gap with the locally measured exponent,
  so the attach falls out of the representation. See "The Method of
  Moderation" (Wu, Tokuoka, and Carroll) for the dual-anchor construction this
  chart is drawn from, and Lentini and Keller (1980, SIAM J. Numer. Anal. 17,
  577-604) for the principle of carrying asymptotic boundary information at
  the truncation of an unbounded domain.

  A measured caution for estimation pipelines: evaluating policies through a
  nonlinear chart (log -> interpolate -> exp) inside an OUTER estimation loop
  can imprint interpolation micro-structure on the estimation objective and
  inflate the cross-start scatter of optimizer stops, even while point
  estimates stay unbiased. Charts of this kind belong at solve time (grid
  placement, boundary conditions, representation); prefer representations
  linear in c on objective-evaluation paths. The ``interp_kind="pchip"``
  option (shape-preserving slopes) removes most of the effect where a chart
  must be consumed by an optimizer.

Validity guards throughout follow a transparent-fallback convention: on any
unhealthy geometry (top knot on or above the limiting line, non-approaching
slope, non-positive pivot) the object warns and keeps the stock exponential
tail rather than risk a divergent extrapolation.
"""

import warnings

import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

from HARK.interpolation import CubicHermiteInterp, LinearInterp

__all__ = [
    "PowerLawDecayLinearInterp",
    "PowerLawDecayCubicHermiteInterp",
    "MoMLogGapChartInterp",
    "retrofit_powerlaw",
    "chartify_in_place",
]


class PowerLawDecayLinearInterp(LinearInterp):
    """``LinearInterp`` whose above-grid decay toward the limiting line is a
    power law in ``(x + h)`` instead of an exponential in ``x - x_top``.

    Construct exactly like ``LinearInterp(x, y, intercept_limit, slope_limit)``.
    When the base class engages decay extrapolation (limits supplied, top slope
    distinct from ``slope_limit``), this subclass re-uses its
    ``decay_extrap_A``/``decay_extrap_B`` and replaces only the tail's
    functional form.

    Parameters
    ----------
    decay_extrap_Q : float, optional
        Explicit exponent override (for externally estimated tail exponents).
        When None, ``Q = decay_extrap_B * (x_top + h)`` from the top-knot
        level+slope pair.
    q_diagnostics : object, optional
        Arbitrary diagnostics stashed on the interpolant as ``local_q_diag``
        for post-solve inspection.
    """

    def __init__(
        self,
        x_list,
        y_list,
        intercept_limit=None,
        slope_limit=None,
        lower_extrap=False,
        decay_extrap_Q=None,
        q_diagnostics=None,
    ):
        super().__init__(x_list, y_list, intercept_limit, slope_limit, lower_extrap)
        self._q_override = None if decay_extrap_Q is None else float(decay_extrap_Q)
        self.local_q_diag = q_diagnostics
        self.decay_extrap_form = "exp"  # until validated below
        if not getattr(self, "decay_extrap", False):
            return
        level_diff = float(self.decay_extrap_A)
        ok = (
            slope_limit is not None
            and slope_limit > 0.0
            and level_diff > 0.0
            and float(self.decay_extrap_B) > 0.0
        )
        if ok:
            pivot = float(self.x_list[-1]) + intercept_limit / slope_limit
            ok = pivot > 0.0
        if not ok:
            warnings.warn(
                "PowerLawDecayLinearInterp: the top knot is not strictly below "
                "the limiting line with slope strictly above slope_limit "
                f"(A={level_diff:.6g}, B={float(self.decay_extrap_B):.6g}, "
                f"slope_limit={slope_limit!r}); disabling decay extrapolation "
                "for this interpolant."
            )
            self.decay_extrap = False
            return
        self.decay_extrap_pivot = pivot
        if self._q_override is not None and self._q_override > 0.0:
            self.decay_extrap_Q = self._q_override
        else:
            self.decay_extrap_Q = float(self.decay_extrap_B) * pivot
        self.decay_extrap_form = "powerlaw"

    def _evalOrDer(self, x, _eval, _Der):
        out = super()._evalOrDer(x, _eval, _Der)
        if not (
            getattr(self, "decay_extrap", False)
            and getattr(self, "decay_extrap_form", "exp") == "powerlaw"
        ):
            return out
        x = np.asarray(x)
        above = x > self.x_list[-1]
        if not np.any(above):
            return out
        x_temp = x[above] - self.x_list[-1]
        # gap = A * ((x + h)/(x_top + h))**(-Q); x + h = x_temp + pivot.
        # exp(-Q*log1p(.)) is the numerically stable spelling.
        decay = self.decay_extrap_A * np.exp(
            -self.decay_extrap_Q * np.log1p(x_temp / self.decay_extrap_pivot)
        )
        k = 0
        if _eval:
            out[k][above] = self.intercept_limit + self.slope_limit * x[above] - decay
            k += 1
        if _Der:
            # d(-gap)/dx = +(Q/(x + h)) * gap
            out[k][above] = (
                self.slope_limit
                + self.decay_extrap_Q / (x_temp + self.decay_extrap_pivot) * decay
            )
        return out


class PowerLawDecayCubicHermiteInterp(CubicHermiteInterp):
    """``CubicHermiteInterp`` (level and slope matched at every knot) whose
    above-grid decay toward the limiting line is the same power law as
    ``PowerLawDecayLinearInterp``.

    ``CubicHermiteInterp`` stores its above-top exponential-gap extrapolation
    in ``coeffs[n] = [intercept_limit, slope_limit, gap, slope_diff/gap]``,
    so ``A = gap`` and ``B = -coeffs[n, 3]`` are exactly the objects the
    linear tail uses. This subclass replaces only the above-top functional
    form; interior evaluation and derivatives are inherited untouched (as is
    the scipy-deepcopy immunity of the base class).
    """

    def __init__(
        self,
        x_list,
        y_list,
        dydx_list,
        intercept_limit=None,
        slope_limit=None,
        lower_extrap=False,
        decay_extrap_Q=None,
        q_diagnostics=None,
    ):
        super().__init__(
            x_list, y_list, dydx_list, intercept_limit, slope_limit, lower_extrap
        )
        self._q_override = None if decay_extrap_Q is None else float(decay_extrap_Q)
        self.local_q_diag = q_diagnostics
        self.decay_extrap_form = "exp"
        row = self.coeffs[self.n]
        intercept, slope, A = float(row[0]), float(row[1]), float(row[2])
        B = -float(row[3])
        ok = slope > 0.0 and A > 0.0 and B > 0.0
        if ok:
            pivot = float(self.x_list[-1]) + intercept / slope
            ok = pivot > 0.0
        if not ok:
            warnings.warn(
                "PowerLawDecayCubicHermiteInterp: top knot not strictly below "
                f"the limiting line with approaching slope (A={A:.6g}, "
                f"B={B:.6g}, slope_limit={slope!r}); keeping the exponential "
                "tail for this interpolant."
            )
            return
        self._pl_A = A
        self._pl_intercept = intercept
        self._pl_slope = slope
        self.decay_extrap_pivot = pivot
        if self._q_override is not None and self._q_override > 0.0:
            self.decay_extrap_Q = self._q_override
        else:
            self.decay_extrap_Q = B * pivot
        self.decay_extrap_form = "powerlaw"

    def _pl_gap(self, x_above):
        x_temp = np.asarray(x_above) - self.x_list[self.n - 1]
        decay = self._pl_A * np.exp(
            -self.decay_extrap_Q * np.log1p(x_temp / self.decay_extrap_pivot)
        )
        return x_temp, decay

    def _eval_helper(self, x, out_bot, out_top):
        y = super()._eval_helper(x, out_bot, out_top)
        if getattr(self, "decay_extrap_form", "exp") == "powerlaw" and np.any(out_top):
            _, decay = self._pl_gap(x[out_top])
            y[out_top] = self._pl_intercept + self._pl_slope * x[out_top] - decay
        return y

    def _der_helper(self, x, out_bot, out_top):
        d = super()._der_helper(x, out_bot, out_top)
        if getattr(self, "decay_extrap_form", "exp") == "powerlaw" and np.any(out_top):
            x_temp, decay = self._pl_gap(x[out_top])
            d[out_top] = (
                self._pl_slope
                + self.decay_extrap_Q / (x_temp + self.decay_extrap_pivot) * decay
            )
        return d


class MoMLogGapChartInterp(CubicHermiteInterp):
    """A consumption function represented in Method-of-Moderation style chart
    coordinates: cubic interpolation of ``y = log(gap)`` against
    ``xi = log(x + hNrm)``, where ``gap = MPCmin*(x + hNrm) - c(x)`` is the
    distance below the perfect-foresight bound.

    Constructed from the same knot data as a stock ``CubicHermiteInterp``
    (values and slopes in natural units); the chart stores the transformed
    knots and evaluates back through ``c = MPCmin*(x + h) - exp(y(log(x + h)))``.
    The base-class storage (``x_list``/``y_list``/``dydx_list`` and
    ``distance_criteria``) is kept in NATURAL units, so solver convergence
    semantics are unchanged by adopting the chart.

    Beyond the knot range the chart continues linearly in (xi, y) with its end
    slopes -- above the top this is exactly a power-law gap with the locally
    measured exponent (exposed as ``decay_extrap_Q``); below the bottom knot
    the log-X coordinate compresses the whole [0, x_0] range into a vanishing
    xi interval (X = x + h >= h), so no low-end pathology is reachable.

    Parameters
    ----------
    MPCmin : float
        Slope of the perfect-foresight bound (kappa underbar).
    hNrm : float
        Normalized human wealth h; the bound is ``MPCmin * (x + hNrm)``.
    interp_kind : {"hermite", "pchip"}
        "hermite" uses the exact transformed knot slopes
        ``y' = (x + h) * (MPCmin - c'(x)) / gap``; "pchip" discards them for
        scipy's shape-preserving monotone slopes (see the module docstring's
        estimation caution).
    """

    def __init__(
        self,
        x_list,
        y_list,
        dydx_list,
        MPCmin,
        hNrm,
        interp_kind="hermite",
    ):
        super().__init__(x_list, y_list, dydx_list)
        ok, why = _build_chart_attrs(
            self,
            np.asarray(x_list, float),
            np.asarray(y_list, float),
            np.asarray(dydx_list, float),
            float(MPCmin),
            float(hNrm),
            str(interp_kind),
        )
        if not ok:
            raise ValueError(f"MoMLogGapChartInterp: {why}")

    def _chart_pieces(self, x):
        X = np.maximum(np.asarray(x, dtype=float) + self._chart_h, 1e-300)
        xi = np.log(X)
        lo, hi = self._chart_xi_lo, self._chart_xi_hi
        y = np.empty_like(xi)
        dy = np.empty_like(xi)
        inb = (xi >= lo) & (xi <= hi)
        if np.any(inb):
            y[inb] = self._chart_spline(xi[inb])
            dy[inb] = self._chart_dspline(xi[inb])
        below = xi < lo
        if np.any(below):
            y[below] = self._chart_y_lo + self._chart_s_lo * (xi[below] - lo)
            dy[below] = self._chart_s_lo
        above = xi > hi
        if np.any(above):
            y[above] = self._chart_y_hi + self._chart_s_hi * (xi[above] - hi)
            dy[above] = self._chart_s_hi
        g = np.exp(y)
        return X, g, dy

    def _eval_helper(self, x, out_bot, out_top):
        X, g, _ = self._chart_pieces(x)
        return self._chart_kappa * X - g

    def _der_helper(self, x, out_bot, out_top):
        X, g, dy = self._chart_pieces(x)
        return self._chart_kappa - (g / X) * dy


def _build_chart_attrs(obj, x, c, dc, kappa, h, kind):
    """Compute and install the ``_chart_*`` attributes on ``obj``.

    Shared by the ``MoMLogGapChartInterp`` constructor and
    ``chartify_in_place``. Returns ``(ok, why)``; on refusal nothing is
    installed.
    """
    if not (np.isfinite(h) and kappa > 0.0):
        return False, f"invalid bound (MPCmin={kappa}, hNrm={h})"
    X = x + h
    if not np.all(X > 0.0) or not np.all(np.diff(X) > 0.0):
        return False, "knot X = x + hNrm not positive and increasing"
    g = kappa * X - c
    if not np.all(g > 0.0):
        return False, "non-positive perfect-foresight gap at a knot"
    xi = np.log(X)
    y = np.log(g)
    dydxi = X * (kappa - dc) / g
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(dydxi))):
        return False, "non-finite chart data"
    if kind == "pchip":
        spline = PchipInterpolator(xi, y)
        dspl = spline.derivative()
        s_lo, s_hi = float(dspl(xi[0])), float(dspl(xi[-1]))
    elif kind == "hermite":
        spline = CubicHermiteSpline(xi, y, dydxi)
        s_lo, s_hi = float(dydxi[0]), float(dydxi[-1])
    else:
        return False, f"unknown interp_kind {kind!r}"
    if s_hi >= 0.0:
        return False, f"top end slope {s_hi:.4g} >= 0 (gap not decaying)"
    obj._chart_kappa = kappa
    obj._chart_h = h
    obj._chart_xi_lo = float(xi[0])
    obj._chart_xi_hi = float(xi[-1])
    obj._chart_y_lo = float(y[0])
    obj._chart_y_hi = float(y[-1])
    obj._chart_s_lo = s_lo
    obj._chart_s_hi = s_hi
    obj._chart_spline = spline
    obj._chart_dspline = spline.derivative()
    obj._chart_interp_kind = kind
    obj.decay_extrap = True
    obj.decay_extrap_form = "mom_chart"
    # The chart's above-top continuation is a power law with this exponent.
    obj.decay_extrap_Q = float(-s_hi)
    return True, "ok"


def retrofit_powerlaw(interp, decay_extrap_Q, q_diagnostics=None):
    """IN-PLACE retrofit of a stock ``LinearInterp``/``CubicHermiteInterp``
    (built with limits) into its power-law-tailed counterpart, with an
    explicit exponent.

    A class swap plus the attach attributes: every interior coefficient is
    preserved bit-for-bit and every captured reference to the object sees the
    upgraded tail. Returns True on success; on any unhealthy geometry the
    object is left untouched (stock exponential tail) and False is returned.
    """
    q = None if decay_extrap_Q is None else float(decay_extrap_Q)
    if q is None or q <= 0.0:
        return False
    if isinstance(interp, (PowerLawDecayLinearInterp, PowerLawDecayCubicHermiteInterp)):
        interp._q_override = q
        interp.decay_extrap_Q = q
        interp.local_q_diag = q_diagnostics
        return True
    if type(interp) is CubicHermiteInterp:
        row = interp.coeffs[interp.n]
        intercept, slope, A = float(row[0]), float(row[1]), float(row[2])
        ok = slope > 0.0 and A > 0.0
        if ok:
            pivot = float(interp.x_list[interp.n - 1]) + intercept / slope
            ok = pivot > 0.0
        if not ok:
            warnings.warn(
                "retrofit_powerlaw: unhealthy CubicHermiteInterp top-knot "
                f"geometry (A={A:.6g}, slope_limit={slope:.6g}); keeping the "
                "stock tail."
            )
            return False
        interp.__class__ = PowerLawDecayCubicHermiteInterp
        interp._q_override = q
        interp.local_q_diag = q_diagnostics
        interp._pl_A = A
        interp._pl_intercept = intercept
        interp._pl_slope = slope
        interp.decay_extrap_pivot = pivot
        interp.decay_extrap_Q = q
        interp.decay_extrap_form = "powerlaw"
        return True
    if type(interp) is LinearInterp:
        if not getattr(interp, "decay_extrap", False):
            return False
        intercept = getattr(interp, "intercept_limit", None)
        slope = getattr(interp, "slope_limit", None)
        A = float(interp.decay_extrap_A)
        B = float(interp.decay_extrap_B)
        ok = slope is not None and slope > 0.0 and A > 0.0 and B > 0.0
        if ok:
            pivot = float(interp.x_list[-1]) + intercept / slope
            ok = pivot > 0.0
        if not ok:
            warnings.warn(
                "retrofit_powerlaw: unhealthy LinearInterp top-knot geometry "
                f"(A={A:.6g}, B={B:.6g}); keeping the stock tail."
            )
            return False
        interp.__class__ = PowerLawDecayLinearInterp
        interp._q_override = q
        interp.local_q_diag = q_diagnostics
        interp.decay_extrap_pivot = pivot
        interp.decay_extrap_Q = q
        interp.decay_extrap_form = "powerlaw"
        return True
    return False


def chartify_in_place(interp, MPCmin, hNrm, interp_kind="hermite"):
    """IN-PLACE re-chart of a stock solved ``CubicHermiteInterp`` as a
    ``MoMLogGapChartInterp``. Returns ``(ok, why)``.

    The host's knot values and slopes are evaluated BEFORE the swap (so any
    post-construction coefficient adjustments are represented faithfully); on
    refusal the host is untouched.
    """
    if getattr(interp, "decay_extrap_form", "") == "mom_chart":
        return True, "already chart"
    if type(interp) is not CubicHermiteInterp:
        return (
            False,
            f"host is not a stock CubicHermiteInterp ({type(interp).__name__})",
        )
    x = np.asarray(interp.x_list, dtype=float)
    c = np.asarray(interp(x), dtype=float)
    dc = np.asarray(interp.derivative(x), dtype=float)
    ok, why = _build_chart_attrs(
        interp, x, c, dc, float(MPCmin), float(hNrm), str(interp_kind)
    )
    if not ok:
        return False, why
    interp.__class__ = MoMLogGapChartInterp
    return True, "ok (chartified in place)"
