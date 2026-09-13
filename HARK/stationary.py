"""Stationary solutions of infinite-horizon problems by plug-in fixed-point methods.

Backward induction (``solve_agent`` with ``cycles=0``) reaches the stationary
solution as the limit of a contraction whose modulus can be close to one, most
familiarly at the edge of the growth impatience condition of the consumption
problem, where the limiting MPC goes to zero and thousands of cycles are
needed.  This module separates three things so that faster routes to the same
fixed point can be added without touching the models.

* A **problem**, built by a model, presents its stationary solution as the
  fixed point of a map on a vector (:class:`StationaryProblem`): a starting
  point, bounds, an admissibility check, and the assembly of the model's
  solution object from a converged vector.
* A **method** is a plug-in that finds the fixed point of any such problem and
  knows nothing about models: :func:`iterate` and :func:`anderson` here,
  registered by name in :data:`STATIONARY_METHODS` and extended with
  :func:`register_stationary_method`.
* The **driver**, :func:`solve_stationary_problem`, runs the method named by
  the agent's ``stationary_method``, applies the admissibility check with a
  fallback to plain iteration, and assembles the solution.

For the consumption models the map is the model's own per-period solver,
called with the continuation built from the iterate (:class:`KnotProblem`).
Each evaluation is therefore one backward-induction cycle, and the fixed point
is exactly the one backward induction converges to.
"""

import warnings
from copy import copy

import numpy as np

from HARK.utilities import anderson_accelerate

__all__ = [
    "StationaryProblem",
    "KnotProblem",
    "ValueKnotProblem",
    "STATIONARY_METHODS",
    "register_stationary_method",
    "iterate",
    "anderson",
    "solve_stationary_problem",
]


class StationaryProblem:
    """A stationary solution presented as the fixed point of a map on a vector.

    Subclasses set ``x0`` (the starting vector), optionally ``lo`` and ``hi``
    (bounds every iterate is clipped to), and implement :meth:`sweep`, which
    maps an iterate to the next one, and :meth:`assemble`, which builds the
    model's solution object from a converged iterate.  :meth:`admissible`
    returns None when an iterate may be accepted as a solution and a reason
    otherwise; the default accepts everything.
    """

    x0 = None
    lo = None
    hi = None

    def sweep(self, x):
        raise NotImplementedError

    def assemble(self, x):
        raise NotImplementedError

    def admissible(self, x):
        return None

    def describe(self):
        return type(self).__name__

    def clip(self, x):
        if self.lo is None and self.hi is None:
            return x
        return np.clip(x, self.lo, self.hi)


def iterate(problem, tol, maxit, options=None, verbose=False):
    """Plain iteration ``x <- sweep(x)``: the reference method.

    The same fixed point as every other method, reached at the rate of the
    map's contraction.  Takes no options.
    """
    if options:
        raise ValueError(f"The 'iterate' method takes no options; got {options!r}.")
    x = problem.clip(np.array(problem.x0, dtype=float).ravel())
    move = np.inf
    sweeps = 0
    for sweeps in range(1, maxit + 1):
        x_new = np.asarray(problem.sweep(x), dtype=float).ravel()
        if not np.all(np.isfinite(x_new)):
            raise ValueError(
                f"The map of {problem.describe()} returned non-finite values at "
                f"sweep {sweeps} of plain iteration."
            )
        x_new = problem.clip(x_new)
        move = float(np.max(np.abs(x_new - x)))
        x = x_new
        if verbose and (sweeps <= 3 or sweeps % 100 == 0):
            print(f"iterate sweep {sweeps}: move {move:.3e}")
        if move < tol:
            return x, {
                "method": "iterate",
                "sweeps": sweeps,
                "converged": True,
                "move": move,
                "restarts": 0,
            }
    return x, {
        "method": "iterate",
        "sweeps": sweeps,
        "converged": False,
        "move": move,
        "restarts": 0,
    }


def anderson(problem, tol, maxit, options=None, verbose=False):
    """Anderson mixing of the iterates (:func:`HARK.utilities.anderson_accelerate`).

    Options: ``depth`` (past residual differences mixed, default 20) and
    ``mix_cap`` (largest mixing coefficient, default 10).
    """
    opts = {"depth": 20, "mix_cap": 10.0}
    unknown = set(options or {}) - set(opts)
    if unknown:
        raise ValueError(
            f"Unknown options for the 'anderson' method: {sorted(unknown)}; "
            f"the choices are {sorted(opts)}."
        )
    opts.update(options or {})
    x, info = anderson_accelerate(
        problem.sweep,
        problem.x0,
        depth=opts["depth"],
        tol=tol,
        maxit=maxit,
        lo=problem.lo,
        hi=problem.hi,
        mix_cap=opts["mix_cap"],
        verbose=verbose,
    )
    return x, {"method": "anderson", **info}


STATIONARY_METHODS = {"iterate": iterate, "anderson": anderson}


def register_stationary_method(name, method):
    """Register a fixed-point method under ``name``.

    ``method(problem, tol, maxit, options=None, verbose=False)`` must return
    ``(x, info)`` with ``info`` carrying at least ``sweeps`` (evaluations of
    the map), ``converged`` and ``move`` (the last change of the iterate).
    """
    if not callable(method):
        raise TypeError("A stationary method must be callable.")
    STATIONARY_METHODS[name] = method


def solve_stationary_problem(
    problem, method="anderson", tol=1e-6, maxit=10000, options=None, verbose=False
):
    """Find the fixed point of ``problem`` by ``method`` and assemble the solution.

    If the method fails to converge, or converges to an iterate the problem
    rejects, the driver warns and falls back to plain iteration from the
    problem's starting point, which cannot mix its way to a spurious point.
    A rejected fixed point of plain iteration raises, since the problem then
    has no admissible stationary solution as posed.

    Parameters
    ----------
    problem : StationaryProblem
    method : str
        A key of :data:`STATIONARY_METHODS`.
    tol : float
        Convergence threshold on the largest change of any entry of the
        iterate between sweeps, the measure ``solve_agent`` uses.
    maxit : int
        Maximum number of evaluations of the map, per attempt.
    options : dict, optional
        Method-specific options.
    verbose : bool

    Returns
    -------
    solution : object
        ``problem.assemble(x)`` at the fixed point.
    info : dict
        The method's record plus ``problem``, ``fallback`` (True when plain
        iteration was used after the method failed) and, in that case,
        ``attempted`` (the failed method's record).
    """
    if method not in STATIONARY_METHODS:
        raise ValueError(
            f"Unknown stationary method {method!r}; the registered methods are "
            f"{sorted(STATIONARY_METHODS)}."
        )
    x, info = STATIONARY_METHODS[method](problem, tol, maxit, options, verbose)
    info = {"problem": problem.describe(), **info, "fallback": False}
    reason = problem.admissible(x) if info["converged"] else None
    if method != "iterate" and (not info["converged"] or reason is not None):
        why = (
            f"did not converge in {maxit} sweeps"
            if not info["converged"]
            else f"converged to a rejected point ({reason})"
        )
        warnings.warn(
            f"The {method!r} solve of {problem.describe()} {why}; falling back to "
            "plain iteration."
        )
        attempted = info
        x, info = iterate(problem, tol, maxit, None, verbose)
        info = {
            "problem": problem.describe(),
            **info,
            "fallback": True,
            "attempted": attempted,
        }
        reason = problem.admissible(x) if info["converged"] else None
    if reason is not None:
        raise ValueError(
            f"Plain iteration of {problem.describe()} converged to a rejected "
            f"point: {reason}"
        )
    if not info["converged"]:
        warnings.warn(
            f"The stationary solve of {problem.describe()} did not converge in "
            f"{maxit} sweeps (last move {info['move']:.2e})."
        )
    return problem.assemble(x), info


class KnotProblem(StationaryProblem):
    """The stationary consumption policy as a fixed point of the per-period solver.

    The iterate is consumption at the nodes of the fixed end-of-period asset
    grid in every discrete state, and with cubic interpolation also the
    marginal propensity to consume at the same nodes, stacked into one vector.
    :meth:`sweep` builds the continuation solution from the vector with the
    model's own interpolants, calls the per-period solver once, and reads the
    new values from the knots of the solution it returns.  The fixed point is
    therefore exactly the one backward induction converges to, and each
    evaluation costs one of its cycles.

    The scalars the solver carries from period to period (``mNrmMin``,
    ``hNrm``, ``MPCmin``, ``MPCmax``) are held at their limits, found first by
    accelerating the solver's own scalar recursion on a two-node grid, where
    the policy work is negligible; the assembled solution then carries the
    exact limits rather than the stock loop's still-drifting values.

    Subclasses bind the model's primitives and supply four pieces:
    :meth:`solve_one_period` (the per-period solver on a given grid),
    :meth:`cFuncs` (the per-state consumption functions of a solution),
    :meth:`make_cFunc` (a consumption function from its knots, as the solver
    builds it) and :meth:`make_continuation` (a solution object from per-state
    consumption functions and the scalars).  :meth:`make_vFunc` is needed for
    the value stage.
    """

    scalar_fields = ("mNrmMin", "hNrm", "MPCmin", "MPCmax")

    def __init__(self, aXtraGrid, cubic, seed_solution):
        self.aXtraGrid = np.asarray(aXtraGrid, dtype=float)
        self.J = self.aXtraGrid.size
        self.cubic = bool(cubic)
        self.scalars = self._scalar_limits(seed_solution)
        self.S = self.scalars["hNrm"].size
        # One pass on the real grid from the seed gives, for every state, the
        # fixed bottom knot and the asset nodes the solver works with.
        first = self.solve_one_period(
            self._with_scalars(copy(seed_solution)), self.aXtraGrid, cubic=self.cubic
        )
        knots = [self._knots_of(f) for f in self.cFuncs(first)]
        self.bottom = [
            (m[0], c[0], mpc[0] if mpc is not None else None) for m, c, mpc in knots
        ]
        self.a_nodes = np.vstack([m[1:] - c[1:] for m, c, _ in knots])
        # Seed: perfect-foresight consumption at each node.
        MPCmin = self.scalars["MPCmin"][:, None]
        hNrm = self.scalars["hNrm"][:, None]
        c0 = np.maximum(MPCmin * (self.a_nodes + hNrm) / (1.0 - MPCmin), 1e-10)
        self.warm_start = None
        if self.cubic:
            # The cubic pair is warm-started from the converged linear policy: a
            # valid policy gives a strictly increasing endogenous grid, which the
            # cubic interpolant needs, and the linear fixed point is close.
            shape = c0.shape
            c_lin, self.warm_start = anderson_accelerate(
                lambda v: self._sweep_from(np.reshape(v, shape), None),
                np.ravel(c0),
                tol=1e-9,
                maxit=10000,
                lo=1e-10,
            )
            c0 = np.reshape(c_lin, shape)
            # Seed the MPC with the slope of the linear policy along its own
            # endogenous grid; a constant seed makes the cubic continuation
            # overshoot between knots, which the solver rejects.
            m0 = self.a_nodes + c0
            mpc0 = np.vstack(
                [np.clip(np.gradient(c0[i], m0[i]), 1e-6, 1.0) for i in range(self.S)]
            )
            self.x0 = self._pack(c0, mpc0)
            self.lo = np.concatenate([np.full(c0.size, 1e-10), np.full(c0.size, 1e-6)])
            self.hi = np.concatenate([np.full(c0.size, np.inf), np.ones(c0.size)])
        else:
            self.x0 = self._pack(c0, None)
            self.lo = np.full(c0.size, 1e-10)
        self.x_solution = None

    # ----- pieces the model supplies -------------------------------------

    def solve_one_period(self, solution_next, aXtraGrid, vFuncBool=False, cubic=None):
        raise NotImplementedError

    def cFuncs(self, solution):
        raise NotImplementedError

    def vFuncs(self, solution):
        raise NotImplementedError

    def make_cFunc(self, m, c, mpc, state):
        raise NotImplementedError

    def make_vFunc(self, m, vNvrs, c_on_m, state):
        raise NotImplementedError

    def make_continuation(self, cFuncs, vFuncs=None):
        raise NotImplementedError

    def scalar_value(self, name, values):
        """A scalar field in the shape the solution object carries it."""
        return values

    # ----- generic machinery ----------------------------------------------

    def describe(self):
        return f"{type(self).__name__} ({self.S} state{'s' if self.S != 1 else ''}, {self.J} nodes)"

    def _with_scalars(self, solution, scalars=None):
        scalars = self.scalars if scalars is None else scalars
        for name in self.scalar_fields:
            setattr(solution, name, self.scalar_value(name, scalars[name]))
        return solution

    def _scalar_limits(self, seed_solution):
        """Limits of the solver's scalar recursion, by mixing it on a two-node grid.

        The MPC fields are mixed in logarithms: their recursions have a
        spurious fixed point at zero, which clipping at a positive floor would
        make stable, and in logarithms it sits at minus infinity.
        """
        two_nodes = self.aXtraGrid[[0, -1]]
        names = self.scalar_fields
        sizes = [np.atleast_1d(getattr(seed_solution, n)).size for n in names]
        splits = np.cumsum(sizes)[:-1]
        is_mpc = np.concatenate(
            [np.full(k, n.startswith("MPC")) for n, k in zip(names, sizes)]
        )

        def to_vector(sol):
            v = np.concatenate(
                [np.atleast_1d(getattr(sol, n)).astype(float) for n in names]
            )
            return np.where(is_mpc, np.log(np.maximum(v, 1e-300)), v)

        def to_scalars(v):
            w = np.where(is_mpc, np.exp(v), v)
            return dict(zip(names, np.split(w, splits)))

        def sweep(v):
            try:
                sol = self.solve_one_period(
                    self._with_scalars(copy(seed_solution), to_scalars(v)),
                    two_nodes,
                    cubic=False,
                )
            except (ValueError, FloatingPointError, ZeroDivisionError):
                return np.full(v.shape, np.nan)
            return to_vector(sol)

        v0 = to_vector(seed_solution)
        lo = np.where(is_mpc, -60.0, -np.inf)
        hi = np.where(is_mpc, 0.0, np.inf)
        lo[np.concatenate([np.full(k, n == "hNrm") for n, k in zip(names, sizes)])] = (
            0.0
        )
        # A few plain steps first: from the terminal seed the map is nearly a
        # translation in the log MPC, which mixing cannot accelerate and can
        # overshoot into.  1e-11 absolute: human wealth is of order 100, so a
        # tighter threshold sits at the rounding floor.
        v = v0
        for _ in range(10):
            v_new = sweep(v)
            if not np.all(np.isfinite(v_new)):
                break
            v = np.clip(v_new, lo, hi)
        v, info = anderson_accelerate(
            sweep, v, depth=10, tol=1e-11, maxit=10000, lo=lo, hi=hi
        )
        if not info["converged"] or np.any(v[is_mpc] < -40.0):
            v = v0
            for _ in range(200000):
                v_new = sweep(v)
                if not np.all(np.isfinite(v_new)):
                    raise ValueError(
                        "The solver rejected the scalar recursion of the stationary "
                        "problem; check the model's parameters."
                    )
                v_new = np.clip(v_new, lo, hi)
                done = np.max(np.abs(v_new - v)) < 1e-11
                v = v_new
                if done:
                    break
            else:
                raise ValueError(
                    "The scalar recursions of the stationary problem do not "
                    "converge (human wealth or the limiting MPC keeps growing); "
                    "check the finite human wealth and growth impatience conditions."
                )
        return to_scalars(v)

    @staticmethod
    def _knots_of(cFunc):
        f = cFunc.functions[0]
        mpc = getattr(f, "dydx_list", None)
        return (
            np.asarray(f.x_list, dtype=float),
            np.asarray(f.y_list, dtype=float),
            None if mpc is None else np.asarray(mpc, dtype=float),
        )

    def _pack(self, c, mpc):
        return (
            np.ravel(c) if mpc is None else np.concatenate([np.ravel(c), np.ravel(mpc)])
        )

    def _unpack(self, x):
        n = self.S * self.J
        c = np.reshape(x[:n], (self.S, self.J))
        mpc = np.reshape(x[n:], (self.S, self.J)) if self.cubic else None
        return c, mpc

    def continuation(self, c, mpc=None, vFuncs=None):
        """The continuation solution whose consumption knots are ``c`` (and ``mpc``)."""
        cFuncs = []
        for i in range(self.S):
            m_bottom, c_bottom, mpc_bottom = self.bottom[i]
            m = np.insert(self.a_nodes[i] + c[i], 0, m_bottom)
            c_for = np.insert(c[i], 0, c_bottom)
            mpc_for = None
            if mpc is not None:
                mpc_for = np.insert(mpc[i], 0, mpc_bottom)
                if np.any(np.diff(m) <= 0.0):
                    # An iterate away from the fixed point can make the
                    # endogenous grid non-monotone, which the cubic interpolant
                    # rejects; the converged grid is strictly increasing, so
                    # this never acts on it.
                    m = np.maximum.accumulate(m) + 1e-12 * np.arange(m.size)
            cFuncs.append(self.make_cFunc(m, c_for, mpc_for, i))
        return self._with_scalars(self.make_continuation(cFuncs, vFuncs))

    def _sweep_from(self, c, mpc):
        """One application of the per-period solver; linear when ``mpc`` is None."""
        cubic = mpc is not None
        solution = self.solve_one_period(
            self.continuation(c, mpc), self.aXtraGrid, cubic=cubic
        )
        knots = [self._knots_of(f) for f in self.cFuncs(solution)]
        c_new = np.vstack([k[1][1:] for k in knots])
        mpc_new = np.vstack([k[2][1:] for k in knots]) if cubic else None
        return self._pack(c_new, mpc_new)

    def sweep(self, x):
        try:
            return self._sweep_from(*self._unpack(x))
        except (ValueError, FloatingPointError, ZeroDivisionError):
            # The solver rejects an iterate outside its domain (consumption
            # driven negative between knots, a degenerate endogenous grid);
            # a non-finite image makes the method restart from a good iterate.
            return np.full(np.shape(x), np.nan)

    def assemble(self, x):
        self.x_solution = np.array(x, dtype=float)
        c, mpc = self._unpack(self.x_solution)
        return self.solve_one_period(self.continuation(c, mpc), self.aXtraGrid)

    def admissible(self, x):
        """Reject a top knot above the perfect-foresight line in any state.

        A stationary buffer-stock policy lies below ``c = MPCmin * (m + hNrm)``
        and approaches it from below, so a converged iterate whose top knot sits
        above the line is a fixed point of the truncated grid, not a solution.
        """
        c, _ = self._unpack(x)
        c_top = c[:, -1]
        m_top = self.a_nodes[:, -1] + c_top
        line = self.scalars["MPCmin"] * (m_top + self.scalars["hNrm"])
        excess = (c_top - line) / np.maximum(1.0, np.abs(line))
        worst = int(np.argmax(excess))
        if excess[worst] > 1e-8:
            return (
                f"the top knot of state {worst} is {100.0 * excess[worst]:.2g} "
                "percent above the perfect-foresight line"
            )
        lowest = int(np.argmin(c_top / np.maximum(line, 1e-300)))
        if c_top[lowest] < 0.1 * line[lowest]:
            # Consumption clipped at its floor is a fixed point of the map too;
            # a real policy runs within a few percent of the line at the top.
            return (
                f"the top knot of state {lowest} is below a tenth of the "
                "perfect-foresight line, a degenerate fixed point"
            )
        return None

    def value_problem(self):
        """The value function on the converged policy, as a problem of its own."""
        if self.x_solution is None:
            raise ValueError("Solve the policy before posing the value problem.")
        return ValueKnotProblem(self)


class ValueKnotProblem(StationaryProblem):
    """The value function on a fixed policy as a fixed point of the same solver.

    The iterate is the logarithm of the inverse-utility value at
    ``m = mNrmMin + aXtraGrid`` in every state.  :meth:`sweep` attaches the
    value continuation built from it to the converged policy continuation of a
    :class:`KnotProblem`, calls the per-period solver with ``vFuncBool=True``,
    and reads the new inverse values from the knots of the value function it
    returns.  Logarithms because the inverse value spans orders of magnitude
    across the grid and has a degenerate fixed point at zero, which mixing in
    levels could reach; the convergence threshold is therefore relative.
    """

    def __init__(self, policy):
        self.policy = policy
        c, mpc = policy._unpack(policy.x_solution)
        self.c, self.mpc = c, mpc
        self.m_grid = policy.scalars["mNrmMin"][:, None] + policy.aXtraGrid[None, :]
        base = policy.continuation(c, mpc)
        self.c_on_m = np.vstack(
            [f(self.m_grid[i]) for i, f in enumerate(policy.cFuncs(base))]
        )
        # Seed: the perfect-foresight value with the limiting MPC, whose inverse
        # is consumption scaled by MPCmin ** (1 / (CRRA - 1)); exact in the
        # limit of large m and the right order of magnitude everywhere.
        CRRA = getattr(policy, "CRRA", None)
        if CRRA is None or CRRA == 1.0:
            seed = self.c_on_m
        else:
            seed = self.c_on_m * policy.scalars["MPCmin"][:, None] ** (
                1.0 / (CRRA - 1.0)
            )
        self.x0 = np.log(np.maximum(np.ravel(seed), 1e-300))
        self.lo = np.full(self.x0.size, -600.0)

    def describe(self):
        return f"value of {self.policy.describe()}"

    def _continuation(self, vNvrs):
        vFuncs = [
            self.policy.make_vFunc(self.m_grid[i], vNvrs[i], self.c_on_m[i], i)
            for i in range(self.policy.S)
        ]
        return self.policy.continuation(self.c, self.mpc, vFuncs)

    def _solve(self, x):
        vNvrs = np.exp(np.reshape(x, self.m_grid.shape))
        return self.policy.solve_one_period(
            self._continuation(vNvrs), self.policy.aXtraGrid, vFuncBool=True
        )

    def sweep(self, x):
        try:
            solution = self._solve(x)
        except (ValueError, FloatingPointError, ZeroDivisionError):
            return np.full(np.shape(x), np.nan)
        vNvrs = np.concatenate(
            [
                np.asarray(f.vFuncNvrs.y_list, dtype=float)[1:]
                for f in self.policy.vFuncs(solution)
            ]
        )
        return np.log(np.maximum(vNvrs, 1e-300))

    def assemble(self, x):
        return self._solve(x)
