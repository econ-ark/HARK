"""
Functions for building heterogeneous agent sequence space Jacobian matrices from
HARK AgentType instances. The top-level functions are accessible as methods on
AgentType itself.
"""

from time import time
from copy import deepcopy
import numpy as np
from HARK._numba import njit


def _prepare_ssj_computation(
    agent, outcomes, grids, norm, solved, verbose, newborn_growth=None
):
    """
    Shared setup for make_basic_SSJ_matrices and calc_shock_response_manually.
    Validates the agent, normalizes outcomes, optionally solves the long run model,
    builds transition matrices, and finds the steady state distribution.

    Parameters
    ----------
    agent : AgentType
        Agent for which setup should be performed.
    outcomes : str or [str]
        Names of outcome variables of interest (will be normalized to a list).
    grids : dict
        Dictionary of dictionaries with discretizing grid information.
    norm : str or None
        Name of the model variable to normalize by for Harmenberg aggregation.
    solved : bool
        Whether the agent's model has already been solved.
    verbose : bool
        Whether to display timing/progress to screen.
    newborn_growth : float or None
        Growth factor of the normalizing level that newborns inherit (see
        AgentSimulator.make_transition_matrices). None takes the agent's
        PermGroFacAgg, or 1.0 if it has none.

    Returns
    -------
    setup : dict
        Dictionary containing:
        - newborn_growth : float     newborn_growth, with None resolved
        - outcomes : [str]           normalized list of outcome variable names
        - no_list : bool             True if outcomes was passed as a single string
        - simulator_backup : object  agent._simulator backup, or None if not present
        - LR_soln : object           deepcopy of the long run solution
        - X : object                 reference to agent._simulator
        - LR_trans : np.array        long run transition matrix
        - LR_period : object         the long run period object from the simulator
        - LR_outcomes : [np.array]   outcome matrices in the long run model
        - outcome_grids : [np.array] outcome grids in the long run model
        - SS_dstn : np.array         steady state distribution
    """
    if (agent.cycles > 0) or (agent.T_cycle != 1):
        raise ValueError(
            "This function is only compatible with one period infinite horizon models!"
        )
    if not isinstance(outcomes, list):
        outcomes = [outcomes]
        no_list = True
    else:
        no_list = False
    if newborn_growth is None:
        newborn_growth = float(
            np.asarray(getattr(agent, "PermGroFacAgg", 1.0)).ravel()[0]
        )

    # Store the simulator if it exists
    if hasattr(agent, "_simulator"):
        simulator_backup = agent._simulator
    else:
        simulator_backup = None

    # Solve the long run model if it wasn't already
    if not solved:
        t0 = time()
        agent.solve()
        t1 = time()
        if verbose:
            print(
                "Solving the long run model took {:.3f}".format(t1 - t0) + " seconds."
            )
    LR_soln = deepcopy(agent.solution[0])

    # Construct the transition matrix for the long run model
    t0 = time()
    agent.initialize_sym()
    X = agent._simulator  # for easier referencing
    X.make_transition_matrices(grids, norm, newborn_growth=newborn_growth)
    LR_trans = X.trans_arrays[0].copy()  # the transition matrix in LR model
    LR_period = X.periods[0]
    LR_outcomes = []
    outcome_grids = []
    for var in outcomes:
        if var not in X.periods[0].matrices:
            raise ValueError(
                "Outcome " + var + " was requested but has no transition matrix."
            )
        if var not in X.periods[0].grids:
            raise ValueError(
                "Outcome " + var + " was requested but no grid was provided!"
            )
        LR_outcomes.append(X.periods[0].matrices[var])
        outcome_grids.append(X.periods[0].grids[var])
    t1 = time()
    if verbose:
        print(
            "Making the transition matrix for the long run model took {:.3f}".format(
                t1 - t0
            )
            + " seconds."
        )

    # Find the steady state for the long run model
    t0 = time()
    X.find_steady_state()
    SS_dstn = X.steady_state_dstn.copy()
    t1 = time()
    if verbose:
        print(
            "Finding the long run steady state took {:.3f}".format(t1 - t0)
            + " seconds."
        )

    return {
        "newborn_growth": newborn_growth,
        "outcomes": outcomes,
        "no_list": no_list,
        "simulator_backup": simulator_backup,
        "LR_soln": LR_soln,
        "X": X,
        "LR_trans": LR_trans,
        "LR_period": LR_period,
        "LR_outcomes": LR_outcomes,
        "outcome_grids": outcome_grids,
        "SS_dstn": SS_dstn,
    }


def _perturb_shock(agent, shock):
    """
    Retrieve and validate the named shock attribute on an agent, returning the
    scalar base value and a flag indicating whether it is stored as a singleton list.

    Parameters
    ----------
    agent : AgentType
        Agent whose shock attribute will be read.
    shock : str
        Name of the shock attribute to perturb.

    Returns
    -------
    base_shock_value : float or np.floating
        The scalar value of the shock attribute (unwrapped from a list if needed).
    shock_is_list : bool
        True if the shock attribute is stored as a list, False otherwise.
    """
    if not hasattr(agent, shock):
        raise ValueError(
            "The agent doesn't have anything called " + shock + " to perturb!"
        )
    base = getattr(agent, shock)
    if isinstance(base, list):
        base_shock_value = base[0]
        shock_is_list = True
    else:
        base_shock_value = base
        shock_is_list = False
    if isinstance(base_shock_value, bool) or not isinstance(
        base_shock_value, (float, np.floating)
    ):
        raise TypeError(
            "The shock attribute '"
            + shock
            + "' must be a scalar float (Python float or np.floating), but got "
            + str(type(base_shock_value))
            + "."
        )
    return base_shock_value, shock_is_list


def _restore_agent(agent, LR_soln, simulator_backup):
    """
    Reset the agent to its original long run state after SSJ or impulse response
    computation, restoring the simulator backup if one was saved.

    Parameters
    ----------
    agent : AgentType
        Agent to restore.
    LR_soln : object
        Long run solution to reinstall as the agent's solution.
    simulator_backup : object or None
        The original simulator to restore, or None if no backup was stored.
    """
    agent.solution = [LR_soln]
    agent.cycles = 0
    agent._simulator.reset()
    if simulator_backup is not None:
        agent._simulator = simulator_backup
    else:
        del agent._simulator


def make_basic_SSJ_matrices(
    agent,
    shock,
    outcomes,
    grids,
    eps=1e-4,
    T_max=300,
    norm=None,
    solved=False,
    construct=True,
    offset=False,
    verbose=False,
    newborn_growth=None,
    ghost=False,
):
    """
    Constructs one or more sequence space Jacobian (SSJ) matrices for specified
    outcomes over one shock variable. It is "basic" in the sense that it only
    works for "one period infinite horizon" models, as in the original SSJ paper.

    Parameters
    ----------
    agent : AgentType
        Agent for which the SSJ(s) should be constructed. Must have T_cycle=1
        and cycles=0, or the function will throw an error. Must have a model
        file defined or this won't work at all.
    shock : str
        Name of the variable that Jacobians will be computed with respect to.
        It does not need to be a "shock" in a modeling sense, but it must be a
        single-valued parameter (possibly a singleton list) that can be changed.
    outcomes : str or [str]
        Names of outcome variables of interest; an SSJ matrix will be constructed
        for each variable named here. If a single string is passed, the output
        will be a single np.array. If a list of strings are passed, the output
        will be a list of SSJ matrices in the order specified here.
    grids : dict
        Dictionary of dictionaries with discretizing grid information. The grids
        should include all arrival variables other than those that are normalized
        out. They should also include all variables named in outcomes, except
        outcomes that are continuation variables that remap to arrival variables.
        Grid specification must include number of nodes N, should also include
        min and max if the variable is continuous.
    eps : float
        Amount by which to perturb the shock variable. The default is 1e-4.
    T_max : int
        Size of the SSJ matrices: the maximum number of periods to consider.
        The default is 300.
    norm : str or None
        Name of the model variable whose realized value weights the probability
        masses, for Harmenberg (income-weighted) aggregation, if any. This lets
        the grid over permanent income be omitted as an explicit state. For HARK's
        permanent-income models name the growth factor of the *level*, 'G'
        (PermGroFac * PermShk); the shock alone, 'PermShk', is exact only without
        deterministic growth or without mortality, or when newborns inherit the
        growth (see newborn_growth) -- otherwise the stationary distribution
        overweights the young and normalized aggregates are biased.
    solved : bool
        Whether the agent's model has already been solved. If False (default),
        it will be solved as the very first step. Solving the agent's long run
        model before constructing SSJ matrices has the advantage of not needing
        to re-solve the long run model for each shock variable.
    construct : bool
        Whether the construct (update) method should be run after the shock is
        updated. The default is True, which is the "safe" option. If the shock
        variable is a parameter that enters the model only *directly*, rather
        than being used to build a more complex model input, then this can be
        set to False to save a (very) small amount of time during computation.
        If it is set to False improperly, the SSJs will be very wrong, potentially
        just zero everywhere.
    offset : bool
        Whether the shock variable is "offset in time" for the solver, with a
        default of False. This should be set to True if the named shock variable
        (or the constructed model input that it affects) is indexed by t+1 from
        the perspective of the solver. For example, the period t solver for the
        ConsIndShock model takes in risk free interest factor Rfree as an argument,
        but it represents the value of R that will occur at the start of t+1.
    verbose : bool
        Whether to display timing/progress to screen. The default is False.
    newborn_growth : float or None
        Per-period growth factor of the normalizing level that newborns inherit
        (a common trend); the weights under norm are divided by it. None (the
        default) takes the agent's PermGroFacAgg, which is 1.0 for most HARK
        agents: newborns arrive at a fixed level. Setting it equal to PermGroFac
        makes all growth a trend that newborns inherit, under which weighting by
        the shock alone is exact.
    ghost : bool
        Whether to difference the perturbed finite-horizon solution against an
        *unperturbed* finite-horizon solution of the same length, solved from the
        same long run solution (a "ghost run"), rather than against the long run
        solution itself. The default is False. The two baselines differ by the
        residual of the long run solve, which backward induction keeps converging
        away from over the finite horizon; for an impatient household solved at
        the default tolerance the residual is negligible, but for a very patient
        one (growth-patience factor close to one) it is divided by eps and shows
        up as spurious entries far from the diagonal of the SSJ. The ghost run
        removes it exactly, at the cost of one extra finite-horizon solve and
        matrix build. Its drift from the long run solution is reported when
        verbose is True.

    Returns
    -------
    SSJ : np.array or [np.array]
        One or more sequence space Jacobian arrays over the outcome variables
        with respect to the named shock variable.
    """
    setup = _prepare_ssj_computation(
        agent, outcomes, grids, norm, solved, verbose, newborn_growth
    )
    newborn_growth = setup["newborn_growth"]
    outcomes = setup["outcomes"]
    no_list = setup["no_list"]
    simulator_backup = setup["simulator_backup"]
    LR_soln = setup["LR_soln"]
    LR_trans = setup["LR_trans"]
    LR_period = setup["LR_period"]
    LR_outcomes = setup["LR_outcomes"]
    outcome_grids = setup["outcome_grids"]
    SS_dstn = setup["SS_dstn"]

    try:
        base_shock_value, shock_is_list = _perturb_shock(agent, shock)
        Tm1_soln, period_Tm1, period_T = _solve_perturbed_Tm1(
            agent,
            shock,
            base_shock_value,
            shock_is_list,
            eps,
            construct,
            LR_soln,
            verbose,
        )
        _solve_finite_horizon(
            agent,
            shock,
            base_shock_value,
            shock_is_list,
            T_max,
            construct,
            Tm1_soln,
            verbose,
        )
        TmX_trans, TmX_outcomes = _build_finite_horizon_matrices(
            agent,
            period_Tm1,
            period_T,
            LR_period,
            grids,
            norm,
            offset,
            T_max,
            outcomes,
            verbose,
            newborn_growth,
        )

        J = len(outcomes)
        K = SS_dstn.size
        if ghost:
            # The ghost run: the same finite horizon with no perturbation, from
            # the same long run solution. Only its pushes of the steady state
            # distribution (one per period) are kept, not its matrices.
            base_D, base_Y = _run_ghost_chain(
                agent,
                shock,
                base_shock_value,
                shock_is_list,
                construct,
                LR_soln,
                LR_period,
                grids,
                norm,
                offset,
                T_max,
                outcomes,
                outcome_grids,
                SS_dstn,
                newborn_growth,
                verbose,
            )
            if verbose:
                _report_ghost_drift(
                    base_D, base_Y, LR_trans, LR_outcomes, outcome_grids, SS_dstn
                )
            D_dstn_array, dY_news_array = _compute_finite_horizon_derivatives_vs_ghost(
                T_max,
                J,
                TmX_trans,
                TmX_outcomes,
                base_D,
                base_Y,
                outcome_grids,
                SS_dstn,
                verbose,
            )
        else:
            D_dstn_array, dY_news_array = _compute_finite_horizon_derivatives(
                T_max,
                J,
                outcomes,
                TmX_trans,
                TmX_outcomes,
                LR_trans,
                LR_outcomes,
                outcome_grids,
                SS_dstn,
                verbose,
            )

        FN = _build_fake_news(
            T_max,
            J,
            K,
            dY_news_array,
            D_dstn_array,
            LR_trans,
            LR_outcomes,
            outcome_grids,
            verbose,
        )

        # Construct the SSJ matrices, one for each outcome variable
        t0 = time()
        SSJ_array = FN.copy()
        for t in range(1, T_max):
            SSJ_array[:, 1:, t] += SSJ_array[:, :-1, t - 1]
        SSJ_array /= eps
        if norm is not None:
            # The outcome arrays carry the growth of the normalizing level within
            # the period, so the responses above are per unit of the arrival-state
            # level; divide by the steady-state mass growth to express them per
            # unit of the period's (post-growth) level, the Harmenberg aggregate.
            SSJ_array /= float(np.sum(np.dot(SS_dstn, LR_outcomes[0])))
        SSJ = [SSJ_array[j, :, :] for j in range(J)]  # unpack into a list of arrays
        _log_timing(verbose, "Constructing the sequence space Jacobians", t0, time())

        if no_list:
            return SSJ[0]
        else:
            return SSJ

    finally:
        _restore_agent(agent, LR_soln, simulator_backup)


def _log_timing(verbose, label, t0, t1):
    if verbose:
        print(label + " took {:.3f}".format(t1 - t0) + " seconds.")


def _shock_value(base_value, shock_is_list, eps=0.0):
    return [base_value + eps] if shock_is_list else base_value + eps


def _solve_perturbed_Tm1(
    agent, shock, base_shock_value, shock_is_list, eps, construct, LR_soln, verbose
):
    t0 = time()
    agent.cycles = 1
    agent.assign_parameters(
        **{shock: _shock_value(base_shock_value, shock_is_list, eps)}
    )
    if construct:
        agent.update()
    agent.solve(from_solution=LR_soln)
    agent.initialize_sym()
    Tm1_soln = deepcopy(agent.solution[0])
    period_Tm1 = agent._simulator.periods[0]
    period_T = agent._simulator.periods[-1]
    _log_timing(verbose, "Solving period T-1 with a perturbed variable", t0, time())
    return Tm1_soln, period_Tm1, period_T


def _solve_finite_horizon(
    agent, shock, base_shock_value, shock_is_list, T_max, construct, Tm1_soln, verbose
):
    t0 = time()
    agent.cycles = T_max - 1
    agent.assign_parameters(**{shock: _shock_value(base_shock_value, shock_is_list)})
    if construct:
        agent.update()
    agent.solve(from_solution=Tm1_soln)
    _log_timing(
        verbose,
        "Solving the finite horizon model for " + str(T_max - 1) + " more periods",
        t0,
        time(),
    )


def _build_finite_horizon_matrices(
    agent,
    period_Tm1,
    period_T,
    LR_period,
    grids,
    norm,
    offset,
    T_max,
    outcomes,
    verbose,
    newborn_growth=1.0,
):
    t0 = time()
    agent.initialize_sym()
    X = agent._simulator
    X.periods[-1] = period_Tm1
    if offset:
        for name in X.periods[-1].content.keys():
            if name not in X.solution:
                X.periods[-1].content[name] = LR_period.content[name]
        X.periods[-1].distribute_content()
        X.periods = X.periods[1:] + [period_T]
    X.make_transition_matrices(
        grids, norm, fake_news_timing=True, newborn_growth=newborn_growth
    )
    TmX_trans = deepcopy(X.trans_arrays)
    TmX_outcomes = [
        [X.periods[t].matrices[var] for var in outcomes] for t in range(T_max)
    ]
    _log_timing(
        verbose,
        "Constructing transition arrays for the finite horizon model",
        t0,
        time(),
    )
    return TmX_trans, TmX_outcomes


def _compute_finite_horizon_derivatives(
    T_max,
    J,
    outcomes,
    TmX_trans,
    TmX_outcomes,
    LR_trans,
    LR_outcomes,
    outcome_grids,
    SS_dstn,
    verbose,
):
    t0 = time()
    D_dstn_array = calc_derivs_of_state_dstns(
        T_max, J, np.array(TmX_trans), LR_trans, SS_dstn
    )
    dY_news_array = np.empty((T_max, J))
    for j in range(J):
        temp_outcomes = np.array([TmX_outcomes[t][j] for t in range(T_max)])
        dY_news_array[:, j] = calc_derivs_of_policy_funcs(
            T_max, temp_outcomes, LR_outcomes[j], outcome_grids[j], SS_dstn
        )
    _log_timing(verbose, "Calculating derivatives by first differences", t0, time())
    return D_dstn_array, dY_news_array


def _run_ghost_chain(
    agent,
    shock,
    base_shock_value,
    shock_is_list,
    construct,
    LR_soln,
    LR_period,
    grids,
    norm,
    offset,
    T_max,
    outcomes,
    outcome_grids,
    SS_dstn,
    newborn_growth,
    verbose,
):
    """
    Solve the unperturbed finite-horizon chain (the ghost run) exactly as the
    perturbed one is solved -- period T-1 from the long run solution, then the
    remaining periods -- build its transition and outcome matrices with the same
    timing, and return the per-period pushes of the steady state distribution
    through them: base_D[t] = trans_t^T SS_dstn and base_Y[t, j] = the average of
    outcome j after period t's policies. The matrices themselves are discarded.
    """
    t0 = time()
    Tm1_soln, period_Tm1, period_T = _solve_perturbed_Tm1(
        agent,
        shock,
        base_shock_value,
        shock_is_list,
        0.0,
        construct,
        LR_soln,
        False,
    )
    _solve_finite_horizon(
        agent,
        shock,
        base_shock_value,
        shock_is_list,
        T_max,
        construct,
        Tm1_soln,
        False,
    )
    ghost_trans, ghost_outcomes = _build_finite_horizon_matrices(
        agent,
        period_Tm1,
        period_T,
        LR_period,
        grids,
        norm,
        offset,
        T_max,
        outcomes,
        False,
        newborn_growth,
    )
    J = len(outcomes)
    base_D = np.empty((T_max, SS_dstn.size))
    base_Y = np.empty((T_max, J))
    for t in range(T_max):
        base_D[t, :] = np.dot(np.asarray(ghost_trans[t]).T, SS_dstn)
        for j in range(J):
            base_Y[t, j] = np.dot(
                np.dot(np.asarray(ghost_outcomes[t][j]).T, SS_dstn), outcome_grids[j]
            )
    del ghost_trans, ghost_outcomes
    _log_timing(verbose, "Solving and pushing through the ghost run", t0, time())
    return base_D, base_Y


def _report_ghost_drift(base_D, base_Y, LR_trans, LR_outcomes, outcome_grids, SS_dstn):
    """
    Print how far the ghost run's pushes drift from the long run solution's own
    push: the part of the naive (steady-state-differenced) derivatives that is
    the long run solve's residual rather than a response to the perturbation.
    """
    LR_D = np.dot(LR_trans.T, SS_dstn)
    LR_Y = np.array(
        [
            np.dot(np.dot(mat.T, SS_dstn), grid)
            for mat, grid in zip(LR_outcomes, outcome_grids)
        ]
    )
    drift_D = np.max(np.abs(base_D - LR_D[None, :]))
    drift_Y = np.max(np.abs(base_Y - LR_Y[None, :]))
    print(
        "Ghost run drift from the long run solution: max |change in the pushed "
        "distribution| = {:.3e}, max |change in average outcomes| = {:.3e} (the "
        "latter divided by eps is the spurious SSJ entry the ghost removes).".format(
            drift_D, drift_Y
        )
    )


def _compute_finite_horizon_derivatives_vs_ghost(
    T_max,
    J,
    TmX_trans,
    TmX_outcomes,
    base_D,
    base_Y,
    outcome_grids,
    SS_dstn,
    verbose,
):
    """
    The finite-horizon derivatives differenced against the ghost run's pushes
    instead of the long run solution's (same conventions as
    _compute_finite_horizon_derivatives, whose indexing they share).
    """
    t0 = time()
    D_dstn_array = calc_derivs_of_state_dstns_vs_ghost(
        T_max, np.array(TmX_trans), base_D, SS_dstn
    )
    dY_news_array = np.empty((T_max, J))
    for j in range(J):
        temp_outcomes = np.array([TmX_outcomes[t][j] for t in range(T_max)])
        dY_news_array[:, j] = calc_derivs_of_policy_funcs_vs_ghost(
            T_max, temp_outcomes, base_Y[:, j].copy(), outcome_grids[j], SS_dstn
        )
    _log_timing(
        verbose, "Calculating derivatives by first differences (ghost)", t0, time()
    )
    return D_dstn_array, dY_news_array


def _build_fake_news(
    T_max,
    J,
    K,
    dY_news_array,
    D_dstn_array,
    LR_trans,
    LR_outcomes,
    outcome_grids,
    verbose,
):
    t0 = time()
    expectation_vectors = np.empty((J, K))
    for j in range(J):
        expectation_vectors[j, :] = np.dot(LR_outcomes[j], outcome_grids[j])
    FN = make_fake_news_matrices(
        T_max,
        J,
        dY_news_array,
        D_dstn_array,
        LR_trans.T,
        expectation_vectors.copy(),
    )
    _log_timing(verbose, "Constructing the fake news matrices", t0, time())
    return FN


def _lc_surviving_mass(matrices, K, norm):
    """
    Mass carried by the survivors from each arrival state of one life-cycle
    period: the survival probability, or under norm the survivors' income-
    weighted mass (which includes the growth of the normalizing level realized
    within the period, relative to newborn_growth), as AgentSimulator.
    make_transition_matrices reads it. Ones when the model has no mortality.
    """
    if "dead" not in matrices:
        return np.ones(K)
    dead = matrices["dead"]
    if norm is None:
        return 1.0 - dead[:, 1]
    return dead[:, 0]


def _lc_cohort_dstns(newborn_dstn, trans_by_age, surv_by_age):
    """
    Arrival distributions of one birth cohort at each age, carrying its mass:
    newborns have mass one, and the mass that survives from each arrival state
    (surv_by_age[a], per state) moves along the survivors' transition
    (trans_by_age[a], rows conditional on survival).
    """
    dstns = [np.asarray(newborn_dstn, dtype=float).copy()]
    for a in range(len(trans_by_age) - 1):
        dstns.append(np.dot(surv_by_age[a] * dstns[a], trans_by_age[a]))
    return dstns


def make_flat_LC_SSJ_matrices(
    agent,
    shock,
    outcomes,
    grids,
    eps=1e-4,
    T_max=100,
    norm=None,
    trend=None,
    pop_gro=1.0,
    prod_gro=1.0,
    solved=False,
    age_agg=True,
    construct=True,
    offset=False,
    verbose=False,
    newborn_growth=1.0,
    ghost=False,
):
    """
    Constructs one or more sequence space Jacobian (SSJ) matrices for specified
    outcomes over one shock variable. This version of the function is for life-
    cycle models with "flat" demographic dynamics: the long run distribution of
    ages is stable. This requires that survival probability is not endogenous to
    agent actions and thus cannot be affected by shocks.

    "Flat" demographic dynamics permit two very specific growth trends: constant
    population growth and constant aggregate productivity growth.

    This algorithm (and some of the code) are directly taken from Mateo Velasquez
    and Bence Bardoczy's paper on life-cycle Jacobians, and its accompanying repo.

    Parameters
    ----------
    agent : AgentType
        Agent for which the SSJ(s) should be constructed. Must have cycles = 1
        or the function will throw an error. Must have a model file defined or
        this won't work at all.
    shock : str
        Name of the variable that Jacobians will be computed with respect to.
        It does not need to be a "shock" in a modeling sense, but it must be a
        single-valued parameter (possibly a singleton list) that can be changed.
    outcomes : str or [str]
        Names of outcome variables of interest; an SSJ matrix will be constructed
        for each variable named here. If a single string is passed, the output
        will be a single np.array. If a list of strings are passed, the output
        will be a list of SSJ matrices in the order specified here.
    grids : dict
        Dictionary of dictionaries with discretizing grid information. The grids
        should include all arrival variables other than those that are normalized
        out. They should also include all variables named in outcomes, except
        outcomes that are continuation variables that remap to arrival variables.
        Grid specification must include number of nodes N, should also include
        min and max if the variable is continuous.
    eps : float
        Amount by which to perturb the shock variable. The default is 1e-4.
    T_max : int
        Size of the SSJ matrices: the maximum number of periods to consider.
        The default is 100.
    norm : str or None
        Name of the block variable whose realized value weights each probability
        mass, for Harmenberg (income-weighted) aggregation, if any; it lets the
        grid over permanent income be omitted as an explicit state. Two usages
        are consistent, and give the same Jacobians for shocks that leave the
        income process alone: ``norm='G'`` (the growth factor of the normalized
        level, ``PermGroFac * PermShk`` in HARK's model files) with ``trend=None``,
        under which each cohort's income-weighted mass carries the deterministic
        growth by age and responds to shocks that change it; or ``norm='PermShk'``
        with ``trend='PermGroFac'``, under which the shock alone weights the
        masses and the deterministic growth is applied to the outcomes by age
        afterwards. Naming a variable that already includes the growth *and*
        passing ``trend`` counts it twice. Under ``norm`` the outcome arrays carry
        the growth of the normalizing level realized within the period, and the
        Jacobians are expressed per unit of the newborn cohort's first-period
        level (the same convention as ``trend``, whose factor starts at one).
    trend : str or None
        Name of the model variable that represents the "normalization trend factor"
        for the outcomes. For example, most consumption-saving models in HARK are
        normalized by permanent income, which grows by factor `PermGroFac` each
        period of the life-cycle; `PermGroFac` should be named as the `trend` for
        any model outputs that are normalized by permanent income (i.e. they have
        `Nrm` in their name) when ``norm`` weights by the shock alone. In contrast,
        if you wanted the fraction of agents that have `Lbr > 0.0` for
        `LaborIntMargConsumerType`, a binary indicator for this outcome should
        *not* have `PermGroFac` named as the `trend`-- you don't want to upweight
        people who have accumulated more income growth more when calculating the
        employment rate! Leave it None when ``norm`` already carries the growth.
    pop_gro : float
        Constant population growth factor, defaulting to 1. Each successive
        birth cohort is this factor bigger than the prior birth cohort. With flat
        demographic dynamics, this results in the entire population growing by
        this factor each period as well. NOT YET IMPLEMENTED
    prod_gro : float
        Constant aggregate productivity growth factor, defaulting to 1. Each
        successive birth cohort has permanent labor productivity that is this
        factor bigger than the prior cohort. With flat demographic dynamics,
        this results in aggregate productivity growing by this factor as well.
        NOT YET IMPLEMENTED
    solved : bool
        Whether the agent's model has already been solved. If False (default),
        it will be solved as the very first step. Solving the agent's long run
        model before constructing SSJ matrices has the advantage of not needing
        to re-solve the long run model for each shock variable.
    age_agg : bool
        Whether the returned SSJs should combine effects across ages (default True)
        or leave effects disaggregated by age (False). When False, each SSJ will
        be shape (T_age, T_max, T_max), and the overall SSJ matrix can be found by
        doing np.sum(SSJ, axis=0).
    construct : bool
        Whether the construct (update) method should be run after the shock is
        updated. The default is True, which is the "safe" option. If the shock
        variable is a parameter that enters the model only *directly*, rather
        than being used to build a more complex model input, then this can be
        set to False to save a (very) small amount of time during computation.
        If it is set to False improperly, the SSJs will be very wrong, potentially
        just zero everywhere.
    offset : bool
        Whether the shock variable is "offset in time" for the solver, with a
        default of False. This should be set to True if the named shock variable
        (or the constructed model input that it affects) is indexed by t+1 from
        the perspective of the solver. For example, the period t solver for the
        ConsIndShock model takes in risk free interest factor Rfree as an argument,
        but it represents the value of R that will occur at the start of t+1.
    verbose : bool
        Whether to display timing/progress to screen. The default is False.
    newborn_growth : float
        Per-period growth factor of the normalizing level that successive newborn
        cohorts inherit (a common trend), by which the weights under ``norm`` are
        divided. With flat demographics and no aggregate productivity growth,
        which is all this function implements (see ``prod_gro``), newborns arrive
        at a fixed level and the default of 1.0 is the consistent value; it is not
        taken from the agent's ``PermGroFacAgg``, because that is exactly the
        cohort trend this function does not yet model.
    ghost : bool
        Accepted for interface parity with make_basic_SSJ_matrices; must be
        False. A life-cycle solution is exact backward induction from a terminal
        period, with no fixed-point residual for a ghost run to remove.

    Returns
    -------
    SSJ : np.array or [np.array]
        One or more sequence space Jacobian arrays over the outcome variables
        with respect to the named shock variable. Each is shape (T_max, T_max).
    """
    if ghost:
        raise NotImplementedError(
            "ghost=True applies to the infinite-horizon builder only; a life-cycle "
            "solution has no fixed-point residual for a ghost run to remove."
        )
    if agent.cycles != 1:
        raise ValueError("This function is only compatible with life-cycle models!")
    if not isinstance(outcomes, list):
        outcomes = [outcomes]
        no_list = True
    else:
        no_list = False
    J = len(outcomes)

    # Check for attempts to use future functionality
    if pop_gro != 1.0:
        raise ValueError(
            "Population growth is not yet implemented for make_flat_LC_SSJ_matrices!"
        )
    if prod_gro != 1.0:
        raise ValueError(
            "Productivity growth is not yet implemented for make_flat_LC_SSJ_matrices!"
        )

    # Store the simulator if it exists
    simulator_backup = agent._simulator if hasattr(agent, "_simulator") else None

    # Make sure the shock variable is age-varying
    if shock in agent.time_inv:
        temp = getattr(agent, shock)
        original_shock_value = temp
        setattr(agent, shock, agent.T_cycle * [temp])
        agent.del_from_time_inv(shock)
        agent.add_to_time_vary(shock)
        shock_was_time_inv = True
    else:
        shock_was_time_inv = False

    # Solve the long run model if it wasn't already
    if not solved:
        t0 = time()
        agent.solve()
        t1 = time()
        if verbose:
            print(
                "Solving the long run model took {:.3f}".format(t1 - t0) + " seconds."
            )
    LR_soln = deepcopy(agent.solution)

    try:
        t0 = time()
        agent.initialize_sym()
        X = agent._simulator  # for easier referencing

        # Transition matrices for the long run model. Naming every period skips
        # the population-level replacement and its stationarity check, which a
        # cohort model does not use: it handles death age by age itself.
        X.make_transition_matrices(
            grids,
            norm,
            for_t=range(len(X.periods)),
            newborn_growth=newborn_growth,
        )
        LR_trans = deepcopy(X.trans_arrays)  # the transition matrices in LR model
        T_age = len(LR_trans)
        K = X.newborn_dstn.size
        # Mass that survives from each arrival state at each age: the survival
        # probability, or under norm the survivors' income-weighted mass, which
        # carries the growth of the normalizing level realized within the period
        LR_surv = [
            _lc_surviving_mass(X.periods[t].matrices, K, norm) for t in range(T_age)
        ]
        if T_max < T_age:
            raise ValueError(
                "T_max must be greater than or equal to T_age in order to pad "
                "fake_news_array without truncation."
            )
        LR_outcomes = []
        outcome_grids = []
        for var in outcomes:
            try:
                LR_outcomes.append([X.periods[t].matrices[var] for t in range(T_age)])
                outcome_grids.append([X.periods[t].grids[var] for t in range(T_age)])
            except KeyError as exc:
                raise KeyError(
                    "Outcome " + var + " was requested, but no grid was provided!"
                ) from exc

        # Extract the normalizing trend
        if trend is not None:
            trend_adj_fac = np.array(
                [X.periods[t].content[trend] for t in range(T_age)]
            )
            trend_adj_fac[0] = 1.0
            trend_adj_cum = np.cumprod(trend_adj_fac)
        else:
            trend_adj_cum = np.ones(T_age)

        t1 = time()
        if verbose:
            print(
                "Making the transition matrix for the long run model took {:.3f}".format(
                    t1 - t0
                )
                + " seconds."
            )

        # Find the steady state for the long run model
        t0 = time()
        X.simulate_cohort_by_grids(outcomes=["dead"])
        SS_outcomes = {}
        for j in range(J):
            name = outcomes[j]
            SS_outcomes[name] = [
                np.dot(LR_outcomes[j][t], outcome_grids[j][t]) for t in range(T_age)
            ]

        # The cohort's arrival distribution by age, carrying its mass. Under norm
        # this is the income-weighted mass, so older cohorts carry the growth of
        # the normalizing level accumulated since birth.
        SS_dstn = _lc_cohort_dstns(X.newborn_dstn, LR_trans, LR_surv)

        # Population size for the per capita normalization counts people: the
        # unweighted survival rates by age
        survival_by_age = 1.0 - X.history_avg["dead"]
        pop_sum = float(
            np.sum(np.cumprod(np.concatenate(([1.0], survival_by_age[:-1]))))
        )

        t1 = time()
        if verbose:
            print(
                "Finding the long run steady state took {:.3f}".format(t1 - t0)
                + " seconds."
            )

        # Construct the "expectation vectors" for all outcomes at all ages
        t0 = time()
        E_vecs = {}
        for j in range(J):
            name = outcomes[j]
            E_temp = [[SS_outcomes[name][a].copy()] for a in range(T_age)]
            for t in range(1, T_age):
                for a in range(T_age - t):
                    E_temp[a].append(
                        LR_surv[a] * np.dot(LR_trans[a], E_temp[a + 1][-1])
                    )
            E_vecs[name] = E_temp
        t1 = time()

        # Rearrange the expectation vectors for better access later
        E_curly = [
            np.stack(
                [
                    np.stack([E_vecs[name][a][t] for name in outcomes])
                    for t in range(T_age - a)
                ]
            )
            for a in range(T_age)
        ]

        if verbose:
            print(
                "Constructing expectation vectors took {:.3f}".format(t1 - t0)
                + " seconds."
            )

        # Each entry of the E_vecs dictionary is a nested list. The outer index of the
        # list is a, the age at t=0, and the inner index is time period t. The elements
        # in the nested list are expectation vectors: the expected value of the outcome
        # in period t conditional on being age a and at state space gridpoint n at t=0.

        # Initialize the fake news matrices for each output
        fake_news_array = np.zeros((J, T_age, T_age, T_age))
        # Dimensions of fake news arrays:
        # dim 0 --> j: index of outcome variable
        # dim 1 --> a: age in period t
        # dim 2 --> t: periods since news arrived
        # dim 3 --> s: periods ahead about which the news arrived

        # Loop over ages of the model and have the news shock apply at each one;
        # k is the age index at which the shock arrives
        t0 = time()
        for k in reversed(range(T_age)):
            # Adjust the timing for "offset" shocks
            l = k - int(offset)
            shock_val_orig = getattr(agent, shock)[l]
            shock_val_new = shock_val_orig + eps

            # Perturb the shock variable at age k, which corresponds to "solver period" l
            if l >= 0:
                getattr(agent, shock)[l] = shock_val_new

                # Solve the model backwards from age l
                if construct:
                    agent.construct()
                agent.solve(from_solution=LR_soln[l + 1], from_t=l + 1)
            else:
                agent.solution = LR_soln

            # Build transitions and outcomes up to age k. Don't use "fake news timing" option!
            agent.initialize_sym()
            X = agent._simulator  # for easier typing
            if l < 0:
                setattr(X.periods[0], shock, shock_val_new)
            X.make_transition_matrices(
                grids, norm, for_t=range(k + 1), newborn_growth=newborn_growth
            )
            shocked_trans = deepcopy(X.trans_arrays)
            # Under norm a shock to the income process moves the surviving mass
            # itself (a level response), so the news term uses the shocked masses
            shocked_surv = [
                _lc_surviving_mass(X.periods[a].matrices, K, norm) for a in range(k + 1)
            ]
            shocked_outcomes = []
            for var in outcomes:
                temp_outcomes = []
                for a in range(k + 1):
                    temp_outcomes.append(X.periods[a].matrices[var])
                shocked_outcomes.append(temp_outcomes)

            # Update the t=0 row of the fake news matrices
            for j in range(J):
                for a in range(k + 1):
                    temp = np.dot(
                        SS_dstn[a], shocked_outcomes[j][a] - LR_outcomes[j][a]
                    )
                    fake_news_array[j, a, 0, k - a] += np.dot(temp, outcome_grids[j][a])

            # Update the other t rows of the fake news matrices
            for a in range(k + 1):
                if a >= T_age - 1:
                    continue
                D_dstn_news = (
                    np.dot(shocked_surv[a] * SS_dstn[a], shocked_trans[a])
                    - SS_dstn[a + 1]
                )
                update_FN_mats(
                    fake_news_array, E_curly[a + 1], D_dstn_news, T_age, a, k
                )

            # Reset the shock variable at age l
            if l >= 0:
                getattr(agent, shock)[l] = shock_val_orig

        t1 = time()
        if verbose:
            print(
                "Making fake news arrays for each period of the problem took {:.3f}".format(
                    t1 - t0
                )
                + " seconds."
            )

        t0 = time()
        # Pad out the fake news array with zeros
        FN_pad = np.zeros((J, T_age, T_max, T_max))
        FN_pad[:, :, :T_age, :T_age] = fake_news_array

        # Construct age-specific Jacobian matrices
        SSJ_by_age = FN_pad.copy()
        for t in range(1, T_max):
            SSJ_by_age[:, :, 1:, t] += SSJ_by_age[:, :, :-1, t - 1]

        # Apply normalization factors
        SSJ_by_age *= np.reshape(trend_adj_cum, (1, T_age, 1, 1))
        SSJ_by_age /= pop_sum * eps
        if norm is not None:
            # Outcome arrays carry the within-period growth of the normalizing level,
            # so express responses per unit of the newborn cohort's first-period
            # level, as trend (factor starting at one) and make_basic_SSJ_matrices do.
            SSJ_by_age /= float(np.sum(np.dot(SS_dstn[0], LR_outcomes[0][0])))

        t1 = time()
        if verbose:
            print(
                "Constructing the sequence space Jacobians took {:.3f}".format(t1 - t0)
                + " seconds."
            )

        # Structure and return outputs, aggregating by age if requested
        SSJ = [SSJ_by_age[j, :, :, :] for j in range(J)]
        if age_agg:
            for j in range(J):
                SSJ[j] = np.sum(SSJ[j], axis=0)
        if no_list:
            return SSJ[0]
        else:
            return SSJ

    finally:
        # Make sure the agent wasn't unexpectedly mutated in this method
        _restore_agent(agent, LR_soln, simulator_backup)
        if shock_was_time_inv:
            setattr(agent, shock, original_shock_value)
            agent.del_from_time_vary(shock)
            agent.add_to_time_inv(shock)


def calc_shock_response_manually(
    agent,
    shock,
    outcomes,
    grids,
    s=0,
    eps=1e-4,
    T_max=300,
    norm=None,
    solved=False,
    construct=[],
    offset=False,
    verbose=False,
    newborn_growth=None,
    ghost=False,
):
    """
    Compute an AgentType instance's timepath of outcome responses to learning at
    t=0 that the named shock variable will be perturbed at t=s. This is equivalent
    to calculating only the s-th column of the SSJs *manually*, rather than using
    the fake news algorithm. This function can be used to verify and/or debug the
    output of the fake news SSJ algorithm.

    Important: Mortality (or death and replacement generally) should be turned
    off in the model (via parameter values) for this to work properly. Or does it?

    Parameters
    ----------
    agent : AgentType
        Agent for which the response(s) should be calculated. Must have T_cycle=1
        and cycles=0, or the function will throw an error. Must have a model
        file defined or this won't work at all.
    shock : str
        Name of the variable that the response will be computed with respect to.
        It does not need to be a "shock" in a modeling sense, but it must be a
        single-valued parameter (possibly a singleton list) that can be changed.
    outcomes : str or [str]
        Names of outcome variables of interest; an SSJ matrix will be constructed
        for each variable named here. If a single string is passed, the output
        will be a single np.array. If a list of strings are passed, the output
        will be a list of dYdX vectors in the order specified here.
    grids : dict
        Dictionary of dictionaries with discretizing grid information. The grids
        should include all arrival variables other than those that are normalized
        out. They should also include all variables named in outcomes, except
        outcomes that are continuation variables that remap to arrival variables.
        Grid specification must include number of nodes N, should also include
        min and max if the variable is continuous.
    s : int
        Period in which the shock variable is perturbed, relative to current t=0.
        The default is 0.
    eps : float
        Amount by which to perturb the shock variable. The default is 1e-4.
    T_max : int
        The length of the simulation for this exercise. The default is 300.
    norm : str or None
        Name of the model variable whose realized value weights the probability
        masses, for Harmenberg (income-weighted) aggregation, if any. This lets
        the grid over permanent income be omitted as an explicit state. For HARK's
        permanent-income models name the growth factor of the *level*, 'G'
        (PermGroFac * PermShk); the shock alone, 'PermShk', is exact only without
        deterministic growth or without mortality, or when newborns inherit the
        growth (see newborn_growth) -- otherwise the stationary distribution
        overweights the young and normalized aggregates are biased.
    solved : bool
        Whether the agent's model has already been solved. If False (default),
        it will be solved as the very first step.
    construct : [str]
        List of constructed objects that will be changed by perturbing shock.
        These should all share an "offset status" (True or False). Default is [].
    offset : bool
        Whether the shock variable is "offset in time" for the solver, with a
        default of False. This should be set to True if the named shock variable
        (or the constructed model input that it affects) is indexed by t+1 from
        the perspective of the solver. For example, the period t solver for the
        ConsIndShock model takes in risk free interest factor Rfree as an argument,
        but it represents the value of R that will occur at the start of t+1.
    verbose : bool
        Whether to display timing/progress to screen. The default is False.

    ghost : bool
        Whether to difference the perturbed path against an unperturbed finite-
        horizon path of the same length (a "ghost run") rather than against the
        long run averages; see make_basic_SSJ_matrices. The default is False.

    Returns
    -------
    dYdX : np.array or [np.array]
        One or more vectors of length T_max.
    """
    setup = _prepare_ssj_computation(
        agent, outcomes, grids, norm, solved, verbose, newborn_growth
    )
    newborn_growth = setup["newborn_growth"]
    outcomes = setup["outcomes"]
    no_list = setup["no_list"]
    simulator_backup = setup["simulator_backup"]
    LR_soln = setup["LR_soln"]
    X = setup["X"]
    LR_outcomes = setup["LR_outcomes"]
    outcome_grids = setup["outcome_grids"]
    SS_dstn = setup["SS_dstn"]

    SS_outcomes = [np.dot(mat.T, SS_dstn) for mat in LR_outcomes]
    SS_avgs = [
        np.dot(ss, grid) / (np.sum(ss) if norm is not None else 1.0)
        for ss, grid in zip(SS_outcomes, outcome_grids)
    ]

    try:
        # Make a temporary agent to construct the perturbed constructed objects
        t0 = time()
        temp_agent = deepcopy(agent)
        base_shock_value, shock_is_list = _perturb_shock(agent, shock)
        if shock_is_list:
            temp_value = [base_shock_value + eps]
        else:
            temp_value = base_shock_value + eps
        temp_dict = {shock: temp_value}
        temp_agent.assign_parameters(**temp_dict)
        if len(construct) > 0:
            temp_agent.update()
        for var in construct:
            temp_dict[var] = getattr(temp_agent, var)

        # Build the finite horizon version of this agent
        FH_agent = deepcopy(agent)
        FH_agent.del_param("solution")
        FH_agent.del_param("_simulator")
        FH_agent.del_from_time_vary("solution")
        FH_agent.del_from_time_inv(shock)
        FH_agent.add_to_time_vary(shock)
        FH_agent.del_from_time_inv(*construct)
        FH_agent.add_to_time_vary(*construct)
        finite_dict = {"T_cycle": T_max, "cycles": 1}
        for var in FH_agent.time_vary:
            if var in construct:
                sequence = [deepcopy(getattr(agent, var)[0]) for t in range(T_max)]
                sequence[s] = deepcopy(getattr(temp_agent, var)[0])
            else:
                sequence = T_max * [deepcopy(getattr(agent, var)[0])]
            finite_dict[var] = sequence
        shock_seq = T_max * [base_shock_value]
        shock_seq[s] = base_shock_value + eps
        finite_dict[shock] = shock_seq
        FH_agent.assign_parameters(**finite_dict)
        del temp_agent
        t1 = time()
        if verbose:
            print(
                "Building the finite horizon agent took {:.3f}".format(t1 - t0)
                + " seconds."
            )

        # Solve the finite horizon agent
        t0 = time()
        FH_agent.solve(from_solution=LR_soln)
        t1 = time()
        if verbose:
            print(
                "Solving the "
                + str(T_max)
                + " period problem took {:.3f}".format(t1 - t0)
                + " seconds."
            )

        # Build transition matrices for the finite horizon problem
        t0 = time()
        FH_agent.initialize_sym()
        FH_agent._simulator.make_transition_matrices(
            grids, norm=norm, fake_news_timing=True, newborn_growth=newborn_growth
        )
        t1 = time()
        if verbose:
            print(
                "Constructing transition matrices took {:.3f}".format(t1 - t0)
                + " seconds."
            )

        # Use grid simulation to find the timepath of requested variables, and compute
        # the derivative with respect to baseline outcomes
        t0 = time()
        FH_agent._simulator.simulate_cohort_by_grids(outcomes, from_dstn=SS_dstn)
        baselines = SS_avgs
        if ghost:
            # The ghost run: the same finite horizon agent with no perturbation,
            # solved from the same long run solution; its path is the baseline.
            GH_agent = deepcopy(agent)
            GH_agent.del_param("solution")
            GH_agent.del_param("_simulator")
            GH_agent.del_from_time_vary("solution")
            GH_agent.del_from_time_inv(shock)
            GH_agent.add_to_time_vary(shock)
            GH_agent.del_from_time_inv(*construct)
            GH_agent.add_to_time_vary(*construct)
            ghost_dict = {"T_cycle": T_max, "cycles": 1}
            for var in GH_agent.time_vary:
                ghost_dict[var] = T_max * [deepcopy(getattr(agent, var)[0])]
            ghost_dict[shock] = T_max * [base_shock_value]
            GH_agent.assign_parameters(**ghost_dict)
            GH_agent.solve(from_solution=LR_soln)
            GH_agent.initialize_sym()
            GH_agent._simulator.make_transition_matrices(
                grids, norm=norm, fake_news_timing=True, newborn_growth=newborn_growth
            )
            GH_agent._simulator.simulate_cohort_by_grids(outcomes, from_dstn=SS_dstn)
            baselines = [GH_agent._simulator.history_avg[var] for var in outcomes]
            del GH_agent
        dYdX = []
        for j, var in enumerate(outcomes):
            diff_path = (FH_agent._simulator.history_avg[var] - baselines[j]) / eps
            if offset:
                dYdX.append(diff_path[1:])
            else:
                dYdX.append(diff_path[:-1])
        t1 = time()
        if verbose:
            print(
                "Calculating impulse responses by grid simulation took {:.3f}".format(
                    t1 - t0
                )
                + " seconds."
            )

        del FH_agent
        if no_list:
            return dYdX[0]
        else:
            return dYdX
    finally:
        _restore_agent(agent, LR_soln, simulator_backup)


@njit
def calc_derivs_of_state_dstns(T, J, trans_by_t, trans_LR, SS_dstn):  # pragma: no cover
    """
    Numba-compatible helper function to calculate the derivative of the state
    distribution by period.

    Parameters
    ----------
    T : int
        Maximum time horizon for the fake news algorithm.
    J : int
        Number of outcomes of interest.
    trans_by_t : np.array
        Array of shape (T,K,K) representing the transition matrix in each period.
    trans_LR : np.array
        Array of shape (K,K) representing the long run transition matrix.
    SS_dstn : np.array
        Array of size K representing the long run steady state distribution.

    Returns
    -------
    D_dstn_news : np.array
        Array of shape (T,K) representing dD_1^s from the SSJ paper, where K
        is the number of arrival state space nodes.

    """
    K = SS_dstn.size
    D_dstn_news = np.empty((T, K))  # this is dD_1^s in the SSJ paper (equation 24)
    for t in range(T - 1, -1, -1):
        D_dstn_news[T - t - 1, :] = np.dot((trans_by_t[t, :, :] - trans_LR).T, SS_dstn)
    return D_dstn_news


@njit
def calc_derivs_of_policy_funcs(T, Y_by_t, Y_LR, Y_grid, SS_dstn):  # pragma: no cover
    """
    Numba-compatible helper function to calculate the derivative of an outcome
    function in each period.

    Parameters
    ----------
    T : int
        Maximum time horizon for the fake news algorithm.
    Y_by_t : np.array
        Array of shape (T,K,N) with the stochastic outcome, mapping from K arrival
        state space nodes to N outcome space nodes, for each of the T periods.
    Y_LR : np.array
        Array of shape (K,N) representing the stochastic outcome in the long run.
    Y_grid : np.array
        Array of size N representing outcome space gridpoints.
    SS_dstn : np.array
        Array of size K representing the long run steady state distribution.

    Returns
    -------
    dY_news : np.array
        Array of size T representing the change in average outcome in each period
        when the shock arrives unexpectedly in that period.
    """
    dY_news = np.empty(T)  # this is dY_0^s in the SSJ paper (equation 24)
    for t in range(T - 1, -1, -1):
        temp = (Y_by_t[t, :, :] - Y_LR).T
        dY_news[T - t - 1] = np.dot(np.dot(temp, SS_dstn), Y_grid)
    return dY_news


@njit
def calc_derivs_of_state_dstns_vs_ghost(
    T, trans_by_t, base_D, SS_dstn
):  # pragma: no cover
    """
    Numba-compatible helper: the derivative of the state distribution by period,
    differenced against the ghost run's push of the steady state distribution
    through the unperturbed transition of the same period.

    Parameters
    ----------
    T : int
        Maximum time horizon for the fake news algorithm.
    trans_by_t : np.array
        Array of shape (T,K,K) representing the perturbed transition matrix in each period.
    base_D : np.array
        Array of shape (T,K): the ghost run's push trans_ghost[t]^T SS_dstn in each period.
    SS_dstn : np.array
        Array of size K representing the long run steady state distribution.

    Returns
    -------
    D_dstn_news : np.array
        Array of shape (T,K) representing dD_1^s from the SSJ paper.
    """
    K = SS_dstn.size
    D_dstn_news = np.empty((T, K))
    for t in range(T - 1, -1, -1):
        D_dstn_news[T - t - 1, :] = (
            np.dot(trans_by_t[t, :, :].T, SS_dstn) - base_D[t, :]
        )
    return D_dstn_news


@njit
def calc_derivs_of_policy_funcs_vs_ghost(
    T, Y_by_t, base_Y, Y_grid, SS_dstn
):  # pragma: no cover
    """
    Numba-compatible helper: the change in the average outcome in each period,
    differenced against the ghost run's average outcome in the same period.

    Parameters
    ----------
    T : int
        Maximum time horizon for the fake news algorithm.
    Y_by_t : np.array
        Array of shape (T,K,N) with the perturbed stochastic outcome in each period.
    base_Y : np.array
        Array of size T: the ghost run's average outcome in each period.
    Y_grid : np.array
        Array of size N representing outcome space gridpoints.
    SS_dstn : np.array
        Array of size K representing the long run steady state distribution.

    Returns
    -------
    dY_news : np.array
        Array of size T representing dY_0^s from the SSJ paper.
    """
    dY_news = np.empty(T)
    for t in range(T - 1, -1, -1):
        dY_news[T - t - 1] = (
            np.dot(np.dot(Y_by_t[t, :, :].T, SS_dstn), Y_grid) - base_Y[t]
        )
    return dY_news


@njit
def make_fake_news_matrices(T, J, dY, D_dstn, trans_LR, E):  # pragma: no cover
    """
    Numba-compatible function to calculate the fake news array from first order
    perturbation information.

    Parameters
    ----------
    T : int
        Maximum time horizon for the fake news algorithm.
    J : int
        Number of outcomes of interest.
    dY : int
        Array shape (T,J) representing dY_0 from the SSJ paper.
    D_dstn : np.array
        Array of shape (T,K) representing dD_1^s from the SSJ paper, where K
        is the number of arrival state space nodes.
    trans_LR : np.array
        Array of shape (K,K) representing the transpose of the long run transition matrix.
    E : np.array
        Initial expectation vectors combined into a single array of shape (J,K).

    Returns
    -------
    FN : np.array
        Fake news array of shape (J,T,T).
    """
    FN = np.empty((J, T, T))
    FN[:, 0, :] = dY.T  # Fill in row zero
    for t in range(1, T):  # Loop over other rows
        for s in range(T):
            FN[:, t, s] = np.dot(E, D_dstn[s, :])
        E = np.dot(E, trans_LR)
    return FN


@njit
def update_FN_mats(FN_mats, evecs, dD1, A, a, k):  # pragma: no cover
    """
    This is adapted from Mateo's code.

    FN_mats: (J, A, A, A)
    evecs : (T, J, G)
    dD1   : (G,)
    """
    J = FN_mats.shape[0]
    G = dD1.shape[0]

    for j in range(J):
        for t in range(1, A - a):
            # compute dot(evecs[t-1, j, :], dD1) by hand
            v = 0.0
            for g in range(G):
                v += evecs[t - 1, j, g] * dD1[g]
            FN_mats[j, a + t, t, k - a] += v


def _scalar_parameter(value, name):
    """
    Return a parameter of a one-period infinite-horizon model as a float,
    accepting a float or a singleton list or array (the form HARK's time-varying
    parameters take when T_cycle is 1).
    """
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size != 1:
        raise ValueError(
            name
            + " must be a single number (or a singleton list) for this check, which "
            "is for one-period infinite-horizon models."
        )
    return float(arr[0])


def flow_budget_residuals(SSJ_C, SSJ_A, Rfree, LivPrb, SSJ_Y=None, newborn_growth=1.0):
    """
    Residuals of the household flow-budget identity along each column of a set
    of sequence space Jacobians from make_basic_SSJ.

    In the units make_basic_SSJ reports (responses per unit of the period's
    normalizing level), the responses of consumption C, end-of-period assets A
    and labor income Y to a perturbation dated s satisfy, at every date t,

        dC[t, s] + dA[t, s] - rho * dA[t-1, s] - dY[t, s] = cash[t, s],

    with dA[-1, s] = 0 and rho = Rfree * LivPrb / newborn_growth: survivors bring
    Rfree times their end-of-period assets into the next period, the dead bring
    nothing (their replacements' initial assets belong to the steady state and
    do not respond), and the normalizing level grows by newborn_growth per
    period. The left side is the period's uses of resources less what arrives
    from last period and from labor income, so cash[t, s] is what the
    perturbation itself injects at date t: nothing, for a perturbation that
    reaches the household through labor income (with dY included) or through
    preferences; for a perturbation of the return factor, the return on the
    assets brought into the period, LivPrb * A_ss / newborn_growth, on the date
    the perturbed return is paid (row s for make_basic_SSJ with offset=True) and
    nothing elsewhere. Without SSJ_Y the residual is the cash delivered including
    labor income; for a perturbation of income that is its delivery on its own
    date, the same amount in every column.

    Parameters
    ----------
    SSJ_C : np.array
        Sequence space Jacobian of consumption (cNrm), shape (T, T).
    SSJ_A : np.array
        Sequence space Jacobian of end-of-period assets (aNrm), shape (T, T).
    Rfree : float or [float]
        Risk free return factor of the long run model.
    LivPrb : float or [float]
        Survival probability of the long run model.
    SSJ_Y : np.array or None
        Sequence space Jacobian of labor income (yNrm), shape (T, T), when it
        was requested as an outcome. The default is None.
    newborn_growth : float
        Per-period growth factor of the normalizing level that newborns inherit,
        as passed to make_basic_SSJ (whose default of None means the agent's
        PermGroFacAgg, 1.0 for most agents). The default is 1.0.

    Returns
    -------
    residuals : np.array
        The residual dC + dA - rho * dA(-1) - dY (dY omitted when SSJ_Y is None),
        shape (T, T); entry [t, s] is date t of the perturbation dated s.
    """
    C = np.asarray(SSJ_C, dtype=float)
    A = np.asarray(SSJ_A, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1] or A.shape != C.shape:
        raise ValueError("SSJ_C and SSJ_A must be square arrays of the same shape.")
    rho = (
        _scalar_parameter(Rfree, "Rfree")
        * _scalar_parameter(LivPrb, "LivPrb")
        / float(newborn_growth)
    )
    residuals = C + A
    residuals[1:, :] -= rho * A[:-1, :]
    if SSJ_Y is not None:
        Y = np.asarray(SSJ_Y, dtype=float)
        if Y.shape != C.shape:
            raise ValueError("SSJ_Y must have the same shape as SSJ_C and SSJ_A.")
        residuals -= Y
    return residuals


def check_flow_budget(
    SSJ_C,
    SSJ_A,
    Rfree,
    LivPrb,
    SSJ_Y=None,
    newborn_growth=1.0,
    A_ss=None,
    tol=1e-6,
):
    """
    Check sequence space Jacobians from make_basic_SSJ against the household
    flow-budget identity (see flow_budget_residuals), raising a ValueError that
    names the offending date and perturbation date if it fails.

    With SSJ_Y given, the residual must vanish at every date in every column.
    If A_ss is given, the perturbation is one of the return factor and the
    residual must instead equal LivPrb * A_ss / newborn_growth on the diagonal
    (the return on the assets brought into the period) and vanish elsewhere.
    Without SSJ_Y or A_ss the residual is the cash the perturbation delivers,
    and the check is the weaker one that needs no income Jacobian: every column
    delivers only on its own date, and the same amount in every column (the
    same experiment moved in time), the amount being taken from the columns
    themselves.

    Parameters
    ----------
    SSJ_C, SSJ_A, Rfree, LivPrb, SSJ_Y, newborn_growth
        As in flow_budget_residuals.
    A_ss : float or None
        Steady state end-of-period assets per unit of the period's normalizing
        level (the long run average of aNrm, as get_long_run_average reports
        it), for a perturbation of the return factor. The default is None.
        LivPrb * A_ss / newborn_growth equals the assets brought into the period
        only up to the discretization of the assets-to-capital transition (of
        the order of 1e-6 with a few hundred grid nodes), so give this case a
        tolerance to match; the check without SSJ_Y and A_ss is exact.
    tol : float
        Largest violation allowed, relative to the largest entry of the
        Jacobians and of the cash delivered. The default is 1e-6.

    Returns
    -------
    worst : float
        The largest violation found, relative to that scale.
    """
    residuals = flow_budget_residuals(
        SSJ_C, SSJ_A, Rfree, LivPrb, SSJ_Y, newborn_growth
    )
    T = residuals.shape[0]
    if A_ss is not None:
        cash = _scalar_parameter(LivPrb, "LivPrb") * float(A_ss) / float(newborn_growth)
    elif SSJ_Y is None:
        cash = float(np.median(np.diag(residuals)))
    else:
        cash = 0.0
    expected = np.zeros_like(residuals)
    expected[np.arange(T), np.arange(T)] = cash
    scale = max(np.max(np.abs(SSJ_C)), np.max(np.abs(SSJ_A)), abs(cash))
    if SSJ_Y is not None:
        scale = max(scale, np.max(np.abs(SSJ_Y)))
    if scale == 0.0:
        scale = 1.0
    violation = np.abs(residuals - expected) / scale
    worst = float(np.max(violation))
    if worst > tol:
        t, s = np.unravel_index(np.argmax(violation), violation.shape)
        raise ValueError(
            "The flow-budget identity fails: at date t={} of the perturbation dated "
            "s={} the residual is {:.4e} where {:.4e} was expected ({:.2e} of the "
            "Jacobians' scale; tolerance {:.1e}).".format(
                t, s, residuals[t, s], expected[t, s], worst, tol
            )
        )
    return worst


def aggregate_SSJs(SSJs, weights):
    """
    Aggregate sequence space Jacobians over types as a weighted sum.

    Parameters
    ----------
    SSJs : list
        One entry per type: the output of make_basic_SSJ for the same shock,
        outcomes and T_max on each type, so an np.array, a list of them or a
        dict of them, in the same structure for every type.
    weights : array-like
        One weight per type, its share of the aggregate. In the units of
        make_basic_SSJ (responses per unit of each type's normalizing level) the
        response per unit of the population's level uses each type's share of
        that level: its population share when the types' newborns start at the
        same level and share the same survival and growth (they then have the
        same mean level whatever their preferences), and its population share
        times its mean level relative to the population's otherwise. The weights
        are used as given, not normalized.

    Returns
    -------
    SSJ : np.array, [np.array] or {str: np.array}
        The weighted sum, in the structure of one type's entry.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    if len(SSJs) == 0 or weights.size != len(SSJs):
        raise ValueError("Pass one weight per type, and at least one type.")
    return _weighted_sum(list(SSJs), weights)


def _weighted_sum(items, weights):
    first = items[0]
    if isinstance(first, dict):
        keys = list(first.keys())
        for item in items[1:]:
            if not isinstance(item, dict) or set(item.keys()) != set(keys):
                raise ValueError("Every type must have the same outcome names.")
        return {
            key: _weighted_sum([item[key] for item in items], weights) for key in keys
        }
    if isinstance(first, (list, tuple)):
        n = len(first)
        for item in items[1:]:
            if not isinstance(item, (list, tuple)) or len(item) != n:
                raise ValueError("Every type must have the same number of outcomes.")
        return [_weighted_sum([item[j] for item in items], weights) for j in range(n)]
    arrays = [np.asarray(item, dtype=float) for item in items]
    shape = arrays[0].shape
    for arr in arrays[1:]:
        if arr.shape != shape:
            raise ValueError("Every type's Jacobian must have the same shape.")
    total = np.zeros(shape)
    for w, arr in zip(weights, arrays):
        total += w * arr
    return total
