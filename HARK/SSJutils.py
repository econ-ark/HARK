"""
Functions for building heterogeneous agent sequence space Jacobian matrices from
HARK AgentType instances. The top-level functions are accessible as methods on
AgentType itself.
"""

from time import time
from copy import deepcopy
import numpy as np
from numba import njit


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
        Name of the model variable to normalize by for Harmenberg aggregation,
        if any. For many HARK models, this should be 'PermShk', which enables
        the grid over permanent income to be omitted as an explicit state.
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

    Returns
    -------
    SSJ : np.array or [np.array]
        One or more sequence space Jacobian arrays over the outcome variables
        with respect to the named shock variable.
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

    # Store the simulator if it exists
    if hasattr(agent, "_simulator"):
        simulator_backup = agent._simulator

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
    X.make_transition_matrices(grids, norm)
    LR_trans = X.trans_arrays[0].copy()  # the transition matrix in LR model
    LR_period = X.periods[0]
    LR_outcomes = []
    outcome_grids = []
    for var in outcomes:
        try:
            LR_outcomes.append(X.periods[0].matrices[var])
            outcome_grids.append(X.periods[0].grids[var])
        except:
            raise ValueError(
                "Outcome " + var + " was requested, but no grid was provided!"
            )
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
    SS_outcomes = []
    for j in range(len(outcomes)):
        SS_outcomes.append(np.dot(LR_outcomes[j].transpose(), SS_dstn))
    t1 = time()
    if verbose:
        print(
            "Finding the long run steady state took {:.3f}".format(t1 - t0)
            + " seconds."
        )

    # Solve back one period while perturbing the shock variable
    t0 = time()
    try:
        base = getattr(agent, shock)
    except:
        raise ValueError(
            "The agent doesn't have anything called " + shock + " to perturb!"
        )
    if isinstance(base, list):
        base_shock_value = base[0]
        shock_is_list = True
    else:
        base_shock_value = base
        shock_is_list = False
    if not isinstance(base_shock_value, float):
        raise TypeError(
            "Only a single real-valued object can be perturbed in this way!"
        )
    agent.cycles = 1
    if shock_is_list:
        temp_value = [base_shock_value + eps]
    else:
        temp_value = base_shock_value + eps
    temp_dict = {shock: temp_value}
    agent.assign_parameters(**temp_dict)
    if construct:
        agent.update()
    agent.solve(from_solution=LR_soln)
    agent.initialize_sym()
    Tm1_soln = deepcopy(agent.solution[0])
    period_Tm1 = agent._simulator.periods[0]
    period_T = agent._simulator.periods[-1]
    t1 = time()
    if verbose:
        print(
            "Solving period T-1 with a perturbed variable took {:.3f}".format(t1 - t0)
            + " seconds."
        )

    # Set up and solve the agent for T_max-1 more periods
    t0 = time()
    agent.cycles = T_max - 1
    if shock_is_list:
        orig_dict = {shock: [base_shock_value]}
    else:
        orig_dict = {shock: base_shock_value}
    agent.assign_parameters(**orig_dict)
    if construct:
        agent.update()
    agent.solve(from_solution=Tm1_soln)
    t1 = time()
    if verbose:
        print(
            "Solving the finite horizon model for "
            + str(T_max - 1)
            + " more periods took {:.3f}".format(t1 - t0)
            + " seconds."
        )

    # Construct transition and outcome matrices for the "finite horizon"
    t0 = time()
    agent.initialize_sym()
    X = agent._simulator  # for easier typing
    X.periods[-1] = period_Tm1  # substitute period T-1 from above
    if offset:
        for name in X.periods[-1].content.keys():
            if name not in X.solution:  # sub in proper T-1 non-solution info
                X.periods[-1].content[name] = LR_period.content[name]
        X.periods[-1].distribute_content()
        X.periods = X.periods[1:] + [period_T]
    X.make_transition_matrices(grids, norm, fake_news_timing=True)
    TmX_trans = deepcopy(X.trans_arrays)
    TmX_outcomes = []
    for t in range(T_max):
        Tmt_outcomes = []
        for var in outcomes:
            Tmt_outcomes.append(X.periods[t].matrices[var])
        TmX_outcomes.append(Tmt_outcomes)
    t1 = time()
    if verbose:
        print(
            "Constructing transition arrays for the finite horizon model took {:.3f}".format(
                t1 - t0
            )
            + " seconds."
        )

    # Calculate derivatives of transition and outcome matrices by first differences
    t0 = time()
    J = len(outcomes)
    K = SS_dstn.size
    D_dstn_array = calc_derivs_of_state_dstns(
        T_max, J, np.array(TmX_trans), LR_trans, SS_dstn
    )
    dY_news_array = np.empty((T_max, J))
    for j in range(J):
        temp_outcomes = np.array([TmX_outcomes[t][j] for t in range(T_max)])
        dY_news_array[:, j] = calc_derivs_of_policy_funcs(
            T_max, temp_outcomes, LR_outcomes[j], outcome_grids[j], SS_dstn
        )
    t1 = time()
    if verbose:
        print(
            "Calculating derivatives by first differences took {:.3f}".format(t1 - t0)
            + " seconds."
        )

    # Construct the "fake news" matrices, one for each outcome variable
    t0 = time()
    expectation_vectors = np.empty((J, K))  # Initialize expectation vectors
    for j in range(J):
        expectation_vectors[j, :] = np.dot(LR_outcomes[j], outcome_grids[j])
    FN = make_fake_news_matrices(
        T_max, J, dY_news_array, D_dstn_array, LR_trans.T, expectation_vectors.copy()
    )
    t1 = time()
    if verbose:
        print(
            "Constructing the fake news matrices took {:.3f}".format(t1 - t0)
            + " seconds."
        )

    # Construct the SSJ matrices, one for each outcome variable
    t0 = time()
    SSJ_array = calc_ssj_from_fake_news_matrices(T_max, J, FN, eps)
    SSJ = [SSJ_array[j, :, :] for j in range(J)]  # unpack into a list of arrays
    t1 = time()
    if verbose:
        print(
            "Constructing the sequence space Jacobians took {:.3f}".format(t1 - t0)
            + " seconds."
        )

    # Reset the agent to its original state and return the output
    agent.solution = [LR_soln]
    agent.cycles = 0
    agent._simulator.reset()
    try:
        agent._simulator = simulator_backup
    except:
        del agent._simulator
    if no_list:
        return SSJ[0]
    else:
        return SSJ


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
        Name of the model variable to normalize by for Harmenberg aggregation,
        if any. For many HARK models, this should be 'PermShk', which enables
        the grid over permanent income to be omitted as an explicit state.
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

    Returns
    -------
    dYdX : np.array or [np.array]
        One or more vectors of length T_max.
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

    # Store the simulator if it exists
    if hasattr(agent, "_simulator"):
        simulator_backup = agent._simulator

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
    X.make_transition_matrices(grids, norm)
    LR_outcomes = []
    outcome_grids = []
    for var in outcomes:
        try:
            LR_outcomes.append(X.periods[0].matrices[var])
            outcome_grids.append(X.periods[0].grids[var])
        except:
            raise ValueError(
                "Outcome " + var + " was requested, but no grid was provided!"
            )
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
    SS_outcomes = []
    SS_avgs = []
    for j in range(len(outcomes)):
        SS_outcomes.append(np.dot(LR_outcomes[j].transpose(), SS_dstn))
        SS_avgs.append(np.dot(SS_outcomes[j], outcome_grids[j]))
    t1 = time()
    if verbose:
        print(
            "Finding the long run steady state took {:.3f}".format(t1 - t0)
            + " seconds."
        )

    # Make a temporary agent to construct the perturbed constructed objects
    t0 = time()
    temp_agent = deepcopy(agent)
    try:
        base = getattr(agent, shock)
    except:
        raise ValueError(
            "The agent doesn't have anything called " + shock + " to perturb!"
        )
    if isinstance(base, list):
        base_shock_value = base[0]
        shock_is_list = True
    else:
        base_shock_value = base
        shock_is_list = False
    if not isinstance(base_shock_value, float):
        raise TypeError(
            "Only a single real-valued object can be perturbed in this way!"
        )
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
        grids, norm=norm, fake_news_timing=True
    )
    t1 = time()
    if verbose:
        print(
            "Constructing transition matrices took {:.3f}".format(t1 - t0) + " seconds."
        )

    # Use grid simulation to find the timepath of requested variables, and compute
    # the derivative with respect to baseline outcomes
    t0 = time()
    FH_agent._simulator.simulate_cohort_by_grids(outcomes, from_dstn=SS_dstn)
    dYdX = []
    for j, var in enumerate(outcomes):
        diff_path = (FH_agent._simulator.history_avg[var] - SS_avgs[j]) / eps
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

    # Reset the agent to its original state and return the output
    del FH_agent
    agent.solution = [LR_soln]
    agent.cycles = 0
    agent._simulator.reset()
    try:
        agent._simulator = simulator_backup
    except:
        del agent._simulator
    if no_list:
        return dYdX[0]
    else:
        return dYdX


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
def calc_ssj_from_fake_news_matrices(T, J, FN, dx):  # pragma: no cover
    """
    Numba-compatible function to calculate the HA-SSJ from fake news matrices.

    Parameters
    ----------
    T : int
        Maximum time horizon for the fake news algorithm.
    J : int
        Number of outcomes of interest.
    FN : np.array
        Fake news array of shape (J,T,T).
    dx : float
        Size of the perturbation of the shock variables (epsilon).

    Returns
    -------
    SSJ : np.array
        HA-SSJ array of shape (J,T,T).
    """
    SSJ = np.empty((J, T, T))
    SSJ[:, 0, :] = FN[:, 0, :]  # Fill in row zero
    SSJ[:, :, 0] = FN[:, :, 0]  # Fill in column zero
    for t in range(1, T):  # Loop over other rows
        for s in range(1, T):  # Loop over other columns
            SSJ[:, t, s] = SSJ[:, t - 1, s - 1] + FN[:, t, s]
    SSJ *= dx**-1.0  # Scale by dx
    return SSJ
