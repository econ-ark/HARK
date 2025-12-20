"""
Classes to solve and simulate consumption-savings model with a discrete, exogenous,
stochastic Markov state.  The only solver here extends ConsIndShockModel to
include a Markov state; the interest factor, permanent growth factor, and income
distribution can vary with the discrete state.
"""

import numpy as np

from HARK import AgentType, NullFunc
from HARK.Calibration.Income.IncomeProcesses import (
    construct_markov_lognormal_income_process_unemployment,
    get_PermShkDstn_from_IncShkDstn_markov,
    get_TranShkDstn_from_IncShkDstn_markov,
)
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsumerSolution,
    IndShockConsumerType,
    make_basic_CRRA_solution_terminal,
    make_lognormal_kNrm_init_dstn,
    make_lognormal_pLvl_init_dstn,
)
from HARK.distributions import MarkovProcess, Uniform, expected, DiscreteDistribution
from HARK.interpolation import (
    CubicInterp,
    LinearInterp,
    LowerEnvelope,
    IndexedInterp,
    MargMargValueFuncCRRA,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.rewards import (
    UtilityFuncCRRA,
    CRRAutility,
    CRRAutility_inv,
    CRRAutility_invP,
    CRRAutilityP,
    CRRAutilityP_inv,
    CRRAutilityP_invP,
    CRRAutilityPP,
)
from HARK.utilities import make_assets_grid

__all__ = ["MarkovConsumerType"]

utility = CRRAutility
utilityP = CRRAutilityP
utilityPP = CRRAutilityPP
utilityP_inv = CRRAutilityP_inv
utility_invP = CRRAutility_invP
utility_inv = CRRAutility_inv
utilityP_invP = CRRAutilityP_invP


###############################################################################

# Define some functions that can be used as constructors for MrkvArray


def make_simple_binary_markov(T_cycle, Mrkv_p11, Mrkv_p22):
    """
    Make a list of very simple Markov arrays between two binary states by specifying
    diagonal elements in each period (probability of remaining in that state).

    Parameters
    ----------
    T_cycle : int
        Number of non-terminal periods in this instance's sequential problem.
    Mrkv_p11 : [float]
        List or array of probabilities of remaining in the first state between periods.
    Mrkv_p22 : [float]
        List or array of probabilities of remaining in the second state between periods.

    Returns
    -------
    MrkvArray : [np.array]
        List of 2x2 Markov transition arrays, one for each non-terminal period.
    """
    p11 = np.array(Mrkv_p11)
    p22 = np.array(Mrkv_p22)

    if len(p11) != T_cycle or len(p22) != T_cycle:
        raise ValueError("Length of p11 and p22 probabilities must equal T_cycle!")
    if np.any(p11 > 1.0) or np.any(p22 > 1.0):
        raise ValueError("The p11 and p22 probabilities must not exceed 1!")
    if np.any(p11 < 0.0) or np.any(p22 < 0.0):
        raise ValueError("The p11 and p22 probabilities must not be less than 0!")

    MrkvArray = [
        np.array([[p11[t], 1.0 - p11[t]], [1.0 - p22[t], p22[t]]])
        for t in range(T_cycle)
    ]
    return MrkvArray


def make_ratchet_markov(T_cycle, Mrkv_ratchet_probs):
    """
    Make a list of "ratchet-style" Markov transition arrays, in which transitions
    are strictly *one way* and only by one step. Each element of the ratchet_probs
    list is a size-N vector giving the probability of progressing from state i to
    state to state i+1 in that period; progress from the topmost state reverts the
    agent to the 0th state. Set ratchet_probs[t][-1] to zero to make absorbing state.

    Parameters
    ----------
    T_cycle : int
        Number of non-terminal periods in this instance's sequential problem.
    Mrkv_ratchet_probs : [np.array]
        List of vectors of "ratchet probabilities" for each period.

    Returns
    -------
    MrkvArray : [np.array]
        List of NxN Markov transition arrays, one for each non-terminal period.
    """
    if len(Mrkv_ratchet_probs) != T_cycle:
        raise ValueError("Length of Mrkv_ratchet_probs must equal T_cycle!")

    N = Mrkv_ratchet_probs[0].size  # number of discrete states
    StateCount = np.array([Mrkv_ratchet_probs[t].size for t in range(T_cycle)])
    if np.any(StateCount != N):
        raise ValueError(
            "All periods of the problem must have the same number of discrete states!"
        )

    MrkvArray = []
    for t in range(T_cycle):
        if np.any(Mrkv_ratchet_probs[t] > 1.0):
            raise ValueError("Ratchet probabilities cannot exceed 1!")
        if np.any(Mrkv_ratchet_probs[t] < 0.0):
            raise ValueError("Ratchet probabilities cannot be below 0!")

        MrkvArray_t = np.zeros((N, N))
        for i in range(N):
            p_go = Mrkv_ratchet_probs[t][i]
            p_stay = 1.0 - p_go
            if i < (N - 1):
                i_next = i + 1
            else:
                i_next = 0
            MrkvArray_t[i, i] = p_stay
            MrkvArray_t[i, i_next] = p_go

        MrkvArray.append(MrkvArray_t)

    return MrkvArray


def make_MrkvInitDstn(MrkvPrbsInit, RNG):
    """
    The constructor function for MrkvInitDstn, the distribution of Markov states
    at model birth.

    Parameters
    ----------
    MrkvPrbsInit : np.array
        Stochastic vector specifying the distribution of initial discrete states.
    RNG : np.random.RandomState
        Agent's internal random number generator.

    Returns
    -------
    MrkvInitDstn : DiscreteDistribution
        Distribution from which discrete states at birth can be drawn.
    """
    seed = RNG.integers(0, 2**31 - 1)
    vals = np.arange(MrkvPrbsInit.size, dtype=int)
    MrkvInitDstn = DiscreteDistribution(pmv=MrkvPrbsInit, atoms=vals, seed=seed)
    return MrkvInitDstn


###############################################################################


def make_markov_solution_terminal(CRRA, MrkvArray):
    """
    Make the terminal period solution for a consumption-saving model with a discrete
    Markov state. Simply makes a basic terminal solution for IndShockConsumerType
    and then replicates the attributes N times for the N states in the terminal period.

    Parameters
    ----------
    CRRA : float
        Coefficient of relative risk aversion.
    MrkvArray : [np.array]
        List of Markov transition probabilities arrays. Only used to find the
        number of discrete states in the terminal period.

    Returns
    -------
    solution_terminal : ConsumerSolution
        Terminal period solution to the Markov consumption-saving problem.
    """
    solution_terminal_basic = make_basic_CRRA_solution_terminal(CRRA)
    StateCount_T = MrkvArray[-1].shape[1]
    N = StateCount_T  # for shorter typing

    # Make replicated terminal period solution: consume all resources, no human wealth, minimum m is 0
    solution_terminal = ConsumerSolution(
        cFunc=N * [solution_terminal_basic.cFunc],
        vFunc=N * [solution_terminal_basic.vFunc],
        vPfunc=N * [solution_terminal_basic.vPfunc],
        vPPfunc=N * [solution_terminal_basic.vPPfunc],
        mNrmMin=np.zeros(N),
        hNrm=np.zeros(N),
        MPCmin=np.ones(N),
        MPCmax=np.ones(N),
    )
    solution_terminal.cFuncX = IndexedInterp(solution_terminal.cFunc)
    return solution_terminal


def solve_one_period_ConsMarkov(
    solution_next,
    IncShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rfree,
    PermGroFac,
    MrkvArray,
    BoroCnstArt,
    aXtraGrid,
    vFuncBool,
    CubicBool,
):
    """
    Solves a single period consumption-saving problem with risky income and
    stochastic transitions between discrete states, in a Markov fashion.  Has
    identical inputs as the ConsIndShock, except for a discrete Markov transition
    rule MrkvArray.  Markov states can differ in their interest factor, permanent
    growth factor, and income distribution, so the inputs Rfree, PermGroFac, and
    IncShkDstn are lists specifying those values in each (succeeding) Markov state.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncShkDstn : [distribution.Distribution]
        A length N list of income distributions in each succeeding Markov state.
        Each income distribution is a discrete approximation to the income process
        at the beginning of the succeeding period.
    LivPrb : [float]
        Survival probability; likelihood of being alive at the beginning of the
        succeeding period conditional on the current state.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : [float]
        Risk free interest factor on end-of-period assets for each Markov
        state in the succeeding period.
    PermGroGac : [float]
        Expected permanent income growth factor at the end of this period
        for each Markov state in the succeeding period.
    MrkvArray : numpy.array
        An NxN array representing a Markov transition matrix between discrete
        states.  The i,j-th element of MrkvArray is the probability of
        moving from state i in period t to state j in period t+1.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  If it is less than the natural borrowing constraint,
        then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
        rowing constraint.
    aXtraGrid: np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear inter-
        polation.

    Returns
    -------
    solution : ConsumerSolution
        The solution to the single period consumption-saving problem. Includes
        a consumption function cFunc (using cubic or linear splines), a marg-
        inal value function vPfunc, a minimum acceptable level of normalized
        market resources mNrmMin, normalized human wealth hNrm, and bounding
        MPCs MPCmin and MPCmax.  It might also have a value function vFunc
        and marginal marginal value function vPPfunc.  All of these attributes
        are lists or arrays, with elements corresponding to the current
        Markov state.  E.g. solution.cFunc[0] is the consumption function
        when in the i=0 Markov state this period.
    """
    # Relabel the inputs that vary across Markov states
    IncShkDstn_list = IncShkDstn
    Rfree_list = np.array(Rfree)
    LivPrb_list = np.array(LivPrb)
    PermGroFac_list = np.array(PermGroFac)
    StateCountNow = MrkvArray.shape[0]
    StateCountNext = MrkvArray.shape[1]

    # Define the utility function
    uFunc = UtilityFuncCRRA(CRRA)

    # Initialize the natural borrowing constraint when entering each succeeding state
    BoroCnstNat_temp = np.zeros(StateCountNext) + np.nan

    # Find the natural borrowing constraint conditional on next period's state
    for j in range(StateCountNext):
        PermShkMinNext = np.min(IncShkDstn_list[j].atoms[0])
        TranShkMinNext = np.min(IncShkDstn_list[j].atoms[1])
        BoroCnstNat_temp[j] = (
            (solution_next.mNrmMin[j] - TranShkMinNext)
            * (PermGroFac_list[j] * PermShkMinNext)
            / Rfree_list[j]
        )

    # Initialize the natural borrowing constraint and minimum value of mNrm for
    # *this* period's Markov states, as well as a "dependency table"
    BoroCnstNat_list = np.zeros(StateCountNow) + np.nan
    mNrmMin_list = np.zeros(StateCountNow) + np.nan
    BoroCnstDependency = np.zeros((StateCountNow, StateCountNext)) + np.nan

    # The natural borrowing constraint in each current state is the *highest*
    # among next-state-conditional natural borrowing constraints that could
    # occur from this current state.
    for i in range(StateCountNow):
        possible_next_states = MrkvArray[i, :] > 0
        BoroCnstNat_list[i] = np.max(BoroCnstNat_temp[possible_next_states])

        # Explicitly handle the "None" case:
        if BoroCnstArt is None:
            mNrmMin_list[i] = BoroCnstNat_list[i]
        else:
            mNrmMin_list[i] = np.max([BoroCnstNat_list[i], BoroCnstArt])
        BoroCnstDependency[i, :] = BoroCnstNat_list[i] == BoroCnstNat_temp
    # Also creates a Boolean array indicating whether the natural borrowing
    # constraint *could* be hit when transitioning from i to j.

    # Initialize end-of-period (marginal) value functions, expected income conditional
    # on the next state, and probability of getting the worst income shock in each
    # succeeding period state
    BegOfPrd_vFunc_list = []
    BegOfPrd_vPfunc_list = []
    Ex_IncNextAll = np.zeros(StateCountNext) + np.nan
    WorstIncPrbAll = np.zeros(StateCountNext) + np.nan

    # Loop through each next-period-state and calculate the beginning-of-period
    # (marginal) value function
    for j in range(StateCountNext):
        # Condition values on next period's state (and record a couple for later use)
        Rfree = Rfree_list[j]
        PermGroFac = PermGroFac_list[j]
        LivPrb = LivPrb_list[j]
        # mNrmMinNow = self.mNrmMin_list[state_index]
        BoroCnstNat = BoroCnstNat_temp[j]

        # Unpack the income distribution in next period's Markov state
        IncShkDstn = IncShkDstn_list[j]
        ShkPrbsNext = IncShkDstn.pmv
        PermShkValsNext = IncShkDstn.atoms[0]
        TranShkValsNext = IncShkDstn.atoms[1]
        PermShkMinNext = np.min(PermShkValsNext)
        TranShkMinNext = np.min(TranShkValsNext)

        # Calculate the probability that we get the worst possible income draw
        IncNext = PermShkValsNext * TranShkValsNext
        WorstIncNext = PermShkMinNext * TranShkMinNext
        WorstIncPrb = np.sum(ShkPrbsNext[IncNext == WorstIncNext])
        # WorstIncPrb is the "Weierstrass p" concept: the odds we get the WORST thing

        DiscFacEff = DiscFac  # survival probability LivPrb represents probability
        # from *current* state, so DiscFacEff is just DiscFac for now

        # Unpack next period's (marginal) value function
        vFuncNext = solution_next.vFunc[j]  # This is None when vFuncBool is False
        vPfuncNext = solution_next.vPfunc[j]
        vPPfuncNext = solution_next.vPPfunc[j]  # This is None when CubicBool is False

        # Compute expected income next period and record worst income probability
        Ex_IncNextAll[j] = np.dot(ShkPrbsNext, PermShkValsNext * TranShkValsNext)
        WorstIncPrbAll[j] = WorstIncPrb

        # Construct the BEGINNING-of-period marginal value function conditional
        # on next period's state and add it to the list of value functions

        # Get data to construct the end-of-period marginal value function (conditional on next state)
        aNrmNext = np.asarray(aXtraGrid) + BoroCnstNat

        # Define local functions for taking future expectations
        def calc_mNrmNext(S, a, R):
            return R / (PermGroFac * S["PermShk"]) * a + S["TranShk"]

        def calc_vNext(S, a, R):
            return (
                S["PermShk"] ** (1.0 - CRRA) * PermGroFac ** (1.0 - CRRA)
            ) * vFuncNext(calc_mNrmNext(S, a, R))

        def calc_vPnext(S, a, R):
            return S["PermShk"] ** (-CRRA) * vPfuncNext(calc_mNrmNext(S, a, R))

        def calc_vPPnext(S, a, R):
            return S["PermShk"] ** (-CRRA - 1.0) * vPPfuncNext(calc_mNrmNext(S, a, R))

        # Calculate beginning-of-period marginal value of assets at each gridpoint
        vPfacEff = DiscFacEff * Rfree * PermGroFac ** (-CRRA)
        BegOfPrd_vPnext = vPfacEff * expected(
            calc_vPnext, IncShkDstn, args=(aNrmNext, Rfree)
        )

        # "Decurved" marginal value
        BegOfPrd_vPnvrsNext = uFunc.derinv(BegOfPrd_vPnext, order=(1, 0))

        # Make the beginning-of-period pseudo-inverse marginal value of assets
        # function conditionalon next period's state
        if CubicBool:
            # Calculate end-of-period marginal marginal value of assets at each gridpoint
            vPPfacEff = DiscFacEff * Rfree * Rfree * PermGroFac ** (-CRRA - 1.0)
            BegOfPrd_vPPnext = vPPfacEff * expected(
                calc_vPPnext, IncShkDstn, args=(aNrmNext, Rfree)
            )
            # "Decurved" marginal marginal value
            BegOfPrd_vPnvrsPnext = BegOfPrd_vPPnext * uFunc.derinv(
                BegOfPrd_vPnext, order=(1, 1)
            )

            # Construct the end-of-period marginal value function conditional on the next state.
            BegOfPrd_vPnvrsFunc = CubicInterp(
                aNrmNext,
                BegOfPrd_vPnvrsNext,
                BegOfPrd_vPnvrsPnext,
                lower_extrap=True,
            )
            # TODO: Should not be lower extrap, add point at BoroCnstNat
        else:
            BegOfPrd_vPnvrsFunc = LinearInterp(
                aNrmNext, BegOfPrd_vPnvrsNext, lower_extrap=True
            )
            # TODO: Should not be lower extrap, add point at BoroCnstNat

        # "Recurve" the pseudo-inverse marginal value function
        BegOfPrd_vPfunc = MargValueFuncCRRA(BegOfPrd_vPnvrsFunc, CRRA)
        BegOfPrd_vPfunc_list.append(BegOfPrd_vPfunc)

        # Construct the beginning-of-period value functional conditional on next
        # period's state and add it to the list of value functions
        if vFuncBool:
            # Calculate end-of-period value, its derivative, and their pseudo-inverse
            BegOfPrd_vNext = DiscFacEff * expected(
                calc_vNext, IncShkDstn, args=(aNrmNext, Rfree)
            )
            # value transformed through inverse utility
            BegOfPrd_vNvrsNext = uFunc.inv(BegOfPrd_vNext)
            BegOfPrd_vNvrsPnext = BegOfPrd_vPnext * uFunc.derinv(
                BegOfPrd_vNext, order=(0, 1)
            )
            BegOfPrd_vNvrsNext = np.insert(BegOfPrd_vNvrsNext, 0, 0.0)
            BegOfPrd_vNvrsPnext = np.insert(
                BegOfPrd_vNvrsPnext, 0, BegOfPrd_vNvrsPnext[0]
            )
            # This is a very good approximation, vNvrsPP = 0 at the asset minimum

            # Construct the end-of-period value function
            aNrm_temp = np.insert(aNrmNext, 0, BoroCnstNat)
            BegOfPrd_vNvrsFunc = LinearInterp(
                aNrm_temp,
                BegOfPrd_vNvrsNext,
            )
            BegOfPrd_vFunc = ValueFuncCRRA(BegOfPrd_vNvrsFunc, CRRA)
            BegOfPrd_vFunc_list.append(BegOfPrd_vFunc)

    # BegOfPrdvP is marginal value conditional on *next* period's state.
    # Take expectations over Markov transitions to get EndOfPrdvP conditional on
    # *this* period's Markov state.

    # Find unique values of minimum acceptable end-of-period assets (and the
    # current period states for which they apply).
    aNrmMin_unique, Mrkv_inverse = np.unique(BoroCnstNat_list, return_inverse=True)
    possible_transitions = MrkvArray > 0

    # Initialize end-of-period marginal value (and marg marg value) at each
    # asset gridpoint for each current period state
    EndOfPrd_vP = np.zeros((StateCountNow, aXtraGrid.size))
    EndOfPrd_vPP = np.zeros((StateCountNow, aXtraGrid.size))

    # Calculate end-of-period marginal value (and marg marg value) at each
    # asset gridpoint for each current period state, grouping current states
    # by their natural borrowing constraint
    for k in range(aNrmMin_unique.size):
        # Get the states for which this minimum applies amd the aNrm grid for
        # this set of current states
        aNrmMin = aNrmMin_unique[k]  # minimum assets for this pass
        which_states = Mrkv_inverse == k
        aNrmNow = aNrmMin + aXtraGrid  # assets grid for this pass

        # Make arrays to hold successor period's beginning-of-period (marginal)
        # marginal value if we transition to it
        BegOfPrd_vPnext = np.zeros((StateCountNext, aXtraGrid.size))
        BegOfPrd_vPPnext = np.zeros((StateCountNext, aXtraGrid.size))

        # Loop through future Markov states and fill in those values, but only
        # look at future states that can actually be reached from our current
        # set of states (for this natural borrowing constraint value)
        for j in range(StateCountNext):
            if not np.any(np.logical_and(possible_transitions[:, j], which_states)):
                continue

            BegOfPrd_vPnext[j, :] = BegOfPrd_vPfunc_list[j](aNrmNow)
            # Add conditional end-of-period (marginal) marginal value to the arrays
            if CubicBool:
                BegOfPrd_vPPnext[j, :] = BegOfPrd_vPfunc_list[j].derivativeX(aNrmNow)

        # Weight conditional marginal values by transition probabilities
        # to get unconditional marginal (marginal) value at each gridpoint.
        EndOfPrd_vP_temp = np.dot(MrkvArray, BegOfPrd_vPnext)

        # Only take the states for which this asset minimum applies
        EndOfPrd_vP[which_states, :] = EndOfPrd_vP_temp[which_states, :]

        # Do the same thing for marginal marginal value
        if CubicBool:
            EndOfPrd_vPP_temp = np.dot(MrkvArray, BegOfPrd_vPPnext)
            EndOfPrd_vPP[which_states, :] = EndOfPrd_vPP_temp[which_states, :]

    # Store the results as attributes of self, scaling end of period marginal value by survival probability from each current state
    LivPrb_tiled = np.tile(
        np.reshape(LivPrb_list, (StateCountNow, 1)), (1, aXtraGrid.size)
    )
    EndOfPrd_vP = LivPrb_tiled * EndOfPrd_vP
    if CubicBool:
        EndOfPrd_vPP = LivPrb_tiled * EndOfPrd_vPP

    # Calculate the bounding MPCs and PDV of human wealth for each state

    # Calculate probability of getting the "worst" income shock and transition
    # from each current state
    WorstIncPrb_array = BoroCnstDependency * np.tile(
        np.reshape(WorstIncPrbAll, (1, StateCountNext)), (StateCountNow, 1)
    )
    temp_array = MrkvArray * WorstIncPrb_array
    WorstIncPrbNow = np.sum(temp_array, axis=1)

    # Calculate expectation of upper bound of next period's MPC
    Ex_MPCmaxNext = (
        np.dot(temp_array, Rfree_list ** (1.0 - CRRA) * solution_next.MPCmax ** (-CRRA))
        / WorstIncPrbNow
    ) ** (-1.0 / CRRA)

    # Calculate limiting upper bound of MPC this period for each Markov state
    DiscFacEff_temp = DiscFac * LivPrb_list
    MPCmaxNow = 1.0 / (
        1.0 + ((DiscFacEff_temp * WorstIncPrbNow) ** (1.0 / CRRA)) / Ex_MPCmaxNext
    )
    MPCmaxEff = MPCmaxNow
    MPCmaxEff[BoroCnstNat_list < mNrmMin_list] = 1.0

    # Calculate the current Markov-state-conditional PDV of human wealth, correctly
    # accounting for risky returns and risk aversion
    hNrmPlusIncNext = Ex_IncNextAll + solution_next.hNrm
    R_adj = np.dot(MrkvArray, Rfree_list ** (1.0 - CRRA))
    hNrmNow = (
        np.dot(MrkvArray, (PermGroFac_list / Rfree_list**CRRA) * hNrmPlusIncNext)
        / R_adj
    )

    # Calculate the lower bound on MPC as m gets arbitrarily large
    temp = (
        DiscFacEff_temp
        * np.dot(
            MrkvArray, solution_next.MPCmin ** (-CRRA) * Rfree_list ** (1.0 - CRRA)
        )
    ) ** (1.0 / CRRA)
    MPCminNow = 1.0 / (1.0 + temp)

    # Find consumption and market resources corresponding to each end-of-period
    # assets point for each state (and add an additional point at the lower bound)
    aNrmNow = (aXtraGrid)[np.newaxis, :] + np.array(BoroCnstNat_list)[:, np.newaxis]
    cNrmNow = uFunc.derinv(EndOfPrd_vP, order=(1, 0))
    mNrmNow = cNrmNow + aNrmNow  # Endogenous mNrm gridpoints
    cNrmNow = np.hstack((np.zeros((StateCountNow, 1)), cNrmNow))
    mNrmNow = np.hstack((np.reshape(mNrmMin_list, (StateCountNow, 1)), mNrmNow))

    # Calculate the MPC at each market resource gridpoint in each state (if desired)
    if CubicBool:
        dcda = EndOfPrd_vPP / uFunc.der(cNrmNow[:, 1:], order=2)  # drop first
        MPCnow = dcda / (dcda + 1.0)
        MPCnow = np.hstack((np.reshape(MPCmaxNow, (StateCountNow, 1)), MPCnow))

    # Initialize an empty solution to which we'll add state-conditional solutions
    solution = ConsumerSolution()

    # Loop through each current period state and add its solution to the overall solution
    for i in range(StateCountNow):
        # Set current-Markov-state-conditional human wealth and MPC bounds
        hNrmNow_i = hNrmNow[i]
        MPCminNow_i = MPCminNow[i]
        mNrmMin_i = mNrmMin_list[i]

        # Construct the consumption function by combining the constrained and unconstrained portions
        cFuncNowCnst = LinearInterp(
            np.array([mNrmMin_list[i], mNrmMin_list[i] + 1.0]), np.array([0.0, 1.0])
        )
        if CubicBool:
            cFuncNowUnc = CubicInterp(
                mNrmNow[i, :],
                cNrmNow[i, :],
                MPCnow[i, :],
                MPCminNow_i * hNrmNow_i,
                MPCminNow_i,
            )
        else:
            cFuncNowUnc = LinearInterp(
                mNrmNow[i, :], cNrmNow[i, :], MPCminNow_i * hNrmNow_i, MPCminNow_i
            )
        cFuncNow = LowerEnvelope(cFuncNowUnc, cFuncNowCnst)

        # Make the marginal (marginal) value function
        vPfuncNow = MargValueFuncCRRA(cFuncNow, CRRA)
        if CubicBool:
            vPPfuncNow = MargMargValueFuncCRRA(cFuncNow, CRRA)
        else:
            vPPfuncNow = NullFunc()

        # Make the value function for this state if requested
        if vFuncBool:
            # Make state-conditional grids of market resources and consumption
            mNrm_for_vFunc = mNrmMin_i + aXtraGrid
            cNrm_for_vFunc = cFuncNow(mNrm_for_vFunc)
            aNrm_for_vFunc = mNrm_for_vFunc - cNrm_for_vFunc

            # Calculate end-of-period value at each gridpoint
            BegOfPrd_v_temp = np.zeros((StateCountNow, aXtraGrid.size))
            for j in range(StateCountNext):
                if possible_transitions[i, j]:
                    BegOfPrd_v_temp[j, :] = BegOfPrd_vFunc_list[j](aNrm_for_vFunc)
            EndOfPrd_v = np.dot(MrkvArray[i, :], BegOfPrd_v_temp)

            # Calculate (normalized) value and marginal value at each gridpoint
            v_now = uFunc(cNrm_for_vFunc) + EndOfPrd_v
            vP_now = uFunc.der(cNrm_for_vFunc)

            # Make a "decurved" value function with the inverse utility function
            # value transformed through inverse utility
            vNvrs_now = uFunc.inv(v_now)
            vNvrsP_now = vP_now * uFunc.derinv(v_now, order=(0, 1))
            mNrm_temp = np.insert(mNrm_for_vFunc, 0, mNrmMin_i)  # add the lower bound
            vNvrs_now = np.insert(vNvrs_now, 0, 0.0)
            vNvrsP_now = np.insert(
                vNvrsP_now, 0, MPCmaxEff[i] ** (-CRRA / (1.0 - CRRA))
            )
            # MPCminNvrs = MPCminNow[i] ** (-CRRA / (1.0 - CRRA))
            vNvrsFuncNow = LinearInterp(
                mNrm_temp,
                vNvrs_now,
                # vNvrsP_now,
            )  # MPCminNvrs * hNrmNow_i, MPCminNvrs)
            # The bounding function for the pseudo-inverse value function is incorrect.
            # TODO: Resolve this strange issue; extrapolation is suppressed for now.

            # "Recurve" the decurved value function and add it to the list
            vFuncNow = ValueFuncCRRA(vNvrsFuncNow, CRRA)

        else:
            vFuncNow = NullFunc()

        # Make the current-Markov-state-conditional solution
        solution_cond = ConsumerSolution(
            cFunc=cFuncNow,
            vFunc=vFuncNow,
            vPfunc=vPfuncNow,
            vPPfunc=vPPfuncNow,
            mNrmMin=mNrmMin_i,
        )

        # Add the current-state-conditional solution to the overall period solution
        solution.append_solution(solution_cond)

    # Add the lower bounds of market resources, MPC limits, human resources,
    # and the value functions to the overall solution, then return it
    solution.mNrmMin = mNrmMin_list
    solution.hNrm = hNrmNow
    solution.MPCmin = MPCminNow
    solution.MPCmax = MPCmaxNow
    solution.cFuncX = IndexedInterp(solution.cFunc)
    return solution


####################################################################################################
####################################################################################################

# Make a dictionary of constructors for the markov consumption-saving model
markov_constructor_dict = {
    "IncShkDstn": construct_markov_lognormal_income_process_unemployment,
    "PermShkDstn": get_PermShkDstn_from_IncShkDstn_markov,
    "TranShkDstn": get_TranShkDstn_from_IncShkDstn_markov,
    "aXtraGrid": make_assets_grid,
    "MrkvArray": make_simple_binary_markov,
    "solution_terminal": make_markov_solution_terminal,
    "kNrmInitDstn": make_lognormal_kNrm_init_dstn,
    "pLvlInitDstn": make_lognormal_pLvl_init_dstn,
    "MrkvInitDstn": make_MrkvInitDstn,
}

# Make a dictionary with parameters for the default constructor for kNrmInitDstn
default_kNrmInitDstn_params = {
    "kLogInitMean": -12.0,  # Mean of log initial capital
    "kLogInitStd": 0.0,  # Stdev of log initial capital
    "kNrmInitCount": 15,  # Number of points in initial capital discretization
}

# Make a dictionary with parameters for the default constructor for pLvlInitDstn
default_pLvlInitDstn_params = {
    "pLogInitMean": 0.0,  # Mean of log permanent income
    "pLogInitStd": 0.0,  # Stdev of log permanent income
    "pLvlInitCount": 15,  # Number of points in initial capital discretization
}

# Default parameters to make IncShkDstn using construct_lognormal_income_process_unemployment
default_IncShkDstn_params = {
    "PermShkStd": np.array(
        [[0.1, 0.1]]
    ),  # Standard deviation of log permanent income shocks
    "PermShkCount": 7,  # Number of points in discrete approximation to permanent income shocks
    "TranShkStd": np.array(
        [[0.1, 0.1]]
    ),  # Standard deviation of log transitory income shocks
    "TranShkCount": 7,  # Number of points in discrete approximation to transitory income shocks
    "UnempPrb": np.array([0.05, 0.05]),  # Probability of unemployment while working
    "IncUnemp": np.array(
        [0.3, 0.3]
    ),  # Unemployment benefits replacement rate while working
    "T_retire": 0,  # Period of retirement (0 --> no retirement)
    "UnempPrbRet": None,  # Probability of "unemployment" while retired
    "IncUnempRet": None,  # "Unemployment" benefits when retired
}

# Default parameters to make aXtraGrid using make_assets_grid
default_aXtraGrid_params = {
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 20,  # Maximum end-of-period "assets above minimum" value
    "aXtraNestFac": 3,  # Exponential nesting factor for aXtraGrid
    "aXtraCount": 48,  # Number of points in the grid of "assets above minimum"
    "aXtraExtra": None,  # Additional other values to add in grid (optional)
}

# Default parameters to make MrkvArray using make_simple_binary_markov
default_MrkvArray_params = {
    "Mrkv_p11": [0.9],  # Probability of remaining in binary state 1
    "Mrkv_p22": [0.4],  # Probability of remaining in binary state 2
}

# Make a dictionary to specify an idiosyncratic income shocks consumer type
init_indshk_markov = {
    # BASIC HARK PARAMETERS REQUIRED TO SOLVE THE MODEL
    "cycles": 1,  # Finite, non-cyclic model
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "constructors": markov_constructor_dict,  # See dictionary above
    "pseudo_terminal": False,  # Terminal period really does exist
    "global_markov": False,  # Whether the Markov state is shared across agents
    # PRIMITIVE RAW PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": [np.array([1.03, 1.03])],  # Interest factor on retained assets
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [np.array([0.98, 0.98])],  # Survival probability after each period
    "PermGroFac": [np.array([0.99, 1.03])],  # Permanent income growth factor
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint
    "vFuncBool": False,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Whether to use cubic spline interpolation when True
    # (Uses linear spline interpolation for cFunc when False)
    # PARAMETERS REQUIRED TO SIMULATE THE MODEL
    "AgentCount": 10000,  # Number of agents of this type
    "T_age": None,  # Age after which simulated agents are automatically killed
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
    "MrkvPrbsInit": np.array([1.0, 0.0]),  # Initial distribution of discrete state
    # (The portion of PermGroFac attributable to aggregate productivity growth)
    "NewbornTransShk": False,  # Whether Newborns have transitory shock
    # ADDITIONAL OPTIONAL PARAMETERS
    "PerfMITShk": False,  # Do Perfect Foresight MIT Shock
    # (Forces Newborns to follow solution path of the agent they replaced if True)
    "neutral_measure": False,  # Whether to use permanent income neutral measure (see Harmenberg 2021)
}
init_indshk_markov.update(default_IncShkDstn_params)
init_indshk_markov.update(default_aXtraGrid_params)
init_indshk_markov.update(default_MrkvArray_params)
init_indshk_markov.update(default_kNrmInitDstn_params)
init_indshk_markov.update(default_pLvlInitDstn_params)


class MarkovConsumerType(IndShockConsumerType):
    """
    An agent in the Markov consumption-saving model.  His problem is defined by a sequence
    of income distributions, survival probabilities, discount factors, and permanent
    income growth rates, as well as time invariant values for risk aversion, the
    interest rate, the grid of end-of-period assets, and how he is borrowing constrained.
    """

    time_vary_ = IndShockConsumerType.time_vary_ + ["MrkvArray"]

    # Mrkv is both a shock and a state
    shock_vars_ = IndShockConsumerType.shock_vars_ + ["Mrkv"]
    state_vars = IndShockConsumerType.state_vars + ["Mrkv"]
    default_ = {
        "params": init_indshk_markov,
        "solver": solve_one_period_ConsMarkov,
        "model": "ConsMarkov.yaml",
    }
    distributions = [
        "IncShkDstn",
        "PermShkDstn",
        "TranShkDstn",
        "kNrmInitDstn",
        "pLvlInitDstn",
        "MrkvInitDstn",
    ]

    def check_markov_inputs(self):
        """
        Many parameters used by MarkovConsumerType are arrays.  Make sure those arrays are the
        right shape.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        StateCount = self.MrkvArray[0].shape[0]

        # Check that arrays are the right shape
        for t in range(self.T_cycle):
            if not isinstance(self.Rfree[t], np.ndarray) or self.Rfree[t].shape != (
                StateCount,
            ):
                raise ValueError(
                    "Rfree[t] not the right shape, it should be an array of Rfree of all the states."
                )

        # Check that arrays in lists are the right shape
        for MrkvArray_t in self.MrkvArray:
            if not isinstance(MrkvArray_t, np.ndarray) or MrkvArray_t.shape != (
                StateCount,
                StateCount,
            ):
                raise ValueError(
                    "MrkvArray not the right shape, it should be of the size states*states."
                )
        for LivPrb_t in self.LivPrb:
            if not isinstance(LivPrb_t, np.ndarray) or LivPrb_t.shape != (StateCount,):
                raise ValueError(
                    "Array in LivPrb is not the right shape, it should be an array of length equal to number of states"
                )
        for PermGroFac_t in self.PermGroFac:
            if not isinstance(PermGroFac_t, np.ndarray) or PermGroFac_t.shape != (
                StateCount,
            ):
                raise ValueError(
                    "Array in PermGroFac is not the right shape, it should be an array of length equal to number of states"
                )

        # Now check the income distribution.
        # Note IncShkDstn is (potentially) time-varying, so it is in time_vary.
        # Therefore it is a list, and each element of that list responds to the income distribution
        # at a particular point in time.
        for IncShkDstn_t in self.IncShkDstn:
            if not isinstance(IncShkDstn_t, list):
                raise ValueError(
                    "self.IncShkDstn is time varying and so must be a list"
                    + "of lists of Distributions, one per Markov State. Found "
                    + f"{self.IncShkDstn} instead"
                )
            elif len(IncShkDstn_t) != StateCount:
                raise ValueError(
                    "List in IncShkDstn is not the right length, it should be length equal to number of states"
                )

    def pre_solve(self):
        """
        Check to make sure that the inputs that are specific to MarkovConsumerType
        are of the right shape (if arrays) or length (if lists).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        AgentType.pre_solve(self)
        self.check_markov_inputs()
        self.construct("solution_terminal")

    def initialize_sim(self):
        self.shocks["Mrkv"] = np.zeros(self.AgentCount, dtype=int)
        IndShockConsumerType.initialize_sim(self)

        # Need to initialize markov state to be the same for all agents
        if self.global_markov:
            base_draw = Uniform(seed=self.RNG.integers(0, 2**31 - 1)).draw(1)
            Cutoffs = np.cumsum(np.array(self.MrkvPrbsInit))
            self.shocks["Mrkv"] = np.ones(self.AgentCount) * np.searchsorted(
                Cutoffs, base_draw
            ).astype(int)
        self.shocks["Mrkv"] = self.shocks["Mrkv"].astype(int)

    def sim_death(self):
        """
        Determines which agents die this period and must be replaced.  Uses the sequence in LivPrb
        to determine survival probabilities for each agent.

        Parameters
        ----------
        None

        Returns
        -------
        which_agents : np.array(bool)
            Boolean array of size AgentCount indicating which agents die.
        """
        # Determine who dies
        LivPrb = np.array(self.LivPrb)[
            self.t_cycle - 1, self.shocks["Mrkv"]
        ]  # Time has already advanced, so look back one
        DiePrb = 1.0 - LivPrb
        DeathShks = Uniform(seed=self.RNG.integers(0, 2**31 - 1)).draw(
            N=self.AgentCount
        )
        which_agents = DeathShks < DiePrb
        if self.T_age is not None:  # Kill agents that have lived for too many periods
            too_old = self.t_age >= self.T_age
            which_agents = np.logical_or(which_agents, too_old)
        return which_agents

    def sim_birth(self, which_agents):
        """
        Makes new Markov consumer by drawing initial normalized assets, permanent income levels, and
        discrete states. Calls IndShockConsumerType.sim_birth, then draws from initial Markov distribution.

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        """
        # Get initial assets and permanent income
        IndShockConsumerType.sim_birth(self, which_agents)

        # Markov state is not changed if it is set at the global level
        if not self.global_markov:
            N = np.sum(which_agents)
            self.state_now["Mrkv"][which_agents] = self.MrkvInitDstn.draw(N)

    def get_markov_states(self):
        """
        Draw new Markov states for each agent in the simulated population, using
        the attribute MrkvArray to determine transition probabilities.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Don't change Markov state for those who were just born (unless global_markov)
        dont_change = self.t_age == 0
        if self.t_sim == 0:  # Respect initial distribution of Markov states
            dont_change[:] = True

        # Determine which agents are in which states right now
        MrkvPrev = self.shocks["Mrkv"]
        MrkvNow = np.zeros(self.AgentCount, dtype=int)

        # Draw new Markov states for each agent
        for t in range(self.T_cycle):
            markov_process = MarkovProcess(
                self.MrkvArray[t], seed=self.RNG.integers(0, 2**31 - 1)
            )
            right_age = self.t_cycle == t
            MrkvNow[right_age] = markov_process.draw(MrkvPrev[right_age])
        if not self.global_markov:
            MrkvNow[dont_change] = MrkvPrev[dont_change]

        self.shocks["Mrkv"] = MrkvNow.astype(int)

    def get_shocks(self):
        """
        Gets new Markov states and permanent and transitory income shocks for this period.  Samples
        from IncShkDstn for each period-state in the cycle.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.get_markov_states()
        MrkvNow = self.shocks["Mrkv"]

        # Now get income shocks for each consumer, by cycle-time and discrete state
        PermShkNow = np.zeros(self.AgentCount)  # Initialize shock arrays
        TranShkNow = np.zeros(self.AgentCount)
        for t in range(self.T_cycle):
            for j in range(self.MrkvArray[t].shape[0]):
                these = np.logical_and(t == self.t_cycle, j == MrkvNow)
                N = np.sum(these)
                if N > 0:
                    IncShkDstnNow = self.IncShkDstn[t - 1][
                        j
                    ]  # set current income distribution
                    PermGroFacNow = self.PermGroFac[t - 1][
                        j
                    ]  # and permanent growth factor

                    # Get random draws of income shocks from the discrete distribution
                    EventDraws = IncShkDstnNow.draw_events(N)
                    PermShkNow[these] = (
                        IncShkDstnNow.atoms[0][EventDraws] * PermGroFacNow
                    )  # permanent "shock" includes expected growth
                    TranShkNow[these] = IncShkDstnNow.atoms[1][EventDraws]
        newborn = self.t_age == 0
        PermShkNow[newborn] = 1.0
        TranShkNow[newborn] = 1.0
        self.shocks["PermShk"] = PermShkNow
        self.shocks["TranShk"] = TranShkNow

    def read_shocks_from_history(self):
        """
        A slight modification of AgentType.read_shocks that makes sure that MrkvNow is int, not float.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        IndShockConsumerType.read_shocks_from_history(self)
        self.shocks["Mrkv"] = self.shocks["Mrkv"].astype(int)

    def get_Rport(self):
        """
        Returns an array of size self.AgentCount with interest factor that varies with discrete state.
        This represents the portfolio return in this model.

        Parameters
        ----------
        None

        Returns
        -------
        RfreeNow : np.array
             Array of size self.AgentCount with risk free interest rate for each agent.
        """
        RfreeNow = np.zeros(self.AgentCount)
        for t in range(self.T_cycle):
            these = self.t_cycle == t
            RfreeNow[these] = self.Rfree[t][self.shocks["Mrkv"][these]]
        return RfreeNow

    def get_controls(self):
        """
        Calculates consumption for each consumer of this type using the consumption functions.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        MPCnow = np.zeros(self.AgentCount) + np.nan
        J = self.MrkvArray[0].shape[0]

        MrkvBoolArray = np.zeros((J, self.AgentCount), dtype=bool)
        for j in range(J):
            MrkvBoolArray[j, :] = j == self.shocks["Mrkv"]

        for t in range(self.T_cycle):
            right_t = t == self.t_cycle
            for j in range(J):
                these = np.logical_and(right_t, MrkvBoolArray[j, :])
                cNrmNow[these], MPCnow[these] = (
                    self.solution[t]
                    .cFunc[j]
                    .eval_with_derivative(self.state_now["mNrm"][these])
                )
        self.controls["cNrm"] = cNrmNow
        self.MPCnow = MPCnow

    def get_poststates(self):
        super().get_poststates()
        self.state_now["Mrkv"] = self.shocks["Mrkv"].copy()

    def calc_bounding_values(self):  # pragma: nocover
        """
        Calculate human wealth plus minimum and maximum MPC in an infinite
        horizon model with only one period repeated indefinitely.  Store results
        as attributes of self.  Human wealth is the present discounted value of
        expected future income after receiving income this period, ignoring mort-
        ality.  The maximum MPC is the limit of the MPC as m --> mNrmMin.  The
        minimum MPC is the limit of the MPC as m --> infty.  Results are all
        np.array with elements corresponding to each Markov state.

        NOT YET IMPLEMENTED FOR THIS CLASS

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        raise NotImplementedError()

    def make_euler_error_func(self, mMax=100, approx_inc_dstn=True):  # pragma: nocover
        """
        Creates a "normalized Euler error" function for this instance, mapping
        from market resources to "consumption error per dollar of consumption."
        Stores result in attribute eulerErrorFunc as an interpolated function.
        Has option to use approximate income distribution stored in self.IncShkDstn
        or to use a (temporary) very dense approximation.

        NOT YET IMPLEMENTED FOR THIS CLASS

        Parameters
        ----------
        mMax : float
            Maximum normalized market resources for the Euler error function.
        approx_inc_dstn : Boolean
            Indicator for whether to use the approximate discrete income distri-
            bution stored in self.IncShkDstn[0], or to use a very accurate
            discrete approximation instead.  When True, uses approximation in
            IncShkDstn; when False, makes and uses a very dense approximation.

        Returns
        -------
        None
        """
        raise NotImplementedError()

    def check_conditions(self, verbose=None):  # pragma: nocover
        raise NotImplementedError()

    def calc_limiting_values(self):  # pragma: nocover
        raise NotImplementedError()
