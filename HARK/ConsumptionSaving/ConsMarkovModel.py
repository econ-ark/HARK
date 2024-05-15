"""
Classes to solve and simulate consumption-savings model with a discrete, exogenous,
stochastic Markov state.  The only solver here extends ConsIndShockModel to
include a Markov state; the interest factor, permanent growth factor, and income
distribution can vary with the discrete state.
"""

import numpy as np

from HARK import AgentType, NullFunc
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsumerSolution,
    IndShockConsumerType,
    PerfForesightConsumerType,
)
from HARK.distribution import MarkovProcess, Uniform, expected
from HARK.interpolation import (
    CubicInterp,
    LinearInterp,
    LowerEnvelope,
    MargMargValueFuncCRRA,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)


import scipy.sparse as sp
from HARK.utilities import(
    
    jump_to_grid_1D,
    jump_to_grid_2D,
    gen_tran_matrix_1D,
    gen_tran_matrix_2D,
    )

from copy import deepcopy

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

__all__ = ["MarkovConsumerType"]

utility = CRRAutility
utilityP = CRRAutilityP
utilityPP = CRRAutilityPP
utilityP_inv = CRRAutilityP_inv
utility_invP = CRRAutility_invP
utility_inv = CRRAutility_inv
utilityP_invP = CRRAutilityP_invP


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
            BegOfPrd_vNvrsFunc = CubicInterp(
                aNrm_temp, BegOfPrd_vNvrsNext, BegOfPrd_vNvrsPnext
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
            vNvrsFuncNow = CubicInterp(
                mNrm_temp,
                vNvrs_now,
                vNvrsP_now,
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
    return solution


####################################################################################################
####################################################################################################


class MarkovConsumerType(IndShockConsumerType):
    """
    An agent in the Markov consumption-saving model.  His problem is defined by a sequence
    of income distributions, survival probabilities, discount factors, and permanent
    income growth rates, as well as time invariant values for risk aversion, the
    interest rate, the grid of end-of-period assets, and how he is borrowing constrained.
    """

    time_vary_ = IndShockConsumerType.time_vary_ + ["MrkvArray"]

    # Is "Mrkv" a shock or a state?
    shock_vars_ = IndShockConsumerType.shock_vars_ + ["Mrkv"]
    state_vars = IndShockConsumerType.state_vars + ["Mrkv"]

    def __init__(self, **kwds):
        IndShockConsumerType.__init__(self, **kwds)
        self.solve_one_period = solve_one_period_ConsMarkov

        if not hasattr(self, "global_markov"):
            self.global_markov = False

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
        if isinstance(self.Rfree,list):
        
            for rfree_t in self.Rfree:
                
                # Check that arrays are the right shape
                if not isinstance(rfree_t, np.ndarray) or rfree_t.shape != (StateCount,):
                    raise ValueError(
                        "Rfree not the right shape, it should an array of Rfree of all the states."
                    )
        else:
            
            # Check that arrays are the right shape
            if not isinstance(self.Rfree, np.ndarray) or self.Rfree.shape != (StateCount,):
                raise ValueError(
                    "Rfree not the right shape, it should an array of Rfree of all the states."
                )

        # Check that arrays in lists are the right shape
        for MrkvArray_t in self.MrkvArray:
            if not isinstance(MrkvArray_t, np.ndarray) or MrkvArray_t.shape != (
                StateCount,
                StateCount,
            ):
                raise ValueError(
                    "MrkvArray not the right shape, it should be of the size states*statres."
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
        # at a particular point in time.  Each income distribution at a point in time should itself
        # be a list, with each element corresponding to the income distribution
        # conditional on a particular Markov state.
        # TODO: should this be a numpy array too?
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

    def update_solution_terminal(self):
        """
        Update the terminal period solution.  This method should be run when a
        new AgentType is created or when CRRA changes.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        IndShockConsumerType.update_solution_terminal(self)

        # Make replicated terminal period solution: consume all resources, no human wealth, minimum m is 0
        StateCount = self.MrkvArray[0].shape[0]
        self.solution_terminal.cFunc = StateCount * [self.cFunc_terminal_]
        self.solution_terminal.vFunc = StateCount * [self.solution_terminal.vFunc]
        self.solution_terminal.vPfunc = StateCount * [self.solution_terminal.vPfunc]
        self.solution_terminal.vPPfunc = StateCount * [self.solution_terminal.vPPfunc]
        self.solution_terminal.mNrmMin = np.zeros(StateCount)
        self.solution_terminal.hRto = np.zeros(StateCount)
        self.solution_terminal.MPCmax = np.ones(StateCount)
        self.solution_terminal.MPCmin = np.ones(StateCount)

    def initialize_sim(self):
        self.shocks["Mrkv"] = np.zeros(self.AgentCount, dtype=int)
        IndShockConsumerType.initialize_sim(self)
        if (
            self.global_markov
        ):  # Need to initialize markov state to be the same for all agents
            base_draw = Uniform(seed=self.RNG.integers(0, 2**31 - 1)).draw(1)
            Cutoffs = np.cumsum(np.array(self.MrkvPrbsInit))
            self.shocks["Mrkv"] = np.ones(self.AgentCount) * np.searchsorted(
                Cutoffs, base_draw
            ).astype(int)
        self.shocks["Mrkv"] = self.shocks["Mrkv"].astype(int)

    def reset_rng(self):
        """
        Extended method that ensures random shocks are drawn from the same sequence
        on each simulation, which is important for structural estimation.  This
        method is called automatically by initialize_sim().

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        PerfForesightConsumerType.reset_rng(self)

        # Reset IncShkDstn if it exists (it might not because reset_rng is called at init)
        if hasattr(self, "IncShkDstn"):
            T = len(self.IncShkDstn)
            for t in range(T):
                for dstn in self.IncShkDstn[t]:
                    dstn.reset()

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
        IndShockConsumerType.sim_birth(
            self, which_agents
        )  # Get initial assets and permanent income
        if (
            not self.global_markov
        ):  # Markov state is not changed if it is set at the global level
            N = np.sum(which_agents)
            base_draws = Uniform(seed=self.RNG.integers(0, 2**31 - 1)).draw(N)
            Cutoffs = np.cumsum(np.array(self.MrkvPrbsInit))
            self.shocks["Mrkv"][which_agents] = np.searchsorted(
                Cutoffs, base_draws
            ).astype(int)

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
        dont_change = (
            self.t_age == 0
        )  # Don't change Markov state for those who were just born (unless global_markov)
        if self.t_sim == 0:  # Respect initial distribution of Markov states
            dont_change[:] = True

        # Determine which agents are in which states right now
        J = self.MrkvArray[0].shape[0]
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

    def get_Rfree(self):
        """
        Returns an array of size self.AgentCount with interest factor that varies with discrete state.

        Parameters
        ----------
        None

        Returns
        -------
        RfreeNow : np.array
             Array of size self.AgentCount with risk free interest rate for each agent.
        """
        RfreeNow = self.Rfree[self.shocks["Mrkv"]]
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

    def calc_bounding_values(self):
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



    def calc_transition_matrix(self, shk_dstn = None):
            '''
            Calculates how the distribution of agents across market resources 
            transitions from one period to the next. If finite horizon problem, then calculates
            a list of transition matrices, consumption and asset policy grids for each period of the problem. 
            The transition matrix/matrices and consumption and asset policy grid(s) are stored as attributes of self.
            
            
            Parameters
            ----------
                shk_dstn: list 
                    list of income shock distributions

            Returns
            -------
            None
            
            ''' 
            
            self.state_num = len(self.MrkvArray[0])

            if self.cycles == 0: 
            
                markov_array = self.MrkvArray
                
                eigen, ss_dstn = sp.linalg.eigs(markov_array[0].T , k=1, which='LM')
                
                
                ss_dstn = ss_dstn[:,0] / np.sum(ss_dstn[:,0]) # Steady state distribution of employed/unemployed 

                states = len(ss_dstn)

                
               
                if shk_dstn == None:
                    shk_dstn = self.IncShkDstn[0]
                
                dist_mGrid = self.dist_mGrid #Grid of market resources
                dist_pGrid = self.dist_pGrid #Grid of permanent incomes
                
                self.cPol_Grid = []
                self.aPol_Grid  = []
                
                bNext = []
                shk_prbs = []
                tran_shks = []
                perm_shks = []
                
                for i in range(states):
                    c_next = self.solution[0].cFunc[i](dist_mGrid)
                    self.cPol_Grid.append(c_next)
                    
                    a_next_i = dist_mGrid - c_next
                    self.aPol_Grid .append(a_next_i)
                    
                    if type(self.Rfree) == list:
                         b_next_i = self.Rfree[0][0] * a_next_i
                    else:
                         b_next_i = self.Rfree * a_next_i
                         
                    bNext.append(b_next_i)
                    
                    shk_prbs.append(shk_dstn[i].pmv)
                    tran_shks.append(shk_dstn[i].atoms[1])
                    perm_shks.append(shk_dstn[i].atoms[0])
                    
     

                LivPrb = self.LivPrb[0][0] # Update probability of staying alive
                
                
                if len(dist_pGrid) == 1: 
                    
                    self.tran_matrix = []

                    for i in range(self.state_num):
                        
                        TranMatrix_i =  np.zeros( (len(dist_mGrid),len(dist_mGrid)) ) 

                        for j in range(self.state_num):
                            NewBornDist = jump_to_grid_1D(tran_shks[j],shk_prbs[j],dist_mGrid)

                            TranMatrix_i +=  gen_tran_matrix_1D(dist_mGrid,bNext[i],self.MrkvArray[0][i][j]*shk_prbs[j],perm_shks[j],tran_shks[j], LivPrb,NewBornDist)
                        self.tran_matrix.append(TranMatrix_i)
                    
                    self.prb_dstn = ss_dstn.real
                    
                    
                else:
                    
                    self.tran_matrix = []

                    for i in range(self.state_num):
                        
                        TranMatrix_i =  np.zeros( (len(dist_mGrid),len(dist_mGrid)) ) 

                        for j in range(self.state_num):
                            NewBornDist = jump_to_grid_2D(tran_shks[j],np.ones_like(tran_shks[i]),shk_prbs[j],dist_mGrid,dist_pGrid)

                            TranMatrix_i +=  gen_tran_matrix_2D(dist_mGrid,dist_pGrid,bNext[i],self.MrkvArray[0][i][j]*shk_prbs[j],perm_shks[j],tran_shks[j], LivPrb,NewBornDist)
                        self.tran_matrix.append(TranMatrix_i)
                    
                    self.prb_dstn = ss_dstn.real
                    
        
            elif self.cycles > 1:
                print('calc_transition_matrix requires cycles = 0 or cycles = 1')
                
            elif self.T_cycle!= 0:
                
                # for finite horizon, we can account for changing levels of prb_unemp because of endogenous job finding probability by imposing a list of these values, so for q'th period, the probability is slightly different
                if shk_dstn == None:
                    shk_dstn = self.IncShkDstn
                
                self.tran_matrix = []

                dist_mGrid =  self.dist_mGrid
                #print(len(dist_mGrid))

                self.cPol_Grid = []
                self.aPol_Grid  = []
                self.prb_dstn = []
                dstn_0 = self.dstn_0 
                
                for k in range(self.T_cycle):
                
                    if type(self.dist_pGrid) == list:
                        dist_pGrid = self.dist_pGrid[k] #Permanent income grid this period
                    else:
                        dist_pGrid = self.dist_pGrid #If here then use prespecified permanent income grid
                    
                    bNext = []
                    shk_prbs = []
                    tran_shks = []
                    perm_shks = []
                    cPol_Grid_k = []
                    aPol_Grid_k = []

                    for i in range(self.state_num):
                        
                        c_next = self.solution[k].cFunc[i](dist_mGrid)
                        cPol_Grid_k.append(c_next)
                        
                        a_next_i = dist_mGrid - c_next
                        aPol_Grid_k.append(a_next_i)
                        
                        if type(self.Rfree)==list:
                            b_next_i = self.Rfree[k][0]*a_next_i
                            bNext.append(b_next_i)
                        else:
                            b_next_i = self.Rfree*a_next_i
                            bNext.append(b_next_i)
                            
                            
                        shk_prbs.append(shk_dstn[k][i].pmv)
                        tran_shks.append(shk_dstn[k][i].atoms[1])
                        perm_shks.append(shk_dstn[k][i].atoms[0])
                        
                    
                    self.cPol_Grid.append(cPol_Grid_k)
                    #print(len(cPol_Grid_k))

                    self.aPol_Grid.append(aPol_Grid_k)

            
                    LivPrb = self.LivPrb[k][0] # Update probability of staying alive this period
                    
         
                    if len(dist_pGrid) == 1: 
                        
                        dstn_0 = np.dot(self.MrkvArray[k].T, dstn_0) #transposed to have columns sum up to one , I think this is the distribution of employed vs unemployed
                        
                        tran_matrix_t = []

                        for i in range(self.state_num):
                            
                            TranMatrix_i =  np.zeros( (len(dist_mGrid),len(dist_mGrid)) ) 

                            for j in range(self.state_num):
                                
                                NewBornDist = jump_to_grid_1D(tran_shks[j],self.MrkvArray[k][i][j]*shk_prbs[j],dist_mGrid)

                                TranMatrix_i +=  gen_tran_matrix_1D(dist_mGrid,bNext[i],self.MrkvArray[k][i][j]*shk_prbs[j],perm_shks[j],tran_shks[j], LivPrb,NewBornDist)
                            tran_matrix_t.append(TranMatrix_i)
                        
                        self.tran_matrix.append(deepcopy(tran_matrix_t))
                   
                        self.prb_dstn.append(dstn_0)
                        
                  
                    else:
                        
                        
                        dstn_0 = np.dot(self.MrkvArray[k].T, dstn_0) #transposed to have columns sum up to one , I think this is the distribution of employed vs unemployed
                        
                        tran_matrix_t = []

                        for i in range(self.state_num):
                            
                            TranMatrix_i =  np.zeros( (len(dist_mGrid),len(dist_mGrid)) ) 

                            for j in range(self.state_num):
                                
                                NewBornDist = jump_to_grid_2D(tran_shks[j],np.ones_like(tran_shks[i]),self.MrkvArray[k][i][j]*shk_prbs[j],dist_mGrid,dist_pGrid)

                                TranMatrix_i +=  gen_tran_matrix_2D(dist_mGrid,dist_pGrid,bNext[i],self.MrkvArray[k][i][j]*shk_prbs[j],perm_shks[j],tran_shks[j], LivPrb,NewBornDist)
                            tran_matrix_t.append(TranMatrix_i)
                        
                        self.tran_matrix.append(deepcopy(tran_matrix_t))
                   
                        self.prb_dstn.append(dstn_0)
                        
                        
                    
                               
                    

        
    def calc_ergodic_dist(self, transition_matrix = None):
            
            '''
            Calculates the ergodic distribution across normalized market resources and
            permanent income as the eigenvector associated with the eigenvalue 1.
            The distribution is stored as attributes of self both as a vector and as a reshaped array with the ij'th element representing
            the probability of being at the i'th point on the mGrid and the j'th
            point on the pGrid.
            
            Parameters
            ----------
            transition_matrix: List 
                        transition matrix whose ergordic distribution is to be solved

            Returns
            -------
            None
            '''
            
            
            #if transition_matrix == None:
                #transition_matrix = [self.tran_matrix]
            self.vec_erg_dstns = []
            for i in range(self.state_num):
                
                eigen_i, ergodic_distr_i = sp.linalg.eigs(self.tran_matrix[i] , k=1 , which='LM')  # Solve for ergodic distribution
                ergodic_distr_i = ergodic_distr_i.real/np.sum(ergodic_distr_i.real)
                
                #erg_dstn = compute_erg_dstn(self.tran_matrix[i])
                #self.ergodic_distrs.append(erg_dstn)

                ergodic_distr_i = ergodic_distr_i.T[0]
                self.vec_erg_dstns.append(ergodic_distr_i)
                
                
            
    def compute_steady_state(self,IncShkDstn_ntrl_msr):
        
           
            

        #solve the consumer's problem

        self.solve()

        self.define_distribution_grid(dist_pGrid = np.array([1]))
        self.calc_transition_matrix(IncShkDstn_ntrl_msr)
        self.calc_ergodic_dist()
        
        
        C = 0
        A = 0
        for i in range(self.state_num):
            
            C += self.prb_dstn[i]*np.dot(self.cPol_Grid[i],self.vec_erg_dstns[i])
            A += self.prb_dstn[i]*np.dot(self.aPol_Grid[i],self.vec_erg_dstns[i])


        self.A_ss = A
        self.C_ss = C
        
        return C , A 
    
    
    def make_euler_error_func(self, mMax=100, approx_inc_dstn=True):
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

        Notes
        -----
        This method is not used by any other code in the library. Rather, it is here
        for expository and benchmarking purposes.
        """
        raise NotImplementedError()
