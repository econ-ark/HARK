"""
Classes to solve and simulate consumption-savings model with a discrete, exogenous,
stochastic Markov state.  The only solver here extends ConsIndShockModel to
include a Markov state; the interest factor, permanent growth factor, and income
distribution can vary with the discrete state.
"""
from copy import deepcopy

import numpy as np
from HARK import AgentType
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsIndShockSolver,
    ConsumerSolution,
    IndShockConsumerType,
    PerfForesightConsumerType,
)
from HARK.distribution import (
    DiscreteDistribution,
    MarkovProcess,
    Uniform,
    calc_expectation,
)
from HARK.interpolation import (
    CubicInterp,
    LinearInterp,
    LowerEnvelope,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.utilities import (
    CRRAutility,
    CRRAutility_inv,
    CRRAutility_invP,
    CRRAutilityP,
    CRRAutilityP_inv,
    CRRAutilityP_invP,
    CRRAutilityPP,
)

__all__ = ["ConsMarkovSolver", "MarkovConsumerType"]

utility = CRRAutility
utilityP = CRRAutilityP
utilityPP = CRRAutilityPP
utilityP_inv = CRRAutilityP_inv
utility_invP = CRRAutility_invP
utility_inv = CRRAutility_inv
utilityP_invP = CRRAutilityP_invP


class ConsMarkovSolver(ConsIndShockSolver):
    """
    A class to solve a single period consumption-saving problem with risky income
    and stochastic transitions between discrete states, in a Markov fashion.
    Extends ConsIndShockSolver, with identical inputs but for a discrete
    Markov state, whose transition rule is summarized in MrkvArray.  Markov
    states can differ in their interest factor, permanent growth factor, live probability, and
    income distribution, so the inputs Rfree, PermGroFac, IncShkDstn, and LivPrb are
    now arrays or lists specifying those values in each (succeeding) Markov state.
    """

    def __init__(
        self,
        solution_next,
        IncShkDstn_list,
        LivPrb,
        DiscFac,
        CRRA,
        Rfree_list,
        PermGroFac_list,
        MrkvArray,
        BoroCnstArt,
        aXtraGrid,
        vFuncBool,
        CubicBool,
    ):
        """
        Constructor for a new solver for a one period problem with risky income
        and transitions between discrete Markov states.  In the descriptions below,
        N is the number of discrete states.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncShkDstn_list : [distribution.Distribution]
            A length N list of income distributions in each succeeding Markov
            state.  Each income distribution is a
            discrete approximation to the income process at the
            beginning of the succeeding period.
        LivPrb : np.array
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period for each Markov state.
        DiscFac : float
            Intertemporal discount factor for future utility.
        CRRA : float
            Coefficient of relative risk aversion.
        Rfree_list : np.array
            Risk free interest factor on end-of-period assets for each Markov
            state in the succeeding period.
        PermGroFac_list : np.array
            Expected permanent income growth factor at the end of this period
            for each Markov state in the succeeding period.
        MrkvArray : np.array
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
        None
        """
        # Set basic attributes of the problem

        self.solution_next = solution_next
        self.IncShkDstn_list = IncShkDstn_list
        self.LivPrb = LivPrb
        self.DiscFac = DiscFac
        self.CRRA = CRRA
        self.BoroCnstArt = BoroCnstArt
        self.aXtraGrid = aXtraGrid
        self.vFuncBool = vFuncBool
        self.CubicBool = CubicBool
        self.Rfree_list = Rfree_list
        self.PermGroFac_list = PermGroFac_list
        self.MrkvArray = MrkvArray
        self.StateCount = MrkvArray.shape[0]

        self.def_utility_funcs()

    def solve(self):
        """
        Solve the one period problem of the consumption-saving model with a Markov state.

        Parameters
        ----------
        none

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
        # Find the natural borrowing constraint in each current state
        self.def_boundary()

        # Initialize end-of-period (marginal) value functions
        self.EndOfPrdvFunc_list = []
        self.EndOfPrdvPfunc_list = []
        self.Ex_IncNextAll = (
            np.zeros(self.StateCount) + np.nan
        )  # expected income conditional on the next state
        self.WorstIncPrbAll = (
            np.zeros(self.StateCount) + np.nan
        )  # probability of getting the worst income shock in each next period state

        # Loop through each next-period-state and calculate the end-of-period
        # (marginal) value function
        for j in range(self.StateCount):
            # Condition values on next period's state (and record a couple for later use)
            self.condition_on_state(j)
            self.Ex_IncNextAll[j] = np.dot(
                self.ShkPrbsNext, self.PermShkValsNext * self.TranShkValsNext
            )
            self.WorstIncPrbAll[j] = self.WorstIncPrb

            # Construct the end-of-period marginal value function conditional
            # on next period's state and add it to the list of value functions
            EndOfPrdvPfunc_cond = self.make_EndOfPrdvPfuncCond()
            self.EndOfPrdvPfunc_list.append(EndOfPrdvPfunc_cond)

            # Construct the end-of-period value functional conditional on next
            # period's state and add it to the list of value functions
            if self.vFuncBool:
                EndOfPrdvFunc_cond = self.make_EndOfPrdvFuncCond()
                self.EndOfPrdvFunc_list.append(EndOfPrdvFunc_cond)

        # EndOfPrdvP_cond is EndOfPrdvP conditional on *next* period's state.
        # Take expectations to get EndOfPrdvP conditional on *this* period's state.
        self.calc_EndOfPrdvP()

        # Calculate the bounding MPCs and PDV of human wealth for each state
        self.calc_HumWealth_and_BoundingMPCs()

        # Find consumption and market resources corresponding to each end-of-period
        # assets point for each state (and add an additional point at the lower bound)
        aNrm = (
            np.asarray(self.aXtraGrid)[np.newaxis, :]
            + np.array(self.BoroCnstNat_list)[:, np.newaxis]
        )
        self.get_points_for_interpolation(self.EndOfPrdvP, aNrm)
        cNrm = np.hstack((np.zeros((self.StateCount, 1)), self.cNrmNow))
        mNrm = np.hstack(
            (np.reshape(self.mNrmMin_list, (self.StateCount, 1)), self.mNrmNow)
        )

        # Package and return the solution for this period
        self.BoroCnstNat = self.BoroCnstNat_list
        solution = self.make_solution(cNrm, mNrm)
        return solution

    def def_boundary(self):
        """
        Find the borrowing constraint for each current state and save it as an
        attribute of self for use by other methods.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        self.BoroCnstNatAll = np.zeros(self.StateCount) + np.nan
        # Find the natural borrowing constraint conditional on next period's state
        for j in range(self.StateCount):
            PermShkMinNext = np.min(self.IncShkDstn_list[j].atoms[0])
            TranShkMinNext = np.min(self.IncShkDstn_list[j].atoms[1])
            self.BoroCnstNatAll[j] = (
                (self.solution_next.mNrmMin[j] - TranShkMinNext)
                * (self.PermGroFac_list[j] * PermShkMinNext)
                / self.Rfree_list[j]
            )

        self.BoroCnstNat_list = np.zeros(self.StateCount) + np.nan
        self.mNrmMin_list = np.zeros(self.StateCount) + np.nan
        self.BoroCnstDependency = np.zeros((self.StateCount, self.StateCount)) + np.nan
        # The natural borrowing constraint in each current state is the *highest*
        # among next-state-conditional natural borrowing constraints that could
        # occur from this current state.
        for i in range(self.StateCount):
            possible_next_states = self.MrkvArray[i, :] > 0
            self.BoroCnstNat_list[i] = np.max(self.BoroCnstNatAll[possible_next_states])

            # Explicitly handle the "None" case:
            if self.BoroCnstArt is None:
                self.mNrmMin_list[i] = self.BoroCnstNat_list[i]
            else:
                self.mNrmMin_list[i] = np.max(
                    [self.BoroCnstNat_list[i], self.BoroCnstArt]
                )
            self.BoroCnstDependency[i, :] = (
                self.BoroCnstNat_list[i] == self.BoroCnstNatAll
            )
        # Also creates a Boolean array indicating whether the natural borrowing
        # constraint *could* be hit when transitioning from i to j.

    def condition_on_state(self, state_index):
        """
        Temporarily assume that a particular Markov state will occur in the
        succeeding period, and condition solver attributes on this assumption.
        Allows the solver to construct the future-state-conditional marginal
        value function (etc) for that future state.

        Parameters
        ----------
        state_index : int
            Index of the future Markov state to condition on.

        Returns
        -------
        none
        """
        # Set future-state-conditional values as attributes of self
        self.IncShkDstn = self.IncShkDstn_list[state_index]
        self.Rfree = self.Rfree_list[state_index]
        self.PermGroFac = self.PermGroFac_list[state_index]
        self.vPfuncNext = self.solution_next.vPfunc[state_index]
        self.mNrmMinNow = self.mNrmMin_list[state_index]
        self.BoroCnstNat = self.BoroCnstNatAll[state_index]
        self.set_and_update_values(
            self.solution_next, self.IncShkDstn, self.LivPrb, self.DiscFac
        )
        self.DiscFacEff = (
            self.DiscFac
        )  # survival probability LivPrb represents probability from
        # *current* state, so DiscFacEff is just DiscFac for now

        # These lines have to come after set_and_update_values to override the definitions there
        self.vPfuncNext = self.solution_next.vPfunc[state_index]
        if self.CubicBool:
            self.vPPfuncNext = self.solution_next.vPPfunc[state_index]
        if self.vFuncBool:
            self.vFuncNext = self.solution_next.vFunc[state_index]

    def calc_EndOfPrdvPP(self):
        """
        Calculates end-of-period marginal marginal value using a pre-defined
        array of next period market resources in self.mNrmNext.

        Parameters
        ----------
        none

        Returns
        -------
        EndOfPrdvPP : np.array
            End-of-period marginal marginal value of assets at each value in
            the grid of assets.
        """

        def vpp_next(shocks, a_nrm, Rfree):
            return shocks[0] ** (-self.CRRA - 1.0) * self.vPPfuncNext(
                self.m_nrm_next(shocks, a_nrm, Rfree)
            )

        EndOfPrdvPP = (
            self.DiscFacEff
            * self.Rfree
            * self.Rfree
            * self.PermGroFac ** (-self.CRRA - 1.0)
            * calc_expectation(self.IncShkDstn, vpp_next, self.aNrmNow, self.Rfree)
        )
        return EndOfPrdvPP

    def make_EndOfPrdvFuncCond(self):
        """
        Construct the end-of-period value function conditional on next period's
        state.

        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal value of assets corresponding to the
            asset values in self.aNrmNow.
        Returns
        -------
        none
        """

        def v_lvl_next(shocks, a_nrm, Rfree):
            return (
                shocks[0] ** (1.0 - self.CRRA) * self.PermGroFac ** (1.0 - self.CRRA)
            ) * self.vFuncNext(self.m_nrm_next(shocks, a_nrm, Rfree))

        EndOfPrdv_cond = self.DiscFacEff * calc_expectation(
            self.IncShkDstn, v_lvl_next, self.aNrmNow, self.Rfree
        )
        EndOfPrdvNvrs = self.u.inv(
            EndOfPrdv_cond
        )  # value transformed through inverse utility
        EndOfPrdvNvrsP = self.EndOfPrdvP_cond * self.u.derinv(
            EndOfPrdv_cond, order=(0, 1)
        )
        EndOfPrdvNvrs = np.insert(EndOfPrdvNvrs, 0, 0.0)
        EndOfPrdvNvrsP = np.insert(
            EndOfPrdvNvrsP, 0, EndOfPrdvNvrsP[0]
        )  # This is a very good approximation, vNvrsPP = 0 at the asset minimum
        aNrm_temp = np.insert(self.aNrmNow, 0, self.BoroCnstNat)
        EndOfPrdvNvrsFunc = CubicInterp(aNrm_temp, EndOfPrdvNvrs, EndOfPrdvNvrsP)
        EndOfPrdvFunc_dond = ValueFuncCRRA(EndOfPrdvNvrsFunc, self.CRRA)

        return EndOfPrdvFunc_dond

    def calc_EndOfPrdvPcond(self):
        """
        Calculate end-of-period marginal value of assets at each point in aNrmNow
        conditional on a particular state occuring in the next period.

        Parameters
        ----------
        None

        Returns
        -------
        EndOfPrdvP : np.array
            A 1D array of end-of-period marginal value of assets.
        """
        EndOfPrdvPcond = ConsIndShockSolver.calc_EndOfPrdvP(self)
        return EndOfPrdvPcond

    def make_EndOfPrdvPfuncCond(self):
        """
        Construct the end-of-period marginal value function conditional on next
        period's state.

        Parameters
        ----------
        None

        Returns
        -------
        EndofPrdvPfunc_cond : MargValueFuncCRRA
            The end-of-period marginal value function conditional on a particular
            state occuring in the succeeding period.
        """
        # Get data to construct the end-of-period marginal value function (conditional on next state)
        self.aNrm_cond = self.prepare_to_calc_EndOfPrdvP()
        self.EndOfPrdvP_cond = self.calc_EndOfPrdvPcond()
        EndOfPrdvPnvrs_cond = self.u.derinv(
            self.EndOfPrdvP_cond, order=(1, 0)
        )  # "decurved" marginal value
        if self.CubicBool:
            EndOfPrdvPP_cond = self.calc_EndOfPrdvPP()
            EndOfPrdvPnvrsP_cond = EndOfPrdvPP_cond * self.u.derinv(
                self.EndOfPrdvP_cond, order=(1, 1)
            )  # "decurved" marginal marginal value

        # Construct the end-of-period marginal value function conditional on the next state.
        if self.CubicBool:
            EndOfPrdvPnvrsFunc_cond = CubicInterp(
                self.aNrm_cond,
                EndOfPrdvPnvrs_cond,
                EndOfPrdvPnvrsP_cond,
                lower_extrap=True,
            )
        else:
            EndOfPrdvPnvrsFunc_cond = LinearInterp(
                self.aNrm_cond, EndOfPrdvPnvrs_cond, lower_extrap=True
            )
        EndofPrdvPfunc_cond = MargValueFuncCRRA(
            EndOfPrdvPnvrsFunc_cond, self.CRRA
        )  # "recurve" the interpolated marginal value function
        return EndofPrdvPfunc_cond

    def calc_EndOfPrdvP(self):
        """
        Calculates end of period marginal value (and marginal marginal) value
        at each aXtra gridpoint for each current state, unconditional on the
        future Markov state (i.e. weighting conditional end-of-period marginal
        value by transition probabilities).

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        # Find unique values of minimum acceptable end-of-period assets (and the
        # current period states for which they apply).
        aNrmMin_unique, state_inverse = np.unique(
            self.BoroCnstNat_list, return_inverse=True
        )
        self.possible_transitions = self.MrkvArray > 0

        # Calculate end-of-period marginal value (and marg marg value) at each
        # asset gridpoint for each current period state
        EndOfPrdvP = np.zeros((self.StateCount, self.aXtraGrid.size))
        EndOfPrdvPP = np.zeros((self.StateCount, self.aXtraGrid.size))
        for k in range(aNrmMin_unique.size):
            aNrmMin = aNrmMin_unique[k]  # minimum assets for this pass
            which_states = (
                state_inverse == k
            )  # the states for which this minimum applies
            aGrid = aNrmMin + self.aXtraGrid  # assets grid for this pass
            EndOfPrdvP_all = np.zeros((self.StateCount, self.aXtraGrid.size))
            EndOfPrdvPP_all = np.zeros((self.StateCount, self.aXtraGrid.size))
            for j in range(self.StateCount):
                if np.any(
                    np.logical_and(self.possible_transitions[:, j], which_states)
                ):  # only consider a future state if one of the relevant states could transition to it
                    EndOfPrdvP_all[j, :] = self.EndOfPrdvPfunc_list[j](aGrid)
                    if (
                        self.CubicBool
                    ):  # Add conditional end-of-period (marginal) marginal value to the arrays
                        EndOfPrdvPP_all[j, :] = self.EndOfPrdvPfunc_list[j].derivativeX(
                            aGrid
                        )
            # Weight conditional marginal (marginal) values by transition probs
            # to get unconditional marginal (marginal) value at each gridpoint.
            EndOfPrdvP_temp = np.dot(self.MrkvArray, EndOfPrdvP_all)
            EndOfPrdvP[which_states, :] = EndOfPrdvP_temp[
                which_states, :
            ]  # only take the states for which this asset minimum applies
            if self.CubicBool:
                EndOfPrdvPP_temp = np.dot(self.MrkvArray, EndOfPrdvPP_all)
                EndOfPrdvPP[which_states, :] = EndOfPrdvPP_temp[which_states, :]

        # Store the results as attributes of self, scaling end of period marginal value by survival probability from each current state
        LivPrb_tiled = np.tile(
            np.reshape(self.LivPrb, (self.StateCount, 1)), (1, self.aXtraGrid.size)
        )
        self.EndOfPrdvP = LivPrb_tiled * EndOfPrdvP
        if self.CubicBool:
            self.EndOfPrdvPP = LivPrb_tiled * EndOfPrdvPP

    def calc_HumWealth_and_BoundingMPCs(self):
        """
        Calculates human wealth and the maximum and minimum MPC for each current
        period state, then stores them as attributes of self for use by other methods.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        # Upper bound on MPC at lower m-bound
        WorstIncPrb_array = self.BoroCnstDependency * np.tile(
            np.reshape(self.WorstIncPrbAll, (1, self.StateCount)), (self.StateCount, 1)
        )
        temp_array = self.MrkvArray * WorstIncPrb_array
        WorstIncPrbNow = np.sum(
            temp_array, axis=1
        )  # Probability of getting the "worst" income shock and transition from each current state
        ExMPCmaxNext = (
            np.dot(
                temp_array,
                self.Rfree_list ** (1.0 - self.CRRA)
                * self.solution_next.MPCmax ** (-self.CRRA),
            )
            / WorstIncPrbNow
        ) ** (-1.0 / self.CRRA)
        DiscFacEff_temp = self.DiscFac * self.LivPrb
        self.MPCmaxNow = 1.0 / (
            1.0
            + ((DiscFacEff_temp * WorstIncPrbNow) ** (1.0 / self.CRRA)) / ExMPCmaxNext
        )
        self.MPCmaxEff = self.MPCmaxNow
        self.MPCmaxEff[self.BoroCnstNat_list < self.mNrmMin_list] = 1.0
        # State-conditional PDV of human wealth
        hNrmPlusIncNext = self.Ex_IncNextAll + self.solution_next.hNrm
        self.hNrmNow = np.dot(
            self.MrkvArray, (self.PermGroFac_list / self.Rfree_list) * hNrmPlusIncNext
        )
        # Lower bound on MPC as m gets arbitrarily large
        temp = (
            DiscFacEff_temp
            * np.dot(
                self.MrkvArray,
                self.solution_next.MPCmin ** (-self.CRRA)
                * self.Rfree_list ** (1.0 - self.CRRA),
            )
        ) ** (1.0 / self.CRRA)
        self.MPCminNow = 1.0 / (1.0 + temp)

    def make_solution(self, cNrm, mNrm):
        """
        Construct an object representing the solution to this period's problem.

        Parameters
        ----------
        cNrm : np.array
            Array of normalized consumption values for interpolation.  Each row
            corresponds to a Markov state for this period.
        mNrm : np.array
            Array of normalized market resource values for interpolation.  Each
            row corresponds to a Markov state for this period.

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
        solution = (
            ConsumerSolution()
        )  # An empty solution to which we'll add state-conditional solutions
        # Calculate the MPC at each market resource gridpoint in each state (if desired)
        if self.CubicBool:
            dcda = self.EndOfPrdvPP / self.u.der(np.array(self.cNrmNow), order=2)
            MPC = dcda / (dcda + 1.0)
            self.MPC_temp = np.hstack(
                (np.reshape(self.MPCmaxNow, (self.StateCount, 1)), MPC)
            )
            interpfunc = self.make_cubic_cFunc
        else:
            interpfunc = self.make_linear_cFunc

        # Loop through each current period state and add its solution to the overall solution
        for i in range(self.StateCount):
            # Set current-period-conditional human wealth and MPC bounds
            self.hNrmNow_j = self.hNrmNow[i]
            self.MPCminNow_j = self.MPCminNow[i]
            if self.CubicBool:
                self.MPC_temp_j = self.MPC_temp[i, :]

            # Construct the consumption function by combining the constrained and unconstrained portions
            self.cFuncNowCnst = LinearInterp(
                [self.mNrmMin_list[i], self.mNrmMin_list[i] + 1.0], [0.0, 1.0]
            )
            cFuncNowUnc = interpfunc(mNrm[i, :], cNrm[i, :])
            cFuncNow = LowerEnvelope(cFuncNowUnc, self.cFuncNowCnst)

            # Make the marginal value function and pack up the current-state-conditional solution
            vPfuncNow = MargValueFuncCRRA(cFuncNow, self.CRRA)
            solution_cond = ConsumerSolution(
                cFunc=cFuncNow, vPfunc=vPfuncNow, mNrmMin=self.mNrmMinNow
            )
            if (
                self.CubicBool
            ):  # Add the state-conditional marginal marginal value function (if desired)
                solution_cond = self.add_vPPfunc(solution_cond)

            # Add the current-state-conditional solution to the overall period solution
            solution.append_solution(solution_cond)

        # Add the lower bounds of market resources, MPC limits, human resources,
        # and the value functions to the overall solution
        solution.mNrmMin = self.mNrmMin_list
        solution = self.add_MPC_and_human_wealth(solution)
        if self.vFuncBool:
            vFuncNow = self.make_vFunc(solution)
            solution.vFunc = vFuncNow

        # Return the overall solution to this period
        return solution

    def make_linear_cFunc(self, mNrm, cNrm):
        """
        Make a linear interpolation to represent the (unconstrained) consumption
        function conditional on the current period state.

        Parameters
        ----------
        mNrm : np.array
            Array of normalized market resource values for interpolation.
        cNrm : np.array
            Array of normalized consumption values for interpolation.

        Returns
        -------
        cFuncUnc: an instance of HARK.interpolation.LinearInterp
        """
        cFuncUnc = LinearInterp(
            mNrm, cNrm, self.MPCminNow_j * self.hNrmNow_j, self.MPCminNow_j
        )
        return cFuncUnc

    def make_cubic_cFunc(self, mNrm, cNrm):
        """
        Make a cubic interpolation to represent the (unconstrained) consumption
        function conditional on the current period state.

        Parameters
        ----------
        mNrm : np.array
            Array of normalized market resource values for interpolation.
        cNrm : np.array
            Array of normalized consumption values for interpolation.

        Returns
        -------
        cFuncUnc: an instance of HARK.interpolation.CubicInterp
        """
        cFuncUnc = CubicInterp(
            mNrm,
            cNrm,
            self.MPC_temp_j,
            self.MPCminNow_j * self.hNrmNow_j,
            self.MPCminNow_j,
        )
        return cFuncUnc

    def make_vFunc(self, solution):
        """
        Construct the value function for each current state.

        Parameters
        ----------
        solution : ConsumerSolution
            The solution to the single period consumption-saving problem. Must
            have a consumption function cFunc (using cubic or linear splines) as
            a list with elements corresponding to the current Markov state.  E.g.
            solution.cFunc[0] is the consumption function when in the i=0 Markov
            state this period.

        Returns
        -------
        vFuncNow : [ValueFuncCRRA]
            A list of value functions (defined over normalized market resources
            m) for each current period Markov state.
        """
        vFuncNow = []  # Initialize an empty list of value functions
        # Loop over each current period state and construct the value function
        for i in range(self.StateCount):
            # Make state-conditional grids of market resources and consumption
            mNrmMin = self.mNrmMin_list[i]
            mGrid = mNrmMin + self.aXtraGrid
            cGrid = solution.cFunc[i](mGrid)
            aGrid = mGrid - cGrid

            # Calculate end-of-period value at each gridpoint
            EndOfPrdv_all = np.zeros((self.StateCount, self.aXtraGrid.size))
            for j in range(self.StateCount):
                if self.possible_transitions[i, j]:
                    EndOfPrdv_all[j, :] = self.EndOfPrdvFunc_list[j](aGrid)
            EndOfPrdv = np.dot(self.MrkvArray[i, :], EndOfPrdv_all)

            # Calculate (normalized) value and marginal value at each gridpoint
            vNrmNow = self.u(cGrid) + EndOfPrdv
            vPnow = self.u.der(cGrid)

            # Make a "decurved" value function with the inverse utility function
            vNvrs = self.u.inv(vNrmNow)  # value transformed through inverse utility
            vNvrsP = vPnow * self.u.derinv(vNrmNow, order=(0, 1))
            mNrm_temp = np.insert(mGrid, 0, mNrmMin)  # add the lower bound
            vNvrs = np.insert(vNvrs, 0, 0.0)
            vNvrsP = np.insert(
                vNvrsP, 0, self.MPCmaxEff[i] ** (-self.CRRA / (1.0 - self.CRRA))
            )
            MPCminNvrs = self.MPCminNow[i] ** (-self.CRRA / (1.0 - self.CRRA))
            vNvrsFunc_i = CubicInterp(
                mNrm_temp, vNvrs, vNvrsP, MPCminNvrs * self.hNrmNow[i], MPCminNvrs
            )

            # "Recurve" the decurved value function and add it to the list
            vFunc_i = ValueFuncCRRA(vNvrsFunc_i, self.CRRA)
            vFuncNow.append(vFunc_i)
        return vFuncNow


def _solve_ConsMarkov(
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
    identical inputs as solveConsIndShock, except for a discrete
    Markov transitionrule MrkvArray.  Markov states can differ in their interest
    factor, permanent growth factor, and income distribution, so the inputs Rfree,
    PermGroFac, and IncShkDstn are arrays or lists specifying those values in each
    (succeeding) Markov state.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncShkDstn_list : [distribution.Distribution]
        A length N list of income distributions in each succeeding Markov
        state.  Each income distribution is
        a discrete approximation to the income process at the
        beginning of the succeeding period.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree_list : np.array
        Risk free interest factor on end-of-period assets for each Markov
        state in the succeeding period.
    PermGroGac_list : float
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
    solver = ConsMarkovSolver(
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
    )
    solution_now = solver.solve()
    return solution_now


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
        self.solve_one_period = _solve_ConsMarkov

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
            base_draw = Uniform(seed=self.RNG.randint(0, 2**31 - 1)).draw(1)
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
        DeathShks = Uniform(seed=self.RNG.randint(0, 2**31 - 1)).draw(
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
            base_draws = Uniform(seed=self.RNG.randint(0, 2**31 - 1)).draw(N)
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
                self.MrkvArray[t], seed=self.RNG.randint(0, 2**31 - 1)
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
