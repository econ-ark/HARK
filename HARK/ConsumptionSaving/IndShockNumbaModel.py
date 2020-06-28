from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from interpolation import interp
from numba import njit

from HARK import makeOnePeriodOOSolver, HARKobject
from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    MargValueFunc,
    ConsumerSolution,
    ConsPerfForesightSolver,
)
from HARK.interpolation import LowerEnvelope, LinearInterp
from HARK.utilities import (
    CRRAutility,
    CRRAutilityP,
    CRRAutilityPP,
    CRRAutilityP_inv,
    CRRAutility_invP,
    CRRAutility_inv,
    CRRAutilityP_invP,
)

utility = CRRAutility
utilityP = CRRAutilityP
utilityPP = CRRAutilityPP
utilityP_inv = CRRAutilityP_inv
utility_invP = CRRAutility_invP
utility_inv = CRRAutility_inv
utilityP_invP = CRRAutilityP_invP


@njit
def tile(A, reps):
    return A.repeat(reps[0]).reshape(A.size, -1).transpose()


@njit
def insert(arr, obj, values, axis=-1):
    return np.append(np.array(values), arr)


@njit
def solveConsIndShockNumba(
    DiscFacEff,
    CRRA,
    Rfree,
    PermGroFac,
    BoroCnstNat,
    aXtraGrid,
    TranShkValsNext,
    PermShkValsNext,
    ShkPrbsNext,
    sn_cFunc_x_list,
    sn_cFunc_y_list,
):
    """
    Prepare to calculate end-of-period marginal value by creating an array
    of market resources that the agent could have next period, considering
    the grid of end-of-period assets and the distribution of shocks he might
    experience next period.
    """

    # We define aNrmNow all the way from BoroCnstNat up to max(self.aXtraGrid)
    # even if BoroCnstNat < BoroCnstArt, so we can construct the consumption
    # function as the lower envelope of the (by the artificial borrowing con-
    # straint) uconstrained consumption function, and the artificially con-
    # strained consumption function.
    aNrm = np.asarray(aXtraGrid) + BoroCnstNat
    ShkCount = TranShkValsNext.size
    aNrm_temp = tile(aNrm, (ShkCount, 1))

    # Tile arrays of the income shocks and put them into useful shapes
    aNrmCount = aNrm.shape[0]
    PermShkVals_temp = (tile(PermShkValsNext, (aNrmCount, 1))).transpose()
    TranShkVals_temp = (tile(TranShkValsNext, (aNrmCount, 1))).transpose()
    ShkPrbs_temp = (tile(ShkPrbsNext, (aNrmCount, 1))).transpose()

    # Get cash on hand next period
    mNrmNext = Rfree / (PermGroFac * PermShkVals_temp) * aNrm_temp + TranShkVals_temp
    # CDC 20191205: This should be divided by LivPrb[0] for Blanchard insurance

    """
    Calculate end-of-period marginal value of assets at each point in aNrmNow.
    Does so by taking a weighted sum of next period marginal values across
    income shocks (in a preconstructed grid self.mNrmNext).
    """

    cFuncNext = interp(sn_cFunc_x_list, sn_cFunc_y_list, mNrmNext.flatten())
    vPfuncNext = (cFuncNext ** -CRRA).reshape(mNrmNext.shape)

    EndOfPrdvP = (
        DiscFacEff
        * Rfree
        * PermGroFac ** (-CRRA)
        * np.sum(PermShkVals_temp ** (-CRRA) * vPfuncNext * ShkPrbs_temp, axis=0)
    )

    """
    Finds interpolation points (c,m) for the consumption function.
    """

    cNrmNow = EndOfPrdvP ** (-1.0 / CRRA)
    mNrmNow = cNrmNow + aNrm

    # Limiting consumption is zero as m approaches mNrmMin
    cNrm = insert(cNrmNow, 0, 0.0, axis=-1)
    mNrm = insert(mNrmNow, 0, BoroCnstNat, axis=-1)

    return cNrm, mNrm


class ConsIndShockNumbaSolverBasic(HARKobject):
    """
    This class solves a single period of a standard consumption-saving problem,
    using linear interpolation and without the ability to calculate the value
    function.  ConsIndShockSolver inherits from this class and adds the ability
    to perform cubic interpolation and to calculate the value function.

    Note that this class does not have its own initializing method.  It initial-
    izes the same problem in the same way as ConsIndShockSetup, from which it
    inherits.
    """

    def __init__(
        self,
        solution_next,
        IncomeDstn,
        LivPrb,
        DiscFac,
        CRRA,
        Rfree,
        PermGroFac,
        BoroCnstArt,
        aXtraGrid,
        vFuncBool,
        CubicBool,
    ):
        """
        Constructor for a new solver-setup for problems with income subject to
        permanent and transitory shocks.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncomeDstn : [np.array]
            A list containing three arrays of floats, representing a discrete
            approximation to the income process between the period being solved
            and the one immediately following (in solution_next). Order: event
            probabilities, permanent shocks, transitory shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.
        DiscFac : float
            Intertemporal discount factor for future utility.
        CRRA : float
            Coefficient of relative risk aversion.
        Rfree : float
            Risk free interest factor on end-of-period assets.
        PermGroFac : float
            Expected permanent income growth factor at the end of this period.
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
        self.assignParameters(
            solution_next,
            IncomeDstn,
            LivPrb,
            DiscFac,
            CRRA,
            Rfree,
            PermGroFac,
            BoroCnstArt,
            aXtraGrid,
            vFuncBool,
            CubicBool,
        )
        self.defUtilityFuncs()

    def assignParameters(
        self,
        solution_next,
        IncomeDstn,
        LivPrb,
        DiscFac,
        CRRA,
        Rfree,
        PermGroFac,
        BoroCnstArt,
        aXtraGrid,
        vFuncBool,
        CubicBool,
    ):
        """
        Assigns period parameters as attributes of self for use by other methods

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncomeDstn : [np.array]
            A list containing three arrays of floats, representing a discrete
            approximation to the income process between the period being solved
            and the one immediately following (in solution_next). Order: event
            probabilities, permanent shocks, transitory shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.
        DiscFac : float
            Intertemporal discount factor for future utility.
        CRRA : float
            Coefficient of relative risk aversion.
        Rfree : float
            Risk free interest factor on end-of-period assets.
        PermGroFac : float
            Expected permanent income growth factor at the end of this period.
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
        none
        """
        ConsPerfForesightSolver.assignParameters(
            self,
            solution_next,
            DiscFac,
            LivPrb,
            CRRA,
            Rfree,
            PermGroFac,
            BoroCnstArt,
            None,
        )
        self.aXtraGrid = aXtraGrid
        self.IncomeDstn = IncomeDstn
        self.vFuncBool = vFuncBool
        self.CubicBool = CubicBool

    def defUtilityFuncs(self):
        """
        Defines CRRA utility function for this period (and its derivatives,
        and their inverses), saving them as attributes of self for other methods
        to use.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        ConsPerfForesightSolver.defUtilityFuncs(self)
        self.uPinv = lambda u: utilityP_inv(u, gam=self.CRRA)
        self.uPinvP = lambda u: utilityP_invP(u, gam=self.CRRA)
        self.uinvP = lambda u: utility_invP(u, gam=self.CRRA)
        if self.vFuncBool:
            self.uinv = lambda u: utility_inv(u, gam=self.CRRA)

    def setAndUpdateValues(self, solution_next, IncomeDstn, LivPrb, DiscFac):
        """
        Unpacks some of the inputs (and calculates simple objects based on them),
        storing the results in self for use by other methods.  These include:
        income shocks and probabilities, next period's marginal value function
        (etc), the probability of getting the worst income shock next period,
        the patience factor, human wealth, and the bounding MPCs.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncomeDstn : distribution.DiscreteDistribution
            A DiscreteDistribution with a pmf
            and two point value arrays in X, order:
            permanent shocks, transitory shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.
        DiscFac : float
            Intertemporal discount factor for future utility.

        Returns
        -------
        None
        """
        self.DiscFacEff = DiscFac * LivPrb  # "effective" discount factor
        self.ShkPrbsNext = IncomeDstn.pmf
        self.PermShkValsNext = IncomeDstn.X[0]
        self.TranShkValsNext = IncomeDstn.X[1]
        self.PermShkMinNext = np.min(self.PermShkValsNext)
        self.TranShkMinNext = np.min(self.TranShkValsNext)
        self.vPfuncNext = solution_next.vPfunc
        self.WorstIncPrb = np.sum(
            self.ShkPrbsNext[
                (self.PermShkValsNext * self.TranShkValsNext)
                == (self.PermShkMinNext * self.TranShkMinNext)
            ]
        )

        if self.CubicBool:
            self.vPPfuncNext = solution_next.vPPfunc

        if self.vFuncBool:
            self.vFuncNext = solution_next.vFunc

        # Update the bounding MPCs and PDV of human wealth:
        self.PatFac = ((self.Rfree * self.DiscFacEff) ** (1.0 / self.CRRA)) / self.Rfree
        self.MPCminNow = 1.0 / (1.0 + self.PatFac / solution_next.MPCmin)
        self.ExIncNext = np.dot(
            self.ShkPrbsNext, self.TranShkValsNext * self.PermShkValsNext
        )
        self.hNrmNow = (
            self.PermGroFac / self.Rfree * (self.ExIncNext + solution_next.hNrm)
        )
        self.MPCmaxNow = 1.0 / (
            1.0
            + (self.WorstIncPrb ** (1.0 / self.CRRA))
            * self.PatFac
            / solution_next.MPCmax
        )

        self.cFuncLimitIntercept = self.MPCminNow * self.hNrmNow
        self.cFuncLimitSlope = self.MPCminNow

    def defBoroCnst(self, BoroCnstArt):
        """
        Defines the constrained portion of the consumption function as cFuncNowCnst,
        an attribute of self.  Uses the artificial and natural borrowing constraints.

        Parameters
        ----------
        BoroCnstArt : float or None
            Borrowing constraint for the minimum allowable assets to end the
            period with.  If it is less than the natural borrowing constraint,
            then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
            rowing constraint.

        Returns
        -------
        none
        """
        # Calculate the minimum allowable value of money resources in this period
        self.BoroCnstNat = (
            (self.solution_next.mNrmMin - self.TranShkMinNext)
            * (self.PermGroFac * self.PermShkMinNext)
            / self.Rfree
        )

        # Note: need to be sure to handle BoroCnstArt==None appropriately.
        # In Py2, this would evaluate to 5.0:  np.max([None, 5.0]).
        # However in Py3, this raises a TypeError. Thus here we need to directly
        # address the situation in which BoroCnstArt == None:
        if BoroCnstArt is None:
            self.mNrmMinNow = self.BoroCnstNat
        else:
            self.mNrmMinNow = np.max([self.BoroCnstNat, BoroCnstArt])
        if self.BoroCnstNat < self.mNrmMinNow:
            self.MPCmaxEff = 1.0  # If actually constrained, MPC near limit is 1
        else:
            self.MPCmaxEff = self.MPCmaxNow

        # Define the borrowing constraint (limiting consumption function)
        self.cFuncNowCnst = LinearInterp(
            np.array([self.mNrmMinNow, self.mNrmMinNow + 1]), np.array([0.0, 1.0])
        )

    def prepareToSolve(self):
        """
        Perform preparatory work before calculating the unconstrained consumption
        function.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        self.setAndUpdateValues(
            self.solution_next, self.IncomeDstn, self.LivPrb, self.DiscFac
        )
        self.defBoroCnst(self.BoroCnstArt)

    def solve(self):
        """
        Solves a one period consumption saving problem with risky income.

        Parameters
        ----------
        None

        Returns
        -------
        solution : ConsumerSolution
            The solution to the one period problem.
        """

        sn_cFunc = self.solution_next.cFunc

        if isinstance(sn_cFunc, LowerEnvelope):
            x_list_1 = sn_cFunc.functions[0].x_list
            x_list_2 = sn_cFunc.functions[1].x_list
            sn_cFunc_x_list = np.sort(np.unique(np.append(x_list_1, x_list_2)))
        else:
            sn_cFunc_x_list = sn_cFunc.x_list

        sn_cFunc_y_list = self.solution_next.cFunc(sn_cFunc_x_list)
        self.cNrm, self.mNrm = solveConsIndShockNumba(
            self.DiscFacEff,
            self.CRRA,
            self.Rfree,
            self.PermGroFac,
            self.BoroCnstNat,
            self.aXtraGrid,
            self.TranShkValsNext,
            self.PermShkValsNext,
            self.ShkPrbsNext,
            sn_cFunc_x_list,
            sn_cFunc_y_list,
        )

        """
        Constructs a basic solution for this period, including the consumption
        function and marginal value function.
        """

        # Makes a linear interpolation to represent the (unconstrained) consumption function.
        # Construct the unconstrained consumption function
        cFuncNowUnc = LinearInterp(
            self.mNrm, self.cNrm, self.cFuncLimitIntercept, self.cFuncLimitSlope
        )

        # Combine the constrained and unconstrained functions into the true consumption function
        cFuncNow = LowerEnvelope(cFuncNowUnc, self.cFuncNowCnst)

        # Make the marginal value function and the marginal marginal value function
        vPfuncNow = MargValueFunc(cFuncNow, self.CRRA)

        # Pack up the solution and return it
        solution = ConsumerSolution(
            cFunc=cFuncNow,
            vPfunc=vPfuncNow,
            mNrmMin=self.mNrmMinNow,
            hNrm=self.hNrmNow,
            MPCmin=self.MPCminNow,
            MPCmax=self.MPCmaxEff,
        )

        return solution


class IndShockConsumerTypeNumba(IndShockConsumerType):
    def __init__(self, **kwargs):
        IndShockConsumerType.__init__(self, **kwargs)

        # Add consumer-type specific objects, copying to create independent versions
        if (not self.CubicBool) and (not self.vFuncBool):
            solver = ConsIndShockNumbaSolverBasic
        # else:  # Use the "advanced" solver if either is requested
        #     solver = ConsIndShockSolverNumba

        self.solveOnePeriod = makeOnePeriodOOSolver(solver)
