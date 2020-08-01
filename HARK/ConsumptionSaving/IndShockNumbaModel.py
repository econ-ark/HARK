from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy

import numpy as np
from numba import njit

from HARK import makeOnePeriodOOSolver, Solution
from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    MargValueFunc,
    ConsumerSolution,
    ConsPerfForesightSolver,
    MargMargValueFunc,
    ValueFunc,
)
from HARK.ConsumptionSaving.ConsIndShockNumba import ConsPerfForesightSolverNumba
from HARK.interpolation import LowerEnvelope, LinearInterp, CubicInterp
from HARK.numba import (
    CRRAutility,
    CRRAutilityP,
    CRRAutilityPP,
    CRRAutilityP_inv,
    CRRAutility_invP,
    CRRAutility_inv,
    CRRAutilityP_invP,
    LinearInterpFast,
)
from HARK.numba import splrep, splevec

utility = CRRAutility
utilityP = CRRAutilityP
utilityPP = CRRAutilityPP
utilityP_inv = CRRAutilityP_inv
utility_invP = CRRAutility_invP
utility_inv = CRRAutility_inv
utilityP_invP = CRRAutilityP_invP


class IndShockSolution(Solution):
    distance_criteria = ["cNrm", "mNrm", "mNrmMin"]

    def __init__(
        self,
        mNrm=np.array([0.0, 1.0]),
        cNrm=np.array([0.0, 1.0]),
        cFuncLimitIntercept=None,
        cFuncLimitSlope=None,
        mNrmMin=0.0,
        hNrm=0.0,
        MPCmin=1.0,
        MPCmax=1.0,
        ExIncNext=0.0,
    ):
        self.mNrm = mNrm
        self.cNrm = cNrm
        self.cFuncLimitIntercept = cFuncLimitIntercept
        self.cFuncLimitSlope = cFuncLimitSlope
        self.mNrmMin = mNrmMin
        self.hNrm = hNrm
        self.MPCmin = MPCmin
        self.MPCmax = MPCmax
        self.ExIncNext = ExIncNext


@njit(cache=True)
def _np_tile(A, reps):
    return A.repeat(reps[0]).reshape(A.size, -1).transpose()


@njit(cache=True)
def _np_insert(arr, obj, values, axis=-1):
    return np.append(np.array(values), arr)


@njit(cache=True)
def _prepareToSolveConsIndShockNumba(
    DiscFac,
    LivPrb,
    CRRA,
    Rfree,
    PermGroFac,
    BoroCnstArt,
    aXtraGrid,
    hNrmNext,
    mNrmMinNext,
    MPCminNext,
    MPCmaxNext,
    PermShkValsNext,
    TranShkValsNext,
    ShkPrbsNext,
):
    DiscFacEff = DiscFac * LivPrb  # "effective" discount factor
    PermShkMinNext = np.min(PermShkValsNext)
    TranShkMinNext = np.min(TranShkValsNext)
    WorstIncPrb = np.sum(
        ShkPrbsNext[
            (PermShkValsNext * TranShkValsNext) == (PermShkMinNext * TranShkMinNext)
        ]
    )

    # Update the bounding MPCs and PDV of human wealth:
    PatFac = ((Rfree * DiscFacEff) ** (1.0 / CRRA)) / Rfree
    MPCminNow = 1.0 / (1.0 + PatFac / MPCminNext)
    ExIncNext = np.dot(ShkPrbsNext, TranShkValsNext * PermShkValsNext)
    hNrmNow = PermGroFac / Rfree * (ExIncNext + hNrmNext)
    MPCmaxNow = 1.0 / (1.0 + (WorstIncPrb ** (1.0 / CRRA)) * PatFac / MPCmaxNext)

    cFuncLimitIntercept = MPCminNow * hNrmNow
    cFuncLimitSlope = MPCminNow

    # Calculate the minimum allowable value of money resources in this period
    BoroCnstNat = (mNrmMinNext - TranShkMinNext) * (PermGroFac * PermShkMinNext) / Rfree

    # Note: need to be sure to handle BoroCnstArt==None appropriately.
    # In Py2, this would evaluate to 5.0:  np.max([None, 5.0]).
    # However in Py3, this raises a TypeError. Thus here we need to directly
    # address the situation in which BoroCnstArt == None:
    if BoroCnstArt is None:
        mNrmMinNow = BoroCnstNat
    else:
        mNrmMinNow = np.max(np.array([BoroCnstNat, BoroCnstArt]))
    if BoroCnstNat < mNrmMinNow:
        MPCmaxEff = 1.0  # If actually constrained, MPC near limit is 1
    else:
        MPCmaxEff = MPCmaxNow

    # We define aNrmNow all the way from BoroCnstNat up to max(self.aXtraGrid)
    # even if BoroCnstNat < BoroCnstArt, so we can construct the consumption
    # function as the lower envelope of the (by the artificial borrowing con-
    # straint) uconstrained consumption function, and the artificially con-
    # strained consumption function.
    aNrmNow = np.asarray(aXtraGrid) + BoroCnstNat
    ShkCount = TranShkValsNext.size
    aNrm_temp = _np_tile(aNrmNow, (ShkCount, 1))

    # Tile arrays of the income shocks and put them into useful shapes
    aNrmCount = aNrmNow.shape[0]
    PermShkVals_temp = (_np_tile(PermShkValsNext, (aNrmCount, 1))).transpose()
    TranShkVals_temp = (_np_tile(TranShkValsNext, (aNrmCount, 1))).transpose()
    ShkPrbs_temp = (_np_tile(ShkPrbsNext, (aNrmCount, 1))).transpose()

    # Get cash on hand next period
    mNrmNext = Rfree / (PermGroFac * PermShkVals_temp) * aNrm_temp + TranShkVals_temp
    # CDC 20191205: This should be divided by LivPrb[0] for Blanchard insurance

    return (
        DiscFacEff,
        BoroCnstNat,
        cFuncLimitIntercept,
        cFuncLimitSlope,
        mNrmMinNow,
        hNrmNow,
        MPCminNow,
        MPCmaxEff,
        ExIncNext,
        mNrmNext,
        PermShkVals_temp,
        ShkPrbs_temp,
        aNrmNow,
    )


@njit(cache=True)
def _solveConsIndShockLinearNumba(
    mNrmMinNext,
    mNrmNext,
    CRRA,
    mNrmUnc,
    cNrmUnc,
    DiscFacEff,
    Rfree,
    PermGroFac,
    PermShkVals_temp,
    ShkPrbs_temp,
    aNrmNow,
    BoroCnstNat,
    cFuncInterceptNext,
    cFuncSlopeNext,
):
    # interp does not take ndarray inputs on 3rd argument, so flatten then reshape
    mNrmCnst = np.array([mNrmMinNext, mNrmMinNext + 1])
    cNrmCnst = np.array([0.0, 1.0])
    cFuncNextCnst = LinearInterpFast(mNrmNext.flatten(), mNrmCnst, cNrmCnst)
    cFuncNextUnc = LinearInterpFast(
        mNrmNext.flatten(), mNrmUnc, cNrmUnc, cFuncInterceptNext, cFuncSlopeNext
    )
    cFuncNext = np.minimum(cFuncNextCnst, cFuncNextUnc)
    vPfuncNext = utilityP(cFuncNext, CRRA).reshape(mNrmNext.shape)

    EndOfPrdvP = (
        DiscFacEff
        * Rfree
        * PermGroFac ** (-CRRA)
        * np.sum(PermShkVals_temp ** (-CRRA) * vPfuncNext * ShkPrbs_temp, axis=0)
    )

    # Finds interpolation points (c,m) for the consumption function.

    cNrmNow = utilityP_inv(EndOfPrdvP, CRRA)
    mNrmNow = cNrmNow + aNrmNow

    # Limiting consumption is zero as m approaches mNrmMin
    cNrm = _np_insert(cNrmNow, 0, 0.0, axis=-1)
    mNrm = _np_insert(mNrmNow, 0, BoroCnstNat, axis=-1)

    return (
        cNrm,
        mNrm,
    )


@njit(cache=True)
def _solveConsIndShockCubicNumba(
    DiscFacEff,
    CRRA,
    Rfree,
    PermGroFac,
    BoroCnstNat,
    aXtraGrid,
    TranShkValsNext,
    PermShkValsNext,
    ShkPrbsNext,
    MPCmaxNow,
    sn_cFunc_x_list,
    sn_cFunc_y_list,
    sn_cFunc_dydx,
):
    """
        Numba global method to solve ConsIndShockModel.

        Parameters
        ----------
        DiscFacEff
        CRRA
        Rfree
        PermGroFac
        BoroCnstNat
        aXtraGrid
        TranShkValsNext
        PermShkValsNext
        ShkPrbsNext
        sn_cFunc_x_list
        sn_cFunc_y_list

        Returns
        -------

        """

    # this is where linear and cubic start to differ

    # interp does not take ndarray inputs on 3rd argument, so flatten then reshape
    cFuncCoeffs = splrep(sn_cFunc_x_list, sn_cFunc_y_list)
    cFuncNext = splevec(
        mNrmNext.flatten(), sn_cFunc_x_list, sn_cFunc_y_list, cFuncCoeffs
    )
    vPfuncNext = (cFuncNext ** -CRRA).reshape(mNrmNext.shape)

    EndOfPrdvP = (
        DiscFacEff
        * Rfree
        * PermGroFac ** (-CRRA)
        * np.sum(PermShkVals_temp ** (-CRRA) * vPfuncNext * ShkPrbs_temp, axis=0)
    )
    # Finds interpolation points (c,m) for the consumption function.

    cNrmNow = EndOfPrdvP ** (-1.0 / CRRA)
    mNrmNow = cNrmNow + aNrmNow

    # Limiting consumption is zero as m approaches mNrmMin
    cNrm = _np_insert(cNrmNow, 0, 0.0, axis=-1)
    mNrm = _np_insert(mNrmNow, 0, BoroCnstNat, axis=-1)

    MPCCoeffs = splrep(sn_cFunc_x_list, sn_cFunc_dydx)
    MPCNext = splevec(mNrmNext.flatten(), sn_cFunc_x_list, sn_cFunc_dydx, MPCCoeffs)
    vPPfuncNext = (MPCNext * (-CRRA * cFuncNext ** (-CRRA - 1.0))).reshape(
        mNrmNext.shape
    )

    EndOfPrdvPP = (
        DiscFacEff
        * Rfree
        * Rfree
        * PermGroFac ** (-CRRA - 1.0)
        * np.sum(PermShkVals_temp ** (-CRRA - 1.0) * vPPfuncNext * ShkPrbs_temp, axis=0)
    )
    dcda = EndOfPrdvPP / utilityPP(cNrm[1:], CRRA)
    MPC = dcda / (dcda + 1.0)
    MPC = _np_insert(MPC, 0, MPCmaxNow)

    return (
        EndOfPrdvP,
        cNrm,
        mNrm,
        cNrmNow,
        mNrmNow,
        mNrmNext,
        MPC,
        PermShkVals_temp,
        ShkPrbs_temp,
        aNrmNow,
    )


@njit(cache=True)
def _addvFuncNumba(
    DiscFacEff,
    CRRA,
    PermGroFac,
    BoroCnstNat,
    aXtraGrid,
    MPCmaxEff,
    MPCminNow,
    aNrmNow,
    mNrmMinNow,
    EndOfPrdvP,
    mNrmNext,
    PermShkVals_temp,
    ShkPrbs_temp,
    sn_cFunc_x_list,
    cFunc_x_list,
    cFunc_y_list,
    sn_vFunc_y_list,
):
    """
    Construct the end-of-period value function for this period, storing it
    as an attribute of self for use by other methods.
    """

    vFuncCoeffs = splrep(sn_cFunc_x_list, sn_vFunc_y_list)
    vFuncNext = splevec(
        mNrmNext.flatten(), sn_cFunc_x_list, sn_vFunc_y_list, vFuncCoeffs
    ).reshape(mNrmNext.shape)

    VLvlNext = (
        PermShkVals_temp ** (1.0 - CRRA) * PermGroFac ** (1.0 - CRRA)
    ) * vFuncNext
    EndOfPrdv = DiscFacEff * np.sum(VLvlNext * ShkPrbs_temp, axis=0)

    # value transformed through inverse utility
    EndOfPrdvNvrs = utility_inv(EndOfPrdv, CRRA)
    EndOfPrdvNvrsP = EndOfPrdvP * utility_invP(EndOfPrdv, CRRA)
    EndOfPrdvNvrs = _np_insert(EndOfPrdvNvrs, 0, 0.0)

    # This is a very good approximation, vNvrsPP = 0 at the asset minimum
    EndOfPrdvNvrsP = _np_insert(EndOfPrdvNvrsP, 0, EndOfPrdvNvrsP[0])
    aNrm_temp = _np_insert(aNrmNow, 0, BoroCnstNat)

    """
    Creates the value function for this period, defined over market resources m.
    self must have the attribute EndOfPrdvFunc in order to execute.
    """
    # Compute expected value and marginal value on a grid of market resources
    mNrm_temp = mNrmMinNow + aXtraGrid

    # current cfunc
    cNrmCoeffs = splrep(cFunc_x_list, cFunc_y_list)
    cNrmNow = splevec(
        mNrm_temp.flatten(), cFunc_x_list, cFunc_y_list, cNrmCoeffs
    ).reshape(mNrm_temp.shape)

    aNrmNow = mNrm_temp - cNrmNow

    EOPvFCoeffs = splrep(aNrm_temp, EndOfPrdvNvrs)
    EOPvFaNN = splevec(aNrmNow.flatten(), aNrm_temp, EndOfPrdvNvrs, EOPvFCoeffs)
    EOPvFaNN = (EOPvFaNN ** (1.0 - CRRA) / (1.0 - CRRA)).reshape(aNrmNow.shape)

    vNrmNow = utility(cNrmNow, CRRA) + EOPvFaNN
    vPnow = utilityP(cNrmNow, CRRA)

    # Construct the beginning-of-period value function
    vNvrs = utility_inv(vNrmNow, CRRA)  # value transformed through inverse utility
    vNvrsP = vPnow * utility_invP(vNrmNow, CRRA)
    mNrm_temp = _np_insert(mNrm_temp, 0, mNrmMinNow)
    vNvrs = _np_insert(vNvrs, 0, 0.0)
    vNvrsP = _np_insert(vNvrsP, 0, MPCmaxEff ** (-CRRA / (1.0 - CRRA)))
    MPCminNvrs = MPCminNow ** (-CRRA / (1.0 - CRRA))

    return (
        aNrm_temp,
        EndOfPrdvNvrs,
        EndOfPrdvNvrsP,
        mNrm_temp,
        vNvrs,
        vNvrsP,
        MPCminNvrs,
    )


class ConsIndShockNumbaSolverBasic(ConsPerfForesightSolverNumba):
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

        self.ShkPrbsNext = self.IncomeDstn.pmf
        self.PermShkValsNext = self.IncomeDstn.X[0]
        self.TranShkValsNext = self.IncomeDstn.X[1]

        (
            self.DiscFacEff,
            self.BoroCnstNat,
            self.cFuncLimitIntercept,
            self.cFuncLimitSlope,
            self.mNrmMinNow,
            self.hNrmNow,
            self.MPCminNow,
            self.MPCmaxEff,
            self.ExIncNext,
            self.mNrmNext,
            self.PermShkVals_temp,
            self.ShkPrbs_temp,
            self.aNrmNow,
        ) = _prepareToSolveConsIndShockNumba(
            self.DiscFac,
            self.LivPrb,
            self.CRRA,
            self.Rfree,
            self.PermGroFac,
            self.BoroCnstArt,
            self.aXtraGrid,
            self.solution_next.hNrm,
            self.solution_next.mNrmMin,
            self.solution_next.MPCmin,
            self.solution_next.MPCmax,
            self.PermShkValsNext,
            self.TranShkValsNext,
            self.ShkPrbsNext,
        )

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

        self.cNrm, self.mNrm = _solveConsIndShockLinearNumba(
            self.solution_next.mNrmMin,
            self.mNrmNext,
            self.CRRA,
            self.solution_next.mNrm,
            self.solution_next.cNrm,
            self.DiscFacEff,
            self.Rfree,
            self.PermGroFac,
            self.PermShkVals_temp,
            self.ShkPrbs_temp,
            self.aNrmNow,
            self.BoroCnstNat,
            self.solution_next.cFuncLimitIntercept,
            self.solution_next.cFuncLimitSlope,
        )

        # Pack up the solution and return it
        solution = IndShockSolution(
            self.mNrm,
            self.cNrm,
            self.cFuncLimitIntercept,
            self.cFuncLimitSlope,
            self.mNrmMinNow,
            self.hNrmNow,
            self.MPCminNow,
            self.MPCmaxEff,
            self.ExIncNext,
        )

        return solution


class ConsIndShockSolverNumba(ConsIndShockNumbaSolverBasic):
    """
        This class solves a single period of a standard consumption-saving problem.
        It inherits from ConsIndShockSolverBasic, adding the ability to perform cubic
        interpolation and to calculate the value function.
        """

    def solve(self):
        """
        Solves the single period consumption-saving problem using the method of
        endogenous gridpoints.  Solution includes a consumption function cFunc
        (using cubic or linear splines), a marginal value function vPfunc, a min-
        imum acceptable level of normalized market resources mNrmMin, normalized
        human wealth hNrm, and bounding MPCs MPCmin and MPCmax.  It might also
        have a value function vFunc and marginal marginal value function vPPfunc.

        Parameters
        ----------
        none

        Returns
        -------
        solution : ConsumerSolution
            The solution to the single period consumption-saving problem.
        """
        # Make arrays of end-of-period assets and end-of-period marginal value

        sn_cFunc_x_list = self.aXtraGrid

        # Construct a basic solution for this period
        if self.CubicBool:
            (
                sn_cFunc_y_list,
                sn_cFunc_dydx,
            ) = self.solution_next.cFunc.eval_with_derivative(sn_cFunc_x_list)

            (
                self.EndOfPrdvP,
                self.cNrm,
                self.mNrm,
                self.cNrmNow,
                self.mNrmNow,
                self.mNrmNext,
                self.MPC,
                self.PermShkVals_temp,
                self.ShkPrbs_temp,
                self.aNrmNow,
            ) = _solveConsIndShockCubicNumba(
                self.DiscFacEff,
                self.CRRA,
                self.Rfree,
                self.PermGroFac,
                self.BoroCnstNat,
                self.aXtraGrid,
                self.TranShkValsNext,
                self.PermShkValsNext,
                self.ShkPrbsNext,
                self.MPCmaxNow,
                sn_cFunc_x_list,
                sn_cFunc_y_list,
                sn_cFunc_dydx,
            )

            cFuncNowUnc = CubicInterp(
                self.mNrm,
                self.cNrm,
                self.MPC,
                self.MPCminNow * self.hNrmNow,
                self.MPCminNow,
            )
        else:
            sn_cFunc_y_list = self.solution_next.cFunc(sn_cFunc_x_list)

            (
                self.EndOfPrdvP,
                self.cNrm,
                self.mNrm,
                self.cNrmNow,
                self.mNrmNow,
                self.mNrmNext,
                self.PermShkVals_temp,
                self.ShkPrbs_temp,
                self.aNrmNow,
            ) = _solveConsIndShockLinearNumba(
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

        solution = self.addSSmNrm(solution)  # find steady state m

        # Add the value function if requested, as well as the marginal marginal
        # value function if cubic splines were used (to prepare for next period)
        if self.vFuncBool:
            cFunc_x_list = self.aXtraGrid
            cFunc_y_list = solution.cFunc(cFunc_x_list)
            sn_vFunc_y_list = self.solution_next.vFunc(cFunc_x_list)

            (
                self.aNrm_temp,
                self.EndOfPrdvNvrs,
                self.EndOfPrdvNvrsP,
                self.mNrm_temp,
                self.vNvrs,
                self.vNvrsP,
                self.MPCminNvrs,
            ) = _addvFuncNumba(
                self.DiscFacEff,
                self.CRRA,
                self.PermGroFac,
                self.BoroCnstNat,
                self.aXtraGrid,
                self.MPCmaxEff,
                self.MPCminNow,
                self.aNrmNow,
                self.mNrmMinNow,
                self.EndOfPrdvP,
                self.mNrmNext,
                self.PermShkVals_temp,
                self.ShkPrbs_temp,
                sn_cFunc_x_list,
                cFunc_x_list,
                cFunc_y_list,
                sn_vFunc_y_list,
            )

            EndOfPrdvNvrsFunc = CubicInterp(
                self.aNrm_temp, self.EndOfPrdvNvrs, self.EndOfPrdvNvrsP
            )

            self.EndOfPrdvFunc = ValueFunc(EndOfPrdvNvrsFunc, self.CRRA)

            vNvrsFuncNow = CubicInterp(
                self.mNrm_temp,
                self.vNvrs,
                self.vNvrsP,
                self.MPCminNvrs * self.hNrmNow,
                self.MPCminNvrs,
            )
            vFuncNow = ValueFunc(vNvrsFuncNow, self.CRRA)

            solution.vFunc = vFuncNow

        if self.CubicBool:
            vPPfuncNow = MargMargValueFunc(solution.cFunc, self.CRRA)
            solution.vPPfunc = vPPfuncNow
        return solution


class IndShockConsumerTypeNumba(IndShockConsumerType):
    solution_terminal_ = IndShockSolution()

    def __init__(self, **kwargs):
        IndShockConsumerType.__init__(self, **kwargs)

        # Add consumer-type specific objects, copying to create independent versions
        if (not self.CubicBool) and (not self.vFuncBool):
            solver = ConsIndShockNumbaSolverBasic
        else:  # Use the "advanced" solver if either is requested
            solver = ConsIndShockSolverNumba

        self.solveOnePeriod = makeOnePeriodOOSolver(solver)

    def updateSolutionTerminal(self):
        pass

    def postSolve(self):
        self.solution_fast = deepcopy(self.solution)

        if self.cycles == 0:
            terminal = 1
        else:
            terminal = self.cycles

        for i in range(terminal):
            solution = self.solution[i]

            # Define the borrowing constraint (limiting consumption function)
            cFuncNowCnst = LinearInterp(
                np.array([solution.mNrmMin, solution.mNrmMin + 1]), np.array([0.0, 1.0])
            )

            """
            Constructs a basic solution for this period, including the consumption
            function and marginal value function.
            """

            # Makes a linear interpolation to represent the (unconstrained) consumption function.
            # Construct the unconstrained consumption function
            cFuncNowUnc = LinearInterp(
                solution.mNrm,
                solution.cNrm,
                solution.cFuncLimitIntercept,
                solution.cFuncLimitSlope,
            )

            # Combine the constrained and unconstrained functions into the true consumption function
            cFuncNow = LowerEnvelope(cFuncNowUnc, cFuncNowCnst)

            # Make the marginal value function and the marginal marginal value function
            vPfuncNow = MargValueFunc(cFuncNow, self.CRRA)

            # Pack up the solution and return it
            consumer_solution = ConsumerSolution(
                cFunc=cFuncNow,
                vPfunc=vPfuncNow,
                mNrmMin=solution.mNrmMin,
                hNrm=solution.hNrm,
                MPCmin=solution.MPCmin,
                MPCmax=solution.MPCmax,
            )

            self.solution[i] = consumer_solution
