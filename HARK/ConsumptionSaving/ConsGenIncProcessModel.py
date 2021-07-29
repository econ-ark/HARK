"""
Classes to solve consumption-saving models with idiosyncratic shocks to income
in which shocks are not necessarily fully transitory or fully permanent.  Extends
ConsIndShockModel by explicitly tracking persistent income as a state variable,
and allows (log) persistent income to follow an AR1 process rather than random walk.
"""
from copy import deepcopy
import numpy as np
from HARK import AgentType, MetricObject, make_one_period_oo_solver
from HARK.distribution import DiscreteDistribution
from HARK.interpolation import (
    LowerEnvelope2D,
    BilinearInterp,
    VariableLowerBoundFunc2D,
    LinearInterpOnInterp1D,
    LinearInterp,
    CubicInterp,
    UpperEnvelope,
    ValueFuncCRRA,
    MargValueFuncCRRA,
    MargMargValueFuncCRRA
)
from HARK.utilities import (
    CRRAutility,
    CRRAutilityP,
    CRRAutilityPP,
    CRRAutilityP_inv,
    CRRAutility_invP,
    CRRAutility_inv,
    CRRAutilityP_invP,
    get_percentiles,
)
from HARK.distribution import Lognormal, Uniform
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsIndShockSetup,
    ConsumerSolution,
    IndShockConsumerType,
    init_idiosyncratic_shocks,
)

__all__ = [
    "pLvlFuncAR1",
    "ConsGenIncProcessSolver",
    "GenIncProcessConsumerType",
    "IndShockExplicitPermIncConsumerType",
    "PersistentShockConsumerType",
    "init_explicit_perm_inc",
    "init_persistent_shocks",
]

utility = CRRAutility
utilityP = CRRAutilityP
utilityPP = CRRAutilityPP
utilityP_inv = CRRAutilityP_inv
utility_invP = CRRAutility_invP
utility_inv = CRRAutility_inv
utilityP_invP = CRRAutilityP_invP


class pLvlFuncAR1(MetricObject):
    """
    A class for representing AR1-style persistent income growth functions.

    Parameters
    ----------
    pLogMean : float
        Log persistent income level toward which we are drawn.
    PermGroFac : float
        Autonomous (e.g. life cycle) pLvl growth (does not AR1 decay).
    Corr : float
        Correlation coefficient on log income.
    """

    def __init__(self, pLogMean, PermGroFac, Corr):
        self.pLogMean = pLogMean
        self.LogGroFac = np.log(PermGroFac)
        self.Corr = Corr

    def __call__(self, pLvlNow):
        """
        Returns expected persistent income level next period as a function of
        this period's persistent income level.

        Parameters
        ----------
        pLvlNow : np.array
            Array of current persistent income levels.

        Returns
        -------
        pLvlNext : np.array
            Identically shaped array of next period persistent income levels.
        """
        pLvlNext = np.exp(
            self.Corr * np.log(pLvlNow)
            + (1.0 - self.Corr) * self.pLogMean
            + self.LogGroFac
        )
        return pLvlNext


###############################################################################


class ConsGenIncProcessSolver(ConsIndShockSetup):
    """
    A class for solving one period problem of a consumer who experiences persistent and
    transitory shocks to his income.  Unlike in ConsIndShock, consumers do not
    necessarily have the same predicted level of p next period as this period
    (after controlling for growth).  Instead, they have  a function that translates
    current persistent income into expected next period persistent income (subject
    to shocks).

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncShkDstn : distribution.Distribution
        A discrete
        approximation to the income process between the period being solved
        and the one immediately following (in solution_next). Order: event
        probabilities, persistent shocks, transitory shocks.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    pLvlNextFunc : float
        Expected persistent income next period as a function of current pLvl.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.
    aXtraGrid: np.array
        Array of "extra" end-of-period (normalized) asset values-- assets
        above the absolute minimum acceptable level.
    pLvlGrid: np.array
        Array of persistent income levels at which to solve the problem.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear interpolation.
    """

    def __init__(
        self,
        solution_next,
        IncShkDstn,
        LivPrb,
        DiscFac,
        CRRA,
        Rfree,
        pLvlNextFunc,
        BoroCnstArt,
        aXtraGrid,
        pLvlGrid,
        vFuncBool,
        CubicBool,
    ):
        """
        Constructor for a new solver for a one period problem with idiosyncratic
        shocks to persistent and transitory income, with persistent income tracked
        as a state variable rather than normalized out.
        """
        self.solution_next = solution_next
        self.IncShkDstn = IncShkDstn
        self.LivPrb = LivPrb
        self.DiscFac = DiscFac
        self.CRRA = CRRA
        self.Rfree = Rfree
        self.pLvlNextFunc = pLvlNextFunc
        self.BoroCnstArt = BoroCnstArt
        self.aXtraGrid = aXtraGrid
        self.pLvlGrid = pLvlGrid
        self.vFuncBool = vFuncBool
        self.CubicBool = CubicBool
        self.PermGroFac = 0.0

        self.def_utility_funcs()

    def set_and_update_values(self, solution_next, IncShkDstn, LivPrb, DiscFac):
        """
        Unpacks some of the inputs (and calculates simple objects based on them),
        storing the results in self for use by other methods.  These include:
        income shocks and probabilities, next period's marginal value function
        (etc), the probability of getting the worst income shock next period,
        the patience factor, human wealth, and the bounding MPCs.  Human wealth
        is stored as a function of persistent income.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncShkDstn : distribution.Distribution
            A discrete
            approximation to the income process between the period being solved
            and the one immediately following (in solution_next).
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.
        DiscFac : float
            Intertemporal discount factor for future utility.

        Returns
        -------
        None
        """
        # Run basic version of this method
        ConsIndShockSetup.set_and_update_values(
            self, solution_next, IncShkDstn, LivPrb, DiscFac
        )
        self.mLvlMinNext = solution_next.mLvlMin

        # Replace normalized human wealth (scalar) with human wealth level as function of persistent income
        self.hNrmNow = 0.0
        pLvlCount = self.pLvlGrid.size
        IncShkCount = self.PermShkValsNext.size
        pLvlNext = (
            np.tile(self.pLvlNextFunc(self.pLvlGrid), (IncShkCount, 1))
            * np.tile(self.PermShkValsNext, (pLvlCount, 1)).transpose()
        )
        hLvlGrid = (
            1.0
            / self.Rfree
            * np.sum(
                (
                    np.tile(self.TranShkValsNext, (pLvlCount, 1)).transpose() * pLvlNext
                    + solution_next.hLvl(pLvlNext)
                )
                * np.tile(self.ShkPrbsNext, (pLvlCount, 1)).transpose(),
                axis=0,
            )
        )
        self.hLvlNow = LinearInterp(
            np.insert(self.pLvlGrid, 0, 0.0), np.insert(hLvlGrid, 0, 0.0)
        )

    def def_BoroCnst(self, BoroCnstArt):
        """
        Defines the constrained portion of the consumption function as cFuncNowCnst,
        an attribute of self.

        Parameters
        ----------
        BoroCnstArt : float or None
            Borrowing constraint for the minimum allowable assets to end the
            period with.  If it is less than the natural borrowing constraint,
            then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
            rowing constraint.

        Returns
        -------
        None
        """
        # Make temporary grids of income shocks and next period income values
        ShkCount = self.TranShkValsNext.size
        pLvlCount = self.pLvlGrid.size
        PermShkVals_temp = np.tile(
            np.reshape(self.PermShkValsNext, (1, ShkCount)), (pLvlCount, 1)
        )
        TranShkVals_temp = np.tile(
            np.reshape(self.TranShkValsNext, (1, ShkCount)), (pLvlCount, 1)
        )
        pLvlNext_temp = (
            np.tile(
                np.reshape(self.pLvlNextFunc(self.pLvlGrid), (pLvlCount, 1)),
                (1, ShkCount),
            )
            * PermShkVals_temp
        )

        # Find the natural borrowing constraint for each persistent income level
        aLvlMin_candidates = (
            self.mLvlMinNext(pLvlNext_temp) - TranShkVals_temp * pLvlNext_temp
        ) / self.Rfree
        aLvlMinNow = np.max(aLvlMin_candidates, axis=1)
        self.BoroCnstNat = LinearInterp(
            np.insert(self.pLvlGrid, 0, 0.0), np.insert(aLvlMinNow, 0, 0.0)
        )

        # Define the minimum allowable mLvl by pLvl as the greater of the natural and artificial borrowing constraints
        if self.BoroCnstArt is not None:
            self.BoroCnstArt = LinearInterp(
                np.array([0.0, 1.0]), np.array([0.0, self.BoroCnstArt])
            )
            self.mLvlMinNow = UpperEnvelope(self.BoroCnstArt, self.BoroCnstNat)
        else:
            self.mLvlMinNow = self.BoroCnstNat

        # Define the constrained consumption function as "consume all" shifted by mLvlMin
        cFuncNowCnstBase = BilinearInterp(
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        )
        self.cFuncNowCnst = VariableLowerBoundFunc2D(cFuncNowCnstBase, self.mLvlMinNow)

    def prepare_to_calc_EndOfPrdvP(self):
        """
        Prepare to calculate end-of-period marginal value by creating an array
        of market resources that the agent could have next period, considering
        the grid of end-of-period normalized assets, the grid of persistent income
        levels, and the distribution of shocks he might experience next period.

        Parameters
        ----------
        None

        Returns
        -------
        aLvlNow : np.array
            2D array of end-of-period assets; also stored as attribute of self.
        pLvlNow : np.array
            2D array of persistent income levels this period.
        """
        ShkCount = self.TranShkValsNext.size
        pLvlCount = self.pLvlGrid.size
        aNrmCount = self.aXtraGrid.size
        pLvlNow = np.tile(self.pLvlGrid, (aNrmCount, 1)).transpose()
        aLvlNow = np.tile(self.aXtraGrid, (pLvlCount, 1)) * pLvlNow + self.BoroCnstNat(
            pLvlNow
        )
        pLvlNow_tiled = np.tile(pLvlNow, (ShkCount, 1, 1))
        aLvlNow_tiled = np.tile(
            aLvlNow, (ShkCount, 1, 1)
        )  # shape = (ShkCount,pLvlCount,aNrmCount)
        if self.pLvlGrid[0] == 0.0:  # aLvl turns out badly if pLvl is 0 at bottom
            aLvlNow[0, :] = self.aXtraGrid
            aLvlNow_tiled[:, 0, :] = np.tile(self.aXtraGrid, (ShkCount, 1))

        # Tile arrays of the income shocks and put them into useful shapes
        PermShkVals_tiled = np.transpose(
            np.tile(self.PermShkValsNext, (aNrmCount, pLvlCount, 1)), (2, 1, 0)
        )
        TranShkVals_tiled = np.transpose(
            np.tile(self.TranShkValsNext, (aNrmCount, pLvlCount, 1)), (2, 1, 0)
        )
        ShkPrbs_tiled = np.transpose(
            np.tile(self.ShkPrbsNext, (aNrmCount, pLvlCount, 1)), (2, 1, 0)
        )

        # Get cash on hand next period
        pLvlNext = self.pLvlNextFunc(pLvlNow_tiled) * PermShkVals_tiled
        mLvlNext = self.Rfree * aLvlNow_tiled + pLvlNext * TranShkVals_tiled

        # Store and report the results
        self.ShkPrbs_temp = ShkPrbs_tiled
        self.pLvlNext = pLvlNext
        self.mLvlNext = mLvlNext
        self.aLvlNow = aLvlNow
        return aLvlNow, pLvlNow

    def calc_EndOfPrdvP(self):
        """
        Calculates end-of-period marginal value of assets at each state space
        point in aLvlNow x pLvlNow. Does so by taking a weighted sum of next
        period marginal values across income shocks (in preconstructed grids
        self.mLvlNext x self.pLvlNext).

        Parameters
        ----------
        None

        Returns
        -------
        EndOfPrdVP : np.array
            A 2D array of end-of-period marginal value of assets.
        """
        EndOfPrdvP = (
            self.DiscFacEff
            * self.Rfree
            * np.sum(
                self.vPfuncNext(self.mLvlNext, self.pLvlNext) * self.ShkPrbs_temp,
                axis=0,
            )
        )
        return EndOfPrdvP

    def make_EndOfPrdvFunc(self, EndOfPrdvP):
        """
        Construct the end-of-period value function for this period, storing it
        as an attribute of self for use by other methods.

        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal value of assets corresponding to the
            asset values in self.aLvlNow x self.pLvlGrid.

        Returns
        -------
        none
        """
        vLvlNext = self.vFuncNext(
            self.mLvlNext, self.pLvlNext
        )  # value in many possible future states
        EndOfPrdv = self.DiscFacEff * np.sum(
            vLvlNext * self.ShkPrbs_temp, axis=0
        )  # expected value, averaging across states
        EndOfPrdvNvrs = self.uinv(
            EndOfPrdv
        )  # value transformed through inverse utility
        EndOfPrdvNvrsP = EndOfPrdvP * self.uinvP(EndOfPrdv)

        # Add points at mLvl=zero
        EndOfPrdvNvrs = np.concatenate(
            (np.zeros((self.pLvlGrid.size, 1)), EndOfPrdvNvrs), axis=1
        )
        if hasattr(self, "MedShkDstn"):
            EndOfPrdvNvrsP = np.concatenate(
                (np.zeros((self.pLvlGrid.size, 1)), EndOfPrdvNvrsP), axis=1
            )
        else:
            EndOfPrdvNvrsP = np.concatenate(
                (
                    np.reshape(EndOfPrdvNvrsP[:, 0], (self.pLvlGrid.size, 1)),
                    EndOfPrdvNvrsP,
                ),
                axis=1,
            )
            # This is a very good approximation, vNvrsPP = 0 at the asset minimum
        aLvl_temp = np.concatenate(
            (
                np.reshape(self.BoroCnstNat(self.pLvlGrid), (self.pLvlGrid.size, 1)),
                self.aLvlNow,
            ),
            axis=1,
        )

        # Make an end-of-period value function for each persistent income level in the grid
        EndOfPrdvNvrsFunc_list = []
        for p in range(self.pLvlGrid.size):
            EndOfPrdvNvrsFunc_list.append(
                CubicInterp(
                    aLvl_temp[p, :] - self.BoroCnstNat(self.pLvlGrid[p]),
                    EndOfPrdvNvrs[p, :],
                    EndOfPrdvNvrsP[p, :],
                )
            )
        EndOfPrdvNvrsFuncBase = LinearInterpOnInterp1D(
            EndOfPrdvNvrsFunc_list, self.pLvlGrid
        )

        # Re-adjust the combined end-of-period value function to account for the natural borrowing constraint shifter
        EndOfPrdvNvrsFunc = VariableLowerBoundFunc2D(
            EndOfPrdvNvrsFuncBase, self.BoroCnstNat
        )
        self.EndOfPrdvFunc = ValueFuncCRRA(EndOfPrdvNvrsFunc, self.CRRA)

    def get_points_for_interpolation(self, EndOfPrdvP, aLvlNow):
        """
        Finds endogenous interpolation points (c,m) for the consumption function.

        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal values.
        aLvlNow : np.array
            Array of end-of-period asset values that yield the marginal values
            in EndOfPrdvP.

        Returns
        -------
        c_for_interpolation : np.array
            Consumption points for interpolation.
        m_for_interpolation : np.array
            Corresponding market resource points for interpolation.
        """
        cLvlNow = self.uPinv(EndOfPrdvP)
        mLvlNow = cLvlNow + aLvlNow

        # Limiting consumption is zero as m approaches mNrmMin
        c_for_interpolation = np.concatenate(
            (np.zeros((self.pLvlGrid.size, 1)), cLvlNow), axis=-1
        )
        m_for_interpolation = np.concatenate(
            (
                self.BoroCnstNat(np.reshape(self.pLvlGrid, (self.pLvlGrid.size, 1))),
                mLvlNow,
            ),
            axis=-1,
        )

        # Limiting consumption is MPCmin*mLvl as p approaches 0
        m_temp = np.reshape(
            m_for_interpolation[0, :], (1, m_for_interpolation.shape[1])
        )
        m_for_interpolation = np.concatenate((m_temp, m_for_interpolation), axis=0)
        c_for_interpolation = np.concatenate(
            (self.MPCminNow * m_temp, c_for_interpolation), axis=0
        )

        return c_for_interpolation, m_for_interpolation

    def use_points_for_interpolation(self, cLvl, mLvl, pLvl, interpolator):
        """
        Constructs a basic solution for this period, including the consumption
        function and marginal value function.

        Parameters
        ----------
        cLvl : np.array
            Consumption points for interpolation.
        mLvl : np.array
            Corresponding market resource points for interpolation.
        pLvl : np.array
            Corresponding persistent income level points for interpolation.
        interpolator : function
            A function that constructs and returns a consumption function.

        Returns
        -------
        solution_now : ConsumerSolution
            The solution to this period's consumption-saving problem, with a
            consumption function, marginal value function, and minimum m.
        """
        # Construct the unconstrained consumption function
        cFuncNowUnc = interpolator(mLvl, pLvl, cLvl)

        # Combine the constrained and unconstrained functions into the true consumption function
        cFuncNow = LowerEnvelope2D(cFuncNowUnc, self.cFuncNowCnst)

        # Make the marginal value function
        vPfuncNow = self.make_vPfunc(cFuncNow)

        # Pack up the solution and return it
        solution_now = ConsumerSolution(cFunc=cFuncNow, vPfunc=vPfuncNow, mNrmMin=0.0)
        return solution_now

    def make_vPfunc(self, cFunc):
        """
        Constructs the marginal value function for this period.

        Parameters
        ----------
        cFunc : function
            Consumption function this period, defined over market resources and
            persistent income level.

        Returns
        -------
        vPfunc : function
            Marginal value (of market resources) function for this period.
        """
        vPfunc = MargValueFuncCRRA(cFunc, self.CRRA)
        return vPfunc

    def make_vFunc(self, solution):
        """
        Creates the value function for this period, defined over market resources
        m and persistent income p.  self must have the attribute EndOfPrdvFunc in
        order to execute.

        Parameters
        ----------
        solution : ConsumerSolution
            The solution to this single period problem, which must include the
            consumption function.

        Returns
        -------
        vFuncNow : ValueFuncCRRA
            A representation of the value function for this period, defined over
            market resources m and persistent income p: v = vFuncNow(m,p).
        """
        mSize = self.aXtraGrid.size
        pSize = self.pLvlGrid.size

        # Compute expected value and marginal value on a grid of market resources
        pLvl_temp = np.tile(self.pLvlGrid, (mSize, 1))  # Tile pLvl across m values
        mLvl_temp = (
            np.tile(self.mLvlMinNow(self.pLvlGrid), (mSize, 1))
            + np.tile(np.reshape(self.aXtraGrid, (mSize, 1)), (1, pSize)) * pLvl_temp
        )
        cLvlNow = solution.cFunc(mLvl_temp, pLvl_temp)
        aLvlNow = mLvl_temp - cLvlNow
        vNow = self.u(cLvlNow) + self.EndOfPrdvFunc(aLvlNow, pLvl_temp)
        vPnow = self.uP(cLvlNow)

        # Calculate pseudo-inverse value and its first derivative (wrt mLvl)
        vNvrs = self.uinv(vNow)  # value transformed through inverse utility
        vNvrsP = vPnow * self.uinvP(vNow)

        # Add data at the lower bound of m
        mLvl_temp = np.concatenate(
            (np.reshape(self.mLvlMinNow(self.pLvlGrid), (1, pSize)), mLvl_temp), axis=0
        )
        vNvrs = np.concatenate((np.zeros((1, pSize)), vNvrs), axis=0)
        vNvrsP = np.concatenate(
            (np.reshape(vNvrsP[0, :], (1, vNvrsP.shape[1])), vNvrsP), axis=0
        )

        # Add data at the lower bound of p
        MPCminNvrs = self.MPCminNow ** (-self.CRRA / (1.0 - self.CRRA))
        m_temp = np.reshape(mLvl_temp[:, 0], (mSize + 1, 1))
        mLvl_temp = np.concatenate((m_temp, mLvl_temp), axis=1)
        vNvrs = np.concatenate((MPCminNvrs * m_temp, vNvrs), axis=1)
        vNvrsP = np.concatenate((MPCminNvrs * np.ones((mSize + 1, 1)), vNvrsP), axis=1)

        # Construct the pseudo-inverse value function
        vNvrsFunc_list = []
        for j in range(pSize + 1):
            pLvl = np.insert(self.pLvlGrid, 0, 0.0)[j]
            vNvrsFunc_list.append(
                CubicInterp(
                    mLvl_temp[:, j] - self.mLvlMinNow(pLvl),
                    vNvrs[:, j],
                    vNvrsP[:, j],
                    MPCminNvrs * self.hLvlNow(pLvl),
                    MPCminNvrs,
                )
            )
        vNvrsFuncBase = LinearInterpOnInterp1D(
            vNvrsFunc_list, np.insert(self.pLvlGrid, 0, 0.0)
        )  # Value function "shifted"
        vNvrsFuncNow = VariableLowerBoundFunc2D(vNvrsFuncBase, self.mLvlMinNow)

        # "Re-curve" the pseudo-inverse value function into the value function
        vFuncNow = ValueFuncCRRA(vNvrsFuncNow, self.CRRA)
        return vFuncNow

    def make_basic_solution(self, EndOfPrdvP, aLvl, pLvl, interpolator):
        """
        Given end of period assets and end of period marginal value, construct
        the basic solution for this period.

        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal values.
        aLvl : np.array
            Array of end-of-period asset values that yield the marginal values
            in EndOfPrdvP.
        pLvl : np.array
            Array of persistent income levels that yield the marginal values
            in EndOfPrdvP (corresponding pointwise to aLvl).
        interpolator : function
            A function that constructs and returns a consumption function.

        Returns
        -------
        solution_now : ConsumerSolution
            The solution to this period's consumption-saving problem, with a
            consumption function, marginal value function, and minimum m.
        """
        cLvl, mLvl = self.get_points_for_interpolation(EndOfPrdvP, aLvl)
        pLvl_temp = np.concatenate(
            (np.reshape(self.pLvlGrid, (self.pLvlGrid.size, 1)), pLvl), axis=-1
        )
        pLvl_temp = np.concatenate((np.zeros((1, mLvl.shape[1])), pLvl_temp))
        solution_now = self.use_points_for_interpolation(
            cLvl, mLvl, pLvl_temp, interpolator
        )
        return solution_now

    def make_linear_cFunc(self, mLvl, pLvl, cLvl):
        """
        Makes a quasi-bilinear interpolation to represent the (unconstrained)
        consumption function.

        Parameters
        ----------
        mLvl : np.array
            Market resource points for interpolation.
        pLvl : np.array
            Persistent income level points for interpolation.
        cLvl : np.array
            Consumption points for interpolation.

        Returns
        -------
        cFuncUnc : LinearInterp
            The unconstrained consumption function for this period.
        """
        cFunc_by_pLvl_list = []  # list of consumption functions for each pLvl
        for j in range(pLvl.shape[0]):
            pLvl_j = pLvl[j, 0]
            m_temp = mLvl[j, :] - self.BoroCnstNat(pLvl_j)
            c_temp = cLvl[j, :]  # Make a linear consumption function for this pLvl
            if pLvl_j > 0:
                cFunc_by_pLvl_list.append(
                    LinearInterp(
                        m_temp,
                        c_temp,
                        lower_extrap=True,
                        slope_limit=self.MPCminNow,
                        intercept_limit=self.MPCminNow * self.hLvlNow(pLvl_j),
                    )
                )
            else:
                cFunc_by_pLvl_list.append(
                    LinearInterp(m_temp, c_temp, lower_extrap=True)
                )
        pLvl_list = pLvl[:, 0]
        cFuncUncBase = LinearInterpOnInterp1D(
            cFunc_by_pLvl_list, pLvl_list
        )  # Combine all linear cFuncs
        cFuncUnc = VariableLowerBoundFunc2D(
            cFuncUncBase, self.BoroCnstNat
        )  # Re-adjust for natural borrowing constraint (as lower bound)
        return cFuncUnc

    def make_cubic_cFunc(self, mLvl, pLvl, cLvl):
        """
        Makes a quasi-cubic spline interpolation of the unconstrained consumption
        function for this period.  Function is cubic splines with respect to mLvl,
        but linear in pLvl.

        Parameters
        ----------
        mLvl : np.array
            Market resource points for interpolation.
        pLvl : np.array
            Persistent income level points for interpolation.
        cLvl : np.array
            Consumption points for interpolation.

        Returns
        -------
        cFuncUnc : CubicInterp
            The unconstrained consumption function for this period.
        """
        # Calculate the MPC at each gridpoint
        EndOfPrdvPP = (
            self.DiscFacEff
            * self.Rfree
            * self.Rfree
            * np.sum(
                self.vPPfuncNext(self.mLvlNext, self.pLvlNext) * self.ShkPrbs_temp,
                axis=0,
            )
        )
        dcda = EndOfPrdvPP / self.uPP(np.array(cLvl[1:, 1:]))
        MPC = dcda / (dcda + 1.0)
        MPC = np.concatenate((np.reshape(MPC[:, 0], (MPC.shape[0], 1)), MPC), axis=1)
        # Stick an extra MPC value at bottom; MPCmax doesn't work
        MPC = np.concatenate(
            (self.MPCminNow * np.ones((1, self.aXtraGrid.size + 1)), MPC), axis=0
        )

        # Make cubic consumption function with respect to mLvl for each persistent income level
        cFunc_by_pLvl_list = []  # list of consumption functions for each pLvl
        for j in range(pLvl.shape[0]):
            pLvl_j = pLvl[j, 0]
            m_temp = mLvl[j, :] - self.BoroCnstNat(pLvl_j)
            c_temp = cLvl[j, :]  # Make a cubic consumption function for this pLvl
            MPC_temp = MPC[j, :]
            if pLvl_j > 0:
                cFunc_by_pLvl_list.append(
                    CubicInterp(
                        m_temp,
                        c_temp,
                        MPC_temp,
                        lower_extrap=True,
                        slope_limit=self.MPCminNow,
                        intercept_limit=self.MPCminNow * self.hLvlNow(pLvl_j),
                    )
                )
            else:  # When pLvl=0, cFunc is linear
                cFunc_by_pLvl_list.append(
                    LinearInterp(m_temp, c_temp, lower_extrap=True)
                )
        pLvl_list = pLvl[:, 0]
        cFuncUncBase = LinearInterpOnInterp1D(
            cFunc_by_pLvl_list, pLvl_list
        )  # Combine all linear cFuncs
        cFuncUnc = VariableLowerBoundFunc2D(cFuncUncBase, self.BoroCnstNat)
        # Re-adjust for lower bound of natural borrowing constraint
        return cFuncUnc

    def add_MPC_and_human_wealth(self, solution):
        """
        Take a solution and add human wealth and the bounding MPCs to it.

        Parameters
        ----------
        solution : ConsumerSolution
            The solution to this period's consumption-saving problem.

        Returns:
        ----------
        solution : ConsumerSolution
            The solution to this period's consumption-saving problem, but now
            with human wealth and the bounding MPCs.
        """
        solution.hNrm = 0.0  # Can't have None or set_and_update_values breaks, should fix
        solution.hLvl = self.hLvlNow
        solution.mLvlMin = self.mLvlMinNow
        solution.MPCmin = self.MPCminNow
        solution.MPCmax = 0.0  # MPCmax is actually a function in this model
        return solution

    def add_vPPfunc(self, solution):
        """
        Adds the marginal marginal value function to an existing solution, so
        that the next solver can evaluate vPP and thus use cubic interpolation.

        Parameters
        ----------
        solution : ConsumerSolution
            The solution to this single period problem, which must include the
            consumption function.

        Returns
        -------
        solution : ConsumerSolution
            The same solution passed as input, but with the marginal marginal
            value function for this period added as the attribute vPPfunc.
        """
        vPPfuncNow = MargMargValueFuncCRRA(solution.cFunc, self.CRRA)
        solution.vPPfunc = vPPfuncNow
        return solution

    def solve(self):
        """
        Solves a one period consumption saving problem with risky income, with
        persistent income explicitly tracked as a state variable.

        Parameters
        ----------
        None

        Returns
        -------
        solution : ConsumerSolution
            The solution to the one period problem, including a consumption
            function (defined over market resources and persistent income), a
            marginal value function, bounding MPCs, and human wealth as a func-
            tion of persistent income.  Might also include a value function and
            marginal marginal value function, depending on options selected.
        """
        aLvl, pLvl = self.prepare_to_calc_EndOfPrdvP()
        EndOfPrdvP = self.calc_EndOfPrdvP()
        if self.vFuncBool:
            self.make_EndOfPrdvFunc(EndOfPrdvP)
        if self.CubicBool:
            interpolator = self.make_cubic_cFunc
        else:
            interpolator = self.make_linear_cFunc
        solution = self.make_basic_solution(EndOfPrdvP, aLvl, pLvl, interpolator)
        solution = self.add_MPC_and_human_wealth(solution)
        if self.vFuncBool:
            solution.vFunc = self.make_vFunc(solution)
        if self.CubicBool:
            solution = self.add_vPPfunc(solution)
        return solution


###############################################################################

# -----------------------------------------------------------------------------
# ----- Define additional parameters for the persistent shocks model ----------
# -----------------------------------------------------------------------------

pLvlPctiles = np.concatenate(
    (
        [0.001, 0.005, 0.01, 0.03],
        np.linspace(0.05, 0.95, num=19),
        [0.97, 0.99, 0.995, 0.999],
    )
)
PrstIncCorr = 0.98  # Serial correlation coefficient for permanent income

# Make a dictionary for the "explicit permanent income" idiosyncratic shocks model
init_explicit_perm_inc = init_idiosyncratic_shocks.copy()
init_explicit_perm_inc["pLvlPctiles"] = pLvlPctiles
init_explicit_perm_inc["PermGroFac"] = [
    1.0
]  # long run permanent income growth doesn't work yet
init_explicit_perm_inc["aXtraMax"] = 30
init_explicit_perm_inc["aXtraExtra"] = [0.005, 0.01]


class GenIncProcessConsumerType(IndShockConsumerType):
    """
    A consumer type with idiosyncratic shocks to persistent and transitory income.
    His problem is defined by a sequence of income distributions, survival prob-
    abilities, and persistent income growth functions, as well as time invariant
    values for risk aversion, discount factor, the interest rate, the grid of
    end-of-period assets, and an artificial borrowing constraint.

    See init_explicit_perm_inc for a dictionary of the
    keywords that should be passed to the constructor.

    Parameters
    ----------
    cycles : int
        Number of times the sequence of periods should be solved.
    """

    cFunc_terminal_ = BilinearInterp(
        np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0.0, 1.0]), np.array([0.0, 1.0])
    )
    solution_terminal_ = ConsumerSolution(
        cFunc=cFunc_terminal_, mNrmMin=0.0, hNrm=0.0, MPCmin=1.0, MPCmax=1.0
    )

    state_vars = ['pLvl',"mLvl",'aLvl']

    def __init__(self, **kwds):
        params = init_explicit_perm_inc.copy()
        params.update(kwds)

        # Initialize a basic ConsumerType
        IndShockConsumerType.__init__(self, **params)
        self.solve_one_period = make_one_period_oo_solver(ConsGenIncProcessSolver)

        # a poststate?
        self.state_now['aLvl'] = None
        self.state_prev['aLvl'] = None

        # better way to do this...
        self.state_now["mLvl"] = None
        self.state_prev["mLvl"] = None

    def pre_solve(self):
        #        AgentType.pre_solve()
        self.update_solution_terminal()

    def update(self):
        """
        Update the income process, the assets grid, the persistent income grid,
        and the terminal solution.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        IndShockConsumerType.update(self)
        self.update_pLvlNextFunc()
        self.update_pLvlGrid()

    def update_solution_terminal(self):
        """
        Update the terminal period solution.  This method should be run when a
        new AgentType is created or when CRRA changes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.solution_terminal.vFunc = ValueFuncCRRA(self.cFunc_terminal_, self.CRRA)
        self.solution_terminal.vPfunc = MargValueFuncCRRA(self.cFunc_terminal_, self.CRRA)
        self.solution_terminal.vPPfunc = MargMargValueFuncCRRA(
            self.cFunc_terminal_, self.CRRA
        )
        self.solution_terminal.hNrm = 0.0  # Don't track normalized human wealth
        self.solution_terminal.hLvl = lambda p: np.zeros_like(p)
        # But do track absolute human wealth by persistent income
        self.solution_terminal.mLvlMin = lambda p: np.zeros_like(p)
        # And minimum allowable market resources by perm inc

    def update_pLvlNextFunc(self):
        """
        A dummy method that creates a trivial pLvlNextFunc attribute that has
        no persistent income dynamics.  This method should be overwritten by
        subclasses in order to make (e.g.) an AR1 income process.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pLvlNextFuncBasic = LinearInterp(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        self.pLvlNextFunc = self.T_cycle * [pLvlNextFuncBasic]
        self.add_to_time_vary("pLvlNextFunc")

    def install_retirement_func(self):
        """
        Installs a special pLvlNextFunc representing retirement in the correct
        element of self.pLvlNextFunc.  Draws on the attributes T_retire and
        pLvlNextFuncRet.  If T_retire is zero or pLvlNextFuncRet does not
        exist, this method does nothing.  Should only be called from within the
        method update_pLvlNextFunc, which ensures that time is flowing forward.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if (not hasattr(self, "pLvlNextFuncRet")) or self.T_retire == 0:
            return
        t = self.T_retire
        self.pLvlNextFunc[t] = self.pLvlNextFuncRet

    def update_pLvlGrid(self):
        """
        Update the grid of persistent income levels.  Currently only works for
        infinite horizon models (cycles=0) and lifecycle models (cycles=1).  Not
        clear what to do about cycles>1 because the distribution of persistent
        income will be different within a period depending on how many cycles
        have elapsed.  This method uses a simulation approach to generate the
        pLvlGrid at each period of the cycle, drawing on the initial distribution
        of persistent income, the pLvlNextFuncs, and the attribute pLvlPctiles.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        LivPrbAll = np.array(self.LivPrb)

        # Simulate the distribution of persistent income levels by t_cycle in a lifecycle model
        if self.cycles == 1:
            pLvlNow = Lognormal(
                self.pLvlInitMean, sigma=self.pLvlInitStd, seed=31382
            ).draw(self.AgentCount)
            pLvlGrid = []  # empty list of time-varying persistent income grids
            # Calculate distribution of persistent income in each period of lifecycle
            for t in range(len(self.PermShkStd)):
                if t > 0:
                    PermShkNow = self.PermShkDstn[t - 1].draw(N=self.AgentCount)
                    pLvlNow = self.pLvlNextFunc[t - 1](pLvlNow) * PermShkNow
                pLvlGrid.append(get_percentiles(pLvlNow, percentiles=self.pLvlPctiles))

        # Calculate "stationary" distribution in infinite horizon (might vary across periods of cycle)
        elif self.cycles == 0:
            T_long = 1000  # Number of periods to simulate to get to "stationary" distribution
            pLvlNow = Lognormal(
                mu=self.pLvlInitMean, sigma=self.pLvlInitStd, seed=31382
            ).draw(self.AgentCount)
            t_cycle = np.zeros(self.AgentCount, dtype=int)
            for t in range(T_long):
                LivPrb = LivPrbAll[
                    t_cycle
                ]  # Determine who dies and replace them with newborns
                draws = Uniform(seed=t).draw(self.AgentCount)
                who_dies = draws > LivPrb
                pLvlNow[who_dies] = Lognormal(
                    self.pLvlInitMean, self.pLvlInitStd, seed=t + 92615
                ).draw(np.sum(who_dies))
                t_cycle[who_dies] = 0

                for j in range(self.T_cycle):  # Update persistent income
                    these = t_cycle == j
                    PermShkTemp = self.PermShkDstn[j].draw(N=np.sum(these))
                    pLvlNow[these] = self.pLvlNextFunc[j](pLvlNow[these]) * PermShkTemp
                t_cycle = t_cycle + 1
                t_cycle[t_cycle == self.T_cycle] = 0

            # We now have a "long run stationary distribution", extract percentiles
            pLvlGrid = []  # empty list of time-varying persistent income grids
            for t in range(self.T_cycle):
                these = t_cycle == t
                pLvlGrid.append(
                    get_percentiles(pLvlNow[these], percentiles=self.pLvlPctiles)
                )

        # Throw an error if cycles>1
        else:
            assert False, "Can only handle cycles=0 or cycles=1!"

        # Store the result and add attribute to time_vary
        self.pLvlGrid = pLvlGrid
        self.add_to_time_vary("pLvlGrid")

    def sim_birth(self, which_agents):
        """
        Makes new consumers for the given indices.  Initialized variables include aNrm and pLvl, as
        well as time variables t_age and t_cycle.  Normalized assets and persistent income levels
        are drawn from lognormal distributions given by aNrmInitMean and aNrmInitStd (etc).

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        """
        # Get and store states for newly born agents
        N = np.sum(which_agents)  # Number of new consumers to make
        aNrmNow_new = Lognormal(
            self.aNrmInitMean, self.aNrmInitStd, seed=self.RNG.randint(0, 2 ** 31 - 1)
        ).draw(N)
        self.state_now['pLvl'][which_agents] = Lognormal(
            self.pLvlInitMean, self.pLvlInitStd, seed=self.RNG.randint(0, 2 ** 31 - 1)
        ).draw(N)
        self.state_now['aLvl'][which_agents] = aNrmNow_new * self.state_now['pLvl'][which_agents]
        self.t_age[which_agents] = 0  # How many periods since each agent was born
        self.t_cycle[
            which_agents
        ] = 0  # Which period of the cycle each agent is currently in

    def transition(self):
        """
        Calculates updated values of normalized market resources
        and persistent income level for each
        agent.  Uses pLvlNow, aLvlNow, PermShkNow, TranShkNow.

        Parameters
        ----------
        None

        Returns
        -------
        pLvlNow
        mLvlNow
        """
        aLvlPrev = self.state_prev['aLvl']
        RfreeNow = self.get_Rfree()

        # Calculate new states: normalized market resources
        # and persistent income level
        pLvlNow = np.zeros_like(aLvlPrev)

        for t in range(self.T_cycle):
            these = t == self.t_cycle
            pLvlNow[these] = (
                self.pLvlNextFunc[t - 1](self.state_prev['pLvl'][these])
                * self.shocks['PermShk'][these]
            )

        #state value
        bLvlNow = RfreeNow * aLvlPrev  # Bank balances before labor income

        # Market resources after income - state value
        mLvlNow = bLvlNow + \
                  self.shocks['TranShk'] * \
                  pLvlNow

        return (pLvlNow,
                mLvlNow)


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
        cLvlNow = np.zeros(self.AgentCount) + np.nan
        MPCnow = np.zeros(self.AgentCount) + np.nan

        for t in range(self.T_cycle):
            these = t == self.t_cycle
            cLvlNow[these] = self.solution[t].cFunc(
                self.state_now["mLvl"][these], self.state_now['pLvl'][these]
            )
            MPCnow[these] = self.solution[t].cFunc.derivativeX(
                self.state_now["mLvl"][these], self.state_now['pLvl'][these]
            )
        self.controls["cLvl"] = cLvlNow
        self.MPCnow = MPCnow

    def get_poststates(self):
        """
        Calculates end-of-period assets for each consumer of this type.
        Identical to version in IndShockConsumerType but uses Lvl rather than Nrm variables.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.state_now['aLvl'] = self.state_now["mLvl"] - self.controls["cLvl"]
        # moves now to prev
        AgentType.get_poststates(self)


###############################################################################


class IndShockExplicitPermIncConsumerType(GenIncProcessConsumerType):
    """
    A consumer type with idiosyncratic shocks to permanent and transitory income.
    The problem is defined by a sequence of income distributions, survival prob-
    abilities, and permanent income growth rates, as well as time invariant values
    for risk aversion, discount factor, the interest rate, the grid of end-of-
    period assets, and an artificial borrowing constraint.  This agent type is
    identical to an IndShockConsumerType but for explicitly tracking pLvl as a
    state variable during solution.  There is no real economic use for it.
    """

    def update_pLvlNextFunc(self):
        """
        A method that creates the pLvlNextFunc attribute as a sequence of
        linear functions, indicating constant expected permanent income growth
        across permanent income levels.  Draws on the attribute PermGroFac, and
        installs a special retirement function when it exists.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pLvlNextFunc = []
        for t in range(self.T_cycle):
            pLvlNextFunc.append(
                LinearInterp(np.array([0.0, 1.0]), np.array([0.0, self.PermGroFac[t]]))
            )

        self.pLvlNextFunc = pLvlNextFunc
        self.add_to_time_vary("pLvlNextFunc")


###############################################################################


# Make a dictionary for the "persistent idiosyncratic shocks" model
init_persistent_shocks = init_explicit_perm_inc.copy()
init_persistent_shocks["PrstIncCorr"] = PrstIncCorr


class PersistentShockConsumerType(GenIncProcessConsumerType):
    """
    Type with idiosyncratic shocks to persistent ('Prst') and transitory income.
    The problem is defined by a sequence of income distributions, survival prob-
    abilities, and persistent income growth rates, as well as time invariant values
    for risk aversion, discount factor, the interest rate, the grid of end-of-
    period assets, an artificial borrowing constraint, and the AR1 correlation
    coefficient for (log) persistent income.

    Parameters
    ----------

    """

    def __init__(self, **kwds):
        params = init_persistent_shocks.copy()
        params.update(kwds)

        GenIncProcessConsumerType.__init__(self, **params)

    def update_pLvlNextFunc(self):
        """
        A method that creates the pLvlNextFunc attribute as a sequence of
        AR1-style functions.  Draws on the attributes PermGroFac and PrstIncCorr.
        If cycles=0, the product of PermGroFac across all periods must be 1.0,
        otherwise this method is invalid.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        pLvlNextFunc = []
        pLogMean = self.pLvlInitMean  # Initial mean (log) persistent income

        for t in range(self.T_cycle):
            pLvlNextFunc.append(
                pLvlFuncAR1(pLogMean, self.PermGroFac[t], self.PrstIncCorr)
            )
            pLogMean += np.log(self.PermGroFac[t])

        self.pLvlNextFunc = pLvlNextFunc
        self.add_to_time_vary("pLvlNextFunc")
