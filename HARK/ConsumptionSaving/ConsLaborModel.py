"""
Subclasses of AgentType representing consumers who make decisions about how much
labor to supply, as well as a consumption-saving decision.

It currently only has
model A: labor supply on the intensive margin (unit interval) with CRRA utility
from a composite good (of consumption and leisure), with transitory and permanent
productivity shocks.  Agents choose their quantities of labor and consumption after
observing both of these shocks, so the transitory shock is a state variable.

model B: labor supply on the extensive margin (whether to work) with non-separable
CRRA utility from consumption and disutility from job search, with transitory and
permanent productivity shocks. Agent is subject to an exogenous probability of being
fired if he is currently employed. If he is unemployed, he need to decide how much
effort he will exert to be re-employed, which induces disutility.
"""
import sys

from copy import copy, deepcopy
import numpy as np
from scipy.optimize import brentq
from HARK.core import Solution, AgentType
from HARK.utilities import CRRAutilityP, CRRAutilityP_inv, CRRAutility, CRRAutility_inv
from HARK.interpolation import LinearInterp, LinearInterpOnInterp1D, VariableLowerBoundFunc2D, BilinearInterp, \
    ConstantFunction, LowerEnvelope
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType, MargValueFunc, init_idiosyncratic_shocks, \
    ValueFunc
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK.ConsumptionSaving.ConsGenIncProcessModel import ValueFunc2D, MargValueFunc2D
from HARK.distribution import DiscreteDistribution, combineIndepDstns
import matplotlib.pyplot as plt


class ConsumerLaborSolution(HARKobject):
    """
    A class for representing one period of the solution to a Consumer Labor problem.
    """

    distance_criteria = ["cFunc", "LbrFunc"]

    def __init__(self, cFunc=None, LbrFunc=None, vFunc=None, vPfunc=None, bNrmMin=None):
        """
        The constructor for a new ConsumerSolution object.

        Parameters
        ----------
        cFunc : function
            The consumption function for this period, defined over normalized
            bank balances and the transitory productivity shock: cNrm = cFunc(bNrm,TranShk).
        LbrFunc : function
            The labor supply function for this period, defined over normalized
            bank balances 0.751784276198: Lbr = LbrFunc(bNrm,TranShk).
        vFunc : function
            The beginning-of-period value function for this period, defined over
            normalized bank balances 0.751784276198: v = vFunc(bNrm,TranShk).
        vPfunc : function
            The beginning-of-period marginal value (of bank balances) function for
            this period, defined over normalized bank balances 0.751784276198: vP = vPfunc(bNrm,TranShk).
        bNrmMin: float
            The minimum allowable bank balances for this period, as a function of
            the transitory shock. cFunc, LbrFunc, etc are undefined for bNrm < bNrmMin(TranShk).

        Returns
        -------
        None
        """
        if cFunc is not None:
            self.cFunc = cFunc
        if LbrFunc is not None:
            self.LbrFunc = LbrFunc
        if vFunc is not None:
            self.vFunc = vFunc
        if vPfunc is not None:
            self.vPfunc = vPfunc
        if bNrmMin is not None:
            self.bNrmMin = bNrmMin


def solveConsLaborIntMarg(
    solution_next,
    PermShkDstn,
    TranShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rfree,
    PermGroFac,
    BoroCnstArt,
    aXtraGrid,
    TranShkGrid,
    vFuncBool,
    CubicBool,
    WageRte,
    LbrCost,
):
    """
    Solves one period of the consumption-saving model with endogenous labor supply
    on the intensive margin by using the endogenous grid method to invert the first
    order conditions for optimal composite consumption and between consumption and
    leisure, obviating any search for optimal controls.

    Parameters
    ----------
    solution_next : ConsumerLaborSolution
        The solution to the next period's problem; must have the attributes
        vPfunc and bNrmMinFunc representing marginal value of bank balances and
        minimum (normalized) bank balances as a function of the transitory shock.
    PermShkDstn: [np.array]
        Discrete distribution of permanent productivity shocks.
    TranShkDstn: [np.array]
        Discrete distribution of transitory productivity shocks.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor.
    CRRA : float
        Coefficient of relative risk aversion over the composite good.
    Rfree : float
        Risk free interest rate on assets retained at the end of the period.
    PermGroFac : float
        Expected permanent income growth factor for next period.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  Currently not handled, must be None.
    aXtraGrid: np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    TranShkGrid: np.array
        Grid of transitory shock values to use as a state grid for interpolation.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.  Not yet handled, must be False.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear interpolation.
        Cubic interpolation is not yet handled, must be False.
    WageRte: float
        Wage rate per unit of labor supplied.
    LbrCost: float
        Cost parameter for supplying labor: u_t = U(x_t), x_t = c_t*z_t^LbrCost,
        where z_t is leisure = 1 - Lbr_t.

    Returns
    -------
    solution_now : ConsumerLaborSolution
        The solution to this period's problem, including a consumption function
        cFunc, a labor supply function LbrFunc, and a marginal value function vPfunc;
        each are defined over normalized bank balances and transitory prod shock.
        Also includes bNrmMinNow, the minimum permissible bank balances as a function
        of the transitory productivity shock.
    """
    # Make sure the inputs for this period are valid: CRRA > LbrCost/(1+LbrCost)
    # and CubicBool = False.  CRRA condition is met automatically when CRRA >= 1.
    frac = 1.0 / (1.0 + LbrCost)
    if CRRA <= frac * LbrCost:
        print(
            "Error: make sure CRRA coefficient is strictly greater than alpha/(1+alpha)."
        )
        sys.exit()
    if BoroCnstArt is not None:
        print("Error: Model cannot handle artificial borrowing constraint yet. ")
        sys.exit()
    if vFuncBool or CubicBool is True:
        print("Error: Model cannot handle cubic interpolation yet.")
        sys.exit()

    # Unpack next period's solution and the productivity shock distribution, and define the inverse (marginal) utilty function
    vPfunc_next = solution_next.vPfunc
    TranShkPrbs = TranShkDstn.pmf
    TranShkVals = TranShkDstn.X
    PermShkPrbs = PermShkDstn.pmf
    PermShkVals = PermShkDstn.X
    TranShkCount = TranShkPrbs.size
    PermShkCount = PermShkPrbs.size
    uPinv = lambda X: CRRAutilityP_inv(X, gam=CRRA)

    # Make tiled versions of the grid of a_t values and the components of the shock distribution
    aXtraCount = aXtraGrid.size
    bNrmGrid = aXtraGrid  # Next period's bank balances before labor income

    # Replicated axtraGrid of b_t values (bNowGrid) for each transitory (productivity) shock
    bNrmGrid_rep = np.tile(np.reshape(bNrmGrid, (aXtraCount, 1)), (1, TranShkCount))

    # Replicated transitory shock values for each a_t state
    TranShkVals_rep = np.tile(np.reshape(TranShkVals, (1, TranShkCount)), (aXtraCount, 1))

    # Replicated transitory shock probabilities for each a_t state
    TranShkPrbs_rep = np.tile(np.reshape(TranShkPrbs, (1, TranShkCount)), (aXtraCount, 1))

    # Construct a function that gives marginal value of next period's bank balances *just before* the transitory shock arrives
    # Next period's marginal value at every transitory shock and every bank balances gridpoint
    vPNext = vPfunc_next(bNrmGrid_rep, TranShkVals_rep)

    # Integrate out the transitory shocks (in TranShkVals direction) to get expected vP just before the transitory shock
    vPbarNext = np.sum(vPNext * TranShkPrbs_rep, axis=1)

    # Transformed marginal value through the inverse marginal utility function to "decurve" it
    vPbarNvrsNext = uPinv(vPbarNext)

    # Linear interpolation over b_{t+1}, adding a point at minimal value of b = 0.
    vPbarNvrsFuncNext = LinearInterp(np.insert(bNrmGrid, 0, 0.0), np.insert(vPbarNvrsNext, 0, 0.0))

    # "Recurve" the intermediate marginal value function through the marginal utility function
    vPbarFuncNext = MargValueFunc(vPbarNvrsFuncNext, CRRA)

    # Get next period's bank balances at each permanent shock from each end-of-period asset values
    # Replicated grid of a_t values for each permanent (productivity) shock
    aNrmGrid_rep = np.tile(np.reshape(aXtraGrid, (aXtraCount, 1)), (1, PermShkCount))

    # Replicated permanent shock values for each a_t value
    PermShkVals_rep = np.tile(np.reshape(PermShkVals, (1, PermShkCount)), (aXtraCount, 1))

    # Replicated permanent shock probabilities for each a_t value
    PermShkPrbs_rep = np.tile(np.reshape(PermShkPrbs, (1, PermShkCount)), (aXtraCount, 1))
    bNrmNext = (Rfree / (PermGroFac * PermShkVals_rep)) * aNrmGrid_rep

    # Calculate marginal value of end-of-period assets at each a_t gridpoint
    # Get marginal value of bank balances next period at each shock
    vPbarNext = (PermGroFac * PermShkVals_rep) ** (-CRRA) * vPbarFuncNext(bNrmNext)

    # Take expectation across permanent income shocks
    EndOfPrdvP = (DiscFac * Rfree * LivPrb * np.sum(vPbarNext * PermShkPrbs_rep,
                                                    axis=1, keepdims=True))

    # Compute scaling factor for each transitory shock
    TranShkScaleFac_temp = (frac * (WageRte * TranShkGrid) ** (LbrCost * frac)
        * (LbrCost ** (-LbrCost * frac) + LbrCost ** frac ))

    # Flip it to be a row vector
    TranShkScaleFac = np.reshape(TranShkScaleFac_temp, (1, TranShkGrid.size))

    # Use the first order condition to compute an array of "composite good" x_t values corresponding to (a_t,theta_t) values
    xNow = (np.dot(EndOfPrdvP, TranShkScaleFac)) ** (-1.0 / (CRRA - LbrCost * frac))

    # Transform the composite good x_t values into consumption c_t and leisure z_t values
    TranShkGrid_rep = np.tile(np.reshape(TranShkGrid, (1, TranShkGrid.size)), (aXtraCount, 1))
    xNowPow = xNow ** frac  # Will use this object multiple times in math below

     # Find optimal consumption from optimal composite good
    cNrmNow = (((WageRte * TranShkGrid_rep) / LbrCost) ** (LbrCost * frac)) * xNowPow

    # Find optimal leisure from optimal composite good
    LsrNow = (LbrCost / (WageRte * TranShkGrid_rep)) ** frac * xNowPow

    # The zero-th transitory shock is TranShk=0, and the solution is to not work: Lsr = 1, Lbr = 0.
    cNrmNow[:, 0] = uPinv(EndOfPrdvP.flatten())
    LsrNow[:, 0] = 1.0

    # Agent cannot choose to work a negative amount of time. When this occurs, set
    # leisure to one and recompute consumption using simplified first order condition.
    # Find where labor would be negative if unconstrained
    violates_labor_constraint = (LsrNow > 1.0)
    EndOfPrdvP_temp = np.tile(np.reshape(EndOfPrdvP, (aXtraCount, 1)), (1, TranShkCount))
    cNrmNow[violates_labor_constraint] = uPinv(EndOfPrdvP_temp[violates_labor_constraint])
    LsrNow[violates_labor_constraint] = 1.0  # Set up z=1, upper limit

    # Calculate the endogenous bNrm states by inverting the within-period transition
    aNrmNow_rep = np.tile(np.reshape(aXtraGrid, (aXtraCount, 1)), (1, TranShkGrid.size))
    bNrmNow = aNrmNow_rep - WageRte * TranShkGrid_rep + cNrmNow + WageRte * TranShkGrid_rep * LsrNow

    # Add an extra gridpoint at the absolute minimal valid value for b_t for each TranShk;
    # this corresponds to working 100% of the time and consuming nothing.
    bNowArray = np.concatenate(
        (np.reshape(-WageRte * TranShkGrid, (1, TranShkGrid.size)), bNrmNow), axis=0
    )
    # Consume nothing
    cNowArray = np.concatenate((np.zeros((1, TranShkGrid.size)), cNrmNow), axis=0)
    # And no leisure!
    LsrNowArray = np.concatenate((np.zeros((1, TranShkGrid.size)), LsrNow), axis=0)
    LsrNowArray[0, 0] = 1.0  # Don't work at all if TranShk=0, even if bNrm=0
    LbrNowArray = 1.0 - LsrNowArray  # Labor is the complement of leisure

    # Get (pseudo-inverse) marginal value of bank balances using end of period
    # marginal value of assets (envelope condition), adding a column of zeros
    # zeros on the left edge, representing the limit at the minimum value of b_t.
    vPnvrsNowArray = np.concatenate((np.zeros((1, TranShkGrid.size)), uPinv(EndOfPrdvP_temp)))

    # Construct consumption and marginal value functions for this period
    bNrmMinNow = LinearInterp(TranShkGrid, bNowArray[0, :])

    # Loop over each transitory shock and make a linear interpolation to get lists
    # of optimal consumption, labor and (pseudo-inverse) marginal value by TranShk
    cFuncNow_list = []
    LbrFuncNow_list = []
    vPnvrsFuncNow_list = []
    for j in range(TranShkGrid.size):
        # Adjust bNrmNow for this transitory shock, so bNrmNow_temp[0] = 0
        bNrmNow_temp = (bNowArray[:, j] - bNowArray[0, j])

        # Make consumption function for this transitory shock
        cFuncNow_list.append(LinearInterp(bNrmNow_temp, cNowArray[:, j]))

        # Make labor function for this transitory shock
        LbrFuncNow_list.append(LinearInterp(bNrmNow_temp, LbrNowArray[:, j]))

        # Make pseudo-inverse marginal value function for this transitory shock
        vPnvrsFuncNow_list.append(LinearInterp(bNrmNow_temp, vPnvrsNowArray[:, j]))

    # Make linear interpolation by combining the lists of consumption, labor and marginal value functions
    cFuncNowBase = LinearInterpOnInterp1D(cFuncNow_list, TranShkGrid)
    LbrFuncNowBase = LinearInterpOnInterp1D(LbrFuncNow_list, TranShkGrid)
    vPnvrsFuncNowBase = LinearInterpOnInterp1D(vPnvrsFuncNow_list, TranShkGrid)

    # Construct consumption, labor, pseudo-inverse marginal value functions with
    # bNrmMinNow as the lower bound.  This removes the adjustment in the loop above.
    cFuncNow = VariableLowerBoundFunc2D(cFuncNowBase, bNrmMinNow)
    LbrFuncNow = VariableLowerBoundFunc2D(LbrFuncNowBase, bNrmMinNow)
    vPnvrsFuncNow = VariableLowerBoundFunc2D(vPnvrsFuncNowBase, bNrmMinNow)

    # Construct the marginal value function by "recurving" its pseudo-inverse
    vPfuncNow = MargValueFunc2D(vPnvrsFuncNow, CRRA)

    # Make a solution object for this period and return it
    solution = ConsumerLaborSolution(cFunc=cFuncNow,
                                     LbrFunc=LbrFuncNow,
                                     vPfunc=vPfuncNow,
                                     bNrmMin=bNrmMinNow
                                     )
    return solution


class LaborIntMargConsumerType(IndShockConsumerType):

    """
    A class representing agents who make a decision each period about how much
    to consume vs save and how much labor to supply (as a fraction of their time).
    They get CRRA utility from a composite good x_t = c_t*z_t^alpha, and discount
    future utility flows at a constant factor.
    """

    time_vary_ = copy(IndShockConsumerType.time_vary_)
    time_vary_ += ["WageRte"]
    time_inv_ = copy(IndShockConsumerType.time_inv_)

    def __init__(self, cycles=1, **kwds):
        """
        Instantiate a new consumer type with given data.
        See init_labor_intensive for a dictionary of
        the keywords that should be passed to the constructor.

        Parameters
        ----------
        cycles : int
            Number of times the sequence of periods should be solved.

        Returns
        -------
        None
        """
        params = init_labor_intensive.copy()
        params.update(kwds)

        IndShockConsumerType.__init__(self, cycles=cycles, **params)
        self.pseudo_terminal = False
        self.solveOnePeriod = solveConsLaborIntMarg
        self.update()

    def update(self):
        """
        Update the income process, the assets grid, and the terminal solution.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.updateIncomeProcess()
        self.updateAssetsGrid()
        self.updateTranShkGrid()
        self.updateLbrCost()

    def updateLbrCost(self):
        """
        Construct the age-varying cost of working LbrCost using the attribute LbrCostCoeffs.
        This attribute should be a 1D array of arbitrary length, representing polynomial
        coefficients (over t_cycle) for (log) LbrCost.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        Coeffs = self.LbrCostCoeffs
        N = len(Coeffs)
        age_vec = np.arange(self.T_cycle)
        LbrCostBase = np.zeros(self.T_cycle)
        for n in range(N):
            LbrCostBase += Coeffs[n] * age_vec ** n
        LbrCost = np.exp(LbrCostBase)
        self.LbrCost = LbrCost.tolist()
        self.addToTimeVary("LbrCost")

    def calcBoundingValues(self):
        """
        Calculate human wealth plus minimum and maximum MPC in an infinite
        horizon model with only one period repeated indefinitely.  Store results
        as attributes of self.  Human wealth is the present discounted value of
        expected future income after receiving income this period, ignoring mort-
        ality.  The maximum MPC is the limit of the MPC as m --> mNrmMin.  The
        minimum MPC is the limit of the MPC as m --> infty.

        NOT YET IMPLEMENTED FOR THIS CLASS

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        raise NotImplementedError()

    def makeEulerErrorFunc(self, mMax=100, approx_inc_dstn=True):
        """
        Creates a "normalized Euler error" function for this instance, mapping
        from market resources to "consumption error per dollar of consumption."
        Stores result in attribute eulerErrorFunc as an interpolated function.
        Has option to use approximate income distribution stored in self.IncomeDstn
        or to use a (temporary) very dense approximation.

        NOT YET IMPLEMENTED FOR THIS CLASS

        Parameters
        ----------
        mMax : float
            Maximum normalized market resources for the Euler error function.
        approx_inc_dstn : Boolean
            Indicator for whether to use the approximate discrete income distri-
            bution stored in self.IncomeDstn[0], or to use a very accurate
            discrete approximation instead.  When True, uses approximation in
            IncomeDstn; when False, makes and uses a very dense approximation.

        Returns
        -------
        None
        """
        raise NotImplementedError()

    def getStates(self):
        """
        Calculates updated values of normalized bank balances and permanent income
        level for each agent.  Uses pLvlNow, aNrmNow, PermShkNow.  Calls the getStates
        method for the parent class, then erases mNrmNow, which cannot be calculated
        until after getControls in this model.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        IndShockConsumerType.getStates(self)
        self.mNrmNow[:] = np.nan  # Delete market resource calculation

    def getControls(self):
        """
        Calculates consumption and labor supply for each consumer of this type
        using the consumption and labor functions in each period of the cycle.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        MPCnow = np.zeros(self.AgentCount) + np.nan
        LbrNow = np.zeros(self.AgentCount) + np.nan
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            cNrmNow[these] = self.solution[t].cFunc(
                self.bNrmNow[these], self.shocks['TranShkNow'][these]
            )  # Assign consumption values
            MPCnow[these] = self.solution[t].cFunc.derivativeX(
                self.bNrmNow[these], self.shocks['TranShkNow'][these]
            )  # Assign marginal propensity to consume values (derivative)
            LbrNow[these] = self.solution[t].LbrFunc(
                self.bNrmNow[these], self.shocks['TranShkNow'][these]
            )  # Assign labor supply
        self.cNrmNow = cNrmNow
        self.MPCnow = MPCnow
        self.LbrNow = LbrNow

    def getPostStates(self):
        """
        Calculates end-of-period assets for each consumer of this type.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        mNrmNow = np.zeros(self.AgentCount) + np.nan
        aNrmNow = np.zeros(self.AgentCount) + np.nan
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            mNrmNow[these] = (
                self.bNrmNow[these] + self.LbrNow[these] * self.shocks['TranShkNow'][these]
            )  # mNrm = bNrm + yNrm
            aNrmNow[these] = mNrmNow[these] - self.cNrmNow[these]  # aNrm = mNrm - cNrm
        self.mNrmNow = mNrmNow
        self.aNrmNow = aNrmNow

    def updateTranShkGrid(self):
        """
        Create a time-varying list of arrays for TranShkGrid using TranShkDstn,
        which is created by the method updateIncomeProcess().  Simply takes the
        transitory shock values and uses them as a state grid; can be improved.
        Creates the attribute TranShkGrid.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        TranShkGrid = []  # Create an empty list for TranShkGrid that will be updated
        for t in range(self.T_cycle):
            TranShkGrid.append(
                self.TranShkDstn[t].X
            )  # Update/ Extend the list of TranShkGrid with the TranShkVals for each TranShkPrbs
        self.TranShkGrid = TranShkGrid  # Save that list in self (time-varying)
        self.addToTimeVary(
            "TranShkGrid"
        )  # Run the method addToTimeVary from AgentType to add TranShkGrid as one parameter of time_vary list

    def updateSolutionTerminal(self):
        """
        Updates the terminal period solution and solves for optimal consumption
        and labor when there is no future.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        t = -1
        TranShkGrid = self.TranShkGrid[t]
        LbrCost = self.LbrCost[t]
        WageRte = self.WageRte[t]

        bNrmGrid = np.insert(
            self.aXtraGrid, 0, 0.0
        )  # Add a point at b_t = 0 to make sure that bNrmGrid goes down to 0
        bNrmCount = bNrmGrid.size  # 201
        TranShkCount = TranShkGrid.size  # = (7,)
        bNrmGridTerm = np.tile(
            np.reshape(bNrmGrid, (bNrmCount, 1)), (1, TranShkCount)
        )  # Replicated bNrmGrid for each transitory shock theta_t
        TranShkGridTerm = np.tile(
            TranShkGrid, (bNrmCount, 1)
        )  # Tile the grid of transitory shocks for the terminal solution. (201,7)

        # Array of labor (leisure) values for terminal solution
        LsrTerm = np.minimum(
            (LbrCost / (1.0 + LbrCost))
            * (bNrmGridTerm / (WageRte * TranShkGridTerm) + 1.0),
            1.0,
        )
        LsrTerm[0, 0] = 1.0
        LbrTerm = 1.0 - LsrTerm

        # Calculate market resources in terminal period, which is consumption
        mNrmTerm = bNrmGridTerm + LbrTerm * WageRte * TranShkGridTerm
        cNrmTerm = mNrmTerm  # Consume everything we have

        # Make a bilinear interpolation to represent the labor and consumption functions
        LbrFunc_terminal = BilinearInterp(LbrTerm, bNrmGrid, TranShkGrid)
        cFunc_terminal = BilinearInterp(cNrmTerm, bNrmGrid, TranShkGrid)

        # Compute the effective consumption value using consumption value and labor value at the terminal solution
        xEffTerm = LsrTerm ** LbrCost * cNrmTerm
        vNvrsFunc_terminal = BilinearInterp(xEffTerm, bNrmGrid, TranShkGrid)
        vFunc_terminal = ValueFunc2D(vNvrsFunc_terminal, self.CRRA)

        # Using the envelope condition at the terminal solution to estimate the marginal value function
        vPterm = LsrTerm ** LbrCost * CRRAutilityP(xEffTerm, gam=self.CRRA)
        vPnvrsTerm = CRRAutilityP_inv(
            vPterm, gam=self.CRRA
        )  # Evaluate the inverse of the CRRA marginal utility function at a given marginal value, vP

        vPnvrsFunc_terminal = BilinearInterp(vPnvrsTerm, bNrmGrid, TranShkGrid)
        vPfunc_terminal = MargValueFunc2D(
            vPnvrsFunc_terminal, self.CRRA
        )  # Get the Marginal Value function

        bNrmMin_terminal = ConstantFunction(
            0.0
        )  # Trivial function that return the same real output for any input

        self.solution_terminal = ConsumerLaborSolution(
            cFunc=cFunc_terminal,
            LbrFunc=LbrFunc_terminal,
            vFunc=vFunc_terminal,
            vPfunc=vPfunc_terminal,
            bNrmMin=bNrmMin_terminal,
        )

    def plotcFunc(self, t, bMin=None, bMax=None, ShkSet=None):
        """
        Plot the consumption function by bank balances at a given set of transitory shocks.

        Parameters
        ----------
        t : int
            Time index of the solution for which to plot the consumption function.
        bMin : float or None
            Minimum value of bNrm at which to begin the plot.  If None, defaults
            to the minimum allowable value of bNrm for each transitory shock.
        bMax : float or None
            Maximum value of bNrm at which to end the plot.  If None, defaults
            to bMin + 20.
        ShkSet : [float] or None
            Array or list of transitory shocks at which to plot the consumption
            function.  If None, defaults to the TranShkGrid for this time period.

        Returns
        -------
        None
        """
        if ShkSet is None:
            ShkSet = self.TranShkGrid[t]

        for j in range(len(ShkSet)):
            TranShk = ShkSet[j]
            if bMin is None:
                bMin_temp = self.solution[t].bNrmMin(TranShk)
            else:
                bMin_temp = bMin
            if bMax is None:
                bMax_temp = bMin_temp + 20.0
            else:
                bMax_temp = bMax

            B = np.linspace(bMin_temp, bMax_temp, 300)
            C = self.solution[t].cFunc(B, TranShk * np.ones_like(B))
            plt.plot(B, C)
        plt.xlabel("Beginning of period bank balances")
        plt.ylabel("Normalized consumption level")
        plt.show()

    def plotLbrFunc(self, t, bMin=None, bMax=None, ShkSet=None):
        """
        Plot the labor supply function by bank balances at a given set of transitory shocks.

        Parameters
        ----------
        t : int
            Time index of the solution for which to plot the labor supply function.
        bMin : float or None
            Minimum value of bNrm at which to begin the plot.  If None, defaults
            to the minimum allowable value of bNrm for each transitory shock.
        bMax : float or None
            Maximum value of bNrm at which to end the plot.  If None, defaults
            to bMin + 20.
        ShkSet : [float] or None
            Array or list of transitory shocks at which to plot the labor supply
            function.  If None, defaults to the TranShkGrid for this time period.

        Returns
        -------
        None
        """
        if ShkSet is None:
            ShkSet = self.TranShkGrid[t]

        for j in range(len(ShkSet)):
            TranShk = ShkSet[j]
            if bMin is None:
                bMin_temp = self.solution[t].bNrmMin(TranShk)
            else:
                bMin_temp = bMin
            if bMax is None:
                bMax_temp = bMin_temp + 20.0
            else:
                bMax_temp = bMax

            B = np.linspace(bMin_temp, bMax_temp, 300)
            L = self.solution[t].LbrFunc(B, TranShk * np.ones_like(B))
            plt.plot(B, L)
        plt.xlabel("Beginning of period bank balances")
        plt.ylabel("Labor supply")
        plt.show()


        
class LaborSearchConsumerType(MarkovConsumerType):
    '''
    A consumer type with idiosyncratic shocks to permanent and transitory income. His utility function
    is non-separable. And he is subject to an exogenous probability of being fired if he is currently
    employed. If he is unemployed, he need to decide how much effort he will exert to be re-employed.
    His problem is defined by a sequence of income distributions, survival probabilities, and permanent
    income growth rates, as well as time invariant values for risk aversion, discount factor, the interest
    rate, the grid of end-of-period assets, the exogenous unemployment probability and an artificial
    borrowing constraint.
    '''

    def __init__(self, cycles=1, **kwds):
        '''
        Instantiate a new consumer type with given data.

        Parameters
        ----------
        cycles : int
            Number of times the sequence of periods should be solved.

        Returns
        -------
        None
        '''

        MarkovConsumerType.__init__(self, cycles=cycles, **kwds)
        self.solveOnePeriod = self.solveConsLaborSearch

    def checkMarkovInputs(self):
        '''
        This overwrites a method of the parent class that would not work as intended in this model.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        pass

    def preSolve(self):
        '''
        Check to make sure that the inputs that are specific to LaborSearchConsumerType
        are of the right shape (if arrays) or length (if lists). If they are of right shape,
        then update the parameters used for solver function.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        AgentType.preSolve(self)
        self.update()

    def update(self):
        '''
        Update the income process, the assets grid, the terminal solution, and growth factor.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        MarkovConsumerType.update(self)
        self.updatePermGroFac()

    def updatePermGroFac(self):
        '''
        Construct the attribute PermGroFac as a time-varying. It should have the length of unemployment state + 1.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        PermGroFac = []
        for j in range(0, self.T_cycle):
            PermGroFac_by_age = np.asarray(self.PermGroFacUnemp)
            PermGroFac_by_age = np.insert(PermGroFac_by_age, 0, self.PermGroFacEmp[j])
            PermGroFac.append(PermGroFac_by_age)

        self.PermGroFac = deepcopy(PermGroFac)
        self.addToTimeVary('PermGroFac')

    def updateIncomeProcess(self):
        '''
        constructs the attributes IncomeDstn, PermShkDstn, and TranShkDstn using the primitive attributes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        IncUnemp_all = deepcopy(self.IncUnemp)
        UnempPrb = self.UnempPrb

        self.IncUnemp = [0.0 for _ in range(0, len(IncUnemp_all))]
        self.UnempPrb = 0.0

        IndShockConsumerType.updateIncomeProcess(self)
        IncomeDstn_emp = deepcopy(self.IncomeDstn)
        PermShkDstn_emp = deepcopy(self.PermShkDstn)
        TranShkDstn_emp = deepcopy(self.TranShkDstn)

        PermShkDstn_unemp = DiscreteDistribution(np.array([1.0]), np.array([1.0]))
        TranShkDstn_unemp = []
        IncomeDstn_by_unemp = []
        for n in range(0, self.Statecount):
            TranDiscDstn_tempobj = DiscreteDistribution(np.array([1.0]), np.array(IncUnemp_all[n]))
            TranShkDstn_unemp.append(TranDiscDstn_tempobj)
            IncomeDstn_tempobj = combineIndepDstns(PermShkDstn_unemp, TranDiscDstn_tempobj)
            IncomeDstn_by_unemp.append(IncomeDstn_tempobj)

        IncomeDstn = []
        TranShkDstn = []
        PermShkDstn = []

        for j in range(0, self.T_cycle):
            IncomeDstn_temp = [IncomeDstn_emp[j]]
            TranShkDstn_temp = [TranShkDstn_emp[j]]
            PermShkDstn_temp = [PermShkDstn_emp[j]]
            for incdstn, transdstn in zip(IncomeDstn_by_unemp, TranShkDstn_unemp):
                IncomeDstn_temp.append(incdstn)
                TranShkDstn_temp.append(transdstn)
                PermShkDstn_temp.append(PermShkDstn_unemp)

            IncomeDstn.append(IncomeDstn_temp)
            TranShkDstn.append(TranShkDstn_temp)
            PermShkDstn.append(PermShkDstn_temp)

        self.TranShkDstn = deepcopy(TranShkDstn)
        self.IncomeDstn = deepcopy(IncomeDstn)
        self.PermShkDstn = deepcopy(PermShkDstn)

        self.addToTimeVary('TranShkDstn', 'PermShkDstn', 'IncomeDstn')

        self.IncUnemp = IncUnemp_all
        self.UnempPrb = UnempPrb

    def solveConsLaborSearch(self, solution_next, IncomeDstn, LivPrb, DiscFac, CRRA, Rfree, PermGroFac, BoroCnstArt,
                             SearchCost, aXtraGrid, SepaRte):
        '''
        Solver function that solves for one period.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one-period problem.
        IncomeDstn: [[np.array]]
            A length N list of income distributions in each succeeding unemployment state.
            Each income distribution contains three arrays of floats,representing a discrete
            approximation to the income process at the beginning of the succeeding period.
            Order: event probabilities, permanent shocks, transitory shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of the next period.
        DiscFac : float
            Intertemporal discount factor for future utility.
        CRRA : float
            Coefficient of relative risk aversion.
        Rfree : float
            Risk free interest factor on end-of-period assets.
        PermGroFac : list
            Expected permanent income growth factor in each unemployment state.
        BoroCnstArt : float or None
            Artificial borrowing constraint, as a multiple of permanent income.
            Can be None, indicating no artificial constraint.
        SearchCost: float
            Cost of employment parameter at this age.
        SepaRte: float
            The job separation probability at this age.
        aXtraGrid: np.array
            Gridpoints represents “normalized assets above minimum”, at which
            the model will be solved by the endogenous grid method.

        Returns
        -------
        solution : ConsumerSolution
            The solution to this period's problem.
        '''

        # 1
        utilitySepa = lambda c: CRRAutility(c, CRRA)
        utility = lambda c, s: utilitySepa(c) * (1.0 - s ** SearchCost) ** (1.0 - CRRA)
        utilityPc = lambda c, s: CRRAutilityP(c, CRRA) * (1 - s ** SearchCost) ** (1.0 - CRRA)
        utility_inv = lambda u: CRRAutility_inv(u, CRRA)
        utilityP_inv = lambda uP: CRRAutilityP_inv(uP, CRRA)

        # 2
        EndOfPrdvFunc_by_state_next = []
        EndOfPrdvPfunc_by_state_next = []

        # 3
        BoroCnstNat = np.zeros(self.StateCount) + np.nan

        # 4
        for n in range(self.StateCount):
            # (a)
            # unpack "shock distributions" into different arrays for later use
            ShkProb = IncomeDstn[n].pmf
            ShkValPerm = IncomeDstn[n].X[0]
            ShkValTran = IncomeDstn[n].X[1]

            # (b)
            # make tiled version of shocks
            ShkPrbs_rep = np.tile(ShkProb, (aXtraGrid.size, 1))
            ShkValPerm_rep = np.tile(ShkValPerm, (aXtraGrid.size, 1))
            ShkValTran_rep = np.tile(ShkValTran, (aXtraGrid.size, 1))

            # (c)
            # calculate the minimum value of a_t given next period unemployment state and minimum shock realizations,
            # this serves as the natural borrowing constraint for this state.
            PermShkMinNext = np.min(ShkValPerm)
            TranShkMinNext = np.min(ShkValTran)
            BoroCnstNat[n] = (solution_next.mNrmMin[n] - TranShkMinNext) * (PermGroFac[n] * PermShkMinNext) / Rfree

            # (d)
            aNrmNowGrid = np.asarray(aXtraGrid) + BoroCnstNat[n]
            ShkCount = ShkValPerm.size
            aNrmNowGrid_rep = np.tile(aNrmNowGrid, (ShkCount, 1))
            aNrmNowGrid_rep = aNrmNowGrid_rep.transpose()

            # (e)
            # using equation on the top of page2 to calculate the next period m values.
            mNrmNext = Rfree / (PermGroFac[n] * ShkValPerm_rep) * aNrmNowGrid_rep + ShkValTran_rep

            # (f)
            # evaluate future realizations of value and marginal value at above m values.
            vFunc_next = solution_next.vFunc[n](mNrmNext)
            vPfunc_next = solution_next.vPfunc[n](mNrmNext)

            # (g)
            # calculate the pseudo (marginal) value function value
            EndOfPrdv = DiscFac * LivPrb * PermGroFac[n] ** (1.0 - CRRA) * \
                        np.sum(ShkValPerm_rep ** (1.0 - CRRA) * vFunc_next * ShkPrbs_rep, axis=1)
            EndOfPrdvP = DiscFac * LivPrb * PermGroFac[n] ** (-CRRA) * Rfree * \
                         np.sum(ShkValPerm_rep ** (-CRRA) * vPfunc_next * ShkPrbs_rep, axis=1)

            # (h)
            # evaluate the inverse pseudo (marginal) utility function
            EndOfPrdvNvrs_cond = utility_inv(EndOfPrdv)
            EndOfPrdvNvrs_cond = np.insert(EndOfPrdvNvrs_cond, 0, 0.0)
            EndOfPrdvPnvrs_cond = utilityP_inv(EndOfPrdvP)
            EndOfPrdvPnvrs_cond = np.insert(EndOfPrdvPnvrs_cond, 0, 0.0)

            # (i)
            aNrmNowGrid = np.insert(aNrmNowGrid, 0, BoroCnstNat[n])
            EndOfPrdvNvrsFunc_cond = LinearInterp(aNrmNowGrid, EndOfPrdvNvrs_cond)
            EndOfPrdvPnvrsFunc_cond = LinearInterp(aNrmNowGrid, EndOfPrdvPnvrs_cond)
            EndOfPrdvFunc_cond = ValueFunc(EndOfPrdvNvrsFunc_cond, CRRA)
            EndOfPrdvPfunc_cond = MargValueFunc(EndOfPrdvPnvrsFunc_cond, CRRA)

            # (j)
            EndOfPrdvFunc_by_state_next.append(EndOfPrdvFunc_cond)
            EndOfPrdvPfunc_by_state_next.append(EndOfPrdvPfunc_cond)

        # 5
        cFunc_by_state = []
        sFunc_by_state = []
        vFunc_by_state = []
        vPfunc_by_state = []

        # 6
        # (a)
        # calculate the minimum m value in this state
        mNrmMinNow = (BoroCnstArt if BoroCnstArt is not None and BoroCnstArt > BoroCnstNat[1] else BoroCnstNat[1])

        # (b)
        aNrmGrid_temp = aXtraGrid + BoroCnstNat[1]
        if BoroCnstArt is not None:
            sortInd = np.searchsorted(aNrmGrid_temp, BoroCnstArt)
            aNrmGrid_temp = np.insert(aNrmGrid_temp, sortInd, BoroCnstArt)

        EndOfPrdv_unemp = EndOfPrdvFunc_by_state_next[1](aNrmGrid_temp)
        EndOfPrdvP_unemp = EndOfPrdvPfunc_by_state_next[1](aNrmGrid_temp)

        # (c)
        EndOfPrdv_emp = EndOfPrdvFunc_by_state_next[0](aNrmGrid_temp)
        EndOfPrdvP_emp = EndOfPrdvPfunc_by_state_next[0](aNrmGrid_temp)

        # (e)
        cNrmGrid_temp = []
        sGrid_temp = []

        for vVal_n, vPval_n, vVal_0, vPval_0 in zip(EndOfPrdv_unemp, EndOfPrdvP_unemp, EndOfPrdv_emp, EndOfPrdvP_emp):
            cVal_FOCs = lambda s_t: ((vVal_0 - vVal_n) / (SearchCost * s_t ** (SearchCost - 1.0) *
                                                          (1.0 - s_t ** SearchCost) ** (-CRRA))) ** (1 / (1.0 - CRRA))
            cVal_FOCc = lambda s_t: ((s_t * vPval_0 + (1.0 - s_t) * vPval_n) /
                                     (1.0 - s_t ** SearchCost) ** (1.0 - CRRA)) ** (-1 / CRRA)
            cDiff_temp = lambda s_t: cVal_FOCc(s_t) - cVal_FOCs(s_t)
            sVal = brentq(cDiff_temp, 1e-6, 1.0 - 1e-6)
            cVal = ((sVal * vPval_0 + (1.0 - sVal[0]) * vPval_n) / (1.0 - sVal[0] ** SearchCost) ** (1.0 - CRRA)) ** \
                   (-1 / CRRA)
            cNrmGrid_temp.append(cVal)
            sGrid_temp.append(sVal)

        cNrmGrid_temp = np.asarray(cNrmGrid_temp)
        sGrid_temp = np.asarray(sGrid_temp)

        # (f)
        mNrmGrid_temp = aNrmGrid_temp + cNrmGrid_temp
        mNrmGrid_temp = np.insert(mNrmGrid_temp, 0, BoroCnstNat)
        cNrmGrid_temp = np.insert(cNrmGrid_temp, 0, 0.0)
        sGrid_temp = np.insert(sGrid_temp, 0, 0.0)

        cFuncUnc_at_this_state = LinearInterp(mNrmGrid_temp, cNrmGrid_temp)
        sFuncUnc_at_this_state = LinearInterp(mNrmGrid_temp, sGrid_temp)

        # (g)
        mCnst = np.array([mNrmMinNow[1], mNrmMinNow[1] + 1.0])
        cCnst = np.array([0.0, 1.0])
        cFuncCnst = LinearInterp(mCnst, cCnst)

        # (h)
        cFuncNow = LowerEnvelope(cFuncUnc_at_this_state, cFuncCnst)
        cFunc_by_state.append(cFuncNow)

        # (i)
        # the following steps are calculating the s-function when liquidity is constrained:
        # 1) Make a grid of m_t values on the open interval bounded below by BoroCnstArt and bounded above by the m_t
        #    value associated with \underline{a}; 10-20 is fine. Call it mNrmGrid_cnst.
        # 2) At each of those m_t values, the agent is liquidity constrained and will end the period with BoroCnstArt
        #    in assets, so calculate c_t = m_t – BoroCnstArt for this grid.
        # 3) Loop through the m_t values and solve (10) for s_t at each one, assuming a_t = \underline{a} and c_t
        #    is the value just calculated. Store these in sGrid_cnst. This is solving the first order condition
        #    for optimal s_t when knowing both c_t and a_t because the agent is liquidity constrained at this m_t.
        # 4) Make mNrmGrid_adj as a truncated version of mNrmGrid_temp, cutting off its lower values that are
        #    associated with a_t < BoroCnstArt. Do the same for sGrid_temp to make sGrid_adj.
        # 5) Concatenate mNrmGrid_cnst and mNrmGrid_adj, and concatenate sGrid_cnst with sGrid_adj.
        #    In this step, sticking together the constrained and unconstrained search functions.
        # 6) Insert a zero at the beginning of both those arrays, then make a LinearInterp; that’s sFunc and should
        #    be appended to sFunc_by_state.

        cEndPoint = cFunc_by_state[1](BoroCnstArt)
        mEndPoint = cEndPoint + BoroCnstArt
        mStartPoint = BoroCnstArt
        mNrmGrid_cnst = np.linspace(mStartPoint, mEndPoint, 20)

        cVal_cnst = mNrmGrid_cnst - BoroCnstArt
        vEmp_cnst = EndOfPrdvFunc_by_state_next[0](BoroCnstArt)
        vUnemp_cnst = EndOfPrdvFunc_by_state_next[1](BoroCnstArt)

        sGrid_cnst = []
        for c, vEmp, vUnemp in zip(cVal_cnst, vEmp_cnst, vUnemp_cnst):
            sFunc = lambda s_t: SearchCost * s_t ** (SearchCost - 1.0) * (1.0 - s_t ** SearchCost) ** (-CRRA) * c ** \
                                (1.0 - CRRA) - (vEmp - vUnemp)
            sGrid_cnst.append(brentq(sFunc, 1e-6, 1.0 - 1e-6))
        sGrid_cnst = np.asarray(sGrid_cnst)

        sortInd = np.searchsorted(aNrmGrid_temp, BoroCnstArt)
        mNrmGrid_adj = mNrmGrid_temp[sortInd:]
        sGrid_adj = sGrid_temp[sortInd:]

        mNrmGrid_cct = np.concatenate((mNrmGrid_cnst, mNrmGrid_adj), axis=None)
        sGrid_cct = np.concatenate((sGrid_cnst, sGrid_adj), axis=None)
        mNrmGrid_cct = np.insert(mNrmGrid_cct, 0, 0.0)
        sGrid_cct = np.insert(sGrid_cct, 0, 0.0)

        # (f)
        sFuncNow = LinearInterp(mNrmGrid_cct, sGrid_cct)
        sFunc_by_state.append(sFuncNow)

        # (k)
        # the following steps are calculating the marginal value function:
        # 1) Make a new grid of m_t values by adding mNrmMinNow to aXtraGrid,
        #    This starts *just above* the minimum permissible value of m_t this period.
        # 2) Evaluate the consumption and search functions at that grid of m_t values.
        # 3) Compute marginal utility of consumption (partial derivative of utility wrt c_t) at these values.
        #    That’s a marginal value grid.
        # 4) Run that through the inverse marginal utility function; that’s a pseudo-inverse marginal value grid.
        # 5) Prepend mNrmMinNow onto the beginning of the m_t grid, and prepend a zero onto
        #    the pseudo-inverse marginal value grid.
        # 6) Make a LinearInterp over m_t and pseudo-inverse marginal value;
        #    that’s the pseudo inverse marginal value function.
        # 7) Make a MargValueFunction using the pseudo-inverse marginal value function and CRRA.
        # 8) Stick that into the list of state-conditional marginal value function.s

        mNrmGrid_new = mNrmMinNow + aXtraGrid
        cValOnThisNewmGrid = cFunc_by_state[1](mNrmGrid_new)
        sValOnThisNewmGrid = sFunc_by_state[1](mNrmGrid_new)
        vPvalOnThisNewmGrid = utilityPc(cValOnThisNewmGrid, sValOnThisNewmGrid)

        vPvalNrvsOnThisNewmGrid = utility_inv(vPvalOnThisNewmGrid)
        mNrmGrid_new = np.insert(mNrmGrid_new, 0, mNrmMinNow)
        vPvalNrvsOnThisNewmGrid = np.insert(vPvalNrvsOnThisNewmGrid, 0, 0.0)
        vPfuncNow = LinearInterp(mNrmGrid_new, vPvalNrvsOnThisNewmGrid)
        vPfunc_by_state.append(MargValueFunc(vPfuncNow, CRRA))

        # (l)
        __mGrid = aXtraGrid + mNrmMinNow[1]
        cGridOnThisGrid = cFunc_by_state[1](__mGrid)
        sGridOnThisGrid = sFunc_by_state[1](__mGrid)
        aGridOnThisGrid = __mGrid - cGridOnThisGrid
        EndOfPrdvFuncUnempOnThisGrid = EndOfPrdvFunc_by_state_next[1](aGridOnThisGrid)
        EndOfPrdvFuncEmpOnThisGrid = EndOfPrdvFunc_by_state_next[0](aGridOnThisGrid)

        vFuncOnThisGrid = utility(cGridOnThisGrid, sGridOnThisGrid) + \
                          sGridOnThisGrid * EndOfPrdvFuncEmpOnThisGrid + \
                          (1.0 - sGridOnThisGrid) * EndOfPrdvFuncUnempOnThisGrid

        # (m)
        vFuncNvrsOnThisGrid = utility_inv(vFuncOnThisGrid)
        vFuncNvrsOnThisGrid = np.insert(vFuncNvrsOnThisGrid, 0, 0.0)
        __mGrid = np.insert(__mGrid, 0, mNrmMinNow[1])
        vFuncNrvs = LinearInterp(__mGrid, vFuncNvrsOnThisGrid)

        # (n)
        vFuncNow = ValueFunc(vFuncNrvs, CRRA)
        vFunc_by_state.append(vFuncNow)

        # 7
        # agent is employed
        mNrmMinNowEmp = (BoroCnstArt if BoroCnstArt is not None and BoroCnstArt > BoroCnstNat[0] else BoroCnstNat[0])

        __aNrmGrid = aXtraGrid + BoroCnstNat[0]
        EndOfPrdvPEmp = EndOfPrdvFunc_by_state_next[0](__aNrmGrid)
        EndOfPrdvPUnemp = EndOfPrdvPfunc_by_state_next[1](__aNrmGrid)

        __cNrmGrid = ((1.0 - SepaRte) * EndOfPrdvPEmp + SepaRte * EndOfPrdvPUnemp) ** (-1.0 / CRRA)
        mNrmGridEmp = __aNrmGrid + __cNrmGrid
        cFuncEmpUncnst = LinearInterp(mNrmGridEmp, __cNrmGrid)
        cFuncEmpCnst = LinearInterp(np.array([mNrmMinNowEmp, mNrmMinNowEmp + 1.0]), np.array([0.0, 1.0]))
        __cFunc = LowerEnvelope(cFuncEmpUncnst, cFuncEmpCnst)
        cFunc_by_state = np.insert(cFunc_by_state, 0, __cFunc)

        sGridEmp = np.zeros(mNrmGridEmp.size)
        __sFunc = LinearInterp(mNrmGridEmp, sGridEmp)
        sFunc_by_state = np.insert(sFunc_by_state, 0, __sFunc)

        __vPfunc = MargValueFunc(cFunc_by_state[0], CRRA)
        vPfunc_by_state = np.insert(vPfunc_by_state, 0, __vPfunc)

        __mGridEmp = aXtraGrid + mNrmMinNowEmp
        __cValEmpOnThisGrid = cFunc_by_state[0](__mGridEmp)
        __aValEmpOnThisGrid = __mGridEmp - __cValEmpOnThisGrid
        __EndOfPrdvFuncUnempOnThisGrid = EndOfPrdvFunc_by_state_next[1](__aValEmpOnThisGrid)
        __EndOfPrdvFuncEmpOnThisGrid = EndOfPrdvFunc_by_state_next[0](__aValEmpOnThisGrid)
        __vFuncOnThisGrid = utilitySepa(__cValEmpOnThisGrid) + (1.0 - SepaRte) * __EndOfPrdvFuncEmpOnThisGrid + \
                            SepaRte * __EndOfPrdvFuncUnempOnThisGrid
        __vFuncNrvsOnThisGrid = utility_inv(__vFuncOnThisGrid)
        __vFuncNrvsOnThisGrid = np.insert(__vFuncNrvsOnThisGrid, 0, 0.0)
        __mGridEmp = np.insert(__mGridEmp, 0, mNrmMinNowEmp)
        __vFuncNrvs = LinearInterp(__mGridEmp, __vFuncNrvsOnThisGrid)
        __vFuncEmp = ValueFunc(__vFuncNrvs, CRRA)
        vFunc_by_state = np.insert(vFunc_by_state, 0, __vFuncEmp)

        mNrmMin_list = np.array([mNrmMinNowEmp, mNrmMinNow])

        # 8
        # construct an object called solution now
        solution_now = ConsumerLaborSolution(cFunc=cFunc_by_state, vFunc=vFunc_by_state, vPfunc=vPfunc_by_state)
        solution_now.sFunc = sFunc_by_state
        solution_now.mNrmMin = mNrmMin_list

        return solution_now        
        
                
        
# Make a default dictionary for the intensive margin labor supply model
init_labor_intensive = copy(init_idiosyncratic_shocks)
init_labor_intensive["LbrCostCoeffs"] = [-1.0]
init_labor_intensive["WageRte"] = [1.0]
init_labor_intensive["IncUnemp"] = 0.0
init_labor_intensive[
    "TranShkCount"
] = 15  # Crank up permanent shock count - Number of points in discrete approximation to transitory income shocks
init_labor_intensive["PermShkCount"] = 16  # Crank up permanent shock count
init_labor_intensive[
    "aXtraCount"
] = 200  # May be important to have a larger number of gridpoints (than 48 initially)
init_labor_intensive["aXtraMax"] = 80.0
init_labor_intensive["BoroCnstArt"] = None

# Make a dictionary for intensive margin labor supply model with finite lifecycle
init_labor_lifecycle = init_labor_intensive.copy()
init_labor_lifecycle["PermGroFac"] = [
    1.01,
    1.01,
    1.01,
    1.01,
    1.01,
    1.02,
    1.02,
    1.02,
    1.02,
    1.02,
]
init_labor_lifecycle["PermShkStd"] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
init_labor_lifecycle["TranShkStd"] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
init_labor_lifecycle["LivPrb"] = [
    0.99,
    0.9,
    0.8,
    0.7,
    0.6,
    0.5,
    0.4,
    0.3,
    0.2,
    0.1,
]  # Living probability decreases as time moves forward.
init_labor_lifecycle["WageRte"] = [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
]  # Wage rate in a lifecycle
init_labor_lifecycle["LbrCostCoeffs"] = [
    -2.0,
    0.4,
]  # Assume labor cost coeffs is a polynomial of degree 1
init_labor_lifecycle["T_cycle"] = 10
# init_labor_lifecycle['T_retire']   = 7 # IndexError at line 774 in interpolation.py.
init_labor_lifecycle[
    "T_age"
] = 11  # Make sure that old people die at terminal age and don't turn into newborns!
