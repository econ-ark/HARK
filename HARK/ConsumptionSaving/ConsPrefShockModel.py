"""
Extensions to ConsIndShockModel concerning models with preference shocks.
It currently only two models:

1) An extension of ConsIndShock, but with an iid lognormal multiplicative shock each period.
2) A combination of (1) and ConsKinkedR, demonstrating how to construct a new model
   by inheriting from multiple classes.
"""

import numpy as np

from HARK import NullFunc
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsumerSolution,
    IndShockConsumerType,
    KinkedRconsumerType,
    make_assets_grid,
    make_basic_CRRA_solution_terminal,
)
from HARK.Calibration.Income.IncomeProcesses import (
    construct_lognormal_income_process_unemployment,
    get_PermShkDstn_from_IncShkDstn,
    get_TranShkDstn_from_IncShkDstn,
)
from HARK.distribution import MeanOneLogNormal, expected
from HARK.interpolation import (
    CubicInterp,
    LinearInterp,
    LinearInterpOnInterp1D,
    LowerEnvelope,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.rewards import UtilityFuncCRRA

__all__ = [
    "PrefShockConsumerType",
    "KinkyPrefConsumerType",
]


def make_lognormal_PrefShkDstn(
    T_cycle,
    PrefShkStd,
    PrefShkCount,
    RNG,
    PrefShk_tail_N=0,
    PrefShk_tail_order=np.e,
    PrefShk_tail_bound=[0.02, 0.98],
):
    """
    Make a discretized mean one lognormal preference shock distribution for each
    period of the agent's problem.

    Parameters
    ----------
    T_cycle : int
        Number of non-terminal periods in the agent's cycle.
    PrefShkStd : [float]
        Standard deviation of log preference shocks in each period.
    PrefShkCount : int
        Number of equiprobable preference shock nodes in the "body" of the distribution.
    RNG : RandomState
        The AgentType's internal random number generator.
    PrefShk_tail_N : int
        Number of shock nodes in each "tail" of the distribution (optional).
    PrefShk_tail_order : float
        Scaling factor for tail nodes (optional).
    PrefShk_tail_bound : [float,float]
        CDF bounds for tail nodes (optional).

    Returns
    -------
    PrefShkDstn : [DiscreteDistribution]
        List of discretized lognormal distributions for shocks.
    """
    PrefShkDstn = []  # discrete distributions of preference shocks
    for t in range(T_cycle):
        PrefShkStd = PrefShkStd[t]
        new_dstn = MeanOneLogNormal(
            sigma=PrefShkStd, seed=RNG.integers(0, 2**31 - 1)
        ).discretize(
            N=PrefShkCount,
            method="equiprobable",
            tail_N=PrefShk_tail_N,
            tail_order=PrefShk_tail_order,
            tail_bound=PrefShk_tail_bound,
        )
        PrefShkDstn.append(new_dstn)
    return PrefShkDstn


###############################################################################

# Make a dictionary of constructors for the preference shock model
prefshk_constructor_dict = {
    "IncShkDstn": construct_lognormal_income_process_unemployment,
    "PermShkDstn": get_PermShkDstn_from_IncShkDstn,
    "TranShkDstn": get_TranShkDstn_from_IncShkDstn,
    "aXtraGrid": make_assets_grid,
    "PrefShkDstn": make_lognormal_PrefShkDstn,
    "solution_terminal": make_basic_CRRA_solution_terminal,
}

# Default parameters to make IncShkDstn using construct_lognormal_income_process_unemployment
default_IncShkDstn_params = {
    "PermShkStd": [0.1],  # Standard deviation of log permanent income shocks
    "PermShkCount": 7,  # Number of points in discrete approximation to permanent income shocks
    "TranShkStd": [0.1],  # Standard deviation of log transitory income shocks
    "TranShkCount": 7,  # Number of points in discrete approximation to transitory income shocks
    "UnempPrb": 0.05,  # Probability of unemployment while working
    "IncUnemp": 0.3,  # Unemployment benefits replacement rate while working
    "T_retire": 0,  # Period of retirement (0 --> no retirement)
    "UnempPrbRet": 0.005,  # Probability of "unemployment" while retired
    "IncUnempRet": 0.0,  # "Unemployment" benefits when retired
}

# Default parameters to make aXtraGrid using construct_assets_grid
default_aXtraGrid_params = {
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 20,  # Maximum end-of-period "assets above minimum" value
    "aXtraNestFac": 3,  # Exponential nesting factor for aXtraGrid
    "aXtraCount": 48,  # Number of points in the grid of "assets above minimum"
    "aXtraExtra": None,  # Additional other values to add in grid (optional)
}

# Default parameters to make PrefShkDstn using make_lognormal_PrefShkDstn
default_PrefShkDstn_params = {
    "PrefShkCount": 12,  # Number of points in discrete approximation to preference shock dist
    "PrefShk_tail_N": 4,  # Number of "tail points" on each end of pref shock dist
    "PrefShkStd": [0.30],  # Standard deviation of utility shocks
}

# Make a dictionary to specify an preference shocks consumer type
init_preference_shocks = {
    # BASIC HARK PARAMETERS REQUIRED TO SOLVE THE MODEL
    "cycles": 1,  # Finite, non-cyclic model
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "constructors": prefshk_constructor_dict,  # See dictionary above
    # PRIMITIVE RAW PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": 1.03,  # Interest factor on retained assets
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability after each period
    "PermGroFac": [1.01],  # Permanent income growth factor
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint
    "vFuncBool": False,  # Whether to calculate the value function during solution
    "CubicBool": False,  # Whether to use cubic spline interpolation when True
    # (Uses linear spline interpolation for cFunc when False)
    # PARAMETERS REQUIRED TO SIMULATE THE MODEL
    "AgentCount": 10000,  # Number of agents of this type
    "T_age": None,  # Age after which simulated agents are automatically killed
    "aNrmInitMean": 0.0,  # Mean of log initial assets
    "aNrmInitStd": 1.0,  # Standard deviation of log initial assets
    "pLvlInitMean": 0.0,  # Mean of log initial permanent income
    "pLvlInitStd": 0.0,  # Standard deviation of log initial permanent income
    "PermGroFacAgg": 1.0,  # Aggregate permanent income growth factor
    # (The portion of PermGroFac attributable to aggregate productivity growth)
    "NewbornTransShk": False,  # Whether Newborns have transitory shock
    # ADDITIONAL OPTIONAL PARAMETERS
    "PerfMITShk": False,  # Do Perfect Foresight MIT Shock
    # (Forces Newborns to follow solution path of the agent they replaced if True)
    "neutral_measure": False,  # Whether to use permanent income neutral measure (see Harmenberg 2021)
}
init_preference_shocks.update(default_IncShkDstn_params)
init_preference_shocks.update(default_aXtraGrid_params)
init_preference_shocks.update(default_PrefShkDstn_params)


# Specify default parameters that differ in "kinky preference" model compared to base PrefShockConsumerType
kinky_pref_different_params = {
    "Rboro": 1.20,  # Interest factor on assets when borrowing, a < 0
    "Rsave": 1.02,  # Interest factor on assets when saving, a > 0
    "BoroCnstArt": None,  # Kinked R only matters if borrowing is allowed
}

# Make a dictionary to specify a "kinky preference" consumer
init_kinky_pref = init_preference_shocks.copy()  # Start with base above
init_kinky_pref.update(kinky_pref_different_params)  # Change just a few params
del init_kinky_pref["Rfree"]


class PrefShockConsumerType(IndShockConsumerType):
    """
    A class for representing consumers who experience multiplicative shocks to
    utility each period, specified as iid lognormal.

    See ConsumerParameters.init_pref_shock for a dictionary of
    the keywords that should be passed to the constructor.

    Parameters
    ----------

    """

    shock_vars_ = IndShockConsumerType.shock_vars_ + ["PrefShk"]

    def __init__(self, **kwds):
        params = init_preference_shocks.copy()
        params.update(kwds)

        IndShockConsumerType.__init__(self, **params)
        self.solve_one_period = solve_one_period_ConsPrefShock

    def pre_solve(self):
        self.update_solution_terminal()

    def update(self):
        """
        Updates the assets grid, income process, terminal period solution, and
        preference shock process.  A very slight extension of IndShockConsumerType.update()
        for the preference shock model.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        IndShockConsumerType.update(
            self
        )  # Update assets grid, income process, terminal solution
        self.update_pref_shock_process()  # Update the discrete preference shock process

    def update_pref_shock_process(self):
        """
        Make a discrete preference shock structure for each period in the cycle
        for this agent type, storing them as attributes of self for use in the
        solution (and other methods).

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        self.construct("PrefShkDstn")
        self.add_to_time_vary("PrefShkDstn")

    def reset_rng(self):
        """
        Reset the RNG behavior of this type.  This method is called automatically
        by initialize_sim(), ensuring that each simulation run uses the same sequence
        of random shocks; this is necessary for structural estimation to work.
        This method extends IndShockConsumerType.reset_rng() to also reset elements
        of PrefShkDstn.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        IndShockConsumerType.reset_rng(self)

        # Reset PrefShkDstn if it exists (it might not because reset_rng is called at init)
        if hasattr(self, "PrefShkDstn"):
            for dstn in self.PrefShkDstn:
                dstn.reset()

    def get_shocks(self):
        """
        Gets permanent and transitory income shocks for this period as well as preference shocks.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        IndShockConsumerType.get_shocks(
            self
        )  # Get permanent and transitory income shocks
        PrefShkNow = np.zeros(self.AgentCount)  # Initialize shock array
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            N = np.sum(these)
            if N > 0:
                PrefShkNow[these] = self.PrefShkDstn[t].draw(N)
        self.shocks["PrefShk"] = PrefShkNow

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
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            cNrmNow[these] = self.solution[t].cFunc(
                self.state_now["mNrm"][these], self.shocks["PrefShk"][these]
            )
        self.controls["cNrm"] = cNrmNow
        return None

    def calc_bounding_values(self):
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


class KinkyPrefConsumerType(PrefShockConsumerType, KinkedRconsumerType):
    """
    A class for representing consumers who experience multiplicative shocks to
    utility each period, specified as iid lognormal and different interest rates
    on borrowing vs saving.

    See init_kinky_pref for a dictionary of the keywords
    that should be passed to the constructor.

    Parameters
    ----------

    """

    def __init__(self, **kwds):
        params = init_kinky_pref.copy()
        params.update(kwds)
        kwds = params
        IndShockConsumerType.__init__(self, **kwds)
        self.solve_one_period = solve_one_period_ConsKinkyPref
        self.add_to_time_inv("Rboro", "Rsave")
        self.del_from_time_inv("Rfree")

    def pre_solve(self):
        self.update_solution_terminal()

    def get_Rfree(self):  # Specify which get_Rfree to use
        return KinkedRconsumerType.get_Rfree(self)


###############################################################################


def solve_one_period_ConsPrefShock(
    solution_next,
    IncShkDstn,
    PrefShkDstn,
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
    Solves one period of a consumption-saving model with idiosyncratic shocks to
    permanent and transitory income, with one risk free asset and CRRA utility.
    The consumer also faces iid preference shocks as a multiplicative shifter to
    their marginal utility of consumption.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to the succeeding one period problem.
    IncShkDstn : distribution.Distribution
        A discrete approximation to the income process between the period being
        solved and the one immediately following (in solution_next). Order:
        permanent shocks, transitory shocks.
    PrefShkDstn : distribution.Distribution
        Discrete distribution of the multiplicative utility shifter.  Order:
        probabilities, preference shocks.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    PermGroGac : float
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
    solution: ConsumerSolution
        The solution to the single period consumption-saving problem.  Includes
        a consumption function cFunc (using linear splines), a marginal value
        function vPfunc, a minimum acceptable level of normalized market re-
        sources mNrmMin, normalized human wealth hNrm, and bounding MPCs MPCmin
        and MPCmax.  It might also have a value function vFunc.  The consumption
        function is defined over normalized market resources and the preference
        shock, c = cFunc(m,PrefShk), but the (marginal) value function is defined
        unconditionally on the shock, just before it is revealed.
    """
    # Define the current period utility function and effective discount factor
    uFunc = UtilityFuncCRRA(CRRA)
    DiscFacEff = DiscFac * LivPrb  # "effective" discount factor

    # Unpack next period's income and preference shock distributions
    ShkPrbsNext = IncShkDstn.pmv
    PermShkValsNext = IncShkDstn.atoms[0]
    TranShkValsNext = IncShkDstn.atoms[1]
    PermShkMinNext = np.min(PermShkValsNext)
    TranShkMinNext = np.min(TranShkValsNext)
    PrefShkPrbs = PrefShkDstn.pmv
    PrefShkVals = PrefShkDstn.atoms.flatten()

    # Calculate the probability that we get the worst possible income draw
    IncNext = PermShkValsNext * TranShkValsNext
    WorstIncNext = PermShkMinNext * TranShkMinNext
    WorstIncPrb = np.sum(ShkPrbsNext[IncNext == WorstIncNext])
    # WorstIncPrb is the "Weierstrass p" concept: the odds we get the WORST thing

    # Unpack next period's (marginal) value function
    vFuncNext = solution_next.vFunc  # This is None when vFuncBool is False
    vPfuncNext = solution_next.vPfunc
    vPPfuncNext = solution_next.vPPfunc  # This is None when CubicBool is False

    # Update the bounding MPCs and PDV of human wealth:
    PatFac = ((Rfree * DiscFacEff) ** (1.0 / CRRA)) / Rfree
    try:
        MPCminNow = 1.0 / (1.0 + PatFac / solution_next.MPCmin)
    except:
        MPCminNow = 0.0
    Ex_IncNext = np.dot(ShkPrbsNext, TranShkValsNext * PermShkValsNext)
    hNrmNow = PermGroFac / Rfree * (Ex_IncNext + solution_next.hNrm)
    temp_fac = (WorstIncPrb ** (1.0 / CRRA)) * PatFac
    MPCmaxNow = 1.0 / (1.0 + temp_fac / solution_next.MPCmax)

    # Calculate the minimum allowable value of money resources in this period
    PermGroFacEffMin = (PermGroFac * PermShkMinNext) / Rfree
    BoroCnstNat = (solution_next.mNrmMin - TranShkMinNext) * PermGroFacEffMin

    # Set the minimum allowable (normalized) market resources based on the natural
    # and artificial borrowing constraints
    if BoroCnstArt is None:
        mNrmMinNow = BoroCnstNat
    else:
        mNrmMinNow = np.max([BoroCnstNat, BoroCnstArt])

    # Set the upper limit of the MPC (at mNrmMinNow) based on whether the natural
    # or artificial borrowing constraint actually binds
    if BoroCnstNat < mNrmMinNow:
        MPCmaxEff = 1.0  # If actually constrained, MPC near limit is 1
    else:
        MPCmaxEff = MPCmaxNow  # Otherwise, it's the MPC calculated above

    # Define the borrowing-constrained consumption function
    cFuncNowCnst = LinearInterp(
        np.array([mNrmMinNow, mNrmMinNow + 1.0]), np.array([0.0, 1.0])
    )

    # Construct the assets grid by adjusting aXtra by the natural borrowing constraint
    aNrmNow = np.asarray(aXtraGrid) + BoroCnstNat

    # Define local functions for taking future expectations
    def calc_mNrmNext(S, a, R):
        return R / (PermGroFac * S["PermShk"]) * a + S["TranShk"]

    def calc_vNext(S, a, R):
        return (S["PermShk"] ** (1.0 - CRRA) * PermGroFac ** (1.0 - CRRA)) * vFuncNext(
            calc_mNrmNext(S, a, R)
        )

    def calc_vPnext(S, a, R):
        return S["PermShk"] ** (-CRRA) * vPfuncNext(calc_mNrmNext(S, a, R))

    def calc_vPPnext(S, a, R):
        return S["PermShk"] ** (-CRRA - 1.0) * vPPfuncNext(calc_mNrmNext(S, a, R))

    # Calculate end-of-period marginal value of assets at each gridpoint
    vPfacEff = DiscFacEff * Rfree * PermGroFac ** (-CRRA)
    EndOfPrdvP = vPfacEff * expected(calc_vPnext, IncShkDstn, args=(aNrmNow, Rfree))

    # Find optimal consumption corresponding to each aNrm, PrefShk combination
    cNrm_base = uFunc.derinv(EndOfPrdvP, order=(1, 0))
    PrefShkCount = PrefShkVals.size
    PrefShk_temp = np.tile(
        np.reshape(PrefShkVals ** (1.0 / CRRA), (PrefShkCount, 1)),
        (1, cNrm_base.size),
    )
    cNrmNow = np.tile(cNrm_base, (PrefShkCount, 1)) * PrefShk_temp
    mNrmNow = cNrmNow + np.tile(aNrmNow, (PrefShkCount, 1))
    # These are the endogenous gridpoints, as usual

    # Add the bottom point to the c and m arrays
    m_for_interpolation = np.concatenate(
        (BoroCnstNat * np.ones((PrefShkCount, 1)), mNrmNow), axis=1
    )
    c_for_interpolation = np.concatenate((np.zeros((PrefShkCount, 1)), cNrmNow), axis=1)

    # Construct the consumption function as a cubic or linear spline interpolation
    # for each value of PrefShk, interpolated across those values.
    if CubicBool:
        # This is not yet supported, not sure why we never got to it
        raise (
            ValueError,
            "Cubic interpolation is not yet supported by the preference shock model!",
        )

    # Make the preference-shock specific consumption functions
    cFuncs_by_PrefShk = []
    for j in range(PrefShkCount):
        MPCmin_j = MPCminNow * PrefShkVals[j] ** (1.0 / CRRA)
        cFunc_this_shk = LowerEnvelope(
            LinearInterp(
                m_for_interpolation[j, :],
                c_for_interpolation[j, :],
                intercept_limit=hNrmNow * MPCmin_j,
                slope_limit=MPCmin_j,
            ),
            cFuncNowCnst,
        )
        cFuncs_by_PrefShk.append(cFunc_this_shk)

    # Combine the list of consumption functions into a single interpolation
    cFuncNow = LinearInterpOnInterp1D(cFuncs_by_PrefShk, PrefShkVals)

    # Make the ex ante marginal value function (before the preference shock)
    m_grid = aXtraGrid + mNrmMinNow
    vP_vec = np.zeros_like(m_grid)
    for j in range(PrefShkCount):  # numeric integration over the preference shock
        vP_vec += (
            uFunc.der(cFuncs_by_PrefShk[j](m_grid)) * PrefShkPrbs[j] * PrefShkVals[j]
        )
    vPnvrs_vec = uFunc.derinv(vP_vec, order=(1, 0))
    vPfuncNow = MargValueFuncCRRA(LinearInterp(m_grid, vPnvrs_vec), CRRA)

    # Define this period's marginal marginal value function
    if CubicBool:
        pass  # This is impossible to reach right now
    else:
        vPPfuncNow = NullFunc()  # Dummy object

    # Construct this period's value function if requested
    if vFuncBool:
        # Calculate end-of-period value, its derivative, and their pseudo-inverse
        EndOfPrdv = DiscFacEff * expected(calc_vNext, IncShkDstn, args=(aNrmNow, Rfree))
        EndOfPrdvNvrs = uFunc.inv(
            EndOfPrdv
        )  # value transformed through inverse utility
        EndOfPrdvNvrsP = EndOfPrdvP * uFunc.derinv(EndOfPrdv, order=(0, 1))
        EndOfPrdvNvrs = np.insert(EndOfPrdvNvrs, 0, 0.0)
        EndOfPrdvNvrsP = np.insert(EndOfPrdvNvrsP, 0, EndOfPrdvNvrsP[0])
        # This is a very good approximation, vNvrsPP = 0 at the asset minimum

        # Construct the end-of-period value function
        aNrm_temp = np.insert(aNrmNow, 0, BoroCnstNat)
        EndOfPrd_vNvrsFunc = CubicInterp(aNrm_temp, EndOfPrdvNvrs, EndOfPrdvNvrsP)
        EndOfPrd_vFunc = ValueFuncCRRA(EndOfPrd_vNvrsFunc, CRRA)

        # Compute expected value and marginal value on a grid of market resources,
        # accounting for all of the discrete preference shocks
        mNrm_temp = mNrmMinNow + aXtraGrid
        v_temp = np.zeros_like(mNrm_temp)
        vP_temp = np.zeros_like(mNrm_temp)
        for j in range(PrefShkCount):
            this_shock = PrefShkVals[j]
            this_prob = PrefShkPrbs[j]
            cNrm_temp = cFuncNow(mNrm_temp, this_shock * np.ones_like(mNrm_temp))
            aNrm_temp = mNrm_temp - cNrm_temp
            v_temp += this_prob * (
                this_shock * uFunc(cNrm_temp) + EndOfPrd_vFunc(aNrm_temp)
            )
            vP_temp += this_prob * this_shock * uFunc.der(cNrm_temp)

        # Construct the beginning-of-period value function
        # value transformed through inverse utility
        vNvrs_temp = uFunc.inv(v_temp)
        vNvrsP_temp = vP_temp * uFunc.derinv(v_temp, order=(0, 1))
        mNrm_temp = np.insert(mNrm_temp, 0, mNrmMinNow)
        vNvrs_temp = np.insert(vNvrs_temp, 0, 0.0)
        vNvrsP_temp = np.insert(vNvrsP_temp, 0, MPCmaxEff ** (-CRRA / (1.0 - CRRA)))
        MPCminNvrs = MPCminNow ** (-CRRA / (1.0 - CRRA))
        vNvrsFuncNow = CubicInterp(
            mNrm_temp, vNvrs_temp, vNvrsP_temp, MPCminNvrs * hNrmNow, MPCminNvrs
        )
        vFuncNow = ValueFuncCRRA(vNvrsFuncNow, CRRA)

    else:
        vFuncNow = NullFunc()  # Dummy object

    # Create and return this period's solution
    solution_now = ConsumerSolution(
        cFunc=cFuncNow,
        vFunc=vFuncNow,
        vPfunc=vPfuncNow,
        vPPfunc=vPPfuncNow,
        mNrmMin=mNrmMinNow,
        hNrm=hNrmNow,
        MPCmin=MPCminNow,
        MPCmax=MPCmaxEff,
    )
    return solution_now


def solve_one_period_ConsKinkyPref(
    solution_next,
    IncShkDstn,
    PrefShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rboro,
    Rsave,
    PermGroFac,
    BoroCnstArt,
    aXtraGrid,
    vFuncBool,
    CubicBool,
):
    """
    Solves one period of a consumption-saving model with idiosyncratic shocks to
    permanent and transitory income, with a risk free asset and CRRA utility.
    In this variation, the interest rate on borrowing Rboro exceeds the interest
    rate on saving Rsave. The consumer also faces iid preference shocks as a multi-
    plicative shifter to their marginal utility of consumption.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncShkDstn : distribution.Distribution
        A discrete approximation to the income process between the period being
        solved and the one immediately following (in solution_next).
    PrefShkDstn : distribution.Distribution
        Discrete distribution of the multiplicative utility shifter.  Order:
        probabilities, preference shocks.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    Rboro: float
        Interest factor on assets between this period and the succeeding
        period when assets are negative.
    Rsave: float
        Interest factor on assets between this period and the succeeding
        period when assets are positive.
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
    solution_now : ConsumerSolution
        The solution to the single period consumption-saving problem.  Includes
        a consumption function cFunc (using linear splines), a marginal value
        function vPfunc, a minimum acceptable level of normalized market re-
        sources mNrmMin, normalized human wealth hNrm, and bounding MPCs MPCmin
        and MPCmax.  It might also have a value function vFunc.  The consumption
        function is defined over normalized market resources and the preference
        shock, c = cFunc(m,PrefShk), but the (marginal) value function is defined
        unconditionally on the shock, just before it is revealed.
    """
    # Verifiy that there is actually a kink in the interest factor
    assert (
        Rboro >= Rsave
    ), "Interest factor on debt less than interest factor on savings!"
    # If the kink is in the wrong direction, code should break here. If there's
    # no kink at all, then just use the ConsIndShockModel solver.
    if Rboro == Rsave:
        solution_now = solve_one_period_ConsPrefShock(
            solution_next,
            IncShkDstn,
            PrefShkDstn,
            LivPrb,
            DiscFac,
            CRRA,
            Rboro,
            PermGroFac,
            BoroCnstArt,
            aXtraGrid,
            vFuncBool,
            CubicBool,
        )
        return solution_now

    # Define the current period utility function and effective discount factor
    uFunc = UtilityFuncCRRA(CRRA)
    DiscFacEff = DiscFac * LivPrb  # "effective" discount factor

    # Unpack next period's income and preference shock distributions
    ShkPrbsNext = IncShkDstn.pmv
    PermShkValsNext = IncShkDstn.atoms[0]
    TranShkValsNext = IncShkDstn.atoms[1]
    PermShkMinNext = np.min(PermShkValsNext)
    TranShkMinNext = np.min(TranShkValsNext)
    PrefShkPrbs = PrefShkDstn.pmv
    PrefShkVals = PrefShkDstn.atoms.flatten()

    # Calculate the probability that we get the worst possible income draw
    IncNext = PermShkValsNext * TranShkValsNext
    WorstIncNext = PermShkMinNext * TranShkMinNext
    WorstIncPrb = np.sum(ShkPrbsNext[IncNext == WorstIncNext])
    # WorstIncPrb is the "Weierstrass p" concept: the odds we get the WORST thing

    # Unpack next period's (marginal) value function
    vFuncNext = solution_next.vFunc  # This is None when vFuncBool is False
    vPfuncNext = solution_next.vPfunc
    vPPfuncNext = solution_next.vPPfunc  # This is None when CubicBool is False

    # Update the bounding MPCs and PDV of human wealth:
    PatFac = ((Rsave * DiscFacEff) ** (1.0 / CRRA)) / Rsave
    PatFacAlt = ((Rboro * DiscFacEff) ** (1.0 / CRRA)) / Rboro
    try:
        MPCminNow = 1.0 / (1.0 + PatFac / solution_next.MPCmin)
    except:
        MPCminNow = 0.0
    Ex_IncNext = np.dot(ShkPrbsNext, TranShkValsNext * PermShkValsNext)
    hNrmNow = (PermGroFac / Rsave) * (Ex_IncNext + solution_next.hNrm)
    temp_fac = (WorstIncPrb ** (1.0 / CRRA)) * PatFacAlt
    MPCmaxNow = 1.0 / (1.0 + temp_fac / solution_next.MPCmax)

    # Calculate the minimum allowable value of money resources in this period
    PermGroFacEffMin = (PermGroFac * PermShkMinNext) / Rboro
    BoroCnstNat = (solution_next.mNrmMin - TranShkMinNext) * PermGroFacEffMin

    # Set the minimum allowable (normalized) market resources based on the natural
    # and artificial borrowing constraints
    if BoroCnstArt is None:
        mNrmMinNow = BoroCnstNat
    else:
        mNrmMinNow = np.max([BoroCnstNat, BoroCnstArt])

    # Set the upper limit of the MPC (at mNrmMinNow) based on whether the natural
    # or artificial borrowing constraint actually binds
    if BoroCnstNat < mNrmMinNow:
        MPCmaxEff = 1.0  # If actually constrained, MPC near limit is 1
    else:
        MPCmaxEff = MPCmaxNow  # Otherwise, it's the MPC calculated above

    # Define the borrowing-constrained consumption function
    cFuncNowCnst = LinearInterp(
        np.array([mNrmMinNow, mNrmMinNow + 1.0]), np.array([0.0, 1.0])
    )

    # Construct the assets grid by adjusting aXtra by the natural borrowing constraint
    aNrmNow = np.sort(
        np.hstack((np.asarray(aXtraGrid) + mNrmMinNow, np.array([0.0, 0.0])))
    )

    # Make a 1D array of the interest factor at each asset gridpoint
    Rfree = Rsave * np.ones_like(aNrmNow)
    Rfree[aNrmNow < 0] = Rboro
    i_kink = np.argwhere(aNrmNow == 0.0)[0][0]
    Rfree[i_kink] = Rboro

    # Define local functions for taking future expectations
    def calc_mNrmNext(S, a, R):
        return R / (PermGroFac * S["PermShk"]) * a + S["TranShk"]

    def calc_vNext(S, a, R):
        return (S["PermShk"] ** (1.0 - CRRA) * PermGroFac ** (1.0 - CRRA)) * vFuncNext(
            calc_mNrmNext(S, a, R)
        )

    def calc_vPnext(S, a, R):
        return S["PermShk"] ** (-CRRA) * vPfuncNext(calc_mNrmNext(S, a, R))

    def calc_vPPnext(S, a, R):
        return S["PermShk"] ** (-CRRA - 1.0) * vPPfuncNext(calc_mNrmNext(S, a, R))

    # Calculate end-of-period marginal value of assets at each gridpoint
    vPfacEff = DiscFacEff * Rfree * PermGroFac ** (-CRRA)
    EndOfPrdvP = vPfacEff * expected(calc_vPnext, IncShkDstn, args=(aNrmNow, Rfree))

    # Find optimal consumption corresponding to each aNrm, PrefShk combination
    cNrm_base = uFunc.derinv(EndOfPrdvP, order=(1, 0))
    PrefShkCount = PrefShkVals.size
    PrefShk_temp = np.tile(
        np.reshape(PrefShkVals ** (1.0 / CRRA), (PrefShkCount, 1)),
        (1, cNrm_base.size),
    )
    cNrmNow = np.tile(cNrm_base, (PrefShkCount, 1)) * PrefShk_temp
    mNrmNow = cNrmNow + np.tile(aNrmNow, (PrefShkCount, 1))
    # These are the endogenous gridpoints, as usual

    # Add the bottom point to the c and m arrays
    m_for_interpolation = np.concatenate(
        (BoroCnstNat * np.ones((PrefShkCount, 1)), mNrmNow), axis=1
    )
    c_for_interpolation = np.concatenate((np.zeros((PrefShkCount, 1)), cNrmNow), axis=1)

    # Construct the consumption function as a cubic or linear spline interpolation
    # for each value of PrefShk, interpolated across those values.
    if CubicBool:
        # This is not yet supported, not sure why we never got to it
        raise (
            ValueError,
            "Cubic interpolation is not yet supported by the preference shock model!",
        )

    # Make the preference-shock specific consumption functions
    cFuncs_by_PrefShk = []
    for j in range(PrefShkCount):
        MPCmin_j = MPCminNow * PrefShkVals[j] ** (1.0 / CRRA)
        cFunc_this_shk = LowerEnvelope(
            LinearInterp(
                m_for_interpolation[j, :],
                c_for_interpolation[j, :],
                intercept_limit=hNrmNow * MPCmin_j,
                slope_limit=MPCmin_j,
            ),
            cFuncNowCnst,
        )
        cFuncs_by_PrefShk.append(cFunc_this_shk)

    # Combine the list of consumption functions into a single interpolation
    cFuncNow = LinearInterpOnInterp1D(cFuncs_by_PrefShk, PrefShkVals)

    # Make the ex ante marginal value function (before the preference shock)
    m_grid = aXtraGrid + mNrmMinNow
    vP_vec = np.zeros_like(m_grid)
    for j in range(PrefShkCount):  # numeric integration over the preference shock
        vP_vec += (
            uFunc.der(cFuncs_by_PrefShk[j](m_grid)) * PrefShkPrbs[j] * PrefShkVals[j]
        )
    vPnvrs_vec = uFunc.derinv(vP_vec, order=(1, 0))
    vPfuncNow = MargValueFuncCRRA(LinearInterp(m_grid, vPnvrs_vec), CRRA)

    # Define this period's marginal marginal value function
    if CubicBool:
        pass  # This is impossible to reach right now
    else:
        vPPfuncNow = NullFunc()  # Dummy object

    # Construct this period's value function if requested
    if vFuncBool:
        # Calculate end-of-period value, its derivative, and their pseudo-inverse
        EndOfPrdv = DiscFacEff * expected(calc_vNext, IncShkDstn, args=(aNrmNow, Rfree))
        EndOfPrdvNvrs = uFunc.inv(
            EndOfPrdv
        )  # value transformed through inverse utility
        EndOfPrdvNvrsP = EndOfPrdvP * uFunc.derinv(EndOfPrdv, order=(0, 1))
        EndOfPrdvNvrs = np.insert(EndOfPrdvNvrs, 0, 0.0)
        EndOfPrdvNvrsP = np.insert(EndOfPrdvNvrsP, 0, EndOfPrdvNvrsP[0])
        # This is a very good approximation, vNvrsPP = 0 at the asset minimum

        # Construct the end-of-period value function
        aNrm_temp = np.insert(aNrmNow, 0, BoroCnstNat)
        EndOfPrd_vNvrsFunc = CubicInterp(aNrm_temp, EndOfPrdvNvrs, EndOfPrdvNvrsP)
        EndOfPrd_vFunc = ValueFuncCRRA(EndOfPrd_vNvrsFunc, CRRA)

        # Compute expected value and marginal value on a grid of market resources,
        # accounting for all of the discrete preference shocks
        mNrm_temp = mNrmMinNow + aXtraGrid
        v_temp = np.zeros_like(mNrm_temp)
        vP_temp = np.zeros_like(mNrm_temp)
        for j in range(PrefShkCount):
            this_shock = PrefShkVals[j]
            this_prob = PrefShkPrbs[j]
            cNrm_temp = cFuncNow(mNrm_temp, this_shock * np.ones_like(mNrm_temp))
            aNrm_temp = mNrm_temp - cNrm_temp
            v_temp += this_prob * (
                this_shock * uFunc(cNrm_temp) + EndOfPrd_vFunc(aNrm_temp)
            )
            vP_temp += this_prob * this_shock * uFunc.der(cNrm_temp)

        # Construct the beginning-of-period value function
        # value transformed through inverse utility
        vNvrs_temp = uFunc.inv(v_temp)
        vNvrsP_temp = vP_temp * uFunc.derinv(v_temp, order=(0, 1))
        mNrm_temp = np.insert(mNrm_temp, 0, mNrmMinNow)
        vNvrs_temp = np.insert(vNvrs_temp, 0, 0.0)
        vNvrsP_temp = np.insert(vNvrsP_temp, 0, MPCmaxEff ** (-CRRA / (1.0 - CRRA)))
        MPCminNvrs = MPCminNow ** (-CRRA / (1.0 - CRRA))
        vNvrsFuncNow = CubicInterp(
            mNrm_temp, vNvrs_temp, vNvrsP_temp, MPCminNvrs * hNrmNow, MPCminNvrs
        )
        vFuncNow = ValueFuncCRRA(vNvrsFuncNow, CRRA)

    else:
        vFuncNow = NullFunc()  # Dummy object

    # Create and return this period's solution
    solution_now = ConsumerSolution(
        cFunc=cFuncNow,
        vFunc=vFuncNow,
        vPfunc=vPfuncNow,
        vPPfunc=vPPfuncNow,
        mNrmMin=mNrmMinNow,
        hNrm=hNrmNow,
        MPCmin=MPCminNow,
        MPCmax=MPCmaxEff,
    )
    return solution_now
