"""
This module has consumption-saving models with both wealth-in-utility and consumption
habits. There are two AgentType subclasses: WealthHabitConsumerType represents agents
with u(c,h,w) preferences who can save in a single risk-free asset; WealthHabitPortfolioConsumerType
represents agents with the same preferences, but who can allocate their wealth between
a risk-free and risky asset. The latter uses a "modular" solver that connects the consumption-
saving solver for WealthHabitConsumerType to the portfolio-allocation solver used by
HabitPortfolioConsumerType, so that very little new code is required for the new model.
"""

import numpy as np
from HARK.core import get_it_from
from HARK.utilities import make_exponential_grid
from HARK.interpolation import (
    LinearInterp,
    LinearInterpOnInterp1D,
    ConstantFunction,
    Curvilinear2DInterp,
    LowerEnvelope2D,
    BilinearInterp,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.distributions import expected
from HARK.Calibration.Income.IncomeProcesses import (
    construct_lognormal_income_process_unemployment,
    get_PermShkDstn_from_IncShkDstn,
    get_TranShkDstn_from_IncShkDstn,
)
from HARK.utilities import make_assets_grid
from HARK.rewards import UtilityFuncCRRA
from HARK.ConsumptionSaving.ConsIndShockModel import (
    make_lognormal_kNrm_init_dstn,
    make_lognormal_pLvl_init_dstn,
)
from HARK.ConsumptionSaving.ConsHabitModel import (
    make_lognormal_habit_init_dstn,
    make_habit_grid,
    make_habit_solution_terminal,
    make_dense_grids,
    HabitConsumerType,
    solve_optimal_share_habit,
)
from HARK.Calibration.Assets.AssetProcesses import (
    make_lognormal_RiskyDstn,
    calc_ShareLimit_for_CRRA,
)
from HARK.ConsumptionSaving.ConsRiskyAssetModel import make_simple_ShareGrid


class HabitFormationWealthInUtilityInverter:
    """
    A class for solving the first order conditions of a consumption-saving model
    with habit formation and wealth-in-utility. In this notation, HabitRte is a
    parameter on the unit interval representing how fast the habit stock evolves;
    a value of zero means no habit dynamics and a value of one means that H_t = c_t,
    complete updating. HabitWgt is also on the unit interval and represents the
    exponent on the habit stock, which is used as a divisor on consumption in the
    utility function. The parameter WealthShare represents the Cobb-Douglas share
    for wealth (end-of-period assets) in the utility function; the complementary
    share is allocated to consumption (with habits).

    Instances of this class take three arguments when called as a function: end-of-
    period habit stock H, end-of-period assets a, and a transformation of end-of
    period marginal value chi. It returns the consumption c and habit stock h values
    that solve the first order conditions for this end-of-period state.

    kappa = c / (a + xi)
    eta = H / (a + xi)
    chi = (W_a(a,H) - lambda * W_H(a,H)) ** (-1/rho) / (a + xi) ** (-alpha * (1-omega) * (1-1/rho) + 1)

    a = m - c
    H = lambda * c + (1-lambda) * h
    m' = R a / psi + theta
    h' = H / psi

    X = (c / h**alpha) ** (1-omega) * (a + xi) ** omega
    U(X) = X**(1-rho) / (1-rho)

    Parameters
    ----------
    CRRA : float
        Coefficient of relative risk aversion, rho.
    HabitRte : float
        Rate of habit stock updating with new consumption, lambda. Must be greater
        than zero but less than one.
    HabitWgt : float
        Weight of habit stock in preferences, alpha; exponent on habits as a divisor
        in utility function. Must be greater than zero but less than one.
    WealthShare : float
        Cobb-Douglas share for wealth in the utility function, omega. Complementary
        share is for (habit-modified) consumption.
    WealthShift : float
        Non-negative additive shifter for wealth in the utility function, xi.
    EtaMin : float
        Smallest value in the relative habit grid to consider.
    EtaMax : float
        Largest value in the relative habit grid to consider.
    EtaCount : int
        Number of gridpoints in the relative habit stock grid.
    EtaOrder : float
        Strictly positive exponential order for the habit stock grid.
    z_bound : float, optional
        Absolute value on the auxiliary variable z's boundary (default 15).
        z represents values that are input into a logit transformation.
    z_N : int, optional
        Number of interpolating gridpoints to use for auxiliary variable z (default 501).
    """

    def __init__(
        self,
        CRRA,
        HabitRte,
        HabitWgt,
        WealthShare,
        WealthShift,
        EtaMin,
        EtaMax,
        EtaCount,
        EtaOrder,
        z_bound=15.0,
        z_N=501,
    ):
        # Parameter validation
        if HabitRte > 1.0:
            raise ValueError("HabitRte must be no greater than 1!")
        if HabitRte <= 0.0:
            raise ValueError("HabitRte must be strictly positive!")
        if HabitWgt > 1.0:
            raise ValueError("HabitWgt must be no greater than 1!")
        if HabitWgt <= 0.0:
            raise ValueError("HabitWgt must be strictly positive!")

        # Make grids
        kappa_limit = (1.0 - WealthShare) / WealthShare
        eta_crit = HabitRte * kappa_limit  # upper bounds are equal here
        z_vec = np.linspace(-z_bound, z_bound, z_N)
        exp_z = np.exp(z_vec)
        frac_grid = exp_z / (1.0 + exp_z)
        hGrid = make_exponential_grid(1e-3, EtaMax, EtaCount - 1, EtaOrder)
        idx = np.searchsorted(hGrid, eta_crit)
        hGrid = np.insert(hGrid, idx, eta_crit)
        hMesh = np.tile(np.reshape(hGrid, (1, EtaCount)), (z_N, 1))
        kMesh = np.empty_like(hMesh)
        for j in range(EtaCount):
            top = np.minimum(kappa_limit, hGrid[j] / HabitRte)
            kMesh[:, j] = top * frac_grid

        # Calculate chi
        fac1 = ((hMesh - HabitRte * kMesh) / (1.0 - HabitRte)) ** (
            -HabitWgt * (1.0 - WealthShare) * (1.0 - 1.0 / CRRA)
        )
        fac2 = (
            (1.0 - WealthShare) * kMesh ** (-WealthShare)
            - WealthShare * kMesh ** (1.0 - WealthShare)
        ) ** (-1.0 / CRRA)
        fac3 = kMesh ** (1.0 - WealthShare)
        chi = fac1 * fac2 * fac3

        # Make a z-from-chi function for each value of eta
        funcs_by_eta = []
        for j in range(EtaCount):
            funcs_by_eta.append(LinearInterp(chi[:, j], z_vec, lower_extrap=True))

        # Combine them into a single 2D interpolator
        zFromChiAndEtaFunc = LinearInterpOnInterp1D(funcs_by_eta, hGrid)

        # Store data on self
        self.func = zFromChiAndEtaFunc
        self.rate = HabitRte
        self.shift = WealthShift
        self.share = WealthShare

    def __call__(self, a, H, chi):
        """
        Find the consumption and habit values that solve the FOC based on end-of-period information.
        """
        a_bar = a + self.shift
        eta = H / a_bar
        z = self.func(chi, eta)
        exp_z = np.exp(z)
        limit = np.minimum(eta / self.rate, (1.0 - self.share) / self.share)
        kappa = limit * exp_z / (1 + exp_z)
        c = kappa * a_bar
        h = (H - self.rate * c) / (1.0 - self.rate)
        return c, h

    def cFunc(self, a, H, chi):
        return self(a, H, chi)[0]  # just return consumption

    def hFunc(self, a, H, chi):
        return self(a, H, chi)[1]  # just return habit stock


def make_wealth_habit_inverter(
    CRRA,
    HabitRte,
    HabitWgt,
    WealthShare,
    WealthShift,
    EtaMin,
    EtaMax,
    EtaCount,
    EtaOrder,
):
    return HabitFormationWealthInUtilityInverter(
        CRRA,
        HabitRte,
        HabitWgt,
        WealthShare,
        WealthShift,
        EtaMin,
        EtaMax,
        EtaCount,
        EtaOrder,
    )


def calc_marg_values(S, k, hpre, rho, R, Gamma, alpha, lamda, omega, xi, beta, C, Vp):
    """
    Helper function for computing expected marginal value with respect to market
    resources and habit stock. Used internally by solve_one_period_ConsWealthHabit.

    The code here uses "math notation" for quick programming. See the only place
    in the code where this function is used for a translation of the symbols.
    """
    psi = S["PermShk"]
    theta = S["TranShk"]
    G = psi * Gamma
    m = R * k / G + theta
    h = hpre / G
    c = C(m, h)
    a = m - c
    H = lamda * c + (1 - lamda) * h
    dvdH = beta * Vp(a, H)
    X = c ** (1 - omega) * (a + xi) ** omega * h ** (-alpha * (1 - omega))
    Xpow = X ** (1 - rho)
    dudc = (1 - omega) * Xpow / c
    dudh = -alpha * (1 - omega) * Xpow / h
    dvdm = dudc + lamda * dvdH
    dvdh = dudh + (1 - lamda) * dvdH
    G_adj = G ** ((1 - rho) * (1 - alpha * (1 - omega)) - 1)
    dvdk = R * G_adj * dvdm
    dvdh = G_adj * dvdh
    return dvdk, dvdh


###############################################################################


def solve_one_period_ConsWealthHabit(
    solution_next,
    IncShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rfree,
    PermGroFac,
    BoroCnstArt,
    aXtraGrid,
    HabitGrid,
    FOCinverter,
    HabitWgt,
    HabitRte,
    WealthShare,
    WealthShift,
    mXtraGrid,
    hGridDense,
):
    """
    Solve one period of the consumption-saving model with habit formation and
    wealth in the utility function.

    Parameters
    ----------
    solution_next : dict
        Dictionary with next period's solution.
    IncShkDstn : DiscreteDistribution
        Discretized permanent and transitory income shock distribution this period.
    LivPrb : float
        Survival probability at the end of this period.
    DiscFac : float
        Intertemporal discount factor.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Interest factor on capital at the start of this period.
    PermGroFac : float
        Permanent income growth factor at the start of this period.
    BoroCnstArt : float or None
        Artificial borrowing constraint on assets at the end of this period,
        as a fraction of permanent income.
    aXtraGrid : np.array
        Grid of "assets above minimum".
    HabitGrid : np.array
        Grid of consumption habit stocks on which to solve the problem.
    FOCinverter : HabitFormationWealthInUtilityInverter
        Function that inverts the first order conditions to yield optimal consumption
        and the decision-time habit stock from which it was chosen.
    HabitWgt : float
        Exponent on habit stock, which is used as a divisor on consumption in
        the utility function. Should be on unit interval.
    HabitRte : float
        Rate at which habit stock is updated by new consumption: H = lambda*c + (1-lambda)*h.
        Should be on the unit interval.
    WealthShare : float
        Cobb-Douglas share for wealth (assets) in the utility function.
    WealthShift : float
        Non-negative additive shifter for wealth in the utility function.
    mXtraGrid : np.array
        Dense grid of market resources, used to "re-interpolate" the curvilinear
        consumption function onto a rectilinear grid.
    hGridDense : np.array
        Dense grid of habit stocks, used to "re-interpolate" the curvilinear
        consumption function onto a rectilinear grid.

    Returns
    -------
    solution_now : dict
        Solution to this period's problem, with the following keys:
        cFunc : Consumption function over (mNrm, hNrm).
        ShareFunc : Risky asset share function over (wNrm, HNrm).
        dvdkFunc : Marginal value of beginning-of-period capital, defined on (kNrm, hPre).
        dvdhFunc : Marginal value of beginning-of-period habit stock, defined on (kNrm, hPre).
        kNrmMin : Minimum allowable beginning-of-period capital.
    """
    U = UtilityFuncCRRA(CRRA)
    DiscFacEff = DiscFac * LivPrb

    # Make end-of-period state grids
    aNrmMin = np.maximum(solution_next["kNrmMin"], -WealthShift)
    aGrid = aXtraGrid + aNrmMin
    aNrm, HNrm = np.meshgrid(aGrid, HabitGrid, indexing="ij")

    # Solve for this period's consumption function
    if type(solution_next["dvdkFunc"]) is ConstantFunction:
        # This is the terminal period, and the consumption function is linear (maybe with a kink)
        if (WealthShift > 0.0) and (WealthShare > 0.0):
            m_cusp = (1 - WealthShare) / WealthShare * WealthShift
            m_terminal = np.array([0.0, m_cusp, m_cusp + 1.0])
            c_terminal = np.array([0.0, m_cusp, m_cusp + (1.0 - WealthShare)])
        else:
            m_terminal = np.array([0.0, 1.0])
            c_terminal = np.array([0.0, 1.0 - WealthShare])
        cFunc_base = LinearInterp(m_terminal, c_terminal)
        cFunc = LinearInterpOnInterp1D([cFunc_base, cFunc_base], np.array([0.0, 1.0]))
        mNrmMin = 0.0

    else:
        # Evaluate end-of-period marginal value on those grids
        EndOfPrd_dvda = DiscFacEff * solution_next["dvdkFunc"](aNrm, HNrm)
        EndOfPrd_dvdH = DiscFacEff * solution_next["dvdhFunc"](aNrm, HNrm)

        # Calculate chi, the transformation of end-of-period marginal value
        chi_numer = U.derinv(EndOfPrd_dvda - HabitRte * EndOfPrd_dvdH)
        aShift = aNrm + WealthShift
        chi_denom = aShift ** (
            -HabitWgt * (1.0 - WealthShare) * (1.0 - 1.0 / CRRA) + 1.0
        )
        chi = chi_numer / chi_denom

        # Recover c and h using the FOC inverter, then find endogenous m gridpoints
        cNrm, hNrm = FOCinverter(aNrm, HNrm, chi)
        mNrm = aNrm + cNrm

        # Construct the unconstrained consumption as a Curvilinear2Dinterp
        cNrm = np.concatenate((np.zeros((1, HabitGrid.size)), cNrm), axis=0)
        mNrm = np.concatenate((aNrmMin * np.ones((1, HabitGrid.size)), mNrm), axis=0)
        hBot = (
            np.reshape(hNrm[0, :], (1, HabitGrid.size))
            if HabitRte == 1.0
            else np.reshape(HabitGrid / (1.0 - HabitRte), (1, HabitGrid.size))
        )
        hNrm = np.concatenate((hBot, hNrm), axis=0)
        cFuncUnc_base = Curvilinear2DInterp(cNrm, mNrm, hNrm)

        # Re-interpolate the curvilinear consumption function onto an ordinary grid
        mGridDense = mXtraGrid + aNrmMin
        mMesh, hMesh = np.meshgrid(mGridDense, hGridDense, indexing="ij")
        cMesh = cFuncUnc_base(mMesh, hMesh)
        cMesh = np.concatenate((np.zeros((mGridDense.size, 1)), cMesh), axis=1)
        cMesh = np.concatenate((np.zeros((1, hGridDense.size + 1)), cMesh), axis=0)
        cFuncUnc = BilinearInterp(
            cMesh, np.insert(mGridDense, 0, aNrmMin), np.insert(hGridDense, 0, 0.0)
        )

        # Add the constrained consumption function to that
        if (BoroCnstArt is not None) and (BoroCnstArt > -np.inf):
            cFuncCnst_temp = LinearInterp([BoroCnstArt, BoroCnstArt + 1.0], [0.0, 1.0])
            cFuncCnst = LinearInterpOnInterp1D(
                [cFuncCnst_temp, cFuncCnst_temp], np.array([0.0, 1.0])
            )
            cFunc = LowerEnvelope2D(cFuncUnc, cFuncCnst)
            mNrmMin = np.maximum(aNrmMin, BoroCnstArt)
        else:
            cFunc = cFuncUnc
            mNrmMin = aNrmMin

    # Calculate the natural borrowing constraint
    PermShkVals = IncShkDstn.atoms[0, :]
    TranShkVals = IncShkDstn.atoms[1, :]
    kNrmMin_cand = (mNrmMin - TranShkVals) / Rfree * (PermShkVals * PermGroFac)
    kNrmMin = np.max(kNrmMin_cand)

    # Make beginning-of-period state grids
    kGrid = kNrmMin + aXtraGrid
    kNrm, hPre = np.meshgrid(kGrid, HabitGrid, indexing="ij")

    # Compute expected marginal value over income shocks from beginning-of-period states
    dvdk, dvdh = expected(
        calc_marg_values,
        IncShkDstn,
        args=(
            kNrm,
            hPre,
            CRRA,
            Rfree,
            PermGroFac,
            HabitWgt,
            HabitRte,
            WealthShare,
            WealthShift,
            DiscFacEff,
            cFunc,
            solution_next["dvdhFunc"],
        ),
    )

    # Transform and package the marginal value functions
    dvdkNvrs = np.concatenate((np.zeros((1, HabitGrid.size)), U.derinv(dvdk)), axis=0)
    dvdkNvrsFunc = BilinearInterp(dvdkNvrs, np.insert(kGrid, 0, kNrmMin), HabitGrid)
    dvdkFunc = MargValueFuncCRRA(dvdkNvrsFunc, CRRA)
    dvdhNvrs = U.inv(dvdh)
    dvdhNvrsFunc = BilinearInterp(dvdhNvrs, kGrid, HabitGrid)
    dvdhFunc = ValueFuncCRRA(dvdhNvrsFunc, CRRA)

    # Package the solution as a dictionary and return it
    solution_now = {
        "cFunc": cFunc,
        "dvdkFunc": dvdkFunc,
        "dvdhFunc": dvdhFunc,
        "kNrmMin": kNrmMin,
        "distance_criteria": ["cFunc"],
    }
    return solution_now


###############################################################################


def solve_one_period_WealthHabitPortfolio(
    solution_next,
    IncShkDstn,
    RiskyDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rfree,
    PermGroFac,
    BoroCnstArt,
    aXtraGrid,
    HabitGrid,
    ShareGrid,
    ShareLimit,
    FOCinverter,
    HabitWgt,
    HabitRte,
    WealthShare,
    WealthShift,
    mXtraGrid,
    hGridDense,
):
    """
    Solve one period of the consumption-saving model with habit formation, wealth
    in the utility function, and portfolio allocation. This solver uses a modular
    approach, separating the portfolio allocation and consumption-saving problems.

    Parameters
    ----------
    solution_next : dict
        Dictionary with next period's solution.
    IncShkDstn : DiscreteDistribution
        Discretized permanent and transitory income shock distribution this period.
    RiskyDstn : DiscreteDistribution
        Discretized risky asset return distribution.
    LivPrb : float
        Survival probability at the end of this period.
    DiscFac : float
        Intertemporal discount factor.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Interest factor on capital at the start of this period.
    PermGroFac : float
        Permanent income growth factor at the start of this period.
    BoroCnstArt : float or None
        Artificial borrowing constraint on assets at the end of this period,
        as a fraction of permanent income.
    aXtraGrid : np.array
        Grid of "assets above minimum".
    HabitGrid : np.array
        Grid of consumption habit stocks on which to solve the problem.
    ShareGrid : np.array
        Grid of risky share values on [0,1].
    ShareLimit : float
        Merton-Samuelson limiting share as wealth -> infinity.
    FOCinverter : HabitFormationWealthInUtilityInverter
        Function that inverts the first order conditions to yield optimal consumption
        and the decision-time habit stock from which it was chosen.
    HabitWgt : float
        Exponent on habit stock, which is used as a divisor on consumption in
        the utility function. Should be on unit interval.
    HabitRte : float
        Rate at which habit stock is updated by new consumption: H = lambda*c + (1-lambda)*h.
        Should be on the unit interval.
    WealthShare : float
        Cobb-Douglas share for wealth (assets) in the utility function.
    WealthShift : float
        Non-negative additive shifter for wealth in the utility function.
    mXtraGrid : np.array
        Dense grid of market resources, used to "re-interpolate" the curvilinear
        consumption function onto a rectilinear grid.
    hGridDense : np.array
        Dense grid of habit stocks, used to "re-interpolate" the curvilinear
        consumption function onto a rectilinear grid.

    Returns
    -------
    solution_now : dict
        Solution to this period's problem, with the following keys:
        cFunc : Consumption function over (mNrm, hNrm).
        dvdkFunc : Marginal value of beginning-of-period capital, defined on (kNrm, hPre).
        dvdhFunc : Marginal value of beginning-of-period habit stock, defined on (kNrm, hPre).
        kNrmMin : Minimum allowable beginning-of-period capital.
    """
    # Solve the portfolio allocation problem, yielding the "mid period solution"
    solution_mid = solve_optimal_share_habit(
        solution_next,
        RiskyDstn,
        LivPrb,
        DiscFac,
        CRRA,
        Rfree,
        BoroCnstArt,
        aXtraGrid,
        HabitGrid,
        ShareGrid,
        ShareLimit,
    )

    # Solve the consumption-saving problem, yielding the solution to this period
    solution_now = solve_one_period_ConsWealthHabit(
        solution_mid,
        IncShkDstn,
        1.0,  # LivPrb accounted for above, turn off
        1.0,  # DiscFac accounted for above, turn off
        CRRA,
        1.0,  # Rfree accounted for above, turn off
        PermGroFac,
        BoroCnstArt,
        aXtraGrid,
        HabitGrid,
        FOCinverter,
        HabitWgt,
        HabitRte,
        WealthShare,
        WealthShift,
        mXtraGrid,
        hGridDense,
    )

    # Add the risky share function to the period's solution and return it
    solution_now["ShareFunc"] = solution_mid["ShareFunc"]
    return solution_now


###############################################################################

# Make a dictionary of constructors for the habit formation model
WealthHabitConsumerType_constructors_default = {
    "kNrmInitDstn": make_lognormal_kNrm_init_dstn,
    "pLvlInitDstn": make_lognormal_pLvl_init_dstn,
    "HabitInitDstn": make_lognormal_habit_init_dstn,
    "IncShkDstn": construct_lognormal_income_process_unemployment,
    "PermShkDstn": get_PermShkDstn_from_IncShkDstn,
    "TranShkDstn": get_TranShkDstn_from_IncShkDstn,
    "aXtraGrid": make_assets_grid,
    "FOCinverter": make_wealth_habit_inverter,
    "HabitGrid": make_habit_grid,
    "DenseGrids": make_dense_grids,
    "mXtraGrid": get_it_from("DenseGrids"),
    "hGridDense": get_it_from("DenseGrids"),
    "solution_terminal": make_habit_solution_terminal,
}

# Make a dictionary with parameters for the default constructor for kNrmInitDstn
WealthHabitConsumerType_kNrmInitDstn_default = {
    "kLogInitMean": -12.0,  # Mean of log initial capital
    "kLogInitStd": 0.0,  # Stdev of log initial capital
    "kNrmInitCount": 15,  # Number of points in initial capital discretization
}

# Make a dictionary with parameters for the default constructor for pLvlInitDstn
WealthHabitConsumerType_pLvlInitDstn_default = {
    "pLogInitMean": 0.0,  # Mean of log permanent income
    "pLogInitStd": 0.0,  # Stdev of log permanent income
    "pLvlInitCount": 15,  # Number of points in initial capital discretization
}

# Make a dictionary with parameters for the default constructor for HabitInitDstn
WealthHabitConsumerType_HabitInitDstn_default = {
    "hLogInitMean": -0.5,  # Mean of log habit stock
    "hLogInitStd": 0.2,  # Stdev of log initial habit stock
    "HabitInitCount": 15,  # Number of points in initial habit stock discretization
}

# Default parameters to make IncShkDstn using construct_lognormal_income_process_unemployment
WealthHabitConsumerType_IncShkDstn_default = {
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

# Default parameters to make aXtraGrid using make_assets_grid
WealthHabitConsumerType_aXtraGrid_default = {
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 30,  # Maximum end-of-period "assets above minimum" value
    "aXtraNestFac": 2,  # Exponential nesting factor for aXtraGrid
    "aXtraCount": 100,  # Number of points in the grid of "assets above minimum"
    "aXtraExtra": None,  # Additional other values to add in grid (optional)
}

# Default parameters to make HabitGrid using make_habit_grid
WealthHabitConsumerType_HabitGrid_default = {
    "HabitMin": 0.2,
    "HabitMax": 5.0,
    "HabitCount": 41,
    "HabitOrder": 2.0,
}

# Default parameters to make the FOC inverter
WealthHabitConsumerType_inverter_default = {
    "EtaMin": 1e-4,
    "EtaMax": 100.0,
    "EtaCount": 301,
    "EtaOrder": 2.5,
}

# Make a dictionary to specify an habit formation consumer type
WealthHabitConsumerType_solving_default = {
    # BASIC HARK PARAMETERS REQUIRED TO SOLVE THE MODEL
    "cycles": 1,  # Finite, non-cyclic model
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "pseudo_terminal": True,  # It's a fake stub
    "constructors": WealthHabitConsumerType_constructors_default,  # See dictionary above
    # PRIMITIVE RAW PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": [1.03],  # Interest factor on retained assets
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability after each period
    "PermGroFac": [1.01],  # Permanent income growth factor
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint
    "HabitWgt": 0.5,  # Weight on consumption habit; exponent on habit divisor in utility
    "HabitRte": 0.2,  # Speed of consumption habit updating
    "WealthShare": 0.2,  # Cobb-Douglas share on assets in the utility function
    "WealthShift": 0.0,  # Additive shifter for wealth in the utility function
    "DenseFactor": 3,  # Density factor for re-interpolation
}
WealthHabitConsumerType_simulation_default = {
    # PARAMETERS REQUIRED TO SIMULATE THE MODEL
    "AgentCount": 10000,  # Number of agents of this type
    "T_age": None,  # Age after which simulated agents are automatically killed
}

WealthHabitConsumerType_defaults = {}
WealthHabitConsumerType_defaults.update(WealthHabitConsumerType_IncShkDstn_default)
WealthHabitConsumerType_defaults.update(WealthHabitConsumerType_kNrmInitDstn_default)
WealthHabitConsumerType_defaults.update(WealthHabitConsumerType_pLvlInitDstn_default)
WealthHabitConsumerType_defaults.update(WealthHabitConsumerType_HabitInitDstn_default)
WealthHabitConsumerType_defaults.update(WealthHabitConsumerType_aXtraGrid_default)
WealthHabitConsumerType_defaults.update(WealthHabitConsumerType_HabitGrid_default)
WealthHabitConsumerType_defaults.update(WealthHabitConsumerType_inverter_default)
WealthHabitConsumerType_defaults.update(WealthHabitConsumerType_solving_default)
WealthHabitConsumerType_defaults.update(WealthHabitConsumerType_simulation_default)


class WealthHabitConsumerType(HabitConsumerType):
    r"""
    A class for representing consumers who form consumption habits and derive utility
    directly from holding wealth. Agents get flow utility according to a CRRA felicity
    function that depends on a Cobb-Douglas combination of habit-modified consumption
    and retained assets. The habit stock evolves as a weighted average of current con-
    sumption and prior habit stock. Consumers can save in a single risk-free asset.

    .. math::
        \newcommand{\CRRA}{\rho}
        \newcommand{\LivPrb}{\mathsf{S}}
        \newcommand{\PermGroFac}{\Gamma}
        \newcommand{\Rfree}{\mathsf{R}}
        \newcommand{\DiscFac}{\beta}
        \newcommand{\HabitWgt}{\alpha}
        \newcommand{\HabitRte}{\lambda}
        \newcommand{\WealthShare}{\omega}
        \newcommand{\WealthShift}{\xi}

        \begin{align*}
        v_t(m_t,h_t) &= \max_{c_t}u(c_t,h_t,a_t) + \DiscFac \LivPrb_t \mathbb{E}_{t} \left[ (\PermGroFac_{t+1} \psi_{t+1})^{(1-(1-\WealthShare)\HabitWgt)(1-\CRRA)} v_{t+1}(m_{t+1}, h_{t+1}) \right] \\
        & \text{s.t.}  \\
        a_t &= m_t - c_t, \\
        H_t &= \HabitRte c_t + (1-\HabitRte) h_t, \\
        a_t &\geq \underline{a}, \\
        m_{t+1} &= a_t \Rfree_{t+1}/(\PermGroFac_{t+1} \psi_{t+1}) + \theta_{t+1}, \\
        h_{t+1} &= H_t / (\PermGroFac_{t+1} \psi_{t+1}), \\
        (\psi_{t+1},\theta_{t+1}) &\sim F_{t+1}, \\
        \mathbb{E}[\psi] &= 1, \\
        u(c,h,a) &= \frac{\left( (c/h^\HabitWgt)^{1-\WealthShare} (a+\WealthShift)^\WealthShare \right)^{1-\CRRA}}{1-\CRRA}.
        \end{align*}
    """

    default_ = {
        "params": WealthHabitConsumerType_defaults,
        "solver": solve_one_period_ConsWealthHabit,
        "model": "ConsHabit.yaml",
        "track_vars": ["aNrm", "cNrm", "mNrm", "hNrm", "pLvl"],
    }

    time_inv_ = HabitConsumerType.time_inv_ + ["WealthShare", "WealthShift"]


###############################################################################

# Make a dictionary of constructors for the wealth-habit-portfolio model
WealthHabitPortfolio_constructors_default = (
    WealthHabitConsumerType_constructors_default.copy()
)
WealthHabitPortfolio_additional_constructors = {
    "RiskyDstn": make_lognormal_RiskyDstn,
    "ShareGrid": make_simple_ShareGrid,
    "ShareLimit": calc_ShareLimit_for_CRRA,
}
WealthHabitPortfolio_constructors_default.update(
    WealthHabitPortfolio_additional_constructors
)

WealthHabitPortfolio_RiskyDstn_default = {
    "RiskyAvg": 1.08,
    "RiskyStd": 0.18,
    "RiskyCount": 5,
}

WealthHabitPortfolio_ShareGrid_default = {
    "ShareCount": 26,
}

WealthHabitPortfolioConsumerType_defaults = WealthHabitConsumerType_defaults.copy()
WealthHabitPortfolioConsumerType_defaults["constructors"] = (
    WealthHabitPortfolio_constructors_default
)
WealthHabitPortfolioConsumerType_defaults.update(WealthHabitPortfolio_RiskyDstn_default)
WealthHabitPortfolioConsumerType_defaults.update(WealthHabitPortfolio_ShareGrid_default)


class WealthHabitPortfolioConsumerType(WealthHabitConsumerType):
    r"""
    A class for representing consumers who form consumption habits, get utility
    directly from holding wealth, and can allocate their wealth between a risky
    and riskless asset. Agents get flow utility according to a CRRA felicity function
    that depends on current consumption, the habit stock, and retained wealth.
    The habit stock evolves as a weighted average of current consumption and prior
    habit stock.

    This type's solver uses a "modular" approach that separates the portfolio allo-
    cation and consumption-saving problems into two functions. The latter function
    is just the solver for WealthHabitConsumerType. Consequently, the consumption
    function is defined over (m,h) but the risky share function is defined over (w,H).

    .. math::
        \newcommand{\CRRA}{\rho}
        \newcommand{\LivPrb}{\mathsf{S}}
        \newcommand{\PermGroFac}{\Gamma}
        \newcommand{\Rfree}{\mathsf{R}}
        \newcommand{\Risky}{\mathfrak{R}}
        \newcommand{\DiscFac}{\beta}
        \newcommand{\HabitWgt}{\alpha}
        \newcommand{\HabitRte}{\lambda}
        \newcommand{\WealthShare}{\omega}
        \newcommand{\WealthShift}{\xi}

        \begin{align*}
        v_t(m_t,h_t) &= \max_{c_t, s_t} u(c_t,h_t,w_t) + \DiscFac \LivPrb_t
            \mathbb{E}_{t} \left[ (\PermGroFac_{t+1} \psi_{t+1})^{(1-\HabitWgt)(1-\CRRA)}
            v_{t+1}(m_{t+1}, h_{t+1}) \right] \\
        & \text{s.t.}  \\
        w_t &= m_t - c_t, \\
        H_t &= \HabitRte c_t + (1-\HabitRte) h_t, \\
        w_t &\geq 0, \\
        s_t &\in [0,1], \\
        a_t &= R_t w_t, \\
        R_{t} &= s_t \Risky_{t} + (1-s_t) \Rfree_{t}, \\
        m_{t+1} &= a_t / (\PermGroFac_{t+1} \psi_{t+1}) + \theta_{t+1}, \\
        h_{t+1} &= H_t / (\PermGroFac_{t+1} \psi_{t+1}), \\
        u(c,h,w) &= \frac{((c/h^\HabitWgt)^{1-\WealthShare} (w+\WealthShift)^{\WealthShare}})^{1-\CRRA}}{1-\CRRA}.
        \end{align*}
    """

    default_ = {
        "params": WealthHabitPortfolioConsumerType_defaults,
        "solver": solve_one_period_WealthHabitPortfolio,
        "model": "ConsHabitPortfolioAlt.yaml",
        "track_vars": ["aNrm", "cNrm", "mNrm", "hNrm", "Share", "pLvl"],
    }

    time_inv_ = WealthHabitConsumerType.time_inv_ + ["RiskyDstn", "ShareGrid"]
    time_vary_ = WealthHabitConsumerType.time_vary_ + ["ShareLimit"]
    shock_vars_ = WealthHabitConsumerType.shock_vars_ + ["Risky"]
    distributions = WealthHabitConsumerType.distributions + ["RiskyDstn"]
