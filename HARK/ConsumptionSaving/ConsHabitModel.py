"""
This module has consumption-models with habit formation. Right now, it only has
a single basic model with permanent and transitory income shocks, one risk-free
asset, and a habit stock that evolves as a weighted average of current consumption
and the prior habit stock.
"""

import numpy as np
from HARK.utilities import make_exponential_grid
from HARK.interpolation import (
    LinearInterp,
    LinearInterpOnInterp1D,
    ConstantFunction,
    Curvilinear2DInterp,
    LowerEnvelope2D,
    IdentityFunction,
    BilinearInterp,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.distributions import expected, Lognormal
from HARK.core import AgentType
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


def make_lognormal_habit_init_dstn(hLogInitMean, hLogInitStd, HabitInitCount, RNG):
    """
    Construct a lognormal distribution for (normalized) initial habit stock
    of newborns, hNrm. This is the default constructor for HabitInitDstn.

    Parameters
    ----------
    hLogInitMean : float
        Mean of log habit stockfor newborns.
    hLogInitStd : float
        Stdev of log habit stock for newborns.
    HabitInitCount : int
        Number of points in the discretization.
    RNG : np.random.RandomState
        Agent's internal RNG.

    Returns
    -------
    HabitInitDstn : DiscreteDistribution
        Discretized distribution of initial capital holdings for newborns.
    """
    dstn = Lognormal(
        mu=hLogInitMean,
        sigma=hLogInitStd,
        seed=RNG.integers(0, 2**31 - 1),
    )
    HabitInitDstn = dstn.discretize(HabitInitCount)
    return HabitInitDstn


class HabitFormationInverter:
    """
    A class for solving the first order conditions of a consumption-saving model
    with habit formation. In this notation, HabitRte is a parameter on the unit
    interval representing how fast the habit stock evolves; a value of zero means
    no habit dynamics and a value of one means that H_t = c_t, complete updating.
    HabitWgt is also on the unit interval and represents the exponent on the
    habit stock, which is used as a divisor on consumption in the utility function.

    Instances of this class take two arguments when called as a function: end-of-
    period habit stock H and transformed end-of-period marginal value chi.

    chi = (W_a(a,H) - alpha * W_H(a,H)) ** (-1/CRRA)

    a = m - c
    H = alpha * c + (1-alpha) * h
    m' = R a / psi + theta
    h' = H / psi
    """

    def __init__(
        self,
        CRRA,
        HabitRte,
        HabitWgt,
        ChiMax,
        ChiCount,
        ChiOrder,
        HabitMax,
        HabitCount,
        HabitOrder,
    ):
        xGrid = make_exponential_grid(0.0, ChiMax, ChiCount, ChiOrder)
        hGrid = make_exponential_grid(0.0, HabitMax, HabitCount, HabitOrder)
        hMesh, xMesh = np.meshgrid(hGrid, xGrid, indexing="ij")

        PostHabit = (
            HabitRte * hMesh ** (HabitWgt * (1.0 - 1.0 / CRRA)) * xMesh
            + (1.0 - HabitRte) * hMesh
        )

        func_by_chi = [LinearInterp(PostHabit[:, j], hGrid) for j in range(ChiCount)]
        func = LinearInterpOnInterp1D(func_by_chi, xGrid)

        self.hFunc = func
        self.alpha = HabitRte

    def __call__(self, H, chi):
        h = self.hFunc(H, chi)
        alpha = self.alpha
        c = (H - (1 - alpha) * h) / (alpha)
        return c, h

    def cFunc(self, H, chi):
        h = self.hFunc(H, chi)
        alpha = self.alpha
        c = (H - (1 - alpha) * h) / (alpha)
        return c


def make_inverter(
    CRRA,
    HabitRte,
    HabitWgt,
    ChiMax,
    ChiCount,
    ChiOrder,
    HabitMax,
    HabitCount,
    HabitOrder,
):
    return HabitFormationInverter(
        CRRA,
        HabitRte,
        HabitWgt,
        ChiMax,
        ChiCount,
        ChiOrder,
        HabitMax,
        HabitCount,
        HabitOrder,
    )


def make_habit_grid(HabitMin, HabitMax, HabitCount, HabitOrder):
    return make_exponential_grid(HabitMin, HabitMax, HabitCount, HabitOrder)


def make_habit_solution_terminal():
    """
    Make a pseudo-terminal solution for the habit formation model, which has zero
    functions for (marginal) value.
    """
    dvdkFunc_terminal = ConstantFunction(0.0)
    dvdhFunc_terminal = ConstantFunction(0.0)
    solution_terminal = {
        "dvdkFunc": dvdkFunc_terminal,
        "dvdhFunc": dvdhFunc_terminal,
        "kNrmMin": 0.0,
    }
    return solution_terminal


def calc_marg_values(S, k, hpre, rho, R, G, gamma, alpha, beta, C, Vp):
    psi = S["PermShk"]
    theta = S["TranShk"]
    gro = psi * G
    m = R * k / gro + theta
    h = hpre / gro
    c = C(m, h)
    a = m - c
    H = alpha * c + (1 - alpha) * h
    dvdH = beta * Vp(a, H)
    temp = h ** (-gamma * (1 - rho))
    dudc = temp * c ** (-rho)
    dudh = c ** (1 - rho) * (-gamma) * temp / h
    dvdm = dudc + alpha * dvdH
    dvdh = dudh + (1 - alpha) * dvdH
    gro_adj = gro ** ((1 - rho) * (1 - gamma) - 1.0)
    dvdk = gro_adj * dvdm
    dvdh = gro_adj * dvdh
    return dvdk, dvdh


###############################################################################


def solve_one_period_ConsHabit(
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
):
    """
    Solve one period of the consumption-saving model with habit formation.

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
    FOCinverter : HabitFormationInverter
        Function that inverts the first order conditions to yield optimal consumption
        and the decision-time habit stock from which it was chose.
    HabitWgt : float
        Exponent on habit stock, which is used as a divisor on consumption in
        the utility function: U(c,h) = u(c / h**gamma). Should be on unit interval.
    HabitRte : float
        Rate at which habit stock is updated by new consumption: H = alpha*c + (1-alpha)*h.
        Should be on the unit interval.

    Returns
    -------
    solution_now : dict
        Solution to this period's problem, with the following keys:
        cFunc : Consumption function over (mNrm, hNrm).
        dvdkFunc : Marginal value of beginning-of-period capital, defined on (kNrm, hPre).
        dvdhFunc : Marginal value of beginning-of-period habit stock, defined on (kNrm, hPre).
        kNrmMin : Minimum allowable beginning-of-period capital.
    """
    U = UtilityFuncCRRA(CRRA)

    # Make end-of-period state grids
    aNrmMin = solution_next["kNrmMin"]
    aGrid = aXtraGrid + aNrmMin
    aNrm, HNrm = np.meshgrid(aGrid, HabitGrid, indexing="ij")

    # Calculate the natural borrowing constraint
    DiscFacEff = DiscFac * LivPrb
    PermShkVals = IncShkDstn.atoms[0, :]
    TranShkVals = IncShkDstn.atoms[1, :]
    BoroCnstNat_cand = (aNrmMin - TranShkVals) / Rfree * PermShkVals
    BoroCnstNat = np.max(BoroCnstNat_cand)

    if type(solution_next["dvdkFunc"]) is ConstantFunction:
        # This is the terminal period, and the consumption function is to consume all
        cFunc = IdentityFunction(i_dim=0, n_dims=2)
        kNrmMin = BoroCnstNat

    else:
        # Evaluate end-of-period marginal value on those grids, then calculate chi
        EndOfPrd_dvda = DiscFacEff * solution_next["dvdkFunc"](aNrm, HNrm)
        EndOfPrd_dvdH = DiscFacEff * solution_next["dvdhFunc"](aNrm, HNrm)
        chi = (EndOfPrd_dvda - HabitRte * EndOfPrd_dvdH) ** (-1.0 / CRRA)

        # Recover c and h using the FOC inverter, then find endogenous m gridpoints
        cNrm, hNrm = FOCinverter(HNrm, chi)
        mNrm = aNrm + cNrm

        # Construct the unconstrained consumption as a Curvilinear2Dinterp
        cNrm = np.concatenate((np.zeros((1, HabitGrid.size)), cNrm), axis=0)
        mNrm = np.concatenate((aNrmMin * np.ones((1, HabitGrid.size)), mNrm), axis=0)
        hBot = np.reshape(HabitGrid / (1.0 - HabitRte), (1, HabitGrid.size))
        hNrm = np.concatenate((hBot, hNrm), axis=0)
        cFuncUnc = Curvilinear2DInterp(cNrm, mNrm, hNrm)

        # Add the constrained consumption function to that
        if (BoroCnstArt is not None) and (BoroCnstArt > -np.inf):
            cFuncCnst_temp = LinearInterp([BoroCnstArt, BoroCnstArt + 1.0], [0.0, 1.0])
            cFuncCnst = LinearInterpOnInterp1D(
                [cFuncCnst_temp, cFuncCnst_temp], np.array([0.0, 1.0])
            )
            cFunc = LowerEnvelope2D(cFuncUnc, cFuncCnst)
            kNrmMin = np.maximum(BoroCnstArt, BoroCnstNat)
        else:
            cFunc = cFuncUnc
            kNrmMin = BoroCnstNat

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
            DiscFacEff,
            cFunc,
            solution_next["dvdhFunc"],
        ),
    )
    dvdk *= Rfree

    # Transform and package the marginal value functions
    dvdkNvrs = np.concatenate((np.zeros((1, HabitGrid.size)), U.derinv(dvdk)), axis=0)
    dvdkNvrsFunc = BilinearInterp(dvdkNvrs, np.insert(kGrid, 0, kNrmMin), HabitGrid)
    dvdkFunc = MargValueFuncCRRA(dvdkNvrsFunc, CRRA)
    dvdhNvrs = np.concatenate((np.zeros((1, HabitGrid.size)), U.inv(dvdh)), axis=0)
    dvdhNvrsFunc = BilinearInterp(dvdhNvrs, np.insert(kGrid, 0, kNrmMin), HabitGrid)
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


# Make a dictionary of constructors for the habit formation model
HabitConsumerType_constructors_default = {
    "kNrmInitDstn": make_lognormal_kNrm_init_dstn,
    "pLvlInitDstn": make_lognormal_pLvl_init_dstn,
    "HabitInitDstn": make_lognormal_habit_init_dstn,
    "IncShkDstn": construct_lognormal_income_process_unemployment,
    "PermShkDstn": get_PermShkDstn_from_IncShkDstn,
    "TranShkDstn": get_TranShkDstn_from_IncShkDstn,
    "aXtraGrid": make_assets_grid,
    "FOCinverter": make_inverter,
    "HabitGrid": make_habit_grid,
    "solution_terminal": make_habit_solution_terminal,
}

# Make a dictionary with parameters for the default constructor for kNrmInitDstn
HabitConsumerType_kNrmInitDstn_default = {
    "kLogInitMean": -12.0,  # Mean of log initial capital
    "kLogInitStd": 0.0,  # Stdev of log initial capital
    "kNrmInitCount": 15,  # Number of points in initial capital discretization
}

# Make a dictionary with parameters for the default constructor for pLvlInitDstn
HabitConsumerType_pLvlInitDstn_default = {
    "pLogInitMean": 0.0,  # Mean of log permanent income
    "pLogInitStd": 0.0,  # Stdev of log permanent income
    "pLvlInitCount": 15,  # Number of points in initial capital discretization
}

# Make a dictionary with parameters for the default constructor for HabitInitDstn
HabitConsumerType_HabitInitDstn_default = {
    "hLogInitMean": -0.5,  # Mean of log habit stock
    "hLogInitStd": 0.2,  # Stdev of log initial habit stock
    "HabitInitCount": 15,  # Number of points in initial habit stock discretization
}

# Default parameters to make IncShkDstn using construct_lognormal_income_process_unemployment
HabitConsumerType_IncShkDstn_default = {
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
HabitConsumerType_aXtraGrid_default = {
    "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
    "aXtraMax": 20,  # Maximum end-of-period "assets above minimum" value
    "aXtraNestFac": 2,  # Exponential nesting factor for aXtraGrid
    "aXtraCount": 48,  # Number of points in the grid of "assets above minimum"
    "aXtraExtra": None,  # Additional other values to add in grid (optional)
}

# Default parameters to make HabitGrid using make_habit_grid
HabitConsumerType_HabitGrid_default = {
    "HabitMin": 0.2,
    "HabitMax": 5.0,
    "HabitCount": 51,
    "HabitOrder": 1.5,
}

# Default parameters to make the FOC inverter
HabitConsumerType_inverter_default = {
    "ChiMax": 50.0,
    "ChiCount": 251,
    "ChiOrder": 1.5,
}

# Make a dictionary to specify an habit formation consumer type
HabitConsumerType_solving_default = {
    # BASIC HARK PARAMETERS REQUIRED TO SOLVE THE MODEL
    "cycles": 1,  # Finite, non-cyclic model
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "pseudo_terminal": True,  # It's a fake stub
    "constructors": HabitConsumerType_constructors_default,  # See dictionary above
    # PRIMITIVE RAW PARAMETERS REQUIRED TO SOLVE THE MODEL
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": [1.03],  # Interest factor on retained assets
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability after each period
    "PermGroFac": [1.01],  # Permanent income growth factor
    "BoroCnstArt": 0.0,  # Artificial borrowing constraint
    "HabitWgt": 0.5,  # Weight on consumption habit
    "HabitRte": 0.2,  # Speed of consumption habit updating
}
HabitConsumerType_simulation_default = {
    # PARAMETERS REQUIRED TO SIMULATE THE MODEL
    "AgentCount": 10000,  # Number of agents of this type
    "T_age": None,  # Age after which simulated agents are automatically killed
}

HabitConsumerType_defaults = {}
HabitConsumerType_defaults.update(HabitConsumerType_IncShkDstn_default)
HabitConsumerType_defaults.update(HabitConsumerType_kNrmInitDstn_default)
HabitConsumerType_defaults.update(HabitConsumerType_pLvlInitDstn_default)
HabitConsumerType_defaults.update(HabitConsumerType_HabitInitDstn_default)
HabitConsumerType_defaults.update(HabitConsumerType_aXtraGrid_default)
HabitConsumerType_defaults.update(HabitConsumerType_HabitGrid_default)
HabitConsumerType_defaults.update(HabitConsumerType_inverter_default)
HabitConsumerType_defaults.update(HabitConsumerType_solving_default)
HabitConsumerType_defaults.update(HabitConsumerType_simulation_default)


class HabitConsumerType(AgentType):
    r"""
    A class for representing consumers who form consumption habits. Agents get
    flow utility according to a CRRA felicity function that depends on both current
    consumption and the habit stock h_t. The habit stock evolves as a weighted
    average of current consumption and prior habit stock. Consumers can save in
    a single risk-free asset, so this model is an extension of the workhorse
    IndShockConsumerType to include a habit stock.

    .. math::
        \newcommand{\CRRA}{\rho}
        \newcommand{\LivPrb}{\mathsf{S}}
        \newcommand{\PermGroFac}{\Gamma}
        \newcommand{\Rfree}{\mathsf{R}}
        \newcommand{\DiscFac}{\beta}
        \newcommand{\HabitWgt}{\gamma}
        \newcommand{\HabitRte}{\alpha}
        \begin{align*}
        v_t(m_t,h_t) &= \max_{c_t,h_t}u(c_t) + \DiscFac \LivPrb_t \mathbb{E}_{t} \left[ ((1-\HabitWgt)(\PermGroFac_{t+1} \psi_{t+1}))^{1-\CRRA} v_{t+1}(m_{t+1}) \right], \\
        & \text{s.t.}  \\
        a_t &= m_t - c_t, \\
        H_t &= \HabitRte c_t + (1-\HabitRte) h_t, \\
        a_t &\geq \underline{a}, \\
        m_{t+1} &= a_t \Rfree_{t+1}/(\PermGroFac_{t+1} \psi_{t+1}) + \theta_{t+1}, \\
        h_{t+1} &= H_t / (\PermGroFac_{t+1} \psi_{t+1}), \\
        (\psi_{t+1},\theta_{t+1}) &\sim F_{t+1}, \\
        \mathbb{E}[\psi] &= 1, \\
        u(c,h) &= \frac{(c/h^\HabitWgt)^{1-\CRRA}}{1-\CRRA}.
        \end{align*}
    """

    default_ = {
        "params": HabitConsumerType_defaults,
        "solver": solve_one_period_ConsHabit,
        "model": "ConsHabit.yaml",
        "track_vars": ["aNrm", "cNrm", "mNrm", "hNrm", "pLvl"],
    }

    time_inv_ = [
        "DiscFac",
        "CRRA",
        "BoroCnstArt",
        "aXtraGrid",
        "HabitGrid",
        "FOCinverter",
        "HabitWgt",
        "HabitRte",
    ]
    time_vary_ = ["IncShkDstn", "Rfree", "PermGroFac", "LivPrb"]

    shock_vars_ = ["PermShk", "TranShk"]
    distributions = [
        "IncShkDstn",
        "PermShkDstn",
        "TranShkDstn",
        "kNrmInitDstn",
        "pLvlInitDstn",
        "HabitInitDstn",
    ]
