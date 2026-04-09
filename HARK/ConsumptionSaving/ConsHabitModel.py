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
from HARK.core import AgentType, get_it_from
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
from HARK.Calibration.Assets.AssetProcesses import (
    make_lognormal_RiskyDstn,
    calc_ShareLimit_for_CRRA,
)
from HARK.ConsumptionSaving.ConsRiskyAssetModel import make_simple_ShareGrid


def make_lognormal_habit_init_dstn(hLogInitMean, hLogInitStd, HabitInitCount, RNG):
    """
    Construct a lognormal distribution for (normalized) initial habit stock
    of newborns, hNrm. This is the default constructor for HabitInitDstn.

    Parameters
    ----------
    hLogInitMean : float
        Mean of log habit stock for newborns.
    hLogInitStd : float
        Stdev of log habit stock for newborns.
    HabitInitCount : int
        Number of points in the discretization.
    RNG : np.random.RandomState
        Agent's internal RNG.

    Returns
    -------
    HabitInitDstn : DiscreteDistribution
        Discretized distribution of initial habit stock for newborns.
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

    chi = (W_a(a,H) - lambda * W_H(a,H)) ** (-1/rho)

    a = m - c
    H = lambda * c + (1-lambda) * h
    m' = R a / psi + theta
    h' = H / psi

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
    ChiMax : float
        Largest value of "transformed marginal value" to consider in the grid.
        The minimum value is always zero. These chi values are "consumption-like".
    ChiCount : int
        Number of gridpoints in the "transformed marginal value" grid.
    ChiOrder : float
        Strictly positive exponential order for the "transformed marginal value" grid.
    HabitMax : float
        Largest value in the habit grid to consider; minimum is always zero.
    HabitCount : int
        Number of gridpoints in the habit stock grid.
    HabitOrder : float
        Strictly positive exponential order for the habit stock grid.
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
        # Validation
        if HabitRte > 1.0:
            raise ValueError("HabitRte must be no greater than 1!")
        if HabitRte <= 0.0:
            raise ValueError("HabitRte must be strictly positive!")
        if HabitWgt > 1.0:
            raise ValueError("HabitWgt must be no greater than 1!")
        if HabitWgt <= 0.0:
            raise ValueError("HabitWgt must be strictly positive!")

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
        self.rate = HabitRte

    def __call__(self, H, chi):
        h = self.hFunc(H, chi)
        rate = self.rate
        c = (H - (1 - rate) * h) / rate
        return c, h

    def cFunc(self, H, chi):
        return self(H, chi)[0]  # just return consumption


def make_habit_inverter(
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


def make_dense_grids(
    HabitMin,
    HabitMax,
    HabitCount,
    HabitOrder,
    aXtraMin,
    aXtraMax,
    aXtraCount,
    aXtraNestFac,
    aXtraExtra,
    DenseFactor,
):
    HabitGridDense = make_exponential_grid(
        HabitMin, HabitMax, HabitCount * DenseFactor, HabitOrder
    )
    mXtraGridDense = make_assets_grid(
        aXtraMin, aXtraMax, aXtraCount * DenseFactor, aXtraExtra, aXtraNestFac
    )
    out = {"hGridDense": HabitGridDense, "mXtraGrid": mXtraGridDense}
    return out


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


def calc_marg_values(S, k, hpre, rho, R, Gamma, alpha, lamda, beta, C, Vp):
    """
    Helper function for computing expected marginal value with respect to market
    resources and habit stock. Used internally by solve_one_period_ConsHabit.

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
    temp = h ** (-alpha * (1 - rho))
    dudc = temp * c ** (-rho)
    dudh = c ** (1 - rho) * (-alpha) * temp / h
    dvdm = dudc + lamda * dvdH
    dvdh = dudh + (1 - lamda) * dvdH
    G_adj = G ** ((1 - rho) * (1 - alpha) - 1.0)
    dvdk = R * G_adj * dvdm
    dvdh = G_adj * dvdh
    return dvdk, dvdh


def calc_mid_dvds(shocks, w_nrm, H_nrm, share, rfree, dvdkFunc_next):
    """
    Compute middle-of-period marginal value of risky asset share by taking expec-
    tations over risky return shocks. Uses the next period's marginal value func-
    tions directly (they already integrate over income shocks).

    Parameters
    ----------
    shocks : float or np.array
        Risky return realizations.
    w_nrm : np.array
        Pre-return savings (w = m - c).
    H_nrm : np.array
        End-of-period habit stock.
    share : np.array
        Risky share.
    rfree : float
        Risk-free return factor.
    dvdkFunc_next : callable
        Next period's marginal value of capital, dvdk(k, hPre).

    Returns
    -------
    dvds : np.array
        Marginal value of risky share (zero at optimum).
    """
    ex_ret = shocks - rfree
    rport = rfree + share * ex_ret
    a_nrm = rport * w_nrm  # post-return assets = next period's kNrm

    dvdk = dvdkFunc_next(a_nrm, H_nrm)
    dvds = ex_ret * w_nrm * dvdk
    return dvds


def calc_mid_dvdx(shocks, w_nrm, H_nrm, share, rfree, dvdkFunc_next, dvdhFunc_next):
    """
    Compute middle-of-period marginal value of wealth and habit stock by taking
    expectations over  risky return shocks. Uses the next period's marginal value
    function directly (they already integrate over income shocks).

    Parameters
    ----------
    shocks : float or np.array
        Risky return realizations.
    w_nrm : np.array
        Pre-return savings (w = m - c).
    H_nrm : np.array
        End-of-period habit stock.
    share : np.array
        Risky share.
    rfree : float
        Risk-free return factor.
    dvdkFunc_next : callable
        Next period's marginal value of capital, dvdk(k, hPre).
    dvdhFunc_next : callable
        Next period's marginal value of habit stock, dvdh(k, hPre).

    Returns
    -------
    dvdw : np.array
        Marginal value of pre-return savings.
    dvdH : np.array
        Marginal value of end-of-period habit stock.
    """
    ex_ret = shocks - rfree
    rport = rfree + share * ex_ret
    a_nrm = rport * w_nrm  # post-return assets = next period's kNrm

    dvdk = dvdkFunc_next(a_nrm, H_nrm)
    dvdw = rport * dvdk
    dvdH = dvdhFunc_next(a_nrm, H_nrm)
    return dvdw, dvdH


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
    mXtraGrid,
    hGridDense,
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
        and the decision-time habit stock from which it was chosen.
    HabitWgt : float
        Exponent on habit stock, which is used as a divisor on consumption in
        the utility function: U(c,h) = u(c / h**alpha). Should be on unit interval.
    HabitRte : float
        Rate at which habit stock is updated by new consumption: H = lambda*c + (1-lambda)*h.
        Should be on the unit interval.
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
    U = UtilityFuncCRRA(CRRA)
    DiscFacEff = DiscFac * LivPrb

    # Make end-of-period state grids
    aNrmMin = solution_next["kNrmMin"]
    aGrid = aXtraGrid + aNrmMin
    aNrm, HNrm = np.meshgrid(aGrid, HabitGrid, indexing="ij")

    # Construct the consumption function
    if type(solution_next["dvdkFunc"]) is ConstantFunction:
        # This is the terminal period, and the consumption function is to consume all
        cFunc = IdentityFunction(i_dim=0, n_dims=2)
        mNrmMin = 0.0

    else:
        # Evaluate end-of-period marginal value on those grids, then calculate chi
        EndOfPrd_dvda = DiscFacEff * solution_next["dvdkFunc"](aNrm, HNrm)
        EndOfPrd_dvdH = DiscFacEff * solution_next["dvdhFunc"](aNrm, HNrm)
        chi = U.derinv(EndOfPrd_dvda - HabitRte * EndOfPrd_dvdH)

        # Recover c and h using the FOC inverter, then find endogenous m gridpoints
        cNrm, hNrm = FOCinverter(HNrm, chi)
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

    # Calculate the natural borrowing constraint (lowest allowable beginning-of-period capital)
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


def solve_optimal_share_habit(
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
):
    """
    Solve only the portfolio allocation problem of the consumption-saving model
    with habit formation.

    Parameters
    ----------
    solution_next : dict
        Dictionary with next period's solution.
    RiskyDstn : DiscreteDistribution
        Discretized risky asset return distribution.
    LivPrb : float
        Survival probability at the end of this period.
    DiscFac : float
        Intertemporal discount factor.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Interest factor on risk-free asset.
    BoroCnstArt : float or None
        Artificial borrowing constraint on end-of-period assets.
    aXtraGrid : np.array
        Grid of "assets above minimum".
    HabitGrid : np.array
        Grid of habit stock values.
    ShareGrid : np.array
        Grid of risky share values on [0,1].
    ShareLimit : float
        Merton-Samuelson limiting share as wealth -> infinity.

    Returns
    -------
    solution_mid : dict
        Solution to the portfolio allocation problem, which has a similar form
        as the solution to the period overall. The dictionary key dvdkFunc refers
        to dvdwFunc, marginal value of post-consumption wealth. The key dvdHfunc
        is the *mid-period* marginal value of post-consumption habit stock, just
        before the portfolio allocation decision is made.
    """
    U = UtilityFuncCRRA(CRRA)
    DiscFacEff = DiscFac * LivPrb

    # Unpack next period's solution
    dvdkFunc_next = solution_next["dvdkFunc"]
    dvdhFunc_next = solution_next["dvdhFunc"]

    # This solver assumes the post-consumption wealth state satisfies wNrm >= 0.
    # Reject unsupported borrowing configurations rather than silently producing
    # share policies on an inconsistent state space.
    if not np.isclose(BoroCnstArt, 0.0):
        raise NotImplementedError(
            "solve_optimal_share_habit only supports BoroCnstArt == 0.0; "
            "negative borrowing constraints are not incorporated into the "
            "post-consumption wealth/share state space."
        )

    # Minimum post-consumption wealth in the supported no-borrowing case
    wNrmMin = 0.0

    # Handle the terminal period: share doesn't matter, marginal value still zero
    if isinstance(dvdkFunc_next, ConstantFunction):
        ShareFunc_mid = ConstantFunction(ShareLimit)
        dvdwFunc = ConstantFunction(0.0)
        dvdHmidFunc = ConstantFunction(0.0)
        solution_mid = {
            "ShareFunc": ShareFunc_mid,
            "dvdkFunc": dvdwFunc,
            "dvdhFunc": dvdHmidFunc,
            "kNrmMin": wNrmMin,
        }
        return solution_mid

    # For each (w, H, s), compute E_R[Rport * dvdkFunc_next(Rport*w, H)]
    # and E_R[dvdhFunc_next(Rport*w, H)] and
    # E_R[(Risky-Rfree)*w * dvdkFunc_next(Rport*w, H)].

    # Build 3D meshes (w, H, s). These are for mid-period state space points
    # combined with candidate risky share values
    wGrid = aXtraGrid + wNrmMin
    wCount = wGrid.size
    HabitCount = HabitGrid.size
    ShareCount = ShareGrid.size
    wMesh, Hmesh, sMesh = np.meshgrid(wGrid, HabitGrid, ShareGrid, indexing="ij")

    # Compute expected marginal value of wealth, habit stock, and risky share for each (w,H,S)
    dvds_mid = DiscFacEff * expected(
        calc_mid_dvds,
        RiskyDstn,
        args=(wMesh, Hmesh, sMesh, Rfree, dvdkFunc_next),
    )

    # For each (w, H), find optimal share where dvds == 0 by looking for a
    # sign change: dvds goes from positive to negative
    focs = dvds_mid
    crossing = np.logical_and(focs[:, :, 1:] <= 0.0, focs[:, :, :-1] >= 0.0)
    share_idx = np.argmax(crossing, axis=2)

    # Find the optimal risky share by solving a linear equation for the FOC
    # crossing point, given that we know upper and lower bounding points for it
    w_idx, h_idx = np.meshgrid(np.arange(wCount), np.arange(HabitCount), indexing="ij")
    bot_s = ShareGrid[share_idx]
    top_s = ShareGrid[np.minimum(share_idx + 1, ShareCount - 1)]
    bot_f = focs[w_idx, h_idx, share_idx]
    top_f = focs[w_idx, h_idx, np.minimum(share_idx + 1, ShareCount - 1)]
    alpha_interp = np.where(
        (top_f - bot_f) != 0.0,
        1.0 - top_f / (top_f - bot_f),
        0.5,
    )
    Share_opt = (1.0 - alpha_interp) * bot_s + alpha_interp * top_s

    # Handle corner solutions
    constrained_top = focs[:, :, -1] > 0.0
    constrained_bot = focs[:, :, 0] < 0.0
    Share_opt[constrained_top] = 1.0
    Share_opt[constrained_bot] = 0.0

    # Construct the share function over (w,H)
    ShareFunc_by_HNrm = [
        LinearInterp(wGrid, Share_opt[:, j], ShareLimit, 0.0, lower_extrap=True)
        for j in range(HabitCount)
    ]
    ShareFunc_mid = LinearInterpOnInterp1D(ShareFunc_by_HNrm, HabitGrid)

    # Extract optimized end-of-period marginal values at optimal share
    wNrm, HNrm = np.meshgrid(wGrid, HabitGrid, indexing="ij")
    dvdw_opt, dvdH_opt = DiscFacEff * expected(
        calc_mid_dvdx,
        RiskyDstn,
        args=(wNrm, HNrm, Share_opt, Rfree, dvdkFunc_next, dvdhFunc_next),
    )

    # Build interpolant for mid-period marginal value of habit stock on (w, H) grid;
    # dvdH_opt already includes DiscFacEff.
    dvdHnvrs = U.inv(dvdH_opt)
    dvdHNvrsFunc = BilinearInterp(dvdHnvrs, wGrid, HabitGrid)
    dvdHmidFunc = ValueFuncCRRA(dvdHNvrsFunc, CRRA)

    # Build interpolant for mid-period marginal value of wealth (after consumption
    # but before portfolio allocation); incorporates DiscFacEff
    dvdwNvrs = np.concatenate(
        (np.zeros((1, HabitGrid.size)), U.derinv(dvdw_opt)), axis=0
    )
    dvdwNvrsFunc = BilinearInterp(dvdwNvrs, np.insert(wGrid, 0, wNrmMin), HabitGrid)
    dvdwFunc = MargValueFuncCRRA(dvdwNvrsFunc, CRRA)

    # Package and return the mid-period solution as a dictionary
    solution_mid = {
        "ShareFunc": ShareFunc_mid,
        "dvdkFunc": dvdwFunc,
        "dvdhFunc": dvdHmidFunc,
        "kNrmMin": wNrmMin,
    }
    return solution_mid


###############################################################################


def solve_one_period_HabitPortfolio(
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
    mXtraGrid,
    hGridDense,
):
    """
    Solve one period of the consumption-saving model with habit formation and
    portfolio choice. This version uses a "modular" structure, solving the portfolio
    problem and consumption problem using separate sub-functions. The latter sub-
    function is literally the basic habit formation solver with some parameters
    turned off (i.e. set to 1.0) because they were handled in the portfolio block.

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
    FOCinverter : HabitFormationInverter
        Function that inverts the first order conditions to yield optimal consumption
        and the decision-time habit stock from which it was chosen.
    HabitWgt : float
        Exponent on habit stock, which is used as a divisor on consumption in
        the utility function: U(c,h) = u(c / h**alpha). Should be on unit interval.
    HabitRte : float
        Rate at which habit stock is updated by new consumption: H = lambda*c + (1-lambda)*h.
        Should be on the unit interval.
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
        ShareFunc : Risky share function over (wNrm, HNrm).
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
    solution_now = solve_one_period_ConsHabit(
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
        mXtraGrid,
        hGridDense,
    )

    # Add the risky share function to the period's solution and return it
    solution_now["ShareFunc"] = solution_mid["ShareFunc"]
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
    "FOCinverter": make_habit_inverter,
    "HabitGrid": make_habit_grid,
    "DenseGrids": make_dense_grids,
    "mXtraGrid": get_it_from("DenseGrids"),
    "hGridDense": get_it_from("DenseGrids"),
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
    "aXtraMax": 30,  # Maximum end-of-period "assets above minimum" value
    "aXtraNestFac": 2,  # Exponential nesting factor for aXtraGrid
    "aXtraCount": 72,  # Number of points in the grid of "assets above minimum"
    "aXtraExtra": None,  # Additional other values to add in grid (optional)
}

# Default parameters to make HabitGrid using make_habit_grid
HabitConsumerType_HabitGrid_default = {
    "HabitMin": 0.2,
    "HabitMax": 5.0,
    "HabitCount": 41,
    "HabitOrder": 2.0,
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
    "HabitWgt": 0.5,  # Weight on consumption habit; exponent on habit divisor in utility
    "HabitRte": 0.2,  # Speed of consumption habit updating
    "DenseFactor": 3,  # Density factor for re-interpolation
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
        \newcommand{\HabitWgt}{\alpha}
        \newcommand{\HabitRte}{\lambda}

        \begin{align*}
        v_t(m_t,h_t) &= \max_{c_t}u(c_t,h_t) + \DiscFac \LivPrb_t \mathbb{E}_{t} \left[ (\PermGroFac_{t+1} \psi_{t+1})^{(1-\HabitWgt)(1-\CRRA)} v_{t+1}(m_{t+1}, h_{t+1}) \right] \\
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
        "mXtraGrid",
        "hGridDense",
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


###############################################################################

# Make a dictionary of constructors for the habit-portfolio model
HabitPortfolio_constructors_default = HabitConsumerType_constructors_default.copy()
HabitPortfolio_additional_constructors = {
    "RiskyDstn": make_lognormal_RiskyDstn,
    "ShareGrid": make_simple_ShareGrid,
    "ShareLimit": calc_ShareLimit_for_CRRA,
}
HabitPortfolio_constructors_default.update(HabitPortfolio_additional_constructors)

HabitPortfolio_RiskyDstn_default = {
    "RiskyAvg": 1.08,
    "RiskyStd": 0.18,
    "RiskyCount": 5,
}

HabitPortfolio_ShareGrid_default = {
    "ShareCount": 26,
}

HabitPortfolioConsumerType_defaults = HabitConsumerType_defaults.copy()
HabitPortfolioConsumerType_defaults["constructors"] = (
    HabitPortfolio_constructors_default
)
HabitPortfolioConsumerType_defaults.update(HabitPortfolio_RiskyDstn_default)
HabitPortfolioConsumerType_defaults.update(HabitPortfolio_ShareGrid_default)


class HabitPortfolioConsumerType(HabitConsumerType):
    r"""
    A class for representing consumers who form consumption habits and can allocate
    their wealth between a risky and riskless asset. Agents get flow utility according
    to a CRRA felicity function that depends on both current consumption and the habit
    stock h_t. The habit stock evolves as a weighted average of current consumption
    and prior habit stock.

    This type's solver uses a "modular" approach that separates the portfolio allo-
    cation and consumption-saving problems into two functions. The latter function
    is just the solver for HabitConsumerType. Consequently, the consumption function
    is defined over (m,h) but the risky share function is defined over (w,H).

    .. math::
        \newcommand{\CRRA}{\rho}
        \newcommand{\LivPrb}{\mathsf{S}}
        \newcommand{\PermGroFac}{\Gamma}
        \newcommand{\Rfree}{\mathsf{R}}
        \newcommand{\Risky}{\mathfrak{R}}
        \newcommand{\DiscFac}{\beta}
        \newcommand{\HabitWgt}{\alpha}
        \newcommand{\HabitRte}{\lambda}

        \begin{align*}
        v_t(m_t,h_t) &= \max_{c_t, s_t} u(c_t,h_t) + \DiscFac \LivPrb_t
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
        (\psi_{t+1}, \theta_{t+1}) &\sim F_{t+1}, \\
        \Risky_{t} & \sim G, \\
        u(c,h) &= \frac{(c/h^\HabitWgt)^{1-\CRRA}}{1-\CRRA}.
        \end{align*}
    """

    default_ = {
        "params": HabitPortfolioConsumerType_defaults,
        "solver": solve_one_period_HabitPortfolio,
        "model": "ConsHabitPortfolio.yaml",
        "track_vars": ["aNrm", "cNrm", "mNrm", "hNrm", "Share", "pLvl"],
    }

    time_inv_ = HabitConsumerType.time_inv_ + ["RiskyDstn", "ShareGrid"]
    time_vary_ = HabitConsumerType.time_vary_ + ["ShareLimit"]
    shock_vars_ = HabitConsumerType.shock_vars_ + ["Risky"]
    distributions = HabitConsumerType.distributions + ["RiskyDstn"]
