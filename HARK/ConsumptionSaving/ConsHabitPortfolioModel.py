"""
Consumption-saving model with habit formation and portfolio choice between a
risk-free and a risky asset. Combines the habit formation model from
ConsHabitModel.py with the portfolio choice structure from
ConsWealthPortfolioModel.py / ConsPortfolioModel.py.

The agent's problem has a two-dimensional decision-time state (m_t, h_t) and
two controls (c_t, s_t). The solution method uses:
  1. Share search: for each (w, H) on end-of-period grids, find s* where the
     portfolio FOC is zero by integrating over risky returns and using the next
     period's marginal value functions dvdkFunc_next and dvdhFunc_next.
  2. Habit EGM: recover optimal consumption c_t and decision-time habit h_t
     from end-of-period state H_t using the FOC-inverter.
  3. Marginal values: compute dvdkFunc and dvdhFunc by integrating over income
     shocks, using the current period's policy functions and a precomputed
     continuation habit value interpolant.
"""

import numpy as np
from HARK.utilities import make_assets_grid
from HARK.interpolation import (
    LinearInterp,
    ConstantFunction,
    Curvilinear2DInterp,
    LowerEnvelope2D,
    LinearInterpOnInterp1D,
    IdentityFunction,
    BilinearInterp,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.distributions import expected
from HARK.core import AgentType
from HARK.Calibration.Income.IncomeProcesses import (
    construct_lognormal_income_process_unemployment,
    get_PermShkDstn_from_IncShkDstn,
    get_TranShkDstn_from_IncShkDstn,
)
from HARK.Calibration.Assets.AssetProcesses import (
    make_lognormal_RiskyDstn,
    calc_ShareLimit_for_CRRA,
)
from HARK.ConsumptionSaving.ConsRiskyAssetModel import make_simple_ShareGrid
from HARK.ConsumptionSaving.ConsHabitModel import (
    make_inverter,
    make_habit_grid,
    make_lognormal_habit_init_dstn,
)
from HARK.ConsumptionSaving.ConsIndShockModel import (
    make_lognormal_kNrm_init_dstn,
    make_lognormal_pLvl_init_dstn,
)
from HARK.rewards import UtilityFuncCRRA


###############################################################################
# Helper functions for expectations
###############################################################################


def calc_end_dvdx_habit(
    shocks, w_nrm, H_nrm, share, rfree, dvdkFunc_next, dvdhFunc_next
):
    """
    Compute end-of-period marginal values by taking expectations over risky
    return shocks. Uses the next period's marginal value functions directly
    (they already integrate over income shocks).

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
    dvds : np.array
        Marginal value of risky share (zero at optimum).
    dvdH : np.array
        Marginal value of end-of-period habit stock.
    """
    ex_ret = shocks - rfree
    rport = rfree + share * ex_ret
    a_nrm = rport * w_nrm  # post-return assets = next period's kNrm

    dvdk = dvdkFunc_next(a_nrm, H_nrm)
    dvdw = rport * dvdk
    dvds = ex_ret * w_nrm * dvdk
    dvdH = dvdhFunc_next(a_nrm, H_nrm)
    return dvdw, dvds, dvdH


def calc_marg_values_port(S, k, hpre, rho, Gamma, alpha, lamda, cFunc, dvdH_cont_func):
    """
    Compute beginning-of-period marginal values of capital (k) and pre-period
    habit stock (hPre) by taking expectations over income shocks. Uses the
    current period's cFunc and a precomputed continuation habit value function.

    In the YAML dynamics, bNrm = kNrm / G (portfolio return already embedded
    in kNrm via the twist aNrm -> kNrm, where aNrm = Rport * wNrm).
    """
    psi = S["PermShk"]
    theta = S["TranShk"]
    G = psi * Gamma
    m = k / G + theta
    h = hpre / G
    c = cFunc(m, h)
    w = m - c
    H = lamda * c + (1 - lamda) * h

    # dvdH_cont already includes DiscFacEff and E_R[dvdhFunc_next]
    dvdH_cont = dvdH_cont_func(w, H)

    temp = h ** (-alpha * (1 - rho))
    dudc = temp * c ** (-rho)
    dvdm = dudc + lamda * dvdH_cont
    dudh = c ** (1 - rho) * (-alpha) * temp / h
    dvdh = dudh + (1 - lamda) * dvdH_cont
    G_adj = G ** ((1 - rho) * (1 - alpha) - 1.0)
    dvdk = G_adj * dvdm
    dvdh_out = G_adj * dvdh
    return dvdk, dvdh_out


###############################################################################
# Terminal solution
###############################################################################


def make_habit_portfolio_solution_terminal():
    """
    Make a pseudo-terminal solution for the habit-portfolio model.
    All value functions are zero (constant). ShareFunc is a constant at 0.
    """
    dvdkFunc_terminal = ConstantFunction(0.0)
    dvdhFunc_terminal = ConstantFunction(0.0)
    solution_terminal = {
        "dvdkFunc": dvdkFunc_terminal,
        "dvdhFunc": dvdhFunc_terminal,
        "kNrmMin": 0.0,
    }
    return solution_terminal


###############################################################################
# Main solver
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
):
    """
    Solve one period of the consumption-saving model with habit formation and
    portfolio choice.

    Parameters
    ----------
    solution_next : dict
        Dictionary with next period's solution.
    IncShkDstn : DiscreteDistribution
        Discretized permanent and transitory income shock distribution.
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
    PermGroFac : float
        Permanent income growth factor.
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
    FOCinverter : HabitFormationInverter
        Inverts the FOC to recover (c, h) from (H, chi).
    HabitWgt : float
        Exponent on habit stock in utility function.
    HabitRte : float
        Rate of habit stock updating.

    Returns
    -------
    solution_now : dict
        Solution to this period's problem.
    """
    U = UtilityFuncCRRA(CRRA)
    DiscFacEff = DiscFac * LivPrb

    # Unpack next period's solution
    dvdkFunc_next = solution_next["dvdkFunc"]
    dvdhFunc_next = solution_next["dvdhFunc"]

    # Minimum savings is zero (can't borrow)
    wNrmMin = 0.0
    if (BoroCnstArt is not None) and (BoroCnstArt > -np.inf):
        kNrmMin = BoroCnstArt
    else:
        kNrmMin = 0.0

    if isinstance(dvdkFunc_next, ConstantFunction):
        # ============================================================
        # Terminal period: consume everything, share doesn't matter
        # ============================================================
        cFunc = IdentityFunction(i_dim=0, n_dims=2)
        ShareFunc = ConstantFunction(ShareLimit)
        dvdH_cont_func = ConstantFunction(0.0)

    else:
        # ============================================================
        # Stage 1: Optimal risky share
        # ============================================================
        # For each (w, H, s), compute E_R[Rport * dvdkFunc_next(Rport*w, H)]
        # and E_R[(Risky-Rfree)*w * dvdkFunc_next(Rport*w, H)] and
        # E_R[dvdhFunc_next(Rport*w, H)].

        wGrid = aXtraGrid + wNrmMin
        wCount = wGrid.size
        HabitCount = HabitGrid.size
        ShareCount = ShareGrid.size

        # Build 3D meshes: (w, H, s)
        w_3d = np.tile(wGrid[:, np.newaxis, np.newaxis], (1, HabitCount, ShareCount))
        H_3d = np.tile(HabitGrid[np.newaxis, :, np.newaxis], (wCount, 1, ShareCount))
        s_3d = np.tile(ShareGrid[np.newaxis, np.newaxis, :], (wCount, HabitCount, 1))

        end_dvdw_3d, end_dvds_3d, end_dvdH_3d = DiscFacEff * expected(
            calc_end_dvdx_habit,
            RiskyDstn,
            args=(w_3d, H_3d, s_3d, Rfree, dvdkFunc_next, dvdhFunc_next),
        )

        # For each (w, H), find optimal share where dvds == 0
        focs = end_dvds_3d

        # Find sign change: dvds goes from positive to negative
        crossing = np.logical_and(focs[:, :, 1:] <= 0.0, focs[:, :, :-1] >= 0.0)
        share_idx = np.argmax(crossing, axis=2)

        w_idx, h_idx = np.meshgrid(
            np.arange(wCount), np.arange(HabitCount), indexing="ij"
        )

        bot_s = ShareGrid[share_idx]
        top_s = ShareGrid[np.minimum(share_idx + 1, ShareCount - 1)]
        bot_f = focs[w_idx, h_idx, share_idx]
        top_f = focs[w_idx, h_idx, np.minimum(share_idx + 1, ShareCount - 1)]
        alpha_interp = np.where(
            (top_f - bot_f) != 0.0,
            1.0 - top_f / (top_f - bot_f),
            0.5,
        )
        alpha_interp = np.clip(alpha_interp, 0.0, 1.0)

        Share_opt = (1.0 - alpha_interp) * bot_s + alpha_interp * top_s

        # Handle corner solutions
        constrained_top = focs[:, :, -1] > 0.0
        constrained_bot = focs[:, :, 0] < 0.0
        Share_opt[constrained_top] = 1.0
        Share_opt[constrained_bot] = 0.0

        # Extract optimized end-of-period marginal values at optimal share
        bot_dvdw = end_dvdw_3d[w_idx, h_idx, share_idx]
        top_dvdw = end_dvdw_3d[w_idx, h_idx, np.minimum(share_idx + 1, ShareCount - 1)]
        dvdw_opt = (1.0 - alpha_interp) * bot_dvdw + alpha_interp * top_dvdw
        dvdw_opt[constrained_top] = end_dvdw_3d[:, :, -1][constrained_top]
        dvdw_opt[constrained_bot] = end_dvdw_3d[:, :, 0][constrained_bot]

        bot_dvdH = end_dvdH_3d[w_idx, h_idx, share_idx]
        top_dvdH = end_dvdH_3d[w_idx, h_idx, np.minimum(share_idx + 1, ShareCount - 1)]
        dvdH_opt = (1.0 - alpha_interp) * bot_dvdH + alpha_interp * top_dvdH
        dvdH_opt[constrained_top] = end_dvdH_3d[:, :, -1][constrained_top]
        dvdH_opt[constrained_bot] = end_dvdH_3d[:, :, 0][constrained_bot]

        # Build interpolant for continuation habit value on (w, H) grid.
        # dvdH_opt already includes DiscFacEff. We store it for Stage 3 (below).
        dvdH_cont_func = BilinearInterp(dvdH_opt, wGrid, HabitGrid)

        # ============================================================
        # Stage 2: Optimal consumption via habit EGM
        # ============================================================

        # chi = u'^{-1}(dvdw_opt - lambda * dvdH_opt) per the habit FOC
        chi = U.derinv(dvdw_opt - HabitRte * dvdH_opt)

        # Recover (c, h) from (H, chi)
        cNrm, hNrm = FOCinverter(
            np.tile(HabitGrid[np.newaxis, :], (wCount, 1)),
            chi,
        )
        mNrm = np.tile(wGrid[:, np.newaxis], (1, HabitCount)) + cNrm

        # Add constrained boundary point (c=0, h at boundary)
        cNrm_aug = np.concatenate((np.zeros((1, HabitCount)), cNrm), axis=0)
        mNrm_aug = np.concatenate((wNrmMin * np.ones((1, HabitCount)), mNrm), axis=0)
        if HabitRte == 1.0:
            hBot = np.reshape(hNrm[0, :], (1, HabitCount))
        else:
            hBot = np.reshape(HabitGrid / (1.0 - HabitRte), (1, HabitCount))
        hNrm_aug = np.concatenate((hBot, hNrm), axis=0)

        # Build consumption function
        cFuncUnc = Curvilinear2DInterp(cNrm_aug, mNrm_aug, hNrm_aug)

        if (BoroCnstArt is not None) and (BoroCnstArt > -np.inf):
            cFuncCnst_temp = LinearInterp([BoroCnstArt, BoroCnstArt + 1.0], [0.0, 1.0])
            cFuncCnst = LinearInterpOnInterp1D(
                [cFuncCnst_temp, cFuncCnst_temp], np.array([0.0, 1.0])
            )
            cFunc = LowerEnvelope2D(cFuncUnc, cFuncCnst)
        else:
            cFunc = cFuncUnc

        # Build share function on the same endogenous grid
        Share_aug = np.concatenate(
            (ShareLimit * np.ones((1, HabitCount)), Share_opt), axis=0
        )
        ShareFunc = Curvilinear2DInterp(Share_aug, mNrm_aug, hNrm_aug)

    # ============================================================
    # Stage 3: Marginal value functions
    # ============================================================

    kGrid = kNrmMin + aXtraGrid
    kNrm, hPre = np.meshgrid(kGrid, HabitGrid, indexing="ij")

    dvdk, dvdh = expected(
        calc_marg_values_port,
        IncShkDstn,
        args=(
            kNrm,
            hPre,
            CRRA,
            PermGroFac,
            HabitWgt,
            HabitRte,
            cFunc,
            dvdH_cont_func,
        ),
    )

    dvdkNvrs = np.concatenate((np.zeros((1, HabitGrid.size)), U.derinv(dvdk)), axis=0)
    dvdkNvrsFunc = BilinearInterp(dvdkNvrs, np.insert(kGrid, 0, kNrmMin), HabitGrid)
    dvdkFunc = MargValueFuncCRRA(dvdkNvrsFunc, CRRA)

    dvdhNvrs = np.concatenate((np.zeros((1, HabitGrid.size)), U.inv(dvdh)), axis=0)
    dvdhNvrsFunc = BilinearInterp(dvdhNvrs, np.insert(kGrid, 0, kNrmMin), HabitGrid)
    dvdhFunc = ValueFuncCRRA(dvdhNvrsFunc, CRRA)

    # Package solution
    solution_now = {
        "cFunc": cFunc,
        "ShareFunc": ShareFunc,
        "dvdkFunc": dvdkFunc,
        "dvdhFunc": dvdhFunc,
        "kNrmMin": kNrmMin,
        "distance_criteria": ["cFunc"],
    }
    return solution_now


###############################################################################
# Default parameter dictionaries
###############################################################################

HabitPortfolio_constructors_default = {
    "kNrmInitDstn": make_lognormal_kNrm_init_dstn,
    "pLvlInitDstn": make_lognormal_pLvl_init_dstn,
    "HabitInitDstn": make_lognormal_habit_init_dstn,
    "IncShkDstn": construct_lognormal_income_process_unemployment,
    "PermShkDstn": get_PermShkDstn_from_IncShkDstn,
    "TranShkDstn": get_TranShkDstn_from_IncShkDstn,
    "aXtraGrid": make_assets_grid,
    "RiskyDstn": make_lognormal_RiskyDstn,
    "ShareGrid": make_simple_ShareGrid,
    "ShareLimit": calc_ShareLimit_for_CRRA,
    "FOCinverter": make_inverter,
    "HabitGrid": make_habit_grid,
    "solution_terminal": make_habit_portfolio_solution_terminal,
}

HabitPortfolio_kNrmInitDstn_default = {
    "kLogInitMean": -12.0,
    "kLogInitStd": 0.0,
    "kNrmInitCount": 15,
}

HabitPortfolio_pLvlInitDstn_default = {
    "pLogInitMean": 0.0,
    "pLogInitStd": 0.0,
    "pLvlInitCount": 15,
}

HabitPortfolio_HabitInitDstn_default = {
    "hLogInitMean": -0.5,
    "hLogInitStd": 0.2,
    "HabitInitCount": 15,
}

HabitPortfolio_IncShkDstn_default = {
    "PermShkStd": [0.1],
    "PermShkCount": 7,
    "TranShkStd": [0.1],
    "TranShkCount": 7,
    "UnempPrb": 0.05,
    "IncUnemp": 0.3,
    "T_retire": 0,
    "UnempPrbRet": 0.005,
    "IncUnempRet": 0.0,
}

HabitPortfolio_aXtraGrid_default = {
    "aXtraMin": 0.001,
    "aXtraMax": 50.0,
    "aXtraNestFac": 2,
    "aXtraCount": 48,
    "aXtraExtra": None,
}

HabitPortfolio_HabitGrid_default = {
    "HabitMin": 0.2,
    "HabitMax": 5.0,
    "HabitCount": 31,
    "HabitOrder": 1.5,
}

HabitPortfolio_inverter_default = {
    "ChiMax": 50.0,
    "ChiCount": 251,
    "ChiOrder": 1.5,
}

HabitPortfolio_RiskyDstn_default = {
    "RiskyAvg": 1.08,
    "RiskyStd": 0.18362634887,
    "RiskyCount": 5,
}

HabitPortfolio_ShareGrid_default = {
    "ShareCount": 25,
}

HabitPortfolio_solving_default = {
    "cycles": 1,
    "T_cycle": 1,
    "pseudo_terminal": True,
    "constructors": HabitPortfolio_constructors_default,
    "CRRA": 2.0,
    "Rfree": [1.03],
    "DiscFac": 0.96,
    "LivPrb": [0.98],
    "PermGroFac": [1.01],
    "BoroCnstArt": 0.0,
    "HabitWgt": 0.5,
    "HabitRte": 0.2,
}

HabitPortfolio_simulation_default = {
    "AgentCount": 10000,
    "T_age": None,
}

HabitPortfolioConsumerType_defaults = {}
HabitPortfolioConsumerType_defaults.update(HabitPortfolio_IncShkDstn_default)
HabitPortfolioConsumerType_defaults.update(HabitPortfolio_kNrmInitDstn_default)
HabitPortfolioConsumerType_defaults.update(HabitPortfolio_pLvlInitDstn_default)
HabitPortfolioConsumerType_defaults.update(HabitPortfolio_HabitInitDstn_default)
HabitPortfolioConsumerType_defaults.update(HabitPortfolio_aXtraGrid_default)
HabitPortfolioConsumerType_defaults.update(HabitPortfolio_HabitGrid_default)
HabitPortfolioConsumerType_defaults.update(HabitPortfolio_inverter_default)
HabitPortfolioConsumerType_defaults.update(HabitPortfolio_RiskyDstn_default)
HabitPortfolioConsumerType_defaults.update(HabitPortfolio_ShareGrid_default)
HabitPortfolioConsumerType_defaults.update(HabitPortfolio_solving_default)
HabitPortfolioConsumerType_defaults.update(HabitPortfolio_simulation_default)


###############################################################################
# AgentType subclass
###############################################################################


class HabitPortfolioConsumerType(AgentType):
    r"""
    A class for representing consumers who form consumption habits and can
    allocate their savings between a risk-free and a risky asset. Combines
    HabitConsumerType (habit formation with EGM) and RiskyAssetConsumerType
    (portfolio choice with share search).

    The decision-time state space is (m_t, h_t), and the agent chooses both
    consumption c_t and risky share s_t.

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
            v_{t+1}(m_{t+1}, h_{t+1}) \right], \\
        & \text{s.t.}  \\
        a_t &= m_t - c_t, \\
        H_t &= \HabitRte c_t + (1-\HabitRte) h_t, \\
        a_t &\geq 0, \\
        s_t &\in [0,1], \\
        R_{t+1} &= s_t \Risky_{t+1} + (1-s_t) \Rfree_{t+1}, \\
        m_{t+1} &= a_t R_{t+1}/(\PermGroFac_{t+1} \psi_{t+1}) + \theta_{t+1}, \\
        h_{t+1} &= H_t / (\PermGroFac_{t+1} \psi_{t+1}), \\
        u(c,h) &= \frac{(c/h^\HabitWgt)^{1-\CRRA}}{1-\CRRA}.
        \end{align*}
    """

    default_ = {
        "params": HabitPortfolioConsumerType_defaults,
        "solver": solve_one_period_HabitPortfolio,
        "model": "ConsHabitPortfolio.yaml",
        "track_vars": ["aNrm", "cNrm", "mNrm", "hNrm", "Share", "pLvl"],
    }

    time_inv_ = [
        "DiscFac",
        "CRRA",
        "BoroCnstArt",
        "aXtraGrid",
        "HabitGrid",
        "ShareGrid",
        "FOCinverter",
        "HabitWgt",
        "HabitRte",
        "RiskyDstn",
    ]
    time_vary_ = ["IncShkDstn", "Rfree", "PermGroFac", "LivPrb", "ShareLimit"]

    shock_vars_ = ["PermShk", "TranShk", "Risky"]
    distributions = [
        "IncShkDstn",
        "PermShkDstn",
        "TranShkDstn",
        "RiskyDstn",
        "kNrmInitDstn",
        "pLvlInitDstn",
        "HabitInitDstn",
    ]

    def pre_solve(self):
        if isinstance(self.ShareLimit, list):
            self.add_to_time_vary("ShareLimit")
            self.del_from_time_inv("ShareLimit")
        else:
            self.add_to_time_inv("ShareLimit")
            self.del_from_time_vary("ShareLimit")
