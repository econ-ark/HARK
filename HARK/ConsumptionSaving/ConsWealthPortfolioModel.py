from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from scipy.optimize import fixed_point, minimize_scalar, root

from HARK.ConsumptionSaving.ConsPortfolioModel import (
    PortfolioConsumerType,
    init_portfolio,
)
from HARK.ConsumptionSaving.LegacyOOsolvers import ConsPortfolioSolver
from HARK.core import make_one_period_oo_solver
from HARK.distribution import DiscreteDistribution, calc_expectation
from HARK.interpolation import (
    BilinearInterp,
    ConstantFunction,
    LinearInterp,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.metric import MetricObject
from HARK.rewards import CRRAutilityP_inv
from HARK.utilities import NullFunc

EPSILON = 1e-6


@dataclass
class WealthPortfolioSolution(MetricObject):
    distance_criteria = ["cFunc"]
    cFunc: LinearInterp = NullFunc()  # consumption as a func of wealth m
    cEndFunc: LinearInterp = NullFunc()  # consumption as a func of assets a
    shareFunc: LinearInterp = NullFunc()  # risky share as a func of wealth m
    shareEndFunc: LinearInterp = NullFunc()  # risky share as a func of assets a
    vFunc: ValueFuncCRRA = NullFunc()  # value func as a func of wealth m
    vEndFunc: ValueFuncCRRA = NullFunc()  # eop value func as a func of assets a
    vPfunc: MargValueFuncCRRA = NullFunc()  # marg val func as a func of wealth m
    vPEndFunc: MargValueFuncCRRA = NullFunc()  # eop marg val func as a func of assets a


class WealthPortfolioConsumerType(PortfolioConsumerType):
    time_inv_ = deepcopy(PortfolioConsumerType.time_inv_)
    time_inv_ = time_inv_ + ["WealthShare", "method"]

    def __init__(self, method="root", **kwds):
        params = init_wealth_portfolio.copy()
        params.update(kwds)
        kwds = params

        # Initialize a basic portfolio consumer type
        super().__init__(**kwds)

        self.method = method

        solver = ConsWealthPortfolioSolver

        self.solve_one_period = make_one_period_oo_solver(solver)

    def update_solution_terminal(self):
        # set up with CRRA split with u of c and a

        gamma = self.WealthShare

        mGrid = np.append(0.0, self.aXtraGrid)
        cGrid = (1 - gamma) * mGrid  # CD share of wealth
        aGrid = gamma * mGrid  # CD share of wealth

        cFunc_terminal = LinearInterp(mGrid, cGrid)  # as a function of wealth
        cEndFunc_terminal = LinearInterp(aGrid, cGrid)  # as a function of assets
        shareFunc_terminal = ConstantFunction(0.0)  # this doesn't matter?
        shareEndFunc_terminal = ConstantFunction(0.0)  # this doesn't matter?

        vNvrs = cGrid ** (1.0 - gamma) * aGrid**gamma
        vNvrsFunc = LinearInterp(mGrid, vNvrs)
        vFunc_terminal = ValueFuncCRRA(vNvrsFunc, self.CRRA)

        constant = ((1 - gamma) ** (1 - gamma) * gamma**gamma) ** (1 - self.CRRA)

        vPNvrs = mGrid * CRRAutilityP_inv(constant, self.CRRA)
        vPNvrsFunc = LinearInterp(mGrid, vPNvrs)

        vPfunc_terminal = MargValueFuncCRRA(vPNvrsFunc, self.CRRA)

        self.solution_terminal = WealthPortfolioSolution(
            cFunc=cFunc_terminal,
            cEndFunc=cEndFunc_terminal,
            shareFunc=shareFunc_terminal,
            shareEndFunc=shareEndFunc_terminal,
            vFunc=vFunc_terminal,
            vPfunc=vPfunc_terminal,
        )

    def post_solve(self):
        super().post_solve()

        for solution in self.solution:
            solution.cFuncAdj = solution.cFunc
            solution.cFuncFxd = lambda m, s: solution.cFunc(m)
            share = solution.shareFunc
            solution.ShareFuncAdj = lambda m: np.clip(share(m), 0.0, 1.0)
            solution.ShareFuncFxd = lambda m, s: np.clip(share(m), 0.0, 1.0)


@dataclass
class ConsWealthPortfolioSolver(ConsPortfolioSolver):
    solution_next: WealthPortfolioSolution
    ShockDstn: DiscreteDistribution
    IncShkDstn: DiscreteDistribution
    TranShkDstn: DiscreteDistribution
    RiskyDstn: DiscreteDistribution
    LivPrb: float
    DiscFac: float
    CRRA: float
    Rfree: float
    PermGroFac: float
    BoroCnstArt: float
    aXtraGrid: np.array
    ShareGrid: np.array
    vFuncBool: bool
    AdjustPrb: float
    DiscreteShareBool: bool
    ShareLimit: float
    IndepDstnBool: bool
    WealthShare: float
    method: str

    def __post_init__(self):
        self.def_utility_funcs()

    def def_utility_funcs(self):
        """
        Define temporary functions for utility and its derivative and inverse
        """
        super().def_utility_funcs()

        gamma = self.WealthShare

        # cobb douglas aggregate of consumption and assets
        self.f = lambda c, a: (c ** (1.0 - gamma)) * (a**gamma)
        self.fPc = lambda c, a: (1.0 - gamma) * c ** (-gamma) * a**gamma
        self.fPa = lambda c, a: gamma * c ** (1.0 - gamma) * a ** (gamma - 1.0)

        self.uCD = lambda c, a: self.u(self.f(c, a))
        self.uPc = lambda c, a: self.uP(self.f(c, a)) * self.fPc(c, a)
        self.uPa = lambda c, a: self.uP(self.f(c, a)) * self.fPa(c, a)

        self.CRRA_Alt = self.CRRA * (1 - gamma) + gamma
        self.WealthShare_Alt = gamma * (1 - self.CRRA) / self.CRRA_Alt
        self.muInv = lambda x: CRRAutilityP_inv(x / (1 - gamma), self.CRRA_Alt)

    def set_and_update_values(self):
        """
        Unpacks some of the inputs (and calculates simple objects based on them),
        storing the results in self for use by other methods.
        """

        # Unpack next period's solution
        self.vPfunc_next = self.solution_next.vPfunc
        self.vFunc_next = self.solution_next.vFunc
        self.cFunc_next = self.solution_next.cFunc

        # Flag for whether the natural borrowing constraint is zero

        # Unpack the shock distribution
        TranShks_next = self.IncShkDstn.atoms[1]

        # Flag for whether the natural borrowing constraint is zero
        self.zero_bound = np.min(TranShks_next) == 0.0

    def prepare_to_calc_EndOfPrd(self):
        """
        Prepare to calculate end-of-period marginal values by creating an array
        of market resources that the agent could have next period, considering
        the grid of end-of-period assets and the distribution of shocks he might
        experience next period.
        """

        # Unpack the shock distribution
        Risky_next = self.RiskyDstn.atoms
        RiskyMax = np.max(Risky_next)
        RiskyMin = np.min(Risky_next)

        # bNrm represents R*a, balances after asset return shocks but before income.
        # This just uses the highest risky return as a rough shifter for the aXtraGrid.
        if self.zero_bound:
            # no point at zero
            self.aNrmGrid = self.aXtraGrid
            self.bNrmGrid = np.insert(
                RiskyMax * self.aXtraGrid, 0, RiskyMin * self.aXtraGrid[0]
            )
        else:
            # Add an asset point at exactly zero
            self.aNrmGrid = np.insert(self.aXtraGrid, 0, 0.0)
            self.bNrmGrid = RiskyMax * np.insert(self.aXtraGrid, 0, 0.0)

        # Get grid and shock sizes, for easier indexing
        self.aNrmCount = self.aNrmGrid.size
        self.ShareCount = self.ShareGrid.size

        # Make tiled arrays to calculate future realizations of bNrm
        # and Share when integrating over RiskyDstn
        self.aNrmMat, self.ShareMat = np.meshgrid(
            self.aNrmGrid, self.ShareGrid, indexing="ij"
        )

    def linear_interp(self, x_list, y_list):
        if self.zero_bound:
            # add a point at zero
            return LinearInterp(np.append(0.0, x_list), np.append(0.0, y_list))
        else:
            # already includes zero
            return LinearInterp(x_list, y_list)

    def bilinear_interp(self, f_values, x_list, y_list):
        if self.zero_bound:
            # insert a point at zero
            return BilinearInterp(
                np.insert(f_values, 0, 0.0, axis=0), np.append(0.0, x_list), y_list
            )
        else:
            # already includes zero
            return BilinearInterp(f_values, x_list, y_list)

    def calc_EndOfPrdvP(self):
        """
        Calculate end-of-period marginal value of assets and shares at each point
        in aNrm and ShareGrid. Does so by taking expectation of next period marginal
        values across income and risky return shocks.
        """

        def dvAltdbCondFunc(shocks, b_nrm):
            """
            Evaluate realizations of marginal value of market resources next period
            """

            p_shk = shocks[0] * self.PermGroFac
            t_shk = shocks[1]
            m_nrm_next = b_nrm / p_shk + t_shk

            vP_next = self.vPfunc_next(m_nrm_next)

            return p_shk ** (-self.CRRA) * vP_next

        # Calculate intermediate marginal value of bank balances by taking expectations over income shocks
        dvAltdbGrid = calc_expectation(self.IncShkDstn, dvAltdbCondFunc, self.bNrmGrid)
        dvAltdbNvrsGrid = self.uPinv(dvAltdbGrid)
        dvAltdbNvrsFunc = self.linear_interp(self.bNrmGrid, dvAltdbNvrsGrid)
        dvAltdbFunc = MargValueFuncCRRA(dvAltdbNvrsFunc, self.CRRA)

        def dvOptdxCondFunc(shock, a_nrm, share):
            """
            Evaluate realizations of marginal value of share
            """

            r_diff = shock - self.Rfree
            r_port = self.Rfree + r_diff * share
            b_nrm = a_nrm * r_port

            vP_next = dvAltdbFunc(b_nrm)

            dvda = r_port * vP_next
            dvds = r_diff * a_nrm * vP_next

            return dvda, dvds

        # Calculate end-of-period marginal value of risky portfolio share by taking expectations
        dvOptdxGrid = (
            self.DiscFac
            * self.LivPrb
            * calc_expectation(
                self.RiskyDstn, dvOptdxCondFunc, self.aNrmMat, self.ShareMat
            )
        )

        self.EndOfPrddvda = dvOptdxGrid[0]
        self.EndOfPrddvds = dvOptdxGrid[1]

        self.EndOfPrddvdaNvrs = self.uPinv(self.EndOfPrddvda)
        self.EndOfPrddvdaNvrsFunc = self.bilinear_interp(
            self.EndOfPrddvdaNvrs, self.aNrmGrid, self.ShareGrid
        )
        self.EndOfPrddvdaFunc = MargValueFuncCRRA(self.EndOfPrddvdaNvrsFunc, self.CRRA)

    def calc_EndOfPrdv(self):
        def vAltCondFunc(shocks, b_nrm):
            p_shk = shocks[0] * self.PermGroFac
            t_shk = shocks[1]
            m_nrm_next = b_nrm / p_shk + t_shk

            v_next = self.vFunc_next(m_nrm_next)

            return p_shk ** (1.0 - self.CRRA) * v_next

        vAltGrid = calc_expectation(self.IncShkDstn, vAltCondFunc, self.bNrmGrid)
        vAltNvrsGrid = self.uinv(vAltGrid)
        vAltNvrsFunc = self.linear_interp(self.bNrmGrid, vAltNvrsGrid)
        vAltFunc = ValueFuncCRRA(vAltNvrsFunc, self.CRRA)

        def vOptCondFunc(shock, a_nrm, share):
            r_diff = shock - self.Rfree
            r_port = self.Rfree + r_diff * share
            b_nrm = a_nrm * r_port

            return vAltFunc(b_nrm)

        vOptGrid = (
            self.DiscFac
            * self.LivPrb
            * calc_expectation(
                self.RiskyDstn, vOptCondFunc, self.aNrmMat, self.ShareMat
            )
        )
        vOptNvrsGrid = self.uinv(vOptGrid)
        vOptNvrsFunc = self.bilinear_interp(vOptNvrsGrid, self.aNrmGrid, self.ShareGrid)
        vOptFunc = ValueFuncCRRA(vOptNvrsFunc, self.CRRA)

        self.EndOfPrdvFunc = vOptFunc

    def prepare_to_optimize_share(self):
        # use the same grid for cash on hand
        self.mNrmGrid = self.aXtraGrid * 1.5
        self.mNrmGridW0 = np.append(0.0, self.mNrmGrid)

    def optimize_risky_share(self):
        if self.method in ["root", "fxp"]:  # by root finding
            super().optimize_share()
            self.ShareOpt = self.Share_now
            self.EndOfPrdvPgrid = self.cNrmAdj_now

            # calculate end of period vp function given optimal share
            # this is created regardless of how optimal share is found

            # only need vPfunc if root finding
            self.EndOfPrdvFunc = NullFunc()
            # can be compared against EndofPrddvPgrid
            vP = self.EndOfPrddvdaFunc(self.aNrmGrid, self.ShareOpt)
            vPnvrs = self.uPinv(vP)
            vPnvrsFunc = self.linear_interp(self.aNrmGrid, vPnvrs)
            self.EndOfPrdvPfunc = MargValueFuncCRRA(vPnvrsFunc, self.CRRA)

        elif self.method == "max":  # by grid search maximization
            self.optimize_share_max()

            # only need vFunc if maximizing
            self.EndOfPrdvPfunc = NullFunc()
            # can be compared against EndofPrdvGrid
            vGrid = self.EndOfPrdvFunc(self.aNrmGrid, self.ShareOpt)
            vNvrsGrid = self.uinv(vGrid)
            vNvrsFunc = self.linear_interp(self.aNrmGrid, vNvrsGrid)
            self.EndOfPrdvFunc = ValueFuncCRRA(vNvrsFunc, self.CRRA)

    def optimize_share_max(self):
        def objective_share(share, a_nrm):
            return -self.EndOfPrdvFunc(a_nrm, share)

        share_opt = np.empty_like(self.aNrmGrid)
        v_opt = np.empty_like(self.aNrmGrid)
        for ai in range(self.aNrmGrid.size):
            a_nrm = self.aNrmGrid[ai]
            sol = minimize_scalar(
                objective_share,
                bounds=(0, 1),
                args=a_nrm,
                method="bounded",
            )
            share_opt[ai] = sol.x
            v_opt[ai] = -sol.fun

        self.ShareOpt = share_opt
        self.EndOfPrdvGrid = v_opt

    def prepare_to_optimize_consumption(self):
        self.aNrmGrid = self.aXtraGrid
        self.aNrmGridW0 = np.append(0.0, self.aXtraGrid)

    def optimize_consumption(self):
        if self.method == "root":
            self.optimize_consumption_root()
        elif self.method == "max":
            self.optimize_consumption_max()
        elif self.method == "fxp":
            self.optimize_consumption_fxp()

    def optimize_consumption_max(self):
        def objective_max(c_nrm, m_nrm):
            a_nrm = m_nrm - c_nrm
            value = self.uCD(c_nrm, a_nrm) + self.EndOfPrdvFunc(a_nrm)
            return -value  # negative because we are minimizing

        c_opt = np.empty_like(self.mNrmGrid)
        v_opt = np.empty_like(self.mNrmGrid)
        for mi in range(self.mNrmGrid.size):
            m_nrm = self.mNrmGrid[mi]
            sol = minimize_scalar(
                objective_max,
                method="bounded",
                args=m_nrm,
                bounds=(EPSILON, m_nrm - EPSILON),
            )

            c_opt[mi] = sol.x
            v_opt[mi] = -sol.fun

        self.cNrmGrid = c_opt  # as a function of m
        self.vNrmGrid = v_opt  # as a function of m

    def optimize_consumption_fxp(self):
        gamma = self.WealthShare
        gamma_alt = self.WealthShare_Alt

        def foc_in_doc(c_nrm_prev, a_nrm):
            # first order condition solving for outer c
            num = self.muInv(self.EndOfPrdvPfunc(a_nrm)) * (a_nrm**gamma_alt)
            denom = self.muInv(1 - (c_nrm_prev / a_nrm) * (gamma / (1 - gamma)))

            c_nrm_next = num / denom

            return c_nrm_next

        def foc_outer_c(c_nrm_prev, a_nrm):
            # save as first order condition above without using muInv
            num = self.EndOfPrdvPfunc(a_nrm)
            denom = (1 - gamma) * (a_nrm / c_nrm_prev) ** gamma - gamma * (
                c_nrm_prev / a_nrm
            ) ** (1 - gamma)

            c_nrm_next = (num / denom) ** (-1 / (self.CRRA) * (1 - gamma))

            return c_nrm_next

        def foc_inner_c(c_nrm_prev, a_nrm):
            # first order condition solving for inner c
            lhs = 1 - self.EndOfPrdvPfunc(a_nrm) / self.uPc(c_nrm_prev, a_nrm)
            return lhs * (1 - gamma) / gamma * a_nrm

        def euler_fxp(c_nrm_prev, a_nrm, sign=1):
            euler = (
                self.uPc(c_nrm, a_nrm)
                - self.uPa(c_nrm, a_nrm)
                - self.EndOfPrdvPfunc(a_nrm)
            )
            return np.abs(sign * euler + c_nrm_prev)

        # first guess should be based off previous solution
        c_guess = self.cFunc_next(self.aNrmGrid)

        c_opt = np.empty_like(self.aNrmGrid)

        # conclusion: fixed point itteration is good only when the initial
        # guess is *very* close to the actual solution
        # when c_nrm = c_now FPI provides a nice solution (c_now - c_opt is small)
        # when c_nrm = c_next FPI sometimes breaks and (c_next - c_opt is large)
        for ai in range(self.aNrmGrid.size):
            c_nrm = c_guess[ai]
            a_nrm = self.aNrmGrid[ai]
            try:
                c_opt[ai] = fixed_point(euler_fxp, c_nrm, args=(a_nrm, 1))
            except RuntimeError:
                try:
                    c_opt[ai] = fixed_point(euler_fxp, c_nrm, args=(a_nrm, -1))
                except RuntimeError:
                    print("error")
                    c_opt[ai] = c_nrm

        self.cNrmGrid = c_opt

    def optimize_consumption_root(self):
        def objective_root(c_nrm, m_nrm):
            a_nrm = m_nrm - c_nrm
            euler = (
                self.uPc(c_nrm, a_nrm)
                - self.uPa(c_nrm, a_nrm)
                - self.EndOfPrdvPfunc(a_nrm)
            )
            return euler

        c_opt = np.empty_like(self.mNrmGrid)
        c_init = self.solution_next.cFunc.y_list[1:]

        for mi in range(self.mNrmGrid.size):
            m_nrm = self.mNrmGrid[mi]
            # sol = root_scalar(
            #     objective_root,
            #     args=m_nrm,
            #     method="toms748",
            #     bracket=(EPSILON, m_nrm - EPSILON),
            # )
            # c_opt[mi] = sol.root

            # using multidimensional root finding is much faster
            # can I use a better initial guess?
            c_init = self.solution_next.cFunc.y_list[1:]
            sol = root(objective_root, x0=c_init[mi], args=m_nrm)
            c_opt[mi] = sol.x

        self.cNrmGrid = c_opt  # as a function of m

    def make_basic_solution(self):
        """
        Given end of period assets and end of period marginal values, construct
        the basic solution for this period.
        """

        cNrmGrid = self.cNrmGrid
        cNrmGridW0 = np.append(0.0, self.cNrmGrid)

        if self.method in ["root", "max"]:
            # c solution is a function of m

            mNrmGrid = self.mNrmGrid
            mNrmGridW0 = self.mNrmGridW0

            aNrmGrid = mNrmGrid - cNrmGrid
            aNrmGridW0 = np.append(0.0, aNrmGrid)

        elif self.method == "fxp":
            # c solution is a function of a

            aNrmGrid = self.aXtraGrid
            aNrmGridW0 = np.append(0.0, aNrmGrid)

            mNrmGrid = aNrmGrid + cNrmGrid
            mNrmGridW0 = np.append(0.0, mNrmGrid)

        self.cFunc = LinearInterp(mNrmGridW0, cNrmGridW0)
        self.cEndFunc = LinearInterp(aNrmGridW0, cNrmGridW0)

        if self.method in ["root", "fxp"]:
            vPGrid = self.uPc(cNrmGrid, aNrmGrid)
            vPNvrsGrid = self.uPinv(vPGrid)
            vPNvrsGridW0 = np.append(0.0, vPNvrsGrid)
            vPNvrsFunc = LinearInterp(mNrmGridW0, vPNvrsGridW0)
            self.vPfunc = MargValueFuncCRRA(vPNvrsFunc, self.CRRA)
            self.vPEndFunc = self.EndOfPrdvPfunc

            self.vFunc = NullFunc()
            self.vEndFunc = NullFunc()

        elif self.method == "max":
            vGrid = self.uCD(cNrmGrid, aNrmGrid) + self.EndOfPrdvFunc(aNrmGrid)
            vNvrsGrid = self.uinv(vGrid)
            vNvrsGridW0 = np.append(0.0, vNvrsGrid)
            vNvrsFunc = LinearInterp(mNrmGridW0, vNvrsGridW0)
            self.vFunc = ValueFuncCRRA(vNvrsFunc, self.CRRA)
            self.vEndFunc = self.EndOfPrdvFunc

            self.vPfunc = NullFunc()
            self.vPEndFunc = NullFunc()

        if self.zero_bound:
            # add zero back
            self.shareEndFunc = LinearInterp(
                aNrmGridW0,
                np.append(1.0, self.ShareOpt),
                intercept_limit=self.ShareLimit,
                slope_limit=0.0,
            )

            self.shareFunc = LinearInterp(
                mNrmGridW0,
                np.append(1.0, self.ShareOpt),
                intercept_limit=self.ShareLimit,
                slope_limit=0.0,
            )
        else:
            self.shareEndFunc = LinearInterp(
                aNrmGridW0,
                self.ShareOpt,
                intercept_limit=self.ShareLimit,
                slope_limit=0.0,
            )

            self.shareFunc = LinearInterp(
                mNrmGridW0,
                self.ShareOpt,
                intercept_limit=self.ShareLimit,
                slope_limit=0.0,
            )

    def make_porfolio_solution(self):
        self.solution = WealthPortfolioSolution(
            cFunc=self.cFunc,
            cEndFunc=self.cEndFunc,
            shareFunc=self.shareFunc,
            shareEndFunc=self.shareEndFunc,
            vFunc=self.vFunc,
            vPfunc=self.vPfunc,
        )

        self.solution.shareEndFunc = self.shareEndFunc

    def solve(self):
        # Make arrays of end-of-period assets and end-of-period marginal values
        self.prepare_to_calc_EndOfPrd()
        if self.method in ["root", "fxp"]:
            self.calc_EndOfPrdvP()
        elif self.method == "max":
            self.calc_EndOfPrdv()

        # Construct a basic solution for this period
        self.prepare_to_optimize_share()
        self.optimize_risky_share()

        self.prepare_to_optimize_consumption()
        self.optimize_consumption()

        self.make_basic_solution()
        self.make_porfolio_solution()

        return self.solution


init_wealth_portfolio = init_portfolio.copy()
init_wealth_portfolio["TranShkCount"] = 1
init_wealth_portfolio["TranShkStd"] = [0.0]
init_wealth_portfolio["PermShkStd"] = [0.0]
init_wealth_portfolio["UnempPrb"] = 0.0
init_wealth_portfolio["WealthShare"] = 1 / 3

# from TRP specs
init_wealth_portfolio["RiskyAvg"] = 1.0486
init_wealth_portfolio["RiskyStd"] = 0.1375
init_wealth_portfolio["Rfree"] = 1.016
init_wealth_portfolio["RiskyCount"] = 25
