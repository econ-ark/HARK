"""
Classes to solve consumption-saving models with a bequest motive and
idiosyncratic shocks to income and wealth. All models here assume
separable CRRA utility of consumption and Stone-Geary utility of
savings with geometric discounting of the continuation value and
shocks to income that have transitory and/or permanent components.

It currently solves 2 types of models:
    1) A standard lifecycle model with a terminal and/or accidental bequest motive.
    3) A portfolio choice model with a terminal and/or accidental bequest motive.
"""

import numpy as np

from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsIndShockSolver,
    IndShockConsumerType,
    init_idiosyncratic_shocks,
    init_lifecycle,
)
from HARK.ConsumptionSaving.ConsPortfolioModel import (
    ConsPortfolioSolver,
    PortfolioConsumerType,
    init_portfolio,
)
from HARK.core import make_one_period_oo_solver
from HARK.interpolation import (
    LinearInterp,
    MargMargValueFuncCRRA,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.rewards import UtilityFuncCRRA, UtilityFuncStoneGeary


class BequestWarmGlowConsumerType(IndShockConsumerType):
    # time_inv_ = IndShockConsumerType.time_inv_ + [
    #     "TermBeqCRRA",
    #     "TermBeqFac",
    #     "TermBeqShift",
    # ]
    time_vary_ = IndShockConsumerType.time_vary_ + [
        "BeqCRRA",
        "BeqFac",
        "BeqShift",
    ]

    def __init__(self, **kwds):
        params = init_wealth_in_utility.copy()
        params.update(kwds)

        super().__init__(**params)

        self.solve_one_period = make_one_period_oo_solver(BequestWarmGlowConsumerSolver)

    def update(self):
        super().update()
        self.update_parameters()

    def update_parameters(self):
        if isinstance(self.BeqCRRA, (int, float)):
            self.BeqCRRA = [self.BeqCRRA] * self.T_cycle
        elif len(self.BeqCRRA) == 1:
            self.BeqCRRA *= self.T_cycle
        elif len(self.BeqCRRA) != self.T_cycle:
            raise ValueError(
                "Bequest CRRA parameter must be a single value or a list of length T_cycle"
            )

        if isinstance(self.BeqFac, (int, float)):
            self.BeqFac = [self.BeqFac] * self.T_cycle
        elif len(self.BeqFac) == 1:
            self.BeqFac *= self.T_cycle
        elif len(self.BeqFac) != self.T_cycle:
            raise ValueError(
                "Bequest relative value parameter must be a single value or a list of length T_cycle"
            )

        if isinstance(self.BeqShift, (int, float)):
            self.BeqShift = [self.BeqShift] * self.T_cycle
        elif len(self.BeqShift) == 1:
            self.BeqShift *= self.T_cycle
        elif len(self.BeqShift) != self.T_cycle:
            raise ValueError(
                "Bequest Stone-Geary parameter must be a single value or a list of length T_cycle"
            )

    def update_solution_terminal(self):
        if self.TermBeqFac == 0.0:  # No terminal bequest
            super().update_solution_terminal()
        else:
            utility = UtilityFuncCRRA(self.CRRA)

            warm_glow = UtilityFuncStoneGeary(
                self.TermBeqCRRA, factor=self.TermBeqFac, shifter=self.TermBeqShift
            )

            aNrmGrid = (
                np.append(0.0, self.aXtraGrid)
                if self.TermBeqShift != 0.0
                else self.aXtraGrid
            )
            cNrmGrid = utility.derinv(warm_glow.der(aNrmGrid))
            vGrid = utility(cNrmGrid) + warm_glow(aNrmGrid)
            cNrmGridW0 = np.append(0.0, cNrmGrid)
            mNrmGridW0 = np.append(0.0, aNrmGrid + cNrmGrid)
            vNvrsGridW0 = np.append(0.0, utility.inv(vGrid))

            cFunc_term = LinearInterp(mNrmGridW0, cNrmGridW0)
            vNvrsFunc_term = LinearInterp(mNrmGridW0, vNvrsGridW0)
            vFunc_term = ValueFuncCRRA(vNvrsFunc_term, self.CRRA)
            vPfunc_term = MargValueFuncCRRA(cFunc_term, self.CRRA)
            vPPfunc_term = MargMargValueFuncCRRA(cFunc_term, self.CRRA)

            self.solution_terminal.cFunc = cFunc_term
            self.solution_terminal.vFunc = vFunc_term
            self.solution_terminal.vPfunc = vPfunc_term
            self.solution_terminal.vPPfunc = vPPfunc_term
            self.solution_terminal.mNrmMin = 0.0


class BequestWarmGlowPortfolioType(PortfolioConsumerType, BequestWarmGlowConsumerType):
    def __init__(self, **kwds):
        params = init_portfolio_bequest.copy()
        params.update(kwds)

        super().__init__(**params)

        self.solve_one_period = make_one_period_oo_solver(
            BequestWarmGlowPortfolioSolver
        )

    def update(self):
        return BequestWarmGlowConsumerType.update(self)

    def update_solution_terminal(self):
        return BequestWarmGlowConsumerType.update_solution_terminal(self)


class BequestWarmGlowConsumerSolver(ConsIndShockSolver):
    def __init__(
        self,
        solution_next,
        IncShkDstn,
        LivPrb,
        DiscFac,
        CRRA,
        Rfree,
        PermGroFac,
        BoroCnstArt,
        aXtraGrid,
        BeqCRRA,
        BeqFac,
        BeqShift,
    ):
        self.BeqCRRA = BeqCRRA
        self.BeqFac = BeqFac
        self.BeqShift = BeqShift
        vFuncBool = False
        CubicBool = False

        super().__init__(
            solution_next,
            IncShkDstn,
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

    def def_utility_funcs(self):
        super().def_utility_funcs()

        BeqFacEff = (1.0 - self.LivPrb) * self.BeqFac

        self.warm_glow = UtilityFuncStoneGeary(self.BeqCRRA, BeqFacEff, self.BeqShift)

    def calc_EndOfPrdvP(self):
        EndofPrdvP = super().calc_EndOfPrdvP()

        return EndofPrdvP + self.warm_glow.der(self.aNrmNow)


class BequestWarmGlowPortfolioSolver(ConsPortfolioSolver):
    def __init__(
        self,
        solution_next,
        ShockDstn,
        IncShkDstn,
        RiskyDstn,
        LivPrb,
        DiscFac,
        CRRA,
        Rfree,
        PermGroFac,
        BoroCnstArt,
        aXtraGrid,
        ShareGrid,
        AdjustPrb,
        ShareLimit,
        BeqCRRA,
        BeqFac,
        BeqShift,
    ):
        self.BeqCRRA = BeqCRRA
        self.BeqFac = BeqFac
        self.BeqShift = BeqShift
        vFuncBool = False
        DiscreteShareBool = False
        IndepDstnBool = True

        super().__init__(
            solution_next,
            ShockDstn,
            IncShkDstn,
            RiskyDstn,
            LivPrb,
            DiscFac,
            CRRA,
            Rfree,
            PermGroFac,
            BoroCnstArt,
            aXtraGrid,
            ShareGrid,
            vFuncBool,
            AdjustPrb,
            DiscreteShareBool,
            ShareLimit,
            IndepDstnBool,
        )

    def def_utility_funcs(self):
        super().def_utility_funcs()

        self.warm_glow = UtilityFuncStoneGeary(self.BeqCRRA, self.BeqFac, self.BeqShift)

    def calc_EndOfPrdvP(self):
        super().calc_EndOfPrdvP()

        self.EndofPrddvda = self.EndOfPrddvda + self.warm_glow.der(self.aNrm_tiled)
        self.EndOfPrddvdaNvrs = self.uPinv(self.EndOfPrddvda)


init_wealth_in_utility = init_idiosyncratic_shocks.copy()
init_wealth_in_utility["BeqCRRA"] = init_idiosyncratic_shocks["CRRA"]
init_wealth_in_utility["BeqFac"] = 1.0
init_wealth_in_utility["BeqShift"] = 0.0
# init_wealth_in_utility["TermBeqCRRA"] = init_idiosyncratic_shocks["CRRA"]

init_warm_glow = init_lifecycle.copy()
init_warm_glow["TermBeqCRRA"] = init_lifecycle["CRRA"]
init_warm_glow["TermBeqFac"] = 1.0
init_warm_glow["TermBeqShift"] = 0.0

init_accidental_bequest = init_warm_glow.copy()
init_accidental_bequest["BeqFac"] = 1.0  # Value of bequest relative to consumption
init_accidental_bequest["BeqShift"] = 0.0  # Shifts the utility function
init_accidental_bequest["BeqCRRA"] = init_lifecycle["CRRA"]

init_portfolio_bequest = init_accidental_bequest.copy()
init_portfolio_bequest.update(init_portfolio)
