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
    init_lifecycle,
)
from HARK.ConsumptionSaving.ConsPortfolioModel import (
    ConsPortfolioSolver,
    PortfolioConsumerType,
    init_portfolio,
)
from HARK.core import make_one_period_oo_solver
from HARK.rewards import UtilityFuncStoneGeary


class BequestWarmGlowConsumerType(IndShockConsumerType):
    time_inv_ = IndShockConsumerType.time_inv_ + [
        "TermBeqCRRA",
        "TermBeqRelVal",
        "TermBeqStoneGeary",
    ]
    time_vary_ = IndShockConsumerType.time_vary_ + [
        "BeqCRRA",
        "BeqRelVal",
        "BeqStoneGeary",
    ]

    def __init__(self, **kwds):
        params = init_warm_glow.copy()
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

        if isinstance(self.BeqRelVal, (int, float)):
            self.BeqRelVal = [self.BeqRelVal] * self.T_cycle
        elif len(self.BeqRelVal) == 1:
            self.BeqRelVal *= self.T_cycle
        elif len(self.BeqRelVal) != self.T_cycle:
            raise ValueError(
                "Bequest relative value parameter must be a single value or a list of length T_cycle"
            )

        if isinstance(self.BeqStoneGeary, (int, float)):
            self.BeqStoneGeary = [self.BeqStoneGeary] * self.T_cycle
        elif len(self.BeqStoneGeary) == 1:
            self.BeqStoneGeary *= self.T_cycle
        elif len(self.BeqStoneGeary) != self.T_cycle:
            raise ValueError(
                "Bequest Stone-Geary parameter must be a single value or a list of length T_cycle"
            )

    def update_solution_terminal(self):
        if self.TermBeqRelVal == 0.0:  # No terminal bequest
            super().update_solution_terminal()
        else:
            self.pseudo_terminal = True
            DiscFacEff = self.TermBeqRelVal / self.DiscFac
            TranShkMin = np.min(self.TranShkDstn[0].atoms)
            StoneGearyEff = self.TermBeqStoneGeary - TranShkMin

            warm_glow = UtilityFuncStoneGeary(
                self.TermBeqCRRA, factor=DiscFacEff, shifter=StoneGearyEff
            )

            self.solution_terminal.cFunc = lambda m: m - TranShkMin
            self.solution_terminal.vFunc = lambda m: warm_glow(m)
            self.solution_terminal.vPfunc = lambda m: warm_glow.der(m)
            self.solution_terminal.vPPfunc = lambda m: warm_glow.der(m, order=2)
            self.solution_terminal.mNrmMin = np.maximum(TranShkMin, -StoneGearyEff)


class BequestWarmGlowPortfolioType(PortfolioConsumerType, BequestWarmGlowConsumerType):
    def __init__(self, **kwds):
        super().__init__(**kwds)

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
        BeqRelVal,
        BeqStoneGeary,
    ):
        self.BeqCRRA = BeqCRRA
        self.BeqRelVal = BeqRelVal
        self.BeqStoneGeary = BeqStoneGeary
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

        BeqRelValEff = (1.0 - self.LivPrb) * self.BeqRelVal

        self.warm_glow = UtilityFuncStoneGeary(
            self.BeqCRRA, BeqRelValEff, self.BeqStoneGeary
        )

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
        BeqRelVal,
        BeqStoneGeary,
    ):
        self.BeqCRRA = BeqCRRA
        self.BeqRelVal = BeqRelVal
        self.BeqStoneGeary = BeqStoneGeary
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

        self.warm_glow = UtilityFuncStoneGeary(
            self.BeqCRRA, self.BeqRelVal, self.BeqStoneGeary
        )

    def calc_EndOfPrdvP(self):
        super().calc_EndOfPrdvP()

        self.EndofPrddvda = self.EndOfPrddvda + self.warm_glow.der(self.aNrm_tiled)
        self.EndOfPrddvdaNvrs = self.uPinv(self.EndOfPrddvda)


init_warm_glow = init_lifecycle.copy()
init_warm_glow["TermBeqCRRA"] = init_lifecycle["CRRA"]
init_warm_glow["TermBeqRelVal"] = 1.0
init_warm_glow["TermBeqStoneGeary"] = 0.0
init_warm_glow["BeqRelVal"] = 1.0  # Value of bequest relative to consumption
init_warm_glow["BeqStoneGeary"] = 0.0  # Shifts the utility function
init_warm_glow["BeqCRRA"] = init_lifecycle["CRRA"]


init_portfolio_bequest = init_warm_glow.copy()
init_portfolio_bequest.update(init_portfolio)
