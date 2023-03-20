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


class TerminalBequestWarmGlowConsumerType(IndShockConsumerType):
    time_inv_ = IndShockConsumerType.time_inv_ + [
        "BeqCRRA",
        "BeqRelVal",
        "BeqStoneGeary",
    ]

    def __init__(self, **kwds):
        params = init_warm_glow.copy()
        params.update(kwds)

        super().__init__(**params)

        self.pseudo_terminal = True

    def update_solution_terminal(self):
        DiscFacEff = self.BeqRelVal / self.DiscFac
        TranShkMin = np.min(self.TranShkDstn[0].atoms)
        StoneGearyEff = self.BeqStoneGeary - TranShkMin

        warm_glow = UtilityFuncStoneGeary(
            self.BeqCRRA, factor=DiscFacEff, shifter=StoneGearyEff
        )

        self.solution_terminal.cFunc = lambda m: m - TranShkMin
        self.solution_terminal.vFunc = lambda m: warm_glow(m)
        self.solution_terminal.vPfunc = lambda m: warm_glow.der(m)
        self.solution_terminal.vPPfunc = lambda m: warm_glow.der(m, order=2)
        self.solution_terminal.mNrmMin = np.maximum(TranShkMin, -StoneGearyEff)


class TerminalBequestWarmGlowPortfolioType(
    PortfolioConsumerType, TerminalBequestWarmGlowConsumerType
):
    def __init__(self, **kwds):
        super().__init__(**kwds)

        self.solve_one_period = make_one_period_oo_solver(
            TerminalBequestWarmGlowPortfolioSolver
        )


class AccidentalBequestWarmGlowConsumerType(TerminalBequestWarmGlowConsumerType):
    def __init__(self, **kwds):
        super().__init__(**kwds)

        self.solve_one_period = make_one_period_oo_solver(
            AccidentalBequestWarmGlowSolver
        )


class AccidentalBequestWarmGlowPortfolioType:
    pass


class AccidentalBequestWarmGlowSolver(ConsIndShockSolver):
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

        self.warm_glow = UtilityFuncStoneGeary(
            self.BeqCRRA, self.BeqRelVal, self.BeqStoneGeary
        )

    def calc_EndOfPrdvP(self):
        EndofPrdvP = super().calc_EndOfPrdvP()

        return EndofPrdvP + self.warm_glow.der(self.aNrmNow)


class TerminalBequestWarmGlowPortfolioSolver(ConsPortfolioSolver):
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
init_warm_glow["BeqRelVal"] = 1.0  # Value of bequest relative to consumption
init_warm_glow["BeqStoneGeary"] = 0.0  # Shifts the utility function
init_warm_glow["BeqCRRA"] = init_lifecycle["CRRA"]


init_portfolio_bequest = init_warm_glow.copy()
init_portfolio_bequest.update(init_portfolio)
