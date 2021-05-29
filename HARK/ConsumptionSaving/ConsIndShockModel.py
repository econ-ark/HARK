# -*- coding: utf-8 -*-

from HARK.ConsumptionSaving.ConsIndShockModel_AgentSolve \
    import (
        ConsumerSolution, ConsumerSolutionOneStateCRRA,
        ConsPerfForesightSolver, ConsIndShockSetup,
        ConsIndShockSolverBasic, ConsIndShockSolver,
        ConsKinkedRsolver)

from HARK.ConsumptionSaving.ConsIndShockModel_AgentTypes \
    import (consumer_onestate_nobequest, PerfForesightConsumerType,
            IndShockConsumerType, KinkedRconsumerType)

from HARK.ConsumptionSaving.ConsIndShockModel_AgentDicts \
    import (init_perfect_foresight,
            init_idiosyncratic_shocks, init_kinked_R,
            init_lifecycle, init_cyclical)

"""
Classes to solve canonical consumption-saving models with idiosyncratic shocks
to income.  All models here assume CRRA utility with geometric discounting, no
bequest motive, and income shocks that are fully transitory or fully permanent.

It currently solves three types of models:
   1) `PerfForesightConsumerType`
      * A basic "perfect foresight" consumption-saving model with no uncertainty.
      * Features of the model prepare it for convenient inheritance
   2) `IndShockConsumerType`
      * A consumption-saving model with transitory and permanent income shocks
      * Inherits from PF model
   3) `KinkedRconsumerType`
      * `IndShockConsumerType` model but with an interest rate paid on debt, `Rboro`
        greater than the interest rate earned on savings, `Rboro > `Rsave`

See NARK https://HARK.githhub.io/Documentation/NARK for variable naming conventions.
See https://hark.readthedocs.io for mathematical descriptions of the models being solved.
"""

__all__ = [
    "ConsumerSolution",
    "ConsumerSolutionOneStateCRRA",
    "ConsPerfForesightSolver",
    "ConsIndShockSetup",
    "ConsIndShockSolverBasic",
    "ConsIndShockSolver",
    "ConsKinkedRsolver",
    "consumer_onestate_nobequest",
    "PerfForesightConsumerType",
    "IndShockConsumerType",
    "KinkedRconsumerType",
    "init_perfect_foresight",
    "init_idiosyncratic_shocks",
    "init_kinked_R",
    "init_lifecycle",
    "init_cyclical",
]
