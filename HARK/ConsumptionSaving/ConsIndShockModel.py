# -*- coding: utf-8 -*-

from HARK.ConsumptionSaving.ConsIndShockModel_AgentSolve \
    import (
        ConsumerSolution, ConsumerSolutionOneStateCRRA,
        ConsPerfForesightSolver, ConsIndShockSetup,
        ConsIndShockSolverBasic, ConsIndShockSolver,
        ConsKinkedRsolver)

from HARK.ConsumptionSaving.ConsIndShockModel_AgentTypes \
    import (consumer_terminal_nobequest_onestate, PerfForesightConsumerType,
            IndShockConsumerType, KinkedRconsumerType)

from HARK.ConsumptionSaving.ConsIndShockModel_AgentDicts \
    import (
        init_perfect_foresight,
        init_idiosyncratic_shocks,
        init_kinked_R,
        init_lifecycle,
        init_cyclical)

from HARK.utilities import CRRAutility as utility
from HARK.utilities import CRRAutilityP as utilityP
from HARK.utilities import CRRAutilityPP as utilityPP
from HARK.utilities import CRRAutilityP_inv as utilityP_inv
from HARK.utilities import CRRAutility_invP as utility_invP
from HARK.utilities import CRRAutility_inv as utility_inv
from HARK.utilities import CRRAutilityP as utilityP_invP

#from HARK.ConsumptionSaving.ConsIndShockModel_AgentSolve_EndOfPeriodValue \
#    import (ConsIndShockSetupEOP, ConsIndShockSolverBasicEOP, ConsIndShockSolverEOP)

"""
Classes to define and solve canonical consumption-saving models with a single
state variable.  All models here assume CRRA utility with geometric discounting,
and if income shocks exist they are fully transitory or fully permanent.

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
    "ConsIndShockSetupEOP",
    "ConsIndShockSolverBasicEOP",
    "ConsIndShockSolverEOP",
    "ConsKinkedRsolver",
    "consumer_terminal_nobequest_onestate",
    "PerfForesightConsumerType",
    "IndShockConsumerType",
    "KinkedRconsumerType",
    "init_perfect_foresight",
    "init_idiosyncratic_shocks",
    "init_kinked_R",
    "init_lifecycle",
    "init_cyclical",
    "utility",
    "utilityP",
    "utilityPP",
    "utilityP_inv",
    "utility_invP",
    "utility_inv",
    "utilityP_invP"
]
