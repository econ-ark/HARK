"""
Classes to solve canonical consumption-saving models with idiosyncratic shocks
to income.  All models here assume CRRA utility with geometric discounting, no
bequest motive, and income shocks that are fully transitory or fully permanent.

It currently solves three types of models:
   1) A very basic "perfect foresight" consumption-savings model with no uncertainty.
   2) A consumption-savings model with risk over transitory and permanent income shocks.
   3) The model described in (2), with an interest rate for debt that differs
      from the interest rate for savings.

See NARK https://HARK.githhub.io/Documentation/NARK for information on variable naming conventions.
See HARK documentation for mathematical descriptions of the models being solved.
"""

from HARK.ConsumptionSaving.ConsIndShockModelOld \
    import ConsumerSolution as ConsumerSolutionOld
from HARK.ConsumptionSaving.ConsIndShockModel_AgentSolve \
    import (
        ConsumerSolution, ConsumerSolutionOneNrmStateCRRA,
        ConsPerfForesightSolver, ConsIndShockSetup,
        ConsIndShockSolverBasic, ConsIndShockSolver
    )

from HARK.ConsumptionSaving.ConsIndShockModel_KinkedRSolver \
    import ConsKinkedRsolver

from HARK.ConsumptionSaving.ConsIndShockModel_AgentTypes \
    import (consumer_terminal_nobequest_onestate, PerfForesightConsumerType,
            IndShockConsumerType, KinkedRconsumerType,
            onestate_bequest_warmglow_homothetic
            )

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

__all__ = [
    "ConsumerSolutionOld",
    "ConsumerSolution",
    "ConsumerSolutionOneNrmStateCRRA",
    "ConsPerfForesightSolver",
    "ConsIndShockSetup",
    "ConsIndShockSolverBasic",
    "ConsIndShockSolver",
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
    "utilityP_invP",
    "onestate_bequest_warmglow_homothetic"
]
