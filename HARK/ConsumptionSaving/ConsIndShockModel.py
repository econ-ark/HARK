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
Classes to define and solve canonical consumption-saving models with a single
state variable.  All models assume CRRA utility with geometric discounting,
and if income shocks exist they are fully transitory or fully permanent.

It currently solves three types of models:
   1) `PerfForesightConsumerType`
      * A basic perfect foresight consumption-saving model with no uncertainty.
      * Features of the model prepare it for convenient inheritance
   2) `IndShockConsumerType`
      * A consumption-saving model with transitory and permanent income shocks
      * Inherits from PF model
   3) `KinkedRconsumerType`
      * `IndShockConsumerType` model but with interest rate on debt, `Rboro`
        greater than the interest rate earned on savings, `Rboro > `Rsave`

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
