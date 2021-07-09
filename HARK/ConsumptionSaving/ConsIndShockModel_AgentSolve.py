# -*- coding: utf-8 -*-
from HARK.core import (_log, core_check_condition, MetricObject)

from scipy.optimize import newton as find_zero_newton
from numpy import dot as E_dot  # easier to type
from numpy.testing import assert_approx_equal as assert_approx_equal
import numpy as np
from copy import deepcopy
from builtins import (str, breakpoint)
from types import SimpleNamespace
from IPython.lib.pretty import pprint
from HARK.ConsumptionSaving.ConsIndShockModel_Both import (
    def_reward,
    def_utility_CRRA, def_value_funcs, def_value_CRRA,
    def_transition_chosen__to__next_choice,
    def_transition_chosen__to__next_BOP,
    def_transition_choice__to__chosen,
)
from HARK.distribution import calc_expectation as expect_funcs_given_states
from HARK.interpolation import (CubicInterp, LowerEnvelope, LinearInterp,
                                MargValueFuncCRRA,
                                MargMargValueFuncCRRA)
from HARK import NullFunc
from HARK.ConsumptionSaving.ConsIndShockModelOld \
    import ConsumerSolution as ConsumerSolutionOlder

from HARK.ConsumptionSaving.ConsIndShockModel_Both \
    import (TransitionFunctions, def_transitions)


class agent_solution(MetricObject):
    """
    Framework for solution of a single stage/period of a decision problem.

    Provides a foundational structure that all models will
    share.  It must be specialized and elaborated to solve any
    particular problem.

    Parameters
    ----------
    soln_futr : agent_solution
    Returns
    -------
    soln_crnt : object containing solution to the current period

    Elements of the soln_crnt object contain, but are not limited to:
        Pars : The parameters used in solving the model
        Bilt : Objects constructed and retained from the solution process
        Modl : Equations of the model, in the form of the python code
            that instantiates the computational solution

            At a minimum, this is broken down into:

            states : predetermined variables at the time of decisions
            controls : variables under control of the decisionmaker
            reward : current period payoff as function of states and controls
            transitions : evolution of states
            choices : conditions that determine the agent's choices

            At a minimum, it should result in:

                [dr] : decision rule
                    Maps states into choices
                    Example: consumption function cFunc over market resources
                [v] : value function
                    Bellman value function the agent expects to experience for
                    behaving according to the dynamically optimal plan over
                    the remainder of the horizon.

    stge_kind : dict
        Dictionary with info about this solution stage
        One required entry keeps track of the nature of the stage:
            {'iter_status':'not initialized'}: Before model is set up
            {'iter_status':'finished'}: Stopping requirements are satisfied
                If such requirements are satisfied, {'tolerance':tolerance}
                should exist recording what convergence tolerance was satisfied
            {'iter_status':'iterator'}: Status during iteration
                solution[0].distance_last records the last distance
            {'iter_status':'terminal_partial'}: Bare-bones terminal period
                Does not contain all the info needed to begin solution
                Solver will augment and replace it with 'iterator' stage
        Other uses include keeping track of the nature of the next stage
    completed_cycles : integer
        The number of cycles of the model solved before this call
    solveMethod : str, optional
        The name of the solution method to use, e.g. 'EGM'
    """

    def __init__(self, *args,
                 stge_kind={'iter_status': 'not initialized'},
                 parameters_solver=None,
                 completed_cycles=0,
                 **kwds):

        self.E_tp1_ = Nexspectations()  # Next given this period choices
        self.t_E_ = Prospectations()  # Before this period choices
        self.Pars = Parameters()
        self.Bilt = Built()
        self.Bilt.completed_cycles = completed_cycles
        self.Bilt.stge_kind = stge_kind
        self.Bilt.parameters_solver = parameters_solver
        self.Modl = Elements()

        # Below: Likely transition types in their canonical possible order
        # Each status will end up transiting only to one subsequent status
        # There are multiple possibilities because models may skip many steps
        # For example, with no end of period shocks, you could go directly from
        # the "chosen" (after choice) status to the "next_BOP" status

        self.Modl.Transitions = {  # BOP = Beginnning of Problem/Period
            'BOP__to__choice': {},
            'choice__to__chosen': {},  # or
            'choice__to__next_BOP': {},
            'chosen__to__EOP': {},  # or
            'chosen__to__next_BOP': {},  # or
            'chosen__to__next_choice': {},
            'EOP__to__next_BOP': {},  # or
            'EOP__to__next_choice': {},  # EOP = End of Problem/Period
        }

        # Allow doublestruck or regular E for expectations
        self.𝔼_tp1_ = self.E_tp1_
        self.t_𝔼_ = self.t_E_

    def define_reward():
        # Determine the current period payoff (utility? profit?)
        pass

    def define_transitions(self):
        # Equations that define transitions that affect agent's state
        pass

    def prep_solve_to_finish():
        # Prep work not done in init but that needs to be done before
        # solve_to_finish
        pass

    def prep_solve_this_stge(crnt, futr):
        # Do any prep work that should be accomplished before tackling the
        # actual solution of this stage of the problem
        # Like building the Pars namespace of parameters for this period
        pass

    def solve_prepared_stge(self):
        crnt, futr = self.soln_crnt, self.soln_futr
        self.define_transitions(crnt, futr)
        self.define_reward(crnt)
        self.expectations_after_shocks_and_choices__E_tp1_(crnt)
        self.make_decision_rules_and_value_functions()
        self.expectations_before_shocks_or_choices__t_E()

        return crnt

    def expectations_after_shocks_and_choices__E_tp1_(self):
        self.build_facts()
        pass

    expectations_after_shocks_and_choices__E_tp1_ =\
        expectations_after_shocks_and_choices__𝔼_tp1_

    def expectations_before_shocks_or_choices__t_E(self):
        pass

    expectations_before_shocks_or_choices__t_E =\
        expectations_before_shocks_or_choices__t_𝔼


class Built(SimpleNamespace):
    """Objects built by solvers during course of solution."""

    pass


class Parameters(SimpleNamespace):
    """Parameters (both as passed, and as exposed for convenience). But not modified."""

    pass


class Expectations(SimpleNamespace):
    """Expectations about future period"""

    pass


class Nexspectations(SimpleNamespace):
    """Expectations about future period after current decisions"""

    pass


class Prospectations(SimpleNamespace):
    """Expectations prior to the realization of current period shocks"""

    pass


class ValueFunctions(SimpleNamespace):
    """Expectations across realization of stochastic shocks."""

    pass


class Ante_Choice(SimpleNamespace):
    """Expectations before choices or shocks."""


class Elements(SimpleNamespace):
    """Elements of the model in python/HARK code."""

    pass


class Equations(SimpleNamespace):
    """Description of the model in HARK and python syntax."""

    pass


class Successor(SimpleNamespace):
    """Objects retrieved from successor to the stage
    referenced in "self." Should contain everything needed to reconstruct
    solution to problem of self even if solution_next is not present.
    """

    pass


__all__ = [
    "ConsumerSolutionOlder",
    "ConsumerSolution",
    "ConsumerSolutionOneStateCRRA",
    "ConsPerfForesightSolver",
    "ConsIndShockSetup",
    "ConsIndShockSolverBasic",
    "ConsIndShockSolver",
    "ConsIndShockSetup",
]


# ConsumerSolution basically does nothing except add agent_solution
# content to old ConsumerSolutionOlder, plus documentation


class ConsumerSolution(ConsumerSolutionOlder, agent_solution):
    __doc__ = ConsumerSolutionOlder.__doc__
    __doc__ += """
    stge_kind : dict
        Dictionary with info about this solution stage
        One required entry keeps track of the nature of the stage:
            {'iter_status':'finished'}: Stopping requirements are satisfied
                If stopping requirements are satisfied, {'tolerance':tolerance}
                should exist recording what convergence tolerance was satisfied
            {'iter_status':'iterator'}: Solution during iteration
                solution[0].distance_last records the last distance
            {'iter_status':'terminal_partial'}: Bare-bones terminal period
                Does not contain all the info needed to begin solution
                Solver will augment and replace it with 'iterator' stage
        Other uses include keeping track of the nature of the next stage
    step_info : dict
        Dictionary with info about this step of the solution process
    parameters_solver : dict
        Stores the parameters with which the solver was called
    completed_cycles : integer
        Number of cycles of the model that have been solved before this call
    solveMethod : str, optional
        The name of the solution method to use, e.g. 'EGM'
    """

# CDC 20210426: vPfunc was a bad choice for distance; here we change
# to cFunc but doing so will require recalibrating some of our tests
#    distance_criteria = ["vPfunc"]  # Bad b/c vP(0)=inf; should use cFunc
#    distance_criteria = ["vFunc.dm"]  # Bad b/c vP(0)=inf; should use cFunc
#    distance_criteria = ["mNrmTrg"]  # mNrmTrg a better choice if GICNrm holds
    distance_criteria = ["cFunc"]  # cFunc if the GIC fails
#    breakpoint()

    def __init__(self, *args,
                 # TODO: New items below should be in default ConsumerSolution
                 stge_kind={'iter_status': 'not initialized'},
                 completed_cycles=0,
                 parameters_solver=None,
                 vAdd=None,
                 **kwds):
        ConsumerSolutionOlder.__init__(self, **kwds)
        agent_solution.__init__(self, *args, **kwds)


class ConsumerSolutionOneStateCRRA(ConsumerSolution):
    """
    ConsumerSolution with CRRA utility and geometric discounting.

    Specializes the generic ConsumerSolution object to the case with:
        * Constant Relative Risk Aversion (CRRA) utility
        * Geometric Discounting of Time Separable Utility

    along with a standard set of restrictions on the parameter values of the
    model (like, the time preference factor must be positive).  Under various
    combinations of these assumptions, various conditions imply different
    conclusions.  The suite of minimal restrictions is always evaluated.
    Conditions are evaluated using the `check_conditions` method.  (Further
    information about the conditions can be found in the documentation for
    that method.)  For convenience, we repeat below the documentation for the
    parent ConsumerSolution of this class, all of which applies here.
    """

    __doc__ += ConsumerSolution.__doc__

    time_vary_ = ["LivPrb",  # Age-varying death rates can match mortality data
                  "PermGroFac"]  # Age-varying income growth to match lifecycle
    time_inv_ = ["CRRA", "Rfree", "DiscFac", "BoroCnstArt"]
    state_vars = ['pLvl',  # Initial idiosyncratic permanent income; for sims
                  'PlvlAgg',  # Aggregate permanent income; macro models
                  'bNrm',  # Bank balances beginning of period (pLvl normed)
                  'mNrm',  # Market resources (b + income - normed by pLvl
                  "aNrm"]  # Assets after all actions (pLvl normed)
    shock_vars_ = []

    def __init__(self, *args,
                 CRRA=2.0,
                 **kwds):

        ConsumerSolution.__init__(self, *args, **kwds)

        self.Pars.CRRA = CRRA

        # These have been moved to Bilt to declutter whiteboard:
        del self.hNrm
        del self.vPfunc
        del self.vPPfunc

    def check_conditions(self, soln_crnt, verbose=None):
        """
        Check whether the instance's type satisfies a set of conditions.

        ================================================================
        Acronym        Condition
        ================================================================
        AIC           Absolute Impatience Condition
        RIC           Return Impatience Condition
        GIC           Growth Impatience Condition
        GICLiv        GIC adjusting for constant probability of mortality
        GICNrm        GIC adjusted for uncertainty in permanent income
        FHWC          Finite Human Wealth Condition
        FVAC          Finite Value of Autarky Condition
        ================================================================

        Depending on the configuration of parameter values, some combination of
        these conditions must be satisfied in order for the problem to have
        a nondegenerate soln_crnt. To check which conditions are required,
        in the verbose mode, a reference to the relevant theoretical literature
        is made.

        Parameters
        ----------
        verbose : int

        Specifies different levels of verbosity of feedback. When False, it only reports whether the
        instance's type fails to satisfy a particular condition. When True, it reports all results, i.e.
        the factor values for all conditions.

        soln_crnt : ConsumerSolution
        Solution to the problem described by information
        for the current stage found in Bilt and the succeeding stage.

        Returns
        -------
        None
        """
        soln_crnt.Bilt.conditions = {}  # Keep track of truth of conditions
        soln_crnt.Bilt.degenerate = False  # True: solution is degenerate

        if not hasattr(self, 'verbose'):  # If verbose not set yet
            verbose = 0
        else:
            verbose = verbose if verbose is None else verbose

        msg = '\nFor a model with the following parameter values:\n'
        msg = msg+'\n'+str(soln_crnt.Bilt.parameters_solver)+'\n'

        if verbose >= 2:
            _log.info(msg)
            _log.info(str(soln_crnt.Bilt.parameters_solver))
            np.set_printoptions(threshold=20)  # Don't print huge output
            for key in soln_crnt.Bilt.parameters_solver.keys():
                print('\t'+key+': ', end='')
                pprint(soln_crnt.Bilt.parameters_solver[key])
            msg = '\nThe following results hold:\n'
            _log.info(msg)

        soln_crnt.check_AIC(soln_crnt, verbose)
        soln_crnt.check_FHWC(soln_crnt, verbose)
        soln_crnt.check_RIC(soln_crnt, verbose)
        soln_crnt.check_GICRaw(soln_crnt, verbose)
        soln_crnt.check_GICNrm(soln_crnt, verbose)
        soln_crnt.check_GICLiv(soln_crnt, verbose)
        soln_crnt.check_FVAC(soln_crnt, verbose)

        # degenerate flag is true if the model has no nondegenerate solution
        if hasattr(soln_crnt.Bilt, "BoroCnstArt") \
                and soln_crnt.Pars.BoroCnstArt is not None:
            soln_crnt.degenerate = not soln_crnt.Bilt.RIC
            # If BoroCnstArt exists but RIC fails, limiting soln is c(m)=0
        else:  # No constraint; not degenerate if neither c(m)=0 or \infty
            soln_crnt.degenerate = \
                not soln_crnt.Bilt.RIC or not soln_crnt.Bilt.FHWC

    def check_AIC(self, stge, verbose=None):
        """
        Evaluate and report on the Absolute Impatience Condition
        """
        name = "AIC"

        def test(stge): return stge.Bilt.APF < 1

        messages = {
            True: "\n\nThe Absolute Patience Factor for the supplied parameter values, APF={0.APF}, satisfies the Absolute Impatience Condition (AIC), which requires APF < 1:\n    "+stge.Bilt.AIC_fcts['urlhandle'],
            False: "\n\nThe Absolute Patience Factor for the supplied parameter values, APF={0.APF}, violates the Absolute Impatience Condition (AIC), which requires APF < 1:\n    "+stge.Bilt.AIC_fcts['urlhandle']
        }
        verbose_messages = {
            True: "\n  Because the APF < 1,  the absolute amount of consumption is expected to fall over time.  \n",
            False: "\n  Because the APF > 1, the absolute amount of consumption is expected to grow over time.  \n",
        }

        stge.Bilt.AIC = core_check_condition(name, test, messages, verbose,
                                             verbose_messages, "APF", stge)

    def check_FVAC(self, stge, verbose=None):
        """
        Evaluate and report on the Finite Value of Autarky Condition
        """
        name = "FVAC"
        def test(stge): return stge.Bilt.FVAF < 1

        messages = {
            True: "\n\nThe Finite Value of Autarky Factor for the supplied parameter values, FVAF={0.FVAF}, satisfies the Finite Value of Autarky Condition, which requires FVAF < 1:\n    "+stge.Bilt.FVAC_fcts['urlhandle'],
            False: "\n\nThe Finite Value of Autarky Factor for the supplied parameter values, FVAF={0.FVAF}, violates the Finite Value of Autarky Condition, which requires FVAF:\n    "+stge.Bilt.FVAC_fcts['urlhandle']
        }
        verbose_messages = {
            True: "\n  Therefore, a nondegenerate solution exists if the RIC also holds. ("+stge.Bilt.FVAC_fcts['urlhandle']+")\n",
            False: "\n  Therefore, a nondegenerate solution exits if the RIC holds, but will not exist if the RIC fails unless the FHWC also fails.\n",
        }

        stge.Bilt.FVAC = core_check_condition(name, test, messages, verbose,
                                              verbose_messages, "FVAF", stge)

    def check_GICRaw(self, stge, verbose=None):
        """
        Evaluate and report on the Growth Impatience Condition
        """
        name = "GICRaw"

        def test(stge): return stge.Bilt.GPFRaw < 1

        messages = {
            True: "\n\nThe Growth Patience Factor for the supplied parameter values, GPF={0.GPFRaw}, satisfies the Growth Impatience Condition (GIC), which requires GPF < 1:\n    "+stge.Bilt.GICRaw_fcts['urlhandle'],
            False: "\n\nThe Growth Patience Factor for the supplied parameter values, GPF={0.GPFRaw}, violates the Growth Impatience Condition (GIC), which requires GPF < 1:\n    "+stge.Bilt.GICRaw_fcts['urlhandle'],
        }
        verbose_messages = {
            True: "\n  Therefore,  for a perfect foresight consumer, the ratio of individual wealth to permanent income is expected to fall indefinitely.    \n",
            False: "\n  Therefore, for a perfect foresight consumer whose parameters satisfy the FHWC, the ratio of individual wealth to permanent income is expected to rise toward infinity. \n"
        }
        stge.Bilt.GICRaw = core_check_condition(name, test, messages, verbose,
                                                verbose_messages, "GPFRaw", stge)

    def check_GICLiv(self, stge, verbose=None):
        name = "GICLiv"

        def test(stge): return stge.Bilt.GPFLiv < 1

        messages = {
            True: "\n\nThe Mortality Adjusted Aggregate Growth Patience Factor for the supplied parameter values, GPFLiv={0.GPFLiv}, satisfies the Mortality Adjusted Aggregate Growth Impatience Condition (GICLiv):\n    "+stge.Bilt.GPFLiv_fcts['urlhandle'],
            False: "\n\nThe Mortality Adjusted Aggregate Growth Patience Factor for the supplied parameter values, GPFLiv={0.GPFLiv}, violates the Mortality Adjusted Aggregate Growth Impatience Condition (GICLiv):\n    "+stge.Bilt.GPFLiv_fcts['urlhandle'],
        }
        verbose_messages = {
            True: "\n  Therefore, a target level of the ratio of aggregate market resources to aggregate permanent income exists ("+stge.Bilt.GPFLiv_fcts['urlhandle']+")\n",
            False: "\n  Therefore, a target ratio of aggregate resources to aggregate permanent income may not exist ("+stge.Bilt.GPFLiv_fcts['urlhandle']+")\n",
        }
        stge.Bilt.GICLiv = core_check_condition(name, test, messages, verbose,
                                                verbose_messages, "GPFLiv", stge)

    def check_RIC(self, stge, verbose=None):
        """
        Evaluate and report on the Return Impatience Condition
        """

        name = "RIC"

        def test(stge): return stge.Bilt.RPF < 1

        messages = {
            True: "\n\nThe Return Patience Factor for the supplied parameter values, RPF= {0.RPF}, satisfies the Return Impatience Condition (RIC), which requires RPF < 1:\n    "+stge.Bilt.RPF_fcts['urlhandle'],
            False: "\n\nThe Return Patience Factor for the supplied parameter values, RPF= {0.RPF}, violates the Return Impatience Condition (RIC), which requires RPF < 1:\n    "+stge.Bilt.RPF_fcts['urlhandle'],
        }
        verbose_messages = {
            True: "\n  Therefore, the limiting consumption function is not c(m)=0 for all m\n",
            False: "\n  Therefore, if the FHWC is satisfied, the limiting consumption function is c(m)=0 for all m.\n",
        }
        stge.Bilt.RIC = core_check_condition(name, test, messages, verbose,
                                             verbose_messages, "RPF", stge)

    def check_FHWC(self, stge, verbose=None):
        """
        Evaluate and report on the Finite Human Wealth Condition
        """
        name = "FHWC"

        def test(stge): return stge.Bilt.FHWF < 1

        messages = {
            True: "\n\nThe Finite Human Wealth Factor value for the supplied parameter values, FHWF={0.FHWF}, satisfies the Finite Human Wealth Condition (FHWC), which requires FHWF < 1:\n    "+stge.Bilt.FHWC_fcts['urlhandle'],
            False: "\n\nThe Finite Human Wealth Factor value for the supplied parameter values, FHWF={0.FHWF}, violates the Finite Human Wealth Condition (FHWC), which requires FHWF < 1:\n    "+stge.Bilt.FHWC_fcts['urlhandle'],
        }
        verbose_messages = {
            True: "\n  Therefore, the limiting consumption function is not c(m)=Infinity.\n  Human wealth normalized by permanent income is {0.hNrmInf}.\n",
            False: "\n  Therefore, the limiting consumption function is c(m)=Infinity for all m unless the RIC is also violated.\n  If both FHWC and RIC fail and the consumer faces a liquidity constraint, the limiting consumption function is nondegenerate but has a limiting slope of 0. ("+stge.Bilt.FHWC_fcts['urlhandle']+")\n",
        }
        stge.Bilt.FHWC = core_check_condition(name, test, messages, verbose,
                                              verbose_messages, "FHWF", stge)

    def check_GICNrm(self, stge, verbose=None):
        """
        Check Normalized Growth Patience Factor.
        """
        if not hasattr(stge.Pars, 'IncShkDstn'):
            return  # GICNrm is same as GIC for PF consumer

        name = "GICNrm"

        def test(stge): return stge.Bilt.GPFNrm <= 1

        messages = {
            True: "\n\nThe Normalized Growth Patience Factor GPFNrm for the supplied parameter values, GPFNrm={0.GPFNrm}, satisfies the Normalized Growth Impatience Condition (GICNrm), which requires GPFNrm < 1:\n    "+stge.Bilt.GICNrm_fcts['urlhandle'],
            False: "\n\nThe Normalized Growth Patience Factor GPFNrm for the supplied parameter values, GPFNrm={0.GPFNrm}, violates the Normalized Growth Impatience Condition (GICNrm), which requires GPFNrm < 1:\n    "+stge.Bilt.GICNrm_fcts['urlhandle'],
        }
        verbose_messages = {
            True: "\n  Therefore, a target level of the individual market resources ratio m exists.",
            False: "\n  Therefore, a target ratio of individual market resources to individual permanent income does not exist.  ("+stge.Bilt.GICNrm_fcts['urlhandle']+")\n",
        }

        stge.Bilt.GICNrm = core_check_condition(name, test, messages, verbose,
                                                verbose_messages, "GPFNrm", stge)

    def check_WRIC(self, stge, verbose=None):
        """
        Evaluate and report on the Weak Return Impatience Condition
        [url]/  # WRIC modified to incorporate LivPrb
        """

        if not hasattr(stge, 'IncShkDstn'):
            return  # WRIC is same as RIC for PF consumer

        name = "WRIC"

        def test(stge): return stge.Bilt.WRPF <= 1

        messages = {
            True: "\n\nThe Weak Return Patience Factor value for the supplied parameter values, WRPF={0.WRPF}, satisfies the Weak Return Impatience Condition, which requires WRPF < 1:\n    "+stge.Bilt.WRIC_fcts['urlhandle'],
            False: "\n\nThe Weak Return Patience Factor value for the supplied parameter values, WRPF={0.WRPF}, violates the Weak Return Impatience Condition, which requires WRPF < 1:\n    "+stge.Bilt.WRIC_fcts['urlhandle'],
        }

        verbose_messages = {
            True: "\n  Therefore, a nondegenerate solution exists if the FVAC is also satisfied. ("+stge.Bilt.WRIC_fcts['urlhandle']+")\n",
            False: "\n  Therefore, a nondegenerate solution is not available ("+stge.Bilt.WRIC_fcts['urlhandle']+")\n",
        }
        stge.Bilt.WRIC = core_check_condition(name, test, messages, verbose,
                                              verbose_messages, "WRPF", stge)

    def mNrmTrg_find(self):
        """
        Find value of m at which individual consumer expects m not to change.

        This will exist if the GICNrm holds.

        https://econ-ark.github.io/BufferStockTheory#UniqueStablePoints

        Parameters
        ----------
        solution : ConsumerSolution
            Solution to this period's problem, which must have attribute cFunc.

        Returns
        -------
            The target value mNrmTrg.
        """
        m_init_guess = self.Bilt.mNrmMin + self.E_tp1_.IncNrmNxt
        try:  # Find value where argument is zero
            self.Bilt.mNrmTrg = find_zero_newton(
                self.E_tp1_.m_tp1_minus_m_t,
                m_init_guess)
        except:
            self.Bilt.mNrmTrg = None

        return self.Bilt.mNrmTrg

    def mNrmStE_find(self):
        """
        Find pseudo Steady-State Equilibrium (normalized) market resources m.

        This is the m at which the consumer
        expects level of market resources M to grow at same rate as the level
        of permanent income P.

        This will exist if the GIC holds.

        https://econ-ark.github.io/BufferStockTheory#UniqueStablePoints

        Parameters
        ----------
        self : ConsumerSolution
            Solution to this period's problem, which must have attribute cFunc.

        Returns
        -------
        self : ConsumerSolution
            Same solution that was passed, but now with attribute mNrmStE.
        """
        # Minimum market resources plus E[next income] is okay starting guess

        m_init_guess = self.Bilt.mNrmMin + self.E_tp1_.IncNrmNxt
        try:
            self.Bilt.mNrmStE = find_zero_newton(
                self.E_tp1_.permShk_tp1_times_m_tp1_minus_m_t, m_init_guess)
        except:
            self.Bilt.mNrmStE = None

        # Add mNrmStE to the solution and return it
        return self.Bilt.mNrmStE


class ConsPerfForesightSolver(ConsumerSolutionOneStateCRRA):
    """
    Solve one period perfect foresight CRRA utility consumption-saving problem.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one-period problem.
    DiscFac : float
        Intertemporal discount factor for future utility.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the next period.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period.
    BoroCnstArt : float or None
        Artificial borrowing constraint, as a multiple of permanent income.
        Can be None, indicating no artificial constraint.
    MaxKinks : int, optional
        Maximum number of kink points to allow in the consumption function;
        additional points will be thrown out.  Only relevant in infinite
        horizon model with artificial borrowing constraint.
    """

    # CDC 20200426: MaxKinks adds a lot of complexity to no necessary purpose
    # because everything it accomplishes could be done solving a finite horizon
    # model (including tests of convergence conditions, which can be invoked
    # manually if a user wants them).
    def __init__(
            self, solution_next, DiscFac=1.0, LivPrb=1.0, CRRA=2.0, Rfree=1.0,
            PermGroFac=1.0, BoroCnstArt=None, MaxKinks=None, **kwds):

        soln_futr = self.soln_futr = solution_next
        soln_crnt = self.soln_crnt = ConsumerSolutionOneStateCRRA()

        Pars = soln_crnt.Pars

        # Get solver parameters and store for later use
        # omitting things that could cause recursion
        Pars.__dict__.update(
            {k: v for k, v in {**kwds, **locals()}.items()
             if k not in {'self', 'solution_next', 'kwds', 'soln_futr',
                          'soln_crnt', 'Bilt', 'Pars', 'E_tp1_', 'Modl'}})

        # 'terminal' solution should replace pseudo_terminal:
        if hasattr(soln_futr.Bilt, 'stge_kind') and \
                (soln_futr.Bilt.stge_kind['iter_status'] == 'terminal_partial'):
            soln_crnt.Bilt = deepcopy(soln_futr.Bilt)

        # links for docs; urls are used when "fcts" are added
        self._url_doc_for_solver_get()

        return

    def _url_doc_for_solver_get(self):
        # Generate a url that will locate the documentation
        self.class_name = self.__class__.__name__
        self.soln_crnt.Bilt.url_ref = self.url_ref =\
            "https://econ-ark.github.io/BufferStockTheory"
        self.soln_crnt.Bilt.urlroot = self.urlroot = \
            self.url_ref+'/#'
        self.soln_crnt.Bilt.url_doc = self.url_doc = \
            "https://hark.readthedocs.io/en/latest/search.html?q=" +\
            self.class_name+"&check_keywords=yes&area=default#"

    # def cFunc_from_vFunc(self, m):
    #     #        ρ = self.soln_crnt.Pars.CRRA
    #     vFuncNvrs = self.vFuncNvrs
    #     vInv = vFuncNvrs(m)
    #     vInvP = vFuncNvrs.derivative(m)
    #     cP = self.cFunc.derivative(m)
    #     cVal = vInv ** (cP / vInvP)
    #     return cVal

    def make_cFunc_PF(self):
        """
        Make (piecewise linear) consumption function for this period.

        See PerfForesightConsumerType.ipynb notebook for derivations.
        """
        # Reduce cluttered formulae with local aliases
        crnt, tp1 = self.soln_crnt, self.soln_futr
        Bilt, Pars, E_tp1_ = crnt.Bilt, crnt.Pars, crnt.E_tp1_
        Rfree, PermGroFac, MPCmin = Pars.Rfree, Pars.PermGroFac, Bilt.MPCmin

        BoroCnstArt, DiscLiv, BoroCnstNat = \
            Pars.BoroCnstArt, Pars.DiscLiv, Bilt.BoroCnstNat

        u, u.Nvrs, u.dc.Nvrs = Bilt.u, Bilt.u.Nvrs, Bilt.u.dc.Nvrs
        CRRA, CRRA_tp1 = Pars.CRRA, tp1.Bilt.vFunc.CRRA

        yNrm_tp1 = tp1.Pars.tranShkMin  # for PF model tranShkMin = 1.0

        if BoroCnstArt is None:
            BoroCnstArt = -np.inf

        # Whichever constraint is tighter is the relevant one
        BoroCnst = max(BoroCnstArt, BoroCnstNat)

        # Omit first and last points which define extrapolation below and above
        # the kink points
        mNrm_kinks_tp1 = tp1.cFunc.x_list[:-1][1:]
        cNrm_kinks_tp1 = tp1.cFunc.y_list[:-1][1:]
        vNrm_kinks_tp1 = tp1.vFunc(mNrm_kinks_tp1)

        # Calculate end-of-this-period aNrm vals that would reach those mNrm's
        # There are no shocks in the PF model, so tranShkMin = tranShk = 1.0
        bNrm_kinks_tp1 = (mNrm_kinks_tp1 - yNrm_tp1)
        kNrm_kinks_tp1 = aNrm_kinks = bNrm_kinks_tp1*(PermGroFac/Rfree)

        # Obtain c_t from which unconstrained consumers would land on each
        # kink next period by inverting FOC: c^#_t = (RβΠ)^(-1/ρ) c^#_tp1
        # This is the endogenous gridpoint (kink point number #) today
        # corresponding to each next-period kink (each of which corresponds
        # to a finite-horizon solution ending one more period in the future)

        cNrm_kinks = (((Rfree * DiscLiv) ** (-1/CRRA_tp1)) *
                      PermGroFac * cNrm_kinks_tp1)
        cNrm_kinks_EGM = tp1.Bilt.u.dc.Nvrs(E_tp1_.given_shocks[E_tp1_.v1_pos])

        vNrm_kinks = (DiscLiv * PermGroFac**(1-tp1.Pars.CRRA))*vNrm_kinks_tp1
        vNrm_kinks_EGM = E_tp1_.given_shocks[E_tp1_.v0_pos]

        mNrm_kinks = aNrm_kinks + cNrm_kinks
        mNrm_kinks_EGM = Bilt.aNrmGrid + cNrm_kinks_EGM

        vInv_kinks = u.Nvrs(vNrm_kinks)
        vInv_kinks_EGM = u.Nvrs(vNrm_kinks_EGM)

        vAdd_kinks = mNrm_kinks-mNrm_kinks

        # tranShkMin = tranShkMax = 1.0 for PF model
        mNrmMin_tp1 = \
            tp1.Pars.tranShkMin + BoroCnst * (Rfree/PermGroFac)

        t_E_v_tp1_at_BoroCnst = \
            (DiscLiv * PermGroFac**(1-CRRA_tp1) *
             tp1.vFunc(mNrmMin_tp1))

        t_E_vP_tp1_at_BoroCnst = \
            ((Rfree * DiscLiv) * PermGroFac**(-CRRA_tp1) *
             tp1.vFunc.dm(mNrmMin_tp1))


        # h is the 'horizon': h_t(m_t) is the number of periods it will take
        # before you hit the constraint, after which you remain constrained

        # For any c_t where you are unconstrained today, value is discounted
        # sum of values you will receive during periods between now and t+h,
        # and values you will receive afer h
#        vAdd = # Sum of post-constrained value by gridpoint
#            (DiscLiv * PermGroFac**(1-CRRA))*\
#                (Bilt.u(folw.cFunc_tp1(mNrm_kinks_tp1) # u at next period cusp
#                        +vAdd_tp1) # v from s

        # cusp is point where current period constraint stops binding
        cNrm_cusp = u.dc.Nvrs(t_E_vP_tp1_at_BoroCnst)
        vNrm_cusp = Bilt.u(cNrm_cusp)+t_E_vP_tp1_at_BoroCnst
        vAdd_cusp = t_E_v_tp1_at_BoroCnst
        vInv_cusp = u.Nvrs(vNrm_cusp)
        mNrm_cusp = cNrm_cusp + BoroCnst

        # cusp today vs today's implications of future constraints
        if mNrm_cusp >= mNrm_kinks[-1]:  # tighter than the tightest existing
            mNrm_kinks = np.array(mNrm_cusp)  # looser ones are irrelevant
            cNrm_kinks = np.array(cNrm_cusp)
            vNrm_kinks = np.array(vNrm_cusp)
            vInv_kinks = np.array(vInv_cusp)
            vAdd_kinks = np.array(vAdd_cusp)
        else:
            first_reachable = np.where(mNrm_kinks >= mNrm_cusp)[0][-1]
            if first_reachable < mNrm_kinks.size - 1:  # Keep reachable pts
                mNrm_kinks = mNrm_kinks[first_reachable:-1]
                cNrm_kinks = cNrm_kinks[first_reachable:-1]
                vInv_kinks = vInv_kinks[first_reachable:-1]
                vAdd_kinks = vAdd_kinks[first_reachable:-1]
            mNrm_kinks = np.insert(mNrm_kinks, 0, mNrm_cusp)
            cNrm_kinks = np.insert(cNrm_kinks, 0, cNrm_cusp)
            vNrm_kinks = np.insert(vNrm_kinks, 0, vNrm_cusp)

        vAddGrid = np.append(vAdd_cusp, vAdd_kinks)
        vAddGrid = np.append(vAddGrid, 0.)

        # To guarantee meeting BoroCnst, if mNrm = BoroCnst then cNrm = 0.
        mNrmGrid_unconst = np.append(mNrm_kinks, mNrm_kinks+1)
        cNrmGrid_unconst = np.append(cNrm_kinks, cNrm_kinks+MPCmin)
        aNrmGrid_unconst = mNrmGrid_unconst-cNrmGrid_unconst
        mNrmGrid_tp1_unconst = aNrmGrid_unconst*(Rfree/PermGroFac)+yNrm_tp1
        vNrmGrid_unconst = u(cNrmGrid_unconst) + \
            (DiscLiv * PermGroFac**(1-CRRA_tp1) *
             tp1.vFunc(mNrmGrid_tp1_unconst))
        vInvGrid_unconst = u.Nvrs(vNrmGrid_unconst)
        vInvPGrid_unconst = \
            (((1-CRRA)*vNrmGrid_unconst)**(-1+1/(1-CRRA))) * \
            (cNrmGrid_unconst**(-CRRA))
        c_from_vInvPGrid_unconst = \
            ((vInvPGrid_unconst/(((1-CRRA)*vNrmGrid_unconst) **
                                 (-1+1/(1-CRRA)))))**(-1/CRRA)

        mNrmGrid_const = np.array([BoroCnst, mNrm_cusp, mNrm_cusp+1])
        uNrmGrid_const = np.array([float('inf'), u(mNrm_cusp), float('inf')])
        uInvGrid_const = u.Nvrs(uNrmGrid_const)

        def vAddFunc(m, mNrmGrid, vAddGrid):
            mNrmGridPlus = np.append(mNrmGrid, float('inf'))
            vAddGridPlus = np.append(vAddGrid, vAddGrid[-1])
            from collections import Iterable
            if isinstance(m, Iterable):
                from itertools import repeat
                return np.array(list(map(lambda m, mNrmGridPlus, vAddGridPlus:
                                         vAddGridPlus[np.where(m < mNrmGridPlus)[0][0]], m, repeat(mNrmGridPlus), repeat(vAddGridPlus))))
            else:
                return vAddGridPlus[np.where(m < mNrmGridPlus)[0][0]]

#        mPts = np.linspace(mNrmGrid[0],mNrmGrid[-1],10)

        vInvFunc_unconst = \
            LinearInterp(mNrmGrid_unconst, vInvGrid_unconst)

#        from HARK.utilities import plot_funcs
#        plot_funcs(lambda x: np.heaviside(x-BoroCnst,0.5),1,2)
        uInvFunc_const = \
            LinearInterp(mNrmGrid_const, uInvGrid_const)
        vFunc_const = Bilt.u(uInvGrid_const)+t_E_v_tp1_at_BoroCnst
        vFunc_unconst = Bilt.u(vInvGrid_unconst)

        def vAddFunc(m, mGrid, vAddGrid):
            return vAddGrid[np.where(m < mGrid)[0][0]]

#        vNrmGrid_const=[BoroCnst,u(mNrmGrid_unconst[0])]

        mNrmGrid = np.append([BoroCnst], mNrmGrid_unconst)
        cNrmGrid = np.append(0., cNrmGrid_unconst)
        vInvGrid = np.append(0., vInvGrid_unconst)
#        vInvPGrid = np.append(float('inf'), vInvPGrid_unconst)
#        vInvGrid = np.insert(vInvGrid, 0, -1)

        # Above last kink point, use PF solution
#        mNrmGrid = np.append(mNrmGrid, mNrmGrid[-1]+1)
#        cNrmGrid = np.append(cNrmGrid, cNrmGrid[-1]+MPCmin)
#        aNrmGrid = mNrmGrid - cNrmGrid
#        bNrmGrid_tp1 = aNrmGrid*(Rfree/PermGroFac)
#        mNrmGrid_tp1 = bNrmGrid_tp1+folw.PF_IncNrm_tp1
#        vNrmGrid = u(cNrmGrid)+(DiscLiv * PermGroFac**(1-CRRA_tp1) *
#             folw.vFunc_tp1(mNrmGrid_tp1))
#        vInvGrid = (vNrmGrid*(1-CRRA))**(1/(1-CRRA))

#        vInvGrid = np.append(vInvGrid, vInvGrid[-1]+MPCmin**(-CRRA/(1.0-CRRA)))

        # To guarantee meeting BoroCnst, if mNrm = BoroCnst then cNrm = 0.
        mNrmGrid = np.append([BoroCnst], mNrm_kinks)
        cNrmGrid = np.append(0., cNrm_kinks)

        # Above last kink point, use PF solution
        mNrmGrid = np.append(mNrmGrid, mNrmGrid[-1]+1)
        cNrmGrid = np.append(cNrmGrid, cNrmGrid[-1]+MPCmin)

        self.cFunc = self.soln_crnt.cFunc = Bilt.cFunc = \
            LinearInterp(mNrmGrid, cNrmGrid)

#        breakpoint()

#        vInvFunc_unconst = self.vFuncNvrs = \
#            LinearInterp(mNrmGrid,vInvGrid)

        # self.vNvrsFunc.derivative(m)
        # (vNvrsFunc.derivative(/(((1-CRRA)*vInvGrid_unconst)**(-1+1/(1-CRRA))))

#        cc = self.cFunc_from_vFunc(2.0)
#        cFromvP=uPinv(vP_Grid)
#        cFuncFromvNvrsFunc = LinearInterp(mNrmGrid,cFromvP)
#        from HARK.utilities import plot_funcs
#        plot_funcs([self.cFunc,cFuncFromvNvrsFunc],0,2)

#        print('hi')
#        PF_t_v_tp1_last = (DiscLiv*(PermGroFac ** (1-CRRA_tp1)))*\
#            np.float(folw.vFunc_tp1((Rfree/PermGroFac)*aNrmGrid[-1]+E_tp1_.IncNrmNxt))
#        PF_t_vNvrs_tp1_Grid_2 = \
#            np.append(PF_t_vNvrs_tp1_Grid,PF_t_v_tp1_last)

        # vNvrsGrid = Bilt.uinv(Bilt.u(cNrmGrid)+ folw.u_tp1(PF_t_vNvrs_tp1_Grid))

        # If the mNrm that would unconstrainedly yield next period's bottom pt
#        if BoroCnst > mNrmGrid_pts[0]: # is prohibited by BoroCnst
#            satisfies_BoroCnst = np.where(
#                mNrmGrid_unconst - BoroCnst < cNrm_from_aNrmMin) # True if OK

        # Amount of m in excess of minimum possible m
#        mNrmXtraGrid = mNrmGrid_pts - BoroCnst

        # We think of the stage at which income has been realized
        # as prior to the start of the period when the problem is solved.
        # You "enter" the period with normalized market resources mNrm
        # If this is starting period of life, we want problem to be defined all
        # the way down to the point where c would be forced to be zero
        # That requires us to find the min aNrm with which the period can end
#        cNrmMin = 0. # You cannot spend anything and still satisfy constraint

        # Find first kink point in existing grid
#        kink_min = np.where(mNrmXtraGrid <= cNrmGrid_pts)[0][-1]
#        kink_min = np.where(aNrmGrid_pts <= BoroCnst)[0][-1]

        # Now calculate the minimum mNrm that will be possible for a consumer
        # who lived in the prior period and in that period satisfied the
        # relevant borrowing constraint
#        _bNrmMin = aNrmMin*(Rfree/PermGroFac)
#        mNrmMin = _bNrmMin + BoroCnst
#        mNrmMin_with_income = mNrmMin + folw.PF_IncNrmNxt
 #       if c_at_aNrmMin > aNrmMin+folw.PF_IncNrmNxt-BoroCnst:
        # The u' from ending this period with aNrm = BoroCnst exceeds
        # u' from the last penny of spending, so the consumer
        # at

#            mNrmGrid_pts = np.insert(mNrmGrid_pts,kink_min,mNrmMin_with_income)
#            cNrmGrid_pts = np.insert(cNrmGrid_pts,kink_min,mNrmMin_with_income)

        # Last index where constraint binds (in current period)
#        kink_min = np.where(mNrmXtraGrid <= cNrmGrid)[0][-1]
#        mNrmGrid_pts = np.insert(mNrmGrid_pts,kink_min,mNrmMin)
#        cNrmGrid_pts = np.insert(cNrmGrid_pts,kink_min,cNrmMin)

        # The minimum m if you were at the binding borrowing constraint

        # at the end of the last period (assumes prior period's )
#         mNrmMin_BoroCnstLast = BoroCnst*(Rfree/PermGroFac)

# #        mNrmGrid_pts = np.insert(mNrmGrid,kink_min,bNrmMin+PF_IncNrmNow)
# #        cNrmGrid_pts = np.insert(cNrmGrid,kink_min,bNrmMin+PF_IncNrmNow)
#         kink_max = np.where(mNrmXtraGrid >= cNrmGrid)[0][-1]

#         # Below binding point, c is maximum allowed, which is attained by line
#         # from [BoroCnst,0.] to [mNrmGrid[cusp],mNrmGrid[cusp]]
#         cNrmGrid_pts = np.insert(cNrmGrid,kink_min,)

#         mNrmGrid_pts = np.insert(mNrmGrid,kink_min-1,E_tp1_.IncNrmNxt)
#         cNrmGrid_pts = np.insert(cNrmGrid,kink_min-1,E_tp1_.IncNrmNxt-BoroCnst)

#        mNrmGrid_pts = np.append(mNrmGrid_pts,mNrmGrid_pts[-1]+1.)
#        cNrmGrid_pts = np.append(cNrmGrid_pts,cNrmGrid_pts[-1]+MPCmin)


#         mNrmGrid = np.insert(mNrmGrid,0,mNrmMin)
#         cNrmGrid = np.insert(cNrmGrid,0,0.)


#         if BoroCnstArt+E_tp1_.IncNrmNxt > mNrmGrid[0]:
#             mNrmGrid

#         mNrmGrid = np.append(mNrmGrid,mNrmGrid[-1]+1.0)
#         cNrmGrid = np.append(cNrmGrid,cNrmGrid[-1]+MPCmin)


#         # Add the point corresponding to
#         mNrmGrid = np.unique(np.insert(mNrmGrid,0,E_tp1_.IncNrmNxt-BoroCnstArt))
#         cNrmGrid = np.unique(np.insert(cNrmGrid,0,E_tp1_.IncNrmNxt-BoroCnstArt))


# #        vNvrs_tp1 = (DiscLiv * LivPrb) * folw.vFunc_tp1(mNrmGrid_tp1)
# #        PF_t_vNvrs_tp1_Grid = folw.uinv_tp1(DiscLiv) * \
# #            folw.vFuncNvrs_tp1.y_list
#         # Beginning-of-period-tp1 marginal value vec is vP_tp1
# #        vP_tp1 = folw.uP_tp1(cNrmGrid_tp1)
#         # Corresponding end-of-period-t marginal value is _vP_t
# #        _vP_t = ((DiscLiv * Rfree) * (PermGroFac**(-CRRA_tp1)))*vP_tp1
# #        _vP_t =
#         # Endogenous gridpoints method
#  #       cNrmGrid = Bilt.uPinv(_vP_t)    # EGM step 1: u' inverse yields c
#         mNrmGrid = aNrmGrid + cNrmGrid  # EGM step 2: DBC inverted

#         cNrmGrid = np.unique(np.insert(cNrmGrid,0,E_tp1_.IncNrmNxt-BoroCnstArt))

#         # Add additional point to the list of gridpoints for extrapolation,
#         # using this period's new value of the lower bound of the MPC, which
#         # defines the PF unconstrained problem through the end of the horizon
# #        mNrmGrid_interp_pts = np.append(mNrmGrid, mNrmGrid[-1] + 1.0)
# #        cNrmGrid_interp_pts = np.append(cNrmGrid, cNrmGrid[-1] + MPCmin)
#         # If artificial borrowing constraint binds, combine constrained and
#         # unconstrained consumption functions.

#         # The problem is well-defined down to BoroCnstArt even if in
#         # principle from t you could not get to any m_tp1 < E_tp1_.IncNrmNxt
#         # because nothing prevents you from starting tp1 with m \geq BoroCnstArt
#  #       if BoroCnstArt < mNrmGrid[0] - E_tp1_.IncNrmNxt:
#  #           mNrmGrid_interp_pts = np.append([BoroCnstArt], mNrmGrid_interp_pts)
#  #           cNrmGrid_interp_pts = np.append([BoroCnstArt], cNrmGrid_interp_pts)
# #        else: # BoroCnstArt is irrelevant if BoroCnstNat is tighter
#             # cNrmGridCnst defines points where cnst would bind for each m gpt
# #        cNrmGridCnst = mNrmGrid_interp_pts - BoroCnstArt
#         mXtraGrid = mNrmGrid - BoroCnstArt # m in excess of minimum m
# #        idx_binding =   # c > possible
#         # highest m where cnst binds
#         idx_binding_last = np.where(mNrmXtraGrid <= cNrmGrid)[0][-1]
#         if idx_binding_last < (mNrmGrid.size - 1):
#             print('hi')
#             # # If not the *very last* index, find the the critical level
#             # # of mNrmGrid where artificial cnst begins to bind.
#             # d0 = cNrmGrid[idx] - cNrmGridCnst[idx]
#             # d1 = cNrmGridCnst[idx + 1] - cNrmGrid[idx + 1]
#             # m0 = mNrmGrid[idx]
#             # m1 = mNrmGrid[idx + 1]
#             # alpha = d0 / (d0 + d1)
#             # mCrit = m0 + alpha * (m1 - m0)
#             # # Adjust grids of mNrmGrid and cNrmGrid to account for constraint.
#             # cCrit = mCrit - BoroCnstArt
#             # mNrmGrid = np.concatenate(([BoroCnstArt, mCrit], mNrmGrid[(idx + 1):]))
#             # cNrmGrid = np.concatenate(([0.0, cCrit], cNrmGrid[(idx + 1):]))
#             # aNrmGrid = mNrmGrid-cNrmGrid
#         else:
#             # If it *is* the last index, then there are only three points
#             # that characterize the c function: the artificial borrowing
#             # constraint, the constraint kink, and the extrapolation point
#             mAbve= 1.0 # (cNrmGrid[-1] - cNrmGridCnst[-1]) / (1.0 - MPCmin)
#             mNrm_max_bindpoint = mNrmGrid[idx_binding_last]
#             cNrm_max_bindpoint = cNrmGrid[idx_binding_last]
# #                mCrit = mNrmGrid[-1] + mXtra
# #                cCrit = mCrit - BoroCnstArt
#             mNrmGrid_Xtra = np.array(
#                 [BoroCnstArt, mNrm_max_bindpoint, mNrm_max_bindpoint + 1])
#             cNrmGrid_Xtra = np.array(
#                 [0.0, cNrm_max_bindpoint, cNrm_max_bindpoint + MPCmin])
#             aNrmGrid_Xtra = mNrmGrid_Xtra- cNrmGrid_Xtra
#             # If mNrmGrid, cNrmGrid grids have become too large, throw out last
#             # kink point, being sure to adjust the extrapolation.

#         if mNrmGrid.size > MaxKinks:
#             mNrmGrid = np.concatenate((mNrmGrid[:-2], [cNrmGrid[-3] + 1.0]))
#             cNrmGrid = np.concatenate((cNrmGrid[:-2], [cNrmGrid[-3] + MPCmin]))
#             aNrmGrid = mNrmGrid - cNrmGrid

        # Consumption function is a linear interpolation between kink pts
#        self.cFunc = self.soln_crnt.cFunc = Bilt.cFunc = \
#            LinearInterp(mNrmGrid_pts, cNrmGrid_pts)


#        PF_t_v_tp1_last = (DiscLiv*(PermGroFac ** (1-CRRA_tp1)))*\
#            np.float(folw.vFunc_tp1((Rfree/PermGroFac)*aNrmGrid[-1]+E_tp1_.IncNrmNxt))
#        PF_t_vNvrs_tp1_Grid_2 = \
#            np.append(PF_t_vNvrs_tp1_Grid,PF_t_v_tp1_last)

        # vNvrsGrid = Bilt.uinv(Bilt.u(cNrmGrid)+ folw.u_tp1(PF_t_vNvrs_tp1_Grid))

        # # Calculate the upper bound of the MPC as the slope of bottom segment
        # # In practice, this is always 1.  Code is here for clarity
        # Bilt.MPCmax = ((cNrmGrid_Xtra[1] - cNrmGrid_Xtra[0])/
        #                (mNrmGrid_Xtra[1] - mNrmGrid_Xtra[0]))

        # # Lower bound of mNrm is lowest gridpoint -- usually 0
        # Bilt.mNrmMin = mNrmGrid_Xtra[0]

        # # Add the calculated grids to self.Bilt
        # Bilt.aNrmGrid = aNrmGrid_Xtra
        # Bilt._vP_t = _vP_t
        # Bilt.cNrmGrid = cNrmGrid_Xtra
        # Bilt.mNrmGrid = mNrmGrid_Xtra

        # Add approximation to v and vP
#        breakpoint()
#        Bilt.vNvrs = self.soln_crnt.uinv(_vP_t)

    def from_chosen_states_make_E_tp1_(self, crnt):
        """
        Construct expectations of useful objects from post-choice stage.

        Parameters
        ----------
        crnt : ConsumerSolution
            The solution to the problem without the expectations info.

        Returns
        -------
        crnt : ConsumerSolution
            The given solution, with the relevant namespaces updated to
        contain the constructed info.
        """
        crnt = self.build_facts_infhor()
        crnt = self.build_facts_recursive()
        

        # Reduce cluttered formulae with local aliases
        E_tp1_ = crnt.E_tp1_
        tp1 = self.soln_futr
        Bilt, Pars = crnt.Bilt, crnt.Pars
        Rfree, PermGroFac, DiscLiv = Pars.Rfree, Pars.PermGroFac, Pars.DiscLiv
        CRRA = tp1.vFunc.CRRA

        BoroCnstArt, BoroCnstNat = \
            Pars.BoroCnstArt, Bilt.BoroCnstNat

        if BoroCnstArt is None:
            BoroCnstArt = -np.inf

        # Whichever constraint is tighter is the relevant one
        BoroCnst = max(BoroCnstArt, BoroCnstNat)

        # Omit first and last points which define extrapolation below and above
        # the kink points
        mNrm_kinks_tp1 = tp1.cFunc.x_list[:-1][1:]
        cNrm_kinks_tp1 = tp1.cFunc.y_list[:-1][1:]
        vNrm_kinks_tp1 = tp1.vFunc(mNrm_kinks_tp1)

        # Calculate end-of-this-period aNrm vals that would reach those mNrm's
        # There are no shocks in the PF model, so tranShkMin = tranShk = 1.0
        bNrm_kinks_tp1 = (mNrm_kinks_tp1 - tp1.Pars.tranShkMin)
        aNrm_kinks = bNrm_kinks_tp1*(PermGroFac/Rfree)

        crnt.Bilt.aNrmGrid = aNrm_kinks
        # Level and first derivative of expected value from aNrmGrid points
        v_0 = DiscLiv * \
            PermGroFac ** (1-CRRA) * vNrm_kinks_tp1
        v_1 = DiscLiv * \
            PermGroFac ** (0-CRRA) * tp1.Bilt.u.dc(cNrm_kinks_tp1) * Rfree
        c_0 = cNrm_kinks_tp1

        E_tp1_.given_shocks = np.array([v_0, v_1, c_0])
        E_tp1_.v0_pos, E_tp1_.v1_pos = 0, 1
        E_tp1_.c0_pos = 3

        return crnt

    def build_facts_infhor(self):
        """
        Calculate facts useful for characterizing infinite horizon models.

        Parameters
        ----------
        solution: ConsumerSolution
            Solution that already has minimal requirements (vPfunc, cFunc)

        Returns
        -------
        solution : ConsumerSolution
            The given solution, with the relevant namespaces updated to
        contain the constructed info.
        """
        # Using local variables makes formulae more readable
        soln_crnt = self.soln_crnt  # current
        Bilt, Pars, E_tp1_ = soln_crnt.Bilt, soln_crnt.Pars, soln_crnt.E_tp1_

        urlroot = Bilt.urlroot
        Pars.DiscLiv = Pars.DiscFac * Pars.LivPrb
        # givens are not changed by the calculations below; Bilt and E_tp1_ are
        givens = {**Pars.__dict__}
#        breakpoint()

        APF_fcts = {
            'about': 'Absolute Patience Factor'
        }
        py___code = '((Rfree * DiscLiv) ** (1.0 / CRRA))'
        Bilt.APF = APF = \
            eval(py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        APF_fcts.update({'latexexpr': r'\APF'})
        APF_fcts.update({'_unicode_': r'Þ'})
        APF_fcts.update({'urlhandle': urlroot+'APF'})
        APF_fcts.update({'py___code': py___code})
        APF_fcts.update({'value_now': APF})
        Bilt.APF_fcts = APF_fcts

        AIC_fcts = {
            'about': 'Absolute Impatience Condition'
        }
        AIC_fcts.update({'latexexpr': r'\AIC'})
        AIC_fcts.update({'urlhandle': urlroot+'AIC'})
        AIC_fcts.update({'py___code': 'test: APF < 1'})
        Bilt.AIC_fcts = AIC_fcts

        RPF_fcts = {
            'about': 'Return Patience Factor'
        }
        py___code = 'APF / Rfree'
        Bilt.RPF = RPF = \
            eval(py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        RPF_fcts.update({'latexexpr': r'\RPF'})
        RPF_fcts.update({'_unicode_': r'Þ_R'})
        RPF_fcts.update({'urlhandle': urlroot+'RPF'})
        RPF_fcts.update({'py___code': py___code})
        RPF_fcts.update({'value_now': RPF})
        Bilt.RPF_fcts = RPF_fcts

        RIC_fcts = {
            'about': 'Growth Impatience Condition'
        }
        RIC_fcts.update({'latexexpr': r'\RIC'})
        RIC_fcts.update({'urlhandle': urlroot+'RIC'})
        RIC_fcts.update({'py___code': 'test: RPF < 1'})
        Bilt.RIC_fcts = RIC_fcts

        GPFRaw_fcts = {
            'about': 'Growth Patience Factor'
        }
        py___code = 'APF / PermGroFac'
        Bilt.GPFRaw = GPFRaw = \
            eval(py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        GPFRaw_fcts.update({'latexexpr': r'\GPFRaw'})
        GPFRaw_fcts.update({'_unicode_': r'Þ_Γ'})
        GPFRaw_fcts.update({'urlhandle': urlroot+'GPFRaw'})
        GPFRaw_fcts.update({'py___code': py___code})
        GPFRaw_fcts.update({'value_now': GPFRaw})
        Bilt.GPFRaw_fcts = GPFRaw_fcts

        GICRaw_fcts = {
            'about': 'Growth Impatience Condition'
        }
        GICRaw_fcts.update({'latexexpr': r'\GICRaw'})
        GICRaw_fcts.update({'urlhandle': urlroot+'GICRaw'})
        GICRaw_fcts.update({'py___code': 'test: GPFRaw < 1'})
        Bilt.GICRaw_fcts = GICRaw_fcts

        GPFLiv_fcts = {
            'about': 'Mortality-Adjusted Growth Patience Factor'
        }
        py___code = 'APF * LivPrb / PermGroFac'
        Bilt.GPFLiv = GPFLiv = \
            eval(py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        GPFLiv_fcts.update({'latexexpr': r'\GPFLiv'})
        GPFLiv_fcts.update({'urlhandle': urlroot+'GPFLiv'})
        GPFLiv_fcts.update({'py___code': py___code})
        GPFLiv_fcts.update({'value_now': GPFLiv})
        Bilt.GPFLiv_fcts = GPFLiv_fcts

        GICLiv_fcts = {
            'about': 'Growth Impatience Condition'
        }
        GICLiv_fcts.update({'latexexpr': r'\GICLiv'})
        GICLiv_fcts.update({'urlhandle': urlroot+'GICLiv'})
        GICLiv_fcts.update({'py___code': 'test: GPFLiv < 1'})
        Bilt.GICLiv_fcts = GICLiv_fcts

        RNrm_PF_fcts = {
            'about': 'Growth-Normalized PF Return Factor'
        }
        py___code = 'Rfree/PermGroFac'
        E_tp1_.RNrm_PF = RNrm_PF = \
            eval(py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        RNrm_PF_fcts.update({'latexexpr': r'\PFRNrm'})
        RNrm_PF_fcts.update({'_unicode_': r'R/Γ'})
        RNrm_PF_fcts.update({'py___code': py___code})
        RNrm_PF_fcts.update({'value_now': RNrm_PF})
        E_tp1_.RNrm_PF_fcts = RNrm_PF_fcts

        Inv_RNrm_PF_fcts = {
            'about': 'Inv of Growth-Normalized PF Return Factor'
        }
        py___code = '1 / RNrm_PF'
        E_tp1_.Inv_RNrm_PF = Inv_RNrm_PF = \
            eval(py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        Inv_RNrm_PF_fcts.update({'latexexpr': r'\InvPFRNrm'})
        Inv_RNrm_PF_fcts.update({'_unicode_': r'Γ/R'})
        Inv_RNrm_PF_fcts.update({'py___code': py___code})
        Inv_RNrm_PF_fcts.update({'value_now': Inv_RNrm_PF})
        E_tp1_.Inv_RNrm_PF_fcts = \
            Inv_RNrm_PF_fcts

        FHWF_fcts = {
            'about': 'Finite Human Wealth Factor'
        }
        py___code = 'PermGroFac / Rfree'
        Bilt.FHWF = FHWF = \
            eval(py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        FHWF_fcts.update({'latexexpr': r'\FHWF'})
        FHWF_fcts.update({'_unicode_': r'R/Γ'})
        FHWF_fcts.update({'urlhandle': urlroot+'FHWF'})
        FHWF_fcts.update({'py___code': py___code})
        FHWF_fcts.update({'value_now': FHWF})
        Bilt.FHWF_fcts = \
            FHWF_fcts

        FHWC_fcts = {
            'about': 'Finite Human Wealth Condition'
        }
        FHWC_fcts.update({'latexexpr': r'\FHWC'})
        FHWC_fcts.update({'urlhandle': urlroot+'FHWC'})
        FHWC_fcts.update({'py___code': 'test: FHWF < 1'})
        Bilt.FHWC_fcts = FHWC_fcts

        hNrmInf_fcts = {
            'about': 'Human wealth for inf hor'
        }
        py___code = '1/(1-FHWF) if (FHWF < 1) else float("inf")'
        Bilt.hNrmInf = hNrmInf = \
            eval(py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        hNrmInf_fcts = dict({'latexexpr': r'1/(1-\FHWF)'})
        hNrmInf_fcts.update({'value_now': hNrmInf})
        hNrmInf_fcts.update({'py___code': py___code})
        Bilt.hNrmInf_fcts = hNrmInf_fcts

        DiscGPFRawCusp_fcts = {
            'about': 'DiscFac s.t. GPFRaw = 1'
        }
        py___code = '( PermGroFac                       **CRRA)/(Rfree)'
        Bilt.DiscGPFRawCusp = DiscGPFRawCusp = \
            eval(py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        DiscGPFRawCusp_fcts.update({'latexexpr':
                                    r'\PermGroFac^{\CRRA}/\Rfree'})
        DiscGPFRawCusp_fcts.update({'value_now': DiscGPFRawCusp})
        DiscGPFRawCusp_fcts.update({'py___code': py___code})
        Bilt.DiscGPFRawCusp_fcts = \
            DiscGPFRawCusp_fcts

        DiscGPFLivCusp_fcts = {
            'about': 'DiscFac s.t. GPFLiv = 1'
        }
        py___code = '( PermGroFac                       **CRRA)/(Rfree*LivPrb)'
        Bilt.DiscGPFLivCusp = DiscGPFLivCusp = \
            eval(py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        DiscGPFLivCusp_fcts.update({'latexexpr':
                                    r'\PermGroFac^{\CRRA}/(\Rfree\LivPrb)'})
        DiscGPFLivCusp_fcts.update({'value_now': DiscGPFLivCusp})
        DiscGPFLivCusp_fcts.update({'py___code': py___code})
        Bilt.DiscGPFLivCusp_fcts = DiscGPFLivCusp_fcts

        FVAF_fcts = {  # overwritten by version with uncertainty
            'about': 'Finite Value of Autarky Factor'
        }
        py___code = 'LivPrb * DiscLiv'
        Bilt.FVAF = \
            eval(py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        FVAF_fcts.update({'latexexpr': r'\FVAFPF'})
        FVAF_fcts.update({'urlhandle': urlroot+'FVAFPF'})
        FVAF_fcts.update({'py___code': py___code})
        Bilt.FVAF_fcts = FVAF_fcts

        FVAC_fcts = {  # overwritten by version with uncertainty
            'about': 'Finite Value of Autarky Condition - Perfect Foresight'
        }
        FVAC_fcts.update({'latexexpr': r'\FVACPF'})
        FVAC_fcts.update({'urlhandle': urlroot+'FVACPF'})
        FVAC_fcts.update({'py___code': 'test: FVAFPF < 1'})
        Bilt.FVAC_fcts = FVAC_fcts

        E_tp1_.IncNrmNxt_fcts = {  # Overwritten by version with uncertainty
            'about': 'Expected income next period'
        }
        py___code = '1.0'
        E_tp1_.IncNrmNxt = \
            eval(py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
#        E_tp1_.IncNrmNxt_fcts.update({'latexexpr': r'ExIncNrmNxt'})
#        E_tp1_.IncNrmNxt_fcts.update({'_unicode_': r'R/Γ'})
#        E_tp1_.IncNrmNxt_fcts.update({'urlhandle': urlroot+'ExIncNrmNxt'})
        E_tp1_.IncNrmNxt_fcts.update({'py___code': py___code})
        E_tp1_.IncNrmNxt_fcts.update({'value_now': E_tp1_.IncNrmNxt})
        soln_crnt.E_tp1_.IncNrmNxt_fcts = E_tp1_.IncNrmNxt_fcts

        RNrm_PF_fcts = {
            'about': 'Expected Growth-Normalized Return'
        }
        py___code = 'Rfree / PermGroFac'
        E_tp1_.RNrm_PF = RNrm_PF = \
            eval(py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        RNrm_PF_fcts.update({'latexexpr': r'\PFRNrm'})
        RNrm_PF_fcts.update({'_unicode_': r'R/Γ'})
        RNrm_PF_fcts.update({'urlhandle': urlroot+'PFRNrm'})
        RNrm_PF_fcts.update({'py___code': py___code})
        RNrm_PF_fcts.update({'value_now': RNrm_PF})
        E_tp1_.RNrm_PF_fcts = RNrm_PF_fcts

        RNrm_PF_fcts = {
            'about': 'Expected Growth-Normalized Return'
        }
        py___code = 'Rfree / PermGroFac'
        E_tp1_.RNrm_PF = RNrm_PF = \
            eval(py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        RNrm_PF_fcts.update({'latexexpr': r'\PFRNrm'})
        RNrm_PF_fcts.update({'_unicode_': r'R/Γ'})
        RNrm_PF_fcts.update({'urlhandle': urlroot+'PFRNrm'})
        RNrm_PF_fcts.update({'py___code': py___code})
        RNrm_PF_fcts.update({'value_now': RNrm_PF})
        E_tp1_.RNrm_PF_fcts = RNrm_PF_fcts

        DiscLiv_fcts = {
            'about': 'Mortality-Inclusive Discounting'
        }
        py___code = 'DiscFac * LivPrb'
        Bilt.DiscLiv = DiscLiv = \
            eval(py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        DiscLiv_fcts.update({'latexexpr': r'\PFRNrm'})
        DiscLiv_fcts.update({'_unicode_': r'R/Γ'})
        DiscLiv_fcts.update({'urlhandle': urlroot+'PFRNrm'})
        DiscLiv_fcts.update({'py___code': py___code})
        DiscLiv_fcts.update({'value_now': DiscLiv})
        Bilt.DiscLiv_fcts = DiscLiv_fcts

    def build_facts_recursive(self):
        """
        Calculate results that depend on the last period solved (t+1).

        Returns
        -------
        None.

        """
        soln_crnt = self.soln_crnt
        tp1 = self.soln_futr.Bilt  # tp1 means t+1
        Bilt, Pars, E_tp1_ = soln_crnt.Bilt, soln_crnt.Pars, soln_crnt.E_tp1_

        givens = {**Pars.__dict__, **locals()}
        urlroot = Bilt.urlroot
        Pars.DiscLiv = Pars.DiscFac * Pars.LivPrb

        hNrm_fcts = {
            'about': 'Human Wealth '
        }
        py___code = '((PermGroFac / Rfree) * (1.0 + tp1.hNrm))'
        if soln_crnt.stge_kind['iter_status'] == 'terminal_partial':  # kludge:
            py___code = '0.0'  # hNrm = 0.0 for last period
        Bilt.hNrm = hNrm = \
            eval(py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        hNrm_fcts.update({'latexexpr': r'\hNrm'})
        hNrm_fcts.update({'_unicode_': r'R/Γ'})
        hNrm_fcts.update({'urlhandle': urlroot+'hNrm'})
        hNrm_fcts.update({'py___code': py___code})
        hNrm_fcts.update({'value_now': hNrm})
        Bilt.hNrm_fcts = hNrm_fcts

        BoroCnstNat_fcts = {
            'about': 'Natural Borrowing Constraint'
        }
        py___code = '(tp1.mNrmMin - tranShkMin)*(PermGroFac/Rfree)*permShkMin'
        if soln_crnt.stge_kind['iter_status'] == 'terminal_partial':  # kludge
            py___code = 'hNrm'  # Presumably zero
        Bilt.BoroCnstNat = BoroCnstNat = \
            eval(py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        BoroCnstNat_fcts.update({'latexexpr': r'\BoroCnstNat'})
        BoroCnstNat_fcts.update({'_unicode_': r''})
        BoroCnstNat_fcts.update({'urlhandle': urlroot+'BoroCnstNat'})
        BoroCnstNat_fcts.update({'py___code': py___code})
        BoroCnstNat_fcts.update({'value_now': BoroCnstNat})
        Bilt.BoroCnstNat_fcts = BoroCnstNat_fcts

        BoroCnst_fcts = {
            'about': 'Effective Borrowing Constraint'
        }
        py___code = 'BoroCnstNat if (BoroCnstArt == None) else ' + \
            '(BoroCnstArt if BoroCnstNat < BoroCnstArt else BoroCnstNat)'
        Bilt.BoroCnst = BoroCnst = \
            eval(py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        BoroCnst_fcts.update({'latexexpr': r'\BoroCnst'})
        BoroCnst_fcts.update({'_unicode_': r''})
        BoroCnst_fcts.update({'urlhandle': urlroot+'BoroCnst'})
        BoroCnst_fcts.update({'py___code': py___code})
        BoroCnst_fcts.update({'value_now': BoroCnst})
        Bilt.BoroCnst_fcts = BoroCnst_fcts

        # MPCmax is not a meaningful object in the PF model so is not created
        # there so create it here
        MPCmax_fcts = {
            'about': 'Maximal MPC in current period as m -> mNrmMin'
        }
        py___code = '1.0 / (1.0 + (RPF / tp1.MPCmax))'
        if soln_crnt.stge_kind['iter_status'] == 'terminal_partial':  # kludge:
            soln_crnt.tp1.MPCmax = float('inf')  # => MPCmax = 1 for last per
        Bilt.MPCmax = eval(
            py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        MPCmax_fcts.update({'latexexpr': r''})
        MPCmax_fcts.update({'urlhandle': urlroot+'MPCmax'})
        MPCmax_fcts.update({'py___code': py___code})
        MPCmax_fcts.update({'value_now': Bilt.MPCmax})
        Bilt.MPCmax_fcts = MPCmax_fcts

        mNrmMin_fcts = {
            'about': 'Min m is the max you can borrow'
        }
        py___code = 'BoroCnst'
        Bilt.mNrmMin =  \
            eval(py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        mNrmMin_fcts.update({'latexexpr': r'\mNrmMin'})
        mNrmMin_fcts.update({'py___code': py___code})
        Bilt.mNrmMin_fcts = mNrmMin_fcts

        MPCmin_fcts = {
            'about': 'Minimal MPC in current period as m -> infty'
        }
        py___code = '1.0 / (1.0 + (RPF / tp1.MPCmin))'
        if soln_crnt.stge_kind['iter_status'] == 'terminal_partial':  # kludge:
            py__code = '1.0'
        Bilt.MPCmin = \
            eval(py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        MPCmin_fcts.update({'latexexpr': r''})
        MPCmin_fcts.update({'urlhandle': urlroot+'MPCmin'})
        MPCmin_fcts.update({'py___code': py___code})
        MPCmin_fcts.update({'value_now': Bilt.MPCmin})
        Bilt.MPCmin_fcts = MPCmin_fcts

        MPCmax_fcts = {
            'about': 'Maximal MPC in current period as m -> mNrmMin'
        }
        py___code = '1.0 / (1.0 + (RPF / tp1.MPCmax))'
        if soln_crnt.stge_kind['iter_status'] == 'terminal_partial':  # kludge:
            Bilt.tp1.MPCmax = float('inf')  # => MPCmax = 1 for final period
        Bilt.MPCmax = \
            eval(py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        MPCmax_fcts.update({'latexexpr': r''})
        MPCmax_fcts.update({'urlhandle': urlroot+'MPCmax'})
        MPCmax_fcts.update({'py___code': py___code})
        MPCmax_fcts.update({'value_now': Bilt.MPCmax})
        Bilt.MPCmax_fcts = MPCmax_fcts

        cFuncLimitIntercept_fcts = {
            'about':
                'Vertical intercept of perfect foresight consumption function'}
        py___code = 'MPCmin * hNrm'
        Bilt.cFuncLimitIntercept = \
            eval(py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        cFuncLimitIntercept_fcts.update({'py___code': py___code})
        cFuncLimitIntercept_fcts.update({'latexexpr': r'\MPC \hNrm'})
        soln_crnt.Bilt.cFuncLimitIntercept_fcts = cFuncLimitIntercept_fcts

        cFuncLimitSlope_fcts = {
            'about': 'Slope of limiting consumption function'}
        py___code = 'MPCmin'
        cFuncLimitSlope_fcts.update({'py___code': 'MPCmin'})
        Bilt.cFuncLimitSlope = \
            eval(py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        cFuncLimitSlope_fcts.update({'py___code': py___code})
        cFuncLimitSlope_fcts = dict({'latexexpr': r'\MPCmin'})
        cFuncLimitSlope_fcts.update({'urlhandle': r'\MPC'})
        soln_crnt.Bilt.cFuncLimitSlope_fcts = cFuncLimitSlope_fcts

        # That's the end of things that are identical for PF and non-PF models

        return soln_crnt

    def def_value(self):
        """
        Build value function and store results in Bilt and Modl.value.

        Returns
        -------
        soln : solution object with value functions attached

        """
        return def_value_CRRA(self.soln_crnt, self.soln_crnt.Pars.CRRA)

    def solve_prepared_stage_divert(self):
        """
        Allow alternative solution method in special cases.

        Returns
        -------
        divert : boolean
            If False (usually), continue normal solution
            If True, produce alternative solution and store on self.soln_crnt
        """
        # bare-bones default terminal solution does not have all the facts
        # we need, because it is generic (for any u func) so add the facts
        crnt, futr = self.soln_crnt, self.soln_futr
        if futr.Bilt.stge_kind['iter_status'] != 'terminal_partial':
            return False  # Continue with normal solution procedures
        else:
            crnt = def_reward(crnt,
                              reward=def_utility_CRRA)  # reward = CRRA utility
            crnt.cFunc = crnt.Bilt.cFunc  # make cFunc accessible
            crnt = self.def_value()  # make value functions using cFunc
            crnt.vFunc = crnt.Bilt.vFunc  # make vFunc accessible for distance
            self.build_facts_infhor()
            crnt.Bilt.stge_kind['iter_status'] = 'iterator'  # now it's legit
            return True  # if pseudo_terminal=True, enhanced replaces original

    def solve_prepared_stage(self):  # inside ConsPerfForesightSolver
        """
        Solve the one-period/stage consumption-saving problem.

        Parameters
        ----------
        None (all are already in self)

        Returns
        -------
        solution : ConsumerSolution
            The solution to this period/stage's problem
        """

        if self.solve_prepared_stage_divert():  # Allow bypass of normal soln
            return self.soln_crnt  # created by solve_prepared_stage_divert

        crnt = self.soln_crnt

        crnt = def_transition_chosen__to__next_choice(crnt)
        crnt = self.from_chosen_states_make_E_tp1_(crnt)
        def_reward(crnt, reward=def_utility_CRRA)  # Utility rewards consumer

        crnt = self.build_decision_rules_and_value_functions(crnt)

        return crnt

    # alias for core.py
    solve = solve_prepared_stage

    def build_decision_rules_and_value_functions(self, crnt):
        self.make_cFunc_PF()
        return def_value_funcs(crnt, crnt.Pars.CRRA)

    def make_decision_rules(self, crnt):
        self.make_cFunc_PF()

    def solver_prep_solution_for_an_iteration(self):  # PF
        """
        Prepare the current stage for processing by the one-stage solver.
        """

        soln_crnt = self.soln_crnt

        Bilt, Pars = soln_crnt.Bilt, soln_crnt.Pars

        # Catch degenerate case of zero-variance income distributions
        # Otherwise "test cases" that try the degenerate dstns will fail
        if hasattr(Pars, "tranShkVals") and hasattr(Pars, "permShkVals"):
            if ((Pars.tranShkMin == 1.0) and (Pars.permShkMin == 1.0)):
                soln_crnt.E_tp1_.Inv_permShk = 1.0
                soln_crnt.E_tp1_.uInv_permShk = 1.0
        else:  # Missing trans or permShkVals; assume it's PF model
            Pars.tranShkMin = Pars.permShkMin = 1.0

        # Nothing needs to be done for terminal_partial
        if hasattr(Bilt, 'stge_kind'):
            if 'iter_status' in Bilt.stge_kind:
                if (Bilt.stge_kind['iter_status'] == 'terminal_partial'):
                    # solution_terminal is handmade, do not remake
                    return

        Bilt.stge_kind = \
            soln_crnt.stge_kind = {'iter_status': 'iterator',
                                   'slvr_type': self.__class__.__name__}

        return

    # Disambiguate "prepare_to_solve" from similar method names elsewhere
    # (preserve "prepare_to_solve" as alias because core.py calls it)
    prepare_to_solve = solver_prep_solution_for_an_iteration


##############################################################################

class ConsIndShockSetup(ConsPerfForesightSolver):
    """
    Superclass for solvers of one period consumption-saving problems with
    constant relative risk aversion utility and permanent and transitory shocks
    to labor income, containing code shared among alternative specific solvers.

    N.B.: Because this is a one stge solver, objects that (in the full problem)
    are lists because they are allowed to vary at different stages, are scalars
    here because the value that is appropriate for the current stage is the one
    that will be passed.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncShkDstn : distribution.Distribution
        A discrete approximation to the income process between the period
        being solved and the one immediately following
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  If it is less than the natural borrowing constraint,
        then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
        rowing constraint.
    aXtraGrid: np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear inter-
        polation.
    solveMethod : str, optional
        Solution method to use
    """
    shock_vars = ['tranShkDstn', 'permShkDstn']  # Unemp shock=min(transShkVal)

    # TODO: CDC 20210416: Params shared with PF are in different order. Fix
    def __init__(
            self, solution_next, IncShkDstn, LivPrb, DiscFac, CRRA, Rfree,
            PermGroFac, BoroCnstArt, aXtraGrid, vFuncBool, CubicBool,
            permShkDstn, tranShkDstn,
            solveMethod='EGM',
            shockTiming='EOP',
            solverType='HARK',
            **kwds):
        # First execute PF solver init
        # We must reorder params by hand in case someone tries positional solve

        ConsPerfForesightSolver.__init__(self, solution_next, DiscFac=DiscFac,
                                         LivPrb=LivPrb, CRRA=CRRA,
                                         Rfree=Rfree, PermGroFac=PermGroFac,
                                         BoroCnstArt=BoroCnstArt,
                                         IncShkDstn=IncShkDstn,
                                         permShkDstn=permShkDstn,
                                         tranShkDstn=tranShkDstn,
                                         solveMethod=solveMethod,
                                         shockTiming=shockTiming,
                                         **kwds)

        # ConsPerfForesightSolver.__init__ makes self.soln_crnt
        soln_crnt = self.soln_crnt

        # Things we have built, exogenous parameters, and model structures:
        Bilt, Pars, Modl = soln_crnt.Bilt, soln_crnt.Pars, soln_crnt.Modl

        Modl.solveMethod = solveMethod
        Modl.shockTiming = shockTiming

        Bilt.aXtraGrid = aXtraGrid
        self.CubicBool = CubicBool

        # In which column is each object stored in IncShkDstn?
        Pars.permPos = IncShkDstn.parameters['ShkPosn']['perm']
        Pars.tranPos = IncShkDstn.parameters['ShkPosn']['tran']

        # Bcst are "broadcasted" values: serial list of every permutation
        # Makes it fast to take expectations using 𝔼_dot
        Pars.permShkValsBcst = permShkValsBcst = IncShkDstn.X[Pars.permPos]
        Pars.tranShkValsBcst = tranShkValsBcst = IncShkDstn.X[Pars.tranPos]

        Pars.ShkPrbs = ShkPrbs = IncShkDstn.pmf

        Pars.permShkPrbs = permShkPrbs = permShkDstn.pmf
        Pars.permShkVals = permShkVals = permShkDstn.X
        # Confirm that perm shocks have expectation near one
        assert_approx_equal(𝔼_dot(permShkPrbs, permShkVals), 1.0)

        Pars.tranShkPrbs = tranShkPrbs = tranShkDstn.pmf
        Pars.tranShkVals = tranShkVals = tranShkDstn.X
        # Confirm that tran shocks have expectation near one
        assert_approx_equal(𝔼_dot(tranShkPrbs, tranShkVals), 1.0)

        Pars.permShkMin = permShkMin = np.min(permShkVals)
        Pars.tranShkMin = tranShkMin = np.min(tranShkVals)

        Pars.permShkMax = permShkMax = np.max(permShkVals)
        Pars.tranShkMax = tranShkMax = np.max(tranShkVals)

        Pars.UnempPrb = Pars.tranShkPrbs[0]

        Pars.inc_min_Prb = np.sum(  # All cases where perm and tran Shk are Min
            ShkPrbs[ \
                permShkValsBcst * tranShkValsBcst == permShkMin * tranShkMin
            ]
        )

        Pars.inc_max_Prb = np.sum(  # All cases where perm and tran Shk are Min
            ShkPrbs[ \
                permShkValsBcst * tranShkValsBcst == permShkMax * tranShkMax
            ]
        )
        Pars.inc_max_Val = permShkMax * tranShkMax

    def build_facts_infhor(self):
        """
        Calculate expectations and facts for models with uncertainty.

        For versions with uncertainty in transitory and/or permanent shocks,
        adds to the solution a set of results useful for calculating
        various diagnostic conditions about the problem, and stable
        points (if they exist).

        Parameters
        ----------
        solution: ConsumerSolution
            Solution to this period's problem, which must have attribute cFunc.

        Returns
        -------
        solution : ConsumerSolution
            Same solution that was provided, augmented with the factors

        """
        super().build_facts_infhor()  # Make the facts built by the PF model

        soln_crnt = self.soln_crnt

        Bilt, Pars, E_tp1_ = soln_crnt.Bilt, soln_crnt.Pars, soln_crnt.E_tp1_

        # The 'givens' do not change as facts are constructed
        givens = {**Pars.__dict__, **soln_crnt.__dict__}

        Bilt.𝔼_dot = 𝔼_dot  # add dot product expectations operator to envt
        Bilt.E_dot = E_dot  # plain not doublestruck E

        urlroot = Bilt.urlroot

        # Many other _fcts will have been inherited from the perfect foresight

        # Here we need compute only those objects whose value changes from PF
        # (or does not exist in PF case)

        E_tp1_.IncNrmNxt_fcts = {
            'about': 'Expected income next period'
        }
        py___code = '𝔼_dot(ShkPrbs, tranShkValsBcst * permShkValsBcst)'
        E_tp1_.IncNrmNxt = E_tp1_.IncNrmNxt = eval(
            py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        E_tp1_.IncNrmNxt_fcts.update({'latexexpr': r'ExIncNrmNxt'})
        E_tp1_.IncNrmNxt_fcts.update({'_unicode_': r'𝔼[tranShk permShk]= 1.0'})
        E_tp1_.IncNrmNxt_fcts.update({'urlhandle': urlroot+'ExIncNrmNxt'})
        E_tp1_.IncNrmNxt_fcts.update({'py___code': py___code})
        E_tp1_.IncNrmNxt_fcts.update({'value_now': E_tp1_.IncNrmNxt})
        soln_crnt.E_tp1_.IncNrmNxt_fcts = E_tp1_.IncNrmNxt_fcts

        E_tp1_.Inv_permShk_fcts = {
            'about': 'Expected Inverse of Permanent Shock'
        }
        py___code = '𝔼_dot(1/permShkVals, permShkPrbs)'
        E_tp1_.Inv_permShk = E_tp1_.Inv_permShk = eval(
            py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        E_tp1_.Inv_permShk_fcts.update({'latexexpr': r'\ExInvpermShk'})
        E_tp1_.Inv_permShk_fcts.update({'urlhandle': urlroot+'ExInvpermShk'})
        E_tp1_.Inv_permShk_fcts.update({'py___code': py___code})
        E_tp1_.Inv_permShk_fcts.update({'value_now': E_tp1_.Inv_permShk})
        soln_crnt.E_tp1_.Inv_permShk_fcts = E_tp1_.Inv_permShk_fcts

        E_tp1_.RNrm_fcts = {
            'about': 'Expected Stochastic-Growth-Normalized Return'
        }
        py___code = 'RNrm_PF * E_tp1_.Inv_permShk'
        E_tp1_.RNrm = eval(
            py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        E_tp1_.RNrm_fcts.update({'latexexpr': r'\ExRNrm'})
        E_tp1_.RNrm_fcts.update({'_unicode_': r'𝔼[R/Γψ]'})
        E_tp1_.RNrm_fcts.update({'urlhandle': urlroot+'ExRNrm'})
        E_tp1_.RNrm_fcts.update({'py___code': py___code})
        E_tp1_.RNrm_fcts.update({'value_now': E_tp1_.RNrm})
        E_tp1_.RNrm_fcts = E_tp1_.RNrm_fcts

        E_tp1_.uInv_permShk_fcts = {
            'about': 'Expected Utility for Consuming Permanent Shock'
        }
        py___code = '𝔼_dot(permShkValsBcst**(1-CRRA), ShkPrbs)'
        E_tp1_.uInv_permShk = E_tp1_.uInv_permShk = eval(
            py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        E_tp1_.uInv_permShk_fcts.update({'latexexpr': r'\ExuInvpermShk'})
        E_tp1_.uInv_permShk_fcts.update({'urlhandle': r'ExuInvpermShk'})
        E_tp1_.uInv_permShk_fcts.update({'py___code': py___code})
        E_tp1_.uInv_permShk_fcts.update({'value_now': E_tp1_.uInv_permShk})
        E_tp1_.uInv_permShk_fcts = E_tp1_.uInv_permShk_fcts

        GPFNrm_fcts = {
            'about': 'Normalized Expected Growth Patience Factor'
        }
        py___code = 'GPFRaw * E_tp1_.Inv_permShk'
        Bilt.GPFNrm = eval(py___code, {},
                           {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        GPFNrm_fcts.update({'latexexpr': r'\GPFNrm'})
        GPFNrm_fcts.update({'_unicode_': r'Þ_Γ'})
        GPFNrm_fcts.update({'urlhandle': urlroot+'GPFNrm'})
        GPFNrm_fcts.update({'py___code': py___code})
        Bilt.GPFNrm_fcts = GPFNrm_fcts

        GICNrm_fcts = {
            'about': 'Stochastic Growth Normalized Impatience Condition'
        }
        GICNrm_fcts.update({'latexexpr': r'\GICNrm'})
        GICNrm_fcts.update({'urlhandle': urlroot+'GICNrm'})
        GICNrm_fcts.update({'py___code': 'test: GPFNrm < 1'})
        Bilt.GICNrm_fcts = GICNrm_fcts

        FVAC_fcts = {  # overwrites PF version
            'about': 'Finite Value of Autarky Condition'
        }

        FVAF_fcts = {  # overwrites PF version FVAFPF
            'about': 'Finite Value of Autarky Factor'
        }
        py___code = 'LivPrb * DiscLiv'
        Bilt.FVAF = eval(py___code, {},
                         {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        FVAF_fcts.update({'latexexpr': r'\FVAF'})
        FVAF_fcts.update({'urlhandle': urlroot+'FVAF'})
        FVAF_fcts.update({'py___code': py___code})
        Bilt.FVAF_fcts = FVAF_fcts

        FVAC_fcts = {  # overwrites PF version
            'about': 'Finite Value of Autarky Condition'
        }
        FVAC_fcts.update({'latexexpr': r'\FVAC'})
        FVAC_fcts.update({'urlhandle': urlroot+'FVAC'})
        FVAC_fcts.update({'py___code': 'test: FVAF < 1'})
        Bilt.FVAC_fcts = FVAC_fcts

        WRPF_fcts = {
            'about': 'Weak Return Patience Factor'
        }
        py___code = '(UnempPrb ** (1 / CRRA)) * RPF'
        Bilt.WRPF = WRPF = eval(py___code, {},
                                {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        WRPF_fcts.update({'latexexpr': r'\WRPF'})
        WRPF_fcts.update({'_unicode_': r'℘^(1/\rho) RPF'})
        WRPF_fcts.update({'urlhandle': urlroot+'WRPF'})
        WRPF_fcts.update({'value_now': WRPF})
        WRPF_fcts.update({'py___code': py___code})
        Bilt.WRPF_fcts = WRPF_fcts

        WRIC_fcts = {
            'about': 'Weak Return Impatience Condition'
        }
        WRIC_fcts.update({'latexexpr': r'\WRIC'})
        WRIC_fcts.update({'urlhandle': urlroot+'WRIC'})
        WRIC_fcts.update({'py___code': 'test: WRPF < 1'})
        Bilt.WRIC_fcts = WRIC_fcts

        DiscGPFNrmCusp_fcts = {
            'about': 'DiscFac s.t. GPFNrm = 1'
        }
        py___code = '((PermGroFac/E_tp1_.Inv_permShk)**(CRRA))/Rfree'
        Bilt.DiscGPFNrmCusp = DiscGPFNrmCusp = \
            eval(py___code, {}, {**E_tp1_.__dict__, **Bilt.__dict__, **givens})
        DiscGPFNrmCusp_fcts.update({'latexexpr': ''})
        DiscGPFNrmCusp_fcts.update({'value_now': DiscGPFNrmCusp})
        DiscGPFNrmCusp_fcts.update({'py___code': py___code})
        Bilt.DiscGPFNrmCusp_fcts = DiscGPFNrmCusp_fcts

    def build_facts_recursive(self):
        """
        Calculate recursive facts for current period from next.

        Returns
        -------
        soln_crnt : solution

        """
        super().build_facts_recursive()

        # All the recursive facts are required for PF model so already exist
        # But various lambda functions are interesting when uncertainty exists

        soln_crnt = self.soln_crnt
        Bilt = soln_crnt.Bilt
        Pars = soln_crnt.Pars
        E_tp1_ = soln_crnt.E_tp1_

        # To use these it is necessary to have created an alias to
        # the relevant namespace on the solution object, e.g.
        # E_tp1_ = [soln].E_tp1_
        # Bilt = [soln].Bilt
        # Pars = [soln].Pars

        # Given m, value of c where 𝔼[m_{t+1}]=m_{t}
        E_tp1_.m_tp1_minus_m_t_eq_0 = (
            lambda m_t:
            m_t * (1 - 1/E_tp1_.RNrm) + (1/E_tp1_.RNrm)
        )
        # Given m, value of c where 𝔼[mLev_{t+1}/mLev_{t}]=Bilt.Pars.permGroFac
        # Solves for c in equation at url/#balgrostable
        E_tp1_.permShk_times_m_tp1_minus_m_t_eq_0 = (
            lambda m_t:
            m_t * (1 - E_tp1_.Inv_RNrm_PF) + E_tp1_.Inv_RNrm_PF
        )
        # 𝔼[m_{t+1} pLev_{t+1}/pLev_{t}] as a fn of a_{t}
        E_tp1_.mLev_tp1_Over_pLev_t_from_a_t = (
            lambda a_t:
            𝔼_dot(Pars.PermGroFac *
                  Pars.permShkValsBcst *
                  (E_tp1_.RNrm_PF/Pars.permShkValsBcst) * a_t
                  + Pars.tranShkValsBcst,
                  Pars.ShkPrbs)
        )
        # 𝔼[c_{t+1} pLev_{t+1}/pLev_{t}] as a fn of a_{t}
        E_tp1_.cLev_tp1_Over_pLev_t_from_a_t = (
            lambda a_t:
            𝔼_dot(Pars.PermGroFac *
                  Pars.permShkValsBcst *
                  Bilt.cFunc((E_tp1_.RNrm_PF/Pars.permShkValsBcst) * a_t
                             + Pars.tranShkValsBcst),
                  Pars.ShkPrbs)
        )
        E_tp1_.c_where_E_tp1__m_tp1_minus_m_t_eq_0 = \
            lambda m_t: \
            m_t * (1 - 1/E_tp1_.RNrm) + (1/E_tp1_.RNrm)
        # Solve the equation at url/#balgrostable
        E_tp1_.c_where_E_tp1__permShk_times_m_tp1_minus_m_t_eq_0 = \
            lambda m_t: \
            (m_t * (1 - 1/E_tp1_.RNrm_PF)) + (1/E_tp1_.RNrm_PF)
        # mNrmTrg solves E_tp1_.RNrm*(m - c(m)) + 𝔼[inc_next] - m = 0
        E_tp1_.m_tp1_minus_m_t = (
            lambda m_t:
            E_tp1_.RNrm * (m_t - Bilt.cFunc(m_t)) + E_tp1_.IncNrmNxt - m_t
        )
        E_tp1_.cLev_tp1_Over_pLev_t_from_num_a_t = (
            lambda a_t:
            𝔼_dot(
                Pars.permShkValsBcst * Pars.PermGroFac * Bilt.cFunc(
                    (E_tp1_.RNrm_PF/Pars.permShkValsBcst) *
                    a_t + Pars.tranShkValsBcst
                ),
                Pars.ShkPrbs)
        )
        E_tp1_.cLev_tp1_Over_pLev_t_from_lst_a_t = (
            lambda a_lst: list(map(
                E_tp1_.cLev_tp1_Over_pLev_t_from_num_a_t, a_lst
            ))
        )
        E_tp1_.cLev_tp1_Over_pLev_t_from_a_t = (
            lambda a_t:
            E_tp1_.cLev_tp1_Over_pLev_t_from_lst_a_t(a_t)
            if (type(a_t) == list or type(a_t) == np.ndarray) else
            E_tp1_.cLev_tp1_Over_pLev_t_from_num_a_t(a_t)
        )
        E_tp1_.cLev_tp1_Over_pLev_t_from_lst_m_t = (
            lambda m_t:
            E_tp1_.cLev_tp1_Over_pLev_t_from_lst_a_t(m_t -
                                                     Bilt.cFunc(m_t))
        )
        E_tp1_.cLev_tp1_Over_pLev_t_from_num_m_t = (
            lambda m_t:
            E_tp1_.cLev_tp1_Over_pLev_t_from_num_a_t(m_t -
                                                     Bilt.cFunc(m_t))
        )
        E_tp1_.cLev_tp1_Over_cLev_t_from_m_t = (
            lambda m_t:
            E_tp1_.cLev_tp1_Over_pLev_t_from_lst_m_t(m_t) / Bilt.cFunc(m_t)
            if (type(m_t) == list or type(m_t) == np.ndarray) else
            E_tp1_.cLev_tp1_Over_pLev_t_from_num_m_t(m_t) / Bilt.cFunc(m_t)
        )
        E_tp1_.permShk_tp1_times_m_tp1_minus_m_t = (
            lambda m_t:
            E_tp1_.RNrm_PF * (m_t - Bilt.cFunc(m_t)) + E_tp1_.IncNrmNxt - m_t
        )

        self.soln_crnt = soln_crnt

        return soln_crnt


class ConsIndShockSolverBasic(ConsIndShockSetup):
    """
    Solves a single period of a standard consumption-saving problem.

    Uses linear interpolation and missing the ability to calculate the value
    function.  ConsIndShockSolver inherits from this class and adds the ability
    to perform cubic interpolation and to calculate the value function.

    Note that this class does not have its own initializing method. It initial-
    izes the same problem in the same way as ConsIndShockSetup, from which it
    inherits.
    """

    def make_chosen_state_grid(self):
        """
        Make grid of potential values of state variable(s) after choice(s).

        Parameters
        ----------
        none

        Returns
        -------
        aNrmGrid : np.array
            A 1D array of end-of-period assets; also is made attribute of Bilt.
        """
        # We define aNrmGrid all the way from BoroCnstNat up to max(aXtraGrid)
        # even if BoroCnstNat<BoroCnstArt, so we can construct the consumption
        # function as lower envelope of the (by the artificial borrowing con-
        # straint) unconstrained consumption function, and  artificially con-
        # strained consumption function.
        self.soln_crnt.Bilt.aNrmGrid = np.asarray(
            self.soln_crnt.Bilt.aXtraGrid) + self.soln_crnt.Bilt.BoroCnstNat

        return self.soln_crnt.Bilt.aNrmGrid

    def make_E_tp1_(self, IncShkDstn):
        """
        Calculate expectations after choices but before shocks.

        Parameters
        ----------
        IncShkDstn : DiscreteDistribution
            The distribution of the stochastic shocks to income.
        """
        crnt, futr = self.soln_crnt, self.soln_futr

        Bilt, Pars, E_tp1_ = crnt.Bilt, crnt.Pars, crnt.E_tp1_

        shockTiming = Pars.shockTiming  # this_EOP or next_BOP (or both)

        states_chosen = Bilt.aNrmGrid
        permPos = IncShkDstn.parameters['ShkPosn']['perm']

        if shockTiming == 'EOP':  # shocks happen at end of this period
            CRRA = futr.Bilt.vFunc.CRRA
            Discount = Pars.DiscLiv
            vFunc = futr.Bilt.vFunc
            cFunc = futr.Bilt.cFunc
            PermGroFac = futr.Pars.PermGroFac
        else:  # default to BOP
            CRRA = Pars.CRRA
            Discount = 1.0
            vFunc = crnt.Bilt.vFunc
            cFunc = crnt.Bilt.cFunc
            PermGroFac = Pars.PermGroFac

        Rfree = Pars.Rfree

        # This is the efficient place to compute expectations of anything
        # at very low marginal cost by adding to list of things calculated
        def funcs_to_expect(xfer_shks_bcst, states_chosen):
            # tp1 contains the realizations of the state variables
            next_choice_states = \
                self.transit_chosen__to__next_choice(
                    xfer_shks_bcst, states_chosen, IncShkDstn)
            mNrm = next_choice_states.mNrm
            # Random (Rnd) shocks to permanent income affect mean PermGroFac
            PermGroFacShk = xfer_shks_bcst[permPos]*PermGroFac
            # expected value function derivatives 0, 1, 2
            v_0 = PermGroFacShk ** (1-CRRA-0) * vFunc(mNrm)
            v_1 = PermGroFacShk ** (1-CRRA-1) * vFunc.dm(mNrm) * Rfree
            v_2 = PermGroFacShk ** (1-CRRA-2) * vFunc.dm.dm(mNrm) * Rfree \
                * Rfree
            # cFunc derivatives 0, 1 (level and MPC); no need, but ~zero cost.
            c_0 = cFunc(mNrm)
            c_1 = cFunc.derivative(mNrm)
            return Discount*np.array([v_0, v_1, v_2, c_0, c_1])

        E_tp1_.given_chosen = np.squeeze(
            expect_funcs_given_states(
                IncShkDstn,
                funcs_to_expect,
                states_chosen)
        )
        # Store positions of the various objects for later retrieval
        E_tp1_.v0_pos, E_tp1_.v1_pos, E_tp1_.v2_pos = 0, 1, 2
        E_tp1_.c0_pos, E_tp1_.c1_pos = 4, 5

    def build_cFunc_using_EGM(self):
        """
        Find interpolation points (c, m) for the consumption function.

        Parameters
        ----------
        none

        Returns
        -------
        cFunc : LowerEnvelope or LinerarInterp
        """
        crnt = self.soln_crnt
        Bilt, E_tp1_, Pars = crnt.Bilt, crnt.E_tp1_, crnt.Pars
        v1_pos = E_tp1_.v1_pos  # first derivative of value function at chosen
        u, aNrmGrid, BoroCnstArt = Bilt.u, Bilt.aNrmGrid, Pars.BoroCnstArt

        # Endogenous Gridpoints step
        # [v1_pos]: precalculated first derivative (E_tp1_from_chosen_states)
        cNrmGrid = u.dc.Nvrs(E_tp1_.given_chosen[v1_pos])
        mNrmGrid = aNrmGrid + cNrmGrid

        # Limiting consumption is zero as m approaches BoroCnstNat
        mPlus, cPlus = (
            np.insert(mNrmGrid, 0, Bilt.BoroCnstNat, axis=-1),
            np.insert(cNrmGrid, 0, 0.0, axis=-1))  # c = 0 at BoroCnstNat

        # Store these for future use
        Bilt.cNrmGrid, Bilt.mNrmGrid = cNrmGrid, mNrmGrid

        interpolator = self.make_cFunc_linear  # default is piecewise linear

        if self.CubicBool:
            interpolator = self.make_cFunc_cubic

        # Use the given interpolator to construct the consumption function
        cFuncUnc = interpolator(mPlus, cPlus)  # Unc = Unconstrained (this prd)

        # Combine constrained and unconstrained functions into the true cFunc
        # by choosing the lower of the constrained and unconstrained functions
        # LowerEnvelope should only be used when BoroCnstArt is true
        if BoroCnstArt is None:
            cFunc = cFuncUnc
        else:
            # CDC 20210614: LinearInterp and LowerEnvelope are both handmade
            # We should substitute standard ways to do these things
            # EconForge interpolation.py or scipy.interpolate for interpolation
            Bilt.cFuncCnst = LinearInterp(
                np.array([Bilt.mNrmMin, Bilt.mNrmMin + 1.0]),
                np.array([0.0, 1.0]))
            cFunc = LowerEnvelope(cFuncUnc, Bilt.cFuncCnst, nan_bool=False)

        return cFunc

    def build_decision_rules_and_value_functions(self):
        """
        Construct consumption function and marginal value function.

        Given the grid of end-of-period values of assets a, use the endogenous
        gridpoints method to obtain the corresponding values of consumption,
        and use the dynamic budget constraint to obtain the corresponding value
        of market resources m.

        Parameters
        ----------
        none (relies upon self.soln_crnt.aNrmGrid to exist at invocation)

        Returns
        -------
        solution : ConsumerSolution
            The solution to the single period consumption-saving problem.
        """
        crnt = self.soln_crnt
        Bilt, Pars = crnt.Bilt, crnt.Pars

        crnt.cFunc = Bilt.cFunc = self.build_cFunc_using_EGM()
        crnt = def_value_funcs(crnt, Pars.CRRA)

        return crnt

    def make_cFunc_linear(self, mNrm, cNrm):
        """
        Make linear interpolation for the (unconstrained) consumption function.

        Parameters
        ----------
        mNrm : np.array
            Corresponding market resource points for interpolation.
        cNrm : np.array
            Consumption points for interpolation.

        Returns
        -------
        cFunc_unconstrained : LinearInterp
            The unconstrained consumption function for this period.
        """
        cFunc_unconstrained = LinearInterp(
            mNrm, cNrm,
            self.soln_crnt.Bilt.cFuncLimitIntercept,
            self.soln_crnt.Bilt.cFuncLimitSlope
        )
        return cFunc_unconstrained

    def from_chosen_states_make_E_tp1_(self):
        """
        Calculate circumstances of an agent before the realization of the labor
        income shocks that constitute the transition to the next period's state.

        Resulting solution object contains the value function vFunc and
        its derivatives.  Does not calculate consumption function cFunc:
        that is a consequence of vFunc.da but is calculated in the
        stage that calls this one.

        Parameters
        ----------
        none (all should be on self)

        Returns
        -------
        solution : ConsumerSolution object
            Contains info (like vFunc.da) required to construct consumption
        """
        soln_crnt = self.soln_crnt

        # Add a bunch of useful info to solution object
        # CDC 20200428: "useful" only for a candidate converged solution
        # in an infinite horizon model.  It's virtually costless to compute but
        # usually there would not be much point in computing it for a
        # non-final infhor stage or finhor.  Exception: Better to construct
        # finhor LiqConstr without MaxKinks but user might want to know
        # these facts
        # TODO: Distinguish between those things that need to be computed for
        # "useful" computations in the final stage, and just info,
        # and make mandatory only the computations of the former category
        self.build_facts_infhor()
        self.build_facts_recursive()  # These require solution to successor

        soln_crnt = def_transition_chosen__to__next_choice(soln_crnt)
        soln_crnt = self.make_chosen_state_grid()
        self.make_E_tp1_(self.soln_crnt.Pars.IncShkDstn)

        return soln_crnt

    def solve_prepared_stage(self):  # solve ONE stage (ConsIndShockSolver)
        """
        Solves one period of the consumption-saving problem. 

        The ".Bilt" namespace on the returned solution object includes
            * decision rule (consumption function), cFunc
            * value and marginal value functions vFunc and vFunc.dm
            * a minimum possible level of normalized market resources mNrmMin
            * normalized human wealth hNrm, and bounding MPCs MPCmin and MPCmax.

        If the user sets `CubicBool` to True, cFunc is interpolated
        with a cubic Hermite interpolator.  This is much smoother than the default
        piecewise linear interpolator, and permits construction of the marginal
        marginal value function vFunc.dm.dm (which occurs automatically).

        In principle, the resulting cFunc will be numerically incorect at values
        of market resources where a marginal increment to m would discretely
        change the probability of a future constraint binding, because in
        principle cFunc is nondifferentiable at those points (cf LiqConstr
        REMARK).

        In practice, if
        the distribution of shocks is not too granular, the approximation error
        is minor.

        Parameters
        ----------
        none (all should be on self)

        Returns
        -------
        solution : ConsumerSolution
            The solution to this period/stage's problem.
        """

        if self.solve_prepared_stage_divert():  # Allow bypass of normal soln
            return self.soln_crnt  # created by bypass

        crnt = self.soln_crnt
        Pars = crnt.Pars
        shockTiming, solveMethod = Pars.shockTiming, Pars.solveMethod

        if solveMethod == 'Generic':
            # like, aNrm_{t} -> kNrm_{t+1}:
            crnt = self.def_transition_EOP__to__next_BOP(crnt)
            # draw EOP shocks (if any):
            crnt = self.def_transition_chosen__to__EOP(crnt)
            # create aNrmGrid
            crnt = self.make_chosen_state_grid(crnt)
            # like, β E[R Γ_{t+1}^{-ρ}u'(c_{t+1})]
            crnt = self.from_chosen_states_make_E_tp1_(crnt)
            # Defines current utility
            crnt = self.def_reward(crnt, reward=def_utility_CRRA)
            # like, aNrm = mNrm - cNrm
            crnt = self.def_transition_choice__to__chosen(crnt)
            # like, cFunc and vFunc
            crnt = self.make_decision_rules_and_value_functions(crnt)
            # draw BOP shocks, if any
            crnt = self.def_transition_BOP__to__choice(crnt)
            # expectations before BOP shocks realized: t_E
            crnt = self.t_E_from_BOP_states_make(crnt)
            return crnt

        # if not using generic, then solve using custom method

        # transition for "saver" who has chosen aNrm; slightly different
        # depending on whether shocks are at End or Beginning of period
        if shockTiming == 'EOP':  # Shocks at End of Period
            def_transition_chosen__to__next_choice(crnt)  # result: m_{t+1}
        else:
            def_transition_chosen__to__next_BOP(crnt)  # result: k_{t+1}

        # Given that transition, calculate expectations
        self.from_chosen_states_make_E_tp1_()

        def_reward(crnt, reward=def_utility_CRRA)  # Utility is consumer reward

        # Having calculated E(marginal value, etc) of saving, construct c
        self.build_decision_rules_and_value_functions()

        self.t_E_from_BOP_states_make()  # t_E: Before BOP shocks are realized

        return crnt

    solve = solve_prepared_stage

    def t_E_from_BOP_states_make(self):
        pass

    def transit_chosen__to__next_choice(self, xfer_shks_bcst, chosen_states,
                                        IncShkDstn):
        """
        Return array of values of normalized market resources m
        corresponding to permutations of potential realizations of
        the permanent and transitory income shocks, given the value of
        end-of-period assets aNrm.

        Parameters
        ----------
        xfer_shks_bcst: 2D ndarray
            Permanent and transitory income shocks in 2D ndarray

        aNrm: float
            Normalized end-of-period assets this period

        Returns
        -------
        transit_chosen__to__next_choice : namespace with results of applying transition eqns
        """

        stge = self.soln_crnt
        Pars, Modl = stge.Pars, stge.Modl
        Transitions = stge.Modl.Transitions

        permPos, tranPos = (
            Pars.IncShkDstn.parameters['ShkPosn']['perm'],
            Pars.IncShkDstn.parameters['ShkPosn']['tran'])

        zeros = chosen_states - chosen_states  # zero array of the right size

        xfer_vars = {
            'permShk': xfer_shks_bcst[permPos] + zeros,  # + zeros fixes size
            'tranShk': xfer_shks_bcst[tranPos] + zeros,
            'aNrm': chosen_states
        }

        # Everything needed to execute the transition equations
        Info = {**Pars.__dict__, **xfer_vars}

        chosen__to__next_choice = \
            Transitions['chosen__to__next_choice']

        for eqn_name in chosen__to__next_choice.__dict__['eqns'].keys():
            exec(chosen__to__next_choice.__dict__['eqns'][eqn_name], Info)

        tp1 = SimpleNamespace()
        tp1.mNrm = Info['mNrm']

        return tp1


###############################################################################

class ConsIndShockSolver(ConsIndShockSolverBasic):
    """
    Solves a single period of a standard consumption-saving problem.
    It inherits from ConsIndShockSolverBasic, and adds ability to perform cubic
    interpolation.
    """

    def make_cFunc_cubic(self, mNrm_Vec, cNrm_Vec):
        """
        Make cubic spline interpolation of unconstrained consumption function.

        Requires self.soln_crnt.Bilt.aNrm to have been computed already.

        Parameters
        ----------
        mNrm_Vec : np.array
            Corresponding market resource points for interpolation.
        cNrm_Vec : np.array
            Consumption points for interpolation.

        Returns
        -------
        cFuncUnc : CubicInterp
            The unconstrained consumption function for this period.
        """
        soln_crnt = self.soln_crnt
        Bilt, E_tp1_ = soln_crnt.Bilt, soln_crnt.E_tp1_
        v2_pos = E_tp1_.v2_pos  # second derivative of value function
        u = Bilt.u

        dc_da = E_tp1_.given_chosen[v2_pos] / u.dc.dc(np.array(cNrm_Vec[1:]))
        MPC = dc_da / (dc_da + 1.0)
        MPC = np.insert(MPC, 0, Bilt.MPCmax)

        cFuncUnc = CubicInterp(
            mNrm_Vec, cNrm_Vec, MPC, Bilt.MPCmin *
            Bilt.hNrm, Bilt.MPCmin
        )
        return cFuncUnc
