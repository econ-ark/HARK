# -*- coding: utf-8 -*-
import logging
from builtins import (str, breakpoint)
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
from numpy import dot as E_dot  # easier to type
from numpy.testing import assert_approx_equal
from scipy.optimize import newton as find_zero_newton

from HARK.ConsumptionSaving.ConsIndShockModelOld \
    import ConsumerSolution as ConsumerSolutionOlder
from HARK.ConsumptionSaving.ConsIndShockModel_Both import (
    define_t_reward, def_utility_CRRA, def_value_funcs, def_value_CRRA,
    define_transition
)
from HARK.core import (_log, core_check_condition, MetricObject)
from HARK.distribution import calc_expectation \
    as expect_funcs_across_shocks_given_states
from HARK.interpolation import (CubicInterp, LowerEnvelope, LinearInterp)


class agent_stage_solution(MetricObject):
    """
    Framework for solution of a single stage of a decision problem.

    A "stage" of a problem is the smallest unit into which it is useful
    to break down the problem. Often this will correspond to a time period,
    but sometimes a stage may correspond to subcomponents of a problem
    which are conceived of as being solved simultaneously but which are
    computationally useful to solve sequentially.

    Provides a foundational structure that all models will build on.  It must
    be specialized and elaborated to solve any particular problem.

    Parameters
    ----------
    solution_follows : agent_stage_solution

    Returns
    -------
    solution_current : agent_stage_solution

    Elements of the current solution crnt contain, but are not limited to:
        Pars : The parameters used in solving this stage of the model
        Bilt : Objects constructed and retained from the solution process
        Modl : Equations of the model, in the form of the python code
            that instantiates the computational solution

            This is broken down into:

            States : predetermined variables at the time of decisions
            Controls : variables under control of the decisionmaker
            Reward : current payoff as function of states and controls
            Transitions : evolution of states
            Choices : conditions that determine the agent's choices

            The minimal required element of a solution_current object is
                [vFunc] : value function
                    Bellman value function the agent expects to experience for
                    behaving according to the dynamically optimal plan over
                    the remainder of the horizon.

            Solution objects will usually also contain a 'decision rule' (for
            example a consumption function), although this is not a requirement.

    Other components of a solution object are:

    stge_kind : dict
        Dictionary with info about this solution stage
        One required entry keeps track of the nature of the stage:
            {'iter_status':'not initialized'}: Before model is set up
            {'iter_status':'finished'}: Stopping requirements are satisfied
                If such requirements are satisfied, {'tolerance':tolerance}
                should exist recording what convergence tolerance was satisfied
            {'iter_status':'iterator'}: Status during iteration
                solution[0].distance_last records the last distance
            {'iter_status':'terminal_partial'}:Bare-bones terminal period/stage
                Does not contain all the info needed to begin solution
                Solver will augment and replace it with 'iterator' stage
        Other uses include keeping track of the nature of the next stage
    completed_cycles : integer
        The number of cycles of the model solved before this call
    solveMethod : str, optional
        The name of the solution method to use, e.g. 'EGM'
    solverType : str, optional
        The name of the type of solver ('HARK', 'Dolo')
    eventTiming : str, optional
        Clarifies timing of events whose timing might otherwise be ambiguous
    messaging_level : int, optional
        Controls the amount of information returned to user. Varies by model.
    """

    def __init__(self, *args,
                 stge_kind=None,
                 parameters_solver=None,
                 completed_cycles=0,
                 **kwds):
        if stge_kind is None:
            stge_kind = {'iter_status': 'not initialized'}
        self.E_Next_ = Nexspectations()  # Next given this period/stage choices
        self.Ante_E_ = Prospectations()  # Before this stage shocks/choices
        self.Pars = Parameters()
        self.Bilt = Built()
        self.Bilt.completed_cycles = completed_cycles
        self.Bilt.stge_kind = stge_kind
        self.Bilt.parameters_solver = parameters_solver
        self.Modl = Modelements()
        self.Modl.Transitions = {}

    def define_transitions_possible(self):
        # Below: Transition types in their canonical possible order

        # These equations should be used by both solution and simulation code

        # We use an OrderedDict so that the Monte Carlo simulation machinery
        # will know the order in which they should be executed
        # (even though the Spyder variable explorer seems, confusingly, to be
        # unable to show an OrderedDict in its intrinsic order)

        # The name of each equation corresponds to a variable that will be made
        # and could be preserved in the simulation (or not)

        # Each status will end up transiting only to one subsequent status
        # ("status" = set of state variables with associated decision problem)
        # There are multiple possibilities because models may skip many steps

        # Steps can be skipped when the model is one in which nothing happens
        # in that step, or what happens is so simple that it can be directly
        # captured in a transition equation

        # For example, with no end of stage shocks, you could go directly from
        # the "chosen" (after choice) status to the "next_BOP" status

        # Equations that define transitions that affect agent's state
        transitions_possible = {  # BOP: Beginnning of Problem
            'BOP_to_choice': {},
            'choice_to_chosen': {},  # or
            'choice_to_next_BOP': {},
            'chosen_to_EOP': {},  # or
            'chosen_to_next_BOP': {},  # or
            'chosen_to_next_choice': {},
            'EOP_to_next_BOP': {},  # or
            'EOP_to_next_choice': {},  # EOP: End of Problem/Period
        }

        return transitions_possible

    def describe_model_and_calibration(self, messaging_level, quietly):
        pass


class Built(SimpleNamespace):
    """Objects built by solvers during course of solution."""

    pass


class Parameters(SimpleNamespace):
    """Parameters (as passed, and exposed). But not modified."""

    pass


class Nexspectations(SimpleNamespace):
    """Expectations about future period after current decisions."""

    pass


class Prospectations(SimpleNamespace):
    """Expectations prior to the realization of current period shocks."""

    pass


class ValueFunctions(SimpleNamespace):
    """Expectations across realization of stochastic shocks."""

    pass


class Modelements(SimpleNamespace):
    """Elements of the model in python/HARK code."""

    pass


__all__ = [
    "ConsumerSolutionOlder",
    "ConsumerSolution",
    "ConsumerSolutionOneNrmStateCRRA",
    "ConsPerfForesightSolver",
    "ConsIndShockSetup",
    "ConsIndShockSolverBasic",
    "ConsIndShockSolver",
    "ConsIndShockSetup",
]


# ConsumerSolution does nothing except add agent_stage_solution
# content to original ConsumerSolutionOlder, and set distance_criteria to cFunc

class ConsumerSolution(ConsumerSolutionOlder, agent_stage_solution):
    __doc__ = ConsumerSolutionOlder.__doc__
    __doc__ += """
    In addition, it inherits the attributes of agent_stage_solution.
    """
    # CDC 20210426:
    # vPfunc is unbounded so seems a bad choice for distance; here we change
    # to cFunc but doing so will require recalibrating some of our tests
    #  distance_criteria = ["vPfunc"]  # Bad b/c vP(0)=inf; should use cFunc
    #  distance_criteria = ["vFunc.dm"]  # Bad b/c vP(0)=inf; should use cFunc
    #  distance_criteria = ["mTrgNrm"]  # mTrgNrm better choice if GICNrm holds
    distance_criteria = ["cFunc"]  # cFunc if the GICRaw fails

    def __init__(self, *args,
                 # TODO: New items below should be in default ConsumerSolution
                 stge_kind=None,
                 completed_cycles=0,
                 parameters_solver=None,
                 **kwds):
        ConsumerSolutionOlder.__init__(self, **kwds)
        agent_stage_solution.__init__(self, *args, **kwds)
        if stge_kind is None:
            stge_kind = dict(iter_status='not initialized')


# noinspection PyTypeChecker
class ConsumerSolutionOneNrmStateCRRA(ConsumerSolution):
    """
    ConsumerSolution with CRRA utility and geometric discounting.

    Specializes the generic ConsumerSolution object to the case with:
        * Constant Relative Risk Aversion (CRRA) utility
        * Geometric Discounting of Time Separable Utility
        * Normalizing Growth Factors for "Permanent Income"
          - At the individual and aggregate levels
        * A (potentially nonzero) Probability of Mortality

    and a standard budget constraint involving capital income with a riskfree
    rate of return, and noncapital income that grows by a permanent factor
    (which is used to normalize the problem).

    The model is specified in such a way that it can accommodate either the
    perfect foresight case or the case where noncapital income is subject
    to transitory and permanent shocks, and can accommodate an artificial
    borrowing constraint specified as a proportion of permanent noncapital
    income.

    The class can test for various restrictions on the parameter values of the
    model. A suite of minimal restrictions (like, the time preference factor
    must be nonnegative) is always evaluated. A set of conditions that
    determine infinite horizon charactieristics of the solution can be
    evaluated using the `check_conditions` method.  (Further information about
    the conditions can be found in the documentation for that method.)  For
    convenience, we repeat below the documentation for the parent
    ConsumerSolution of this class, all of which applies here.

    Parameters
    ----------
    DiscFac : float
        Pure intertemporal discount factor for future utility.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the next period.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period.
    """
    __doc__ += ConsumerSolution.__doc__

    def __init__(self, *args,
                 CRRA=2., DiscFac=0.96, LivPrb=1.0, Rfree=1.03, PermGroFac=1.0,
                 **kwds):
        ConsumerSolution.__init__(self, *args, **kwds)

        self.Pars.DiscFac = DiscFac
        self.Pars.LivPrb = LivPrb
        self.Pars.CRRA = CRRA
        self.Pars.Rfree = Rfree
        self.Pars.PermGroFac = PermGroFac

        # These have been moved to Bilt to declutter whiteboard:
        del self.hNrm
        del self.vPfunc
        del self.vPPfunc
        del self.MPCmax
        del self.MPCmin

        self.Bilt.transitions_possible = self.define_transitions_possible()

    def define_transitions_possible(self):
        """
        Construct dictionary containing the possible transition equations.

        Returns
        -------
        possible_transitions : dict
            Names and definitions of the transitions possible within this
        solution object or as an exit from it to the beginning of the next
        object.

        """
        # Working backwards from End of Problem

        # Later we can select among the allowed transitions
        # First, for model with shocks at Beginning of Problem/Period (BOP):
        chosen_to_next_BOP = \
            {'kNrm': 'kNrm = aNrm'}  # k_{t+1} = a_{t}

        choice_to_chosen = \
            {'aNrm': 'aNrm = mNrm - cNrm'}  # a_{t} = m_{t} - c_{t}

        BOP_to_choice = {
            'RNrm': 'RNrm = Rfree / (PermGroFac * PermShk)',
            'bNrm': 'bNrm = kNrm * RNrm',
            'yNrm': 'yNrm = TranShk',
            'mNrm': 'mNrm = bNrm + yNrm'}

        # Now, for model with shocks at End of Problem/Period (EOP)

        chosen_to_next_choice = \
            {'kNrm': 'kNrm = aNrm',
             'RNrm': 'RNrm = Rfree / (PermGroFac * PermShk)',
             'bNrm': 'bNrm = kNrm * RNrm',  # b_{t} = k_{t} RNrm_{t}
             'yNrm': 'yNrm = TranShk',  # y_{t} = \TranShk_{t}
             'mNrm': 'mNrm = bNrm + yNrm'}

        # choice_to_chosen need not be redefined b/c same as defined above

        possible_transitions = \
            {'chosen_to_next_BOP': chosen_to_next_BOP,
             'chosen_to_next_choice': chosen_to_next_choice,
             'choice_to_chosen': choice_to_chosen,
             'BOP_to_choice': BOP_to_choice
             }

        return possible_transitions

    def describe_model_and_calibration(self, messaging_level=logging.INFO,
                                       quietly=False):
        """
        Log a brief description of the model and its calibration.

        Parameters
        ----------
        self : agent_stage_solution

            Solution to the problem described by information for the current
        stage found in Bilt and the succeeding stage.

        quietly : boolean, optional

            If true, suppresses all output

        Returns
        -------
        None
        """

        crnt = self
        Pars, Modl = crnt.Pars, crnt.Modl
        Tran = Modl.Transitions

        if not quietly and messaging_level < logging.WARNING:
            msg = '\n(quietly=False and messaging_level < logging.WARNING, ' \
                + 'so some model information is provided below):\n'
            msg = msg + '\nThe model has the following parameter values:\n'
            _log.setLevel(messaging_level)
            _log.info(msg)
            for key in Pars.__dict__.keys():
                _log.info('\t' + key + ': ' + str(Pars.__dict__[key]))

            msg = "\nThe model's transition equations are:"
            _log.info(msg)
            for key in Tran.keys():
                _log.info('\n' + key + ' step:')
                for eqn_name in Tran[key]['raw_text']:
                    _log.info('\t' + str(Tran[key]['raw_text'][eqn_name]))

    def check_conditions(self, messaging_level=logging.DEBUG, quietly=False):
        """
        Check whether parameters satisfy some possibly interesting conditions.

        ==================================================================
        Acronym        Condition
        ==================================================================
        AIC           Absolute Impatience Condition
        RIC           Return Impatience Condition
        GICRaw        Growth Impatience Condition (not mortality adjusted)
        GICNrm        GIC adjusted for uncertainty in permanent income
        GICLiv        GIC adjusted for constant probability of mortality
        GICLivMod     Aggregate GIC assuming Modigliani bequests
        FHWC          Finite Human Wealth Condition
        WRIC          Weak Return Impatience Condition
        FVAC          Finite Value of Autarky Condition
        ==================================================================

        Depending on the configuration of parameter values, some combination of
        these conditions must be satisfied in order for the problem to have a
        nondegenerate solution. To check which conditions are required, in 
        the verbose mode, a reference to the relevant theoretical literature
        is made.

        Parameters
        ----------
        self : agent_stage_solution

            Solution to the problem described by information for the current
        stage found in Bilt and the succeeding stage.

        messaging_level : int, optional

            Controls verbosity of messages. logging.DEBUG is most verbose,
            logging.INFO is less verbose, logging.WARNING indicates possible
            problems, logging.CRITICAL indicates it is degenerate.

        quietly : boolean, optional

            If true, performs calculations but prints no results

        Returns
        -------
        None
        """
        crnt = self  # A current solution object

        Bilt, Pars = crnt.Bilt, crnt.Pars

        Bilt.conditions = {}  # Keep track of truth of conditions
        Bilt.degenerate = False  # True: solution is degenerate

        self.describe_model_and_calibration(messaging_level, quietly)
        if not quietly:
            _log.info('\n\nBecause messaging_level is >= logging.INFO, ' +
                      'infinite horizon conditions are reported below:\n')
        crnt.check_AIC(crnt, messaging_level, quietly)
        crnt.check_FHWC(crnt, messaging_level, quietly)
        crnt.check_RIC(crnt, messaging_level, quietly)
        crnt.check_GICRaw(crnt, messaging_level, quietly)
        crnt.check_GICNrm(crnt, messaging_level, quietly)
        crnt.check_GICLiv(crnt, messaging_level, quietly)
        crnt.check_GICLivMod(crnt, messaging_level, quietly)
        crnt.check_WRIC(crnt, messaging_level, quietly)
        crnt.check_FVAC(crnt, messaging_level, quietly)

        # degenerate flag is True if the model has no nondegenerate solution
        if hasattr(Bilt, "BoroCnstArt") and Pars.BoroCnstArt is not None:
            if Bilt.FHWC:
                Bilt.degenerate = not Bilt.RIC  # h finite & patient => c(m)=0
            # If BoroCnstArt exists but RIC fails, limiting soln is c(m)=0
        else:  # No BoroCnst; not degenerate if neither c(m)=0 or \infty
            if Bilt.FHWC:
                Bilt.degenerate = not Bilt.RIC  # Finite h requires finite Pat
            else:
                Bilt.degenerate = Bilt.RIC  # infinite h requires impatience

        if Bilt.degenerate:
            _log.critical("Under the given parameter values, " +
                          "the model is degenerate.")

    def check_AIC(self, soln, messaging_level=logging.DEBUG, quietly=False):
        """Evaluate and report on the Absolute Impatience Condition."""
        name = "AIC"

        def test(soln): return soln.Bilt.APFac < 1

        messages = {
            True: f"\nThe Absolute Patience Factor, APFac={soln.Bilt.APFac:.5f} satisfies the Absolute Impatience Condition (AIC), APFac < 1:\n    " +
                  soln.Bilt.AIC_fcts['urlhandle'],
            False: f"\nThe Absolute Patience Factor, APFac={soln.Bilt.APFac:.5f} violates the Absolute Impatience Condition (AIC), APFac < 1:\n    " +
                   soln.Bilt.AIC_fcts['urlhandle']
        }
        verbose_messages = {
            True: "\n    Because the APFac < 1,  the absolute amount of consumption is expected to fall over time.  \n",
            False: "\n    Because the APFac > 1, the absolute amount of consumption is expected to grow over time.  \n",
        }

        soln.Bilt.AIC = core_check_condition(name, test, messages, messaging_level,
                                             verbose_messages, "APFac", soln, quietly)

    # noinspection PyTypeChecker
    def check_FVAC(self, soln, messaging_level=logging.DEBUG, quietly=False):
        """Evaluate and report on the Finite Value of Autarky Condition."""
        name = "FVAC"

        def test(soln): return soln.Bilt.VAFac < 1

        messages = {
            True: f"\nThe Finite Value of Autarky Factor, VAFac={soln.Bilt.VAFac:.5f} satisfies the Finite Value of Autarky Condition, VAFac < 1:\n    " +
                  soln.Bilt.FVAC_fcts['urlhandle'],
            False: f"\nThe Finite Value of Autarky Factor, VAFac={soln.Bilt.VAFac:.5f} violates the Finite Value of Autarky Condition, VAFac:\n    " +
                   soln.Bilt.FVAC_fcts['urlhandle']
        }
        verbose_messages = {
            True: "\n    Therefore, a nondegenerate solution exists if the RIC also holds. (" + soln.Bilt.FVAC_fcts[
                'urlhandle'] + ")\n",
            False: "\n    Therefore, a nondegenerate solution exits if the RIC holds, but will not exist if the RIC fails unless the FHWC also fails.\n",
        }

        # This is bad enough to report as a warning
        if messaging_level == logging.WARNING and quietly is False \
           and soln.Bilt.VAFac > 1:
            _log.warning(messages['False']+verbose_messages['False'])

        soln.Bilt.FVAC = core_check_condition(name, test, messages, messaging_level,
                                              verbose_messages, "VAFac", soln, quietly)

    def check_GICRaw(self, soln, messaging_level=logging.DEBUG, quietly=False):
        """Evaluate and report on the Growth Impatience Condition (GICRaw)."""
        name = "GICRaw"

        def test(soln): return soln.Bilt.GPFacRaw < 1

        messages = {
            True: f"\nThe Growth Patience Factor, GPF={soln.Bilt.GPFacRaw:.5f} satisfies the Growth Impatience Condition (GICRaw), GPF < 1:\n    " +
                  soln.Bilt.GICRaw_fcts['urlhandle'],
            False: f"\nThe Growth Patience Factor, GPF={soln.Bilt.GPFacRaw:.5f} violates the Growth Impatience Condition (GICRaw), GPF < 1:\n    " +
                   soln.Bilt.GICRaw_fcts['urlhandle'],
        }
        verbose_messages = {
            True: "\n    Therefore, for a perfect foresight consumer, the ratio of individual wealth to permanent income is expected to fall indefinitely.    \n",
            False: "\n    Therefore, for a perfect foresight consumer whose parameters satisfy the FHWC, the ratio of individual wealth to permanent income is expected to rise toward infinity. \n"
        }
        soln.Bilt.GICRaw = core_check_condition(name, test, messages, messaging_level,
                                                verbose_messages, "GPFacRaw", soln, quietly)

        # Give them a warning if the model does not satisfy the GICRaw,
        # even if they asked to solve quietly, unless messaging_level is CRITICAL
        if quietly is True:  # Otherwise they will get the info anyway
            if soln.Bilt.GPFacRaw > 1:
                if messaging_level <= logging.CRITICAL:
                    _log.warning(messages[False]+verbose_messages[False])

    def check_GICLiv(self, soln, messaging_level=logging.DEBUG, quietly=False):
        """Evaluate and report on Mortality Adjusted GIC (GICLiv)."""
        name = "GICLiv"

        def test(soln): return soln.Bilt.GPFacLiv < 1

        messages = {
            True: f"\nThe Mortality Adjusted Growth Patience Factor, GPFacLiv={soln.Bilt.GPFacLiv:.5f} satisfies the Mortality Adjusted Growth Impatience Condition (GICLiv):\n    " +
                  soln.Bilt.GPFacLiv_fcts['urlhandle'],
            False: f"\nThe Mortality Adjusted Growth Patience Factor, GPFacLiv={soln.Bilt.GPFacLiv:.5f} violates the Mortality Adjusted Growth Impatience Condition (GICLiv):\n    " +
                   soln.Bilt.GPFacLiv_fcts['urlhandle'],
        }
        verbose_messages = {
            True: "\n    Therefore, a target level of the ratio of aggregate market resources to aggregate permanent income exists.    \n" +
                  soln.Bilt.GPFacLiv_fcts['urlhandle'] + "\n",
            False: "\n    Therefore, a target ratio of aggregate resources to aggregate permanent income may not exist.  \n" +
                   soln.Bilt.GPFacLiv_fcts['urlhandle'] + "\n",
        }
        soln.Bilt.GICLiv = core_check_condition(name, test, messages, messaging_level,
                                                verbose_messages, "GPFacLiv", soln, quietly)

        # This is important enough to warn them even if quietly == True; unless messaging_level = CRITICAL
        if (soln.Bilt.GICLiv == np.False_) and (quietly is True) and (messaging_level < logging.CRITICAL):
            _log.warning(messages[False]+verbose_messages[False])

    def check_GICLivMod(self, soln, messaging_level=logging.DEBUG, quietly=False):
        """Evaluate and report on Mortality Adjusted GIC (GICLivMod)."""
        name = "GICLivMod"

        def test(soln): return soln.Bilt.GPFacLivMod < 1

        messages = {
            True: f"\nThe Mortality Adjusted Growth Patience Factor, GPFacLivMod={soln.Bilt.GPFacLivMod:.5f} satisfies the Mortality Adjusted Growth Impatience Condition (GICLivMod):\n    " +
                  soln.Bilt.GPFacLivMod_fcts['urlhandle'],
            False: f"\nThe Mortality Adjusted Growth Patience Factor, GPFacLivMod={soln.Bilt.GPFacLivMod:.5f} violates the Mortality Adjusted Growth Impatience Condition (GICLivMod):\n    " +
                   soln.Bilt.GPFacLivMod_fcts['urlhandle'],
        }
        verbose_messages = {
            True: "\n    Therefore, a target level of the ratio of aggregate market resources to aggregate permanent income exists.    \n" +
                  soln.Bilt.GPFacLivMod_fcts['urlhandle'] + "\n",
            False: "\n    Therefore, a target ratio of aggregate resources to aggregate permanent income may not exist.  \n" +
                   soln.Bilt.GPFacLivMod_fcts['urlhandle'] + "\n",
        }
        soln.Bilt.GICLivMod = core_check_condition(name, test, messages, messaging_level,
                                                verbose_messages, "GPFacLivMod", soln, quietly)

        # This is important enough to warn them even if quietly == True; unless messaging_level = CRITICAL
        if (soln.Bilt.GICLivMod == np.False_) and (quietly is True) and (messaging_level < logging.CRITICAL):
            _log.warning(messages[False]+verbose_messages[False])

    def check_RIC(self, soln, messaging_level=logging.DEBUG, quietly=False):
        """Evaluate and report on the Return Impatience Condition."""
        name = "RIC"

        def test(soln): return soln.Bilt.RPFac < 1

        messages = {
            True: f"\nThe Return Patience Factor, RPFac={soln.Bilt.RPFac:.5f} satisfies the Return Impatience Condition (RIC), RPFac < 1:\n    " +
                  soln.Bilt.RPFac_fcts['urlhandle'],
            False: f"\nThe Return Patience Factor, RPFac={soln.Bilt.RPFac:.5f} violates the Return Impatience Condition (RIC), RPFac < 1:\n    " +
                   soln.Bilt.RPFac_fcts['urlhandle'],
        }
        verbose_messages = {
            True: "\n    Therefore, the limiting consumption function is not c(m)=0 for all m\n",
            False: "\n    Therefore, if the FHWC is satisfied, the limiting consumption function is c(m)=0 for all m.\n",
        }
        soln.Bilt.RIC = core_check_condition(name, test, messages, messaging_level,
                                             verbose_messages, "RPFac", soln, quietly)

    def check_FHWC(self, soln, messaging_level=logging.DEBUG, quietly=False):
        """Evaluate and report on the Finite Human Wealth Condition."""
        name = "FHWC"

        def test(soln): return soln.Bilt.FHWFac < 1

        messages = {
            True: f"\nThe Finite Human Wealth Factor, FHWFac={soln.Bilt.FHWFac:.5f} satisfies the Finite Human Wealth Condition (FHWC), FHWFac < 1:\n    " +
                  soln.Bilt.FHWC_fcts['urlhandle'],
            False: f"\nThe Finite Human Wealth Factor, FHWFac={soln.Bilt.FHWFac:.5f} violates the Finite Human Wealth Condition (FHWC), FHWFac < 1:\n    " +
                   soln.Bilt.FHWC_fcts['urlhandle'],
        }
        verbose_messages = {
            True: f"\n    Therefore, the limiting consumption function is not c(m)=Infinity.\n\n    Human wealth normalized by permanent income is {soln.Bilt.hNrmInf:.5f}.\n",
            False: "\n    Therefore, the limiting consumption function is c(m)=Infinity for all m unless the RIC is also violated.\n  If both FHWC and RIC fail and the consumer faces a liquidity constraint, the limiting consumption function is nondegenerate but has a limiting slope of 0. (" +
                   soln.Bilt.FHWC_fcts['urlhandle'] + ")\n",
        }
        soln.Bilt.FHWC = core_check_condition(name, test, messages, messaging_level,
                                              verbose_messages, "FHWFac", soln, quietly)

        if (messaging_level < logging.CRITICAL) and (soln.Bilt.FHWC == np.False_):
            _log.info(messages[False]+verbose_messages[False])

    def check_GICNrm(self, soln, messaging_level=logging.DEBUG, quietly=False):
        """Check Normalized Growth Patience Factor."""
        if not hasattr(soln.Pars, 'IncShkDstn'):
            return  # GICNrm is same as GIC for PF consumer

        name = "GICNrm"

        def test(soln): return soln.Bilt.GPFacNrm <= 1

        messages = {
            True: f"\nThe Normalized Growth Patience Factor GPFacNrm, GPFacNrm={soln.Bilt.GPFacNrm:.5f} satisfies the Normalized Growth Impatience Condition (GICNrm), GPFacNrm < 1:\n    " +
                  soln.Bilt.GICNrm_fcts['urlhandle'],
            False: f"\nThe Normalized Growth Patience Factor GPFacNrm, GPFacNrm={soln.Bilt.GPFacNrm:.5f} violates the Normalized Growth Impatience Condition (GICNrm), GPFacNrm < 1:\n    " +
                   soln.Bilt.GICNrm_fcts['urlhandle'],
        }
        verbose_messages = {
            True: "\n    Therefore, a target level of the individual market resources ratio m exists.",
            False: "\n    Therefore, a target ratio of individual market resources to individual permanent income does not exist.  \n"
        }

        soln.Bilt.GICNrm = \
            core_check_condition(name, test, messages, messaging_level,
                                 verbose_messages, "GPFacNrm", soln, quietly)

        # Warn them their model does not satisfy the GICNrm even if they asked
        # for the "quietly" solution -- unless they said "only CRITICAL"

        if (messaging_level < logging.CRITICAL) and (soln.Bilt.GICNrm == np.False_):
            _log.info(messages[False]+verbose_messages[False])

    def check_WRIC(self, soln, messaging_level=logging.DEBUG, quietly=False):
        """Evaluate and report on the Weak Return Impatience Condition."""
        if not hasattr(soln, 'IncShkDstn'):
            return  # WRIC is same as RIC for PF consumer

        name = "WRIC"

        def test(soln): return soln.Bilt.WRPFac <= 1

        messages = {
            True: f"\nThe Weak Return Patience Factor, WRPFac={soln.Bilt.WRPFac:.5f} satisfies the Weak Return Impatience Condition, WRPFac < 1:\n    " +
                  soln.Bilt.WRIC_fcts['urlhandle'],
            False: f"\nThe Weak Return Patience Factor, WRPFac={soln.Bilt.WRPFac:.5f} violates the Weak Return Impatience Condition, WRPFac < 1:\n    " +
                   soln.Bilt.WRIC_fcts['urlhandle'],
        }

        verbose_messages = {
            True: "\n    Therefore, a nondegenerate solution exists if the FVAC is also satisfied. (" +
                  soln.Bilt.WRIC_fcts['urlhandle'] + ")\n",
            False: "\n    Therefore, a nondegenerate solution is not available (" + soln.Bilt.WRIC_fcts[
                'urlhandle'] + ")\n",
        }
        soln.Bilt.WRIC = core_check_condition(
            name, test, messages, messaging_level, verbose_messages, "WRPFac", soln, quietly)

    def mTrgNrm_find(self):
        """
        Find value of m at which individual consumer expects m not to change.

        This will exist if the GICNrm holds.

        https://econ-ark.github.io/BufferStockTheory#UniqueStablePoints

        Returns
        -------
            The target value mTrgNrm.
        """
        m_init_guess = self.Bilt.mNrmMin + self.E_Next_.IncNrmNxt
        try:  # Find value where argument is zero
            self.Bilt.mTrgNrm = find_zero_newton(
                self.E_Next_.m_tp1_minus_m_t,
                m_init_guess)
        except:
            self.Bilt.mTrgNrm = None

        return self.Bilt.mTrgNrm

    def mBalLvl_find(self):
        """
        Find pseudo Steady-State Equilibrium (normalized) market resources m.

        This is the m at which the consumer
        expects level of market resources M to grow at same rate as the level
        of permanent income P.

        This will exist if the GICRaw holds.

        https://econ-ark.github.io/BufferStockTheory#UniqueStablePoints

        Parameters
        ----------
        self : ConsumerSolution
            Solution to this period's problem, which must have attribute cFunc.

        Returns
        -------
        self : ConsumerSolution
            Same solution that was passed, but now with attribute mBalLvl.
        """
        # Minimum market resources plus E[next income] is okay starting guess

        m_init_guess = self.Bilt.mNrmMin + self.E_Next_.IncNrmNxt

        try:
            self.Bilt.mBalLvl = find_zero_newton(
                self.E_Next_.PermShk_tp1_times_m_tp1_Over_m_t_minus_PermGroFac, m_init_guess)
        except:
            self.Bilt.mBalLvl = None

        # Add mBalLvl to the solution and return it
        return self.Bilt.mBalLvl

    def mBalLog_find(self):
        """
        Find mBalLog where expected growth in log mLvl matches growth in log pLvl

        This is the m at which the consumer expects log of market resources  
        to grow at same rate as the log of permanent income

        This will exist if the GICRaw holds.

        Parameters
        ----------
        self : ConsumerSolution
            Solution to this period's problem, which must have attribute cFunc.

        Returns
        -------
        self : ConsumerSolution
            Same solution that was passed, but now with attribute mBalLvl.
        """
        # Minimum market resources plus E[next income] is okay starting guess

        m_init_guess = self.Bilt.mNrmMin + self.E_Next_.IncNrmNxt

        try:
            self.Bilt.mBalLog = find_zero_newton(
                self.E_Next_.mLog_tp1_from_a_t(
                    self.Bilt.cFunc
                    , m_init_guess))
        except:
            self.Bilt.mBalLog = None

        # Add mBalLog to the solution and return it
        return self.Bilt.mBalLog


# Until this point, our objects have been "solution" not "solver" objects.  To
# a "solution" object, "solver" objects add the tools that can generate a
# solution of that kind.  Eventually we aim to have a clear delineation between
# models objects, solution objects, and solvers, so that a model can be
# specified independent of a solution method, a solution can be specified
# independent of a model or solution method, and a solution method can be
# specified that can solve many different models.

# As a first step in that direction, solver classes do not __init__ themselves
# with the __init__ method of their "parent" solution class.  Instead, they
# expect to receive as an argument an instance of a solution object called
# solution_next, and they will construct a solution_current object that begins
# empty and onto which the information describing the model and its solution
# are added step by step.


class ConsPerfForesightSolver(ConsumerSolutionOneNrmStateCRRA):
    """
    Solve (one period of) perfect foresight CRRA utility consumer problem.

    Augments ConsumerSolutionOneNrmStateCRRA_PF solution object with methods
    able to solve a perfect foresight model with those characteristics,
    allowing for a bequest motive and either a natural or an artificial
    borrowing constraint.

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

    # CDC 20200426: MaxKinks adds a lot of complexity to no evident purpose
    # because everything it accomplishes could be done solving a finite horizon
    # model (including tests of convergence conditions, which can be invoked
    # manually if a user wants them).
    def __init__(self, solution_next, DiscFac=0.96, LivPrb=1.0, CRRA=2.0,
                 Rfree=1.0, PermGroFac=1.0, BoroCnstArt=None,
                 MaxKinks=None, solverType='HARK', solveMethod='EGM',
                 eventTiming='EOP', horizon='infinite', *args,
                 **kwds):

        folw = self.solution_follows = solution_next  # abbreviation

        # Do NOT __init__ as a ConsumerSolutionOneNrmStateCRRA object
        # even though that is the parent class
        # Because this is a SOLVER (which, admittedly, at present, can only
        # handle that class). But in principle, this is a solver, not a
        # solution

        # In principle, a solution need not know its solver
        crnt = self.solution_current = \
            ConsumerSolutionOneNrmStateCRRA(
                self, DiscFac, LivPrb, CRRA, Rfree, PermGroFac, eventTiming)

        # Get solver parameters and store for later use
        # omitting things that are not needed

        Pars = crnt.Pars
        Pars.__dict__.update(
            {k: v for k, v in {**kwds, **locals()}.items()
             if k not in {'self', 'solution_next', 'kwds', 'solution_follows',
                          'folw', 'crnt', 'Pars'}})

        # 'terminal' solution should replace pseudo_terminal:
        if hasattr(folw.Bilt, 'stge_kind') and \
                folw.Bilt.stge_kind['iter_status'] == 'terminal_partial':
            crnt.Bilt = deepcopy(folw.Bilt)

        # links for docs; urls are used when "fcts" are added
        self._url_doc_for_solver_get()

        return

    def _url_doc_for_solver_get(self):
        # Generate a url that will locate the documentation
        self.class_name = self.__class__.__name__
        self.solution_current.Bilt.url_ref = self.url_ref = \
            "https://econ-ark.github.io/BufferStockTheory"
        self.solution_current.Bilt.urlroot = self.urlroot = \
            self.url_ref + '/#'
        self.solution_current.Bilt.url_doc = self.url_doc = \
            "https://hark.readthedocs.io/en/latest/search.html?q=" + \
            self.class_name + "&check_keywords=yes&area=default#"

    def from_chosen_states_make_continuation_E_Next_(self, crnt):
        """
        Construct expectations of useful objects from post-choice status.

        Parameters
        ----------
        crnt : agent_stage_solution
            The solution to the problem without the expectations info.

        Returns
        -------
        crnt : agent_stage_solution
            The given solution, with the relevant namespaces updated to
        contain the constructed info.
        """
        self.build_facts_infhor()
        crnt = self.build_facts_recursive()

        # Reduce cluttered formulae with local aliases
        E_Next_, tp1 = crnt.E_Next_, self.solution_follows
        Bilt, Pars = crnt.Bilt, crnt.Pars
        Rfree, PermGroFac, DiscFacLiv = Pars.Rfree, Pars.PermGroFac, Bilt.DiscFacLiv

        CRRA = tp1.vFunc.CRRA

        # Omit first and last pts; they define extrapo below and above kinks
        Bilt.mNrm_kinks_tp1 = mNrm_kinks_tp1 = tp1.cFunc.x_list[:-1][1:]
        Bilt.cNrm_kinks_tp1 = cNrm_kinks_tp1 = tp1.cFunc.y_list[:-1][1:]
        Bilt.vNrm_kinks_tp1 = vNrm_kinks_tp1 = tp1.vFunc(mNrm_kinks_tp1)

        # Calculate end-of-this-period aNrm vals that would reach those mNrm's
        # There are no shocks in the PF model, so TranShkMin = TranShk = 1.0
        bNrm_kinks_tp1 = (mNrm_kinks_tp1 - tp1.Bilt.TranShkMin)
        aNrm_kinks = bNrm_kinks_tp1 * (PermGroFac / Rfree)

        crnt.Bilt.aNrmGrid = aNrm_kinks

        # Level and first derivative of expected value from aNrmGrid points
        v_0 = DiscFacLiv * \
            PermGroFac ** (1 - CRRA) * vNrm_kinks_tp1
        v_1 = DiscFacLiv * \
            PermGroFac ** (0 - CRRA) * tp1.Bilt.u.dc(cNrm_kinks_tp1) * Rfree

        c_0 = cNrm_kinks_tp1

        E_Next_.given_shocks = np.array([v_0, v_1, c_0])

        # Make positions of precomputed objects easy to reference
        E_Next_.v0_pos, E_Next_.v1_pos = 0, 1
        E_Next_.c0_pos = 3

        return crnt

    def make_cFunc_PF(self):
        """
        Make (piecewise linear) consumption function for this period.

        See PerfForesightConsumerType.ipynb notebook for derivations.
        """
        # Reduce cluttered formulae with local aliases
        crnt, tp1 = self.solution_current, self.solution_follows
        Bilt, Pars, E_Next_ = crnt.Bilt, crnt.Pars, crnt.E_Next_
        Rfree, PermGroFac, MPCmin = Pars.Rfree, Pars.PermGroFac, Bilt.MPCmin

        BoroCnstArt, DiscFacLiv, BoroCnstNat = \
            Pars.BoroCnstArt, Bilt.DiscFacLiv, Bilt.BoroCnstNat

        u, u.Nvrs, u.dc.Nvrs = Bilt.u, Bilt.u.Nvrs, Bilt.u.dc.Nvrs
        CRRA_tp1 = tp1.Bilt.vFunc.CRRA

        # define yNrm_tp1 to make formulas below easier to read
        yNrm_tp1 = tp1.Bilt.TranShkMin  # in PF model TranShk[Min,Max] = 1.0

        if BoroCnstArt is None:
            BoroCnstArt = -np.inf

        # Whichever constraint is tighter is the relevant one
        BoroCnst = max(BoroCnstArt, BoroCnstNat)

        # Translate t+1 constraints into their consequences for t
        # Endogenous Gridpoints steps:
        # c today yielding u' equal to discounted u' from each kink in t+1
        cNrm_kinks = crnt.Bilt.u.dc.Nvrs(E_Next_.given_shocks[E_Next_.v1_pos])
        mNrm_kinks = Bilt.aNrmGrid + cNrm_kinks  # Corresponding m

        # Corresponding value and inverse value
        vNrm_kinks = E_Next_.given_shocks[E_Next_.v0_pos]
        vInv_kinks = u.Nvrs(vNrm_kinks)

        # vAdd used later to add some new points; should add zero to existing
        vAdd_kinks = mNrm_kinks - mNrm_kinks  # makes zero array of useful size

        mNrmMin_tp1 = yNrm_tp1 + BoroCnst * (Rfree / PermGroFac)

        # by t_E_ here we mean "discounted back to period t"
        t_E_v_tp1_at_mNrmMin_tp1 = \
            (DiscFacLiv * PermGroFac ** (1 - CRRA_tp1) * tp1.vFunc(mNrmMin_tp1))

        t_E_v1_tp1_at_mNrmMin_tp1 = \
            (((Rfree * DiscFacLiv) * (PermGroFac ** (-CRRA_tp1))
              ) * tp1.vFunc.dm(mNrmMin_tp1))

        # Explanation:
        # Define h as the 'horizon': h_t(m_t) is the number of periods it will take
        # before you hit the constraint, after which you remain constrained

        # The maximum h in a finite horizon model
        # is the remaining number of periods of life: h = T - t

        # If the consumer is sufficiently impatient, there will be levels of
        # m from which the optimal plan will be to run down existing wealth
        # over some horizon less than T - t, and for the remainder of the
        # horizon to set consumption equal to income

        # In a given period t, it will be optimal to spend all resources
        # whenever the marginal utility of doing so exceeds the marginal
        # utility yielded by receiving the minimum possible income next
        # period: u'(m_t) > (discounted) u'(c(y_{t+1}))

        # "cusp" is name for where current period constraint stops binding
        cNrm_cusp = u.dc.Nvrs(t_E_v1_tp1_at_mNrmMin_tp1)
        vNrm_cusp = Bilt.u(cNrm_cusp) + t_E_v_tp1_at_mNrmMin_tp1
        vAdd_cusp = t_E_v_tp1_at_mNrmMin_tp1
        vInv_cusp = u.Nvrs(vNrm_cusp)
        mNrm_cusp = cNrm_cusp + BoroCnst

        # cusp today vs today's implications of future constraints
        if mNrm_cusp >= mNrm_kinks[-1]:  # tighter than the tightest existing
            mNrm_kinks = np.array([mNrm_cusp])  # looser kinka are irrelevant
            cNrm_kinks = np.array([cNrm_cusp])  # forget about them all
            vNrm_kinks = np.array([vNrm_cusp])
            vInv_kinks = np.array([vInv_cusp])
            vAdd_kinks = np.array([vAdd_cusp])
        else:  # keep today's implications of future kinks that matter today
            first_reachable = np.where(mNrm_kinks >= mNrm_cusp)[0][-1]
            if first_reachable < mNrm_kinks.size - 1:  # Keep binding pts
                mNrm_kinks = mNrm_kinks[first_reachable:-1]
                cNrm_kinks = cNrm_kinks[first_reachable:-1]
                vInv_kinks = vInv_kinks[first_reachable:-1]
                vAdd_kinks = vAdd_kinks[first_reachable:-1]
            # Add the new kink introduced by today's constraint
            mNrm_kinks = np.insert(mNrm_kinks, 0, mNrm_cusp)
            cNrm_kinks = np.insert(cNrm_kinks, 0, cNrm_cusp)
            vNrm_kinks = np.insert(vNrm_kinks, 0, vNrm_cusp)

        # Add a point to construct cFunc, vFunc as PF solution beyond last kink
        mNrmGrid_unconst = np.append(mNrm_kinks, mNrm_kinks[-1] + 1)  # m + 1
        cNrmGrid_unconst = np.append(cNrm_kinks, cNrm_kinks[-1] + 1*MPCmin)
        mNrmGrid = np.append([BoroCnst], mNrmGrid_unconst)
        cNrmGrid = np.append(0., cNrmGrid_unconst)

        self.cFunc = self.solution_current.cFunc = Bilt.cFunc = \
            LinearInterp(mNrmGrid, cNrmGrid)

    def def_value(self):
        """
        Build value function and store results in Modl.value.

        Returns
        -------
        soln : solution object with value functions attached

        """
        return def_value_CRRA(self.solution_current,
                              self.solution_current.Pars.CRRA)

    def build_facts_infhor(self):
        """
        Calculate facts useful for characterizing infinite horizon models.

        Returns
        -------
        solution : ConsumerSolution
            The given solution, with the relevant namespaces updated to
        contain the constructed info.
        """
        crnt = self.solution_current  # current
        Bilt, Pars, E_Next_ = crnt.Bilt, crnt.Pars, crnt.E_Next_

        urlroot = Bilt.urlroot
        Bilt.DiscFacLiv = Pars.DiscFac * Pars.LivPrb
        # givens are not changed by calculations below; Bilt and E_Next_ are
        givens = {**Pars.__dict__}

        APFac_fcts = {
            'about': 'Absolute Patience Factor'
        }
        py___code = '((Rfree * DiscFacLiv) ** (1.0 / CRRA))'
        Bilt.APFac = APFac = \
            eval(py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        APFac_fcts.update({'latexexpr': r'\APFac'})
        APFac_fcts.update({'_unicode_': r'Þ'})
        APFac_fcts.update({'urlhandle': urlroot + 'APFacDefn'})
        APFac_fcts.update({'py___code': py___code})
        APFac_fcts.update({'value_now': APFac})
        Bilt.APFac_fcts = APFac_fcts

        AIC_fcts = {
            'about': 'Absolute Impatience Condition'
        }
        AIC_fcts.update({'latexexpr': r'\AIC'})
        AIC_fcts.update({'urlhandle': urlroot + 'AICDefn'})
        AIC_fcts.update({'py___code': 'test: APFac < 1'})
        Bilt.AIC_fcts = AIC_fcts

        RPFac_fcts = {
            'about': 'Return Patience Factor'
        }
        py___code = 'APFac / Rfree'
        Bilt.RPFac = RPFac = \
            eval(py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        RPFac_fcts.update({'latexexpr': r'\RPFac'})
        RPFac_fcts.update({'_unicode_': r'Þ_R'})
        RPFac_fcts.update({'urlhandle': urlroot + 'RPFacDefn'})
        RPFac_fcts.update({'py___code': py___code})
        RPFac_fcts.update({'value_now': RPFac})
        Bilt.RPFac_fcts = RPFac_fcts

        RIC_fcts = {
            'about': 'Growth Impatience Condition'
        }
        RIC_fcts.update({'latexexpr': r'\RIC'})
        RIC_fcts.update({'urlhandle': urlroot + 'RICDefn'})
        RIC_fcts.update({'py___code': 'test: RPFac < 1'})
        Bilt.RIC_fcts = RIC_fcts

        GPFacRaw_fcts = {
            'about': 'Growth Patience Factor'
        }
        py___code = 'APFac / PermGroFac'
        Bilt.GPFacRaw = GPFacRaw = \
            eval(py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        GPFacRaw_fcts.update({'latexexpr': r'\GPFacRaw'})
        GPFacRaw_fcts.update({'_unicode_': r'Þ_Φ'})
        GPFacRaw_fcts.update({'urlhandle': urlroot + 'GPFacRawDefn'})
        GPFacRaw_fcts.update({'py___code': py___code})
        GPFacRaw_fcts.update({'value_now': GPFacRaw})
        Bilt.GPFacRaw_fcts = GPFacRaw_fcts

        GICRaw_fcts = {
            'about': 'Growth Impatience Condition'
        }
        GICRaw_fcts.update({'latexexpr': r'\GICRaw'})
        GICRaw_fcts.update({'urlhandle': urlroot + 'GICRawDefn'})
        GICRaw_fcts.update({'py___code': 'test: GPFacRaw < 1'})
        Bilt.GICRaw_fcts = GICRaw_fcts

        GPFacLiv_fcts = {
            'about': 'Mortality-Adjusted Growth Patience Factor'
        }
        py___code = 'APFac * LivPrb / PermGroFac'
        Bilt.GPFacLiv = GPFacLiv = \
            eval(py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        GPFacLiv_fcts.update({'latexexpr': r'\GPFacLiv'})
        GPFacLiv_fcts.update({'urlhandle': urlroot + 'GPFacLivDefn'})
        GPFacLiv_fcts.update({'py___code': py___code})
        GPFacLiv_fcts.update({'value_now': GPFacLiv})
        Bilt.GPFacLiv_fcts = GPFacLiv_fcts

        GICLiv_fcts = {
            'about': 'Growth Impatience Condition'
        }
        GICLiv_fcts.update({'latexexpr': r'\GICLiv'})
        GICLiv_fcts.update({'urlhandle': urlroot + 'GICLivDefn'})
        GICLiv_fcts.update({'py___code': 'test: GPFacLiv < 1'})
        Bilt.GICLiv_fcts = GICLiv_fcts

        GPFacLivMod_fcts = {
            'about': 'Aggregate GPF with Modigliani Bequests'
        }
        py___code = 'GPFacLiv * LivPrb'
        Bilt.GPFacLivMod = GPFacLivMod = \
            eval(py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        GPFacLivMod_fcts.update({'latexexpr': r'\GPFacLivMod'})
        GPFacLivMod_fcts.update({'urlhandle': urlroot + 'GPFacLivModDefn'})
        GPFacLivMod_fcts.update({'py___code': py___code})
        GPFacLivMod_fcts.update({'value_now': GPFacLivMod})
        Bilt.GPFacLivMod_fcts = GPFacLivMod_fcts

        GICLivMod_fcts = {
            'about': 'Aggregate GIC with Modigliani Mortality'
        }
        GICLivMod_fcts.update({'latexexpr': r'\GICLivMod'})
        GICLivMod_fcts.update({'urlhandle': urlroot + 'GICLivModDefn'})
        GICLivMod_fcts.update({'py___code': 'test: GPFacLiv < 1'})
        Bilt.GICLivMod_fcts = GICLivMod_fcts

        RNrm_PF_fcts = {
            'about': 'Growth-Normalized PF Return Factor'
        }
        py___code = 'Rfree/PermGroFac'
        E_Next_.RNrm_PF = RNrm_PF = \
            eval(py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        RNrm_PF_fcts.update({'latexexpr': r'\PFRNrm'})
        RNrm_PF_fcts.update({'_unicode_': r'R/Φ'})
        RNrm_PF_fcts.update({'py___code': py___code})
        RNrm_PF_fcts.update({'value_now': RNrm_PF})
        E_Next_.RNrm_PF_fcts = RNrm_PF_fcts

        Inv_RNrm_PF_fcts = {
            'about': 'Inv of Growth-Normalized PF Return Factor'
        }
        py___code = '1 / RNrm_PF'
        E_Next_.Inv_RNrm_PF = Inv_RNrm_PF = \
            eval(py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        Inv_RNrm_PF_fcts.update({'latexexpr': r'\InvPFRNrm'})
        Inv_RNrm_PF_fcts.update({'_unicode_': r'Φ/R'})
        Inv_RNrm_PF_fcts.update({'py___code': py___code})
        Inv_RNrm_PF_fcts.update({'value_now': Inv_RNrm_PF})
        E_Next_.Inv_RNrm_PF_fcts = \
            Inv_RNrm_PF_fcts

        FHWFac_fcts = {
            'about': 'Finite Human Wealth Factor'
        }
        py___code = 'PermGroFac / Rfree'
        Bilt.FHWFac = FHWFac = \
            eval(py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        FHWFac_fcts.update({'latexexpr': r'\FHWFac'})
        FHWFac_fcts.update({'_unicode_': r'R/Φ'})
        FHWFac_fcts.update({'urlhandle': urlroot + 'FHWFacDefn'})
        FHWFac_fcts.update({'py___code': py___code})
        FHWFac_fcts.update({'value_now': FHWFac})
        Bilt.FHWFac_fcts = \
            FHWFac_fcts

        FHWC_fcts = {
            'about': 'Finite Human Wealth Condition'
        }
        FHWC_fcts.update({'latexexpr': r'\FHWC'})
        FHWC_fcts.update({'urlhandle': urlroot + 'FHWCDefn'})
        FHWC_fcts.update({'py___code': 'test: FHWFac < 1'})
        Bilt.FHWC_fcts = FHWC_fcts

        hNrmInf_fcts = {
            'about': 'Human wealth for inf hor'
        }
        py___code = '1/(1-FHWFac) if (FHWFac < 1) else float("inf")'
        Bilt.hNrmInf = hNrmInf = \
            eval(py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        hNrmInf_fcts = dict({'latexexpr': r'1/(1-\FHWFac)'})
        hNrmInf_fcts.update({'value_now': hNrmInf})
        hNrmInf_fcts.update({'py___code': py___code})
        Bilt.hNrmInf_fcts = hNrmInf_fcts

        DiscGPFacRawCusp_fcts = {
            'about': 'DiscFac s.t. GPFacRaw = 1'
        }
        py___code = '( PermGroFac                       **CRRA)/(Rfree)'
        Bilt.DiscGPFacRawCusp = DiscGPFacRawCusp = \
            eval(py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        DiscGPFacRawCusp_fcts.update({'latexexpr':
                                    r'\PermGroFac^{\CRRA}/\Rfree'})
        DiscGPFacRawCusp_fcts.update({'value_now': DiscGPFacRawCusp})
        DiscGPFacRawCusp_fcts.update({'py___code': py___code})
        Bilt.DiscGPFacRawCusp_fcts = \
            DiscGPFacRawCusp_fcts

        DiscGPFacLivCusp_fcts = {
            'about': 'DiscFac s.t. GPFacLiv = 1'
        }
        py___code = '( PermGroFac                       **CRRA)/(Rfree*LivPrb)'
        Bilt.DiscGPFacLivCusp = DiscGPFacLivCusp = \
            eval(py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        DiscGPFacLivCusp_fcts.update({'latexexpr':
                                    r'\PermGroFac^{\CRRA}/(\Rfree\LivPrb)'})
        DiscGPFacLivCusp_fcts.update({'value_now': DiscGPFacLivCusp})
        DiscGPFacLivCusp_fcts.update({'py___code': py___code})
        Bilt.DiscGPFacLivCusp_fcts = DiscGPFacLivCusp_fcts

        VAFac_fcts = {  # overwritten by version with uncertainty
            'about': 'Finite Value of Autarky Factor'
        }
        py___code = 'LivPrb * DiscFacLiv'
        Bilt.VAFac = \
            eval(py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        VAFac_fcts.update({'latexexpr': r'\PFVAFac'})
        VAFac_fcts.update({'urlhandle': urlroot + 'PFVAFacDefn'})
        VAFac_fcts.update({'py___code': py___code})
        Bilt.VAFac_fcts = VAFac_fcts

        FVAC_fcts = {  # overwritten by version with uncertainty
            'about': 'Finite Value of Autarky Condition - Perfect Foresight'
        }
        FVAC_fcts.update({'latexexpr': r'\PFFVAC'})
        FVAC_fcts.update({'urlhandle': urlroot + 'PFFVACDefn'})
        FVAC_fcts.update({'py___code': 'test: PFVAFac < 1'})
        Bilt.FVAC_fcts = FVAC_fcts

        E_Next_.IncNrmNxt_fcts = {  # Overwritten by version with uncertainty
            'about': 'Expected income next period'
        }
        py___code = '1.0'
        E_Next_.IncNrmNxt = \
            eval(py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        E_Next_.IncNrmNxt_fcts.update({'py___code': py___code})
        E_Next_.IncNrmNxt_fcts.update({'value_now': E_Next_.IncNrmNxt})
        crnt.E_Next_.IncNrmNxt_fcts = E_Next_.IncNrmNxt_fcts

        RNrm_PF_fcts = {
            'about': 'Expected Growth-Normalized Return'
        }
        py___code = 'Rfree / PermGroFac'
        E_Next_.RNrm_PF = RNrm_PF = \
            eval(py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        RNrm_PF_fcts.update({'latexexpr': r'\PFRNrm'})
        RNrm_PF_fcts.update({'_unicode_': r'R/Φ'})
        RNrm_PF_fcts.update({'urlhandle': urlroot + 'PFRNrmDefn'})
        RNrm_PF_fcts.update({'py___code': py___code})
        RNrm_PF_fcts.update({'value_now': RNrm_PF})
        E_Next_.RNrm_PF_fcts = RNrm_PF_fcts

        DiscFacLiv_fcts = {
            'about': 'Mortality-Inclusive Discounting'
        }
        py___code = 'DiscFac * LivPrb'
        Bilt.DiscFacLiv = DiscFacLiv = \
            eval(py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        DiscFacLiv_fcts.update({'latexexpr': r'\DiscFacLiv'})
        DiscFacLiv_fcts.update({'_unicode_': r'ℒ β'})
        DiscFacLiv_fcts.update({'urlhandle': urlroot + 'PFRNrm'})
        DiscFacLiv_fcts.update({'py___code': py___code})
        DiscFacLiv_fcts.update({'value_now': DiscFacLiv})
        Bilt.DiscFacLiv_fcts = DiscFacLiv_fcts

    def build_facts_recursive(self):
        """
        For t, calculate results that depend on the last period solved (t+1).

        Returns
        -------
        None.

        """
        crnt = self.solution_current
        tp1 = self.solution_follows.Bilt  # tp1 means t+1
        Bilt, Pars, E_Next_ = crnt.Bilt, crnt.Pars, crnt.E_Next_

        givens = {**Pars.__dict__, **locals()}
        urlroot = Bilt.urlroot
        Bilt.DiscFacLiv = Pars.DiscFac * Pars.LivPrb

        hNrm_fcts = {
            'about': 'Human Wealth '
        }
        py___code = '((PermGroFac / Rfree) * (1.0 + tp1.hNrm))'
        if crnt.stge_kind['iter_status'] == 'terminal_partial':  # kludge:
            py___code = '0.0'  # hNrm = 0.0 for last period
        Bilt.hNrm = hNrm = \
            eval(py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        hNrm_fcts.update({'latexexpr': r'\hNrm'})
        hNrm_fcts.update({'_unicode_': r'R/Φ'})
        hNrm_fcts.update({'urlhandle': urlroot + 'hNrmDefn'})
        hNrm_fcts.update({'py___code': py___code})
        hNrm_fcts.update({'value_now': hNrm})
        Bilt.hNrm_fcts = hNrm_fcts

        BoroCnstNat_fcts = {
            'about': 'Natural Borrowing Constraint'
        }
        py___code = '(tp1.mNrmMin - TranShkMin)*(PermGroFac/Rfree)*PermShkMin'
        if crnt.stge_kind['iter_status'] == 'terminal_partial':  # kludge
            py___code = 'hNrm'  # Presumably zero
        Bilt.BoroCnstNat = BoroCnstNat = \
            eval(py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        BoroCnstNat_fcts.update({'latexexpr': r'\BoroCnstNat'})
        BoroCnstNat_fcts.update({'_unicode_': r''})
        BoroCnstNat_fcts.update({'urlhandle': urlroot + 'BoroCnstNat'})
        BoroCnstNat_fcts.update({'py___code': py___code})
        BoroCnstNat_fcts.update({'value_now': BoroCnstNat})
        Bilt.BoroCnstNat_fcts = BoroCnstNat_fcts

        BoroCnst_fcts = {
            'about': 'Effective Borrowing Constraint'
        }
        py___code = 'BoroCnstNat if (BoroCnstArt == None) else ' + \
            '(BoroCnstArt if BoroCnstNat < BoroCnstArt else BoroCnstNat)'
        Bilt.BoroCnst = BoroCnst = \
            eval(py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        BoroCnst_fcts.update({'py___code': py___code})
        BoroCnst_fcts.update({'value_now': BoroCnst})
        Bilt.BoroCnst_fcts = BoroCnst_fcts

        # MPCmax is not a meaningful object in the PF model so is not created
        # there so create it here
        MPCmax_fcts = {
            'about': 'Maximal MPC in current period as m -> mNrmMin'
        }
        py___code = '1.0 / (1.0 + (RPFac / tp1.MPCmax))'
        if crnt.stge_kind['iter_status'] == 'terminal_partial':  # kludge:
            crnt.tp1.MPCmax = float('inf')  # => MPCmax = 1 for last per
        Bilt.MPCmax = eval(
            py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        MPCmax_fcts.update({'latexexpr': r''})
        MPCmax_fcts.update({'urlhandle': urlroot + 'MPCmax'})
        MPCmax_fcts.update({'py___code': py___code})
        MPCmax_fcts.update({'value_now': Bilt.MPCmax})
        Bilt.MPCmax_fcts = MPCmax_fcts

        mNrmMin_fcts = {
            'about': 'Min m is the max you can borrow'
        }
        py___code = 'BoroCnst'
        Bilt.mNrmMin = \
            eval(py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        mNrmMin_fcts.update({'latexexpr': r'\mNrmMin'})
        mNrmMin_fcts.update({'py___code': py___code})
        Bilt.mNrmMin_fcts = mNrmMin_fcts

        MPCmin_fcts = {
            'about': 'Minimal MPC in current period as m -> infty'
        }
        py___code = '1.0 / (1.0 + (RPFac / tp1.MPCmin))'
        if crnt.stge_kind['iter_status'] == 'terminal_partial':  # kludge:
            py__code = '1.0'
        Bilt.MPCmin = \
            eval(py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        MPCmin_fcts.update({'latexexpr': r''})
        MPCmin_fcts.update({'urlhandle': urlroot + 'MPCmin'})
        MPCmin_fcts.update({'py___code': py___code})
        MPCmin_fcts.update({'value_now': Bilt.MPCmin})
        Bilt.MPCmin_fcts = MPCmin_fcts

        MPCmax_fcts = {
            'about': 'Maximal MPC in current period as m -> mNrmMin'
        }
        py___code = '1.0 / (1.0 + (RPFac / tp1.MPCmax))'
        if crnt.stge_kind['iter_status'] == 'terminal_partial':  # kludge:
            Bilt.tp1.MPCmax = float('inf')  # => MPCmax = 1 for final period
        Bilt.MPCmax = \
            eval(py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        MPCmax_fcts.update({'latexexpr': r''})
        MPCmax_fcts.update({'urlhandle': urlroot + 'MPCmax'})
        MPCmax_fcts.update({'py___code': py___code})
        MPCmax_fcts.update({'value_now': Bilt.MPCmax})
        Bilt.MPCmax_fcts = MPCmax_fcts

        cFuncLimitIntercept_fcts = {
            'about':
                'Vertical intercept of perfect foresight consumption function'}
        py___code = 'MPCmin * hNrm'
        Bilt.cFuncLimitIntercept = \
            eval(py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        cFuncLimitIntercept_fcts.update({'py___code': py___code})
        cFuncLimitIntercept_fcts.update({'latexexpr': r'\MPC \hNrm'})
        crnt.Bilt.cFuncLimitIntercept_fcts = cFuncLimitIntercept_fcts

        cFuncLimitSlope_fcts = {
            'about': 'Slope of limiting consumption function'}
        py___code = 'MPCmin'
        cFuncLimitSlope_fcts.update({'py___code': 'MPCmin'})
        Bilt.cFuncLimitSlope = \
            eval(py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        cFuncLimitSlope_fcts.update({'py___code': py___code})
        cFuncLimitSlope_fcts = dict({'latexexpr': r'\MPCmin'})
        cFuncLimitSlope_fcts.update({'urlhandle': r'\MPCminDefn'})
        crnt.Bilt.cFuncLimitSlope_fcts = cFuncLimitSlope_fcts

        # That's the end of things that are identical for PF and non-PF models
        # Models with uncertainty will supplement the above calculations

        return crnt

    def solve_prepared_stage_divert(self):
        """
        Allow alternative solution method in special cases.

        Returns
        -------
        divert : boolean
            If False (usually), continue normal solution
            If True, produce alternative solution and store on self.solution_current
        """
        # bare-bones default terminal solution does not have all the facts
        # we need, because it is generic (for any u func) so add the facts
        crnt, folw = self.solution_current, self.solution_follows
        if folw.Bilt.stge_kind['iter_status'] != 'terminal_partial':
            return False  # Continue with normal solution procedures
        else:
            # Populate it with the proper properties
            crnt = define_t_reward(crnt, def_utility_CRRA)  # Bellman reward
            define_transition(crnt, 'chosen_to_next_choice')
            define_transition(crnt, 'choice_to_chosen')
            crnt.cFunc = crnt.Bilt.cFunc  # make cFunc accessible
            crnt = def_value_CRRA(crnt, crnt.Pars.CRRA)  # make v using cFunc
            self.build_facts_infhor()
            crnt.Bilt.stge_kind['iter_status'] = 'iterator'  # now it's legit
            return True  # if pseudo_terminal=True, enhanced replaces original

    def solve_prepared_stage(self):  # inside ConsPerfForesightSolver
        """
        Solve one stage/period of the consumption-saving problem.

        Parameters
        ----------
        None (all are already in self)

        Returns
        -------
        solution : ConsumerSolution
            The solution to this period/stage's problem
        """
        if self.solve_prepared_stage_divert():  # Allow bypass of normal soln
            return self.solution_current  # created by solve_prepared_stage_divert

        crnt = self.solution_current

        define_transition(crnt, 'chosen_to_next_choice')

        self.from_chosen_states_make_continuation_E_Next_(crnt)

        define_t_reward(crnt, def_utility_CRRA)  # Bellman reward: utility

        self.make_t_decision_rules_and_value_functions()

        return crnt

    # alias for core.py which calls .solve method
    solve = solve_prepared_stage

    def make_t_decision_rules_and_value_functions(self):
        """
        Add decision rules and value funcs to current solution object.

        Returns
        -------
        agent_stage_solution : agent_stage_solution
            Augmented with decision rules and value functions

        """
        self.make_cFunc_PF()
        return def_value_funcs(self.solution_current, self.solution_current.Pars.CRRA)

    def solver_prep_solution_for_an_iteration(self):  # PF
        """Prepare current stage for processing by the one-stage solver."""
        crnt = self.solution_current

        Bilt, Pars = crnt.Bilt, crnt.Pars

        # Catch degenerate case of zero-variance income distributions
        # Otherwise "test cases" that try the degenerate dstns will fail
        if hasattr(Bilt, "TranShkVals") and hasattr(Bilt, "PermShkVals"):
            if ((Bilt.TranShkMin == 1.0) and (Bilt.PermShkMin == 1.0)):
                crnt.E_Next_.Inv_PermShk = 1.0
                crnt.E_Next_.uInv_PermShk = 1.0
        else:  # Missing trans or PermShkVals; assume it's PF model
            Bilt.TranShkMin = Bilt.PermShkMin = 1.0

        # Nothing needs to be done for terminal_partial
        if hasattr(Bilt, 'stge_kind'):
            if 'iter_status' in Bilt.stge_kind:
                if (Bilt.stge_kind['iter_status'] == 'terminal_partial'):
                    # solution_terminal is handmade, do not remake
                    return

        Bilt.stge_kind = \
            crnt.stge_kind = {'iter_status': 'iterator',
                              'slvr_type': self.__class__.__name__}

        return

    # Disambiguate "prepare_to_solve" from similar method names elsewhere
    # (preserve "prepare_to_solve" as alias because core.py calls it)
    prepare_to_solve = solver_prep_solution_for_an_iteration


##############################################################################

class ConsIndShockSetup(ConsPerfForesightSolver):
    """
    Solve one period of CRRA problem with transitory and permanent shocks.

    This is a superclass for solvers of one period consumption problems with
    constant relative risk aversion utility and permanent and transitory shocks
    to labor income, containing code shared among alternative specific solvers.

    N.B.: Because this is a one-time-period solver, objects that (in the full
    problem) are lists because they are allowed to vary at different periods
    (like, income growth at different ages), are scalars here because the value
    that is appropriate for the current period is the one that will be passed.

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
    eventTiming : str, optional
        Stochastic shocks or deterministic evolutions of the problem (aside
        from state variable transitions) can occur between choice stages. The
        informaton about these events needs to be attached to the appropriate
        solution stage, and executed at appropriate point. This option allows
        changing interpretation of an existing variable, e.g. income shocks,
        between the allowed timings.
             'EOP' is 'End of problem/period'
             'BOP' is 'Beginning of problem/period'
             'Both' there are things to do both before and after decision stage
    """

    shock_vars = ['TranShkDstn', 'PermShkDstn']  # Unemp shock=min(transShkVal)

    # TODO: CDC 20210416: Params shared with PF are in different order. Fix
    # noinspection PyTypeChecker
    def __init__(
            self, solution_next, IncShkDstn, LivPrb, DiscFac, CRRA, Rfree,
            PermGroFac, BoroCnstArt, aXtraGrid, vFuncBool, CubicBool,
            PermShkDstn, TranShkDstn,
            solveMethod='EGM',
            eventTiming='EOP',
            horizon='infinite',
            **kwds):
        # First execute PF solver init
        # We must reorder params by hand in case someone tries positional solve
        ConsPerfForesightSolver.__init__(self, solution_next, DiscFac=DiscFac,
                                         LivPrb=LivPrb, CRRA=CRRA,
                                         Rfree=Rfree, PermGroFac=PermGroFac,
                                         BoroCnstArt=BoroCnstArt,
                                         IncShkDstn=IncShkDstn,
                                         PermShkDstn=PermShkDstn,
                                         TranShkDstn=TranShkDstn,
                                         solveMethod=solveMethod,
                                         eventTiming=eventTiming,
                                         horizon=horizon,
                                         **kwds)

        # ConsPerfForesightSolver.__init__ makes self.solution_current
        crnt = self.solution_current

        # Things we have built, exogenous parameters, and model structures:
        Bilt, Modl = crnt.Bilt, crnt.Modl

        Modl.eventTiming = eventTiming
        Modl.horizon = horizon

        Bilt.aXtraGrid = aXtraGrid
        self.CubicBool = CubicBool

        # In which column is each object stored in IncShkDstn?
        Bilt.permPos = IncShkDstn.parameters['ShkPosn']['perm']
        Bilt.tranPos = IncShkDstn.parameters['ShkPosn']['tran']

        # Bcst are "broadcasted" values: serial list of every permutation
        # Makes it fast to take expectations using E_dot
        Bilt.PermShkValsBcst = PermShkValsBcst = IncShkDstn.X[Bilt.permPos]
        Bilt.TranShkValsBcst = TranShkValsBcst = IncShkDstn.X[Bilt.tranPos]

        Bilt.ShkPrbs = ShkPrbs = IncShkDstn.pmf

        Bilt.PermShkPrbs = PermShkPrbs = PermShkDstn.pmf
        Bilt.PermShkVals = PermShkVals = PermShkDstn.X
        # Confirm that perm shocks have expectation near one
        assert_approx_equal(E_dot(PermShkPrbs, PermShkVals), 1.0)

        Bilt.TranShkPrbs = TranShkPrbs = TranShkDstn.pmf
        Bilt.TranShkVals = TranShkVals = TranShkDstn.X
        # Confirm that tran shocks have expectation near one
        assert_approx_equal(E_dot(TranShkPrbs, TranShkVals), 1.0)

        Bilt.PermShkMin = PermShkMin = np.min(PermShkVals)
        Bilt.TranShkMin = TranShkMin = np.min(TranShkVals)

        Bilt.PermShkMax = PermShkMax = np.max(PermShkVals)
        Bilt.TranShkMax = TranShkMax = np.max(TranShkVals)

        Bilt.UnempPrb = Bilt.TranShkPrbs[0]

        Bilt.inc_min_Prb = np.sum(  # All cases where perm and tran Shk are Min
            ShkPrbs[ \
                PermShkValsBcst * TranShkValsBcst == PermShkMin * TranShkMin
            ]
        )

        Bilt.inc_max_Prb = np.sum(  # All cases where perm and tran Shk are Min
            ShkPrbs[ \
                PermShkValsBcst * TranShkValsBcst == PermShkMax * TranShkMax
            ]
        )
        Bilt.inc_max_Val = PermShkMax * TranShkMax

    def build_facts_infhor(self):
        """
        Calculate expectations and facts for models with uncertainty.

        For versions with uncertainty in transitory and/or permanent shocks,
        adds to the solution a set of results useful for calculating
        various diagnostic conditions about the problem, and stable
        points (if they exist).

        Returns
        -------
        solution : ConsumerSolution
            Same solution that was provided, augmented with the factors

        """
        super().build_facts_infhor()  # Make the facts built by the PF model

        crnt = self.solution_current

        Bilt, Pars, E_Next_ = crnt.Bilt, crnt.Pars, crnt.E_Next_

        # The 'givens' do not change as facts are constructed
        givens = {**Pars.__dict__, **crnt.__dict__}

        Bilt.E_dot = E_dot  # add dot product expectations operator to envt

        urlroot = Bilt.urlroot

        # Many other _fcts will have been inherited from the perfect foresight

        # Here we need compute only those objects whose value changes from PF
        # (or does not exist in PF case)

        E_Next_.IncNrmNxt_fcts = {
            'about': 'Expected income next period'
        }
        py___code = 'E_dot(ShkPrbs, TranShkValsBcst * PermShkValsBcst)'
        E_Next_.IncNrmNxt = E_Next_.IncNrmNxt = eval(
            py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
#        E_Next_.IncNrmNxt_fcts.update({'latexexpr': r'ExIncNrmNxt'})
        E_Next_.IncNrmNxt_fcts.update({'_unicode_': r'E[TranShk PermShk]=1.0'})
#        E_Next_.IncNrmNxt_fcts.update({'urlhandle': urlroot + 'ExIncNrmNxt'})
        E_Next_.IncNrmNxt_fcts.update({'py___code': py___code})
        E_Next_.IncNrmNxt_fcts.update({'value_now': E_Next_.IncNrmNxt})
        crnt.E_Next_.IncNrmNxt_fcts = E_Next_.IncNrmNxt_fcts

        E_Next_.Inv_PermShk_fcts = {
            'about': 'Expected Inverse of Permanent Shock'
        }
        py___code = 'E_dot(1/PermShkVals, PermShkPrbs)'
        E_Next_.Inv_PermShk = E_Next_.Inv_PermShk = eval(
            py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
#        E_Next_.Inv_PermShk_fcts.update({'latexexpr': r'\ExInvPermShk'})
#        E_Next_.Inv_PermShk_fcts.update({'urlhandle':urlroot + 'ExInvPermShk'})
        E_Next_.Inv_PermShk_fcts.update({'py___code': py___code})
        E_Next_.Inv_PermShk_fcts.update({'value_now': E_Next_.Inv_PermShk})
        crnt.E_Next_.Inv_PermShk_fcts = E_Next_.Inv_PermShk_fcts

        E_Next_.RNrm_fcts = {
            'about': 'Expected Stochastic-Growth-Normalized Return'
        }
        py___code = 'RNrm_PF * E_Next_.Inv_PermShk'
        E_Next_.RNrm = eval(
            py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        E_Next_.RNrm_fcts.update({'latexexpr': r'\ExRNrm'})
        E_Next_.RNrm_fcts.update({'_unicode_': r'E[R/φ]'})
        E_Next_.RNrm_fcts.update({'urlhandle': urlroot + 'ExRNrm'})
        E_Next_.RNrm_fcts.update({'py___code': py___code})
        E_Next_.RNrm_fcts.update({'value_now': E_Next_.RNrm})
        E_Next_.RNrm_fcts = E_Next_.RNrm_fcts

        E_Next_.uInv_PermShk_fcts = {
            'about': 'Expected Utility for Consuming Permanent Shock'
        }
        py___code = 'E_dot(PermShkValsBcst**(1-CRRA), ShkPrbs)'
        E_Next_.uInv_PermShk = E_Next_.uInv_PermShk = eval(
            py___code, {}, {**E_Next_.__dict__, **Bilt.__dict__, **givens})
#        E_Next_.uInv_PermShk_fcts.update({'latexexpr': r'\ExuInvPermShk'})
        E_Next_.uInv_PermShk_fcts.update({'urlhandle': urlroot+r'ExuInvPermShk'})
        E_Next_.uInv_PermShk_fcts.update({'py___code': py___code})
        E_Next_.uInv_PermShk_fcts.update({'value_now': E_Next_.uInv_PermShk})
        E_Next_.uInv_PermShk_fcts = E_Next_.uInv_PermShk_fcts

        GPFacNrm_fcts = {
            'about': 'Normalized Expected Growth Patience Factor'
        }
        py___code = 'GPFacRaw * E_Next_.Inv_PermShk'
        Bilt.GPFacNrm = eval(py___code, {},
                           {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        GPFacNrm_fcts.update({'latexexpr': r'\GPFacNrm'})
        GPFacNrm_fcts.update({'_unicode_': r'Þ_Φ'})
        GPFacNrm_fcts.update({'urlhandle': urlroot + 'GPFacNrm'})
        GPFacNrm_fcts.update({'py___code': py___code})
        Bilt.GPFacNrm_fcts = GPFacNrm_fcts

        GICNrm_fcts = {
            'about': 'Stochastic Growth Normalized Impatience Condition'
        }
        GICNrm_fcts.update({'latexexpr': r'\GICNrm'})
        GICNrm_fcts.update({'urlhandle': urlroot + 'GICNrm'})
        GICNrm_fcts.update({'py___code': 'test: GPFacNrm < 1'})
        Bilt.GICNrm_fcts = GICNrm_fcts

        FVAC_fcts = {  # overwrites PF version
            'about': 'Finite Value of Autarky Condition'
        }

        VAFac_fcts = {  # overwrites PF version PFVAFac
            'about': 'Finite Value of Autarky Factor'
        }
        py___code = 'LivPrb * DiscFacLiv'
        Bilt.VAFac = eval(py___code, {},
                         {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        VAFac_fcts.update({'latexexpr': r'\VAFac'})
        VAFac_fcts.update({'urlhandle': urlroot + 'VAFacDefn'})
        VAFac_fcts.update({'py___code': py___code})
        Bilt.VAFac_fcts = VAFac_fcts

        FVAC_fcts = {  # overwrites PF version
            'about': 'Finite Value of Autarky Condition'
        }
        FVAC_fcts.update({'latexexpr': r'\FVAC'})
        FVAC_fcts.update({'urlhandle': urlroot + 'FVAC'})
        FVAC_fcts.update({'py___code': 'test: VAFac < 1'})
        Bilt.FVAC_fcts = FVAC_fcts

        WRPFac_fcts = {
            'about': 'Weak Return Patience Factor'
        }
        py___code = '(UnempPrb ** (1 / CRRA)) * RPFac'
        Bilt.WRPFac = WRPFac = \
            eval(py___code, {},
                 {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        WRPFac_fcts.update({'latexexpr': r'\WRPFac'})
        WRPFac_fcts.update({'_unicode_': r'℘^(1/\rho) RPFac'})
        WRPFac_fcts.update({'urlhandle': urlroot + 'WRPFacDefn'})
        WRPFac_fcts.update({'value_now': WRPFac})
        WRPFac_fcts.update({'py___code': py___code})
        Bilt.WRPFac_fcts = WRPFac_fcts

        WRIC_fcts = {
            'about': 'Weak Return Impatience Condition'
        }
        WRIC_fcts.update({'latexexpr': r'\WRIC'})
        WRIC_fcts.update({'urlhandle': urlroot + 'WRIC'})
        WRIC_fcts.update({'py___code': 'test: WRPFac < 1'})
        Bilt.WRIC_fcts = WRIC_fcts

        DiscGPFacNrmCusp_fcts = {
            'about': 'DiscFac s.t. GPFacNrm = 1'
        }
        py___code = '((PermGroFac/E_Next_.Inv_PermShk)**(CRRA))/Rfree'
        Bilt.DiscGPFacNrmCusp = DiscGPFacNrmCusp = \
            eval(py___code, {},
                 {**E_Next_.__dict__, **Bilt.__dict__, **givens})
        DiscGPFacNrmCusp_fcts.update({'latexexpr': ''})
        DiscGPFacNrmCusp_fcts.update({'value_now': DiscGPFacNrmCusp})
        DiscGPFacNrmCusp_fcts.update({'py___code': py___code})
        Bilt.DiscGPFacNrmCusp_fcts = DiscGPFacNrmCusp_fcts

    def build_facts_recursive(self):
        """
        Calculate recursive facts for current period from next.

        Returns
        -------
        crnt : agent_stage_solution

        """
        super().build_facts_recursive()

        # All the recursive facts are required for PF model so already exist
        # But various lambda functions are interesting when uncertainty exists

        crnt = self.solution_current
        Bilt = crnt.Bilt
        Pars = crnt.Pars
        E_Next_ = crnt.E_Next_

        # To use these it is necessary to have created an alias to
        # the relevant namespace on the solution object, e.g.
        # E_Next_ = [soln].E_Next_
        # Bilt = [soln].Bilt
        # Pars = [soln].Pars

        # Given m, value of c where E[m_{t+1}]=m_{t}
        E_Next_.c_where_m_tp1_minus_m_t_eq_0 = (
            lambda m_t:
            m_t * (1 - 1 / E_Next_.RNrm) + (1 / E_Next_.RNrm)
        )
        E_Next_.c_where_PermShk_tp1_times_m_tp1_minus_m_t_eq_0 = (
            lambda m_t:
            m_t * (1 - E_Next_.Inv_RNrm_PF) + E_Next_.Inv_RNrm_PF
        )
        # E[c_{t+1} pLvl_{t+1}/pLvl_{t}] as a fn of a_{t}
        E_Next_.cLvl_tp1_Over_pLvl_t_from_a_t = (
            lambda a_t:
            E_dot(Pars.PermGroFac *
                  Bilt.PermShkValsBcst *
                  Bilt.cFunc((E_Next_.RNrm_PF / Bilt.PermShkValsBcst) * a_t
                             + Bilt.TranShkValsBcst),
                  Bilt.ShkPrbs)
        )
        E_Next_.c_where_E_Next_m_tp1_minus_m_t_eq_0 = \
            lambda m_t: \
            m_t * (1 - 1/E_Next_.RNrm) + (1 / E_Next_.RNrm)

        E_Next_.c_where_E_Next_PermShk_tp1_times_m_tp1_minus_m_t_eq_0 = \
            lambda m_t: \
            (m_t * (1 - 1 / E_Next_.RNrm_PF)) + (1 / E_Next_.RNrm_PF)
        # mTrgNrm solves E_Next_.RNrm*(m - c(m)) + E[inc_next] - m = 0

        E_Next_.m_tp1_minus_m_t = (
            lambda m_t:
            (E_Next_.RNrm * (m_t - Bilt.cFunc(m_t)) + E_Next_.IncNrmNxt)-m_t
        )
        E_Next_.m_tp1_Over_m_t = (
            lambda m_t:
            (E_Next_.RNrm * (m_t - Bilt.cFunc(m_t)) + E_Next_.IncNrmNxt)/m_t
        )
        E_Next_.mLvl_tp1_Over_mLvl_t = (
            lambda m_t:
            (Pars.Rfree * (m_t - Bilt.cFunc(m_t)) +
             Pars.PermGroFac * E_Next_.IncNrmNxt)/m_t
        )
        # Define separately for float ('num') and listlike ('lst'), then combine
        E_Next_.cLvl_tp1_Over_pLvl_t_from_num_a_t = (
            lambda a_t:
            E_dot(
                Bilt.PermShkValsBcst * Pars.PermGroFac * Bilt.cFunc(
                    (E_Next_.RNrm_PF / Bilt.PermShkValsBcst) *
                    a_t + Bilt.TranShkValsBcst
                ),
                Bilt.ShkPrbs)
        )
        E_Next_.cLvl_tp1_Over_pLvl_t_from_lst_a_t = (
            lambda a_lst: list(map(
                E_Next_.cLvl_tp1_Over_pLvl_t_from_num_a_t, a_lst
            ))
        )
        E_Next_.cLvl_tp1_Over_pLvl_t_from_a_t = (
            lambda a_t:
            E_Next_.cLvl_tp1_Over_pLvl_t_from_lst_a_t(a_t)
            if (type(a_t) == list or type(a_t) == np.ndarray) else
            E_Next_.cLvl_tp1_Over_pLvl_t_from_num_a_t(a_t)
        )
        # Define separately for float ('num') and listlike ('lst'), then combine
        # E_Next_.mLog_tp1_from_num_a_t = (
        #     lambda a_t:
        #     E_dot(
        #         np.log(
        #             Pars.PermGroFac *
        #                 (E_Next_.RNrm_PF * a_t
        #                  + Bilt.TranShkValsBcst)
        #         )
        #         ,
        #         Bilt.ShkPrbs)
        # )
        # E_Next_.mLog_tp1_from_lst_a_t = (
        #     lambda a_lst: list(map(
        #         E_Next_.mLog_tp1_from_num_a_t, a_lst
        #     ))
        # )
        # E_Next_.mLog_tp1_from_a_t = (
        #     lambda a_t:
        #     E_Next_.mLog_tp1_from_lst_a_t(a_t)
        #     if (type(a_t) == list or type(a_t) == np.ndarray) else
        #     E_Next_.mLog_tp1_from_num_a_t(a_t)
        # )
        # E_Next_.mLog_tp1_from_num_m_t = (
        #     lambda m_t:
        #     E_dot(
        #         np.log(
        #             Pars.PermGroFac * (E_Next_.RNrm_PF *
        #                                (m_t - Bilt.cFunc(m_t))
        #                                + Bilt.TranShkValsBcst*Bilt.PermShkValsBcst)
        #         )
        #         ,
        #         Bilt.ShkPrbs)
        # )
        # E_Next_.mLog_tp1_from_lst_m_t = (
        #     lambda m_lst: list(map(
        #         E_Next_.mLog_tp1_from_num_m_t, a_lst
        #     ))
        # )
        # E_Next_.mLog_tp1_from_a_t = (
        #     lambda a_t:
        #     E_Next_.mLog_tp1_from_lst_a_t(a_t)
        #     if (type(a_t) == list or type(a_t) == np.ndarray) else
        #     E_Next_.mLog_tp1_from_num_a_t(a_t)
        # )
        E_Next_.k_tp1_from_num_m_t = (
            lambda m_t:
                m_t - Bilt.cFunc(m_t)
        )
        E_Next_.k_tp1_from_lst_m_t = (
            lambda m_lst: list(map(
                E_Next_.k_tp1_from_num_m_t, m_lst
            ))
        )
        E_Next_.k_tp1_from_m_t = (
            lambda m_t:
            E_Next_.k_tp1_from_lst_m_t(m_t)
            if (type(m_t) == list or type(m_t) == np.ndarray) else
            E_Next_.k_tp1_from_num_m_t(m_t)
        )
        E_Next_.a_t_from_num_m_t = (
            lambda m_t:
                m_t - Bilt.cFunc(m_t)
        )
        E_Next_.a_t_from_lst_m_t = (
            lambda m_lst: list(map(
                E_Next_.a_t_from_num_m_t, m_lst
            ))
        )
        E_Next_.a_t_from_m_t = (
            lambda m_t:
            E_Next_.a_t_from_lst_m_t(m_t)
            if (type(m_t) == list or type(m_t) == np.ndarray) else
            E_Next_.a_t_from_num_m_t(m_t)
        )
#         E_Next_.mApx_tp1_from_num_a_t = (
#             lambda a_t:
#             E_dot(
#                                        - ((Bilt.TranShkValsBcst*Bilt.PermShkValsBcst-1) * (Bilt.TranShkValsBcst*Bilt.PermShkValsBcst)-1)/(a_t+1)/2
# #                                       + ((Bilt.TranShkValsBcst-1) * (Bilt.TranShkValsBcst-1)* (Bilt.TranShkValsBcst-1)/(a_t+1))/3
#                 +
#                 np.log(
#                     Pars.PermGroFac * (E_Next_.RNrm_PF *
#                                        a_t + 1
#                                        )
#                 )
#                 ,
#                 Bilt.ShkPrbs)
#         )
#         E_Next_.mApx_tp1_from_lst_a_t = (
#             lambda a_lst: list(map(
#                 E_Next_.mApx_tp1_from_num_a_t, a_lst
#             ))
#         )
#         E_Next_.mApx_tp1_from_a_t = (
#             lambda a_t:
#             E_Next_.mApx_tp1_from_lst_a_t(a_t)
#             if (type(a_t) == list or type(a_t) == np.ndarray) else
#             E_Next_.mApx_tp1_from_num_a_t(a_t)
#         )
#         E_Next_.mDif_tp1_from_num_a_t = (
#             lambda a_t:
#             E_dot(np.log(1.+(((a_t * Pars.Rfree + 1)**(-1))
#                                        *((Bilt.TranShkValsBcst*Bilt.PermShkValsBcst-1))))
#                 ,
#                 Bilt.ShkPrbs)
#         )
#         E_Next_.mDif_tp1_from_lst_a_t = (
#             lambda a_lst: list(map(
#                 E_Next_.mDif_tp1_from_num_a_t, a_lst
#             ))
#         )
#         E_Next_.mDif_tp1_from_a_t = (
#             lambda a_t:
#             E_Next_.mDif_tp1_from_lst_a_t(a_t)
#             if (type(a_t) == list or type(a_t) == np.ndarray) else
#             E_Next_.mDif_tp1_from_num_a_t(a_t)
#         )
        E_Next_.mLvl_tp1_Over_pLvl_t_from_num_a_t = (
            lambda a_t:
            Pars.Rfree * a_t + Pars.PermGroFac
        )
        E_Next_.mLvl_tp1_Over_pLvl_t_from_lst_a_t = (
            lambda a_lst: list(map(
                E_Next_.mLvl_tp1_Over_pLvl_t_from_num_a_t, a_lst
            ))
        )
        E_Next_.mLvl_tp1_Over_pLvl_t_from_a_t = (
            lambda a_t:
            E_Next_.mLvl_tp1_Over_pLvl_t_from_lst_a_t(a_t)
            if (type(a_t) == list or type(a_t) == np.ndarray) else
            E_Next_.mLvl_tp1_Over_pLvl_t_from_num_a_t(a_t)
        )
        E_Next_.m_tp1_from_a_t = (
            lambda a_t:
            E_Next_.RNrm * a_t + E_Next_.IncNrmNxt
        )
        E_Next_.log_PermShk_tp1_times_m_tp1_from_num_a_t = (
            lambda a_t:
            E_dot(np.log(Pars.Rfree * a_t
                  + Pars.PermGroFac * Bilt.TranShkValsBcst*Bilt.PermShkValsBcst)
                 ,Bilt.ShkPrbs)
        )
        E_Next_.log_PermShk_tp1_times_PermShk_t_from_lst_a_t = (
            lambda a_lst: list(map(
                E_Next_.log_PermShk_tp1_times_m_tp1_from_num_a_t, a_lst
            ))
        )
        E_Next_.log_PermShk_tp1_times_m_tp1_from_a_t = (
            lambda a_t:
            E_Next_.log_PermShk_tp1_times_m_tp1_from_lst_a_t(a_t)
            if (type(a_t) == list or type(a_t) == np.ndarray) else
            E_Next_.log_PermShk_tp1_times_m_tp1_from_num_a_t(a_t)
        )
        E_Next_.cLvl_tp1_Over_pLvl_t_from_lst_m_t = (
            lambda m_t:
            E_Next_.cLvl_tp1_Over_pLvl_t_from_lst_a_t(m_t -
                                                      Bilt.cFunc(m_t))
        )
        E_Next_.PermShk_tp1_times_m_tp1_Over_m_t = (
            lambda m_t:
            (Pars.Rfree*(m_t - Bilt.cFunc(m_t)) + Pars.PermGroFac * E_Next_.IncNrmNxt)/m_t
        )
        E_Next_.PermShk_tp1_times_m_tp1_Over_m_t_minus_PermGroFac = (
            lambda m_t:
            E_Next_.PermShk_tp1_times_m_tp1_Over_m_t(m_t) - Pars.PermGroFac
        )
        E_Next_.cLvl_tp1_Over_pLvl_t_from_num_m_t = (
            lambda m_t:
            E_Next_.cLvl_tp1_Over_pLvl_t_from_num_a_t(m_t -
                                                      Bilt.cFunc(m_t))
        )
        E_Next_.cLvl_tp1_Over_cLvl_t_from_m_t = (
            lambda m_t:
            E_Next_.cLvl_tp1_Over_pLvl_t_from_lst_m_t(m_t) / Bilt.cFunc(m_t)
            if (type(m_t) == list or type(m_t) == np.ndarray) else
            E_Next_.cLvl_tp1_Over_pLvl_t_from_num_m_t(m_t) / Bilt.cFunc(m_t)
        )
        E_Next_.mLog_tp1_minus_mLog_t_from_m_t = (
            lambda m_t:
            E_Next_.log_PermShk_tp1_times_m_tp1_from_a_t(E_Next_.a_t_from_m_t(m_t)) \
               - np.log(m_t)
        )
        self.solution_current = crnt

        return crnt


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
        self.solution_current.Bilt.aNrmGrid = np.asarray(
            self.solution_current.Bilt.aXtraGrid) + \
            self.solution_current.Bilt.BoroCnstNat

        return self.solution_current.Bilt.aNrmGrid

    # "Expectorate" = calculate expected values of useful objects across the
    # distribution of shocks at constructed grid of values of the relevant
    # states. I tried to invent a new usage, "Expectate" but Spyder/PyCharm
    # kept flagging it as not a real word, while now they don't complain!

    def make_E_Next_(self, IncShkDstn):
        """
        Expectorate after choices but before end of period (incl. discounting).

        Parameters
        ----------
        IncShkDstn : DiscreteDistribution
            The distribution of the stochastic shocks to income.
        """
        crnt, folw = self.solution_current, self.solution_follows

        Bilt, Pars, E_Next_ = crnt.Bilt, crnt.Pars, crnt.E_Next_

        eventTiming = Pars.eventTiming  # this_EOP or next_BOP (or both)

        states_chosen = Bilt.aNrmGrid
        permPos = IncShkDstn.parameters['ShkPosn']['perm']

        if eventTiming == 'EOP':  # shocks happen at end of this period
            CRRA = folw.Bilt.vFunc.CRRA  # Next CRRA utility normalizes
            Discount = Bilt.DiscFacLiv  # Discount next period
            vFunc = folw.Bilt.vFunc
            cFunc = folw.Bilt.cFunc
            PermGroFac = folw.Pars.PermGroFac
            Rfree = Pars.Rfree
        else:  # default to BOP
            # In this case, we should have computed the 'hard part' and
            # attached it already to the BOP of the folw stage.
            breakpoint()
            # DiscFacLiv = Bilt.DiscFacLiv
            # v0_pos, v1_pos = folw.Ante_E_.v0_pos, folw.Ante_E_.v1_pos
            # v2_pos = folw.Ante_E_.v2_pos

        # This is the efficient place to compute expectations of anything
        # at very low marginal cost by adding to list of things calculated

        def f_to_expect_across_shocks_given_current_state(xfer_shks_bcst,
                                                          states_chosen):
            next_choice_states = self.transit_states_given_shocks(
                'chosen_to_next_choice', states_chosen,
                xfer_shks_bcst, IncShkDstn)
            mNrm = next_choice_states['mNrm']
            # Random shocks to permanent income affect mean PermGroFac
            PermGroFacShk = xfer_shks_bcst[permPos] * PermGroFac
            # expected value function derivatives 0, 1, 2
            v_0 = PermGroFacShk ** (1 - CRRA - 0) * vFunc(mNrm)
            v_1 = PermGroFacShk ** (1 - CRRA - 1) * vFunc.dm(mNrm) * Rfree
            v_2 = PermGroFacShk ** (1 - CRRA - 2) * vFunc.dm.dm(mNrm) * Rfree \
                * Rfree
            # cFunc derivatives 0, 1 (level and MPC); no need, but ~zero cost.
            c_0 = cFunc(mNrm)
            c_1 = cFunc.derivative(mNrm)
            return Discount * np.array([v_0, v_1, v_2, c_0, c_1])

        E_Next_.post_choice = np.squeeze(
            expect_funcs_across_shocks_given_states(
                IncShkDstn,
                f_to_expect_across_shocks_given_current_state,
                states_chosen)
        )
        # Store positions of the various objects for later convenience
        E_Next_.v0_pos, E_Next_.v1_pos, E_Next_.v2_pos = 0, 1, 2
        E_Next_.c0_pos, E_Next_.c1_pos = 4, 5

    def make_Ante_E_(self, IncShkDstn):
        """
        Expectorate before beginning-of-period events, and subsequent choices.

        Parameters
        ----------
        IncShkDstn : DiscreteDistribution
            The distribution of the stochastic shocks to income.
        """
        crnt = self.solution_current

        Bilt, Pars, Ante_E_ = crnt.Bilt, crnt.Pars, crnt.Ante_E_

        eventTiming = Pars.eventTiming  # this_EOP or next_BOP (or both)

        if eventTiming == 'EOP':  # shocks happen at end of this period
            return  # nothing needs to be done

        CRRA, Rfree, permPos = Pars.CRRA, Pars.Rfree, Bilt.permPos
        PermGroFac = Pars.PermGroFac
        vFunc, cFunc = crnt.vFunc, crnt.cFunc
        Discount = 1.0  # Allows formulae to be identical here and in E_Next_
        BOP_state = Bilt.mNrmGrid

        # This is the efficient place to compute expectations of anything
        # at very low marginal cost by adding to list of things calculated

        def f_to_expect_across_shocks_given_current_state(xfer_shks_bcst,
                                                          BOP_state):
            choicestep_states = self.transit_states_given_shocks(
                'BOP_to_choice', BOP_state,
                xfer_shks_bcst, IncShkDstn)
            mNrm = choicestep_states['mNrm']
            # Random shocks to permanent income affect mean PermGroFac
            PermGroFacShk = xfer_shks_bcst[permPos] * PermGroFac
            # expected value function derivatives 0, 1, 2
            v_0 = PermGroFacShk ** (1 - CRRA - 0) * vFunc(mNrm)
            v_1 = PermGroFacShk ** (1 - CRRA - 1) * vFunc.dm(mNrm) * Rfree
            v_2 = PermGroFacShk ** (1 - CRRA - 2) * vFunc.dm.dm(mNrm) * Rfree \
                * Rfree
            # cFunc derivatives 0, 1 (level and MPC); no need, but ~zero cost.
            c_0 = cFunc(mNrm)
            c_1 = cFunc.derivative(mNrm)
            return Discount * np.array([v_0, v_1, v_2, c_0, c_1])

        Ante_E_.ante_choice = np.squeeze(
            expect_funcs_across_shocks_given_states(
                IncShkDstn,
                f_to_expect_across_shocks_given_current_state,
                BOP_state)
        )
        # Store positions of the various objects for later convenience
        Ante_E_.v0_pos, Ante_E_.v1_pos, Ante_E_.v2_pos = 0, 1, 2
        Ante_E_.c0_pos, Ante_E_.c1_pos = 4, 5

    def build_cFunc_using_EGM(self):
        """
        Find interpolation points (c, m) for the consumption function.

        Returns
        -------
        cFunc : LowerEnvelope or LinerarInterp
        """
        crnt = self.solution_current
        Bilt, E_Next_, Pars = crnt.Bilt, crnt.E_Next_, crnt.Pars
        v1_pos = E_Next_.v1_pos  # first derivative of value function at chosen
        u, aNrmGrid, BoroCnstArt = Bilt.u, Bilt.aNrmGrid, Pars.BoroCnstArt

        # Endogenous Gridpoints steps
        # [v1_pos]: precalculated first derivative (E_Next_from_chosen_states)
        cNrmGrid = u.dc.Nvrs(E_Next_.post_choice[v1_pos])
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

    def make_t_decision_rules_and_value_functions(self):
        """
        Construct consumption function and marginal value function.

        Given the grid of end-of-period values of assets a, use the endogenous
        gridpoints method to obtain the corresponding values of consumption,
        and use the dynamic budget constraint to obtain the corresponding value
        of market resources m.

        Parameters
        ----------
        None (relies on self.solution_current.aNrmGrid to exist at invocation)

        Returns
        -------
        solution : ConsumerSolution
            The solution to the single period consumption-saving problem.
        """
        crnt = self.solution_current
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
            self.solution_current.Bilt.cFuncLimitIntercept,
            self.solution_current.Bilt.cFuncLimitSlope
        )
        return cFunc_unconstrained

    def from_chosen_states_make_continuation_E_Next_(self, crnt):
        """
        Expect after choices have been made and before shocks realized.

        Calculate circumstances of an agent before the realization of the labor
        income shocks that constitute transition to the next period's state.

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
        crnt = self.solution_current

        # Add some useful info to solution object
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

        self.make_chosen_state_grid()
        self.make_E_Next_(self.solution_current.Pars.IncShkDstn)

        return crnt

    def solve_prepared_stage(self):  # solve ONE stage (ConsIndShockSolver)
        """
        Solve one period of the consumption-saving problem.

        The ".Bilt" namespace on the returned solution object includes
            * decision rule (consumption function), cFunc
            * value and marginal value functions vFunc and vFunc.dm
            * a minimum possible level of normalized market resources mNrmMin
            * normalized human wealth hNrm; bounding MPCs MPCmin and MPCmax.

        If the user sets `CubicBool` to True, cFunc is interpolated with a
        cubic Hermite interpolator.  This is much smoother than the default
        piecewise linear interpolator, and causes (automatic) construction of
        the marginal marginal value function vFunc.dm.dm (which occurs
        automatically).

        In principle, the resulting cFunc will be numerically incorect at
        values of market resources where a marginal increment to m would
        discretely change the probability of a future constraint binding,
        because in principle cFunc is nondifferentiable at those points (cf
        LiqConstr REMARK).

        In practice, if the distribution of shocks is not too granular, the
        approximation error is minor.

        Parameters
        ----------
        none (all should be on self)

        Returns
        -------
        solution : agent_stage_solution
            The solution to this period/stage's problem.
        """
        if self.solve_prepared_stage_divert():  # Allow bypass of normal soln
            return self.solution_current  # made by solve_prepared_stage_divert

        crnt = self.solution_current

        Pars = crnt.Pars
        eventTiming, solveMethod = Pars.eventTiming, Pars.solveMethod

        if solveMethod == 'Generic':  # Steps that should encompass any problem
            define_transition(crnt, 'EOP_to_next_BOP')
            define_transition(crnt, 'chosen_to_EOP')
            self.from_chosen_states_make_continuation_E_Next_(crnt)
            define_t_reward(crnt, def_utility_CRRA)
            define_transition(crnt, 'choice_to_chosen')
            self.make_t_decision_rules_and_value_functions()
            define_transition(crnt, 'BOP_to_choice')
            self.from_BOP_states_make_Ante_E_()
            return crnt

        # if not using Generic, then solve using custom method

        # transition depends on whether events are EOP or BOP
        if eventTiming == 'EOP':  # Events (shocks, etc) at End of Problem
            define_transition(crnt, 'chosen_to_next_choice')
        else:
            define_transition(crnt, 'chosen_to_next_BOP')

        # Given the transition, calculate expectations
        self.from_chosen_states_make_continuation_E_Next_(crnt)

        # Define transition caused by choice
        define_transition(crnt, 'choice_to_chosen')

        # Define today's reward (utility, in consumer's problem)
        define_t_reward(crnt, def_utility_CRRA)

        # Having calculated E(marginal value, etc) of saving, construct c and v
        self.make_t_decision_rules_and_value_functions()

        # Ante_E_: Before BOP shocks realized (does nothing if no BOP shocks)
        self.from_BOP_states_make_Ante_E_()

        return crnt

    # alias "solve" because core.py expects [agent].solve to solve the model
    solve = solve_prepared_stage  # much easier to remember

    def from_BOP_states_make_Ante_E_(self):
        """Make expectations before beginning-of-period events (shocks)."""
        self.make_Ante_E_(self)

    def transit_states_given_shocks(self, transition_name, starting_states,
                                    shks_permuted, IncShkDstn):
        """
        From starting_states calculate transitions given shock permutations.

        Return array of values of normalized market resources m
        corresponding to permutations of potential realizations of
        the permanent and transitory income shocks, given the value of
        end-of-period assets aNrm.

        Parameters
        ----------
        shks_permuted: 2D ndarray
            Permanent and transitory income shocks in 2D ndarray

        Returns
        -------
        transited : dict with results of applying transition eqns
        """
        soln = self.solution_current
        Pars = soln.Pars
        Transitions = soln.Modl.Transitions

        permPos, tranPos = (
            IncShkDstn.parameters['ShkPosn']['perm'],
            IncShkDstn.parameters['ShkPosn']['tran'])

        zeros = starting_states - starting_states  # zero array of right size

        xfer_vars = {
            'PermShk': shks_permuted[permPos] + zeros,  # + zeros fixes size
            'TranShk': shks_permuted[tranPos] + zeros,  # all pmts for each st
            'aNrm': starting_states
        }

        # Everything needed to execute the transition equations
        Info = {**Pars.__dict__, **xfer_vars}

        transition = Transitions[transition_name]
        for eqn_name in transition['compiled'].keys():
            exec(transition['compiled'][eqn_name], Info)

        return Info


class ConsIndShockSolver(ConsIndShockSolverBasic):
    """
    Solves a single period of a standard consumption-saving problem.

    Inherits from ConsIndShockSolverBasic; adds cubic interpolation.
    """

    def make_cFunc_cubic(self, mNrm_Vec, cNrm_Vec):
        """
        Make cubic spline interpolation of unconstrained consumption function.

        Requires self.solution_current.Bilt.aNrm to have been computed already.

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
        crnt = self.solution_current
        Bilt, E_Next_ = crnt.Bilt, crnt.E_Next_
        v2_pos = E_Next_.v2_pos  # second derivative of value function
        u = Bilt.u

        dc_da = E_Next_.post_choice[v2_pos] / u.dc.dc(np.array(cNrm_Vec[1:]))
        MPC = dc_da / (dc_da + 1.0)
        MPC = np.insert(MPC, 0, Bilt.MPCmax)

        cFuncUnc = CubicInterp(
            mNrm_Vec, cNrm_Vec, MPC, Bilt.MPCmin *
            Bilt.hNrm, Bilt.MPCmin
        )
        return cFuncUnc
