# -*- coding: utf-8 -*-
from HARK.core import (_log, core_check_condition)
from HARK.utilities import CRRAutility
from HARK.utilities import CRRAutilityP
from HARK.utilities import CRRAutilityPP
from HARK.utilities import CRRAutility as utility
from HARK.utilities import CRRAutilityP as utilityP
from HARK.utilities import CRRAutilityPP as utilityPP
from HARK.utilities import CRRAutilityP_inv as utilityP_inv
from HARK.utilities import CRRAutility_invP as utility_invP
from HARK.utilities import CRRAutility_inv as utility_inv
from HARK.utilities import CRRAutilityP_invP as utilityP_invP

from HARK.interpolation import (CubicInterp, LowerEnvelope, LinearInterp,
                                ValueFuncCRRA, MargValueFuncCRRA,
                                MargMargValueFuncCRRA)
from HARK import NullFunc, MetricObject
from scipy.optimize import newton as find_zero_newton
from numpy import dot as E_dot  # easier to type
from numpy.testing import assert_approx_equal as assert_approx_equal
import numpy as np
from copy import deepcopy
from builtins import (str, breakpoint)
from types import SimpleNamespace
from IPython.lib.pretty import pprint
from HARK.ConsumptionSaving.ConsIndShockModel_Both import (
    def_utility, def_value_funcs)
from HARK.distribution import (calc_expectation)


class Built(SimpleNamespace):
    """
    Objects built by solvers during course of solution.
    """
# TODO: Move (to core.py) when vetted/agreed
    pass


class SuccessorInfo(SimpleNamespace):
    """
    Objects retrieved from successor to the stage
    referenced in "self." Should contain everything needed to reconstruct
    solution to problem of self even if solution_next is not present.
    """
# TODO: Move (to core.py) when vetted/agreed
    pass


__all__ = [
    "ConsumerSolutionOld",
    "ConsumerSolution",
    "ConsumerSolutionOneStateCRRA",
    "ConsPerfForesightSolverEOP",
    "ConsPerfForesightSolver",
    "ConsIndShockSetupEOP",
    "ConsIndShockSetup",
    "ConsIndShockSolverBasicEOP",
    "ConsIndShockSolverBasic",
    "ConsIndShockSolverEOP",
    "ConsIndShockSolver",
    "ConsKinkedRsolver",
    "ConsIndShockSetupEOP",
    "ConsIndShockSetup",
    "ConsKinkedRsolver",
]

# ConsumerSolutionOld below is identical to ConsumerSolution in HARK 0.11.0
# New definition of ConsumerSolution, later, augments it


class ConsumerSolutionOld(MetricObject):
    """
    A class representing the solution of a single period of a consumption-saving
    problem.  The solution must include a consumption function and marginal
    value function.
    Here and elsewhere in the code, Nrm indicates that variables are normalized
    by permanent income.
    Parameters
    ----------
    cFunc : function
        The consumption function for this period, defined over market
        resources: c = cFunc(m).
    vFunc : function
        The beginning-of-period value function for this period, defined over
        market resources: v = vFunc(m).
    vPfunc : function
        The beginning-of-period marginal value function for this period,
        defined over market resources: vP = vPfunc(m).
    vPPfunc : function
        The beginning-of-period marginal marginal value function for this
        period, defined over market resources: vPP = vPPfunc(m).
    mNrmMin : float
        The minimum allowable market resources for this period; the consump-
        tion function (etc) are undefined for m < mNrmMin.
    hNrm : float
        Human wealth after receiving income this period: PDV of all future
        income, ignoring mortality.
    MPCmin : float
        Infimum of the marginal propensity to consume this period.
        MPC --> MPCmin as m --> infinity.
    MPCmax : float
        Supremum of the marginal propensity to consume this period.
        MPC --> MPCmax as m --> mNrmMin.
    """

    distance_criteria = ["vPfunc"]

    def __init__(self, cFunc=None, vFunc=None, vPfunc=None, vPPfunc=None,
                 mNrmMin=None, hNrm=None, MPCmin=None, MPCmax=None, *args, **kwargs):
        # Change any missing function inputs to NullFunc
        self.cFunc = cFunc if cFunc is not None else NullFunc()
        self.vFunc = vFunc if vFunc is not None else NullFunc()
        self.vPfunc = vPfunc if vPfunc is not None else NullFunc()
        self.vPPfunc = vPPfunc if vPPfunc is not None else NullFunc()
        self.mNrmMin = mNrmMin
        self.hNrm = hNrm
        self.MPCmin = MPCmin
        self.MPCmax = MPCmax

    def append_solution(self, new_solution):
        """
        Appends one solution to another to create a ConsumerSolution whose
        attributes are lists.  Used in ConsMarkovModel, where we append solutions
        *conditional* on a particular value of a Markov state to each other in
        order to get the entire solution.
        Parameters
        ----------
        new_solution : ConsumerSolution
            The solution to a consumption-saving problem; each attribute is a
            list representing state-conditional values or functions.
        Returns
        -------
        None
        """
        if type(self.cFunc) != list:
            # Then we assume that self is an empty initialized solution instance.
            # Begin by checking this is so.
            assert (
                NullFunc().distance(self.cFunc) == 0
            ), "append_solution called incorrectly!"

            # We will need the attributes of the solution instance to be lists.  Do that here.
            self.cFunc = [new_solution.cFunc]
            self.vFunc = [new_solution.vFunc]
            self.vPfunc = [new_solution.vPfunc]
            self.vPPfunc = [new_solution.vPPfunc]
            self.mNrmMin = [new_solution.mNrmMin]
        else:
            self.cFunc.append(new_solution.cFunc)
            self.vFunc.append(new_solution.vFunc)
            self.vPfunc.append(new_solution.vPfunc)
            self.vPPfunc.append(new_solution.vPPfunc)
            self.mNrmMin.append(new_solution.mNrmMin)



class ConsumerSolution(ConsumerSolutionOld):
    __doc__ = ConsumerSolutionOld.__doc__
    __doc__ += """
    stge_kind : dict
        Dictionary with info about this stage
        One built-in entry keeps track of the nature of the stage:
            {'iter_status':'finished'}: Stopping requirements are satisfied
                If stopping requirements are satisfied, {'tolerance':tolerance}
                should exist recording what convergence tolerance was satisfied
            {'iter_status':'iterator'}: Solution during iteration
                solution[0].distance_last records the last distance
            {'iter_status':'terminal_pseudo'}: Bare-bones terminal period
                Does not contain all the info needed to begin solution
                Solver will augment and replace it with 'iterator' stage
        Other uses include keeping track of the nature of the next stage
    parameters_solver : dict
        Stores the parameters with which the solver was called
    """

    # CDC 20210426: vPfunc is a bad choice; we should change it,
    # but doing so will require recalibrating some of our tests
    distance_criteria = ["vPfunc"]  # Bad: it goes to infinity; would be better to use:
#    distance_criteria = ["mNrmStE"]  # mNrmStE if the GIC holds (and it's not close)
#    distance_criteria = ["cFunc"]  # cFunc if the GIC fails

    def __init__(
            self, cFunc=NullFunc(), vFunc=NullFunc(), vPfunc=NullFunc(),
            vPPfunc=NullFunc(), mNrmMin=float('nan'), hNrm=float('nan'),
            MPCmin=float('nan'), MPCmax=float('nan'),
            stge_kind={'iter_status': 'not initialized'}, parameters_solver=None,
            completed_cycles=0, ** kwds,):

        bilt = self.bilt = Built()
        bilt.cFunc = cFunc
        bilt.vFunc = vFunc
        bilt.vPfunc = vPfunc
        bilt.vPPfunc = vPPfunc
        bilt.mNrmMin = mNrmMin
        bilt.hNrm = hNrm
        bilt.MPCmin = MPCmin
        bilt.MPCmax = MPCmax
        bilt.stge_kind = stge_kind
        bilt.completed_cycles = completed_cycles

    def append_solution(self, new_solution):
        """
        Appends one solution to another to create a ConsumerSolution whose
        attributes are lists.  Used in ConsMarkovModel, where we append solutions
        *conditional* on a particular value of a Markov state to each other in
        order to get the entire solution.
        Parameters
        ----------
        new_solution : ConsumerSolution
            The solution to a consumption-saving problem; each attribute is a
            list representing state-conditional values or functions.
        Returns
        -------
        None
        """
        if type(self.cFunc) != list:
            # Then we assume that self is an empty initialized solution instance.
            # Begin by checking this is so.
            assert (
                NullFunc().distance(self.cFunc) == 0
            ), "append_solution called incorrectly!"

            # We will need the attributes of the solution instance to be lists.  Do that here.
            self.cFunc = [new_solution.cFunc]
            self.vFunc = [new_solution.vFunc]
            self.vPfunc = [new_solution.vPfunc]
            self.vPPfunc = [new_solution.vPPfunc]
            self.mNrmMin = [new_solution.mNrmMin]
        else:
            self.cFunc.append(new_solution.cFunc)
            self.vFunc.append(new_solution.vFunc)
            self.vPfunc.append(new_solution.vPfunc)
            self.vPPfunc.append(new_solution.vPPfunc)
            self.mNrmMin.append(new_solution.mNrmMin)

# CDC 20210509: ConsumerSolutionPlus was added as a stage above
# ConsumerSolution because ConsumerSolution should
# generically handle any and ALL general consumption problems
#  and PerfForesightCRRA should handle the subsubclass that is both PF and CRRA
# Also as a place to instantiate the stge_kind attribute, which should
# ultimately move upstream to become a core attribute of ANY solution


# class ConsumerSolution_ConsPerfForesightSolver(ConsumerSolutionOneStateCRRA):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)  # https://elfi-y.medium.com/super-inherit-your-python-class-196369e3377a
#         breakpoint()
#class ConsumerSolutionOneStateCRRA(ConsumerSolutionOld):
class ConsumerSolutionOneStateCRRA(ConsumerSolution):
    """
    This subclass of ConsumerSolution assumes that the problem has two
    additional characteristics:

        * Constant Relative Risk Aversion (CRRA) utility
        * Geometric Discounting of Time Separable Utility

    along with a standard set of restrictions on the parameter values of the
    model (like, the time preference factor must be positive).  Under various
    combinations of these assumptions, various conditions imply different
    conclusions.  The suite of minimal restrictions is always evaluated.  The set
    of conditions is evaluated using the `check_conditions` method.  (Further
    information about the conditions can be found in the documentation for
    that method.)  For convenience, we repeat below the documentation for the
    parent ConsumerSolution of this class, all of which applies here.
    """
    __doc__ += ConsumerSolution.__doc__
    __doc__ += """
    stge_kind : dict
        Dictionary with info about this stage
        One built-in entry keeps track of the nature of the stage:
            {'iter_status':'finished'}: Stopping requirements are satisfied
                If stopping requirements are satisfied, {'tolerance':tolerance}
                should exist recording what convergence tolerance was satisfied
            {'iter_status':'iterator'}: Solution during iteration
                solution[0].distance_last records the last distance
            {'iter_status':'terminal_pseudo'}: Bare-bones terminal period
                Does not contain all the info needed to begin solution
                Solver will augment and replace it with 'iterator' stage
        Other uses include keeping track of the nature of the next stage
    completed_cycles : A count of the number of iterations executed so far
    parameters_solver : dict
        Stores the parameters with which the solver was called
    """
    time_vary_ = ["LivPrb",  # Age-varying death rates can match mortality data
                  "PermGroFac"]  # Age-varying income growth can match lifecycle
    time_inv_ = ["CRRA", "Rfree", "DiscFac", "BoroCnstArt"]
    state_vars = ['pLvl',  # Initial idiosyncratic permanent income
                  'PlvlAgg',  # Aggregate permanent income
                  'bNrm',  # Bank balances beginning of period (pLvl normed)
                  'mNrm',  # Market resources (b + income) (pLvl normed)
                  "aNrm"]  # Assets after all actions (pLvl normed)
    shock_vars_ = []
    
    def __init__(self, *args,
                 stge_kind={'iter_status': 'not initialized'},
                 completed_cycles=0,
                 parameters_solver=None,
                 CRRA=2.0,
                 u=CRRAutility,
                 uP=CRRAutilityP,
                 uPP=CRRAutilityPP,
                 vAdd=None,
                 **kwds):

        ConsumerSolutionOld.__init__(self, *args, **kwds)
        # We are now in position to define some elements of the
        # dolo representation of the model
        # dolo = self.dolo
        # # These will be updated when more specific assumptions are made
        # dolo.model_name = 'ConsumerSolutionOneStateCRRA'
        # dolo.symbols = {'states': ['mNrm'],
        #                 'controls': ['cNrm'],
        #                 'parameters': ['ρ', 'β', 'Π', 'Rfree', 'Γ'],
        #                 'expectations': ['Euler_cNrm'],
        #                 'transition': ['DBC_mNrm'],
        #                 }

#        global Rfree, PermGroFac, MPCmin, MaxKinks, BoroCnstArt, DiscFac, \
#                Ex_IncNrmNxt, LivPrb, DiscLiv

        # Put into bilt a lot of stuff that was on the whiteboard root level
        bilt = self.bilt = Built()
#        breakpoint()
        
        self = def_utility(self, CRRA)

        bilt.vAdd = vAdd

        bilt.parameters_solver = None
        bilt.cFunc = self.cFunc
        bilt.vFunc = self.vFunc
        bilt.vFunc.dm = self.vPfunc
        bilt.vFunc.ddm = self.vPPfunc
        bilt.u = u
        bilt.u.dc = uP
        bilt.u.ddc = uPP
        bilt.mNrmMin = self.mNrmMin
        bilt.hNrm = self.hNrm
        bilt.MPCmin = self.MPCmin
        bilt.MPCmax = self.MPCmax
        
        del self.mNrmMin
        del self.hNrm
        del self.MPCmin
        del self.MPCmax
        del self.vFunc
        del self.vPfunc
        del self.vPPfunc
        
        bilt.completed_cycles = completed_cycles
        bilt.parameters_solver = parameters_solver
        

    def check_conditions(self, soln_crnt, verbose=None):
        """
        Checks whether the instance's type satisfies the:

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
        verbose: boolean
        Specifies different levels of verbosity of feedback. When False, it only reports whether the
        instance's type fails to satisfy a particular condition. When True, it reports all results, i.e.
        the factor values for all conditions.

        soln_crnt: ConsumerSolution
        Contains the solution to the problem described by information
        for the current stage found in bilt and the succeeding stage found
        in scsr.

        Returns
        -------
        None
        """
        soln_crnt.bilt.conditions = {}  # Keep track of truth of conditions
        soln_crnt.bilt.degenerate = False  # True means solution is degenerate

        if not hasattr(self, 'verbose'):  # If verbose not set yet
            verbose = 0 if verbose is None else verbose
        else:
            verbose = verbose if verbose is None else verbose

        msg = '\nFor a model with the following parameter values:\n'
        msg = msg+'\n'+str(soln_crnt.bilt.parameters_solver)+'\n'

        if verbose >= 2:
            _log.info(msg)
            _log.info(str(soln_crnt.bilt.parameters_solver))
            np.set_printoptions(threshold=20)  # Don't print huge output
            for key in soln_crnt.bilt.parameters_solver.keys():
                print('\t'+key+': ', end='')
                pprint(soln_crnt.bilt.parameters_solver[key])

        msg = '\nThe following results hold:\n'
        if verbose >= 2:
            _log.info(msg)

        soln_crnt.check_AIC(soln_crnt, verbose)
        soln_crnt.check_FHWC(soln_crnt, verbose)
        soln_crnt.check_RIC(soln_crnt, verbose)
        soln_crnt.check_GICRaw(soln_crnt, verbose)
        soln_crnt.check_GICNrm(soln_crnt, verbose)
        soln_crnt.check_GICLiv(soln_crnt, verbose)
        soln_crnt.check_FVAC(soln_crnt, verbose)

        # degenerate flag is true if the model has no nondegenerate solution
        if hasattr(soln_crnt.bilt, "BoroCnstArt") \
                and soln_crnt.bilt.BoroCnstArt is not None:
            soln_crnt.degenerate = not soln_crnt.bilt.RIC
            # If BoroCnstArt exists but RIC fails, limiting soln is c(m)=0
        else:  # If no constraint,
            soln_crnt.degenerate = not soln_crnt.bilt.RIC or \
                not soln_crnt.bilt.FHWC    # c(m)=0 or \infty

    def check_AIC(self, stge, verbose=None):
        """
        Evaluate and report on the Absolute Impatience Condition
        """
        name = "AIC"

        def test(stge): return stge.bilt.APF < 1

        messages = {
            True: "\n\nThe Absolute Patience Factor for the supplied parameter values, APF={0.APF}, satisfies the Absolute Impatience Condition (AIC), which requires APF < 1:\n    "+stge.bilt.AIC_fcts['urlhandle'],
            False: "\n\nThe Absolute Patience Factor for the supplied parameter values, APF={0.APF}, violates the Absolute Impatience Condition (AIC), which requires APF < 1:\n    "+stge.bilt.AIC_fcts['urlhandle']
        }
        verbose_messages = {
            True: "\n  Because the APF < 1,  the absolute amount of consumption is expected to fall over time.  \n",
            False: "\n  Because the APF > 1, the absolute amount of consumption is expected to grow over time.  \n",
        }

#        stge.AIC =
        stge.bilt.AIC = core_check_condition(name, test, messages, verbose,
                                             verbose_messages, "APF", stge)

    def check_FVAC(self, stge, verbose=None):
        """
        Evaluate and report on the Finite Value of Autarky Condition
        """
        name = "FVAC"
        def test(stge): return stge.bilt.FVAF < 1

        messages = {
            True: "\n\nThe Finite Value of Autarky Factor for the supplied parameter values, FVAF={0.FVAF}, satisfies the Finite Value of Autarky Condition, which requires FVAF < 1:\n    "+stge.bilt.FVAC_fcts['urlhandle'],
            False: "\n\nThe Finite Value of Autarky Factor for the supplied parameter values, FVAF={0.FVAF}, violates the Finite Value of Autarky Condition, which requires FVAF:\n    "+stge.bilt.FVAC_fcts['urlhandle']
        }
        verbose_messages = {
            True: "\n  Therefore, a nondegenerate solution exists if the RIC also holds. ("+stge.bilt.FVAC_fcts['urlhandle']+")\n",
            False: "\n  Therefore, a nondegenerate solution exits if the RIC holds, but will not exist if the RIC fails unless the FHWC also fails.\n",
        }

        stge.FVAC = stge.bilt.FVAC = core_check_condition(name, test, messages, verbose,
                                                          verbose_messages, "FVAF", stge)

    def check_GICRaw(self, stge, verbose=None):
        """
        Evaluate and report on the Growth Impatience Condition
        """
        name = "GICRaw"

        def test(stge): return stge.bilt.GPFRaw < 1

        messages = {
            True: "\n\nThe Growth Patience Factor for the supplied parameter values, GPF={0.GPFRaw}, satisfies the Growth Impatience Condition (GIC), which requires GPF < 1:\n    "+stge.bilt.GICRaw_fcts['urlhandle'],
            False: "\n\nThe Growth Patience Factor for the supplied parameter values, GPF={0.GPFRaw}, violates the Growth Impatience Condition (GIC), which requires GPF < 1:\n    "+stge.bilt.GICRaw_fcts['urlhandle'],
        }
        verbose_messages = {
            True: "\n  Therefore,  for a perfect foresight consumer, the ratio of individual wealth to permanent income is expected to fall indefinitely.    \n",
            False: "\n  Therefore, for a perfect foresight consumer whose parameters satisfy the FHWC, the ratio of individual wealth to permanent income is expected to rise toward infinity. \n"
        }
        stge.GICRaw = stge.bilt.GICRaw = core_check_condition(name, test, messages, verbose,
                                                              verbose_messages, "GPFRaw", stge)

    def check_GICLiv(self, stge, verbose=None):
        name = "GICLiv"

        def test(stge): return stge.bilt.GPFLiv < 1

        messages = {
            True: "\n\nThe Mortality Adjusted Aggregate Growth Patience Factor for the supplied parameter values, GPFLiv={0.GPFLiv}, satisfies the Mortality Adjusted Aggregate Growth Impatience Condition (GICLiv):\n    "+stge.bilt.GPFLiv_fcts['urlhandle'],
            False: "\n\nThe Mortality Adjusted Aggregate Growth Patience Factor for the supplied parameter values, GPFLiv={0.GPFLiv}, violates the Mortality Adjusted Aggregate Growth Impatience Condition (GICLiv):\n    "+stge.bilt.GPFLiv_fcts['urlhandle'],
        }
        verbose_messages = {
            True: "\n  Therefore, a target level of the ratio of aggregate market resources to aggregate permanent income exists ("+stge.bilt.GPFLiv_fcts['urlhandle']+")\n",
            False: "\n  Therefore, a target ratio of aggregate resources to aggregate permanent income may not exist ("+stge.bilt.GPFLiv_fcts['urlhandle']+")\n",
        }
#        stge.GICLiv =
        stge.bilt.GICLiv = core_check_condition(name, test, messages, verbose,
                                                verbose_messages, "GPFLiv", stge)

    def check_RIC(self, stge, verbose=None):
        """
        Evaluate and report on the Return Impatience Condition
        """

        name = "RIC"

        def test(stge): return stge.bilt.RPF < 1

        messages = {
            True: "\n\nThe Return Patience Factor for the supplied parameter values, RPF= {0.RPF}, satisfies the Return Impatience Condition (RIC), which requires RPF < 1:\n    "+stge.bilt.RPF_fcts['urlhandle'],
            False: "\n\nThe Return Patience Factor for the supplied parameter values, RPF= {0.RPF}, violates the Return Impatience Condition (RIC), which requires RPF < 1:\n    "+stge.bilt.RPF_fcts['urlhandle'],
        }
        verbose_messages = {
            True: "\n  Therefore, the limiting consumption function is not c(m)=0 for all m\n",
            False: "\n  Therefore, if the FHWC is satisfied, the limiting consumption function is c(m)=0 for all m.\n",
        }
#        stge.RIC =
        stge.bilt.RIC = core_check_condition(name, test, messages, verbose,
                                             verbose_messages, "RPF", stge)

    def check_FHWC(self, stge, verbose=None):
        """
        Evaluate and report on the Finite Human Wealth Condition
        """
        name = "FHWC"

        def test(stge): return stge.bilt.FHWF < 1

        messages = {
            True: "\n\nThe Finite Human Wealth Factor value for the supplied parameter values, FHWF={0.FHWF}, satisfies the Finite Human Wealth Condition (FHWC), which requires FHWF < 1:\n    "+stge.bilt.FHWC_fcts['urlhandle'],
            False: "\n\nThe Finite Human Wealth Factor value for the supplied parameter values, FHWF={0.FHWF}, violates the Finite Human Wealth Condition (FHWC), which requires FHWF < 1:\n    "+stge.bilt.FHWC_fcts['urlhandle'],
        }
        verbose_messages = {
            True: "\n  Therefore, the limiting consumption function is not c(m)=Infinity.\n  Human wealth normalized by permanent income is {0.hNrmInf}.\n",
            False: "\n  Therefore, the limiting consumption function is c(m)=Infinity for all m unless the RIC is also violated.\n  If both FHWC and RIC fail and the consumer faces a liquidity constraint, the limiting consumption function is nondegenerate but has a limiting slope of 0. ("+stge.bilt.FHWC_fcts['urlhandle']+")\n",
        }
#        stge.FHWC =
        stge.bilt.FHWC = core_check_condition(name, test, messages, verbose,
                                              verbose_messages, "FHWF", stge)

    def check_GICNrm(self, stge, verbose=None):
        """
        Check Normalized Growth Patience Factor.
        """
        if not hasattr(stge.bilt, 'IncShkDstn'):
            return  # GICNrm is same as GIC for PF consumer

        name = "GICNrm"

        def test(stge): return stge.bilt.GPFNrm <= 1

        messages = {
            True: "\n\nThe Normalized Growth Patience Factor GPFNrm for the supplied parameter values, GPFNrm={0.GPFNrm}, satisfies the Normalized Growth Impatience Condition (GICNrm), which requires GPFNrm < 1:\n    "+stge.bilt.GICNrm_fcts['urlhandle'],
            False: "\n\nThe Normalized Growth Patience Factor GPFNrm for the supplied parameter values, GPFNrm={0.GPFNrm}, violates the Normalized Growth Impatience Condition (GICNrm), which requires GPFNrm < 1:\n    "+stge.bilt.GICNrm_fcts['urlhandle'],
        }
        verbose_messages = {
            True: "\n  Therefore, a target level of the individual market resources ratio m exists.",
            False: "\n  Therefore, a target ratio of individual market resources to individual permanent income does not exist.  ("+stge.bilt.GICNrm_fcts['urlhandle']+")\n",
        }
#        stge.GICNrm =
        stge.bilt.GICNrm = core_check_condition(name, test, messages, verbose,
                                                verbose_messages, "GPFNrm", stge)

    def check_WRIC(self, stge, verbose=None):
        """
        Evaluate and report on the Weak Return Impatience Condition
        [url]/  # WRIC modified to incorporate LivPrb
        """

        if not hasattr(stge, 'IncShkDstn'):
            return  # WRIC is same as RIC for PF consumer

        name = "WRIC"

        def test(stge): return stge.bilt.WRPF <= 1

        messages = {
            True: "\n\nThe Weak Return Patience Factor value for the supplied parameter values, WRPF={0.WRPF}, satisfies the Weak Return Impatience Condition, which requires WRPF < 1:\n    "+stge.bilt.WRIC_fcts['urlhandle'],
            False: "\n\nThe Weak Return Patience Factor value for the supplied parameter values, WRPF={0.WRPF}, violates the Weak Return Impatience Condition, which requires WRPF < 1:\n    "+stge.bilt.WRIC_fcts['urlhandle'],
        }

        verbose_messages = {
            True: "\n  Therefore, a nondegenerate solution exists if the FVAC is also satisfied. ("+stge.bilt.WRIC_fcts['urlhandle']+")\n",
            False: "\n  Therefore, a nondegenerate solution is not available ("+stge.bilt.WRIC_fcts['urlhandle']+")\n",
        }
#        stge.WRIC =
        stge.bilt.WRIC = core_check_condition(name, test, messages, verbose,
                                              verbose_messages, "WRPF", stge)

    def mNrmTrg_find(self):
        """
        Finds value of(normalized) market resources mNrm at which individual consumer
        expects m not to change.

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

        # Minimum market resources plus next income is OK starting guess
        # Better would be to presere the last value (if it exists)
        # and use that as a starting point

        m_init_guess = self.bilt.mNrmMin + self.bilt.Ex_IncNrmNxt
        try:  # Find value where argument is zero
            self.bilt.mNrmTrg = find_zero_newton(
                self.Ex_m_tp1_minus_m_t,
                m_init_guess)
        except:
            self.bilt.mNrmTrg = None

        return self.bilt.mNrmTrg

    def mNrmStE_find(self):
        """
        Finds value of (normalized) market resources m at which individual consumer
        expects the level of market resources M to grow at the same rate as the level
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
            Same solution that was passed, but now with the attribute mNrmStE.
        """

        # Minimum market resources plus E[next income] is okay starting guess

        m_init_guess = self.bilt.mNrmMin + self.bilt.Ex_IncNrmNxt
        try:
            self.bilt.mNrmStE = find_zero_newton(
                self.Ex_permShk_tp1_times_m_tp1_minus_m_t, m_init_guess)
        except:
            self.bilt.mNrmStE = None

        # Add mNrmStE to the solution and return it
        return self.bilt.mNrmStE

# ConsPerfForesightSolver also incorporates calcs and info useful for
# models in which perfect foresight does not apply, because the contents
# of the PF model are inherited by a variety of non-perfect-foresight models
    def finish_setup_of_default_solution_terminal(self):
        """
        Add to `solution_terminal` characteristics of the agent required
        for solution of the particular type which are not automatically
        created as part of the definition of the generic `solution_terminal.`

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -------
        None
        """
        # If no solution exists for the agent,
        # core.py uses solution_terminal as solution_next

        solution_terminal_bilt = self.solution_terminal.bilt

        # Natural borrowing constraint: Cannot die in debt
        # Measured after income = tranShk*permShk/permShk received
        if not hasattr(solution_terminal_bilt, 'hNrm'):
            _log('warning: hNrm should be set in solution_terminal.')
            _log('assuming solution_terminal.hNrm = 0.')
            solution_terminal_bilt.hNrm = 0.
        solution_terminal_bilt.BoroCnstNat = -solution_terminal_bilt.hNrm

        # Define BoroCnstArt if not yet defined
        if not hasattr(self.parameters, 'BoroCnstArt'):
            solution_terminal_bilt.BoroCnstArt = None
        else:
            solution_terminal_bilt.BoroCnstArt = self.parameters.BoroCnstArt

        solution_terminal_bilt.stge_kind = {'iter_status': 'terminal_pseudo'}

        # Solution options
        if hasattr(self, 'vFuncBool'):
            solution_terminal_bilt.vFuncBool = self.parameters['vFuncBool']
        else:  # default to true
            solution_terminal_bilt.vFuncBool = True

        if hasattr(self, 'CubicBool'):
            solution_terminal_bilt.CubicBool = self.parameters['CubicBool']
        else:  # default to false (linear)
            solution_terminal_bilt.CubicBool = False

        solution_terminal_bilt.parameters = self.parameters
        CRRA = self.CRRA
        solution_terminal_bilt = def_utility(solution_terminal_bilt, CRRA)
        solution_terminal_bilt = def_value_funcs(solution_terminal_bilt, CRRA)

        return solution_terminal_bilt


#class ConsPerfForesightSolverEOP(MetricObject):
class ConsPerfForesightSolverEOP(ConsumerSolutionOneStateCRRA):
    """
    Solves a one period perfect foresight
    CRRA utility consumption-saving problem.

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
    # CDC 20200426: MaxKinks adds a good bit of complexity to little purpose
    # because everything it accomplishes could be done using a finite horizon
    # model (including tests of convergence conditions, which can be invoked
    # manually if a user wants them).

    def __init__(
            self, solution_next, DiscFac=1.0, LivPrb=1.0, CRRA=2.0, Rfree=1.0,
            PermGroFac=1.0, BoroCnstArt=None, MaxKinks=None, **kwds
    ):

        self.soln_futr = soln_futr = solution_next

        # objects whose _tp1 value is neeeded to solve problem of t:
        self.recursive = \
            {'cFunc', 'vFunc', 'vPfunc', 'vPPfunc',  # 'vFuncNvrs',
             'u', 'uP', 'uPP', 'uPinv', 'uPinvP', 'uinvP', 'uinv',
             'hNrm', 'mNrmMin', 'MPCmin', 'MPCmax', 'BoroCnstNat', 'CRRA'
             ,'vAdd'
             }

        self.soln_crnt = ConsumerSolutionOneStateCRRA()

        # Get solver parameters and store for later use
        # omitting things that would cause recursion
        parameters_solver = \
            {k: v for k, v in {**kwds, **locals()}.items()
             if k not in {'self', 'solution_next', 'kwds', 'soln_futr',
                          'bilt_futr', 'soln_crnt', 'bilt'}}

        # 'terminal' solution should replace pseuco_terminal:
        if hasattr(self.soln_futr.bilt, 'stge_kind') and \
                (soln_futr.bilt.stge_kind['iter_status'] == 'terminal_pseudo'):
            self.soln_crnt.bilt = deepcopy(self.soln_futr.bilt)

        # links for docs; urls are used when "fcts" are added
        self.url_doc_for_solver_get()

        self.soln_crnt.bilt.parameters_solver = deepcopy(parameters_solver)
        # Store the exact params with which solver was called
        # except for solution_next and self (to prevent inf recursion)
        for key in parameters_solver:
            setattr(self.soln_crnt.bilt, key, parameters_solver[key])
            setattr(self.soln_crnt.bilt, key, parameters_solver[key])
        return

    # Methods
    def url_doc_for_solver_get(self):
        # Generate a url that will locate the documentation
        self.class_name = self.__class__.__name__
        self.soln_crnt.bilt.url_ref = self.url_ref =\
            "https://econ-ark.github.io/BufferStockTheory"
        self.soln_crnt.bilt.urlroot = self.urlroot = \
            self.url_ref+'/#'
        self.soln_crnt.bilt.url_doc = self.url_doc = \
            "https://hark.readthedocs.io/en/latest/search.html?q=" +\
            self.class_name+"&check_keywords=yes&area=default#"

    def cFunc_from_vFunc(self, m):
        #        ρ = self.soln_crnt.bilt.CRRA
        vFuncNvrs = self.vFuncNvrs
        vInv = vFuncNvrs(m)
        vInvP = vFuncNvrs.derivative(m)
        cP = self.cFunc.derivative(m)
        cVal = vInv ** (cP / vInvP)
#        vP = (vFuncNvrs(m)**(-ρ))*vFuncNvrsP
#        cVal = vP**(-1/ρ)
        return cVal

        # mu = (c[m])**(-ρ)
        # ((1-ρ)*vFuncNvrs(m)**(-ρ))*vInvP = (c[m])**(-ρ)
        # (((1-ρ)*vFuncNvrs(m)**(-ρ))*vInvP)**(-1/ρ) = c[m]
        # vInvP = vFuncNvrs.derivative(m)
        # c = (((1-ρ)*vFuncNvrs(m)**(-ρ))*vInvP)**(-1/ρ)
        # uPinv = self.soln_crnt.bilt.uPinv
        # vFuncNvrs = self.vFuncNvrs
        # vP = vFuncNvrs.derivative(m)/(
        #     (((1/(1-CRRA))*vFuncNvrs(m))**(1-CRRA))**(-1+1/(1-CRRA))
        #     )
        # v = (((1-CRRA)**-1)*(vFuncNvrs(m)**(1-CRRA)))
        # vInvP = (((1-CRRA)*v)**(-1+1/(1-CRRA)))vP
        # vInvP = vFuncNvrs.derivative(m)
        # vP = (c[m]**-CRRA)
        # vInvP = (((1-CRRA)*v)**(-1+1/(1-CRRA)))*(c**-CRRA)
        # vFuncNvrs.derivative(m) = (((1-CRRA)*v)**(-1+1/(1-CRRA)))*(c**-CRRA)
        # ((vFuncNvrs.derivative(m)**(-1/CRRA))/(((1-CRRA)*v)**(-1+1/(1-CRRA)))) = c**-CRRA
        # c = \
        #     (((vFuncNvrs.derivative(m)**(-1/CRRA))/(((1-CRRA)*v)**(-1+1/(1-CRRA)))))**(-1/CRRA)
        # (((1-CRRA)*v)**(-1+1/(1-CRRA)))vP =
        # ((1-CRRA)*v)**(CRRA/(1-CRRA))
        # vP = (vFuncNvrs.derivative(m)/(
        #     (1-CRRA)*
        #     ((1-CRRA)*(vInv**(-CRRA/(1-CRRA)))) #
#        vP = (self.vNvrsFunc.derivative(m)/
#              (((1-CRRA)*self.vFunc(m))**(-1+1/(1-CRRA))))
#            (((1-CRRA)*self.vNvrsFunc(m))**(-1+1/(1-CRRA))))
#        cVal = vFuncNvrs.derivative(vP)
        return cVal

    def make_cFunc_PF(self):
        """
        Makes the (linear) consumption function for this period.  See the 
        PerfForesightConsumerType.ipynb notebook for derivations.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Reduce cluttered formulae with local aliases

        bild, folw = self.soln_crnt.bilt, self.soln_crnt.folw

        Rfree, PermGroFac, MPCmin, MaxKinks = \
            bild.Rfree, bild.PermGroFac, bild.MPCmin, bild.MaxKinks
        BoroCnstArt, DiscFac, Ex_IncNrmNxt, LivPrb = \
            bild.BoroCnstArt, bild.DiscFac, bild.Ex_IncNrmNxt, bild.LivPrb
        BoroCnstNat = bild.BoroCnstNat
        u = bild.u
        uP = bild.uP
        uinv = bild.uinv
        uPinv = bild.uPinv
        vFunc_tp1 = folw.vFunc_tp1
        vPfunc_tp1 = folw.vPfunc_tp1

        _PF_IncNrm_tp1 = Ex_IncNrmNxt
        DiscLiv = DiscFac * LivPrb
        CRRA = bild.CRRA
        CRRA_tp1 = folw.CRRA_tp1

        if BoroCnstArt is None:
            BoroCnstArt = -np.inf

        # Whichever constraint is tighter is the relevant one
        BoroCnst = max(BoroCnstArt, BoroCnstNat)

        # Omit first and last points which define extrapolation below and above
        # the kink points
        mNrm_kinks_tp1 = folw.cFunc_tp1.x_list[:-1][1:]
        cNrm_kinks_tp1 = folw.cFunc_tp1.y_list[:-1][1:]
        vAdd_tp1 = folw.vAdd_tp1
        vNrm_kinks_tp1 = folw.vFunc_tp1(mNrm_kinks_tp1)

        # Calculate end-of-this-period aNrm vals that would reach those mNrm's
        aNrm_kinks = (mNrm_kinks_tp1 - _PF_IncNrm_tp1)*(PermGroFac/Rfree)

        # Obtain c_t from which unconstrained consumers would land on each
        # kink next period by inverting FOC: c_t = (RβΠ)^(-1/ρ) c_tp1
        # This is the endogenous gridpoint (kink point) today
        # corresponding to each next-period kink (each of which corresponds
        # to a finite-horizon solution ending one more period in the future)

        cNrm_kinks = (((Rfree * DiscLiv) ** (-1/CRRA_tp1)) *
                      PermGroFac * cNrm_kinks_tp1)

        vNrm_kinks = (DiscLiv * PermGroFac**(1-CRRA))*vNrm_kinks_tp1

        mNrm_kinks = aNrm_kinks + cNrm_kinks

        vInv_kinks = uinv(vNrm_kinks)
        
        vAdd_kinks = mNrm_kinks-mNrm_kinks

        # _v_t(aNrm) is value as of the END of period t
        # _v_t'(aNrmMin) = RβΠ (Γ**(-ρ)) v_tp1'(bNrmMin+_PF_IncNrmNxt)
        mNrmMin_tp1 = _PF_IncNrm_tp1 + BoroCnst * (Rfree/PermGroFac)

        _v_t_at_BoroCnst = \
            (DiscLiv * PermGroFac**(1-CRRA_tp1) *
             vFunc_tp1(mNrmMin_tp1))

        _vP_t_at_BoroCnst = \
            ((Rfree * DiscLiv) * PermGroFac**(-CRRA_tp1) *
             vPfunc_tp1(mNrmMin_tp1))

        # h is the 'horizon': h_t(m_t) is the number of periods it will take
        # before you hit the constraint, after which you remain constrained
        
        # For any c_t where you are unconstrained today, value is the discounted
        # sum of values you will receive during periods between now and t+h, 
        # and values you will receive afer h
#        vAdd = # Sum of post-constrained value by gridpoint
#            (DiscLiv * PermGroFac**(1-CRRA))*\
#                (bild.u(folw.cFunc_tp1(mNrm_kinks_tp1) # u at next period cusp
#                        +vAdd_tp1) # v from s


        # cusp is point where current period constraint stops binding
        cNrm_cusp = uPinv(_vP_t_at_BoroCnst)
        vNrm_cusp = bild.u(cNrm_cusp)+_v_t_at_BoroCnst
        vAdd_cusp = _v_t_at_BoroCnst
        vInv_cusp = uinv(vNrm_cusp)
        uInv_cusp = uinv(bild.u(cNrm_cusp))
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
#            vInv_kinks = np.insert(vInv_kinks, 0, vInv_cusp)
#            vAdd_kinks = np.insert(vAdd_kinks, 0, vAdd_cusp)

        vAddGrid = np.append(vAdd_cusp,vAdd_kinks)
        vAddGrid = np.append(vAddGrid,0.)
        # vFuncNvrsSlopeLim = MPCmin ** (-CRRA / (1.0 - CRRA))
        # vFuncNvrs_vals = \
        #     np.array([vFuncNvrsSlopeLim*(mNrm_kinks-BoroCnstNat),
        #               vFuncNvrsSlopeLim*(mNrm_kinks-BoroCnstNat+1)])
        # vFuncNvrs_grid = np.array([mNrm_kinks,mNrm_kinks+1])
        # vFuncNvrs = LinearInterp(vFuncNvrs_grid,vFuncNvrs_vals)
        # vFuncNvrsIntercept = mNrm_kinks[-1]
        # vFuncNvrs_kinks = vFunc_tp1.vFuncNvrs(mNrm_kinks)
        
        # To guarantee meeting BoroCnst, if mNrm = BoroCnst then cNrm = 0.
        mNrmGrid_unconst = np.append(mNrm_kinks, mNrm_kinks+1)
        cNrmGrid_unconst = np.append(cNrm_kinks, cNrm_kinks+MPCmin)
        aNrmGrid_unconst = mNrmGrid_unconst-cNrmGrid_unconst
        mNrmGrid_tp1_unconst = aNrmGrid_unconst*(Rfree/PermGroFac)+_PF_IncNrm_tp1
        vNrmGrid_unconst = u(cNrmGrid_unconst)+(DiscLiv * PermGroFac**(1-CRRA_tp1) *
             vFunc_tp1(mNrmGrid_tp1_unconst))
        vInvGrid_unconst = uinv(vNrmGrid_unconst)
        vInvPGrid_unconst = \
            (((1-CRRA)*vNrmGrid_unconst)**(-1+1/(1-CRRA)))*(cNrmGrid_unconst**(-CRRA))
        c_from_vInvPGrid_unconst = \
            ((vInvPGrid_unconst/(((1-CRRA)*vNrmGrid_unconst)**(-1+1/(1-CRRA)))))**(-1/CRRA)

        mNrmGrid_const = np.array([BoroCnst,mNrm_cusp,mNrm_cusp+1])
        uNrmGrid_const = np.array([float('inf'),u(mNrm_cusp),float('inf')])
        uInvGrid_const = uinv(uNrmGrid_const)
        def vAddFunc(m,mNrmGrid,vAddGrid):
            mNrmGridPlus = np.append(mNrmGrid,float('inf'))
            vAddGridPlus = np.append(vAddGrid,vAddGrid[-1])
            from collections import Iterable
            if isinstance(m,Iterable):
                from itertools import repeat
                return np.array(list(map(lambda m, mNrmGridPlus, vAddGridPlus: \
                                         vAddGridPlus[np.where(m<mNrmGridPlus)[0][0]]
                                ,m
                                ,repeat(mNrmGridPlus)
                                ,repeat(vAddGridPlus))))
            else:
                return vAddGridPlus[np.where(m<mNrmGridPlus)[0][0]]
                
#        mPts = np.linspace(mNrmGrid[0],mNrmGrid[-1],10)
            
        vInvFunc_unconst = \
            LinearInterp(mNrmGrid_unconst,vInvGrid_unconst)
        
#        from HARK.utilities import plot_funcs
#        plot_funcs(lambda x: np.heaviside(x-BoroCnst,0.5),1,2)
        uInvFunc_const = \
            LinearInterp(mNrmGrid_const,uInvGrid_const)
        vFunc_const = bild.u(uInvGrid_const)+_v_t_at_BoroCnst
        vFunc_unconst = bild.u(vInvGrid_unconst)
        
        def vAddFunc(m,mGrid,vAddGrid):
            return vAddGrid[np.where(m<mGrid)[0][0]]
        
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
#        mNrmGrid_tp1 = bNrmGrid_tp1+_PF_IncNrm_tp1
#        vNrmGrid = u(cNrmGrid)+(DiscLiv * PermGroFac**(1-CRRA_tp1) *
#             vFunc_tp1(mNrmGrid_tp1))
#        vInvGrid = (vNrmGrid*(1-CRRA))**(1/(1-CRRA))

#        vInvGrid = np.append(vInvGrid, vInvGrid[-1]+MPCmin**(-CRRA/(1.0-CRRA)))

        # To guarantee meeting BoroCnst, if mNrm = BoroCnst then cNrm = 0.
        mNrmGrid = np.append([BoroCnst],mNrm_kinks)
        cNrmGrid = np.append(0.,cNrm_kinks)
        
        # Above last kink point, use PF solution
        mNrmGrid = np.append(mNrmGrid,mNrmGrid[-1]+1)
        cNrmGrid = np.append(cNrmGrid,cNrmGrid[-1]+MPCmin)
            

        self.cFunc = self.soln_crnt.cFunc = bild.cFunc = \
            LinearInterp(mNrmGrid, cNrmGrid)
            
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
#        PF_t_v_tp1_last = (DiscLiv*(PermGroFac ** (1-folw.CRRA_tp1)))*\
#            np.float(folw.vFunc_tp1((Rfree/PermGroFac)*aNrmGrid[-1]+Ex_IncNrmNxt))
#        PF_t_vNvrs_tp1_Grid_2 = \
#            np.append(PF_t_vNvrs_tp1_Grid,PF_t_v_tp1_last)

        #vNvrsGrid = bild.uinv(bild.u(cNrmGrid)+ folw.u_tp1(PF_t_vNvrs_tp1_Grid))

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
#        mNrmMin_with_income = mNrmMin + _PF_IncNrmNxt
 #       if c_at_aNrmMin > aNrmMin+_PF_IncNrmNxt-BoroCnst:
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

#         mNrmGrid_pts = np.insert(mNrmGrid,kink_min-1,Ex_IncNrmNxt)
#         cNrmGrid_pts = np.insert(cNrmGrid,kink_min-1,Ex_IncNrmNxt-BoroCnst)

#        mNrmGrid_pts = np.append(mNrmGrid_pts,mNrmGrid_pts[-1]+1.)
#        cNrmGrid_pts = np.append(cNrmGrid_pts,cNrmGrid_pts[-1]+MPCmin)


#         mNrmGrid = np.insert(mNrmGrid,0,mNrmMin)
#         cNrmGrid = np.insert(cNrmGrid,0,0.)


#         if BoroCnstArt+Ex_IncNrmNxt > mNrmGrid[0]:
#             mNrmGrid

#         mNrmGrid = np.append(mNrmGrid,mNrmGrid[-1]+1.0)
#         cNrmGrid = np.append(cNrmGrid,cNrmGrid[-1]+MPCmin)


#         # Add the point corresponding to
#         mNrmGrid = np.unique(np.insert(mNrmGrid,0,Ex_IncNrmNxt-BoroCnstArt))
#         cNrmGrid = np.unique(np.insert(cNrmGrid,0,Ex_IncNrmNxt-BoroCnstArt))


# #        vNvrs_tp1 = (DiscLiv * LivPrb) * folw.vFunc_tp1(mNrmGrid_tp1)
# #        PF_t_vNvrs_tp1_Grid = folw.uinv_tp1(DiscLiv) * \
# #            folw.vFuncNvrs_tp1.y_list
#         # Beginning-of-period-tp1 marginal value vec is vP_tp1
# #        vP_tp1 = folw.uP_tp1(cNrmGrid_tp1)
#         # Corresponding end-of-period-t marginal value is _vP_t
# #        _vP_t = ((DiscLiv * Rfree) * (PermGroFac**(-CRRA_tp1)))*vP_tp1
# #        _vP_t =
#         # Endogenous gridpoints method
#  #       cNrmGrid = bild.uPinv(_vP_t)    # EGM step 1: u' inverse yields c
#         mNrmGrid = aNrmGrid + cNrmGrid  # EGM step 2: DBC inverted

#         cNrmGrid = np.unique(np.insert(cNrmGrid,0,Ex_IncNrmNxt-BoroCnstArt))

#         # Add additional point to the list of gridpoints for extrapolation,
#         # using this period's new value of the lower bound of the MPC, which
#         # defines the PF unconstrained problem through the end of the horizon
# #        mNrmGrid_interp_pts = np.append(mNrmGrid, mNrmGrid[-1] + 1.0)
# #        cNrmGrid_interp_pts = np.append(cNrmGrid, cNrmGrid[-1] + MPCmin)
#         # If artificial borrowing constraint binds, combine constrained and
#         # unconstrained consumption functions.

#         # The problem is well-defined down to BoroCnstArt even if in
#         # principle from t you could not get to any m_tp1 < Ex_IncNrmNxt
#         # because nothing prevents you from starting tp1 with m \geq BoroCnstArt
#  #       if BoroCnstArt < mNrmGrid[0] - Ex_IncNrmNxt:
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
#        self.cFunc = self.soln_crnt.cFunc = bild.cFunc = \
#            LinearInterp(mNrmGrid_pts, cNrmGrid_pts)


#        PF_t_v_tp1_last = (DiscLiv*(PermGroFac ** (1-folw.CRRA_tp1)))*\
#            np.float(folw.vFunc_tp1((Rfree/PermGroFac)*aNrmGrid[-1]+Ex_IncNrmNxt))
#        PF_t_vNvrs_tp1_Grid_2 = \
#            np.append(PF_t_vNvrs_tp1_Grid,PF_t_v_tp1_last)

        #vNvrsGrid = bild.uinv(bild.u(cNrmGrid)+ folw.u_tp1(PF_t_vNvrs_tp1_Grid))

        # # Calculate the upper bound of the MPC as the slope of bottom segment
        # # In practice, this is always 1.  Code is here for clarity
        # bild.MPCmax = ((cNrmGrid_Xtra[1] - cNrmGrid_Xtra[0])/
        #                (mNrmGrid_Xtra[1] - mNrmGrid_Xtra[0]))

        # # Lower bound of mNrm is lowest gridpoint -- usually 0
        # bild.mNrmMin = mNrmGrid_Xtra[0]

        # # Add the calculated grids to self.bild
        # bild.aNrmGrid = aNrmGrid_Xtra
        # bild._vP_t = _vP_t
        # bild.cNrmGrid = cNrmGrid_Xtra
        # bild.mNrmGrid = mNrmGrid_Xtra

        # Add approximation to v and vP
#        breakpoint()
#        bild.vNvrs = self.soln_crnt.uinv(_vP_t)

#    def build_infhor_facts_from_params_ConsPerfForesightSolver(self):


    def build_infhor_facts_from_params(self):
        """
            Adds to the solution extensive information and references about
            its elements.

            Parameters
            ----------
            solution: ConsumerSolution
                A solution that already has minimal requirements (vPfunc, cFunc)

            Returns
            -------
            solution : ConsumerSolution
                Same solution that was provided, augmented with facts
        """

        # Using local variables makes formulae below more readable

        soln_crnt = self.soln_crnt
#        scsr = self.soln_crnt.scsr
#     breakpoint()
        bilt = self.soln_crnt.bilt
        folw = self.soln_crnt.folw
        urlroot = bilt.urlroot
        bilt.DiscLiv = bilt.DiscFac * bilt.LivPrb

        APF_fcts = {
            'about': 'Absolute Patience Factor'
        }
        py___code = '((Rfree * DiscLiv) ** (1.0 / CRRA))'
#        soln_crnt.APF = \
        bilt.APF = APF = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        APF_fcts.update({'latexexpr': r'\APF'})
        APF_fcts.update({'_unicode_': r'Þ'})
        APF_fcts.update({'urlhandle': urlroot+'APF'})
        APF_fcts.update({'py___code': py___code})
        APF_fcts.update({'value_now': APF})
        # soln_crnt.fcts.update({'APF': APF_fcts})
#        soln_crnt.APF_fcts = \
        bilt.APF_fcts = APF_fcts

        AIC_fcts = {
            'about': 'Absolute Impatience Condition'
        }
        AIC_fcts.update({'latexexpr': r'\AIC'})
        AIC_fcts.update({'urlhandle': urlroot+'AIC'})
        AIC_fcts.update({'py___code': 'test: APF < 1'})
        # soln_crnt.fcts.update({'AIC': AIC_fcts})
#        soln_crnt.AIC_fcts =
        bilt.AIC_fcts = AIC_fcts

        RPF_fcts = {
            'about': 'Return Patience Factor'
        }
        py___code = 'APF / Rfree'
#        soln_crnt.RPF = \
        bilt.RPF = RPF = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        RPF_fcts.update({'latexexpr': r'\RPF'})
        RPF_fcts.update({'_unicode_': r'Þ_R'})
        RPF_fcts.update({'urlhandle': urlroot+'RPF'})
        RPF_fcts.update({'py___code': py___code})
        RPF_fcts.update({'value_now': RPF})
        # soln_crnt.fcts.update({'RPF': RPF_fcts})
#        soln_crnt.RPF_fcts = \
        bilt.RPF_fcts = RPF_fcts

        RIC_fcts = {
            'about': 'Growth Impatience Condition'
        }
        RIC_fcts.update({'latexexpr': r'\RIC'})
        RIC_fcts.update({'urlhandle': urlroot+'RIC'})
        RIC_fcts.update({'py___code': 'test: RPF < 1'})
        # soln_crnt.fcts.update({'RIC': RIC_fcts})
#        soln_crnt.RIC_fcts = \
        bilt.RIC_fcts = RIC_fcts

        GPFRaw_fcts = {
            'about': 'Growth Patience Factor'
        }
        py___code = 'APF / PermGroFac'
#        soln_crnt.GPFRaw = \
        bilt.GPFRaw = GPFRaw = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        GPFRaw_fcts.update({'latexexpr': '\GPFRaw'})
        GPFRaw_fcts.update({'_unicode_': r'Þ_Γ'})
        GPFRaw_fcts.update({'urlhandle': urlroot+'GPFRaw'})
        GPFRaw_fcts.update({'py___code': py___code})
        GPFRaw_fcts.update({'value_now': GPFRaw})
        # soln_crnt.fcts.update({'GPFRaw': GPFRaw_fcts})
#        soln_crnt.GPFRaw_fcts = \
        bilt.GPFRaw_fcts = GPFRaw_fcts

        GICRaw_fcts = {
            'about': 'Growth Impatience Condition'
        }
        GICRaw_fcts.update({'latexexpr': r'\GICRaw'})
        GICRaw_fcts.update({'urlhandle': urlroot+'GICRaw'})
        GICRaw_fcts.update({'py___code': 'test: GPFRaw < 1'})
        # soln_crnt.fcts.update({'GICRaw': GICRaw_fcts})
#        soln_crnt.GICRaw_fcts = \
        bilt.GICRaw_fcts = GICRaw_fcts

        GPFLiv_fcts = {
            'about': 'Mortality-Adjusted Growth Patience Factor'
        }
        py___code = 'APF * LivPrb / PermGroFac'
#        soln_crnt.GPFLiv = \
        bilt.GPFLiv = GPFLiv = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        GPFLiv_fcts.update({'latexexpr': '\GPFLiv'})
        GPFLiv_fcts.update({'urlhandle': urlroot+'GPFLiv'})
        GPFLiv_fcts.update({'py___code': py___code})
        GPFLiv_fcts.update({'value_now': GPFLiv})
        # soln_crnt.fcts.update({'GPFLiv': GPFLiv_fcts})
#        soln_crnt.GPFLiv_fcts = \
        bilt.GPFLiv_fcts = GPFLiv_fcts

        GICLiv_fcts = {
            'about': 'Growth Impatience Condition'
        }
        GICLiv_fcts.update({'latexexpr': r'\GICLiv'})
        GICLiv_fcts.update({'urlhandle': urlroot+'GICLiv'})
        GICLiv_fcts.update({'py___code': 'test: GPFLiv < 1'})
        # soln_crnt.fcts.update({'GICLiv': GICLiv_fcts})
#        soln_crnt.GICLiv_fcts = \
        bilt.GICLiv_fcts = GICLiv_fcts

        PF_RNrm_fcts = {
            'about': 'Growth-Normalized PF Return Factor'
        }
        py___code = 'Rfree/PermGroFac'
#        soln_crnt.PF_RNrm = \
        bilt.PF_RNrm = PF_RNrm = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        PF_RNrm_fcts.update({'latexexpr': r'\PFRNrm'})
        PF_RNrm_fcts.update({'_unicode_': r'R/Γ'})
        PF_RNrm_fcts.update({'py___code': py___code})
        PF_RNrm_fcts.update({'value_now': PF_RNrm})
        # soln_crnt.fcts.update({'PF_RNrm': PF_RNrm_fcts})
#        soln_crnt.PF_RNrm_fcts = \
        bilt.PF_RNrm_fcts = PF_RNrm_fcts
#        soln_crnt.PF_RNrm = PF_RNrm

        Inv_PF_RNrm_fcts = {
            'about': 'Inv of Growth-Normalized PF Return Factor'
        }
        py___code = '1 / PF_RNrm'
#        soln_crnt.Inv_PF_RNrm = \
        bilt.Inv_PF_RNrm = Inv_PF_RNrm = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        Inv_PF_RNrm_fcts.update({'latexexpr': r'\InvPFRNrm'})
        Inv_PF_RNrm_fcts.update({'_unicode_': r'Γ/R'})
        Inv_PF_RNrm_fcts.update({'py___code': py___code})
        Inv_PF_RNrm_fcts.update({'value_now': Inv_PF_RNrm})
        # soln_crnt.fcts.update({'Inv_PF_RNrm': Inv_PF_RNrm_fcts})
#        soln_crnt.Inv_PF_RNrm_fcts = \
        bilt.Inv_PF_RNrm_fcts = \
            Inv_PF_RNrm_fcts

        FHWF_fcts = {
            'about': 'Finite Human Wealth Factor'
        }
        py___code = 'PermGroFac / Rfree'
#        soln_crnt.FHWF = \
        bilt.FHWF = FHWF = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        FHWF_fcts.update({'latexexpr': r'\FHWF'})
        FHWF_fcts.update({'_unicode_': r'R/Γ'})
        FHWF_fcts.update({'urlhandle': urlroot+'FHWF'})
        FHWF_fcts.update({'py___code': py___code})
        FHWF_fcts.update({'value_now': FHWF})
        # soln_crnt.fcts.update({'FHWF': FHWF_fcts})
#        soln_crnt.FHWF_fcts = \
        bilt.FHWF_fcts = \
            FHWF_fcts

        FHWC_fcts = {
            'about': 'Finite Human Wealth Condition'
        }
        FHWC_fcts.update({'latexexpr': r'\FHWC'})
        FHWC_fcts.update({'urlhandle': urlroot+'FHWC'})
        FHWC_fcts.update({'py___code': 'test: FHWF < 1'})
        # soln_crnt.fcts.update({'FHWC': FHWC_fcts})
#        soln_crnt.FHWC_fcts = \
        bilt.FHWC_fcts = FHWC_fcts

        hNrmInf_fcts = {
            'about': 'Human wealth for inf hor'
        }
        py___code = '1/(1-FHWF) if (FHWF < 1) else float("inf")'
#        soln_crnt.hNrmInf = \
        bilt.hNrmInf = hNrmInf = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        hNrmInf_fcts = dict({'latexexpr': '1/(1-\FHWF)'})
        hNrmInf_fcts.update({'value_now': hNrmInf})
        hNrmInf_fcts.update({'py___code': py___code})
        # soln_crnt.fcts.update({'hNrmInf': hNrmInf_fcts})
#        soln_crnt.hNrmInf_fcts = \
        bilt.hNrmInf_fcts = hNrmInf_fcts

        DiscGPFRawCusp_fcts = {
            'about': 'DiscFac s.t. GPFRaw = 1'
        }
        py___code = '( PermGroFac                       ** CRRA)/(Rfree)'
#        soln_crnt.DiscGPFRawCusp = \
        bilt.DiscGPFRawCusp = DiscGPFRawCusp = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        DiscGPFRawCusp_fcts.update({'latexexpr': '\PermGroFac^{\CRRA}/\Rfree'})
        DiscGPFRawCusp_fcts.update({'value_now': DiscGPFRawCusp})
        DiscGPFRawCusp_fcts.update({'py___code': py___code})
        # soln_crnt.fcts.update({'DiscGPFRawCusp': DiscGPFRawCusp_fcts})
#        soln_crnt.DiscGPFRawCusp_fcts = \
        bilt.DiscGPFRawCusp_fcts = \
            DiscGPFRawCusp_fcts

        DiscGPFLivCusp_fcts = {
            'about': 'DiscFac s.t. GPFLiv = 1'
        }
        py___code = '( PermGroFac                       ** CRRA)/(Rfree*LivPrb)'
#        soln_crnt.DiscGPFLivCusp = \
        bilt.DiscGPFLivCusp = DiscGPFLivCusp = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        DiscGPFLivCusp_fcts.update({'latexexpr': '\PermGroFac^{\CRRA}/(\Rfree\LivPrb)'})
        DiscGPFLivCusp_fcts.update({'value_now': DiscGPFLivCusp})
        DiscGPFLivCusp_fcts.update({'py___code': py___code})
        # soln_crnt.fcts.update({'DiscGPFLivCusp': DiscGPFLivCusp_fcts})
#        soln_crnt.DiscGPFLivCusp_fcts = \
        bilt.DiscGPFLivCusp_fcts = DiscGPFLivCusp_fcts

        FVAF_fcts = {  # overwritten by version with uncertainty
            'about': 'Finite Value of Autarky Factor'
        }
        py___code = 'LivPrb * DiscLiv'
#        soln_crnt.FVAF = \
        bilt.FVAF = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        FVAF_fcts.update({'latexexpr': r'\FVAFPF'})
        FVAF_fcts.update({'urlhandle': urlroot+'FVAFPF'})
        FVAF_fcts.update({'py___code': py___code})
        # soln_crnt.fcts.update({'FVAF': FVAF_fcts})
#        soln_crnt.FVAF_fcts = \
        bilt.FVAF_fcts = FVAF_fcts

        FVAC_fcts = {  # overwritten by version with uncertainty
            'about': 'Finite Value of Autarky Condition - Perfect Foresight'
        }
        FVAC_fcts.update({'latexexpr': r'\FVACPF'})
        FVAC_fcts.update({'urlhandle': urlroot+'FVACPF'})
        FVAC_fcts.update({'py___code': 'test: FVAFPF < 1'})
        # soln_crnt.fcts.update({'FVAC': FVAC_fcts})
#        soln_crnt.FVAC_fcts = \
        bilt.FVAC_fcts = FVAC_fcts

        Ex_IncNrmNxt_fcts = {  # Overwritten by version with uncertainty
            'about': 'Expected income next period'
        }
        py___code = '1.0'
#        soln_crnt.Ex_IncNrmNxt = \
        bilt.Ex_IncNrmNxt = Ex_IncNrmNxt = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
#        Ex_IncNrmNxt_fcts.update({'latexexpr': r'\Ex_IncNrmNxt'})
#        Ex_IncNrmNxt_fcts.update({'_unicode_': r'R/Γ'})
#        Ex_IncNrmNxt_fcts.update({'urlhandle': urlroot+'ExIncNrmNxt'})
        Ex_IncNrmNxt_fcts.update({'py___code': py___code})
        Ex_IncNrmNxt_fcts.update({'value_now': Ex_IncNrmNxt})
        # soln_crnt.fcts.update({'Ex_IncNrmNxt': Ex_IncNrmNxt_fcts})
        soln_crnt.Ex_IncNrmNxt_fcts = soln_crnt.bilt.Ex_IncNrmNxt_fcts = Ex_IncNrmNxt_fcts

        PF_RNrm_fcts = {
            'about': 'Expected Growth-Normalized Return'
        }
        py___code = 'Rfree / PermGroFac'
#        soln_crnt.PF_RNrm = \
        bilt.PF_RNrm = PF_RNrm = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        PF_RNrm_fcts.update({'latexexpr': r'\PFRNrm'})
        PF_RNrm_fcts.update({'_unicode_': r'R/Γ'})
        PF_RNrm_fcts.update({'urlhandle': urlroot+'PFRNrm'})
        PF_RNrm_fcts.update({'py___code': py___code})
        PF_RNrm_fcts.update({'value_now': PF_RNrm})
        # soln_crnt.fcts.update({'PF_RNrm': PF_RNrm_fcts})
#        soln_crnt.PF_RNrm_fcts = \
        bilt.PF_RNrm_fcts = PF_RNrm_fcts

        PF_RNrm_fcts = {
            'about': 'Expected Growth-Normalized Return'
        }
        py___code = 'Rfree / PermGroFac'
#        soln_crnt.PF_RNrm = \
        bilt.PF_RNrm = PF_RNrm = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        PF_RNrm_fcts.update({'latexexpr': r'\PFRNrm'})
        PF_RNrm_fcts.update({'_unicode_': r'R/Γ'})
        PF_RNrm_fcts.update({'urlhandle': urlroot+'PFRNrm'})
        PF_RNrm_fcts.update({'py___code': py___code})
        PF_RNrm_fcts.update({'value_now': PF_RNrm})
        # soln_crnt.fcts.update({'PF_RNrm': PF_RNrm_fcts})
#        soln_crnt.PF_RNrm_fcts = \
        bilt.PF_RNrm_fcts = PF_RNrm_fcts

        DiscLiv_fcts = {
            'about': 'Mortality-Inclusive Discounting'
        }
        py___code = 'DiscFac * LivPrb'
#        soln_crnt.DiscLiv = \
        bilt.DiscLiv = DiscLiv = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        DiscLiv_fcts.update({'latexexpr': r'\PFRNrm'})
        DiscLiv_fcts.update({'_unicode_': r'R/Γ'})
        DiscLiv_fcts.update({'urlhandle': urlroot+'PFRNrm'})
        DiscLiv_fcts.update({'py___code': py___code})
        DiscLiv_fcts.update({'value_now': DiscLiv})
        # soln_crnt.fcts.update({'DiscLiv': DiscLiv_fcts})
#        soln_crnt.DiscLiv_fcts = \
        bilt.DiscLiv_fcts = DiscLiv_fcts

#    def build_recursive_facts_ConsPerfForesightSolver(self):
    def build_recursive_facts(self):

        soln_crnt = self.soln_crnt
        bilt = self.soln_crnt.bilt
        folw = soln_crnt.folw
        urlroot = bilt.urlroot
        bilt.DiscLiv = bilt.DiscFac * bilt.LivPrb

#        breakpoint()
        hNrm_fcts = {
            'about': 'Human Wealth '
        }
        py___code = '((PermGroFac / Rfree) * (1.0 + hNrm_tp1))'
        if soln_crnt.stge_kind['iter_status'] == 'terminal_pseudo':  # kludge:
            #        if soln_crnt.bilt.stge_kind['iter_status'] == 'terminal_pseudo':  # kludge:
            soln_crnt.hNrm_tp1 = -1.0  # causes hNrm = 0 for final period
#        soln_crnt.hNrm = \
        bilt.hNrm = hNrm = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        hNrm_fcts.update({'latexexpr': r'\hNrm'})
        hNrm_fcts.update({'_unicode_': r'R/Γ'})
        hNrm_fcts.update({'urlhandle': urlroot+'hNrm'})
        hNrm_fcts.update({'py___code': py___code})
        hNrm_fcts.update({'value_now': hNrm})
        # soln_crnt.fcts.update({'hNrm': hNrm_fcts})
#        soln_crnt.hNrm_fcts = \
        bilt.hNrm_fcts = hNrm_fcts

        BoroCnstNat_fcts = {
            'about': 'Natural Borrowing Constraint'
        }
        py___code = '(mNrmMin_tp1 - tranShkMin)*(PermGroFac/Rfree)*permShkMin'
        if soln_crnt.stge_kind['iter_status'] == 'terminal_pseudo':  # kludge
            #        if soln_crnt.bilt.stge_kind['iter_status'] == 'terminal_pseudo':  # kludge
            py___code = 'hNrm'  # Presumably zero
#        soln_crnt.BoroCnstNat = \
        bilt.BoroCnstNat = BoroCnstNat = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        BoroCnstNat_fcts.update({'latexexpr': r'\BoroCnstNat'})
        BoroCnstNat_fcts.update({'_unicode_': r''})
        BoroCnstNat_fcts.update({'urlhandle': urlroot+'BoroCnstNat'})
        BoroCnstNat_fcts.update({'py___code': py___code})
        BoroCnstNat_fcts.update({'value_now': BoroCnstNat})
        # soln_crnt.fcts.update({'BoroCnstNat': BoroCnstNat_fcts})
#        soln_crnt.BoroCnstNat_fcts = \
        bilt.BoroCnstNat_fcts = BoroCnstNat_fcts

        BoroCnst_fcts = {
            'about': 'Effective Borrowing Constraint'
        }
        py___code = 'BoroCnstNat if (BoroCnstArt == None) else (BoroCnstArt if BoroCnstNat < BoroCnstArt else BoroCnstNat)'
#        soln_crnt.BoroCnst = \
        bilt.BoroCnst = BoroCnst = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        BoroCnst_fcts.update({'latexexpr': r'\BoroCnst'})
        BoroCnst_fcts.update({'_unicode_': r''})
        BoroCnst_fcts.update({'urlhandle': urlroot+'BoroCnst'})
        BoroCnst_fcts.update({'py___code': py___code})
        BoroCnst_fcts.update({'value_now': BoroCnst})
        # soln_crnt.fcts.update({'BoroCnst': BoroCnst_fcts})
#        soln_crnt.BoroCnst_fcts = \
        bilt.BoroCnst_fcts = BoroCnst_fcts

        # MPCmax is not a meaningful object in the PF model so is not created there
        # so create it here
        MPCmax_fcts = {
            'about': 'Maximal MPC in current period as m -> mNrmMin'
        }
        py___code = '1.0 / (1.0 + (RPF / MPCmax_tp1))'
        if soln_crnt.stge_kind['iter_status'] == 'terminal_pseudo':  # kludge:
            soln_crnt.MPCmax_tp1 = float('inf')  # causes MPCmax = 1 for final period
#        soln_crnt.MPCmax = \
        bilt.MPCmax = MPCmax = eval(
            py___code, {}, {**bilt.__dict__, **folw.__dict__})
        MPCmax_fcts.update({'latexexpr': r''})
        MPCmax_fcts.update({'urlhandle': urlroot+'MPCmax'})
        MPCmax_fcts.update({'py___code': py___code})
        MPCmax_fcts.update({'value_now': MPCmax})
        # soln_crnt.fcts.update({'MPCmax': MPCmax_fcts})
#        soln_crnt.bilt.MPCmax_fcts = \
        bilt.MPCmax_fcts = MPCmax_fcts

        mNrmMin_fcts = {
            'about': 'Min m is the max you can borrow'
        }
        py___code = 'BoroCnst'
#        soln_crnt.mNrmMin = \
        bilt.mNrmMin =  \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        mNrmMin_fcts.update({'latexexpr': r'\mNrmMin'})
        mNrmMin_fcts.update({'py___code': py___code})
        # soln_crnt.fcts.update({'mNrmMin': mNrmMin_fcts})
#        soln_crnt.mNrmMin_fcts = \
        bilt.mNrmMin_fcts = mNrmMin_fcts

        MPCmin_fcts = {
            'about': 'Minimal MPC in current period as m -> infty'
        }
        py___code = '1.0 / (1.0 + (RPF /MPCmin_tp1))'
        if soln_crnt.stge_kind['iter_status'] == 'terminal_pseudo':  # kludge:
            #        if soln_crnt.bilt.stge_kind['iter_status'] == 'terminal_pseudo':  # kludge:
            bilt.MPCmin_tp1 = float('inf')  # causes MPCmin = 1 for final period
#        soln_crnt.MPCmin = \
        bilt.MPCmin = MPCmin = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        MPCmin_fcts.update({'latexexpr': r''})
        MPCmin_fcts.update({'urlhandle': urlroot+'MPCmin'})
        MPCmin_fcts.update({'py___code': py___code})
        MPCmin_fcts.update({'value_now': MPCmin})
        # soln_crnt.fcts.update({'MPCmin': MPCmin_fcts})
#        soln_crnt.MPCmin_fcts = \
        bilt.MPCmin_fcts = MPCmin_fcts

        MPCmax_fcts = {
            'about': 'Maximal MPC in current period as m -> mNrmMin'
        }
        py___code = '1.0 / (1.0 + (RPF / MPCmax_tp1))'
        if soln_crnt.stge_kind['iter_status'] == 'terminal_pseudo':  # kludge:
            #        if soln_crnt.bilt.stge_kind['iter_status'] == 'terminal_pseudo':  # kludge:
            bilt.MPCmax_tp1 = float('inf')  # causes MPCmax = 1 for final period
#        soln_crnt.MPCmax = \
        bilt.MPCmax = MPCmax = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        MPCmax_fcts.update({'latexexpr': r''})
        MPCmax_fcts.update({'urlhandle': urlroot+'MPCmax'})
        MPCmax_fcts.update({'py___code': py___code})
        MPCmax_fcts.update({'value_now': MPCmax})
        # soln_crnt.fcts.update({'MPCmax': MPCmax_fcts})
#        soln_crnt.MPCmax_fcts = \
        bilt.MPCmax_fcts = MPCmax_fcts

        cFuncLimitIntercept_fcts = {
            'about': 'Vertical intercept of perfect foresight consumption function'}
        py___code = 'MPCmin * hNrm'
#        soln_crnt.cFuncLimitIntercept = \
        bilt.cFuncLimitIntercept = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        cFuncLimitIntercept_fcts.update({'py___code': py___code})
        cFuncLimitIntercept_fcts.update({'latexexpr': '\MPC \hNrm'})
#        cFuncLimitIntercept_fcts.update({'urlhandle': ''})
#        cFuncLimitIntercept_fcts.update({'value_now': cFuncLimitIntercept})
#        cFuncLimitIntercept_fcts.update({'cFuncLimitIntercept': cFuncLimitIntercept_fcts})
        soln_crnt.cFuncLimitIntercept_fcts = cFuncLimitIntercept_fcts

        cFuncLimitSlope_fcts = {
            'about': 'Slope of limiting consumption function'}
        py___code = 'MPCmin'
        cFuncLimitSlope_fcts.update({'py___code': 'MPCmin'})
        bilt.cFuncLimitSlope = soln_crnt.cFuncLimitSlope = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        cFuncLimitSlope_fcts.update({'py___code': py___code})
        cFuncLimitSlope_fcts = dict({'latexexpr': '\MPCmin'})
        cFuncLimitSlope_fcts.update({'urlhandle': '\MPC'})
#        cFuncLimitSlope_fcts.update({'value_now': cFuncLimitSlope})
#        stg_crt.fcts.update({'cFuncLimitSlope': cFuncLimitSlope_fcts})
        soln_crnt.cFuncLimitSlope_fcts = cFuncLimitSlope_fcts
        # That's the end of things that are identical for PF and non-PF models

        return soln_crnt

    def solve_prepared_stage(self):  # ConsPerfForesightSolver
        """
        Solves the one-period/stage perfect foresight consumption-saving problem.

        Parameters
        ----------
        None (all should be in self)

        Returns
        -------
        solution : ConsumerSolution
            The solution to this period/stage's problem.
        """
        soln_futr = self.soln_futr
        soln_crnt = self.soln_crnt
        soln_futr_bilt = soln_futr.bilt
        soln_crnt_bilt = soln_crnt.bilt
        CRRA = soln_crnt.bilt.CRRA

        if not hasattr(soln_futr.bilt, 'stge_kind'):
            print('no stge_kind')
            breakpoint()

        if soln_futr.bilt.stge_kind['iter_status'] == 'finished':
            breakpoint()
            # Should not have gotten here
            # because core.py tests whehter solution_last is 'finished'

        if soln_futr_bilt.stge_kind['iter_status'] == 'terminal_pseudo':
            # bare-bones default terminal solution does not have all the facts
            # we want, so add them
            #            breakpoint()
            soln_futr_bilt = soln_crnt_bilt = def_utility(soln_crnt, CRRA)
#            self.build_infhor_facts_from_params_ConsPerfForesightSolver()
            self.build_infhor_facts_from_params()
#            breakpoint()
            soln_futr = soln_crnt = def_value_funcs(soln_crnt, CRRA)
            # Now that they've been added, it's good to go as a source for iteration
            if not hasattr(soln_crnt.bilt, 'stge_kind'):
                print('No stge_kind')
                breakpoint()
            soln_crnt.bilt.stge_kind['iter_status'] = 'iterator'
            soln_crnt.stge_kind = soln_crnt.bilt.stge_kind
            self.soln_crnt.vPfunc = self.soln_crnt.bilt.vPfunc  # Need for distance
            self.soln_crnt.cFunc = self.soln_crnt.bilt.cFunc  # Need for distance
            if hasattr(self.soln_crnt.bilt, 'IncShkDstn'):
                self.soln_crnt.IncShkDstn = self.soln_crnt.bilt.IncShkDstn

#            breakpoint()
            return soln_crnt  # if pseudo_terminal = True, enhanced replaces original

        # self.soln_crnt.bilt.stge_kind = \
        #     self.soln_crnt.stge_kind = {'iter_status': 'iterator',
        #                                 'slvr_type': self.__class.__name}

#        breakpoint()
        CRRA = self.soln_crnt.bilt.CRRA
        self.soln_crnt = def_utility(soln_crnt, CRRA)
#        breakpoint()  # Need to build evPfut here, but previously had it building current
#        self.build_infhor_facts_from_params_ConsPerfForesightSolver()
        self.build_infhor_facts_from_params()
        self.build_recursive_facts()
        self.make_cFunc_PF()
        soln_crnt = def_value_funcs(soln_crnt, CRRA)
#        breakpoint()

        return soln_crnt

    solve = solve_prepared_stage

    def solver_prep_solution_for_an_iteration(self):  # self: solver for this stage
        """
        Prepare the current stage for processing by the one-stage solver.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
#        breakpoint()
        soln_crnt = self.soln_crnt
        soln_futr = self.soln_futr
#        breakpoint()
        bilt = soln_crnt.bilt

        # Organizing principle: folw should have a deepcopy of everything
        # needed to re-solve crnt problem; and everything needed to construct
        # the "fcts" about current stage of the problem, so that the stge could
        # be deepcopied as a standalone object and solved without soln_futr
        # or soln_crnt

        folw = soln_crnt.folw = SuccessorInfo()

        # Catch degenerate case of zero-variance income distributions
        if hasattr(bilt, "tranShkVals") and hasattr(bilt, "permShkVals"):
            if ((bilt.tranShkMin == 1.0) and (bilt.permShkMin == 1.0)):
                bilt.Ex_Inv_permShk = 1.0
                bilt.Ex_uInv_permShk = 1.0
        else:
            bilt.tranShkMin = bilt.permShkMin = 1.0

        if hasattr(bilt, 'stge_kind'):
            if 'iter_status' in bilt.stge_kind:
                if (bilt.stge_kind['iter_status'] == 'terminal_pseudo'):
                    # No work needed in terminal period, which replaces itself
                    return

        if not ('MPCmin' in soln_futr.bilt.__dict__):
            print('Breaking because no MPCmin')
            breakpoint()

        for key in (k for k in self.recursive
                    if k not in
                    {'solution_next', 'bilt', 'stge_kind', 'folw'}):
            setattr(folw, key+'_tp1',
                    soln_futr.bilt.__dict__[key])

        self.soln_crnt.bilt.stge_kind = \
            self.soln_crnt.stge_kind = {'iter_status': 'iterator',
                                        'slvr_type': self.__class__.__name__}

        return

    # Disambiguate confusing "prepare_to_solve" from similar method names elsewhere
    # (preserve "prepare_to_solve" as alias because core.py calls prepare_to_solve)
    prepare_to_solve = solver_prep_solution_for_an_iteration


###############################################################################
# ##############################################################################


class ConsPerfForesightSolver(ConsPerfForesightSolverEOP):
    def __init__(
            self, solution_next, DiscFac=1.0, LivPrb=1.0, CRRA=2.0, Rfree=1.0,
            PermGroFac=1.0, BoroCnstArt=None, MaxKinks=None, **kwds
    ):
        super().__init__(solution_next, DiscFac=DiscFac, LivPrb=LivPrb, CRRA=CRRA,
                         Rfree=Rfree, PermGroFac=PermGroFac,
                         BoroCnstArt=BoroCnstArt, MaxKinks=MaxKinks, **kwds)


class ConsIndShockSetupEOP(ConsPerfForesightSolver):
    """
    A superclass for solvers of one period consumption-saving problems with
    constant relative risk aversion utility and permanent and transitory shocks
    to income, containing code shared among alternative specific solvers.
    Has methods to set up but not solve the one period problem.

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
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported soln_crnt.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear inter-
        polation.
    """
    shock_vars = ['tranShkDstn', 'permShkDstn']  # Unemp shock is min(transShkVal)

    # TODO: CDC 20210416: Params shared with PF are in different order. Fix
    def __init__(
            self, solution_next, IncShkDstn, LivPrb, DiscFac, CRRA, Rfree, PermGroFac, BoroCnstArt, aXtraGrid, vFuncBool,
            CubicBool, permShkDstn, tranShkDstn, **kwds
    ):  # First execute PF solver init
        # We must reorder params by hand in case someone tries positional solve
        #        breakpoint()
        ConsPerfForesightSolver.__init__(self, solution_next, DiscFac=DiscFac, LivPrb=LivPrb, CRRA=CRRA,
                                         Rfree=Rfree, PermGroFac=PermGroFac, BoroCnstArt=BoroCnstArt, IncShkDstn=IncShkDstn, permShkDstn=permShkDstn, tranShkDstn=tranShkDstn, **kwds
                                         )

        # ConsPerfForesightSolver.__init__ makes self.soln_crnt
        # At this point it just has params copied from self.soln_futr, otherwise empty
#        breakpoint()

        soln_crnt = self.soln_crnt

        # Don't want to keep track of anything on self of disposable solver
        bilt = soln_crnt.bilt  # convenient local alias to reduce clutter

        # In which column is each object stored in IncShkDstn?
        permPos = IncShkDstn.parameters['ShkPosn']['perm']
        tranPos = IncShkDstn.parameters['ShkPosn']['tran']

        # Bcst are "broadcasted" values: serial list of every possible combo
        # Makes it easy to take expectations using 𝔼_dot
        bilt.permShkValsBcst = permShkValsBcst = IncShkDstn.X[permPos]
        bilt.tranShkValsBcst = tranShkValsBcst = IncShkDstn.X[tranPos]
        bilt.ShkPrbs = ShkPrbs = IncShkDstn.pmf

        bilt.permShkPrbs = permShkPrbs = permShkDstn.pmf
        bilt.permShkVals = permShkVals = permShkDstn.X
        # Confirm that perm shocks have expectation near one
        assert_approx_equal(𝔼_dot(permShkPrbs, permShkVals), 1.0)

        bilt.tranShkPrbs = tranShkPrbs = tranShkDstn.pmf
        bilt.tranShkVals = tranShkVals = tranShkDstn.X
        # Confirm that tran shocks have expectation near one
        assert_approx_equal(𝔼_dot(tranShkPrbs, tranShkVals), 1.0)

        bilt.permShkMin = permShkMin = np.min(permShkVals)
        bilt.tranShkMin = tranShkMin = np.min(tranShkVals)

        bilt.UnempPrb = tranShkPrbs[0]

        bilt.WorstIncPrb = np.sum(  # All cases where perm and tran Shk are Min
            ShkPrbs[ \
                permShkValsBcst * tranShkValsBcst == permShkMin * tranShkMin
            ]
        )
        bilt.WorstIncVal = permShkMin * tranShkMin

        bilt.aXtraGrid = aXtraGrid
        bilt.vFuncBool = vFuncBool
# e        breakpoint()
        bilt.CubicBool = CubicBool

    def build_infhor_facts_from_params(self):
        """
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
        super().build_infhor_facts_from_params()
        soln_crnt = self.soln_crnt
#        breakpoint()
        bilt = soln_crnt.bilt
        folw = soln_crnt.folw

        urlroot = bilt.urlroot
        # Modify formulae also present in PF model but that must change

        # Many other _fcts will have been inherited from the perfect foresight
        # model of which this model is a descendant
        # Here we need compute only those objects whose value changes
        # or does not exist when
        # the shock distributions are nondegenerate.
        Ex_IncNrmNxt_fcts = {
            'about': 'Expected income next period'
        }
        py___code = '𝔼_dot(ShkPrbs,tranShkValsBcst * permShkValsBcst)'
        bilt.𝔼_dot = 𝔼_dot  # add the expectations operator to envt
#        soln_crnt.Ex_IncNrmNxt = \
        bilt.Ex_IncNrmNxt = Ex_IncNrmNxt = eval(
            #        soln_crnt.Ex_IncNrmNxt = Ex_IncNrmNxt = eval(
            py___code, {}, {**bilt.__dict__, **folw.__dict__})
        Ex_IncNrmNxt_fcts.update({'latexexpr': r'\Ex_IncNrmNxt'})
        Ex_IncNrmNxt_fcts.update({'_unicode_': r'𝔼[\tranShk \permShk] = 1.0'})
        Ex_IncNrmNxt_fcts.update({'urlhandle': urlroot+'ExIncNrmNxt'})
        Ex_IncNrmNxt_fcts.update({'py___code': py___code})
        Ex_IncNrmNxt_fcts.update({'value_now': Ex_IncNrmNxt})
        # soln_crnt.fcts.update({'Ex_IncNrmNxt': Ex_IncNrmNxt_fcts})
        soln_crnt.Ex_IncNrmNxt_fcts = soln_crnt.bilt.Ex_IncNrmNxt_fcts = Ex_IncNrmNxt_fcts

        Ex_Inv_permShk_fcts = {
            'about': 'Expected Inverse of Permanent Shock'
        }
        py___code = '𝔼_dot(1/permShkVals, permShkPrbs)'
#        soln_crnt.Ex_Inv_permShk = \
        bilt.Ex_Inv_permShk = Ex_Inv_permShk = eval(
            py___code, {}, {**bilt.__dict__, **folw.__dict__})
        Ex_Inv_permShk_fcts.update({'latexexpr': r'\ExInvpermShk'})
#        Ex_Inv_permShk_fcts.update({'_unicode_': r'R/Γ'})
        Ex_Inv_permShk_fcts.update({'urlhandle': urlroot+'ExInvpermShk'})
        Ex_Inv_permShk_fcts.update({'py___code': py___code})
        Ex_Inv_permShk_fcts.update({'value_now': Ex_Inv_permShk})
        # soln_crnt.fcts.update({'Ex_Inv_permShk': Ex_Inv_permShk_fcts})
        soln_crnt.Ex_Inv_permShk_fcts = soln_crnt.bilt.Ex_Inv_permShk_fcts = Ex_Inv_permShk_fcts

        Inv_Ex_Inv_permShk_fcts = {
            'about': 'Inverse of Expected Inverse of Permanent Shock'
        }
        py___code = '1/Ex_Inv_permShk'
#        soln_crnt.Inv_Ex_Inv_permShk = \
        bilt.Inv_Ex_Inv_permShk = Inv_Ex_Inv_permShk = eval(
            py___code, {}, {**bilt.__dict__, **folw.__dict__})
        Inv_Ex_Inv_permShk_fcts.update(
            {'latexexpr': '\left(\Ex[\permShk^{-1}]\right)^{-1}'})
        Inv_Ex_Inv_permShk_fcts.update({'_unicode_': r'1/𝔼[Γψ]'})
        Inv_Ex_Inv_permShk_fcts.update({'urlhandle': urlroot+'InvExInvpermShk'})
        Inv_Ex_Inv_permShk_fcts.update({'py___code': py___code})
        Inv_Ex_Inv_permShk_fcts.update({'value_now': Inv_Ex_Inv_permShk})
        # soln_crnt.fcts.update({'Inv_Ex_Inv_permShk': Inv_Ex_Inv_permShk_fcts})
        soln_crnt.Inv_Ex_Inv_permShk_fcts = soln_crnt.bilt.Inv_Ex_Inv_permShk_fcts = Inv_Ex_Inv_permShk_fcts
        # soln_crnt.Inv_Ex_Inv_permShk = Inv_Ex_Inv_permShk

        Ex_RNrm_fcts = {
            'about': 'Expected Stochastic-Growth-Normalized Return'
        }
        py___code = 'PF_RNrm * Ex_Inv_permShk'
#        soln_crnt.Ex_RNrm = \
        bilt.Ex_RNrm = Ex_RNrm = eval(
            py___code, {}, {**bilt.__dict__, **folw.__dict__})
        Ex_RNrm_fcts.update({'latexexpr': r'\ExRNrm'})
        Ex_RNrm_fcts.update({'_unicode_': r'𝔼[R/Γψ]'})
        Ex_RNrm_fcts.update({'urlhandle': urlroot+'ExRNrm'})
        Ex_RNrm_fcts.update({'py___code': py___code})
        Ex_RNrm_fcts.update({'value_now': Ex_RNrm})
        # soln_crnt.fcts.update({'Ex_RNrm': Ex_RNrm_fcts})
#        soln_crnt.Ex_RNrm_fcts = \
        bilt.Ex_RNrm_fcts = Ex_RNrm_fcts

        Inv_Ex_RNrm_fcts = {
            'about': 'Inverse of Expected Stochastic-Growth-Normalized Return'
        }
        py___code = '1/Ex_RNrm'
#        soln_crnt.Inv_Ex_RNrm = \
        bilt.Inv_Ex_RNrm = Inv_Ex_RNrm = eval(
            py___code, {}, {**bilt.__dict__, **folw.__dict__})
        Inv_Ex_RNrm_fcts.update(
            {'latexexpr': '\InvExInvRNrm=\left(\Ex[\permShk^{-1}]\right)^{-1}'})
        Inv_Ex_RNrm_fcts.update({'_unicode_': r'1/𝔼[R/(Γψ)]'})
        Inv_Ex_RNrm_fcts.update({'urlhandle': urlroot+'InvExRNrm'})
        Inv_Ex_RNrm_fcts.update({'py___code': py___code})
        Inv_Ex_RNrm_fcts.update({'value_now': Inv_Ex_RNrm})
        # soln_crnt.fcts.update({'Inv_Ex_RNrm': Inv_Ex_RNrm_fcts})
#        soln_crnt.Inv_Ex_RNrm_fcts = \
        bilt.Inv_Ex_RNrm_fcts = Inv_Ex_RNrm_fcts

        Ex_uInv_permShk_fcts = {
            'about': 'Expected Utility for Consuming Permanent Shock'
        }
        py___code = '𝔼_dot(permShkValsBcst**(1-CRRA), ShkPrbs)'
#        soln_crnt.Ex_uInv_permShk = \
        bilt.Ex_uInv_permShk = Ex_uInv_permShk = eval(
            py___code, {}, {**bilt.__dict__, **folw.__dict__})
        Ex_uInv_permShk_fcts.update({'latexexpr': r'\ExuInvpermShk'})
        Ex_uInv_permShk_fcts.update({'urlhandle': r'ExuInvpermShk'})
        Ex_uInv_permShk_fcts.update({'py___code': py___code})
        Ex_uInv_permShk_fcts.update({'value_now': Ex_uInv_permShk})
        # soln_crnt.fcts.update({'Ex_uInv_permShk': Ex_uInv_permShk_fcts})
#        soln_crnt.Ex_uInv_permShk_fcts = \
        bilt.Ex_uInv_permShk_fcts = Ex_uInv_permShk_fcts

        uInv_Ex_uInv_permShk_fcts = {
            'about': 'Inverted Expected Utility for Consuming Permanent Shock'
        }
        py___code = '1/Ex_uInv_permShk'
#        soln_crnt.uInv_Ex_uInv_permShk = \
        bilt.uInv_Ex_uInv_permShk = uInv_Ex_uInv_permShk = eval(
            py___code, {}, {**bilt.__dict__, **folw.__dict__})
        uInv_Ex_uInv_permShk_fcts.update({'latexexpr': r'\uInvExuInvpermShk'})
        uInv_Ex_uInv_permShk_fcts.update({'urlhandle': urlroot+'uInvExuInvpermShk'})
        uInv_Ex_uInv_permShk_fcts.update({'py___code': py___code})
        uInv_Ex_uInv_permShk_fcts.update({'value_now': uInv_Ex_uInv_permShk})
        # soln_crnt.fcts.update({'uInv_Ex_uInv_permShk': uInv_Ex_uInv_permShk_fcts})
#        soln_crnt.uInv_Ex_uInv_permShk_fcts = \
        bilt.uInv_Ex_uInv_permShk_fcts = uInv_Ex_uInv_permShk_fcts

        GPFNrm_fcts = {
            'about': 'Normalized Expected Growth Patience Factor'
        }
        py___code = 'GPFRaw * Ex_Inv_permShk'
#        soln_crnt.GPFNrm = \
        bilt.GPFNrm = eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        GPFNrm_fcts.update({'latexexpr': r'\GPFNrm'})
        GPFNrm_fcts.update({'_unicode_': r'Þ_Γ'})
        GPFNrm_fcts.update({'urlhandle': urlroot+'GPFNrm'})
        GPFNrm_fcts.update({'py___code': py___code})
        # soln_crnt.fcts.update({'GPFNrm': GPFNrm_fcts})
#        soln_crnt.GPFNrm_fcts = \
        bilt.GPFNrm_fcts = GPFNrm_fcts

        GICNrm_fcts = {
            'about': 'Stochastic Growth Normalized Impatience Condition'
        }
        GICNrm_fcts.update({'latexexpr': r'\GICNrm'})
        GICNrm_fcts.update({'urlhandle': urlroot+'GICNrm'})
        GICNrm_fcts.update({'py___code': 'test: GPFNrm < 1'})
        # soln_crnt.fcts.update({'GICNrm': GICNrm_fcts})
#        soln_crnt.GICNrm_fcts = \
        bilt.GICNrm_fcts = GICNrm_fcts

        FVAC_fcts = {  # overwrites PF version
            'about': 'Finite Value of Autarky Condition'
        }

        FVAF_fcts = {  # overwrites PF version FVAFPF
            'about': 'Finite Value of Autarky Factor'
        }
        py___code = 'LivPrb * DiscLiv'
#        soln_crnt.FVAF = \
        bilt.FVAF = eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        FVAF_fcts.update({'latexexpr': r'\FVAF'})
        FVAF_fcts.update({'urlhandle': urlroot+'FVAF'})
        FVAF_fcts.update({'py___code': py___code})
        # soln_crnt.fcts.update({'FVAF': FVAF_fcts})
#        soln_crnt.FVAF_fcts = \
        bilt.FVAF_fcts = FVAF_fcts

        FVAC_fcts = {  # overwrites PF version
            'about': 'Finite Value of Autarky Condition'
        }
        FVAC_fcts.update({'latexexpr': r'\FVAC'})
        FVAC_fcts.update({'urlhandle': urlroot+'FVAC'})
        FVAC_fcts.update({'py___code': 'test: FVAF < 1'})
        # soln_crnt.fcts.update({'FVAC': FVAC_fcts})
#        soln_crnt.FVAC_fcts = \
        bilt.FVAC_fcts = FVAC_fcts

        WRPF_fcts = {
            'about': 'Weak Return Patience Factor'
        }
        py___code = '(UnempPrb ** (1 / CRRA)) * RPF'
#        soln_crnt.WRPF = \
        bilt.WRPF = WRPF = eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        WRPF_fcts.update({'latexexpr': r'\WRPF'})
        WRPF_fcts.update({'_unicode_': r'℘^(1/\rho) RPF'})
        WRPF_fcts.update({'urlhandle': urlroot+'WRPF'})
        WRPF_fcts.update({'value_now': WRPF})
        WRPF_fcts.update({'py___code': py___code})
        # soln_crnt.fcts.update({'WRPF': WRPF_fcts})
#        soln_crnt.WRPF_fcts = \
        bilt.WRPF_fcts = WRPF_fcts

        WRIC_fcts = {
            'about': 'Weak Return Impatience Condition'
        }
        WRIC_fcts.update({'latexexpr': r'\WRIC'})
        WRIC_fcts.update({'urlhandle': urlroot+'WRIC'})
        WRIC_fcts.update({'py___code': 'test: WRPF < 1'})
        # soln_crnt.fcts.update({'WRIC': WRIC_fcts})
#        soln_crnt.WRIC_fcts = \
        bilt.WRIC_fcts = WRIC_fcts

        DiscGPFNrmCusp_fcts = {
            'about': 'DiscFac s.t. GPFNrm = 1'
        }
        py___code = '((PermGroFac*Inv_Ex_Inv_permShk)**(CRRA))/Rfree'
#        soln_crnt.DiscGPFNrmCusp = \
        bilt.DiscGPFNrmCusp = DiscGPFNrmCusp = \
            eval(py___code, {}, {**bilt.__dict__, **folw.__dict__})
        DiscGPFNrmCusp_fcts.update({'latexexpr': ''})
        DiscGPFNrmCusp_fcts.update({'value_now': DiscGPFNrmCusp})
        DiscGPFNrmCusp_fcts.update({'py___code': py___code})
        # soln_crnt.fcts.update({'DiscGPFNrmCusp': DiscGPFNrmCusp_fcts})
#        soln_crnt.DiscGPFNrmCusp_fcts = \
        bilt.DiscGPFNrmCusp_fcts = DiscGPFNrmCusp_fcts

    def build_recursive_facts(self):
        super().build_recursive_facts()

        soln_crnt = self.soln_crnt
        # Now define some useful lambda functions

        # Given m, value of c where 𝔼[m_{t+1}]=m_{t}
        soln_crnt.c_where_Ex_mtp1_minus_mt_eq_0 = (
            lambda m_t:
            m_t * (1 - soln_crnt.Inv_Ex_RNrm) + (soln_crnt.Inv_Ex_RNrm)
        )

        # Given m, value of c where 𝔼[mLev_{t+1}/mLev_{t}]=soln_crnt.bilt.PermGroFac
        # Solves for c in equation at url/#balgrostable

        soln_crnt.c_where_Ex_permShk_times_mtp1_minus_mt_eq_0 = (
            lambda m_t:
            m_t * (1 - soln_crnt.bilt.Inv_PF_RNrm) + soln_crnt.bilt.Inv_PF_RNrm
        )

        # 𝔼[m_{t+1} pLev_{t+1}/pLev_{t}] as a fn of a_{t}
        soln_crnt.Ex_mLev_tp1_Over_pLev_t_from_a_t = (
            lambda a_t:
            𝔼_dot(soln_crnt.bilt.PermGroFac *
                  soln_crnt.bilt.permShkValsBcst *
                  (soln_crnt.bilt.PF_RNrm/soln_crnt.bilt.permShkValsBcst) * a_t
                  + soln_crnt.bilt.tranShkValsBcst,
                  soln_crnt.bilt.ShkPrbs)
        )

        # 𝔼[c_{t+1} pLev_{t+1}/pLev_{t}] as a fn of a_{t}
        soln_crnt.Ex_cLev_tp1_Over_pLev_t_from_a_t = (
            lambda a_t:
            𝔼_dot(soln_crnt.bilt.PermGroFac *
                  soln_crnt.bilt.permShkValsBcst *
                  soln_crnt.cFunc(
                      (soln_crnt.bilt.PF_RNrm/soln_crnt.bilt.permShkValsBcst) * a_t
                      + soln_crnt.bilt.tranShkValsBcst
                  ),
                  soln_crnt.bilt.ShkPrbs)
        )

        soln_crnt.c_where_Ex_mtp1_minus_mt_eq_0 = \
            lambda m_t: \
            m_t * (1 - 1/soln_crnt.bilt.Ex_RNrm) + (1/soln_crnt.bilt.Ex_RNrm)

        # Solve the equation at url/#balgrostable
        soln_crnt.c_where_Ex_permShk_times_mtp1_minus_mt_eq_0 = \
            lambda m_t: \
            (m_t * (1 - 1/soln_crnt.bilt.PF_RNrm)) + (1/soln_crnt.bilt.PF_RNrm)

        # mNrmTrg solves Ex_RNrm*(m - c(m)) + 𝔼[inc_next] - m = 0
        Ex_m_tp1_minus_m_t = (
            lambda m_t:
            soln_crnt.bilt.Ex_RNrm * (m_t - soln_crnt.cFunc(m_t)) +
            soln_crnt.bilt.Ex_IncNrmNxt - m_t
        )
        soln_crnt.Ex_m_tp1_minus_m_t = \
            soln_crnt.bilt.Ex_m_tp1_minus_m_t = Ex_m_tp1_minus_m_t

        soln_crnt.Ex_cLev_tp1_Over_pLev_t_from_num_a_t = (
            lambda a_t:
            𝔼_dot(
                soln_crnt.bilt.permShkValsBcst * soln_crnt.bilt.PermGroFac * soln_crnt.cFunc(
                    (soln_crnt.bilt.PF_RNrm/soln_crnt.bilt.permShkValsBcst) *
                    a_t + soln_crnt.bilt.tranShkValsBcst
                ),
                soln_crnt.bilt.ShkPrbs)
        )

        soln_crnt.Ex_cLev_tp1_Over_pLev_t_from_lst_a_t = (
            lambda a_lst: list(map(
                soln_crnt.Ex_cLev_tp1_Over_pLev_t_from_aNrm_num, a_lst
            ))
        )
        soln_crnt.bilt.Ex_cLev_tp1_Over_pLev_t_from_a_t = \
            soln_crnt.Ex_cLev_tp1_Over_pLev_t_from_a_t = (
                lambda a_t:
                soln_crnt.Ex_cLev_tp1_Over_pLev_t_from_lst_a_t(a_t)
                if (type(a_t) == list or type(a_t) == np.ndarray) else
                soln_crnt.Ex_cLev_tp1_Over_pLev_t_from_num_a_t(a_t)
            )

        soln_crnt.bilt.Ex_cLev_tp1_Over_pLev_t_from_lst_m_t = \
            soln_crnt.Ex_cLev_tp1_Over_pLev_t_from_lst_m_t = (
                lambda m_t:
                soln_crnt.Ex_cLev_tp1_Over_pLev_t_from_lst_a_t(m_t -
                                                               soln_crnt.cFunc(m_t))
            )

        soln_crnt.Ex_cLev_tp1_Over_pLev_t_from_num_m_t = \
            soln_crnt.bilt.Ex_cLev_tp1_Over_pLev_t_from_num_m_t = (
                lambda m_t:
                soln_crnt.Ex_cLev_tp1_Over_pLev_t_from_num_a_t(m_t -
                                                               soln_crnt.cFunc(m_t))
            )

        soln_crnt.bilt.Ex_cLev_tp1_Over_pLev_t_from_num_m_t = \
            soln_crnt.Ex_cLev_tp1_Over_pLev_t_from_num_m_t = (
                lambda m_t:
                soln_crnt.Ex_cLev_tp1_Over_pLev_t_from_num_a_t(m_t -
                                                               soln_crnt.cFunc(m_t))
            )

        soln_crnt.bilt.Ex_cLev_tp1_Over_cLev_t_from_m_t = \
            soln_crnt.Ex_cLev_tp1_Over_cLev_t_from_m_t = (
                lambda m_t:
                soln_crnt.Ex_cLev_tp1_Over_pLev_t_from_m_t(m_t) /
                soln_crnt.cFunc(m_t)
            )
        soln_crnt.Ex_permShk_tp1_times_m_tp1_minus_m_t = (
            lambda m_t:
            soln_crnt.bilt.PF_RNrm *
            (m_t - soln_crnt.cFunc(m_t)) + 1.0 - m_t
        )

        self.soln_crnt = soln_crnt

        return soln_crnt


class ConsIndShockSetup(ConsIndShockSetupEOP):
    pass

####################################################################################################
# ###################################################################################################


class ConsIndShockSolverBasicEOP(ConsIndShockSetupEOP):
    """
    This class solves a single period of a standard consumption-saving problem,
    using linear interpolation and without the ability to calculate the value
    function.  ConsIndShockSolver inherits from this class and adds the ability
    to perform cubic interpolation and to calculate the value function.

    Note that this class does not have its own initializing method.  It initial-
    izes the same problem in the same way as ConsIndShockSetup, from which it
    inherits.
    """

    def prepare_to_calc_EndOfPrdvP(self):
        """
        Prepare to calculate end-of-period marginal value by creating an array
        of market resources that the agent could have next period, given the
        current grid of end-of-period assets and the distribution of shocks
        they might experience next period.

        Parameters
        ----------
        none

        Returns
        -------
        aNrmGrid : np.array
            A 1D array of end-of-period assets; also stored as attribute of self.soln_crnt.bilt.
        """

        # We define aNrmGrid all the way from BoroCnstNat up to max(self.aXtraGrid)
        # even if BoroCnstNat < BoroCnstArt, so we can construct the consumption
        # function as the lower envelope of the (by the artificial borrowing con-
        # straint) unconstrained consumption function, and the artificially con-
        # strained consumption function.
        self.soln_crnt.bilt.aNrmGrid = np.asarray(
            self.soln_crnt.bilt.aXtraGrid) + self.soln_crnt.bilt.BoroCnstNat

        return self.soln_crnt.bilt.aNrmGrid

    def calc_EndOfPrdvP(self):
        """
        Calculate end-of-period marginal value of assets at each point in aNrm.
        Does so by taking a weighted sum of next period marginal values across
        income shocks (in a preconstructed grid self.soln_crnt.bilt.mNrmNext).

        Parameters
        ----------
        none

        Returns
        -------
        EndOfPrdvP : np.array
            A 1D array of end-of-period marginal value of assets
        """

        bilt = self.soln_crnt.bilt
        folw = self.soln_crnt.folw

        def vP_tp1(shk_vector, a_number):
            return shk_vector[0] ** (-bilt.CRRA) \
                * folw.vPfunc_tp1(self.m_Nrm_tp1(shk_vector, a_number))

        EndOfPrdvP = (
            bilt.DiscFac * bilt.LivPrb
            * bilt.Rfree
            * bilt.PermGroFac ** (-bilt.CRRA)
            * calc_expectation(
                bilt.IncShkDstn,
                vP_tp1,
                bilt.aNrmGrid
            )
        )
        return EndOfPrdvP

    def get_source_points_via_EGM(self, EndOfPrdvP, aNrm):
        """
        Finds interpolation points (c,m) for the consumption function.

        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal values.
        aNrm : np.array
            Array of end-of-period asset values that yield the marginal values
            in EndOfPrdvP.

        Returns
        -------
        c_for_interpolation : np.array
            Consumption points for interpolation.
        m_for_interpolation : np.array
            Corresponding market resource points for interpolation.
        """
#        breakpoint()
#        CRRA = self.soln_crnt.bilt.CRRA
#        self.soln_crnt.bilt = def_utility(self.soln_crnt.bilt, CRRA)
        cNrm = self.soln_crnt.bilt.uPinv(EndOfPrdvP)
        mNrm = cNrm + aNrm

        # Limiting consumption is zero as m approaches mNrmMin
        c_for_interpolation = np.insert(cNrm, 0, 0.0, axis=-1)
        m_for_interpolation = np.insert(mNrm, 0, self.soln_crnt.bilt.BoroCnstNat, axis=-1)

        # Store these for calcvFunc
        self.soln_crnt.bilt.cNrm = cNrm
        self.soln_crnt.bilt.mNrm = mNrm

        return c_for_interpolation, m_for_interpolation

    def use_points_for_interpolation(self, cNrm, mNrm, interpolator):
        """
        Constructs a basic solution for this period, including the consumption
        function and marginal value function.

        Parameters
        ----------
        cNrm : np.array
            (Normalized) consumption points for interpolation.
        mNrm : np.array
            (Normalized) corresponding market resource points for interpolation.
        interpolator : function
            A function that constructs and returns a consumption function.

        Returns
        -------
        solution_interpolating : ConsumerSolution
            The solution to this period's consumption-saving problem, with a
            minimum m, a consumption function, and marginal value function.
        """
        bilt = self.soln_crnt.bilt
        # Use the given interpolator to construct the consumption function
        cFuncUnc = interpolator(mNrm, cNrm)  # Unc=Unconstrained

        # Combine the constrained and unconstrained functions into the true consumption function
        # by choosing the lower of the constrained and unconstrained functions
        # LowerEnvelope should only be used when BoroCnstArt is true
        if bilt.BoroCnstArt is None:
            cFunc = cFuncUnc
        else:
            bilt.cFuncCnst = LinearInterp(
                np.array([bilt.mNrmMin, bilt.mNrmMin + 1]
                         ), np.array([0.0, 1.0]))
            cFunc = LowerEnvelope(cFuncUnc, bilt.cFuncCnst, nan_bool=False)

        # Make the marginal value function and the marginal marginal value function
        vPfunc = MargValueFuncCRRA(cFunc, bilt.CRRA)

        # Pack up the solution and return it
        solution_interpolating = ConsumerSolutionOneStateCRRA(
            cFunc=cFunc,
            vPfunc=vPfunc,
            mNrmMin=bilt.mNrmMin
        )

        return solution_interpolating

    def interpolating_EGM_solution(self, EndOfPrdvP, aNrmGrid, interpolator):
        """
        Given end of period assets and end of period marginal value, construct
        the basic solution for this period.

        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal values.
        aNrmGrid : np.array
            Array of end-of-period asset values that yield the marginal values
            in EndOfPrdvP.

        interpolator : function
            A function that constructs and returns a consumption function.

        Returns
        -------
        sol_EGM : ConsumerSolution
            The EGM solution to this period's consumption-saving problem, with a
            consumption function, marginal value function, and minimum m.
        """
        cNrm, mNrm = self.get_source_points_via_EGM(EndOfPrdvP, aNrmGrid)
        sol_EGM = self.use_points_for_interpolation(cNrm, mNrm, interpolator)

        return sol_EGM

    def make_sol_using_EGM(self):  # Endogenous Gridpts Method
        """
        Given a grid of end-of-period values of assets a, use the endogenous
        gridpoints method to obtain the corresponding values of consumption,
        and use the dynamic budget constraint to obtain the corresponding value
        of market resources m.

        Parameters
        ----------
        none (relies upon self.soln_crnt.aNrm existing before invocation)

        Returns
        -------
        solution : ConsumerSolution
            The solution to the single period consumption-saving problem.
        """
        bilt = self.soln_crnt.bilt
        bilt.aNrmGrid = self.prepare_to_calc_EndOfPrdvP()
        bilt.EndOfPrdvP = self.calc_EndOfPrdvP()

        # Construct a solution for this period
        if bilt.CubicBool:
            soln_crnt = self.interpolating_EGM_solution(
                bilt.EndOfPrdvP, bilt.aNrmGrid,
                interpolator=self.make_cubic_cFunc
            )
        else:
            soln_crnt = self.interpolating_EGM_solution(
                bilt.EndOfPrdvP, bilt.aNrmGrid,
                interpolator=self.make_linear_cFunc
            )
        return soln_crnt

    def make_linear_cFunc(self, mNrm, cNrm):
        """
        Makes a linear interpolation to represent the (unconstrained) consumption function.

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
            mNrm, cNrm, self.soln_crnt.bilt.cFuncLimitIntercept, self.soln_crnt.bilt.cFuncLimitSlope
        )
        return cFunc_unconstrained

#    def solve_prepared_stage(self):  # solves ONE stage of ConsIndShockSolverBasic
    def solve_prepared_stage(self):  # solves ONE stage of ConsIndShockSolverBasic
        """
        Solves one stage (period, in this model) of the consumption-saving problem.  

        Solution includes a decision rule (consumption function), cFunc,
        value and marginal value functions vFunc and vPfunc, 
        a minimum possible level of normalized market resources mNrmMin, 
        normalized human wealth hNrm, and bounding MPCs MPCmin and MPCmax.  

        If the user chooses sets `CubicBool` to True, cFunc
        have a value function vFunc and marginal marginal value function vPPfunc.

        Parameters
        ----------
        none (all should be on self)

        Returns
        -------
        solution : ConsumerSolution
            The solution to this period/stage's problem.
        """
        soln_futr_bilt = self.soln_futr.bilt
        soln_crnt_bilt = self.soln_crnt.bilt
#        soln_futr = self.soln_futr
        soln_crnt = self.soln_crnt
        CRRA = soln_crnt.bilt.CRRA

        if soln_futr_bilt.stge_kind['iter_status'] == 'finished':
            breakpoint()
            # Should not have gotten here
            # core.py tests whehter solution_last is 'finished'

        # If this is the first invocation of solve, just flesh out the
        # terminal_pseudo solution so it is a proper starting point for iteration
        # given the further info that has been added since generic
        # solution_terminal was constructed.  This involves copying its
        # contents into the bilt attribute, then invoking the
        # build_infhor_facts_from_params() method
#        breakpoint()
        if soln_futr_bilt.stge_kind['iter_status'] == 'terminal_pseudo':
            # generic AgentType solution_terminal does not have utility or value
            #            breakpoint()

            soln_futr = soln_crnt = def_utility(soln_crnt, CRRA)
#            print('Test whether value funcs are already defined; they are in PF case ...')
#            breakpoint()
#            soln_futr_bilt = soln_crnt_bilt = def_value_funcs(soln_crnt_bilt, CRRA)
            self.build_infhor_facts_from_params()
            # Now it is good to go as a starting point for backward induction:
            soln_crnt_bilt.stge_kind['iter_status'] = 'iterator'
#            breakpoint()
            self.soln_crnt.vFunc = self.soln_crnt.bilt.vFunc  # Need for distance
            self.soln_crnt.vPfunc = self.soln_crnt.bilt.vPfunc  # Need for distance
            self.soln_crnt.cFunc = self.soln_crnt.bilt.cFunc  # Need for distance
            self.soln_crnt.IncShkDstn = self.soln_crnt.bilt.IncShkDstn
            return self.soln_crnt  # Replaces original "terminal" solution; next soln_futr

        # Add a bunch of useful stuff
        # CDC 20200428: This stuff is "useful" only for a candidate converged solution
        # in an infinite horizon model.  It's not costly to compute but there's not
        # much point in computing most of it for a non-final infhor stage or a finhor model
        # TODO: Distinguish between those things that need to be computed for the
        # "useful" computations in the final stage, and those that are merely info,
        # and make mandatory only the computations of the former category
        self.build_infhor_facts_from_params()
#        if self.soln_futr.bilt.completed_cycles == 1:
#            print('about to call recursive on soln_futr.completed_cycles==1')
#            breakpoint()
        self.build_recursive_facts()

        # Current utility functions colud be different from future
        soln_crnt_bilt = def_utility(soln_crnt, CRRA)
#        breakpoint()
        sol_EGM = self.make_sol_using_EGM()  # Need to add test for finished, change stge_kind if so

#        breakpoint()
        soln_crnt.bilt.cFunc = soln_crnt.cFunc = sol_EGM.bilt.cFunc
        # soln_crnt.bilt.vPfunc = soln_crnt.vPfunc = sol_EGM.vPfunc
        # # Adding vPPfunc does no harm if non-cubic solution is being used
        # soln_crnt.bilt.vPPfunc = MargMargValueFuncCRRA(soln_crnt.bilt.cFunc, soln_crnt.bilt.CRRA)
        # Can't build current value function until current consumption function exists
#        CRRA = soln_crnt.bilt.CRRA
#        soln_crnt.bilt = def_value_funcs(soln_crnt.bilt, CRRA)

        soln_crnt = def_value_funcs(soln_crnt, CRRA)
        soln_crnt.vPfunc = soln_crnt.bilt.vPfunc
        soln_crnt.cFunc = soln_crnt.bilt.cFunc
        if not hasattr(soln_crnt.bilt, 'IncShkDstn'):
            print('not hasattr(soln_crnt.bilt, "IncShkDstn")')
            breakpoint()

        soln_crnt.IncShkDstn = soln_crnt.bilt.IncShkDstn
        # Add the value function if requested, as well as the marginal marginal
        # value function if cubic splines were used for interpolation
        # CDC 20210428: We should just always make the value function.  The cost
        # is trivial and making it optional is not worth the maintainence and
        # mindspace time the option takes in the codebase
        # if soln_crnt.bilt.vFuncBool:
        #     soln_crnt.bilt.vFunc = self.vFunc = self.add_vFunc(soln_crnt, self.EndOfPrdvP)
        # if soln_crnt.bilt.CubicBool:
        #     soln_crnt.bilt.vPPfunc = self.add_vPPfunc(soln_crnt)

        # EndOfPrdvP=self.soln_crnt.bilt.EndOfPrdvP
        # aNrmGrid=self.soln_crnt.bilt.aNrmGrid

        # solnow=self.mbs(EndOfPrdvP, aNrmGrid, self.make_cubic_cFunc)

        return soln_crnt

    solve = solve_prepared_stage

    def m_Nrm_tp1(self, shk_vector, a_number):
        """
        Computes normalized market resources of the next period
        from income shocks and current normalized market resources.

        Parameters
        ----------
        shk_vector: [float]
            Permanent and transitory income shock levels.

        a_number: float
            Normalized market assets this period

        Returns
        -------
        float
           normalized market resources in the next period
        """
        return self.soln_crnt.bilt.Rfree / (self.soln_crnt.bilt.PermGroFac * shk_vector[0]) \
            * a_number + shk_vector[1]


class ConsIndShockSolverBasic(ConsIndShockSolverBasicEOP):
    pass


###############################################################################


class ConsIndShockSolverEOP(ConsIndShockSolverBasicEOP):
    """
    This class solves a single period of a standard consumption-saving problem.
    It inherits from ConsIndShockSolverBasic, and adds the ability to perform cubic
    interpolation and to calculate the value function.
    """

    def make_cubic_cFunc(self, mNrm_Vec, cNrm_Vec):
        """
        Makes a cubic spline interpolation of the unconstrained consumption
        function for this period.

        Requires self.soln_crnt.bilt.aNrm to have been computed already.

        Parameters
        ----------
        mNrm_Vec : np.array
            Corresponding market resource points for interpolation.
        cNrm_Vec : np.array
            Consumption points for interpolation.

        Returns
        -------
        cFunc_unconstrained : CubicInterp
            The unconstrained consumption function for this period.
        """

#        scsr = self.soln_crnt.scsr
        bilt = self.soln_crnt.bilt
        folw = self.soln_crnt.folw

        def vPP_tp1(shk_vector, a_number):
            return shk_vector[0] ** (- bilt.CRRA - 1.0) \
                * folw.vPPfunc_tp1(self.m_Nrm_tp1(shk_vector, a_number))

        EndOfPrdvPP = (
            bilt.DiscFac * bilt.LivPrb
            * bilt.Rfree
            * bilt.Rfree
            * bilt.PermGroFac ** (-bilt.CRRA - 1.0)
            * calc_expectation(
                bilt.IncShkDstn,
                vPP_tp1,
                bilt.aNrmGrid
            )
        )
        dcda = EndOfPrdvPP / bilt.uPP(np.array(cNrm_Vec[1:]))
        MPC = dcda / (dcda + 1.0)
        MPC = np.insert(MPC, 0, bilt.MPCmax)

        cFuncUnc = CubicInterp(
            mNrm_Vec, cNrm_Vec, MPC, bilt.MPCmin *
            bilt.hNrm, bilt.MPCmin
        )
        return cFuncUnc

    def make_EndOfPrdvFunc(self, EndOfPrdvP):
        """
        Construct the end-of-period value function for this period, storing it
        as an attribute of self for use by other methods.

        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal value of assets corresponding to the
            asset values in self.soln_crnt.aNrm.

        Returns
        -------
        none
        """

        breakpoint()
        bilt = self.soln_crnt.bilt

        def v_Lvl_tp1(shk_vector, a_number):
            return (
                shk_vector[0] ** (1.0 - bilt.CRRA)
                * bilt.PermGroFac ** (1.0 - bilt.CRRA)
            ) * bilt.vFuncNxt(self.soln_crnt.m_Nrm_tp1(shk_vector, a_number))
        EndOfPrdv = bilt.DiscLiv * calc_expectation(
            bilt.IncShkDstn, v_Lvl_tp1, self.soln_crnt.aNrm
        )
        EndOfPrdvNvrs = self.soln_crnt.uinv(
            EndOfPrdv
        )  # value transformed through inverse utility
        EndOfPrdvNvrsP = EndOfPrdvP * self.soln_crnt.uinvP(EndOfPrdv)
        EndOfPrdvNvrs = np.insert(EndOfPrdvNvrs, 0, 0.0)
        EndOfPrdvNvrsP = np.insert(
            EndOfPrdvNvrsP, 0, EndOfPrdvNvrsP[0]
        )  # This is a very good approximation, vNvrsPP = 0 at the asset minimum
        aNrm_temp = np.insert(self.soln_crnt.aNrm, 0, self.soln_crnt.BoroCnstNat)
        EndOfPrdvNvrsFunc = CubicInterp(aNrm_temp, EndOfPrdvNvrs, EndOfPrdvNvrsP)
        self.soln_crnt.EndOfPrdvFunc = ValueFuncCRRA(
            EndOfPrdvNvrsFunc, bilt.CRRA)

    def add_vFunc(self, soln_crnt, EndOfPrdvP):
        """
        Creates the value function for this period and adds it to the soln_crnt.

        Parameters
        ----------
        solution : ConsumerSolution
            The solution to this single period problem, likely including the
            consumption function, marginal value function, etc.
        EndOfPrdvP : np.array
            Array of end-of-period marginal value of assets corresponding to the
            asset values in self.soln_crnt.aNrm.

        Returns
        -------
        solution : ConsumerSolution
            The single period solution passed as an input, but now with the
            value function (defined over market resources m) as an attribute.
        """
        self.make_EndOfPrdvFunc(EndOfPrdvP)
        self.vFunc = soln_crnt.vFunc = self.make_vFunc(soln_crnt)
        return soln_crnt.vFunc

    def make_vFunc(self, soln_crnt):
        """
        Creates the value function for this period, defined over market resources m.
        self must have the attribute EndOfPrdvFunc in order to execute.

        Parameters
        ----------
        solution : ConsumerSolution
            The solution to this single period problem, which must include the
            consumption function.

        Returns
        -------
        vFunc : ValueFuncCRRA
            A representation of the value function for this period, defined over
            normalized market resources m: v = vFunc(m).
        """
        # Compute expected value and marginal value on a grid of market resources
        bilt = self.soln_crnt.bilt

        mNrm_temp = bilt.mNrmMin + bilt.aXtraGrid
        cNrm = soln_crnt.cFunc(mNrm_temp)
        aNrm = mNrm_temp - cNrm
        vNrm = bilt.u(cNrm) + self.EndOfPrdvFunc(aNrm)
        vPnow = self.uP(cNrm)

        # Construct the beginning value function
        vNvrs = bilt.uinv(vNrm)  # value transformed through inverse utility
        vNvrsP = vPnow * bilt.uinvP(vNrm)
        mNrm_temp = np.insert(mNrm_temp, 0, bilt.mNrmMin)
        vNvrs = np.insert(vNvrs, 0, 0.0)
        vNvrsP = np.insert(
            vNvrsP, 0, bilt.MPCmaxEff ** (-bilt.CRRA /
                                          (1.0 - bilt.CRRA))
        )
        MPCminNvrs = bilt.MPCmin ** (-bilt.CRRA /
                                     (1.0 - bilt.CRRA))
        vNvrsFunc = CubicInterp(
            mNrm_temp, vNvrs, vNvrsP, MPCminNvrs * bilt.hNrm, MPCminNvrs
        )
        vFunc = ValueFuncCRRA(vNvrsFunc, bilt.CRRA)
        return vFunc

    def add_vPPfunc(self, soln_crnt):
        """
        Adds the marginal marginal value function to an existing solution, so
        that the next solver can evaluate vPP and thus use cubic interpolation.

        Parameters
        ----------
        solution : ConsumerSolution
            The solution to this single period problem, which must include the
            consumption function.

        Returns
        -------
        solution : ConsumerSolution
            The same solution passed as input, but with the marginal marginal
            value function for this period added as the attribute vPPfunc.
        """
        self.vPPfunc = MargMargValueFuncCRRA(soln_crnt.bilt.cFunc, soln_crnt.bilt.CRRA)
        soln_crnt.bilt.vPPfunc = self.vPPfunc
        return soln_crnt.bilt.vPPfunc


class ConsIndShockSolver(ConsIndShockSolverEOP):
    pass

####################################################################################################
####################################################################################################


class ConsKinkedRsolver(ConsIndShockSolver):
    """
    A class to solve a single period consumption-saving problem where the interest
    rate on debt differs from the interest rate on savings.  Inherits from
    ConsIndShockSolver, with nearly identical inputs and outputs.  The key diff-
    erence is that Rfree is replaced by Rsave (a>0) and Rboro (a<0).  The solver
    can handle Rboro == Rsave, which makes it identical to ConsIndShocksolver, but
    it terminates immediately if Rboro < Rsave, as this has a different soln_crnt.

    Parameters
    ----------
    soln_futr : ConsumerSolution
        The solution to next period's one period problem.
    IncShkDstn : distribution.Distribution
        A discrete
        approximation to the income process between the period being solved
        and the one immediately following (in soln_futr).
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
    Rboro: float
        Interest factor on assets between this period and the succeeding
        period when assets are negative.
    Rsave: float
        Interest factor on assets between this period and the succeeding
        period when assets are positive.
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
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported soln_crnt.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear inter-
        polation.
    """

    def __init__(
            self,
            soln_futr,
            IncShkDstn,
            LivPrb,
            DiscFac,
            CRRA,
            Rboro,
            Rsave,
            PermGroFac,
            BoroCnstArt,
            aXtraGrid,
            vFuncBool,
            CubicBool,
    ):
        assert (
            Rboro >= Rsave
        ), "Interest factor on debt less than interest factor on savings!"

        # Initialize the solver.  Most of the steps are exactly the same as in
        # the non-kinked-R basic case, so start with that.
        ConsIndShockSolver.__init__(
            self,
            soln_futr,
            IncShkDstn,
            LivPrb,
            DiscFac,
            CRRA,
            Rboro,
            PermGroFac,
            BoroCnstArt,
            aXtraGrid,
            vFuncBool,
            CubicBool,
        )

        # Assign the interest rates as class attributes, to use them later.
        self.bilt.Rboro = self.Rboro = Rboro
        self.bilt.Rsave = self.Rsave = Rsave
        self.bilt.cnstrct = {'vFuncBool', 'IncShkDstn'}

        self.Rboro = Rboro
        self.Rsave = Rsave
        self.cnstrct = {'vFuncBool', 'IncShkDstn'}

    def make_cubic_cFunc(self, mNrm, cNrm):
        """
        Makes a cubic spline interpolation that contains the kink of the unconstrained
        consumption function for this period.


        ----------
        mNrm : np.array
            Corresponding market resource points for interpolation.
        cNrm : np.array
            Consumption points for interpolation.

        Returns
        -------
        cFunc_unconstrained : CubicInterp
            The unconstrained consumption function for this period.
        """
        # Call the make_cubic_cFunc from ConsIndShockSolver.
        cFuncUncKink = super().make_cubic_cFunc(mNrm, cNrm)

        # Change the coeffients at the kinked points.
        cFuncUncKink.coeffs[self.i_kink + 1] = [
            cNrm[self.i_kink],
            mNrm[self.i_kink + 1] - mNrm[self.i_kink],
            0,
            0,
        ]

        return cFuncUncKink

    def prepare_to_calc_EndOfPrdvP(self):
        """
        Prepare to calculate end-of-period marginal value by creating an array
        of market resources that the agent could have next period, considering
        the grid of end-of-period assets and the distribution of shocks he might
        experience next period.  This differs from the baseline case because
        different savings choices yield different interest rates.

        Parameters
        ----------
        none

        Returns
        -------
        aNrm : np.array
            A 1D array of end-of-period assets; stored as attribute of self.
        """
        KinkBool = (
            self.bilt.Rboro > self.bilt.Rsave
        )  # Boolean indicating that there is actually a kink.
        # When Rboro == Rsave, this method acts just like it did in IndShock.
        # When Rboro < Rsave, the solver would have terminated when it was called.

        # Make a grid of end-of-period assets, including *two* copies of a=0
        if KinkBool:
            aNrm = np.sort(
                np.hstack(
                    (np.asarray(self.aXtraGrid) + self.mNrmMin, np.array([0.0, 0.0]))
                )
            )
        else:
            aNrm = np.asarray(self.aXtraGrid) + self.mNrmMin
            aXtraCount = aNrm.size

        # Make tiled versions of the assets grid and income shocks
        ShkCount = self.bilt.tranShkVals.size
        aNrm_temp = np.tile(aNrm, (ShkCount, 1))
        permShkVals_temp = (np.tile(self.bilt.permShkVals, (aXtraCount, 1))).transpose()
        tranShkVals_temp = (np.tile(self.bilt.tranShkVals, (aXtraCount, 1))).transpose()
        ShkPrbs_temp = (np.tile(self.ShkPrbs, (aXtraCount, 1))).transpose()

        # Make a 1D array of the interest factor at each asset gridpoint
        Rfree_vec = self.bilt.Rsave * np.ones(aXtraCount)
        if KinkBool:
            self.i_kink = (
                np.sum(aNrm <= 0) - 1
            )  # Save the index of the kink point as an attribute
            Rfree_vec[0: self.i_kink] = self.bilt.Rboro
#            Rfree = Rfree_vec
            Rfree_temp = np.tile(Rfree_vec, (ShkCount, 1))

        # Make an array of market resources that we could have next period,
        # considering the grid of assets and the income shocks that could occur
        mNrmNext = (
            Rfree_temp / (self.PermGroFac * permShkVals_temp) * aNrm_temp
            + tranShkVals_temp
        )

        # Recalculate the minimum MPC and human wealth using the interest factor on saving.
        # This overwrites values from set_and_update_values, which were based on Rboro instead.
        if KinkBool:
            RPFTop = (
                (self.bilt.Rsave * self.DiscLiv) ** (1.0 / self.CRRA)
            ) / self.bilt.Rsave
            self.MPCmin = 1.0 / (1.0 + RPFTop / self.soln_crnt.bilt.MPCmin)
            self.hNrm = (
                self.PermGroFac
                / self.bilt.Rsave
                * (
                    𝔼_dot(
                        self.ShkPrbs, self.bilt.tranShkVals * self.bilt.permShkVals
                    )
                    + self.soln_crnt.bilt.hNrm
                )
            )

        # Store some of the constructed arrays for later use and return the assets grid
        self.permShkVals_temp = permShkVals_temp
        self.ShkPrbs_temp = ShkPrbs_temp
        self.mNrmNext = mNrmNext
        self.aNrm = aNrm
        return aNrm
