# -*- coding: utf-8 -*-
import numpy as np
from copy import copy, deepcopy
from builtins import (range, str, breakpoint)
from types import SimpleNamespace
from dolo import yaml_import
import tempfile  # create temp file for dolo yaml import

from HARK.core import (_log, set_verbosity_level)
from HARK.distribution \
    import (add_discrete_outcome_constant_mean, calc_expectation,
            combine_indep_dstns, Lognormal, MeanOneLogNormal, Uniform)
from HARK.interpolation import (LinearInterp)
from HARK import AgentType, make_one_period_oo_solver
from HARK.ConsumptionSaving.ConsIndShockModel_CommonDefs \
    import (def_utility, def_value_funcs,
            construct_assets_grid)
from HARK.ConsumptionSaving.ConsIndShockModel_AgentSolve \
    import (ConsumerSolutionOneStateCRRA,
            ConsPerfForesightSolver,
            ConsIndShockSolverBasic, ConsIndShockSolver,
            ConsKinkedRsolver
            )
from HARK.ConsumptionSaving.ConsIndShockModel_AgentDicts \
    import (init_perfect_foresight, init_idiosyncratic_shocks)

"""
Defines increasingly specialized agent types for one-state-variable
consumption problem.

    * consumer_terminal_nobequest_onestate: The single state variable defined here
    is market resources `m,` the sum of assets from prior choices
    and income earned immediately before consumption decision.
    Incorporates a `nobequest` terminal consumption function
    in which consumption `c = m`

    * PerfForesightConsumerType: Subclass of consumer_terminal_nobequest_onestate
    in which income and asset returns are perfectly predictable
    and utility is CRRA

    * IndShockConsumerType: Subclass of PerfForesightConsumerType
    in which noncapital income has transitory and permanent shocks,
    and the lowest realization of the transitory shock corresponds
    to a one-period spell of `unemployment.'

    * KinkedRconsumerType: Subclass of IndShockConsumerType
    in which the interest factor depends on whether the consumer
    ends the period with positive market assets (earning `Rsave`)
    or negative market assets (paying interest according to
    `Rboro > Rsave`).

"""

__all__ = [
    "AgentTypePlus",
    "consumer_terminal_nobequest_onestate",
    "PerfForesightConsumerType",
    "IndShockConsumerType",
    "KinkedRconsumerType"
]
# TODO: CDC 20210129: After begin vetted, the elements of "Plus" that add to
# the base type from core.py should be merged into that base type. We can leave the "Plus"
# type empty in order to preserve an easy workflow that permits discrete
# proposals for improvements to the core AgentType.


class AgentTypePlus(AgentType):
    """
    AgentType augmented with a few features that should be incorporated into
    the base AgentType
    """
    __doc__ += AgentType.__doc__
    __doc__ += """
    Notes
    -----
    The code defines a number of optional elements that are used to
    to enhance clarity or to allow future functionality.  These include:

    prmtv_par : dictionary
        List of 'prmtv' parameters that are necessary and sufficient to
        define a unique solution with infinite computational power

    aprox_lim : dictionary
        Approximation parameters, including a limiting value.
        As all aprox parameters approach their limits simultaneously,
        the numerical solution should converge to the 'true' solution
        that would be obtained with infinite computational power

    """

    # These three variables are mandatory; they must be overwritten
    # as appropriate
    time_vary = []
    time_inv = []
    state_vars = []

    # https://elfi-y.medium.com/super-inherit-your-python-class-196369e3377a
    def __init__(self, *args, **kwargs):  # Inherit from basic AgentType
        AgentType.__init__(self, *args, **kwargs)

        # The base MetricObject class automatically constructs a list
        # of parameters but for some reason it does not get some
        # of the parameters {'cycles','seed','tolerance'} needed
        # TODO: CDC 20210525: Fix this in MetricObject to reduce clutter here
        self.add_to_given_params = {'time_vary', 'time_inv', 'state_vars',
                                    'cycles', 'seed', 'tolerance'}
        self.update_parameters_for_this_agent_subclass()

    def agent_store_model_params(self, prmtv_par, aprox_lim):
        # When anything cached here changes, solution must be recomputed
        self.prmtv_par_vals = {}
        for par in prmtv_par:
            if hasattr(self, par):
                self.prmtv_par_vals[par] = getattr(self, par)

        self.aprox_par_vals = {}
        for key in aprox_lim:
            if hasattr(self, key):
                self.aprox_par_vals[key] = getattr(self, key)

        # Merge to get all aprox and prmtv params and make a copy
        self.solve_par_vals = \
            deepcopy({**self.prmtv_par_vals, **self.aprox_par_vals})

        # Needs to go on solution_terminal so it can get on the solutions
        self.solution_terminal.bilt.solve_par_vals = self.solve_par_vals

    def update_parameters_for_this_agent_subclass(self):
        # Model class adds parameters explicitly passed; but parameters should also
        # include anything else (even default values not explicitly passed) required
        # to reproduce results of the model

        for add_it in self.add_to_given_params:
            self.parameters.update({add_it: getattr(self, add_it)})

    def agent_update_if_params_have_changed_since_last_solve(self):
        """
        Update any characteristics of the agent that need to be recomputed
        as a result of changes in parameters since the last time the solver was invoked.

        Parameters
        ----------
        None

        Returns
        -------
        None (adds `solve_par_vals_now` dict to self)

        """

        solve_par_vals_now = {}
        if hasattr(self, 'solve_par_vals'):
            for par in self.solve_par_vals:
                solve_par_vals_now[par] = getattr(self, par)

            if not solve_par_vals_now == self.solve_par_vals:
                _log.info('Some model parameter has changed since last update.')
                _log.info('Storing new parameters and updating shocks and grid.')
                self.agent_force_prepare_info_needed_to_begin_solving()  # The AgentType must define its own

    def agent_force_prepare_info_needed_to_begin_solving(self):
        # There are no universally required pre_solve objects
        #
        pass

    # pre_solve is the old name, preserved as an alias because
    # core.py uses it.  New name is clearer
    pre_solve = agent_force_prepare_info_needed_to_begin_solving

    # Universal method to either warn that something went wrong
    # or to mark the solution as having completed.  Should not
    # be overwritten by subclasses; instead, agent-specific
    # post-solve actions are accomplished by agent_post_post_solve
    def post_solve(self):
        if not hasattr(self, 'solution'):
            _log.critical('No solution was returned.')
            return
        else:
            if not type(self.solution) == list:
                _log.critical('Solution is not a list.')
                return
        soln = self.solution[0]
        if not hasattr(soln.bilt, 'stge_kind'):
            #        if not hasattr(soln.bilt, 'stge_kind'):
            _log.warning('Solution does not have attribute stge_kind')
            return
        else:
#            breakpoint()
            soln.bilt.stge_kind['iter_status'] = 'finished'
#            soln.bilt.stge_kind['iter_status'] = 'finished'
        self.agent_post_post_solve()

    # Disambiguation: former "[solver].post_solve"; post_solve is now alias to this
    # it's hard to remember whether "post_solve" is a method of
    # the solver or of the agent.  The answer is the agent; hence the rename
    # The alias below prevents breakage
    agent_post_solve = post_solve_agent = post_solve

    def agent_post_post_solve(self):
        # overwrite this with anything required to be customized for post_solve
        # of a particular agent type.
        # For example, computing stable points for inf hor buffer stock
        # Overwritten in PerfForesightConsumerSolution, carrying over to IndShockConsumerType
        pass


# TODO: CDC: 20210529 consumer_terminal_nobequest_onestate should be changed to
# consumer_onestate and we should define a set of allowed bequest
# choices including at least:
# - nobequest
# - warm_glow
# - capitalist_spirit
#   - warm_glow with bequests as a luxury in Stone-Geary form
#   - implies that bequests are left only if lifetime income high enough
# - dynasty (Barrovian)

class consumer_terminal_nobequest_onestate(AgentTypePlus):
    """
    Minimal requirements for a consumer with one state variable, m:
        * m combines assets from prior history with current income
        * it is referred to as `market resources` throughout the docs

    consumer_terminal_nobequest_onestate class must be inherited by some subclass that
    fleshes out the rest of the characteristics of the agent, e.g. the
    PerfForesightConsumerType or MertonSamuelsonConsumerType or something.

    Parameters
    ----------
    cycles : int
        Number of times the sequence of periods/stages should be solved.

    solution_startfrom : ConsumerSolution, optional

        A prespecified solution for the endpoint of the consumer
    problem. If no value is supplied, the terminal solution defaults
    to the case in which the consumer spends all available resources,
    obtaining no residual utility from any unspent m.

    """

    def __init__(
        self,
        solution_startfrom=None,
        cycles=1,
        pseudo_terminal=False,
        **kwds
    ):

        AgentTypePlus.__init__(
            self,
            solution_terminal=solution_startfrom,  # whether handmade or default
            cycles=cycles,
            pseudo_terminal=False,
            **kwds
        )

        cFunc_terminal_nobequest_ = LinearInterp([0.0, 1.0], [0.0, 1.0])

        # The below config of the 'afterlife' is constructed so that when
        # the standard lifetime transition rules are applied, the nobequest
        # terminal solution defined below is generated.
        # This should work if stge_kind['iter_status']="iterator"
        # This is inoperative if the terminal period is labeled with
        # stge_kind['iter_status']="terminal_pseudo" (because in that case
        # the "terminal_pseudo" final solution is used to construct the
        # augmented "terminal" solution

        solution_afterlife_nobequest_ = ConsumerSolutionOneStateCRRA(
            cFunc=lambda m: float('inf'),
            vFunc=lambda m: 0.0,  # nobequest vFunc same for all utility funcs
            vPfunc=lambda m: 0.0,
            vPPfunc=lambda m: 0.0,
            mNrmMin=0.0,
            hNrm=-1.0,
            MPCmin=float('inf'),
            MPCmax=float('inf'),
            stge_kind={
                'iter_status': 'afterlife',
                'term_type': 'nobequest'},
            completed_cycles=-1
        )

        solution_nobequest_ = ConsumerSolutionOneStateCRRA(  # Omits vFunc b/c u not yet def
            cFunc=cFunc_terminal_nobequest_,
            mNrmMin=0.0,  # Assumes PF model in which minimum mNrmMin is 1.0
            hNrm=0.0,
            MPCmin=1.0,
            MPCmax=1.0,
            stge_kind={
                'iter_status': 'terminal_pseudo',  # will be replaced with iterator
                'term_type': 'nobequest'
            })

        solution_nobequest_.solution_next = solution_afterlife_nobequest_

        # Define solution_terminal_ for legacy/compatability reasons
        # Otherwise would be better to just explicitly use solution_nobequest_
        solution_terminal_ = solution_nobequest_

        # Deepcopy: We will be modifying features of solution_terminal,
        # so make a deepcopy so that if multiple agents get created, we
        # always use the unaltered "master" solution_terminal_
        self.solution_terminal = deepcopy(solution_terminal_)
        self.update_parameters_for_this_agent_subclass()


class PerfForesightConsumerType(consumer_terminal_nobequest_onestate):

    """
    A perfect foresight consumer who has no uncertainty other than
    mortality risk.  Time-separable utility maximization problem is
    defined by a coefficient of relative risk aversion, geometric
    discount factor, interest factor, an artificial borrowing constraint
    (maybe) and time sequences of the permanent income growth rate and survival.

    Parameters
    ----------
    cycles : int
        Number of times the sequence of periods/stages should be solved.
    """
    # fcts : dictionary
    #     For storing meta information about an object in the model,
    #     for example a mathematical derivation or an explanation of
    #     its role in an economic model.

    #     fcts[objectName]['latexexpr'] - Name of variable in LaTeX docs
    #     fcts[objectName]['urlhandle'] - url to further info on it
    #     fcts[objectName]['python_ex'] - python expr creating its value
    #     fcts[objectName]['value_now'] - latest value calculated for it

    time_vary_ = ["LivPrb",  # Age-varying death rates can match mortality data
                  "PermGroFac"]  # Age-varying income growth can match data
    time_inv_ = ["CRRA", "Rfree", "DiscFac", "MaxKinks", "BoroCnstArt"]
    state_vars = ['pLvl',  # Idiosyncratic permanent income
                  'PlvlAgg',  # Aggregate permanent income
                  'bNrm',  # Bank balances beginning of period (pLvl normed)
                  'mNrm',  # Market resources (b + income) (pLvl normed)
                  "aNrm"]  # Assets after all actions (pLvl normed)
    shock_vars_ = []

    # Get default values from the Single Source of Truth
    from HARK.ConsumptionSaving.ConsIndShockModel_AgentDicts \
        import init_perfect_foresight \
            as default
#        import init_perfect_foresight_plus_make

#    default = init_perfect_foresight_plus_make(
#        init_perfect_foresight).parameters

    def __init__(self,
                 cycles=1,  # Default to finite horiz
                 verbose=1,  # little feedback
                 quiet=False,  # do not check conditions
                 solution_startfrom=None,  # Default is no interim solution
                 # TODO: 20210529: CDC: Probably we should use python 3.7+
                 # dataclasses for representing parameters.  Cleaner.
                 BoroCnstArt=default['BoroCnstArt'],
                 MaxKinks=default['MaxKinks'],
                 Rfree=default['Rfree'],
                 CRRA=default['CRRA'],
                 DiscFac=default['DiscFac'],
                 PermGroFac=default['PermGroFac'],
                 LivPrb=default['LivPrb'],
                 T_cycle=default['T_cycle'],
                 PermGroFacAgg=default['PermGroFacAgg'],
                 aNrmInitMean=default['aNrmInitMean'],
                 aNrmInitStd=default['aNrmInitStd'],
                 pLvlInitMean=default['pLvlInitMean'],
                 pLvlInitStd=default['pLvlInitStd'],
                 T_age=default['T_age'],
                 solver=ConsPerfForesightSolver,
                 **kwds
                 ):

        params = init_perfect_foresight.copy()  # Get defaults
        params.update(kwds)  # Replace defaults with passed vals if diff

        consumer_terminal_nobequest_onestate.__init__(
            self,
            solution_startfrom=None,
            cycles=cycles,
            pseudo_terminal=False,
            ** params)

        self.CRRA = CRRA
        self.Rfree = Rfree
        self.DiscFac = DiscFac
        self.LivPrb = LivPrb
        self.PermGroFac = PermGroFac
        self.BoroCnstArt = BoroCnstArt
        self.T_cycle = T_cycle
        self.PermGroFacAgg = PermGroFacAgg
        self.MaxKinks = MaxKinks
        self.aNrmInitMean = aNrmInitMean
        self.aNrmInitStd = aNrmInitStd
        self.pLvlInitMean = pLvlInitMean
        self.pLvlInitStd = pLvlInitStd
        self.T_age = T_age
        self.solver = solver
        self.verbose = verbose
#        self.quiet = quiet
        set_verbosity_level((4 - verbose) * 10)

        self.check_restrictions()  # Make sure it's a minimally valid model
        self.time_vary = deepcopy(self.time_vary_)
        self.time_inv = deepcopy(self.time_inv_)
        self.cycles = deepcopy(self.cycles)

        self.update_parameters_for_this_agent_subclass()  # self.parameters gets new info

        # consumer_terminal_nobequest_onestate creates self.soln_crnt and self.soln_crnt.scsr
        # If they did not provide their own solution_startfrom, use default

        if not hasattr(self, 'solution_startfrom'):
            # enrich generic consumer_terminal_nobequest_onestate terminal function
            # with info specifically needed to solve this particular model
            self.solution_terminal.bilt = \
                self.finish_setup_of_default_solution_terminal()
            # make url that will locate the documentation
            self.url_doc_for_this_agent_type_get()
#        else:
#            # any user-provided solution should already be enriched
#            solution_terminal = solution_startfrom

        # The foregoing is executed by all classes that inherit from the PF model
        # The code below the following "if" is excuted only in the PF case

        self.income_risks_exist = \
            ('permShkStd' in params) or \
            ('tranShkStd' in params) or \
            (('UnempPrb' in params) and (params['UnempPrb'] != 0)) or \
            (('UnempPrbRet' in params) and (params['UnempPrbRet'] != 0))

        if self.income_risks_exist:  # We got here from a model with risks
            return  # Models with risks have different prep

        #
        self.agent_force_prepare_info_needed_to_begin_solving()

        # Store initial params; later used to test if anything changed
        self.agent_store_model_params(params['prmtv_par'],
                                      params['aprox_lim'])

        # Attach one-period(/stage) solver to AgentType
        self.solve_one_period = make_one_period_oo_solver(solver)  # allows user-specified alt

        self.make_solution_for_final_period()  # Populate [instance].solution[0]

#        self.solution[0].bilt.solve_par_vals = self.solve_par_vals
#        self.dolo_defs()

    def add_stable_points_to_solution(self, soln):
        """
        If the model is one characterized by stable points, calculate those and
        attach them to the solution.

        Parameters
        ----------
        soln : ConsumerSolution
            The solution whose stable points are to be calculated
        """
#        breakpoint()
        soln.check_conditions(soln, verbose=0)

        if not soln.bilt.GICRaw:  # no mNrmStE
            wrn = "Because the model's parameters do not satisfy the GIC, it " +\
                "has neither an individual steady state nor a target."
            _log.warning(wrn)
            soln.bilt.mNrmStE = \
                soln.mNrmStE = float('nan')
        else:  # mNrmStE exists; compute it and check mNrmTrg
            # soln.mNrmStE = \
            soln.bilt.mNrmStE = soln.mNrmStE_find()
#            soln.bilt.mNrmStE = soln.mNrmStE_find()
        if not self.income_risks_exist:  # If a PF model, nothing more to do
            return
        else:
            if not hasattr(soln.bilt, 'GICNrm'):  # Should not occur; debug if get here
                _log.critical('soln.bilt has no GICNrm attribute')
                breakpoint()
                return

            if not soln.bilt.GICNrm:
                wrn = "Because the model's parameters do not satisfy the " +\
                    "stochastic-growth-normalized GIC, it does not exhibit " +\
                    "a target level of wealth."
                _log.warning(wrn)
#                soln.mNrmTrg = \
                soln.bilt.mNrmTrg = float('nan')
            else:  # GICNrm exists
                #                breakpoint()
                #                soln.mNrmTrg = \
                soln.bilt.mNrmTrg = soln.mNrmTrg_find()

        return

    # CDC 20210511: The old version of ConsIndShockModel mixed calibration and results
    # between the agent, the solver, and the solution.  The new version puts all results
    # on the solution. This requires a final stage solution to exist from the get-go.
    # The method below tricks the solver into putting a properly enhanced version of
    # solution_terminal into the solution[0] position where it needs to be, leaving
    # the agent in a state where invoking the ".solve()" method as before will
    # accomplish the same things it did before, but from the new starting setup

    def make_solution_for_final_period(self):  # solution[0]=terminal_solution
        # but want to add extra info required for backward induction
        cycles_orig = deepcopy(self.cycles)
        tolerance_orig = deepcopy(self.tolerance)
        self.tolerance = float('inf')  # Any distance satisfies this tolerance!
        if self.cycles > 0:  # Then it's a finite horizon problem
            self.cycles = 0  # Tell it to solve only one period
        self.solve()  # ... means that "solve" will stop after setup ...
        self.tolerance = tolerance_orig  # which leaves us ready to solve
        self.cycles = cycles_orig  # with the original convergence criteria
        self.solution[0].bilt.stge_kind['iter_status'] = 'iterator'
        self.soln_crnt = self.solution[0]  # current soln is now the newly made one

    def agent_post_post_solve(self):  # Overwrites version from AgentTypePlus
        if self.cycles == 0:  # if it's an infinite horizon model
            # Test for the stable points, and if they exist, add them
            self.add_stable_points_to_solution(self.solution[0])

    def check_restrictions(self):
        """
        Check that various restrictions are met for the model class.
        """
        min0Bounded = {  # Things that must be >= 0
            'tranShkStd', 'permShkStd', 'UnempPrb', 'IncUnemp', 'UnempPrbRet', 'IncUnempRet'}

        gt0Bounded = {  # Things that must be >0
            'DiscFac', 'Rfree', 'PermGroFac', 'LivPrb'}

        max1Bounded = {  # Things that must be <= 1
            'LivPrb'}

        gt1Bounded = {  # Things that must be > 1
            'CRRA'}

        for var in min0Bounded:
            if var in self.__dict__['parameters']:
                if self.__dict__['parameters'][var] is not None:
                    # If a list (because time_var), use extremum of list
                    if type(self.__dict__['parameters'][var]) == list:
                        varMin = np.min(self.__dict__['parameters'][var])
                    else:
                        varMin = self.__dict__['parameters'][var]
                    if varMin < 0:
                        raise Exception(var+" is negative with value: " + str(varMin))
        for var in gt0Bounded:
            if self.__dict__['parameters'][var] is not None:
                if var in self.__dict__['parameters']:
                    if type(self.__dict__['parameters'][var]) == list:
                        varMin = np.min(self.__dict__['parameters'][var])
                    else:
                        varMin = self.__dict__['parameters'][var]
                    if varMin <= 0.0:
                        raise Exception(var+" is nonpositive with value: " + str(varMin))

        for var in max1Bounded:
            if self.__dict__['parameters'][var] is not None:
                if var in self.__dict__['parameters']:
                    if type(self.__dict__['parameters'][var]) == list:
                        varMax = np.max(self.__dict__['parameters'][var])
                    else:
                        varMax = self.__dict__['parameters'][var]
                    if varMax > 1.0:
                        raise Exception(var+" is greater than 1 with value: " + str(varMax))

        for var in gt1Bounded:
            if self.__dict__['parameters'][var] is not None:
                if var in self.__dict__['parameters']:
                    if type(self.__dict__['parameters'][var]) == list:
                        varMin = np.min(self.__dict__['parameters'][var])
                    else:
                        varMin = self.__dict__['parameters'][var]
                    if varMin <= 1.0:
                        if var == 'CRRA' and self.__dict__['parameters'][var] == 1.0:
                            _log.info('For log utility, use CRRA very close to 1, like 1.00001')
                        raise Exception(
                            var+" is less than or equal to 1.0 with value: " + str(varMax))
        return

    def check_conditions(self, verbose):

        if not hasattr(self, 'solution'):  # Need a solution to have been computed
            _log.info('Solving final period because conditions are computed on solver')
            self.make_solution_for_final_period()

        soln_crnt = self.solution[0]
        soln_crnt.check_conditions(soln_crnt, verbose)

    # def dolo_defs(self):  # CDC 20210415: Beginnings of Dolo integration
    #     self.symbol_calibration = dict(  # not used yet, just created
    #         states={"mNrm": 2.0,
    #                 "aNrm": 1.0,
    #                 "bNrm": 1.0,
    #                 "pLvl": 1.0,
    #                 "pLvlAgg": 1.0
    #                 },
    #         controls=["cNrm"],
    #         exogenous=[],
    #         parameters={"DiscFac": 0.96, "LivPrb": 1.0, "CRRA": 2.0,
    #                     "Rfree": 1.03, "PermGroFac": 1.0,
    #                     "BoroCnstArt": None,
    #                     }
    #         # Not clear how to specify characteristics of sim starting point
    #     )  # Things all ConsumerSolutions have in common

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

    check_conditions_solver = solver_check_conditions = check_conditions

    def url_doc_for_this_agent_type_get(self):
        # Generate a url that will locate the documentation
        self.class_name = self.__class__.__name__
        self.url_ref = "https://econ-ark.github.io/BufferStockTheory"
        self.urlroot = self.url_ref+'/#'
        self.url_doc = "https://hark.readthedocs.io/en/latest/search.html?q=" +\
            self.class_name+"&check_keywords=yes&area=default#"

    # Prepare PF agent for solution of entire problem
    # Overwrites default version from AgentTypePlus
    # Overwritten by version in ConsIndShockSolver
    def agent_force_prepare_info_needed_to_begin_solving(self):
        # This will be reached by IndShockConsumerTypes when they execute
        # PerfForesightConsumerType.__init__ but will subsequently be
        # overridden by the agent_force_prepare_info_needed_to_begin_solving
        # method attached to the IndShockConsumerType class
        if (type(self) == PerfForesightConsumerType):
            if not hasattr(self, 'BoroCnstArt'):
                if hasattr(self, "MaxKinks"):
                    if self.MaxKinks:
                        # What does it mean to have specified MaxKinks but no BoroCnst?
                        raise(
                            AttributeError(
                                "Kinks are caused by constraints.  \nCannot specify MaxKinks without constraints!  Aborting."
                            ))
                        return
                else:  # If MaxKinks not specified, set to number of cycles
                    self.MaxKinks = self.cycles

    pre_solve = agent_force_prepare_info_needed_to_begin_solving

    def unpack_cFunc(self):
        """ DEPRECATED: Use solution.unpack('cFunc') instead.
        "Unpacks" the consumption functions into their own field for easier access.
        After the model has been solved, the consumption functions reside in the
        attribute cFunc of each element of ConsumerType.solution.  This method
        creates a (time varying) attribute cFunc that contains a list of consumption
        functions.
        Parameters
        ----------
        none
        Returns
        -------
        none
        """
        _log.critical(
            "unpack_cFunc is deprecated and it will soon be removed, "
            "please use unpack('cFunc') instead."
        )
        self.unpack("cFunc")

    unpack_cFunc_from_solution_to_agent = unpack_cFunc

    def initialize_sim(self):
        self.mcrlovars = SimpleNamespace()
        self.mcrlovars.permShkAgg = self.permShkAgg = self.PermGroFacAgg  # Never changes during sim
        # CDC 20210428 it would be good if we could separate the sim from the sol variables like this
        self.mcrlovars.state_now['PlvlAgg'] = self.state_now['PlvlAgg'] = 1.0
        AgentType.initialize_sim(self)

    def birth(self, which_agents):
        """
        Makes new consumers for the given indices.  Initialized variables include aNrm and pLvl, as
        well as time variables t_age and t_cycle.  Normalized assets and permanent income levels
        are drawn from lognormal distributions given by aNrmInitMean and aNrmInitStd (etc).

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        """
        # Get and store states for newly born agents
        N = np.sum(which_agents)  # Number of new consumers to make
        self.mcrlovars.state_now['aNrm'][which_agents] = self.state_now['aNrm'][which_agents] = Lognormal(
            mu=self.aNrmInitMean,
            sigma=self.aNrmInitStd,
            seed=self.RNG.randint(0, 2 ** 31 - 1),
        ).draw(N)
        # why is a now variable set here? Because it's an aggregate.
        pLvlInitMean = self.pLvlInitMean + np.log(
            self.state_now['PlvlAgg']
        )  # Account for newer cohorts having higher permanent income
        self.mcrlovars.state_now['pLvl'][which_agents] = \
            self.state_now['pLvl'][which_agents] = Lognormal(
            pLvlInitMean,
            self.pLvlInitStd,
            seed=self.RNG.randint(0, 2 ** 31 - 1)
        ).draw(N)
        # How many periods since each agent was born
        self.mcrlovars.t_age[which_agents] = self.t_age[which_agents] = 0
        self.mcrlovars.t_cycle[which_agents] = \
            self.t_cycle[
            which_agents
        ] = 0  # Which period of the cycle each agent is currently in
        return None

    mcrlo_birth = birth_mcrlo = birth

    def death(self):
        """
        Determines which agents die this period and must be replaced.  Uses the sequence in LivPrb
        to determine survival probabilities for each agent.

        Parameters
        ----------
        None

        Returns
        -------
        which_agents : np.array(bool)
            Boolean array of size AgentCount indicating which agents die.
        """
        # Determine who dies
        DiePrb_by_t_cycle = 1.0 - np.asarray(self.LivPrb)
        DiePrb = DiePrb_by_t_cycle[
            self.t_cycle - 1
        ]  # Time has already advanced, so look back one
        DeathShks = Uniform(seed=self.RNG.randint(0, 2 ** 31 - 1)).draw(
            N=self.AgentCount
        )
        which_agents = DeathShks < DiePrb
        if self.T_age is not None:  # Kill agents that have lived for too many periods
            too_old = self.t_age >= self.T_age
            which_agents = np.logical_or(which_agents, too_old)
        return which_agents

    mcrlo_death = death_mcrlo = death

    def get_shocks(self):
        """
        Finds permanent and transitory income "shocks" for each agent this period.  When this is a
        perfect foresight model, there are no stochastic shocks: permShk = PermGroFac for each
        agent (according to their t_cycle) and tranShk = 1.0 for all agents.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        PermGroFac = np.array(self.PermGroFac)
        self.shocks['permShk'] = PermGroFac[
            self.t_cycle - 1
        ]  # cycle time has already been advanced
        self.shocks['tranShk'] = np.ones(self.AgentCount)

    get_shocks_mcrlo = mcrlo_get_shocks = get_shocks

    def get_Rfree(self):  # -> mcrlo_get_Rfree.
        # CDC: We should have a generic mcrlo_get_all_params
        """
        Returns an array of size self.AgentCount with self.Rfree in every entry.

        Parameters
        ----------
        None

        Returns
        -------
        Rfree : np.array
            Array of size self.AgentCount with risk free interest rate for each agent.
        """
        Rfree = self.Rfree * np.ones(self.AgentCount)
        return Rfree

    mcrlo_get_Rfree = get_Rfree_mcrlo = get_Rfree

    def transition(self):  # -> mcrlo_trnsitn
        pLvlPrev = self.state_prev['pLvl']
        aNrmPrev = self.state_prev['aNrm']
        Rfree = self.get_Rfree()

        # Calculate new states: normalized market resources and permanent income level
        pLvl = pLvlPrev*self.shocks['permShk']  # Updated permanent income level
        # Updated aggregate permanent productivity level
        PlvlAgg = self.state_prev['PlvlAgg']*self.permShkAgg
        # "Effective" interest factor on normalized assets
        RNrm = Rfree/self.shocks['permShk']
        bNrm = RNrm*aNrmPrev         # Bank balances before labor income
        mNrm = bNrm + self.shocks['tranShk']  # Market resources after income

        return pLvl, PlvlAgg, bNrm, mNrm, None

    transition_mcrlo = mcrlo_transition = transition

    def get_controls(self):  # -> mcrlo_get_ctrls
        """
        Calculates consumption for each consumer of this type using the consumption functions.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        cNrm = np.zeros(self.AgentCount) + np.nan
        MPCnow = np.zeros(self.AgentCount) + np.nan
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            cNrm[these], MPCnow[these] = self.solution[t].cFunc.eval_with_derivative(
                self.state_now['mNrm'][these]
            )
            self.controls['cNrm'] = cNrm

        # MPCnow is not really a control
        self.MPCnow = MPCnow
        return None

    get_controls_mcrlo = mcrlo_get_controls = get_controls

    def get_poststates(self):  # -> mcrlo_get_poststes
        """
        Calculates end-of-period assets for each consumer of this type.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # should this be "", or "Prev"?!?
        self.state_now['aNrm'] = self.state_now['mNrm'] - self.controls['cNrm']
        # Useful in some cases to precalculate asset level
        self.state_now['aLvl'] = self.state_now['aNrm'] * self.state_now['pLvl']

        # moves now to prev
        super().get_poststates()

        return None

    mcrlo_get_poststates = get_poststates_mcrlo = get_poststates


class IndShockConsumerType(PerfForesightConsumerType):

    """
    A consumer with idiosyncratic shocks to permanent and transitory income.
    The problem is defined by a sequence of income distributions, survival prob-
    abilities, and permanent income growth rates, as well as time invariant values
    for risk aversion, the discount factor, the interest rate, the grid of end-of-
    period assets, and (optionally) an artificial borrowing constraint.

    Parameters
    ----------
    cycles : int
        Number of times the sequence of periods should be solved.  If zero,
        the solver will continue until successive policy functions are closer
        than the tolerance specified as a default parameter.

    quiet : boolean, optional
        If True, creates the agent without setting up any solution apparatus
        If False, creates a solution object populated with a solution for
        the final stage.

    solution_startfrom : stge, optional
        A user-specified terminal period/stage solution for the iteration,
        to be used in place of the hardwired solution_terminal.  One use
        might be to set a loose tolerance to get a quick `solution_rough`
        using the default hardwired solution (nobequest), then
        set the tolerance tighter, or change some approximation parameter,
        and resume iteration using `solution_startfrom = solution_rough` until
        the new tolerance is met with the (presumably better but slower)
        approximation parameter.
    """

    # Time invariant parameters
    time_inv_ = PerfForesightConsumerType.time_inv_ + [
        "vFuncBool",
        "CubicBool",
    ]
    time_inv_.remove(  # Unwanted item(s) inherited from PerfForesight
        "MaxKinks"  # PF infhor with MaxKinks equiv to finhor with hor=MaxKinks
    )

    def __init__(self,
                 cycles=1, verbose=1,  quiet=True, solution_startfrom=None,
                 solverType='HARK',
                 solverName=ConsIndShockSolverBasic,
                 **kwds):
        params = init_idiosyncratic_shocks.copy()  # Get default params
        params.update(kwds)  # Update/overwrite defaults with user-specified

        # Inherit characteristics of a PF model with the same parameters
        PerfForesightConsumerType.__init__(self, cycles=cycles,
                                           verbose=verbose, quiet=quiet,
                                           _startfrom=solution_startfrom,
                                           **params)

        self.update_parameters_for_this_agent_subclass()  # Add new pars

        # If precooked terminal stage not provided by user ...
        if not hasattr(self, 'solution_startfrom'):  # .. then init the default
            self.agent_force_prepare_info_needed_to_begin_solving()

        # - Default interpolation method is piecewise linear
        # - Cubic is smoother, works best if problem has no constraints
        # - User may or may not want to create the value function
        # TODO: CDC 20210428: Basic solver is not worth preserving
        # - 1. We might as well always compute vFunc
        # - 2. Cubic vs linear interpolation is not worth two different solvers
        # -    * Cubic should be preserved as an option
        if self.CubicBool or self.vFuncBool:
            solverName = ConsIndShockSolver

        # Attach the corresponding one-stage solver to the agent
        # This is what gets called when the user invokes [instance].solve()
        if (solverType == 'HARK') or (solverType == 'DARKolo'):
            self.solve_one_period = make_one_period_oo_solver(solverName)

        if (solverType == 'dolo') or (solverType == 'DARKolo'):
            # If we want to solve with dolo, set up the model
            self.dolo_model()

        # Store setup parameters so later we can check for changes
        # that necessitate restarting solution process
        self.agent_store_model_params(params['prmtv_par'], params['aprox_lim'])

        # Put the (enhanced) solution_terminal in self.solution[0]
        self.make_solution_for_final_period()

    def dolo_model(self):
        # Create a dolo version of the model
#        breakpoint()
        self.dolo_modl = yaml_import(
            '/Volumes/Data/Code/ARK/DARKolo/chimeras/BufferStock/bufferstock.yaml'
            )
#        tmpyaml = tempfile.NamedTemporaryFile(mode='w+')
#        tmpyaml.write(self.dolo_yaml)
#        tmpyaml.seek(0)  # move to beginning
#        self.dolo_modl = yaml_import(tmpyaml.name)
#        breakpoint()


    # The former "[AgentType].update_pre_solve()" was poor nomenclature --
    #  easy to confuse with the also-existing "[AgentType].pre_solve()" and with
    # "[SolverType].prepare_to_solve()".  The new name,
    #
    # agent_force_prepare_info_needed_to_begin_solving()
    #
    # is better.  The old one
    # is preserved as an alias, below, to prevent breakage of existing code:

    def agent_force_prepare_info_needed_to_begin_solving(self):
        """
        Update any characteristics of the agent that need to be recomputed as a 
        result of changes in parameters since the last time the solver was invoked.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        self.solve_par_vals_now = {}
        if not hasattr(self, 'solve_par_vals'):  # We haven't set it up yet
            self.update_income_process()
            self.update_assets_grid()
        else:  # it has been set up, so see if anything changed
            for par in self.solve_par_vals:
                self.solve_par_vals_now[par] = getattr(self, par)
            if not self.solve_par_vals_now == self.solve_par_vals:
                _log.info('Some model parameter has changed since last update.')
                _log.info('Storing new parameters and updating shocks and grid.')
                self.update_income_process()
                self.update_assets_grid()
                self.solve_par_vals = self.solve_par_vals_now


    pre_solve = agent_force_prepare_info_needed_to_begin_solving

    def update_income_process(self):
        """
        Updates this agent's income process based on its current attributes.

        Parameters
        ----------
        none

        Returns:
        -----------
        none
        """

        (self.IncShkDstn,
            self.permShkDstn,
            self.tranShkDstn,
         ) = self.construct_lognormal_income_process_unemployment()
        self.add_to_time_vary("IncShkDstn", "permShkDstn", "tranShkDstn")
        self.parameters.update({'IncShkDstn': self.IncShkDstn,
                                'permShkDstn': self.permShkDstn,
                                'tranShkDstn': self.tranShkDstn})

    def update_assets_grid(self):
        """
        Updates this agent's end-of-period assets grid by constructing a multi-
        exponentially spaced grid of aXtra values.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        self.aXtraGrid = construct_assets_grid(self)
        self.add_to_time_inv("aXtraGrid")
        self.parameters.update({'aXtraGrid': self.aXtraGrid})

    def reset_rng(self):
        """
        Reset the RNG behavior of this type.  This method is called automatically
        by initialize_sim(), ensuring that each simulation run uses the same sequence
        of random shocks; this is necessary for structural estimation to work.
        This method extends AgentType.reset_rng() to also reset elements of IncShkDstn.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        PerfForesightConsumerType.reset_rng(self)

        # Reset IncShkDstn if it exists (it might not because reset_rng is called at init)
        if hasattr(self, "IncShkDstn"):
            for dstn in self.IncShkDstn:
                dstn.reset()

    mcrlo_reset_rng = reset_rng_mcrlo = reset_rng

    def construct_lognormal_income_process_unemployment(self):
        """
        Generates a sequence of discrete approximations to the income process for each
        life period, from end of life to beginning of life.  Permanent shocks are mean
        one lognormally distributed with standard deviation permShkStd[t] during the
        working life, and degenerate at 1 in the retirement period.  transitory shocks
        are mean one lognormally distributed with a point mass at IncUnemp with
        probability UnempPrb while working; they are mean one with a point mass at
        IncUnempRet with probability UnempPrbRet.  Retirement occurs
        after t=T_retire periods of working.

        Note 1: All time in this function runs forward, from t=0 to t=T

        Note 2: All parameters are passed as attributes of the input parameters.

        Parameters (passed as attributes of the input parameters)
        ----------
        permShkStd : [float]
            List of standard deviations in log permanent income uncertainty during
            the agent's life.
        permShkCount : int
            The number of approximation points to be used in the discrete approxima-
            tion to the permanent income shock distribution.
        tranShkStd : [float]
            List of standard deviations in log transitory income uncertainty during
            the agent's life.
        tranShkCount : int
            The number of approximation points to be used in the discrete approxima-
            tion to the permanent income shock distribution.
        UnempPrb : float
            The probability of becoming unemployed during the working period.
        UnempPrbRet : float
            The probability of not receiving typical retirement income when retired.
        T_retire : int
            The index value for the final working period in the agent's life.
            If T_retire <= 0 then there is no retirement.
        IncUnemp : float
            transitory income received when unemployed.
        IncUnempRet : float
            transitory income received while "unemployed" when retired.
        T_cycle :  int
            Total number of non-terminal periods in the consumer's sequence of periods.

        Returns
        -------
        IncShkDstn :  [distribution.Distribution]
            A list with elements from t = 0 to T_cycle, each of which is a
            discrete approximation to the joint income distribution at at [t]
        permShkDstn : [[distribution.Distribution]]
            A list with elements from t = 0 to T_cycle, each of which is a
            discrete approximation to the permanent shock distribution at [t]
        tranShkDstn : [[distribution.Distribution]]
            A list with elements from t = 0 to T_cycle, each of which is a
            discrete approximation to the transitory shock distribution at [t]
        """
        # Unpack the parameters from the input

        permShkStd = self.permShkStd
        permShkCount = self.permShkCount
        tranShkStd = self.tranShkStd
        tranShkCount = self.tranShkCount
        UnempPrb = self.UnempPrb
        UnempPrbRet = self.UnempPrbRet
        T_retire = self.T_retire
        IncUnemp = self.IncUnemp
        IncUnempRet = self.IncUnempRet
        T_cycle = self.T_cycle

        # make a dictionary of the parameters
        # This is so that, later, we can determine whether any parameters have changed
        parameters = {
            'permShkStd':  self.permShkStd,
            'permShkCount':  self.permShkCount,
            'tranShkStd':  self.tranShkStd,
            'tranShkCount':  self.tranShkCount,
            'UnempPrb':  self.UnempPrb,
            'UnempPrbRet':  self.UnempPrbRet,
            'T_retire':  self.T_retire,
            'IncUnemp':  self.IncUnemp,
            'IncUnempRet':  self.IncUnempRet,
            'T_cycle':  self.T_cycle,
            'ShkPosn': {'perm': 0, 'tran': 1}
        }

        # This is so that, later, we can determine whether another distribution object
        # was constructed using the same method or a different method
        constructed_by = {'method': 'construct_lognormal_income_process_unemployment'}

        IncShkDstn = []  # Discrete approximations to income process in each period
        permShkDstn = []  # Discrete approximations to permanent income shocks
        tranShkDstn = []  # Discrete approximations to transitory income shocks

        # Fill out a simple discrete RV for retirement, with value 1.0 (mean of shocks)
        # in normal times; value 0.0 in "unemployment" times with small prob.
        if T_retire > 0:
            if UnempPrbRet > 0:
                #                permShkValsNxtRet = np.array([1.0, 1.0])  # Permanent income is deterministic in retirement (2 states for temp income shocks)
                tranShkValsRet = np.array(
                    [
                        IncUnempRet,
                        (1.0 - UnempPrbRet * IncUnempRet) / (1.0 - UnempPrbRet),
                    ]
                )
                ShkPrbsRet = np.array([UnempPrbRet, 1.0 - UnempPrbRet])
            else:
                (IncShkDstnRet,
                 permShkDstnRet,
                 tranShkDstnRet,
                 ) = self.construct_lognormal_income_process_unemployment()
                ShkPrbsRet = IncShkDstnRet.pmf

        # Loop to fill in the list of IncShkDstn random variables.
        for t in range(T_cycle):  # Iterate over all periods, counting forward
            if T_retire > 0 and t >= T_retire:
                # Then we are in the "retirement period" and add a retirement income object.
                IncShkDstn.append(deepcopy(IncShkDstnRet))
                permShkDstn.append([np.array([1.0]), np.array([1.0])])
                tranShkDstn.append([ShkPrbsRet, tranShkValsRet])
            else:
                # We are in the "working life" periods.
                tranShkDstn_t = MeanOneLogNormal(sigma=tranShkStd[t]).approx(
                    tranShkCount, tail_N=0
                )
                if UnempPrb > 0:
                    tranShkDstn_t = add_discrete_outcome_constant_mean(
                        tranShkDstn_t, p=UnempPrb, x=IncUnemp
                    )
                permShkDstn_t = MeanOneLogNormal(sigma=permShkStd[t]).approx(
                    permShkCount, tail_N=0
                )
                IncShkDstn.append(
                    combine_indep_dstns(
                        permShkDstn_t,
                        tranShkDstn_t,
                        seed=self.RNG.randint(0, 2 ** 31 - 1),
                    )
                )  # mix the independent distributions
                permShkDstn.append(permShkDstn_t)
                tranShkDstn.append(tranShkDstn_t)

        IncShkDstn[-1].parameters = parameters
        IncShkDstn[-1].constructed_by = constructed_by

        return IncShkDstn, permShkDstn, tranShkDstn

    def get_shocks(self):  # mcrlo simulation tool
        """
        Gets permanent and transitory income shocks for this period.  Samples from IncShkDstn for
        each period in the cycle.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        permShk = np.zeros(self.AgentCount)  # Initialize shock arrays
        tranShk = np.zeros(self.AgentCount)
        newborn = self.t_age == 0
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            N = np.sum(these)
            if N > 0:
                IncShkDstn = self.IncShkDstn[
                    t - 1
                ]  # set current income distribution
                PermGroFac = self.PermGroFac[t - 1]  # and permanent growth factor
                # Get random draws of income shocks from the discrete distribution
                IncShks = IncShkDstn.draw(N)

                permShk[these] = (
                    IncShks[0, :] * PermGroFac
                )  # permanent "shock" includes expected growth
                tranShk[these] = IncShks[1, :]

        # That procedure used the *last* period in the sequence for newborns, but that's not right
        # Redraw shocks for newborns, using the *first* period in the sequence.  Approximation.
        N = np.sum(newborn)
        if N > 0:
            these = newborn
            IncShkDstn = self.IncShkDstn[0]  # set current income distribution
            PermGroFac = self.PermGroFac[0]  # and permanent growth factor

            # Get random draws of income shocks from the discrete distribution
            EventDraws = IncShkDstn.draw_events(N)
            permShk[these] = (
                IncShkDstn.X[0][EventDraws] * PermGroFac
            )  # permanent "shock" includes expected growth
            tranShk[these] = IncShkDstn.X[1][EventDraws]
            #        permShk[newborn] = 1.0
        tranShk[newborn] = 1.0

        # Store the shocks in self
        self.Emp = np.ones(self.AgentCount, dtype=bool)
        self.Emp[tranShk == self.IncUnemp] = False
        self.shocks['permShk'] = permShk
        self.shocks['tranShk'] = tranShk

    get_shocks_mcrlo = mcrlo_get_shocks = get_shocks


# Make a dictionary to specify a "kinked R" idiosyncratic shock consumer
init_kinked_R = dict(
    init_idiosyncratic_shocks,
    **{
        "Rboro": 1.20,  # Interest factor on assets when borrowing, a < 0
        "Rsave": 1.02,  # Interest factor on assets when saving, a > 0
        "BoroCnstArt": None,  # kinked R is a bit silly if borrowing not allowed
        "CubicBool": True,  # kinked R is now compatible with linear cFunc and cubic cFunc
        "aXtraCount": 48,  # ...so need lots of extra gridpoints to make up for it
    }
)
del init_kinked_R["Rfree"]  # get rid of constant interest factor


class KinkedRconsumerType(IndShockConsumerType):
    """
    A consumer type that faces idiosyncratic shocks to income and has a different
    interest factor on saving vs borrowing.  Extends IndShockConsumerType, with
    very small changes.  Solver for this class is currently only compatible with
    linear spline interpolation.

    Parameters
    ----------
    cycles : int
        Number of times the sequence of periods should be solved.
    """

    time_inv_ = copy(IndShockConsumerType.time_inv_)
    time_inv_.remove("Rfree")
    time_inv_ += ["Rboro", "Rsave"]

    def __init__(self, cycles=1, **kwds):
        params = init_kinked_R.copy()
        params.update(kwds)

        # Initialize a basic AgentType
        PerfForesightConsumerType.__init__(self, cycles=cycles, **params)

        # Add consumer-type specific objects, copying to create independent versions
        self.solve_one_period = make_one_period_oo_solver(ConsKinkedRsolver)
        # Make assets grid, income process, terminal solution

    def agent_force_prepare_info_needed_to_begin_solving(self):
        self.update_assets_grid()
        self.update_income_process()

#    pre_solve = agent_force_prepare_info_needed_to_begin_solving

    def calc_bounding_values(self):
        """
        Calculate human wealth plus minimum and maximum MPC in an infinite
        horizon model with only one period repeated indefinitely.  Store results
        as attributes of self.  Human wealth is the present discounted value of
        expected future income after receiving income this period, ignoring mort-
        ality.  The maximum MPC is the limit of the MPC as m --> mNrmMin.  The
        minimum MPC is the limit of the MPC as m --> infty.  This version deals
        with the different interest rates on borrowing vs saving.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Unpack the income distribution and get average and worst outcomes
        permShkVals = self.IncShkDstn[0][1]
        tranShkVals = self.IncShkDstn[0][2]
        ShkPrbs = self.IncShkDstn[0][0]
        Ex_IncNrmNxt = calc_expectation(
            self.IncShkDstn,
            lambda trans, perm: trans * perm
        )
        permShkMinNext = np.min(permShkVals)
        tranShkMinNext = np.min(tranShkVals)
        WorstIncNext = permShkMinNext * tranShkMinNext
        WorstIncPrb = np.sum(
            ShkPrbs[(permShkVals * tranShkVals) == WorstIncNext]
        )

        # Calculate human wealth and the infinite horizon natural borrowing constraint
        hNrm = (Ex_IncNrmNxt * self.PermGroFac[0] / self.Rsave) / (
            1.0 - self.PermGroFac[0] / self.Rsave
        )
        temp = self.PermGroFac[0] * permShkMinNext / self.Rboro
        BoroCnstNat = -tranShkMinNext * temp / (1.0 - temp)

        RPFTop = (self.DiscFac * self.LivPrb * self.Rsave) ** (
            1.0 / self.CRRA
        ) / self.Rsave
        RPFBot = (self.DiscFac * self.LivPrb * self.Rboro) ** (
            1.0 / self.CRRA
        ) / self.Rboro
        if BoroCnstNat < self.BoroCnstArt:
            MPCmax = 1.0  # if natural borrowing constraint is overridden by artificial one, MPCmax is 1
        else:
            MPCmax = 1.0 - WorstIncPrb ** (1.0 / self.CRRA) * RPFBot
            MPCmin = 1.0 - RPFTop

        # Store the results as attributes of self
        self.hNrm = hNrm
        self.MPCmin = MPCmin
        self.MPCmax = MPCmax
        self.IncNext_min = WorstIncNext

    def make_euler_error_func(self, mMax=100, approx_inc_dstn=True):
        """
        Creates a "normalized Euler error" function for this instance, mapping
        from market resources to "consumption error per dollar of consumption."
        Stores result in attribute eulerErrorFunc as an interpolated function.
        Has option to use approximate income distribution stored in self.IncShkDstn
        or to use a (temporary) very dense approximation.

        SHOULD BE INHERITED FROM ConsIndShockModel

        Parameters
        ----------
        mMax : float
            Maximum normalized market resources for the Euler error function.
        approx_inc_dstn : Boolean
            Indicator for whether to use the approximate discrete income distri-
            bution stored in self.IncShkDstn[0], or to use a very accurate
            discrete approximation instead.  When True, uses approximation in
            IncShkDstn; when False, makes and uses a very dense approximation.

        Returns
        -------
        None

        Notes
        -----
        This method is not used by any other code in the library. Rather, it is here
        for expository and benchmarking purposes.
        """
        raise NotImplementedError()

    def get_Rfree(self):
        """
        Returns an array of size self.AgentCount with self.Rboro or self.Rsave in each entry, based
        on whether self.aNrm >< 0.

        Parameters
        ----------
        None

        Returns
        -------
        Rfree : np.array
             Array of size self.AgentCount with risk free interest rate for each agent.
        """
        Rfree = self.Rboro * np.ones(self.AgentCount)
        Rfree[self.state_prev['aNrm'] > 0] = self.Rsave
        return Rfree

    mcrlo_get_Rfree = get_Rfree_mcrlo = get_Rfree
