# -*- coding: utf-8 -*-
import logging
import numpy as np
from copy import copy, deepcopy
from builtins import (range, str, breakpoint)
from types import SimpleNamespace

from HARK.core import (_log, set_verbosity_level)
from HARK.distribution \
    import (add_discrete_outcome_constant_mean,
            combine_indep_dstns, Lognormal, MeanOneLogNormal, Uniform)
from HARK.interpolation import (LinearInterp)
from HARK import AgentType, make_one_period_oo_solver
from HARK.ConsumptionSaving.ConsIndShockModel_CommonDefs \
    import (def_utility, def_value_funcs,
            construct_assets_grid)
from HARK.ConsumptionSaving.ConsIndShockModel_AgentSolve \
    import (ConsumerSolutionOneNrmStateCRRA, ConsPerfForesightSolver,
            ConsIndShockSolverBasic, ConsIndShockSolver,
            )
from HARK.ConsumptionSaving.ConsIndShockModel_AgentDicts \
    import (init_perfect_foresight, init_idiosyncratic_shocks)

from HARK.utilities import CRRAutility

from HARK.utilities import uFunc_CRRA_stone_geary as u_stone_geary
from HARK.utilities import uPFunc_CRRA_stone_geary as uP_stone_geary
from HARK.utilities import uPPFunc_CRRA_stone_geary as uPP_stone_geary


"""
Define increasingly specialized AgentTypes for one-state-variable
consumption problem.

    * consumer_terminal_nobequest_onestate:

        The single state variable defined here is market resources `m,` the sum
    of assets from prior choices and income earned immediately before
    consumption decision.  Incorporates a `nobequest` terminal consumption
    function in which consumption `c = m`

    * PerfForesightConsumerType:

        Subclass of consumer_terminal_nobequest_onestate in which income and
    asset returns are perfectly predictable and utility is CRRA

    * IndShockConsumerType:

        Subclass of PerfForesightConsumerType in which noncapital income has
    transitory and permanent shocks, and the lowest realization of the
    transitory shock corresponds to a one-period spell of `unemployment.'

    * KinkedRconsumerType:

        Subclass of IndShockConsumerType in which the interest factor depends
    on whether the consumer ends the period with positive market assets
    (earning `Rsave`) or negative market assets (paying interest according to
    `Rboro > Rsave`).

"""

__all__ = [
    "AgentTypePlus",
    "consumer_terminal_nobequest_onestate",
    "PerfForesightConsumerType",
    "IndShockConsumerType",
    "KinkedRconsumerType",
    "onestate_bequest_warmglow_homothetic"
]

# TODO: CDC 20210129: After being vetted, the elements of "Plus" that add to
# the base type from core.py should be merged into that base type. We can leave
# the "Plus" type empty in order to preserve an easy workflow that permits
# future proposals for improvements to the core AgentType.


class AgentTypePlus(AgentType):
    """
    Augment AgentType with features that should be incorporated into AgentType.
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

    # Mandatory lists; they must be overwritten as appropriate
    time_vary = []
    time_inv = []
    state_vars = []

    # https://elfi-y.medium.com/super-inherit-your-python-class-196369e3377a
    def __init__(self, *args,
                 verbose=True,  # Equivalent to "True"
                 quietly=False,  # Suppress all output
                 **kwds):  # Inherit from basic AgentType
        AgentType.__init__(self, *args, **kwds)

        # The base MetricObject class automatically constructs a list
        # of parameters but for some reason it does not get some
        # of the parameters {'cycles','seed','tolerance'} needed
        # to reproduce results exactly
        # TODO: CDC 20210525: Fix this in MetricObject to reduce clutter here
        self.add_to_given_params = {'time_vary', 'time_inv', 'state_vars',
                                    'cycles', 'seed', 'tolerance'}
        self.update_parameters_for_this_agent_subclass()

        # Goal: Push everything that will universally be needed down to the
        # AgentType level.  Verbosity is such a thing.
        self.verbose = verbose
        self.quietly = quietly  #
        set_verbosity_level((4 - verbose) * 10)

    def agent_store_model_params(self, prmtv_par, aprox_lim):
        # When anything cached here changes, solution must be recomputed
        prmtv_par_vals = {}
        for par in prmtv_par:
            if hasattr(self, par):
                prmtv_par_vals[par] = getattr(self, par)

        aprox_par_vals = {}
        for key in aprox_lim:
            if hasattr(self, key):
                aprox_par_vals[key] = getattr(self, key)

        # Merge to get all aprox and prmtv params and make a copy
        self.solve_par_vals = \
            deepcopy({**prmtv_par_vals, **aprox_par_vals})

        # Put on solution_terminal so it can get on non-term solution
        self.solution_terminal.Bilt.solve_par_vals = self.solve_par_vals

    def update_parameters_for_this_agent_subclass(self):
        # add_it: (below)
        # class(Model) adds parameters explicitly passed; but parameters should also
        # include anything else (even default values not explicitly passed) required
        # to reproduce exactly ALL results of the model

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

        # Get current parameter values
        solve_par_vals_now = {}
        if hasattr(self, 'solve_par_vals'):
            for par in self.solve_par_vals:
                solve_par_vals_now[par] = getattr(self, par)

            breakpoint()
            # Check whether any of them has changed
            if not (solve_par_vals_now == self.solve_par_vals):
                if not quietly:
                    _log.info('Some model parameter has changed since last update.')
                self._agent_force_prepare_info_needed_to_begin_solving()

    def _agent_force_prepare_info_needed_to_begin_solving(self):
        # There are no universally required pre_solve objects
        pass

    # pre_solve is the old name, preserved as an alias because
    # core.py uses it.  New name is MUCH clearer
    pre_solve = agent_update_if_params_have_changed_since_last_solve

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
        if not hasattr(soln.Bilt, 'stge_kind'):
            _log.warning('Solution does not have attribute stge_kind')
            return
        else:  # breakpoint()
            soln.Bilt.stge_kind['iter_status'] = 'finished'
        self.agent_post_post_solve()

    # Disambiguation: former "[solver].post_solve"; post_solve is now alias
    # it's too hard to remember whether "post_solve" is a method of
    # the solver or of the agent.  The answer is the agent; hence the rename
    agent_post_solve = post_solve_agent = post_solve

    # User-provided handmade solution_terminal is likely to be bare-bones
    # Below is a placeholder for anything that user might want to do
    # programmatially to enhance it
    def finish_setup_of_default_solution_terminal(self):
        """
        Add to `solution_terminal` characteristics of the agent required
        for solution of the particular type which are not automatically
        created as part of the definition of the generic `solution_terminal.`
        """
        pass

    def agent_post_post_solve(self):
        # overwrite this with anything required to be customized for post_solve
        # of a particular agent type.
        # For example, computing stable points for inf hor buffer stock
        # Overwritten in PerfForesightConsumerSolution, carrying over
        # to IndShockConsumerType
        pass

        # If they did not provide their own solution_startfrom, use default
        if not hasattr(self, 'solution_startfrom'):
            # enrich generic consumer_terminal_nobequest_onestate terminal func
            # with info specifically needed to solve this particular model
            self.solution_terminal.Bilt = \
                self.finish_setup_of_default_solution_terminal()
            # make url that will locate the documentation
            self._url_doc_for_this_agent_type_get()
        # any user-provided solution should already be enriched


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

    This class must be inherited by some subclass
    that fleshes out the rest of the characteristics of the agent, e.g. the
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
            self, solution_startfrom=None, cycles=1, pseudo_terminal=False,
            **kwds):

        AgentTypePlus.__init__(
            self, solution_terminal=solution_startfrom,  # handmade or default
            cycles=cycles, pseudo_terminal=False,
            **kwds)

        # The below config of the 'afterlife' is constructed so that when
        # the standard lifetime transition rules are applied, the nobequest
        # terminal solution defined below is generated.
        # This should work if stge_kind['iter_status']="iterator"
        # The afterlife is inoperative if the terminal period is labeled with
        # stge_kind['iter_status']="terminal_partial" (because in that case
        # the "terminal_partial" final solution is used to construct the
        # augmented "terminal" solution)

        # no value in afterlife:
        def vFunc(m): return 0.
        vFunc.dm = vPfunc = vFunc
        vFunc.dm.dm = vPPfunc = vFunc

        def cFunc(m): return float('inf')  # With CRRA utility, c=inf gives v=0
        cFunc.derivativeX = lambda m: float('inf')

        mNrmMin, hNrm, MPCmin, MPCmax = 0.0, -1.0, float('inf'), float('inf')

        solution_afterlife_nobequest_ = ConsumerSolutionOneNrmStateCRRA(
            vFunc=vFunc,
            vPfunc=vPfunc,  # TODO: vPfunc deprecated; remove
            vPPfunc=vPPfunc,  # TODO: vPPfunc deprecated: Remove
            cFunc=cFunc,
            mNrmMin=mNrmMin,  # TODO: mNrmMin should be on Bilt; remove
            hNrm=-hNrm,  # TODO: hNrm should be on Bilt; remove
            MPCmin=MPCmin,  # TODO: hNrm should be on Bilt; remove
            MPCmax=MPCmax,  # TODO: hNrm should be on Bilt; remove
            # TODO: stge_kind should be on Bilt; remove
            stge_kind={
                'iter_status': 'afterlife',
                'term_type': 'nobequest'},
            completed_cycles=-1
        )
        Bilt = solution_afterlife_nobequest_.Bilt
        Bilt.cFunc, Bilt.vFunc, Bilt.mNrmMin, Bilt.hNrm, Bilt.MPCmin, Bilt.MPCmax = \
            cFunc, vFunc, mNrmMin, hNrm, MPCmin, MPCmax

        mNrmMin, hNrm, MPCmin, MPCmax = 0.0, 0.0, 1.0, 1.0

        # This is the solution that would be constructed by applying
        # our normal iteration tools to solution_afterlife_nobequest_

        cFunc_terminal_nobequest_ = LinearInterp([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])

        cFunc = cFunc_terminal_nobequest_

        CRRA = 2.0
        def u(c): CRRAutility(c, CRRA)

        solution_nobequest_ = \
            ConsumerSolutionOneNrmStateCRRA(  # Omit vFunc b/c u not yet def
                cFunc=cFunc_terminal_nobequest_,
                #                vFunc=u,
                mNrmMin=mNrmMin,  # TODO: deprecated; remove
                hNrm=hNrm,  # TODO: should be on Bilt; remove
                MPCmin=MPCmin,  # TODO: should be on Bilt; remove
                MPCmax=MPCmin,  # TODO: should be on Bilt; remove
                stge_kind={
                    'iter_status': 'terminal_partial',  # must be replaced
                    'term_type': 'nobequest'
                })

        Bilt = solution_nobequest_.Bilt
        Bilt.cFunc, Bilt.vFunc, Bilt.mNrmMin, Bilt.hNrm, Bilt.MPCmin, Bilt.MPCmax = \
            cFunc, vFunc, mNrmMin, hNrm, MPCmin, MPCmax

        solution_nobequest_.solution_next = solution_afterlife_nobequest_
        # solution_terminal_ is defined for legacy/compatability reasons
        # Otherwise would be better to just explicitly use solution_nobequest_
        self.solution_terminal_ = solution_terminal_ = solution_nobequest_
        # Deepcopy: We will be modifying features of solution_terminal,
        # so make a deepcopy so that if multiple agents get created, we
        # always use the unaltered "master" solution_terminal_
        self.solution_terminal = deepcopy(solution_terminal_)


class onestate_bequest_warmglow_homothetic(ConsumerSolutionOneNrmStateCRRA):
    """
    Homothetic Stone-Geary bequest utility function with bequests as a luxury.

    Must be inherited by a subclass
    that fleshes out the rest of the characteristics of the agent, e.g. the
    PerfForesightConsumerType or MertonSamuelsonConsumerType or something.

    The bequest utility function is assumed to be of the Stone-Geary form
    and to have a scale reflecting the number of periods worth of consumption
    that it is equivalent to in the limit.  (In the limit as wealth approaches
    infinity, if this parameter were equal to the number of periods of life
    and the pure time preference factor were 1, the consumer would split their
    lifetime resources equally between the bequest and their lifetime
    consumption).

    Parameters
    ----------
    cycles : int
        Number of times the sequence of periods/stages should be solved.

    solution_startfrom : ConsumerSolution, optional
        A prespecified solution for the endpoint of the consumer
    problem. If no value is supplied, the terminal solution defaults
    to the case in which the consumer spends all available resources,
    obtaining no residual utility from any unspent m.

    stone_geary : float
        This parameter is added to the argument of the bequest utility function
    in order to make bequests a luxury good

    equiv_life_periods : float
        Limiting number of periods-worth of consumption that the bequest is
    equivalent to
    """

    def __init__(
            self, solution_startfrom=None, cycles=1, pseudo_terminal=False,
            stone_geary=1.0,
            equiv_life_periods=1.0,
            CRRA=2,
            **kwds):

        ConsumerSolutionOneNrmStateCRRA.__init__(self,
                                                 cycles=cycles,
                                                 pseudo_terminal=False, CRRA=CRRA,
                                                 **kwds)

        Bilt = self.Bilt  # alias

        if (equiv_life_periods == 0.0):
            msg = 'With bequest parameter equiv_life_periods = 0, ' +\
                'the model exhibits no bequest motive.'

            nobequest_agent = consumer_terminal_nobequest_onestate(
            )
            self.solution_terminal = nobequest_agent.solution_terminal

            # Only reason to use the bequest type here instead of nobequest
            # is to get the infrastructure for solving the PF liquidity
            # constrained problem.  That is below.

            # Add infrastructure for piecewise linear PF solution
            Bilt.mNrm_cusp = 0.0  # here 'cusp' => cannot die in debt
            Bilt.vNrm_cusp = -float('inf')  # yields neg inf value
            Bilt.vInv_cusp = 0.0
            Bilt.mNrm_kinks = [Bilt.mNrm_cusp]
            Bilt.vNrm_kinks = [Bilt.vNrm_cusp]
            Bilt.MPC_kinks = [1.0]
            Bilt.hNrm = 0.0
            _log.info(msg)
            return

        if (stone_geary == 0.0):
            msg = 'With stone_geary parameter of zero, the bequest motive ' +\
                'is equivlent to equiv_life_periods worth of consumption.'
            _log.info(msg)

        # The entire bequest enters the utility function
        bequest_entering_utility = LinearInterp(
            [0., 1.], [0., 1.]
        )

        sab = solution_afterlife_bequest_ = ConsumerSolutionOneNrmStateCRRA(
            cFunc=bequest_entering_utility,
            u=u_stone_geary, uP=uP_stone_geary, uPP=uPP_stone_geary,
            vFunc=u_stone_geary, vPfunc=uP_stone_geary,
            vPPfunc=uPP_stone_geary,
            mNrmMin=0.0,
            MPCmin=1.0,
            MPCmax=1.0,
            stge_kind={
                'iter_status': 'afterlife',
                'term_type': 'bequest_warmglow_homothetic'},
            completed_cycles=-1
        )
        ρ = sab.Bilt.CRRA = CRRA
        η = sab.Bilt.stone_geary = stone_geary
        ℶ = sab.Bilt.equiv_life_periods = equiv_life_periods  # Hebrew bet

        if (equiv_life_periods == 0.0):
            Bilt.vNrm_cusp = -float('inf')  # then 'cusp' => cannot die in debt
        else:
            bequest_size = 0.0
            Bilt.vNrm_cusp = CRRAutility(Bilt.mNrm_cusp, CRRA) +\
                ℶ * u_stone_geary(bequest_size, CRRA, stone_geary)

        Bilt.mNrm_kinks = [Bilt.mNrm_cusp]  # zero if no bequest motive
        Bilt.vInv_uncons = [self.Bilt.uinv(Bilt.vNrm_cusp)]
        Bilt.vInv_constr = [self.Bilt.uinv(Bilt.u(0.))]
        # See PerfForesightConsumerType for MPC derivation
        if ℶ == 0.0:
            Bilt.MPC_constr = [1/(1+0.0)]
        else:
            Bilt.MPC_constr = [1/(1+(ℶ**(-1/ρ)))]

    def cFunc(self, m):
        MPC_constr = self.Bilt.MPC_constr
        mNrm_kinks = self.Bilt.mNrm_kinks
        constr_0 = np.heaviside(m-mNrm_kinks[0], 0.)  # 0 if constrnd, else 1
        c_constr = (1-constr_0)*m  # m if m < kink
        c_uncons = constr_0*(c_constr+MPC_constr[0]*(m-mNrm_kinks[0]))
        return c_constr+c_uncons

# It should be possible to swap other bequest motive choices, but this
# has not really been tested (20210718)


class PerfForesightConsumerType(consumer_terminal_nobequest_onestate):
    """
    Consumer with perfect foresight except for potential mortality risk.

    A perfect foresight consumer who has no uncertainty other than
    mortality risk.  Time-separable utility maximization problem is
    defined by a coefficient of relative risk aversion, geometric
    discount factor, interest factor, an artificial borrowing constraint
    (maybe) and time sequences of permanent income growth rate and survival.

    Parameters
    ----------
    cycles : int
        Number of times the sequence of periods/stages should be solved.
    """

    time_vary_ = ["LivPrb",  # Age-varying death rates can match mortality data
                  "PermGroFac"]  # Age-varying income growth to match lifecycle
    time_inv_ = ["CRRA", "Rfree", "DiscFac", "MaxKinks", "BoroCnstArt",
                 "solveMethod", "eventTiming", "solverType",
                 "horizon"
                 ]
    state_vars = ['pLvl',  # Initial idiosyncratic permanent income
                  'PlvlAgg',  # Aggregate permanent income
                  'bNrm',  # Bank balances beginning of period (pLvl normed)
                  'mNrm',  # Market resources (b + income) (pLvl normed)
                  "aNrm"]  # Assets after all actions (pLvl normed)
    shock_vars_ = []

    def __init__(self,
                 cycles=1,  # Default to finite horiz
                 quietly=False,  # if True, do print anything conditions
                 solution_startfrom=None,  # Default: no interim solution
                 solver=ConsPerfForesightSolver,
                 solveMethod='EGM',
                 eventTiming='EOP',
                 solverType='HARK',
                 horizon='infinite',
                 messaging_level=logging.INFO,
                 **kwds
                 ):

        params = init_perfect_foresight.copy()  # Get defaults
        params.update(kwds)  # Replace defaults with passed vals if diff

        consumer_terminal_nobequest_onestate.__init__(
            self, solution_startfrom=None, cycles=cycles,
            pseudo_terminal=False, ** params)

        # Solver and method are:
        # - required to solve
        # - not necessarily set in any ancestral class
        self.solver = solver
        self.solveMethod = solveMethod
        self.eventTiming = eventTiming
        self.solverType = solverType
        self.horizon = horizon
        self.messaging_level = messaging_level
        self.quietly = quietly

        # Things to keep track of for this and child models
        self.check_restrictions()  # Make sure it's a minimally valid model
        self.time_vary = deepcopy(self.time_vary_)
        self.time_inv = deepcopy(self.time_inv_)
        self.cycles = deepcopy(self.cycles)

        self.update_parameters_for_this_agent_subclass()  # store new info

        # If they did not provide their own solution_startfrom, use default
        if not hasattr(self, 'solution_startfrom'):
            # enrich generic consumer_terminal_nobequest_onestate terminal func
            # with info specifically needed to solve this particular model
            self.solution_terminal.Bilt = \
                self.finish_setup_of_default_solution_terminal()
            # make url that will locate the documentation
            self._url_doc_for_this_agent_type_get()
        # any user-provided solution should already be enriched

        # The foregoing is executed by all classes that inherit from PF model
        # The code below the following "if" is excuted only in the PF case

        self.income_risks_exist = \
            ('permShkStd' in params) or \
            ('tranShkStd' in params) or \
            (('UnempPrb' in params) and (params['UnempPrb'] != 0)) or \
            (('UnempPrbRet' in params) and (params['UnempPrbRet'] != 0))

        if self.income_risks_exist:  # We got here from a model with risks
            return  # Models with risks have different prep

        self._agent_force_prepare_info_needed_to_begin_solving()

        # Store initial params; later used to test if anything changed
        self.agent_store_model_params(params['prmtv_par'],
                                      params['aprox_lim'])

        # Attach one-period(/stage) solver to AgentType
        self.solve_one_period = \
            make_one_period_oo_solver(
                solver,
                solveMethod=solveMethod,
                eventTiming=eventTiming
            )  # allows user-specified alt

        self._make_solution_for_final_period()  # Populate instance.solution[0]

    def add_stable_points_to_solution(self, soln):
        """
        If they exist, add any stable points to the model solution object.

        Parameters
        ----------
        soln : agent_solution
            The solution whose stable points are to be calculated
        """

        #  Users can effectively solve approx to inf hor PF model by specifying
        #  the "horizon" (number of periods to be solved). In that special case
        #  the "conditions" are relevant because we are thinking of it as an
        #  inf hor model, so allow that case to slip through the cracks
        if soln.Pars.horizon != 'infinite':
            if soln.Pars.BoroCnstArt is None:
                return
            else:  # finite horizon borrowing constrained model
                if self.income_risks_exist:
                    return  # infinite horizon conditions unimportant

        soln.check_conditions(messaging_level=self.messaging_level, quietly=self.quietly)

        if not soln.Bilt.GICRaw:  # no mNrmStE
            # wrn = "Because the model's parameters do not satisfy the GIC," +\
            #     "it has neither an individual steady state nor a target."
            # _log.warning(wrn)
            soln.Bilt.mNrmStE = soln.mNrmStE = float('nan')
        else:  # mNrmStE exists; compute it and check mNrmTrg
            soln.Bilt.mNrmStE = soln.mNrmStE_find()
        if not self.income_risks_exist:  # If a PF model, nothing more to do
            return
        else:
            if not soln.Bilt.GICNrm:
                soln.Bilt.mNrmTrg = float('nan')
            else:  # GICNrm exists; calculate it
                soln.Bilt.mNrmTrg = soln.mNrmTrg_find()
        return

    # CDC 20210511: The old version of ConsIndShockModel mixed calibration and results
    # between the agent, the solver, and the solution.  The new version puts all results
    # on the solution. This requires a final stage solution to exist from the get-go.
    # The method below tricks the solver into putting a properly enhanced version of
    # solution_terminal into the solution[0] position where it needs to be, leaving
    # the agent in a state where invoking the ".solve()" method as before will
    # accomplish the same things it did before, but from the new starting setup

    def _make_solution_for_final_period(self, messaging_level=logging.INFO,
                                        quietly=True):
        # but want to add extra info required for backward induction
        cycles_orig = deepcopy(self.cycles)
        tolerance_orig = deepcopy(self.tolerance)
        self.tolerance = float('inf')  # Any distance satisfies this tolerance!
        if self.cycles > 0:  # Then it's a finite horizon problem
            self.cycles = 0  # solve only one period (leaving MaxKinks be)
            self.solve()  # do not spout nonsense
        else:  # tolerance of inf means that "solve" will stop after setup ...
            #            breakpoint()
            self.solve()
        self.tolerance = tolerance_orig  # which leaves us ready to solve
        self.cycles = cycles_orig  # with the original convergence criteria
        self.solution[0].Bilt.stge_kind['iter_status'] = 'iterator'
        self.solution[0].Bilt.vAdd = np.array([0.0])  # Amount to add to last v
        self.soln_crnt = self.solution[0]  # current soln is now newly made one

    def agent_post_post_solve(self):  # Overwrites version from AgentTypePlus
        """For infinite horizon models, add stable points (if they exist)."""
        if self.cycles == 0:  # if it's an infinite horizon model
            self.add_stable_points_to_solution(self.solution[0])

    def check_restrictions(self):
        """
        Check that various restrictions are met for the model class.
        """
        min0Bounded = {  # Things that must be >= 0
            'tranShkStd', 'permShkStd', 'UnempPrb', 'IncUnemp', 'UnempPrbRet',
            'IncUnempRet'}

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

    def check_conditions(self, messaging_level=logging.INFO, quietly=False):

        if not hasattr(self, 'solution'):  # A solution must have been computed
            _log.info('Make final soln because conditions are computed there')
            self._make_solution_for_final_period()

        soln_crnt = self.solution[0]
        soln_crnt.check_conditions(messaging_level=logging.INFO, quietly=False)
#       soln_crnt.check_conditions(soln_crnt, verbose)  # real version on soln

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
        """
        # If no solution exists for the agent,
        # core.py uses solution_terminal as solution_next

        solution_terminal = self.solution_terminal

        # Natural borrowing constraint: Cannot die in debt
        # Measured after income = tranShk*permShk/permShk received
        if not hasattr(solution_terminal.Bilt, 'hNrm'):
            _log.warning('warning: hNrm should be set in solution_terminal.')
            _log.warning('assuming solution_terminal.hNrm = 0.')
            solution_terminal.Bilt.hNrm = 0.
        solution_terminal.Bilt.BoroCnstNat = -solution_terminal.Bilt.hNrm

        # Define BoroCnstArt if not yet defined
        if not hasattr(self.parameters, 'BoroCnstArt'):
            solution_terminal.Bilt.BoroCnstArt = None
        else:
            solution_terminal.Bilt.BoroCnstArt = self.parameters.BoroCnstArt

        solution_terminal.Bilt.stge_kind = {'iter_status': 'terminal_partial'}

        # Solution options
        if hasattr(self, 'vFuncBool'):
            solution_terminal.Bilt.vFuncBool = self.parameters['vFuncBool']
        else:  # default to true
            solution_terminal.Bilt.vFuncBool = True

        if hasattr(self, 'CubicBool'):
            solution_terminal.Bilt.CubicBool = self.parameters['CubicBool']
        else:  # default to false (linear)
            solution_terminal.Bilt.CubicBool = False

        solution_terminal.Bilt.parameters = self.parameters
        CRRA = self.CRRA
        solution_terminal.Bilt = def_utility(solution_terminal.Bilt, CRRA)
        solution_terminal.Bilt = def_value_funcs(solution_terminal.Bilt, CRRA)

        return solution_terminal.Bilt

    check_conditions_solver = solver_check_conditions = check_conditions

    def _url_doc_for_this_agent_type_get(self):
        # Generate a url that will locate the documentation
        self.class_name = self.__class__.__name__
        self.url_ref = "https://econ-ark.github.io/BufferStockTheory"
        self.urlroot = self.url_ref+'/#'
        self.url_doc = "https://hark.readthedocs.io/en/latest/search.html?q=" +\
            self.class_name+"&check_keywords=yes&area=default#"

    # Prepare PF agent for solution of entire problem
    # Overwrites default version from AgentTypePlus
    # Overwritten by version in ConsIndShockSolver
    def _agent_force_prepare_info_needed_to_begin_solving(self):
        # This will be reached by IndShockConsumerTypes when they execute
        # PerfForesightConsumerType.__init__ but will subsequently be
        # overridden by the _agent_force_prepare_info_needed_to_begin_solving
        # method attached to the IndShockConsumerType class

        if (type(self) == PerfForesightConsumerType):
            if self.cycles > 0:  # finite horizon, not solving terminal period
                if hasattr(self, 'BoroCnstArt'):
                    if isinstance(self.BoroCnstArt, float):  # 0.0 means no borrowing
                        if self.MaxKinks:  # If they did specify MaxKinks
                            if self.MaxKinks > self.cycles:
                                msg = 'You have requested a number of constraints ' +\
                                    'greater than the number of cycles.  ' +\
                                    'Reducing to MaxKinks = cycles'
                                self.MaxKinks = self.cycles
                                _log.critical(msg)
                        else:  # They specified a BoroCnstArt but no MaxKinks
                            self.MaxKinks = self.cycles
                    else:  # BoroCnstArt is not defined
                        if not hasattr(self, "MaxKinks"):
                            self.MaxKinks = self.cycles
                        else:
                            if self.MaxKinks:
                                # What does it mean to have specified MaxKinks
                                # but no BoroCnstArt?
                                raise(
                                    AttributeError(
                                        "Kinks are caused by constraints.  \n" +
                                        "Cannot specify MaxKinks without constraints!\n" +
                                        "  Aborting."
                                    ))
                            else:
                                self.MaxKinks = self.cycles

    pre_solve = _agent_force_prepare_info_needed_to_begin_solving

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
    Problem is defined by a sequence of income distributions, survival prob-
    abilities, permanent income growth rates, and time invariant values for
    risk aversion, the discount factor, the interest rate, the grid of end-of-
    period assets, and (optionally) an artificial borrowing constraint.

    Parameters
    ----------
    cycles : int
        Number of times the sequence of periods should be solved.  If zero,
        the solver will continue until successive policy functions are closer
        than the tolerance specified as a default parameter.

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
                 cycles=0,  # cycles=0 signals infinite horizon
                 messaging_level=logging.INFO,  quietly=False,
                 solution_startfrom=None,
                 solverType='HARK',
                 solveMethod='EGM',
                 eventTiming='EOP',
                 solverName=ConsIndShockSolverBasic,
                 **kwds):
        params = init_idiosyncratic_shocks.copy()  # Get default params
        params.update(kwds)  # Update/overwrite defaults with user-specified
#        self.messaging_level = messaging_level
#        self.quietly = quietly

        # Inherit characteristics of a PF model with the same parameters
        PerfForesightConsumerType.__init__(self, cycles=cycles,
                                           messaging_level=messaging_level,
                                           quietly=quietly,
                                           **params)

        self.update_parameters_for_this_agent_subclass()  # Add new Pars

        # If precooked terminal stage not provided by user ...
        if not hasattr(self, 'solution_startfrom'):  # .. then init the default
            self._agent_force_prepare_info_needed_to_begin_solving()

        # - Default interpolation method is piecewise linear
        # - Cubic is smoother, works if problem has no constraints
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
            #            breakpoint()
            self.solve_one_period = \
                make_one_period_oo_solver(
                    solverName,
                    solveMethod=solveMethod,
                    eventTiming=eventTiming
                )

        if (solverType == 'dolo') or (solverType == 'DARKolo'):
            # If we want to solve with dolo, set up the model
            self.dolo_model()

        # Store setup parameters so later we can check for changes
        # that necessitate restarting solution process

        self.agent_store_model_params(params['prmtv_par'], params['aprox_lim'])

        # Put the (enhanced) solution_terminal in self.solution[0]

        self._make_solution_for_final_period(messaging_level=messaging_level,
                                             quietly=True)

    def dolo_model(self):
        # Create a dolo version of the model
        return
        from dolo import yaml_import
        self.dolo_modl = yaml_import(
            '/Volumes/Data/Code/ARK/DARKolo/chimeras/BufferStock/bufferstock.yaml'
        )
        if self.verbose >= 2:
            _log.info(self.dolo_modl)

    def _agent_force_prepare_info_needed_to_begin_solving(self):
        """
        Update any characteristics of the agent that need to be recomputed
        as a result of changes in parameters since the last time the solver was 
        invoked.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        solve_par_vals_now = {}
        if not hasattr(self, 'solve_par_vals'):  # We haven't set it up yet
            self.update_income_process()
            self.update_assets_grid()
        else:  # it has been set up, so see if anything changed
            for par in self.solve_par_vals:
                solve_par_vals_now[par] = getattr(self, par)
            if not solve_par_vals_now == self.solve_par_vals:
                if not self.quietly:
                    _log.info('Some parameter has changed since last update.')
                    _log.info('Storing calculated consequences for grid etc.')
                self.update_income_process()
                self.update_assets_grid()

    pre_solve = _agent_force_prepare_info_needed_to_begin_solving

    # The former "[AgentType].update_pre_solve()" was not good nomenclature --
    #  easy to confuse with the also-existing "[AgentType].pre_solve()" and with
    # "[SolverType].prepare_to_solve()".  The new name,
    #
    # _agent_force_prepare_info_needed_to_begin_solving()
    #
    # is better.  The old one
    # is preserved as an alias, below, to prevent breakage of existing code:

    def update_income_process(self):
        """
        Updates agent's income shock specs based on its current attributes.

        Parameters
        ----------
        none

        Returns:
        --------
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
        # Created so, later, we can determine whether any parameters have changed
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

        # constructed_by: later, we can determine whether another distribution
        # object was constructed using the same method or a different method
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
        self.solve_one_period = make_one_period_oo_solver(
            ConsKinkedRsolver)
        # Make assets grid, income process, terminal solution

    def _agent_force_prepare_info_needed_to_begin_solving(self):
        self.update_assets_grid()
        self.update_income_process()

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
