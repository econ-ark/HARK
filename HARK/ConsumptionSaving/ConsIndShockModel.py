# -*- coding: utf-8 -*-
from types import SimpleNamespace as nsp
from builtins import (range, str, breakpoint)
from copy import copy, deepcopy
import numpy as np
from numpy import dot as 𝔼_dot  # expectations (arg0 and arg1 are prb and val)
from scipy.optimize import newton
from HARK import AgentType, NullFunc, MetricObject, make_one_period_oo_solver
from HARK.interpolation import (CubicInterp, LowerEnvelope, LinearInterp, ValueFuncCRRA, MargValueFuncCRRA,
                                MargMargValueFuncCRRA)
from HARK.distribution import (add_discrete_outcome_constant_mean, calc_expectation,
                               combine_indep_dstns, Lognormal, MeanOneLogNormal, Uniform)
from HARK.utilities import (make_grid_exp_mult, CRRAutility, CRRAutilityP, CRRAutilityPP, CRRAutilityP_inv,
                            CRRAutility_invP, CRRAutility_inv, CRRAutilityP_invP)
from HARK.core import (_log, set_verbosity_level, core_check_condition, get_solve_one_period_args)
from HARK.Calibration.Income.IncomeTools import parse_income_spec, parse_time_params, Cagetti_income
from HARK.datasets.SCF.WealthIncomeDist.SCFDistTools import income_wealth_dists_from_scf
from HARK.datasets.life_tables.us_ssa.SSATools import parse_ssa_life_table
from HARK.ConsumptionSaving.ConsModel import (
    TrnsPars,
)


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
    "ConsPerfForesightSolver",
    "ConsIndShockSetup",
    "ConsIndShockSolverBasic",
    "ConsIndShockSolver",
    "ConsKinkedRsolver",
    "PerfForesightConsumerType",
    "IndShockConsumerType",
    "KinkedRconsumerType",
    "init_perfect_foresight",
    "init_idiosyncratic_shocks",
    "init_kinked_R",
    "init_lifecycle",
    "init_cyclical",
]

utility = CRRAutility
utilityP = CRRAutilityP
utilityPP = CRRAutilityPP
utilityP_inv = CRRAutilityP_inv
utility_invP = CRRAutility_invP
utility_inv = CRRAutility_inv
utilityP_invP = CRRAutilityP_invP

# =====================================================================
# === Classes to solve consumption-saving models ===
# =====================================================================


class ConsumerSolution(MetricObject):
    """
    Solution of single period/stage of a consumption/saving problem with
    one state at decision time: market resources `m`, which includes both
    liquid assets and current income.  Defines a consumption function and
    marginal value function.

    Here and elsewhere in the code, Nrm indicates that variables are normalized
    by permanent income.

    Parameters
    ----------
    cFunc : function
        The consumption function for this period/stage, defined over market
        resources: c = cFunc(m).
    vFunc : function
        The beginning value function for this stage, defined over
        market resources: v = vFunc(m).
    vPfunc : function
        The beginning marginal value function for this period,
        defined over market resources: vP = vPfunc(m).
    vPPfunc : function
        The beginning marginal marginal value function for this
        period, defined over market resources: vPP = vPPfunc(m).
    mNrmMin : float
        The minimum allowable market resources for this period; the consump-
        tion and other functions are undefined for m < mNrmMin.
    hNrm : float
        Human wealth after receiving income this period: PDV of all future
        income, ignoring mortality.
    MPCmin : float
        Infimum of the marginal propensity to consume this period.
        MPC --> MPCmin as m --> infinity.
    MPCmax : float
        Supremum of the marginal propensity to consume this period.
        MPC --> MPCmax as m --> mNrmMin.
    stge_kind : dict
        Dictionary with info about this stage
        One built-in entry keeps track of the nature of the stage:
            {'iter_status':'terminal'}: Terminal (last period of existence)
            {'iter_status':'iterator'}: Solution during iteration
            {'iter_status':'finished'}: Stopping requirements are satisfied
                If stopping requirements are satisfied, {'tolerance':tolerance}
                should exist recording what convergence tolerance was satisfied
        Other uses include keeping track of the nature of the next stage
    parameters_solver : dict
        Stores the parameters with which the solver was called
    """

    # CDC 20210426: vPfunc is a bad choice; we should change it,
    # but doing so will require recalibrating some of our tests
    distance_criteria = ["vPfunc"]  # Bad because it goes to infinity; instead:
#    distance_criteria = ["mNrmStE"]  # mNrmStE if the GIC holds (and it's not close)
#    distance_criteria = ["cFunc"]  # cFunc if the GIC fails

    def __init__(
            self,
            cFunc=None,
            vFunc=None,
            vPfunc=None,
            vPPfunc=None,
            mNrmMin=None,
            hNrm=None,
            MPCmin=None,
            MPCmax=None,
            stge_kind=None,
            parameters_solver=None,
            ** kwds,
    ):
        # Change any missing function inputs to NullFunc
        self.cFunc = cFunc if cFunc is not None else NullFunc()
        self.vFunc = vFunc if vFunc is not None else NullFunc()
        self.vPfunc = vPfunc if vPfunc is not None else NullFunc()
        self.vPPfunc = vPPfunc if vPPfunc is not None else NullFunc()
        self.mNrmMin = mNrmMin
        self.hNrm = hNrm
        self.MPCmin = MPCmin
        self.MPCmax = MPCmax
        self.completed_cycles = 0

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

    def mNrmTrg_finder(self):
        """
        Finds value of (normalized) market resources mNrm at which individual consumer
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

        # Minimum market resources plus next income is okay starting guess
        # Better would be to presere the last value (if it exists)
        # and use that as a starting point

        m_init_guess = self.mNrmMin + self.Ex_IncNrmNxt
        try:  # Find value where argument is zero
            self.mNrmTrg = newton(
                self.Ex_m_tp1_minus_m_t,
                m_init_guess)
        except:
            self.mNrmTrg = None

        return self.mNrmTrg

# ConsPerfForesightSolver class incorporates calcs and info useful for
# models in which perfect foresight does not apply, because the contents
# of the PF model are inherited by a variety of non-perfect-foresight models


class ConsPerfForesightSolver(MetricObject):
    """
    A class for solving a one period perfect foresight
    consumption-saving problem.

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
            self,
            # CDC 20210415: solve_one_cycle provides first arg as solution_next
            # Since it is a required positional argument, we could rename it
            # to reflect our new stages terminology here and througout, but
            # we should not do so until we rename similarly in core.py
            solution_next,
            DiscFac=1.0,
            LivPrb=1.0,
            CRRA=2.0,
            Rfree=1.0,
            PermGroFac=1.0,
            BoroCnstArt=None,
            MaxKinks=None,
            **kwds
    ):
        # Preserve the solver's parameters for later use
        tmp = locals()  # There's got to be a better way to do this; lambda?
        parameters_solver = \
            {x: tmp[x] for x in locals() if x not in
             {'self', 'kwds', 'solution_next'}}
        self.parameters_solver = parameters_solver.update(kwds)
        del tmp

        # Give first arg a name that highlights that it's the next "stage"
        self.soln_futr = soln_futr = solution_next

        # If we've been fed a "terminal" solution, use it as current solution
        if soln_futr.stge_kind['iter_status'] == 'terminal':
            self.soln_crnt = deepcopy(soln_futr)
        # Otherwise create receptacle for construction of solution
        else:
            self.soln_crnt = ConsumerSolution()

        # The solver, and its self, should be disposable; once the solution is
        # produced, there should never be any need for anything in it again

        # .Ths: namespace to store specific info needed to solve this problem
        # Organizing principle: Ths should have a copy of everything
        # needed to re-solve its problem; and everything needed to construct
        # the "fcts" about the problem, so that it could be deepcopied
        # as a standalone object and solved without soln_futr or soln_crnt

        self.soln_crnt.Ths = Ths = TrnsPars()

        Ths.hNrm_tp1 = soln_futr.Ths.hNrm
        Ths.BoroCnstNat_tp1 = soln_futr.Ths.BoroCnstNat
        Ths.mNrmMin_tp1 = soln_futr.Ths.mNrmMin
        Ths.MPCmin_tp1 = soln_futr.Ths.MPCmin
        Ths.MPCmax_tp1 = soln_futr.Ths.MPCmax
        Ths.cFunc_tp1 = soln_futr.cFunc
        Ths.vFunc_tp1 = soln_futr.vFunc
        Ths.vPfunc_tp1 = soln_futr.vPfunc
        if hasattr(soln_futr, 'vPPfunc'):
            Ths.vPPfunc_tp1 = soln_futr.vPPfunc

        self.url_doc_for_solver_get()  # make link to docs

        # urls are used when "fcts" are added
        Ths.url_ref = self.url_ref
        Ths.url_doc = self.url_doc
        Ths.urlroot = self.urlroot

        self.soln_crnt.fcts = {}  # Collect facts about the current stage

        # Store to Ths the exact params with which solver was called
        for key in parameters_solver:
            setattr(self.soln_crnt.Ths, key, parameters_solver[key])

    def url_doc_for_solver_get(self):
        # Generate a url that will locate the documentation
        self.class_name = self.__class__.__name__
        self.url_ref = "https://econ-ark.github.io/BufferStockTheory"
        self.urlroot = self.url_ref+'/#'
        self.url_doc = "https://hark.readthedocs.io/en/latest/search.html?q=" +\
            self.class_name+"&check_keywords=yes&area=default#"

    <_useful_for_further_analysis_ConsPerfForesightSolver(self, futr_Ths):
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

        # Using local variables allows formulae below to be more readable

        soln_crnt = self.soln_crnt
        Ths = self.soln_crnt.Ths
        urlroot = Ths.urlroot

        Ths.DiscLiv = Ths.DiscFac * Ths.LivPrb

        APF_fcts = {
            'about': 'Absolute Patience Factor'
        }
        py___code = '((Rfree * DiscLiv) ** (1.0 / CRRA))'
        Ths.APF = APF = eval(py___code, globals(), Ths.__dict__)
        APF_fcts.update({'latexexpr': r'\APF'})
        APF_fcts.update({'_unicode_': r'Þ'})
        APF_fcts.update({'urlhandle': urlroot+'APF'})
        APF_fcts.update({'py___code': py___code})
        APF_fcts.update({'value_now': APF})
        soln_crnt.fcts.update({'APF': APF_fcts})
        soln_crnt.Ths.APF_fcts = APF_fcts

        AIC_fcts = {
            'about': 'Absolute Impatience Condition'
        }
        AIC_fcts.update({'latexexpr': r'\AIC'})
        AIC_fcts.update({'urlhandle': urlroot+'AIC'})
        AIC_fcts.update({'py___code': 'test: APF < 1'})
        soln_crnt.fcts.update({'AIC': AIC_fcts})
        soln_crnt.Ths.AIC_fcts = AIC_fcts

        RPF_fcts = {
            'about': 'Return Patience Factor'
        }
        py___code = 'APF / Rfree'
        Ths.RPF = RPF = eval(py___code, globals(), Ths.__dict__)
        RPF_fcts.update({'latexexpr': r'\RPF'})
        RPF_fcts.update({'_unicode_': r'Þ_R'})
        RPF_fcts.update({'urlhandle': urlroot+'RPF'})
        RPF_fcts.update({'py___code': py___code})
        RPF_fcts.update({'value_now': RPF})
        soln_crnt.fcts.update({'RPF': RPF_fcts})
        soln_crnt.Ths.RPF_fcts = RPF_fcts

        RIC_fcts = {
            'about': 'Growth Impatience Condition'
        }
        RIC_fcts.update({'latexexpr': r'\RIC'})
        RIC_fcts.update({'urlhandle': urlroot+'RIC'})
        RIC_fcts.update({'py___code': 'test: RPF < 1'})
        soln_crnt.fcts.update({'RIC': RIC_fcts})
        soln_crnt.Ths.RIC_fcts = RIC_fcts

        GPFRaw_fcts = {
            'about': 'Growth Patience Factor'
        }
        py___code = 'APF / PermGroFac'
        Ths.GPFRaw = GPFRaw = eval(py___code, globals(), Ths.__dict__)
        GPFRaw_fcts.update({'latexexpr': '\GPFRaw'})
        GPFRaw_fcts.update({'_unicode_': r'Þ_Γ'})
        GPFRaw_fcts.update({'urlhandle': urlroot+'GPFRaw'})
        GPFRaw_fcts.update({'py___code': py___code})
        GPFRaw_fcts.update({'value_now': GPFRaw})
        soln_crnt.fcts.update({'GPFRaw': GPFRaw_fcts})
        soln_crnt.Ths.GPFRaw_fcts = GPFRaw_fcts
        soln_crnt.GPFRaw = GPFRaw

        GICRaw_fcts = {
            'about': 'Growth Impatience Condition'
        }
        GICRaw_fcts.update({'latexexpr': r'\GICRaw'})
        GICRaw_fcts.update({'urlhandle': urlroot+'GICRaw'})
        GICRaw_fcts.update({'py___code': 'test: GPFRaw < 1'})
        soln_crnt.fcts.update({'GICRaw': GICRaw_fcts})
        soln_crnt.Ths.GICRaw_fcts = GICRaw_fcts

        GPFLiv_fcts = {
            'about': 'Mortality-Adjusted Growth Patience Factor'
        }
        py___code = 'APF * LivPrb / PermGroFac'
        Ths.GPFLiv = GPFLiv = eval(py___code, globals(), Ths.__dict__)
        GPFLiv_fcts.update({'latexexpr': '\GPFLiv'})
        GPFLiv_fcts.update({'urlhandle': urlroot+'GPFLiv'})
        GPFLiv_fcts.update({'py___code': py___code})
        GPFLiv_fcts.update({'value_now': GPFLiv})
        soln_crnt.fcts.update({'GPFLiv': GPFLiv_fcts})
        soln_crnt.Ths.GPFLiv_fcts = GPFLiv_fcts
        soln_crnt.GPFLiv = GPFLiv

        GICLiv_fcts = {
            'about': 'Growth Impatience Condition'
        }
        GICLiv_fcts.update({'latexexpr': r'\GICLiv'})
        GICLiv_fcts.update({'urlhandle': urlroot+'GICLiv'})
        GICLiv_fcts.update({'py___code': 'test: GPFLiv < 1'})
        soln_crnt.fcts.update({'GICLiv': GICLiv_fcts})
        soln_crnt.Ths.GICLiv_fcts = GICLiv_fcts

        PF_RNrm_fcts = {
            'about': 'Growth-Normalized PF Return Factor'
        }
        py___code = 'Rfree/PermGroFac'
        Ths.PF_RNrm = PF_RNrm = eval(py___code, globals(), Ths.__dict__)
        PF_RNrm_fcts.update({'latexexpr': r'\PFRNrm'})
        PF_RNrm_fcts.update({'_unicode_': r'R/Γ'})
        PF_RNrm_fcts.update({'py___code': py___code})
        PF_RNrm_fcts.update({'value_now': PF_RNrm})
        soln_crnt.fcts.update({'PF_RNrm': PF_RNrm_fcts})
        soln_crnt.Ths.PF_RNrm_fcts = PF_RNrm_fcts
        soln_crnt.PF_RNrm = PF_RNrm

        Inv_PF_RNrm_fcts = {
            'about': 'Inv of Growth-Normalized PF Return Factor'
        }
        py___code = '1 / PF_RNrm'
        Ths.Inv_PF_RNrm = Inv_PF_RNrm = eval(py___code, globals(), Ths.__dict__)
        Inv_PF_RNrm_fcts.update({'latexexpr': r'\InvPFRNrm'})
        Inv_PF_RNrm_fcts.update({'_unicode_': r'Γ/R'})
        Inv_PF_RNrm_fcts.update({'py___code': py___code})
        Inv_PF_RNrm_fcts.update({'value_now': Inv_PF_RNrm})
        soln_crnt.fcts.update({'Inv_PF_RNrm': Inv_PF_RNrm_fcts})
        soln_crnt.Ths.Inv_PF_RNrm_fcts = Inv_PF_RNrm_fcts
        soln_crnt.Inv_PF_RNrm = Inv_PF_RNrm

        FHWF_fcts = {
            'about': 'Finite Human Wealth Factor'
        }
        py___code = 'PermGroFac / Rfree'
        Ths.FHWF = FHWF = eval(py___code, globals(), Ths.__dict__)
        FHWF_fcts.update({'latexexpr': r'\FHWF'})
        FHWF_fcts.update({'_unicode_': r'R/Γ'})
        FHWF_fcts.update({'urlhandle': urlroot+'FHWF'})
        FHWF_fcts.update({'py___code': py___code})
        FHWF_fcts.update({'value_now': FHWF})
        soln_crnt.fcts.update({'FHWF': FHWF_fcts})
        soln_crnt.Ths.FHWF_fcts = FHWF_fcts
        soln_crnt.FHWF = FHWF

        FHWC_fcts = {
            'about': 'Finite Human Wealth Condition'
        }
        FHWC_fcts.update({'latexexpr': r'\FHWC'})
        FHWC_fcts.update({'urlhandle': urlroot+'FHWC'})
        FHWC_fcts.update({'py___code': 'test: FHWF < 1'})
        soln_crnt.fcts.update({'FHWC': FHWC_fcts})
        soln_crnt.Ths.FHWC_fcts = FHWC_fcts

        hNrmInf_fcts = {
            'about': 'Human wealth for inf hor'
        }
        py___code = '1/(1-FHWF) if (FHWF < 1) else np.inf'
        Ths.hNrmInf = hNrmInf = eval(py___code, globals(), Ths.__dict__)
        hNrmInf_fcts = dict({'latexexpr': '1/(1-\FHWF)'})
        hNrmInf_fcts.update({'value_now': hNrmInf})
        hNrmInf_fcts.update({'py___code': py___code})
        soln_crnt.fcts.update({'hNrmInf': hNrmInf_fcts})
        soln_crnt.Ths.hNrmInf_fcts = hNrmInf_fcts
        soln_crnt.hNrmInf = hNrmInf

        DiscGPFRawCusp_fcts = {
            'about': 'DiscFac s.t. GPFRaw = 1'
        }
        py___code = '( PermGroFac                       ** CRRA)/(Rfree)'
        Ths.DiscGPFRawCusp = DiscGPFRawCusp = eval(py___code, globals(), Ths.__dict__)
        DiscGPFRawCusp_fcts.update({'latexexpr': '\PermGroFac^{\CRRA}/\Rfree'})
        DiscGPFRawCusp_fcts.update({'value_now': DiscGPFRawCusp})
        DiscGPFRawCusp_fcts.update({'py___code': py___code})
        soln_crnt.fcts.update({'DiscGPFRawCusp': DiscGPFRawCusp_fcts})
        soln_crnt.Ths.DiscGPFRawCusp_fcts = DiscGPFRawCusp_fcts
        soln_crnt.DiscGPFRawCusp = DiscGPFRawCusp

        DiscGPFLivCusp_fcts = {
            'about': 'DiscFac s.t. GPFLiv = 1'
        }
        py___code = '( PermGroFac                       ** CRRA)/(Rfree*LivPrb)'
        Ths.DiscGPFLivCusp = DiscGPFLivCusp = eval(py___code, globals(), Ths.__dict__)
        DiscGPFLivCusp_fcts.update({'latexexpr': '\PermGroFac^{\CRRA}/(\Rfree\LivPrb)'})
        DiscGPFLivCusp_fcts.update({'value_now': DiscGPFLivCusp})
        DiscGPFLivCusp_fcts.update({'py___code': py___code})
        soln_crnt.fcts.update({'DiscGPFLivCusp': DiscGPFLivCusp_fcts})
        soln_crnt.Ths.DiscGPFLivCusp_fcts = DiscGPFLivCusp_fcts
        soln_crnt.DiscGPFLivCusp = DiscGPFLivCusp

        FVAF_fcts = {
            'about': 'Finite Value of Autarky Factor'
        }
        py___code = 'LivPrb * DiscLiv'
        Ths.FVAF = FVAF = eval(py___code, globals(), Ths.__dict__)
        FVAF_fcts.update({'latexexpr': r'\FVAFPF'})
        FVAF_fcts.update({'urlhandle': urlroot+'FVAFPF'})
        FVAF_fcts.update({'py___code': py___code})
        soln_crnt.fcts.update({'FVAF': FVAF_fcts})
        soln_crnt.Ths.FVAF_fcts = FVAF_fcts
        soln_crnt.FVAF = FVAF

        FVAC_fcts = {
            'about': 'Finite Value of Autarky Condition - Perfect Foresight'
        }
        FVAC_fcts.update({'latexexpr': r'\FVACPF'})
        FVAC_fcts.update({'urlhandle': urlroot+'FVACPF'})
        FVAC_fcts.update({'py___code': 'test: FVAFPF < 1'})
        soln_crnt.fcts.update({'FVAC': FVAC_fcts})
        soln_crnt.Ths.FVAC_fcts = FVAC_fcts

        hNrm_fcts = {
            'about': 'Human Wealth '
        }
        py___code = '((PermGroFac / Rfree) * (1.0 + hNrm_tp1))'
        if soln_crnt.stge_kind['iter_status'] == 'terminal':  # kludge:
            Ths.hNrm_tp1 = -1.0  # causes hNrm = 0 for final period
        Ths.hNrm = hNrm = eval(py___code, globals(), Ths.__dict__)
        hNrm_fcts.update({'latexexpr': r'\hNrm'})
        hNrm_fcts.update({'_unicode_': r'R/Γ'})
        hNrm_fcts.update({'urlhandle': urlroot+'hNrm'})
        hNrm_fcts.update({'py___code': py___code})
        hNrm_fcts.update({'value_now': hNrm})
        soln_crnt.fcts.update({'hNrm': hNrm_fcts})
        soln_crnt.Ths.hNrm_fcts = hNrm_fcts
        soln_crnt.hNrm = hNrm

        # That's the end of things that are identical for PF and non-PF models

        BoroCnstNat_fcts = {
            'about': 'Natural Borrowing Constraint'
        }
        if soln_crnt.stge_kind['iter_status'] == 'terminal':  # kludge
            Ths.mNrmMin_tp1 = Ths.tranShkMin  # causes BoroCnstNat = 0 in term
        py___code = '(mNrmMin_tp1 - tranShkMin)*(PermGroFac/Rfree)*permShkMin'
        Ths.BoroCnstNat = BoroCnstNat = eval(py___code, globals(), Ths.__dict__)
        BoroCnstNat_fcts.update({'latexexpr': r'\BoroCnstNat'})
        BoroCnstNat_fcts.update({'_unicode_': r''})
        BoroCnstNat_fcts.update({'urlhandle': urlroot+'BoroCnstNat'})
        BoroCnstNat_fcts.update({'py___code': py___code})
        BoroCnstNat_fcts.update({'value_now': BoroCnstNat})
        soln_crnt.fcts.update({'BoroCnstNat': BoroCnstNat_fcts})
        soln_crnt.Ths.BoroCnstNat_fcts = BoroCnstNat_fcts
        soln_crnt.BoroCnstNat = BoroCnstNat

        BoroCnstArt_fcts = {
            'about': 'Artificial Borrowing Constraint'
        }
        py___code = 'BoroCnstArt if (BoroCnstArt and BoroCnstNat < BoroCnstArt) else BoroCnstNat'
        Ths.BoroCnstArt = BoroCnstArt = eval(py___code, globals(), Ths.__dict__)
        BoroCnstArt_fcts.update({'latexexpr': r'\BoroCnstArt'})
        BoroCnstArt_fcts.update({'_unicode_': r''})
        BoroCnstArt_fcts.update({'urlhandle': urlroot+'BoroCnstArt'})
        BoroCnstArt_fcts.update({'py___code': py___code})
        BoroCnstArt_fcts.update({'value_now': BoroCnstArt})
        soln_crnt.fcts.update({'BoroCnstArt': BoroCnstArt_fcts})
        soln_crnt.Ths.BoroCnstArt_fcts = BoroCnstArt_fcts
        soln_crnt.BoroCnstArt = BoroCnstArt

        mNrmMin_fcts = {
            'about': 'Minimum mNrm'
        }
        py___code = 'max([BoroCnstNat, BoroCnstArt])'
        Ths.mNrmMin = mNrmMin = eval(py___code, globals(), Ths.__dict__)
        mNrmMin_fcts.update({'latexexpr': r'\mNrmMin'})
        mNrmMin_fcts.update({'py___code': py___code})
        soln_crnt.fcts.update({'mNrmMin': mNrmMin_fcts})
        soln_crnt.Ths.mNrmMin_fcts = mNrmMin_fcts
        soln_crnt.mNrmMin = mNrmMin

        MPCmin_fcts = {
            'about': 'Minimal MPC in current period as m -> infty'
        }
        py___code = '1.0 / (1.0 + (RPF / MPCmin_tp1))'
        if soln_crnt.stge_kind['iter_status'] == 'terminal':  # kludge:
            Ths.MPCmin_tp1 = float('inf')  # causes MPCmin = 1 for final period
        Ths.MPCmin = MPCmin = eval(py___code, globals(), Ths.__dict__)
        MPCmin_fcts.update({'latexexpr': r''})
        MPCmin_fcts.update({'urlhandle': urlroot+'MPCmin'})
        MPCmin_fcts.update({'py___code': py___code})
        MPCmin_fcts.update({'value_now': MPCmin})
        soln_crnt.fcts.update({'MPCmin': MPCmin_fcts})
        soln_crnt.Ths.MPCmin_fcts = MPCmin_fcts
        soln_crnt.MPCmin = soln_crnt.MPCmin = MPCmin

        MPCmax_fcts = {
            'about': 'Maximal MPC in current period as m -> infty'
        }
        py___code = '1.0 / (1.0 + (RPF / MPCmax_tp1))'
        if soln_crnt.stge_kind['iter_status'] == 'terminal':  # kludge:
            Ths.MPCmax_tp1 = float('inf')  # causes MPCmax = 1 for final period
        Ths.MPCmax = MPCmax = eval(py___code, globals(), Ths.__dict__)
        MPCmax_fcts.update({'latexexpr': r''})
        MPCmax_fcts.update({'urlhandle': urlroot+'MPCmax'})
        MPCmax_fcts.update({'py___code': py___code})
        MPCmax_fcts.update({'value_now': MPCmax})
        soln_crnt.fcts.update({'MPCmax': MPCmax_fcts})
        soln_crnt.Ths.MPCmax_fcts = MPCmax_fcts
        soln_crnt.MPCmax = soln_crnt.MPCmax = MPCmax

        return soln_crnt

    def def_utility_funcs(self, stge):
        """
        Defines CRRA utility function for this period (and its derivatives,
        and their inverses), saving them as attributes of self for other methods
        to use.

        Parameters
        ----------
        solution_stage

        Returns
        -------
        none
        """
        # utility function
        stge.u = lambda c: utility(c, gam=stge.CRRA)  
         # marginal utility function
        stge.uP = lambda c: utilityP(c, gam=stge.CRRA) 
        # marginal marginal utility function
        stge.uPP = lambda c: utilityPP(c, gam=stge.CRRA)  
        
        # Inverses thereof
        stge.uPinv = lambda u: utilityP_inv(u, gam=stge.CRRA)
        stge.uPinvP = lambda u: utilityP_invP(u, gam=stge.CRRA)
        stge.uinvP = lambda u: utility_invP(u, gam=stge.CRRA)
        stge.uinv = lambda u: utility_inv(u, gam=stge.CRRA)  # in case vFuncBool

        return stge

    def def_value_funcs(self, stge):
        """
        Defines the value and marginal value functions for this period.
        mNrmMin.  See PerfForesightConsumerType.ipynb
        for a brief explanation and the links below for a fuller treatment.
        https://github.com/llorracc/SolvingMicroDSOPs/#vFuncPF

        Parameters
        ----------
        solution_stage

        Returns
        -------
        None

        Notes
        -------
        Uses the fact that for a perfect foresight CRRA utility problem,
        if the MPC in period t is :math:`\kappa_{t}`, and relative risk
        aversion :math:`\rho`, then the inverse value vFuncNvrs has a
        constant slope of :math:`\kappa_{t}^{-\rho/(1-\rho)}` and
        vFuncNvrs has value of zero at the lower bound of market resources
        """

        # See PerfForesightConsumerType.ipynb docs for derivations
        vFuncNvrsSlope = stge.MPCmin ** (-stge.CRRA / (1.0 - stge.CRRA))
        vFuncNvrs = LinearInterp(
            np.array([stge.mNrmMin, stge.mNrmMin + 1.0]),
            np.array([0.0, vFuncNvrsSlope]),
        )
        stge.vFunc = ValueFuncCRRA(vFuncNvrs, stge.CRRA)
        stge.vPfunc = MargValueFuncCRRA(stge.cFunc, stge.CRRA)
        return stge

    def make_cFunc_PF(self):
        """
        Makes the (linear) consumption function for this period.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Reduce cluttered formulae with local copies

        CRRA = self.soln_crnt.Ths.CRRA
        Rfree = self.soln_crnt.Ths.Rfree
        PermGroFac = self.soln_crnt.Ths.PermGroFac
        MPCmin = self.soln_crnt.MPCmin
        DiscLiv = self.soln_crnt.Ths.DiscLiv
        MaxKinks = self.soln_crnt.MaxKinks

        # Use local value of BoroCnstArt to prevent comparing None and float
        if self.soln_crnt.Ths.BoroCnstArt is None:
            BoroCnstArt = -np.inf
        else:
            BoroCnstArt = self.soln_crnt.Ths.BoroCnstArt

        # Extract kink points in next period's consumption function;
        # don't take the last one; it only defines extrapolation, is not kink.
        mNrmNext = self.soln_futr.cFunc.x_list[:-1]
        cNrmNext = self.soln_futr.cFunc.y_list[:-1]

        # Calculate end-of-period asset vals that would reach those kink points
        # next period, then invert first order condition to get c. Then find
        # endogenous gridpoint (kink point) today corresponding to each kink
        aNrm = (PermGroFac / Rfree) * (mNrmNext - 1.0)
        cNrm = (DiscLiv * Rfree) ** (-1.0 / CRRA) * (
            PermGroFac * cNrmNext
        )
        mNrm = aNrm + cNrm

        # Add additional point to the list of gridpoints for extrapolation,
        # using the new value of the lower bound of the MPC.
        mNrm = np.append(mNrm, mNrm[-1] + 1.0)
        cNrm = np.append(cNrm, cNrm[-1] + MPCmin)
        # If artificial borrowing constraint binds, combine constrained and
        # unconstrained consumption functions.
        if BoroCnstArt > mNrm[0]:
            # Find the highest index where constraint binds
            cNrmCnst = mNrm - BoroCnstArt
            CnstBinds = cNrmCnst < cNrm
            idx = np.where(CnstBinds)[0][-1]
            if idx < (mNrm.size - 1):
                # If not the *very last* index, find the the critical level
                # of mNrm where artificial borrowing contraint begins to bind.
                d0 = cNrm[idx] - cNrmCnst[idx]
                d1 = cNrmCnst[idx + 1] - cNrm[idx + 1]
                m0 = mNrm[idx]
                m1 = mNrm[idx + 1]
                alpha = d0 / (d0 + d1)
                mCrit = m0 + alpha * (m1 - m0)
                # Adjust grids of mNrm and cNrm to account for constraint.
                cCrit = mCrit - BoroCnstArt
                mNrm = np.concatenate(([BoroCnstArt, mCrit], mNrm[(idx + 1):]))
                cNrm = np.concatenate(([0.0, cCrit], cNrm[(idx + 1):]))
            else:
                # If it *is* the last index, then there are only three points
                # that characterize the c function: the artificial borrowing
                # constraint, the constraint kink, and the extrapolation point
                mXtra = (cNrm[-1] - cNrmCnst[-1]) / (1.0 - MPCmin)
                mCrit = mNrm[-1] + mXtra
                cCrit = mCrit - BoroCnstArt
                mNrm = np.array([BoroCnstArt, mCrit, mCrit + 1.0])
                cNrm = np.array([0.0, cCrit, cCrit + MPCmin])
                # If mNrm, cNrm grids have become too large, throw out last
                # kink point, being sure to adjust the extrapolation.
        if mNrm.size > MaxKinks:
            mNrm = np.concatenate((mNrm[:-2], [mNrm[-3] + 1.0]))
            cNrm = np.concatenate((cNrm[:-2], [cNrm[-3] + MPCmin]))
            # Construct the consumption function as a linear interpolation.
        self.cFunc = self.soln_crnt.cFunc = LinearInterp(mNrm, cNrm)
        # Calculate the upper bound of the MPC as the slope of bottom segment.
        self.soln_crnt.MPCmax = (cNrm[1] - cNrm[0]) / (mNrm[1] - mNrm[0])

        # Add two attributes to enable calc of steady state market resources.
        self.soln_crnt.Ex_IncNrmNxt = 1.0  # Perfect foresight income of 1
        self.soln_crnt.mNrmMin = mNrm[0]  # Relabel for compat w add_mNrmStE

    def solve(self):  # in ConsPerfForesightSolver
        """
        Solves the one-period/stage perfect foresight consumption-saving problem.

        Parameters
        ----------
        None

        Returns
        -------
        solution : ConsumerSolution
            The solution to this period/stage's problem.
        """
        self.soln_crnt.make_cFunc_PF()
        self.soln_crnt = self.soln_crnt.def_value_funcs(self.soln_crnt)

        return self.soln_crnt

    def check_AIC(self, stge, verbose=None):
        """
        Evaluate and report on the Absolute Impatience Condition
        """
        name = "AIC"
   
        def test(stge): return stge.Ths.APF < 1

        messages = {
            True: "\nThe Absolute Patience Factor for the supplied parameter values, APF={0.APF}, satisfies the Absolute Impatience Condition (AIC), which requires APF < 1: "+stge.Ths.AIC_fcts['urlhandle'],
            False: "\nThe Absolute Patience Factor for the supplied parameter values, APF={0.APF}, violates the Absolute Impatience Condition (AIC), which requires APF < 1: "+stge.Ths.AIC_fcts['urlhandle']
        }
        verbose_messages = {
            True: "  Because the APF < 1,  the absolute amount of consumption is expected to fall over time.  \n",
            False: "  Because the APF > 1, the absolute amount of consumption is expected to grow over time.  \n",
        }

        stge.Ths.AIC = core_check_condition(name, test, messages, verbose,
                                            verbose_messages, "APF", stge)

    def check_FVAC(self, stge, verbose=None):
        """
        Evaluate and report on the Finite Value of Autarky Condition
        """
        name = "FVAC"
   
        def test(stge): return stge.Ths.FVAF < 1

        messages = {
            True: "\nThe Finite Value of Autarky Factor for the supplied parameter values, FVAF={0.FVAF}, satisfies the Finite Value of Autarky Condition, which requires FVAF < 1: "+stge.Ths.FVAC_fcts['urlhandle'],
            False: "\nThe Finite Value of Autarky Factor for the supplied parameter values, FVAF={0.FVAF}, violates the Finite Value of Autarky Condition, which requires FVAF: "+stge.Ths.FVAC_fcts['urlhandle']
        }
        verbose_messages = {
            True: "  Therefore, a nondegenerate solution exists if the RIC also holds. ("+stge.Ths.FVAC_fcts['urlhandle']+")\n",
            False: "  Therefore, a nondegenerate solution exits if the RIC holds.\n",
        }

        stge.Ths.FVAC = core_check_condition(name, test, messages, verbose,
                                             verbose_messages, "FVAC", stge)

    def check_GICRaw(self, stge, verbose=None):
        """
        Evaluate and report on the Growth Impatience Condition
        """
        name = "GICRaw"
   
        def test(stge): return stge.Ths.GPFRaw < 1

        messages = {
            True: "\nThe Growth Patience Factor for the supplied parameter values, GPF={0.GPFRaw}, satisfies the Growth Impatience Condition (GIC), which requires GPF < 1: "+stge.Ths.GICRaw_fcts['urlhandle'],
            False: "\nThe Growth Patience Factor for the supplied parameter values, GPF={0.GPFRaw}, violates the Growth Impatience Condition (GIC), which requires GPF < 1: "+stge.Ths.GICRaw_fcts['urlhandle'],
        }
        verbose_messages = {
            True: "  Therefore,  for a perfect foresight consumer, the ratio of individual wealth to permanent income is expected to fall indefinitely.    \n",
            False: "  Therefore, for a perfect foresight consumer, the ratio of individual wealth to permanent income is expected to rise toward infinity. \n"
        }
        stge.Ths.GICRaw = core_check_condition(name, test, messages, verbose,
                                               verbose_messages, "GPFRaw", stge)

    def check_GICLiv(self, stge, verbose=None):
        name = "GICLiv"
   
        def test(stge): return stge.Ths.GPFLiv < 1

        messages = {
            True: "\nThe Mortality Adjusted Aggregate Growth Patience Factor for the supplied parameter values, GPFLiv={0.GPFLiv}, satisfies the Mortality Adjusted Aggregate Growth Imatience Condition (GICLiv): "+stge.Ths.GPFLiv_fcts['urlhandle'],
            False: "\nThe Mortality Adjusted Aggregate Growth Patience Factor for the supplied parameter values, GPFLiv={0.GPFLiv}, violates the Mortality Adjusted Aggregate Growth Imatience Condition (GICLiv): "+stge.Ths.GPFLiv_fcts['urlhandle'],
        }
        verbose_messages = {
            True: "  Therefore, a target level of the ratio of aggregate market resources to aggregate permanent income exists ("+stge.Ths.GPFLiv_fcts['urlhandle']+")\n",
            False: "  Therefore, a target ratio of aggregate resources to aggregate permanent income may not exist ("+stge.Ths.GPFLiv_fcts['urlhandle']+")\n",
        }
        stge.Ths.GICLiv = core_check_condition(name, test, messages, verbose,
                                               verbose_messages, "GPFLiv", stge)

    def check_RIC(self, stge, verbose=None):
        """
        Evaluate and report on the Return Impatience Condition
        """

        name = "RIC"
   
        def test(stge): return stge.Ths.RPF < 1

        messages = {
            True: "\nThe Return Patience Factor for the supplied parameter values, RPF= {0.RPF}, satisfies the Return Impatience Condition (RIC), which requires RPF < 1: "+stge.Ths.RPF_fcts['urlhandle'],
            False: "\nThe Return Patience Factor for the supplied parameter values, RPF= {0.RPF}, violates the Return Impatience Condition (RIC), which requires RPF < 1: "+stge.Ths.RPF_fcts['urlhandle'],
        }
        verbose_messages = {
            True: "  Therefore, the limiting consumption function is not c(m)=0 for all m\n",
            False: "  Therefore, if the FHWC is satisfied, the limiting consumption function is c(m)=0 for all m.\n",
        }
        stge.Ths.RIC = core_check_condition(name, test, messages, verbose,
                                            verbose_messages, "RPF", stge)

    def check_FHWC(self, stge, verbose=None):
        """
        Evaluate and report on the Finite Human Wealth Condition
        """
        name = "FHWC"
   
        def test(stge): return stge.Ths.FHWF < 1

        messages = {
            True: "\nThe Finite Human Wealth Factor value for the supplied parameter values, FHWF={0.FHWF}, satisfies the Finite Human Wealth Condition (FHWC), which requires FHWF < 1: "+stge.Ths.FHWC_fcts['urlhandle'],
            False: "\nThe Finite Human Wealth Factor value for the supplied parameter values, FHWF={0.FHWF}, violates the Finite Human Wealth Condition (FHWC), which requires FHWF < 1: "+stge.Ths.FHWC_fcts['urlhandle'],
        }
        verbose_messages = {
            True: "  Therefore, the limiting consumption function is not c(m)=Infinity ("+stge.Ths.FHWC_fcts['urlhandle']+")\n  Human wealth normalized by permanent income is {0.hNrmInf}.\n",
            False: "  Therefore, the limiting consumption function is c(m)=Infinity for all m unless the RIC is also violated.\n  If both FHWC and RIC fail and the consumer faces a liquidity constraint, the limiting consumption function is nondegenerate but has a limiting slope of 0. ("+stge.Ths.FHWC_fcts['urlhandle']+")\n",
        }
        stge.Ths.FHWC = core_check_condition(name, test, messages, verbose,
                                             verbose_messages, "FHWF", stge)

    def check_GICNrm(self, stge, verbose=None):
        """
        Check Individual Growth Patience Factor.
        """
        name = "GICNrm"
   
        def test(stge): return stge.Ths.GPFNrm <= 1

        messages = {
            True: "\nThe Normalized Growth Patience Factor GPFNrm for the supplied parameter values, GPFNrm={0.GPFNrm}, satisfies the Normalized Growth Impatience Condition (GICNrm), which requires GICNrm < 1: "+stge.Ths.GPFNrm_fcts['urlhandle']+"\n",
            False: "\nThe Normalized Growth Patience Factor GPFNrm for the supplied parameter values, GPFNrm={0.GPFNrm}, violates the Normalized Growth Impatience Condition (GICNrm), which requires GICNrm < 1: "+stge.Ths.GPFNrm_fcts['urlhandle']+"\n",
        }
        verbose_messages = {
            True: " Therefore, a target level of the individual market resources ratio m exists ("+stge.Ths.GICNrm_fcts['urlhandle']+").\n",
            False: " Therefore, a target ratio of individual market resources to individual permanent income does not exist.  ("+stge.Ths.GICNrm_fcts['urlhandle']+")\n",
        }
        stge.Ths.GPFNrm = core_check_condition(name, test, messages, verbose,
                                               verbose_messages, "GPFNrm", stge)

    def check_WRIC(self, stge, verbose=None):
        """
        Evaluate and report on the Weak Return Impatience Condition
        [url]/#WRIC modified to incorporate LivPrb
        """

        name = "WRIC"
   
        def test(stge): return stge.Ths.WRPF <= 1

        messages = {
            True: "\nThe Weak Return Patience Factor value for the supplied parameter values, WRPF={0.WRPF}, satisfies the Weak Return Impatience Condition, which requires WRIF < 1: "+stge.Ths.WRIC_fcts['urlhandle'],
            False: "\nThe Weak Return Patience Factor value for the supplied parameter values, WRPF={0.WRPF}, violates the Weak Return Impatience Condition, which requires WRIF < 1: "+stge.Ths.WRIC_fcts['urlhandle'],
        }

        verbose_messages = {
            True: "  Therefore, a nondegenerate solution exists if the FVAC is also satisfied. ("+stge.Ths.WRIC_fcts['urlhandle']+")\n",
            False: "  Therefore, a nondegenerate solution is not available ("+stge.Ths.WRIC_fcts['urlhandle']+")\n",
        }
        stge.Ths.WRIC = core_check_condition(name, test, messages, verbose,
                                             verbose_messages, "WRPF", stge)

    def check_conditions(self, soln_crnt, verbose=None):
        """
        Checks whether the instance's type satisfies the:

        ============= ===================================================
        Acronym        Condition
        ============= ===================================================
        AIC           Absolute Impatience Condition
        RIC           Return Impatience Condition
        GIC           Growth Impatience Condition
        GICLiv        GIC adjusting for constant probability of mortality
        GICNrm        GIC adjusted for uncertainty in permanent income
        FHWC          Finite Human Wealth Condition
        FVAC          Finite Value of Autarky Condition
        ============= ===================================================

        Depending on the configuration of parameter values, some combination of
        these conditions must be satisfied in order for the problem to have
        a nondegenerate soln_crnt. To check which conditions are required,
        in the verbose mode, a reference to the relevant theoretical literature
        is made.

        Parameters
        ----------
        verbose : boolean
            Specifies different levels of verbosity of feedback. When False, it only reports whether the
            instance's type fails to satisfy a particular condition. When True, it reports all results, i.e.
            the factor values for all conditions.

        Returns
        -------
        None
        """
        self.soln_crnt.Ths.conditions = {}  # Keep track of truth value of conditions
        self.soln_crnt.violated = False  # True means solution is degenerate

#        # This method only checks for the conditions for infinite horizon models
#        # with a 1 period cycle. If these conditions are not met, we exit early.
#        if self.parameters_model['cycles'] != 0 \
#           or self.parameters_model['T_cycle'] > 1:
#            return

        if not hasattr(self, 'verbose'):  # If verbose not set yet
            self.verbose = 0 if verbose is None else verbose
        else:
            verbose = self.verbose if verbose is None else verbose

        self.check_AIC(soln_crnt, verbose)
        self.check_FHWC(soln_crnt, verbose)
        self.check_RIC(soln_crnt, verbose)
        self.check_GICRaw(soln_crnt, verbose)
        self.check_GICLiv(soln_crnt, verbose)
        self.check_FVAC(soln_crnt, verbose)

        # violated flag is true if the model has no nondegenerate solution
        if hasattr(self.soln_crnt.Ths, "BoroCnstArt") \
                and self.soln_crnt.Ths.BoroCnstArt is not None:
            self.soln_crnt.violated = not self.soln_crnt.Ths.conditions["RIC"]
            # If BoroCnstArt exists but RIC fails, limiting soln is c(m)=0
        else:  # If no constraint,
            self.soln_crnt.violated = \
                not self.soln_crnt.Ths.conditions["RIC"] or not self.soln_crnt.Ths.conditions[
                    "FHWC"]    # c(m)=0 or \infty


###############################################################################
###############################################################################


class ConsIndShockSetup(ConsPerfForesightSolver):
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

    # # Get the "further info" method from the perfect foresight solver
    # def add_info_useful_for_further_analysis_ConsPerfForesightSolver(self, soln_crnt):
    #     super().add_fcts_to_soln(soln_crnt)

    def __init__(  # CDC 20210416: Params shared with PF are in different order. Fix
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
            vFuncBool,
            CubicBool,
            permShkDstn,
            tranShkDstn,
            **kwds
    ):
        # Need to reorder params by hand in case someone tries positional solve
        ConsPerfForesightSolver.__init__(self,
                                         solution_next,
                                         DiscFac=DiscFac,
                                         LivPrb=LivPrb,
                                         CRRA=CRRA,
                                         Rfree=Rfree,
                                         PermGroFac=PermGroFac,
                                         BoroCnstArt=BoroCnstArt,
                                         IncShkDstn=IncShkDstn,
                                         permShkDstn=permShkDstn,
                                         tranShkDstn=tranShkDstn,
                                         **kwds
                                         )  # First execute PF solver init


        soln_futr = solution_next
        soln_crnt = self.soln_crnt  # for terminal solution, soln_crnt = soln_futr

        # Ths really refers to end-of-this-period items
        Ths = soln_crnt.Ths

        IncShkDstn = Ths.IncShkDstn
        permShkDstn = Ths.permShkDstn
        tranShkDstn = Ths.tranShkDstn

        # In which column is each object stored in IncShkDstn?
        Ths.permPos = IncShkDstn.parameters['ShkPosn']['perm']
        Ths.tranPos = IncShkDstn.parameters['ShkPosn']['tran']

        # Bcst are "broadcasted" values: serial list of every possible combo
        # Makes it easy to take expectations
        Ths.permShkValsBcst = permShkValsBcst = IncShkDstn.X[Ths.permPos]
        Ths.tranShkValsBcst = tranShkValsBcst = IncShkDstn.X[Ths.tranPos]
        Ths.ShkPrbs = ShkPrbs = IncShkDstn.pmf

        Ths.permShkPrbs = permShkPrbs = permShkDstn.pmf
        Ths.permShkVals = permShkVals = permShkDstn.X

        Ths.tranShkPrbs = tranShkPrbs = tranShkDstn.pmf
        Ths.tranShkVals = tranShkVals = tranShkDstn.X

        Ths.permShkMin = permShkMin = np.min(permShkVals)
        Ths.tranShkMin = tranShkMin = np.min(tranShkVals)

        Ths.UnempPrb = tranShkPrbs[0]

        Ths.WorstIncPrb = np.sum(  # All cases where permShk and tranShk are Min
            ShkPrbs[ \
                permShkValsBcst * tranShkValsBcst == permShkMin * tranShkMin
            ]
        )
        Ths.WorstIncVal = permShkMin * tranShkMin
        # New vars not used in PF model

        Ths.aXtraGrid = aXtraGrid
        Ths.vFuncBool = vFuncBool
        Ths.CubicBool = CubicBool

    # self here is the solver, which knows info about the problem from the agent
    def add_fcts_to_soln(self, futr_Ths):
        """
        For versions with uncertainty in transitory and/or permanent shocks,
        adds to the solution a set of results useful for calculating
        and various diagnostic conditions about the problem, and stable
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
        soln_crnt = self.soln_crnt
        Ths = soln_crnt.Ths

        urlroot = Ths.urlroot

        self.add_info_useful_for_further_analysis_ConsPerfForesightSolver(futr_Ths)

        # Modify formulae also present in PF model but that must change

        py___code = '1/(1 + (WorstIncPrb**(1/CRRA))*(RPF/MPCmax_tp1))'
        if soln_crnt.stge_kind['iter_status'] == 'terminal':  # kludge needed
            Ths.MPCmax_tp1 = float('inf')  # Makes MPCmax come to 1.0
        if soln_crnt.stge_kind['iter_status'] == 'terminal':
            Ths.MPCmax = MPCmax = eval(py___code, globals(), Ths.__dict__)

        # Many other _fcts will have been inherited from the perfect foresight
        # model of which this model is a descendant
        # Here we need compute only those objects whose value changes 
        # or does not exist when
        # the shock distributions are nondegenerate.
        Ex_IncNrmNxt_fcts = {
            'about': 'Expected income next period'
        }
        py___code = '𝔼_dot(ShkPrbs,tranShkValsBcst * permShkValsBcst)'
        Ths.Ex_IncNrmNxt = Ex_IncNrmNxt = \
            eval(py___code, globals(), Ths.__dict__)  # globals enables 𝔼_dot
        Ex_IncNrmNxt_fcts.update({'latexexpr': r'\Ex_IncNrmNxt'})
        Ex_IncNrmNxt_fcts.update({'_unicode_': r'R/Γ'})
        Ex_IncNrmNxt_fcts.update({'urlhandle': urlroot+'ExIncNrmNxt'})
        Ex_IncNrmNxt_fcts.update({'py___code': py___code})
        Ex_IncNrmNxt_fcts.update({'value_now': Ex_IncNrmNxt})
        soln_crnt.fcts.update({'Ex_IncNrmNxt': Ex_IncNrmNxt_fcts})
        soln_crnt.Ths.Ex_IncNrmNxt_fcts = Ex_IncNrmNxt_fcts
 
        Ex_Inv_permShk_fcts = {
            'about': 'Expected Inverse of Permanent Shock'
        }
        py___code = '𝔼_dot(1/permShkVals, permShkPrbs)'
        Ths.Ex_Inv_permShk = Ex_Inv_permShk = \
            eval(py___code, globals(), Ths.__dict__)
        Ex_Inv_permShk_fcts.update({'latexexpr': r'\ExInvpermShk'})
#        Ex_Inv_permShk_fcts.update({'_unicode_': r'R/Γ'})
        Ex_Inv_permShk_fcts.update({'urlhandle': urlroot+'ExInvpermShk'})
        Ex_Inv_permShk_fcts.update({'py___code': py___code})
        Ex_Inv_permShk_fcts.update({'value_now': Ex_Inv_permShk})
        soln_crnt.fcts.update({'Ex_Inv_permShk': Ex_Inv_permShk_fcts})
        soln_crnt.Ths.Ex_Inv_permShk_fcts = Ex_Inv_permShk_fcts

        Inv_Ex_Inv_permShk_fcts = {
            'about': 'Inverse of Expected Inverse of Permanent Shock'
        }
        py___code = '1/Ex_Inv_permShk'
        Ths.Inv_Ex_Inv_permShk = Inv_Ex_Inv_permShk = \
            eval(py___code, globals(), Ths.__dict__)
        Inv_Ex_Inv_permShk_fcts.update(
            {'latexexpr': '\left(\Ex[\permShk^{-1}]\right)^{-1}'})
        Inv_Ex_Inv_permShk_fcts.update({'_unicode_': r'1/𝔼[Γψ]'})
        Inv_Ex_Inv_permShk_fcts.update({'urlhandle': urlroot+'InvExInvpermShk'})
        Inv_Ex_Inv_permShk_fcts.update({'py___code': py___code})
        Inv_Ex_Inv_permShk_fcts.update({'value_now': Inv_Ex_Inv_permShk})
        soln_crnt.fcts.update({'Inv_Ex_Inv_permShk': Inv_Ex_Inv_permShk_fcts})
        soln_crnt.Ths.Inv_Ex_Inv_permShk_fcts = Inv_Ex_Inv_permShk_fcts
        # soln_crnt.Inv_Ex_Inv_permShk = Inv_Ex_Inv_permShk

        Ex_RNrm_fcts = {
            'about': 'Expected Stochastic-Growth-Normalized Return'
        }
        py___code = 'PF_RNrm * Ex_Inv_permShk'
        Ths.Ex_RNrm = Ex_RNrm = \
            eval(py___code, globals(), Ths.__dict__)
        Ex_RNrm_fcts.update({'latexexpr': r'\ExRNrm'})
        Ex_RNrm_fcts.update({'_unicode_': r'𝔼[R/Γψ]'})
        Ex_RNrm_fcts.update({'urlhandle': urlroot+'ExRNrm'})
        Ex_RNrm_fcts.update({'py___code': py___code})
        Ex_RNrm_fcts.update({'value_now': Ex_RNrm})
        soln_crnt.fcts.update({'Ex_RNrm': Ex_RNrm_fcts})
        soln_crnt.Ths.Ex_RNrm_fcts = Ex_RNrm_fcts
        soln_crnt.Ex_RNrm = Ex_RNrm

        Inv_Ex_RNrm_fcts = {
            'about': 'Inverse of Expected Stochastic-Growth-Normalized Return'
        }
        py___code = '1/Ex_RNrm'
        Ths.Inv_Ex_RNrm = Inv_Ex_RNrm = \
            eval(py___code, globals(), Ths.__dict__)
        Inv_Ex_RNrm_fcts.update(
            {'latexexpr': '\InvExInvRNrm=\left(\Ex[\permShk^{-1}]\right)^{-1}'})
        Inv_Ex_RNrm_fcts.update({'_unicode_': r'1/𝔼[R/(Γψ)]'})
        Inv_Ex_RNrm_fcts.update({'urlhandle': urlroot+'InvExRNrm'})
        Inv_Ex_RNrm_fcts.update({'py___code': py___code})
        Inv_Ex_RNrm_fcts.update({'value_now': Inv_Ex_RNrm})
        soln_crnt.fcts.update({'Inv_Ex_RNrm': Inv_Ex_RNrm_fcts})
        soln_crnt.Ths.Inv_Ex_RNrm_fcts = Inv_Ex_RNrm_fcts
        soln_crnt.Inv_Ex_RNrm = Inv_Ex_RNrm

        Ex_uInv_permShk_fcts = {
            'about': 'Expected Utility for Consuming Permanent Shock'
        }
        py___code = '𝔼_dot(permShkValsBcst**(1-CRRA), ShkPrbs)'
        Ths.Ex_uInv_permShk = Ex_uInv_permShk = \
            eval(py___code, globals(), Ths.__dict__)
        Ex_uInv_permShk_fcts.update({'latexexpr': r'\ExuInvpermShk'})
        Ex_uInv_permShk_fcts.update({'urlhandle': r'ExuInvpermShk'})
        Ex_uInv_permShk_fcts.update({'py___code': py___code})
        Ex_uInv_permShk_fcts.update({'value_now': Ex_uInv_permShk})
        soln_crnt.fcts.update({'Ex_uInv_permShk': Ex_uInv_permShk_fcts})
        soln_crnt.Ths.Ex_uInv_permShk_fcts = Ex_uInv_permShk_fcts
        soln_crnt.Ex_uInv_permShk = Ex_uInv_permShk

        py___code = '1/Ex_uInv_permShk'
        uInv_Ex_uInv_permShk_fcts = {
            'about': 'Inverted Expected Utility for Consuming Permanent Shock'
        }
        Ths.uInv_Ex_uInv_permShk = uInv_Ex_uInv_permShk = \
            eval(py___code, globals(), Ths.__dict__)
        uInv_Ex_uInv_permShk_fcts.update({'latexexpr': r'\uInvExuInvpermShk'})
        uInv_Ex_uInv_permShk_fcts.update({'urlhandle': urlroot+'uInvExuInvpermShk'})
        uInv_Ex_uInv_permShk_fcts.update({'py___code': py___code})
        uInv_Ex_uInv_permShk_fcts.update({'value_now': uInv_Ex_uInv_permShk})
        soln_crnt.fcts.update({'uInv_Ex_uInv_permShk': uInv_Ex_uInv_permShk_fcts})
        soln_crnt.Ths.uInv_Ex_uInv_permShk_fcts = uInv_Ex_uInv_permShk_fcts
        soln_crnt.uInv_Ex_uInv_permShk = uInv_Ex_uInv_permShk

        PermGroFacAdj_fcts = {
            'about': 'Uncertainty-Adjusted Permanent Income Growth Factor'
        }
        py___code = 'PermGroFac * Inv_Ex_Inv_permShk'
        Ths.PermGroFacAdj = PermGroFacAdj = \
            eval(py___code, globals(), Ths.__dict__)
        PermGroFacAdj_fcts.update({'latexexpr': r'\PermGroFacAdj'})
        PermGroFacAdj_fcts.update({'urlhandle': urlroot+'PermGroFacAdj'})
        PermGroFacAdj_fcts.update({'value_now': PermGroFacAdj})
        soln_crnt.fcts.update({'PermGroFacAdj': PermGroFacAdj_fcts})
        soln_crnt.Ths.PermGroFacAdj_fcts = PermGroFacAdj_fcts
        soln_crnt.PermGroFacAdj = PermGroFacAdj

        GPFNrm_fcts = {
            'about': 'Normalized Expected Growth Patience Factor'
        }
        py___code = 'GPFRaw * Ex_Inv_permShk'
        Ths.GPFNrm = GPFNrm = \
            eval(py___code, globals(), Ths.__dict__)
        GPFNrm_fcts.update({'latexexpr': r'\GPFNrm'})
        GPFNrm_fcts.update({'_unicode_': r'Þ_Γ'})
        GPFNrm_fcts.update({'urlhandle': urlroot+'GPFNrm'})
        GPFNrm_fcts.update({'py___code': py___code})
        soln_crnt.fcts.update({'GPFNrm': GPFNrm_fcts})
        soln_crnt.Ths.GPFNrm_fcts = GPFNrm_fcts
        soln_crnt.GPFNrm = GPFNrm

        GICNrm_fcts = {
            'about': 'Stochastic Growth Normalized Impatience Condition'
        }
        GICNrm_fcts.update({'latexexpr': r'\GICNrm'})
        GICNrm_fcts.update({'urlhandle': urlroot+'GICNrm'})
        GICNrm_fcts.update({'py___code': 'test: GPFNrm < 1'})
        soln_crnt.fcts.update({'GICNrm': GICNrm_fcts})
        soln_crnt.Ths.GICNrm_fcts = GICNrm_fcts

        FVAC_fcts = {
            'about': 'Finite Value of Autarky Condition'
        }
        FVAC_fcts.update({'latexexpr': r'\FVAC'})
        FVAC_fcts.update({'urlhandle': urlroot+'FVAC'})
        FVAC_fcts.update({'py___code': 'test: FVAF < 1'})
        soln_crnt.fcts.update({'FVAC': FVAC_fcts})
        soln_crnt.Ths.FVAC_fcts = FVAC_fcts

        WRPF_fcts = {
            'about': 'Weak Return Patience Factor'
        }
        py___code = '(UnempPrb ** (1 / CRRA)) * RPF'
        Ths.WRPF = WRPF = \
            eval(py___code, globals(), Ths.__dict__)
        WRPF_fcts.update({'latexexpr': r'\WRPF'})
        WRPF_fcts.update({'_unicode_': r'℘^(1/\rho) RPF'})
        WRPF_fcts.update({'urlhandle': urlroot+'WRPF'})
        WRPF_fcts.update({'value_now': WRPF})
        WRPF_fcts.update({'py___code': py___code})
        soln_crnt.fcts.update({'WRPF': WRPF_fcts})
        soln_crnt.Ths.WRPF_fcts = WRPF_fcts

        WRIC_fcts = {
            'about': 'Weak Return Impatience Condition'
        }
        WRIC_fcts.update({'latexexpr': r'\WRIC'})
        WRIC_fcts.update({'urlhandle': urlroot+'WRIC'})
        WRIC_fcts.update({'py___code': 'test: WRPF < 1'})
        soln_crnt.fcts.update({'WRIC': WRIC_fcts})
        soln_crnt.Ths.WRIC_fcts = WRIC_fcts

        DiscGPFNrmCusp_fcts = {
            'about': 'DiscFac s.t. GPFNrm = 1'
        }
        py___code = '((PermGroFac*Inv_Ex_Inv_permShk)**(CRRA))/Rfree'
        Ths.DiscGPFNrmCusp = DiscGPFNrmCusp = \
            eval(py___code, globals(), Ths.__dict__)
        DiscGPFNrmCusp_fcts.update({'latexexpr': ''})
        DiscGPFNrmCusp_fcts.update({'value_now': DiscGPFNrmCusp})
        DiscGPFNrmCusp_fcts.update({'py___code': py___code})
        soln_crnt.fcts.update({'DiscGPFNrmCusp': DiscGPFNrmCusp_fcts})
        soln_crnt.Ths.DiscGPFNrmCusp_fcts = DiscGPFNrmCusp_fcts
        soln_crnt.DiscGPFNrmCusp = DiscGPFNrmCusp

        if soln_crnt.Inv_PF_RNrm < 1:        # Finite if Rfree > soln_crnt.Ths.PermGroFac
            soln_crnt.hNrmInf = 1/(1-soln_crnt.Inv_PF_RNrm)

        # Given m, value of c where 𝔼[m_{t+1}]=m_{t}
        # url/#
        soln_crnt.c_where_Ex_mtp1_minus_mt_eq_0 = (
            lambda m_t:
            m_t * (1 - soln_crnt.Inv_Ex_RNrm) + (soln_crnt.Inv_Ex_RNrm)
        )

        # Given m, value of c where 𝔼[mLev_{t+1}/mLev_{t}]=soln_crnt.Ths.PermGroFac
        # Solves for c in equation at url/#balgrostable

        soln_crnt.c_where_Ex_permShk_times_mtp1_minus_mt_eq_0 = (
            lambda m_t:
            m_t * (1 - soln_crnt.Inv_PF_RNrm) + soln_crnt.Inv_PF_RNrm
        )

        # 𝔼[c_{t+1} pLev_{t+1}/pLev_{t}] as a fn of a_{t}
        soln_crnt.Ex_cLev_tp1_Over_pLev_t_from_at = (
            lambda a_t:
            𝔼_dot(soln_crnt.Ths.PermGroFac *
                  soln_crnt.Ths.permShkValsBcst *
                  soln_crnt.cFunc(
                      (soln_crnt.PF_RNrm/soln_crnt.Ths.permShkValsBcst) * a_t
                      + soln_crnt.Ths.tranShkValsBcst
                  ),
                  soln_crnt.ShkPrbs)
        )

        soln_crnt.c_where_Ex_mtp1_minus_mt_eq_0 = c_where_Ex_mtp1_minus_mt_eq_0 = (
            lambda m_t:
            m_t * (1 - 1/soln_crnt.Ex_RNrm) + (1/soln_crnt.Ex_RNrm)
        )

        # Solve the equation at url/#balgrostable
        soln_crnt.c_where_Ex_permShk_times_mtp1_minus_mt_eq_0 = \
            c_where_Ex_permShk_times_mtp1_minus_mt_eq_0 = (
                lambda m_t:
                (m_t * (1 - 1/soln_crnt.PF_RNrm)) + (1/soln_crnt.PF_RNrm)
            )

        # mNrmTrg solves Ex_RNrm*(m - c(m)) + 𝔼[inc_next] - m = 0
        Ex_m_tp1_minus_m_t = (
            lambda m_t:
            soln_crnt.Ex_RNrm * (m_t - soln_crnt.cFunc(m_t)) +
            soln_crnt.Ex_IncNrmNxt - m_t
        )
        soln_crnt.Ex_m_tp1_minus_m_t = Ex_m_tp1_minus_m_t

        soln_crnt.Ex_cLev_tp1_Over_pLev_t_from_at = Ex_cLev_tp1_Over_pLev_t_from_at = (
            lambda a_t:
            𝔼_dot(
                soln_crnt.Ths.permShkValsBcst * soln_crnt.Ths.PermGroFac * soln_crnt.cFunc(
                    (soln_crnt.PF_RNrm/soln_crnt.Ths.permShkValsBcst) *
                    a_t + soln_crnt.Ths.tranShkValsBcst
                ),
                soln_crnt.ShkPrbs)
        )

        soln_crnt.Ex_permShk_tp1_times_m_tp1_minus_m_t = \
            Ex_permShk_tp1_times_m_tp1_minus_m_t = (
                lambda m_t: self.soln_crnt.PF_RNrm *
                (m_t - soln_crnt.cFunc(m_t)) + 1.0 - m_t
            )

        for key in soln_crnt.fcts:
            setattr(Ths, key+'_fcts', soln_crnt.fcts[key])

        return soln_crnt

    # self here is the solver, which knows info about the problem from the agent

    def prepare_to_solve(self):  # stage
        """
        Prepare the current stage for processing by the one-stage solver.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        soln_crnt = self.soln_crnt
        Ths = self.Ths = soln_crnt.Ths
        
        # Now that solution exists, put condition-checking tools there 
        soln_crnt.check_conditions = self.check_conditions
        soln_crnt.check_AIC = self.check_AIC
        soln_crnt.check_RIC = self.check_RIC
        soln_crnt.check_FVAC = self.check_FVAC
        soln_crnt.check_GICLiv = self.check_GICLiv
        soln_crnt.check_GICRaw = self.check_GICRaw
        soln_crnt.check_GICNrm = self.check_GICNrm
        soln_crnt.check_FHWC = self.check_FHWC
        soln_crnt.check_WRIC = self.check_WRIC

        # Define a few variables that permit the same formulae to be used for
        # versions with and without uncertainty
        # We are in the perfect foresight model now so these are all 1.0

        self.def_utility_funcs(Ths)

        Ths.PerfFsgt = (type(self) == ConsPerfForesightSolver)

        # If no uncertainty, return the degenerate targets for the PF model
        if hasattr(Ths, "tranShkVals"):  # Then it has transitory shocks
            # Handle the degenerate case where shocks are of size zero
            if ((Ths.tranShkMin == 1.0) and (Ths.permShkMin == 1.0)):
                # But they still might have unemployment risk
                if hasattr(Ths, "UnempPrb"):
                    if ((Ths.UnempPrb == 0.0) or (Ths.IncUnemp == 1.0)):
                        Ths.PerfFsgt = True  # No unemployment risk either
                    else:
                        Ths.PerfFsgt = False  # The only kind of uncertainty is unemployment

        if Ths.PerfFsgt:
            Ths.Ex_Inv_permShk = 1.0
            Ths.Ex_uInv_permShk = 1.0

        return


####################################################################################################
####################################################################################################


class ConsIndShockSolverBasic(ConsIndShockSetup):
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
            A 1D array of end-of-period assets; also stored as attribute of self.soln_crnt.Ths.
        """

        # We define aNrmGrid all the way from BoroCnstNat up to max(self.aXtraGrid)
        # even if BoroCnstNat < BoroCnstArt, so we can construct the consumption
        # function as the lower envelope of the (by the artificial borrowing con-
        # straint) unconstrained consumption function, and the artificially con-
        # strained consumption function.
        self.soln_crnt.Ths.aNrmGrid = np.asarray(
            self.soln_crnt.Ths.aXtraGrid) + self.soln_crnt.Ths.BoroCnstNat

        return self.soln_crnt.Ths.aNrmGrid

    def calc_EndOfPrdvP(self):
        """
        Calculate end-of-period marginal value of assets at each point in aNrm.
        Does so by taking a weighted sum of next period marginal values across
        income shocks (in a preconstructed grid self.soln_crnt.Ths.mNrmNext).

        Parameters
        ----------
        none

        Returns
        -------
        EndOfPrdvP : np.array
            A 1D array of end-of-period marginal value of assets
        """

        def vp_next(shocks, a_Nrm_Val):
            return shocks[0] ** (-self.soln_crnt.Ths.CRRA) \
                * self.soln_crnt.Ths.vPfunc_tp1(self.m_Nrm_tp1(shocks, a_Nrm_Val))

        EndOfPrdvP = (
            self.soln_crnt.Ths.DiscFac * self.soln_crnt.Ths.LivPrb
            * self.soln_crnt.Ths.Rfree
            * self.soln_crnt.Ths.PermGroFac ** (-self.soln_crnt.Ths.CRRA)
            * calc_expectation(
                self.soln_crnt.Ths.IncShkDstn,
                vp_next,
                self.soln_crnt.Ths.aNrmGrid
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
        cNrm = self.soln_crnt.Ths.uPinv(EndOfPrdvP)
        mNrm = cNrm + aNrm

        # Limiting consumption is zero as m approaches mNrmMin
        c_for_interpolation = np.insert(cNrm, 0, 0.0, axis=-1)
        m_for_interpolation = np.insert(mNrm, 0, self.soln_crnt.Ths.BoroCnstNat, axis=-1)

        # Store these for calcvFunc
        self.soln_crnt.Ths.cNrm = cNrm
        self.soln_crnt.Ths.mNrm = mNrm

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
        # Use the given interpolator to construct the consumption function
        cFuncUnc = interpolator(mNrm, cNrm)

        # Combine the constrained and unconstrained functions into the true consumption function
        # by choosing the lower of the constrained and unconstrained functions
        # LowerEnvelope should only be used when BoroCnstArt is true
        if self.soln_crnt.Ths.BoroCnstArt is None:
            cFunc = cFuncUnc
        else:
            self.soln_crnt.Ths.cFuncCnst = LinearInterp(
                np.array([self.soln_crnt.Ths.mNrmMin, self.soln_crnt.Ths.mNrmMin + 1]
                         ), np.array([0.0, 1.0]))
            cFunc = LowerEnvelope(cFuncUnc, self.soln_crnt.Ths.cFuncCnst, nan_bool=False)

        # Make the marginal value function and the marginal marginal value function
        vPfunc = MargValueFuncCRRA(cFunc, self.soln_crnt.Ths.CRRA)

        # Pack up the solution and return it
        solution_interpolating = ConsumerSolution(
            cFunc=cFunc,
            vPfunc=vPfunc,
            mNrmMin=self.soln_crnt.Ths.mNrmMin
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
        sol_EGM = \
            self.use_points_for_interpolation(cNrm, mNrm, interpolator)

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
        self.soln_crnt.Ths.aNrmGrid = self.prepare_to_calc_EndOfPrdvP()
        self.soln_crnt.Ths.EndOfPrdvP = self.calc_EndOfPrdvP()

        # Construct a solution for this period
        if self.soln_crnt.Ths.CubicBool:
            soln_crnt = self.interpolating_EGM_solution(
                self.soln_crnt.Ths.EndOfPrdvP, self.soln_crnt.Ths.aNrmGrid, interpolator=self.make_cubic_cFunc
            )
        else:
            soln_crnt = self.interpolating_EGM_solution(
                self.soln_crnt.Ths.EndOfPrdvP, self.soln_crnt.Ths.aNrmGrid, interpolator=self.make_linear_cFunc
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
            mNrm, cNrm, self.soln_crnt.Ths.cFuncLimitIntercept, self.soln_crnt.Ths.cFuncLimitSlope
        )
        return cFunc_unconstrained

    # ConsIndShockSolverBasic
    def solve(self):  # make self.soln_crnt from self.soln_futr
        """
        Solves (one period/stage of) the single period consumption-saving problem with
        method of endogenous gridpoints.  Solution includes a consumption function
        cFunc (using cubic or linear splines), a marginal value function vPfunc, a min-
        imum acceptable level of normalized market resources mNrmMin, normalized
        human wealth hNrm, and bounding MPCs MPCmin and MPCmax.  It might also
        have a value function vFunc and marginal marginal value function vPPfunc.

        Parameters
        ----------
        none

        Returns
        -------
        solution : ConsumerSolution
            The solution to this period/stage's problem.
        """
        futr = self.soln_futr
        soln_crnt = self.soln_crnt

        if futr.stge_kind['iter_status'] == 'finished':
            soln_crnt.stge_kind['iter_status'] = 'finished'
            _log.error("The model has already been solved.  Aborting.")
            return soln_crnt

        # If this is the first invocation of solve, just flesh out the terminal
        # period solution so it is a proper starting point for iteration
        if futr.stge_kind['iter_status'] == 'terminal':
            self.soln_crnt.Ths_inputs = deepcopy(soln_crnt.Ths)
            self.add_fcts_to_soln(futr.Ths)
            soln_crnt.stge_kind['iter_status'] = 'iterator'
            return soln_crnt  # Replaces original "terminal" solution; next soln_futr

        self.soln_crnt.stge_kind = {'iter_status': 'iterator',
                                    'slvr_type': 'ConsIndShockSolver'}
        # Add a bunch of 
        self.add_fcts_to_soln(futr.Ths)

        sol_EGM = self.make_sol_using_EGM()  # Need to add test for finished, change stge_kind if so
        soln_crnt.Ths.cFunc = soln_crnt.cFunc = sol_EGM.cFunc
        soln_crnt.Ths.vPfunc = soln_crnt.vPfunc = sol_EGM.vPfunc

        # Add the value function if requested, as well as the marginal marginal
        # value function if cubic splines were used for interpolation
        if soln_crnt.Ths.vFuncBool:
            soln_crnt.Ths.vFunc = self.vFunc = self.add_vFunc(soln_crnt, self.EndOfPrdvP)
        if soln_crnt.Ths.CubicBool:
            soln_crnt.Ths.vPPfunc = self.add_vPPfunc(soln_crnt)

        return soln_crnt

    def m_Nrm_tp1(self, shocks, a_Nrm_Val):
        """
        Computes normalized market resources of the next period
        from income shocks and current normalized market resources.

        Parameters
        ----------
        shocks: [float]
            Permanent and transitory income shock levels.

        a_Nrm_Val: float
            Normalized market assets this period

        Returns
        -------
        float
           normalized market resources in the next period
        """
        return self.soln_crnt.Ths.Rfree / (self.soln_crnt.Ths.PermGroFac * shocks[0]) \
            * a_Nrm_Val + shocks[1]


###############################################################################
###############################################################################


class ConsIndShockSolver(ConsIndShockSolverBasic):
    """
    This class solves a single period of a standard consumption-saving problem.
    It inherits from ConsIndShockSolverBasic, and adds the ability to perform cubic
    interpolation and to calculate the value function.
    """

    def make_cubic_cFunc(self, mNrm_Vec, cNrm_Vec):
        """
        Makes a cubic spline interpolation of the unconstrained consumption
        function for this period.

        Requires self.soln_crnt.Ths.aNrm to have been computed already.

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
        def vPP_next(shocks, a_Nrm_Val):
            return shocks[0] ** (- self.soln_crnt.Ths.CRRA - 1.0) \
                * self.soln_crnt.Ths.vPPfunc_tp1(self.m_Nrm_tp1(shocks, a_Nrm_Val))

        EndOfPrdvPP = (
            self.soln_crnt.Ths.DiscFac * self.soln_crnt.Ths.LivPrb
            * self.soln_crnt.Ths.Rfree
            * self.soln_crnt.Ths.Rfree
            * self.soln_crnt.Ths.PermGroFac ** (-self.soln_crnt.Ths.CRRA - 1.0)
            * calc_expectation(
                self.soln_crnt.Ths.IncShkDstn,
                vPP_next,
                self.soln_crnt.Ths.aNrmGrid
            )
        )
        dcda = EndOfPrdvPP / self.soln_crnt.Ths.uPP(np.array(cNrm_Vec[1:]))
        MPC = dcda / (dcda + 1.0)
        MPC = np.insert(MPC, 0, self.soln_crnt.Ths.MPCmax)

        cFuncUnc = CubicInterp(
            mNrm_Vec, cNrm_Vec, MPC, self.soln_crnt.Ths.MPCmin *
            self.soln_crnt.Ths.hNrm, self.soln_crnt.Ths.MPCmin
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
        def v_Lvl_next(shocks, a_Nrm_Val):
            return (
                shocks[0] ** (1.0 - self.soln_crnt.Ths.CRRA)
                * self.stg_cur.Ths.PermGroFac ** (1.0 - self.soln_crnt.Ths.CRRA)
            ) * self.soln_crnt.Ths.vFuncNxt(self.soln_crnt.m_Nrm_tp1(shocks, a_Nrm_Val))
        EndOfPrdv = self.stg_cur.Ths.DiscLiv * calc_expectation(
            self.stg_cur.Ths.IncShkDstn, v_Lvl_next, self.soln_crnt.aNrm
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
            EndOfPrdvNvrsFunc, self.soln_crnt.Ths.CRRA)

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
        mNrm_temp = self.soln_crnt.mNrmMin + self.soln_crnt.aXtraGrid
        cNrm = soln_crnt.cFunc(mNrm_temp)
        aNrm = mNrm_temp - cNrm
        vNrm = self.soln_crnt.u(cNrm) + self.EndOfPrdvFunc(aNrm)
        vPnow = self.uP(cNrm)

        # Construct the beginning value function
        vNvrs = self.soln_crnt.uinv(vNrm)  # value transformed through inverse utility
        vNvrsP = vPnow * self.soln_crnt.uinvP(vNrm)
        mNrm_temp = np.insert(mNrm_temp, 0, self.soln_crnt.mNrmMin)
        vNvrs = np.insert(vNvrs, 0, 0.0)
        vNvrsP = np.insert(
            vNvrsP, 0, self.soln_crnt.MPCmaxEff ** (-self.soln_crnt.Ths.CRRA /
                                                    (1.0 - self.soln_crnt.Ths.CRRA))
        )
        MPCminNvrs = self.soln_crnt.MPCmin ** (-self.soln_crnt.Ths.CRRA /
                                               (1.0 - self.soln_crnt.Ths.CRRA))
        vNvrsFunc = CubicInterp(
            mNrm_temp, vNvrs, vNvrsP, MPCminNvrs * self.soln_crnt.hNrm, MPCminNvrs
        )
        vFunc = ValueFuncCRRA(vNvrsFunc, self.soln_crnt.Ths.CRRA)
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
        self.vPPfunc = MargMargValueFuncCRRA(soln_crnt.Ths.cFunc, soln_crnt.Ths.CRRA)
        soln_crnt.vPPfunc = self.vPPfunc
        return soln_crnt.vPPfunc


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
p        Expected permanent income growth factor at the end of this period.
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
        self.Ths.Rboro = self.Rboro = Rboro
        self.Ths.Rsave = self.Rsave = Rsave
        self.Ths.cnstrct = {'vFuncBool', 'IncShkDstn'}

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
            self.Ths.Rboro > self.Ths.Rsave
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
        ShkCount = self.Ths.tranShkVals.size
        aNrm_temp = np.tile(aNrm, (ShkCount, 1))
        permShkVals_temp = (np.tile(self.Ths.permShkVals, (aXtraCount, 1))).transpose()
        tranShkVals_temp = (np.tile(self.Ths.tranShkVals, (aXtraCount, 1))).transpose()
        ShkPrbs_temp = (np.tile(self.ShkPrbs, (aXtraCount, 1))).transpose()

        # Make a 1D array of the interest factor at each asset gridpoint
        Rfree_vec = self.Ths.Rsave * np.ones(aXtraCount)
        if KinkBool:
            self.i_kink = (
                np.sum(aNrm <= 0) - 1
            )  # Save the index of the kink point as an attribute
            Rfree_vec[0: self.i_kink] = self.Ths.Rboro
            Rfree = Rfree_vec
            Rfree_temp = np.tile(Rfree_vec, (ShkCount, 1))

        # Make an array of market resources that we could have next period,
        # considering the grid of assets and the income shocks that could occur
        mNrmNext = (
            Rfree_temp / (self.PermGroFac * permShkVals_temp) * aNrm_temp
            + Ths.tranShkVals_temp
        )

        # Recalculate the minimum MPC and human wealth using the interest factor on saving.
        # This overwrites values from set_and_update_values, which were based on Rboro instead.
        if KinkBool:
            RPFTop = (
                (self.Ths.Rsave * self.DiscLiv) ** (1.0 / self.CRRA)
            ) / self.Ths.Rsave
            self.MPCmin = 1.0 / (1.0 + RPFTop / self.stg_cur.Ths.MPCmin)
            self.hNrm = (
                self.PermGroFac
                / self.Ths.Rsave
                * (
                    𝔼_dot(
                        self.ShkPrbs, self.Ths.tranShkVals * self.Ths.permShkVals
                    )
                    + self.stg_cur.Ths.hNrm
                )
            )

        # Store some of the constructed arrays for later use and return the assets grid
        self.permShkVals_temp = permShkVals_temp
        self.ShkPrbs_temp = ShkPrbs_temp
        self.mNrmNext = mNrmNext
        self.aNrm = aNrm
        return aNrm


# ============================================================================
# == Classes for representing types of consumer agents (and things they do) ==
# ============================================================================

# Make a dictionary to specify a perfect foresight consumer type
init_perfect_foresight = {
    'CRRA': 2.0,          # Coefficient of relative risk aversion,
    'Rfree': 1.03,        # Interest factor on assets
    'DiscFac': 0.96,      # Intertemporal discount factor
    'LivPrb': [0.98],     # Survival probability
    'PermGroFac': [1.01],  # Permanent income growth factor
    'BoroCnstArt': None,  # Artificial borrowing constraint
    'T_cycle': 1,         # Num of periods in a finite horizon cycle (like, a life cycle)
    'PermGroFacAgg': 1.0,  # Aggregate income growth factor (multiplies individual)
    'MaxKinks': 400,      # Maximum number of grid points to allow in cFunc (should be large)
    'mcrlo_AgentCount': 10000,  # Number of agents of this type (only matters for simulation)
    'aNrmInitMean': 0.0,  # Mean of log initial assets (only matters for simulation)
    'aNrmInitStd': 1.0,  # Standard deviation of log initial assets (only for simulation)
    'mcrlo_pLvlInitMean': 0.0,  # Mean of log initial permanent income (only matters for simulation)
    # Standard deviation of log initial permanent income (only matters for simulation)
    'mcrlo_pLvlInitStd': 0.0,
    # Aggregate permanent income growth factor: portion of PermGroFac attributable to aggregate productivity growth (only matters for simulation)
    'T_age': None,       # Age after which simulated agents are automatically killed
    # Optional extra _fcts about the model and its calibration
}

init_perfect_foresight.update(dict({'_fcts': {'import': 'init_perfect_foresight'}}))

# The info below is optional at present but may become mandatory as the toolkit evolves
# 'Primitives' define the 'true' model that we think of ourselves as trying to solve
# (the limit as approximation error reaches zero)
init_perfect_foresight.update(
    {'prmtv_par': ['CRRA', 'Rfree', 'DiscFac', 'LivPrb', 'PermGroFac', 'BoroCnstArt', 'PermGroFacAgg', 'T_cycle']})
# Approximation parameters define the precision of the approximation
# Limiting values for approximation parameters: values such that, as all such parameters approach their limits,
# the approximation gets arbitrarily close to the 'true' model
init_perfect_foresight.update(  # In principle, kinks exist all the way to infinity
    {'aprox_lim': {'MaxKinks': 'infinity'}})
# The simulation stge of the problem requires additional parameterization
init_perfect_foresight.update(  # The 'primitives' for the simulation
    {'prmtv_sim': ['aNrmInitMean', 'aNrmInitStd', 'mcrlo_pLvlInitMean', 'mcrlo_pLvlInitStd']})
init_perfect_foresight.update({  # Approximation parameters for monte carlo sims
    'mcrlo_sim': ['mcrlo_AgentCount', 'T_age']
})
init_perfect_foresight.update({  # Limiting values that define 'true' simulation
    'mcrlo_lim': {
        'mcrlo_AgentCount': 'infinity',
        'T_age': 'infinity'
    }
})

# Optional more detailed _fcts about various parameters
CRRA_fcts = {
    'about': 'Coefficient of Relative Risk Aversion'}
CRRA_fcts.update({'latexexpr': '\providecommand{\CRRA}{\rho}\CRRA'})
CRRA_fcts.update({'_unicode_': 'ρ'})  # \rho is Greek r: relative risk aversion rrr
CRRA_fcts.update({'prmtv_par': 'True'})
init_perfect_foresight['prmtv_par'].append('CRRA')
init_perfect_foresight['_fcts'].update({'CRRA': CRRA_fcts})
init_perfect_foresight.update({'CRRA_fcts': CRRA_fcts})

DiscFac_fcts = {
    'about': 'Pure time preference rate'}
DiscFac_fcts.update({'latexexpr': '\providecommand{\DiscFac}{\beta}\DiscFac'})
DiscFac_fcts.update({'_unicode_': 'β'})
DiscFac_fcts.update({'prmtv_par': 'True'})
init_perfect_foresight['prmtv_par'].append('DiscFac')
init_perfect_foresight['_fcts'].update({'DiscFac': DiscFac_fcts})
init_perfect_foresight.update({'DiscFac_fcts': DiscFac_fcts})

LivPrb_fcts = {
    'about': 'Probability of survival from this period to next'}
LivPrb_fcts.update({'latexexpr': '\providecommand{\LivPrb}{\Pi}\LivPrb'})
LivPrb_fcts.update({'_unicode_': 'Π'})  # \Pi mnemonic: 'Probability of surival'
LivPrb_fcts.update({'prmtv_par': 'True'})
init_perfect_foresight['prmtv_par'].append('LivPrb')
init_perfect_foresight['_fcts'].update({'LivPrb': LivPrb_fcts})
init_perfect_foresight.update({'LivPrb_fcts': LivPrb_fcts})

Rfree_fcts = {
    'about': 'Risk free interest factor'}
Rfree_fcts.update({'latexexpr': '\providecommand{\Rfree}{\mathsf{R}}\Rfree'})
Rfree_fcts.update({'_unicode_': 'R'})
Rfree_fcts.update({'prmtv_par': 'True'})
init_perfect_foresight['prmtv_par'].append('Rfree')
init_perfect_foresight['_fcts'].update({'Rfree': Rfree_fcts})
init_perfect_foresight.update({'Rfree_fcts': Rfree_fcts})

PermGroFac_fcts = {
    'about': 'Growth factor for permanent income'}
PermGroFac_fcts.update({'latexexpr': '\providecommand{\PermGroFac}{\Gamma}\PermGroFac'})
PermGroFac_fcts.update({'_unicode_': 'Γ'})  # \Gamma is Greek G for Growth
PermGroFac_fcts.update({'prmtv_par': 'True'})
init_perfect_foresight['prmtv_par'].append('PermGroFac')
init_perfect_foresight['_fcts'].update({'PermGroFac': PermGroFac_fcts})
init_perfect_foresight.update({'PermGroFac_fcts': PermGroFac_fcts})

PermGroFacAgg_fcts = {
    'about': 'Growth factor for aggregate permanent income'}
# PermGroFacAgg_fcts.update({'latexexpr': '\providecommand{\PermGroFacAgg}{\Gamma}\PermGroFacAgg'})
# PermGroFacAgg_fcts.update({'_unicode_': 'Γ'})  # \Gamma is Greek G for Growth
PermGroFacAgg_fcts.update({'prmtv_par': 'True'})
init_perfect_foresight['prmtv_par'].append('PermGroFacAgg')
init_perfect_foresight['_fcts'].update({'PermGroFacAgg': PermGroFacAgg_fcts})
init_perfect_foresight.update({'PermGroFacAgg_fcts': PermGroFacAgg_fcts})

BoroCnstArt_fcts = {
    'about': 'If not None, maximum proportion of permanent income borrowable'}
BoroCnstArt_fcts.update({'latexexpr': r'\providecommand{\BoroCnstArt}{\underline{a}}\BoroCnstArt'})
BoroCnstArt_fcts.update({'prmtv_par': 'True'})
init_perfect_foresight['prmtv_par'].append('BoroCnstArt')
init_perfect_foresight['_fcts'].update({'BoroCnstArt': BoroCnstArt_fcts})
init_perfect_foresight.update({'BoroCnstArt_fcts': BoroCnstArt_fcts})

MaxKinks_fcts = {
    'about': 'PF Constrained model solves to period T-MaxKinks,'
    ' where the solution has exactly this many kink points'}
MaxKinks_fcts.update({'prmtv_par': 'False'})
# init_perfect_foresight['prmtv_par'].append('MaxKinks')
init_perfect_foresight['_fcts'].update({'MaxKinks': MaxKinks_fcts})
init_perfect_foresight.update({'MaxKinks_fcts': MaxKinks_fcts})

mcrlo_AgentCount_fcts = {
    'about': 'Number of agents to use in baseline Monte Carlo simulation'}
mcrlo_AgentCount_fcts.update(
    {'latexexpr': '\providecommand{\mcrlo_AgentCount}{N}\mcrlo_AgentCount'})
mcrlo_AgentCount_fcts.update({'mcrlo_sim': 'True'})
mcrlo_AgentCount_fcts.update({'mcrlo_lim': 'infinity'})
# init_perfect_foresight['mcrlo_sim'].append('mcrlo_AgentCount')
init_perfect_foresight['_fcts'].update({'mcrlo_AgentCount': mcrlo_AgentCount_fcts})
init_perfect_foresight.update({'mcrlo_AgentCount_fcts': mcrlo_AgentCount_fcts})

aNrmInitMean_fcts = {
    'about': 'Mean initial population value of aNrm'}
aNrmInitMean_fcts.update({'mcrlo_sim': 'True'})
aNrmInitMean_fcts.update({'mcrlo_lim': 'infinity'})
init_perfect_foresight['mcrlo_sim'].append('aNrmInitMean')
init_perfect_foresight['_fcts'].update({'aNrmInitMean': aNrmInitMean_fcts})
init_perfect_foresight.update({'aNrmInitMean_fcts': aNrmInitMean_fcts})

aNrmInitStd_fcts = {
    'about': 'Std dev of initial population value of aNrm'}
aNrmInitStd_fcts.update({'mcrlo_sim': 'True'})
init_perfect_foresight['mcrlo_sim'].append('aNrmInitStd')
init_perfect_foresight['_fcts'].update({'aNrmInitStd': aNrmInitStd_fcts})
init_perfect_foresight.update({'aNrmInitStd_fcts': aNrmInitStd_fcts})

mcrlo_pLvlInitMean_fcts = {
    'about': 'Mean initial population value of log pLvl'}
mcrlo_pLvlInitMean_fcts.update({'mcrlo_sim': 'True'})
init_perfect_foresight['mcrlo_sim'].append('mcrlo_pLvlInitMean')
init_perfect_foresight['_fcts'].update({'mcrlo_pLvlInitMean': mcrlo_pLvlInitMean_fcts})
init_perfect_foresight.update({'mcrlo_pLvlInitMean_fcts': mcrlo_pLvlInitMean_fcts})

mcrlo_pLvlInitStd_fcts = {
    'about': 'Mean initial std dev of log ppLvl'}
mcrlo_pLvlInitStd_fcts.update({'mcrlo_sim': 'True'})
init_perfect_foresight['mcrlo_sim'].append('mcrlo_pLvlInitStd')
init_perfect_foresight['_fcts'].update({'mcrlo_pLvlInitStd': mcrlo_pLvlInitStd_fcts})
init_perfect_foresight.update({'mcrlo_pLvlInitStd_fcts': mcrlo_pLvlInitStd_fcts})

T_age_fcts = {
    'about': 'Age after which simulated agents are automatically killedl'}
T_age_fcts.update({'mcrlo_sim': 'False'})
init_perfect_foresight['_fcts'].update({'T_age': T_age_fcts})
init_perfect_foresight.update({'T_age_fcts': T_age_fcts})

T_cycle_fcts = {
    'about': 'Number of periods in a "cycle" (like, a lifetime) for this agent type'}
init_perfect_foresight['_fcts'].update({'T_cycle': T_cycle_fcts})
init_perfect_foresight.update({'T_cycle_fcts': T_cycle_fcts})

cycles_fcts = {
    'about': 'Number of times the sequence of periods/stages should be solved'}
init_perfect_foresight['_fcts'].update({'cycle': cycles_fcts})
init_perfect_foresight.update({'cycles_fcts': cycles_fcts})


class OneStateConsumerType(AgentType):
    """
    Construct endpoint for solution of problem of a consumer with
    one state variable, m:

        * m combines assets from prior history with current income

        * it is referred to as `market resources` throughout the docs

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
            pseudo_terminal=True,
            **kwds
    ):
        # This LinearInterp extrapolates the 45 degree line to infinity
        cFunc_terminal_nobequest_ = LinearInterp([0.0, 1.0], [0.0, 1.0])
        cFunc_terminal_ = cFunc_terminal_nobequest_

        solution_nobequest_ = ConsumerSolution(  # Omits vFunc b/c u not yet def
            cFunc=cFunc_terminal_nobequest_,
            mNrmMin=0.0,
            hNrm=0.0,
            MPCmin=1.0,
            MPCmax=1.0,
            stge_kind={
                'iter_status': 'terminal',
                'term_type': 'nobequest',
                'maker_cls': 'OneStateConsumerType'
            }
        )
        def cFunc_afterlife_nobequest_(x): return float('inf')
        def vFunc_afterlife_nobequest_(x): return 0.0
        def vPfunc_afterlife_nobequest_(x): return 0.0
        def vPPfunc_afterlife_nobequest_(x): return 0.0

        # This configuration of the 'afterlife' is constructed so that when
        # the standard lifetime transition rules are applied, the nobequest
        # terminal solution is generated
        solution_afterlife_nobequest_ = ConsumerSolution(
            cFunc=cFunc_afterlife_nobequest_,
            vFunc=vFunc_afterlife_nobequest_,
            vPfunc=vPfunc_afterlife_nobequest_,
            vPPfunc=vPPfunc_afterlife_nobequest_,
            mNrmMin=0.0,
            hNrm=-1.0,
            MPCmin=float('inf'),
            MPCmax=float('inf')
        )

        solution_afterlife_nobequest_.stge_kind = {
            'iter_status': 'afterlife',
            'term_type': 'nobequest',
            'maker_cls': 'OneStateConsumerType'
        }
        solution_nobequest_.solution_next = solution_afterlife_nobequest_

        # Define solution_terminal_ for legacy reasons
        solution_terminal_ = solution_nobequest_
        # If user has not provided a terminal sol
        if not solution_startfrom:  # Then use the default solution_terminal
            solution_startfrom = deepcopy(solution_nobequest_)

        AgentType.__init__(
            self,  # position makes this solution_terminal in AgentType
            solution_terminal=solution_startfrom,  # whether handmade or default
            cycles=cycles,
            pseudo_terminal=False,
            **kwds
        )
        self.soln_crnt = ConsumerSolution()  # Mainly for storing functions, methods
        self.soln_crnt.Ths = nsp()  # For storing variables

        self.dolo_defs()  # Instantiate (partial) dolo description

    def dolo_defs(self):  # CDC 20210415: Beginnings of Dolo integration
        self.symbol_calibration = dict(  # not used yet, just created
            states={"m": 1.0},
            controls=["c"],
        )  # Things all such models have in common


class PerfForesightConsumerType(AgentType):

    """
    A perfect foresight consumer who has no uncertainty other than mortality.
    Problem is defined by a coefficient of relative risk aversion, geometric
    discount factor, interest factor, an artificial borrowing constraint (maybe)
    and time sequences of the permanent income growth rate and survival.

    Parameters
    ----------
    cycles : int
        Number of times the sequence of periods/stages should be solved.
    """

    time_vary_ = ["LivPrb",  # Age-varying death rates can match mortality data
                  "PermGroFac"]  # Age-varying income growth can match data
    time_inv_ = ["CRRA", "Rfree", "DiscFac", "MaxKinks", "BoroCnstArt"]
    state_vars = ['pLvl',  # Idiosyncratic permanent income
                  'PlvlAgg',  # Aggregate permanent income
                  'bNrm',  # Bank balances beginning of period (pLvl normed)
                  'mNrm',  # Market resources (b + income) (pLvl normed)
                  "aNrm"]  # Assets after all actions (pLvl normed)
    shock_vars_ = []

    def __init__(self,
                 cycles=1,  # Default to finite horiz
                 verbose=1,
                 quiet=False,  # do not check conditions
                 solution_startfrom=None,  # Default is no interim solution
                 BoroCnstArt=None,
                 solver=ConsPerfForesightSolver,
                 **kwds
                 ):
        self.kwds_upd = init_perfect_foresight.copy()  # Get defaults
        self.kwds_upd.update(kwds)  # Replace defaults with passed vals if difft
        OneStateConsumerType.__init__(  # Universals for one state var c models
            self,
            solution_startfrom=None,  # defaults to nobequest
            cycles=cycles,
            pseudo_terminal=False,
            ** self.kwds_upd
        )
        # OneStateConsumerType creates self.soln_crnt and self.soln_crnt.Ths

        # If they did not provide their own solution_startfrom, use default
        if not hasattr(self, 'solution_startfrom'):
            # enrich default OneStateConsumerType terminal function
            # with info specifically needed to solve PF CRRA model
            soln_crnt = self.finish_setup_default_term_by_putting_into_soln_crnt()
            # url that will locate the documentation
            self.url_doc_for_this_agent_type_get()
        else:
            # user-provided solution should already be enriched therewith
            soln_crnt = solution_startfrom

        # Construct one-period(/stage) solver
        self.solve_one_period = \
            make_one_period_oo_solver(solver)  # allows user-specified alt

        # Add parameters not already automatically captured
        self.parameters.update({"cycles": self.cycles})
        self.parameters.update({"model_type": self.class_name})

        # Add solver parameters so solution knows where it came from
        soln_crnt.parameters_solver = self.parameters_solver = \
            get_solve_one_period_args(self,
                                      self.solve_one_period, stge_which=0)

        # Store initial model params; later used to test if anything changed
        self.store_model_params(self.kwds_upd['prmtv_par'],
                                self.kwds_upd['aprox_lim'])

        # Honor arguments (if any) provided with solution_startfrom call
        self.verbose = verbose
        set_verbosity_level((4 - verbose) * 10)
        self.quiet = quiet
        self.dolo_defs()

    def dolo_defs(self):  # CDC 20210415: Beginnings of Dolo integration
        self.symbol_calibration = dict(  # not used yet, just created
            states={"mNrm": 2.0,
                    "aNrm": 1.0,
                    "bNrm": 1.0,
                    "pLvl": 1.0,
                    "pLvlAgg": 1.0
                    },
            controls=["cNrm"],
            exogenous=[],
            parameters={"DiscFac": 0.96, "LivPrb": 1.0, "CRRA": 2.0,
                        "Rfree": 1.03, "PermGroFac": 1.0,
                        "BoroCnstArt": None,
                        }
            # Not clear how to specify characteristics of sim starting point
        )  # Things all ConsumerSolutions have in common

    def finish_setup_default_term_by_putting_into_soln_crnt(self):
        """
        Add to solution_terminal characteristics of the agent required
        for solution but which are not automatically added as part of
        the solution process.

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

        # OneStateConsumerType + AgentType => if we got here
        # then self.solution_terminal is default from OneStateConsumerType
        # which is nobequest
        soln_crnt = self.soln_crnt = self.solution_terminal
        soln_futr = self.solution_terminal.solution_next
        soln_crnt.Ths = nsp()

        # Put these at root of solution
        soln_crnt.time_vary = self.time_vary = deepcopy(self.time_vary_)
        soln_crnt.time_inv = self.time_inv = deepcopy(self.time_inv_)

        # Natural borrowing constraint: Cannot die in debt
        soln_crnt.Ths.BoroCnstNat = soln_crnt.BoroCnstNat = -(soln_crnt.hNrm + soln_crnt.mNrmMin)

        self.vFunc = soln_crnt.Ths.vFunc = soln_crnt.vFunc = ValueFuncCRRA(
            soln_crnt.cFunc, self.CRRA)
        self.vPfunc = soln_crnt.Ths.vPfunc = soln_crnt.vPfunc = MargValueFuncCRRA(
            soln_crnt.cFunc, self.CRRA)
        self.vPPfunc = soln_crnt.Ths.vPPfunc = soln_crnt.vPPfunc = MargMargValueFuncCRRA(
            soln_crnt.cFunc, self.CRRA)

        # CDC 20210423:
        # utility u, marginail utility u' is uP, marginal marginal uPP
        soln_crnt.Ths.u = soln_crnt.u = lambda c: utility(c, self.CRRA)
        soln_crnt.Ths.uP = soln_crnt.uP = lambda c: utilityP(c, self.CRRA)
        soln_crnt.Ths.uPP = soln_crnt.uPP = lambda c: utilityPP(c, self.CRRA)

        # Inverses and pseudo-inverses
        soln_crnt.Ths.uPinv = soln_crnt.uPinv = lambda u: utilityP_inv(u, self.CRRA)
        soln_crnt.Ths.uPinvP = soln_crnt.uPinvP = lambda u: utilityP_invP(u, self.CRRA)
        soln_crnt.Ths.uinvP = soln_crnt.uinvP = lambda u: utility_invP(u, self.CRRA)
        soln_crnt.Ths.uinv = soln_crnt.uinv = lambda u: utility_inv(
            u, self.CRRA)  # in case vFuncBool

        soln_crnt.stge_kind = {'iter_status': 'terminal',
                               'slvr_type': 'ConsPerfForesightSolver'
                               }

        # Natural borrowing constraint: Cannot die in debt
        soln_crnt.Ths.mNrmMin = deepcopy(soln_crnt.mNrmMin)
        soln_crnt.Ths.hNrm = deepcopy(soln_crnt.hNrm)
        soln_crnt.Ths.MPCmin = deepcopy(soln_crnt.MPCmin)
        soln_crnt.Ths.MPCmax = deepcopy(soln_crnt.MPCmax)
        soln_crnt.Ths.BoroCnstNat = deepcopy(soln_crnt.BoroCnstNat)

        # Solution options
        soln_crnt.Ths.vFuncBool = self.vFuncBool
        soln_crnt.Ths.CubicBool = self.CubicBool

        return soln_crnt

    def url_doc_for_this_agent_type_get(self):
        # Generate a url that will locate the documentation
        self.class_name = self.__class__.__name__
        self.url_ref = "https://econ-ark.github.io/BufferStockTheory"
        self.urlroot = self.url_ref+'/#'
        self.url_doc = "https://hark.readthedocs.io/en/latest/search.html?q=" +\
            self.class_name+"&check_keywords=yes&area=default#"

    def store_model_params(self, prmtv_par, aprox_lim):
        # When anything cached here changes, solution SHOULD change
        self.prmtv_par_vals = {}
        for par in prmtv_par:
            self.prmtv_par_vals[par] = getattr(self, par)

        self.aprox_par_vals = {}
        for key in aprox_lim:
            self.aprox_par_vals[key] = getattr(self, key)

        # Merge to get all aprox and prmtv params
        self.solve_par_vals = \
            {**self.prmtv_par_vals, **self.aprox_par_vals}

        # Let solver know about all the  params of the modl
        self.solve_one_period.parameters_model = self.parameters

        # and about the ones which, if they change, require iterating
        self.solve_one_period.solve_par_vals = self.solve_par_vals
#        solver.solve_par_vals = self.solve_par_vals

    def check_conditions(self, verbose=3):

        if not hasattr(self, 'solution'):  # Need a solution to have been computed
            _log.info('Solving penultimate period because solution needed to check conditions')
            self.make_solution_for_final_period()

#        self.solution[-1].check_conditions(self.solution[-1], verbose=3)
        self.solution[-1].check_conditions(self.solution[-1], verbose)

    def pre_solve(self):  # Do anything necessary to prepare agent to solve

        if not self.BoroCnstArt:
            if hasattr(self, "MaxKinks"):
                if self.MaxKinks:  # True if MaxKinks is anything other than None
                    raise(
                        AttributeError(
                            "Kinks are caused by constraints.  Cannot specify MaxKinks without constraints!  Ignoring."
                        ))
                self.MaxKinks = np.inf
            return
        # Then it has a borrowing constraint
        if hasattr(self, "MaxKinks"):
            if self.cycles > 0:  # If it's not an infinite horizon model...
                self.MaxKinks = np.inf  # ...there's no need to set MaxKinks
            else:
                raise (
                    AttributeError(
                        "PerfForesightConsumerType requires MaxKinks when BoroCnstArt is not None, cycles == 0."
                    )
                )

    def check_restrictions(self):  # url/#check-restrictions
        """
        A method to check that various restrictions are met for the model class.
        """
        if self.DiscFac <= 0:
            raise Exception("DiscFac is zero or less with value: " + str(self.DiscFac))

        if self.Rfree < 0:
            raise Exception("Rfree is below zero with value: " + str(self.DiscFac))

        if self.PermGroFac < 0:
            raise Exception("PermGroFac is negative with value: " + str(self.PermGroFac))

        if self.LivPrb < 0:
            raise Exception("LivPrb is less than zero with value: " + str(self.LivPrb))

        if self.LivPrb > 1:
            raise Exception("LivPrb is greater than one with value: " + str(self.LivPrb))

        if self.tranShkStd < 0:
            raise Exception("tranShkStd is negative with value: " + str(self.tranShkStd))

        if self.permShkStd < 0:
            raise Exception("permShkStd is negative with value: " + str(self.permShkStd))

        if self.IncUnemp < 0:
            raise Exception("IncUnemp is negative with value: " + str(self.IncUnemp))

        if self.IncUnempRet < 0:
            raise Exception("IncUnempRet is negative with value: " + str(self.IncUnempRet))

        if self.CRRA <= 1:
            raise Exception("CRRA is <= 1 with value: " + str(self.CRRA))

        return

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

    def initialize_sim(self):
        self.permShkAgg = self.PermGroFacAgg  # Never changes during sim
        self.state_now['PlvlAgg'] = 1.0
        AgentType.initialize_sim(self)

    def mcrlo_birth(self, which_agents):
        """
        Makes new consumers for the given indices.  Initialized variables include aNrm and pLvl, as
        well as time variables t_age and t_cycle.  Normalized assets and permanent income levels
        are drawn from lognormal distributions given by aNrmInitMean and aNrmInitStd (etc).

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.mcrlo_AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        """
        # Get and store states for newly born agents
        N = np.sum(which_agents)  # Number of new consumers to make
        self.state_now['aNrm'][which_agents] = Lognormal(
            mu=self.aNrmInitMean,
            sigma=self.aNrmInitStd,
            seed=self.RNG.randint(0, 2 ** 31 - 1),
        ).draw(N)
        # why is a now variable set here? Because it's an aggregate.
        mcrlo_pLvlInitMean = self.mcrlo_pLvlInitMean + np.log(
            self.state_now['PlvlAgg']
        )  # Account for newer cohorts having higher permanent income
        self.state_now['pLvl'][which_agents] = Lognormal(
            mcrlo_pLvlInitMean,
            self.mcrlo_pLvlInitStd,
            seed=self.RNG.randint(0, 2 ** 31 - 1)
        ).draw(N)
        self.t_age[which_agents] = 0  # How many periods since each agent was born
        self.t_cycle[
            which_agents
        ] = 0  # Which period of the cycle each agent is currently in
        return None

    def mcrlo_death(self):
        """
        Determines which agents die this period and must be replaced.  Uses the sequence in LivPrb
        to determine survival probabilities for each agent.

        Parameters
        ----------
        None

        Returns
        -------
        which_agents : np.array(bool)
            Boolean array of size mcrlo_AgentCount indicating which agents die.
        """
        # Determine who dies
        DiePrb_by_t_cycle = 1.0 - np.asarray(self.LivPrb)
        DiePrb = DiePrb_by_t_cycle[
            self.t_cycle - 1
        ]  # Time has already advanced, so look back one
        DeathShks = Uniform(seed=self.RNG.randint(0, 2 ** 31 - 1)).draw(
            N=self.mcrlo_AgentCount
        )
        which_agents = DeathShks < DiePrb
        if self.T_age is not None:  # Kill agents that have lived for too many periods
            too_old = self.t_age >= self.T_age
            which_agents = np.logical_or(which_agents, too_old)
        return which_agents

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
        self.shocks['tranShk'] = np.ones(self.mcrlo_AgentCount)

    def get_Rfree(self):  # -> mcrlo_get_Rfree.
        # CDC: We should have a generic mcrlo_get_all_params
        """
        Returns an array of size self.mcrlo_AgentCount with self.Rfree in every entry.

        Parameters
        ----------
        None

        Returns
        -------
        Rfree : np.array
            Array of size self.mcrlo_AgentCount with risk free interest rate for each agent.
        """
        Rfree = self.Rfree * np.ones(self.mcrlo_AgentCount)
        return Rfree

    def transition(self):  # -> mcrlo_trnsitn
        pLvlPrev = self.state_prev['pLvl']
        aNrmPrev = self.state_prev['aNrm']
        Rfree = self.get_Rfree()

        # Calculate new states: normalized market resources and permanent income level
        pLvl = pLvlPrev*self.shocks['permShk']  # Updated permanent income level
        # Updated aggregate permanent productivity level
        PlvlAgg = self.state_prev['PlvlAgg']*self.permShkAgg
        # "Effective" interest factor on normalized assets
        Reff = Rfree/self.shocks['permShk']
        bNrm = Reff*aNrmPrev         # Bank balances before labor income
        mNrm = bNrm + self.shocks['tranShk']  # Market resources after income

        return pLvl, PlvlAgg, bNrm, mNrm, None

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
        cNrm = np.zeros(self.mcrlo_AgentCount) + np.nan
        MPCnow = np.zeros(self.mcrlo_AgentCount) + np.nan
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            cNrm[these], MPCnow[these] = self.solution[t].cFunc.eval_with_derivative(
                self.state_now['mNrm'][these]
            )
            self.controls['cNrm'] = cNrm

        # MPCnow is not really a control
        self.MPCnow = MPCnow
        return None

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


# Make a dictionary to specify an idiosyncratic income shocks consumer
init_idiosyncratic_shocks = dict(
    init_perfect_foresight,
    **{
        # Income process variables
        "permShkStd": [0.1],  # Standard deviation of log permanent income shocks
        "tranShkStd": [0.1],  # Standard deviation of log transitory income shocks
        "UnempPrb": 0.05,  # Probability of unemployment while working
        "UnempPrbRet": 0.005,  # Probability of "unemployment" while retired
        "IncUnemp": 0.3,  # Unemployment benefits replacement rate
        "IncUnempRet": 0.0,  # "Unemployment" benefits when retired
        "BoroCnstArt": 0.0,  # Artificial borrowing constraint; imposed minimum level of end-of period assets
        "tax_rate": 0.0,  # Flat income tax rate
        "T_retire": 0,  # Period of retirement (0 --> no retirement)
        # Parameters governing construction of income process
        "permShkCount": 7,  # Number of points in discrete approximation to permanent income shocks
        "tranShkCount": 7,  # Number of points in discrete approximation to transitory income shocks
        # parameters governing construction of grid of assets above min value
        "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
        "aXtraMax": 20,     # Maximum end-of-period "assets above minimum" value
        "aXtraNestFac": 3,  # Exponential nesting factor when constructing "assets above minimum" grid
        "aXtraCount": 48,   # Number of points in the grid of "assets above minimum"
        # list other values of "assets above minimum" to add to the grid (e.g., 10000)
        "aXtraExtra": [None],
        "vFuncBool": False,  # Whether to calculate the value function during solution
        "CubicBool": False,  # Use cubic spline interpolation when True, linear interpolation when False
    }
)

# The info above is necessary and sufficient for defining the consumer
# The info below is supplemental
# Some of it is required for further purposes

# Parameters required for a (future) matrix-based discretization of the problem
init_idiosyncratic_shocks.update({
    'matrx_par': {
        'mcrlo_aXtraCount': '100',
        'mcrlo_aXtraMin': init_idiosyncratic_shocks['aXtraMin'],
        'mcrlo_aXtraMax': init_idiosyncratic_shocks['aXtraMax']
    }})
init_idiosyncratic_shocks.update({
    'matrx_lim': {
        'mcrlo_aXtraCount': float('inf'),
        'mcrlo_aXtraMin': float('inf'),
        'mcrlo_aXtraMax': float('inf')
    }})

#  add parameters that were not part of perfect foresight model
# Primitives
init_idiosyncratic_shocks['prmtv_par'].append('permShkStd')
init_idiosyncratic_shocks['prmtv_par'].append('tranShkStd')
init_idiosyncratic_shocks['prmtv_par'].append('UnempPrb')
init_idiosyncratic_shocks['prmtv_par'].append('UnempPrbRet')
init_idiosyncratic_shocks['prmtv_par'].append('IncUnempRet')
init_idiosyncratic_shocks['prmtv_par'].append('BoroCnstArt')
init_idiosyncratic_shocks['prmtv_par'].append('tax_rate')
init_idiosyncratic_shocks['prmtv_par'].append('T_retire')

# Approximation parameters and their limits (if any)
# init_idiosyncratic_shocks['aprox_par'].append('permShkCount')
init_idiosyncratic_shocks['aprox_lim'].update({'permShkCount': 'infinity'})
# init_idiosyncratic_shocks['aprox_par'].append('tranShkCount')
init_idiosyncratic_shocks['aprox_lim'].update({'tranShkCount': 'infinity'})
# init_idiosyncratic_shocks['aprox_par'].append('aXtraMin')
init_idiosyncratic_shocks['aprox_lim'].update({'aXtraMin': float('0.0')})
# init_idiosyncratic_shocks['aprox_par'].append('aXtraMax')
init_idiosyncratic_shocks['aprox_lim'].update({'aXtraMax': float('inf')})
# init_idiosyncratic_shocks['aprox_par'].append('aXtraNestFac')
init_idiosyncratic_shocks['aprox_lim'].update({'aXtraNestFac': None})
# init_idiosyncratic_shocks['aprox_par'].append('aXtraCount')
init_idiosyncratic_shocks['aprox_lim'].update({'aXtraCount': None})

IncShkDstn_fcts = {
    'about': 'Income Shock Distribution: .X[0] and .X[1] retrieve shocks, .pmf retrieves probabilities'}
IncShkDstn_fcts.update({'py___code': r'construct_lognormal_income_process_unemployment'})
init_idiosyncratic_shocks['_fcts'].update({'IncShkDstn': IncShkDstn_fcts})
init_idiosyncratic_shocks.update({'IncShkDstn_fcts': IncShkDstn_fcts})

permShkStd_fcts = {
    'about': 'Standard deviation for lognormal shock to permanent income'}
permShkStd_fcts.update({'latexexpr': '\permShkStd'})
init_idiosyncratic_shocks['_fcts'].update({'permShkStd': permShkStd_fcts})
init_idiosyncratic_shocks.update({'permShkStd_fcts': permShkStd_fcts})

tranShkStd_fcts = {
    'about': 'Standard deviation for lognormal shock to permanent income'}
tranShkStd_fcts.update({'latexexpr': '\tranShkStd'})
init_idiosyncratic_shocks['_fcts'].update({'tranShkStd': tranShkStd_fcts})
init_idiosyncratic_shocks.update({'tranShkStd_fcts': tranShkStd_fcts})

UnempPrb_fcts = {
    'about': 'Probability of unemployment while working'}
UnempPrb_fcts.update({'latexexpr': r'\UnempPrb'})
UnempPrb_fcts.update({'_unicode_': '℘'})
init_idiosyncratic_shocks['_fcts'].update({'UnempPrb': UnempPrb_fcts})
init_idiosyncratic_shocks.update({'UnempPrb_fcts': UnempPrb_fcts})

UnempPrbRet_fcts = {
    'about': '"unemployment" in retirement = big medical shock'}
UnempPrbRet_fcts.update({'latexexpr': r'\UnempPrbRet'})
init_idiosyncratic_shocks['_fcts'].update({'UnempPrbRet': UnempPrbRet_fcts})
init_idiosyncratic_shocks.update({'UnempPrbRet_fcts': UnempPrbRet_fcts})

IncUnemp_fcts = {
    'about': 'Unemployment insurance replacement rate'}
IncUnemp_fcts.update({'latexexpr': '\IncUnemp'})
IncUnemp_fcts.update({'_unicode_': 'μ'})
init_idiosyncratic_shocks['_fcts'].update({'IncUnemp': IncUnemp_fcts})
init_idiosyncratic_shocks.update({'IncUnemp_fcts': IncUnemp_fcts})

IncUnempRet_fcts = {
    'about': 'Size of medical shock (frac of perm inc)'}
init_idiosyncratic_shocks['_fcts'].update({'IncUnempRet': IncUnempRet_fcts})
init_idiosyncratic_shocks.update({'IncUnempRet_fcts': IncUnempRet_fcts})

tax_rate_fcts = {
    'about': 'Flat income tax rate'}
tax_rate_fcts.update({
    'about': 'Size of medical shock (frac of perm inc)'})
init_idiosyncratic_shocks['_fcts'].update({'tax_rate': tax_rate_fcts})
init_idiosyncratic_shocks.update({'tax_rate_fcts': tax_rate_fcts})

T_retire_fcts = {
    'about': 'Period of retirement (0 --> no retirement)'}
init_idiosyncratic_shocks['_fcts'].update({'T_retire': T_retire_fcts})
init_idiosyncratic_shocks.update({'T_retire_fcts': T_retire_fcts})

permShkCount_fcts = {
    'about': 'Num of pts in discrete approx to permanent income shock dstn'}
init_idiosyncratic_shocks['_fcts'].update({'permShkCount': permShkCount_fcts})
init_idiosyncratic_shocks.update({'permShkCount_fcts': permShkCount_fcts})

tranShkCount_fcts = {
    'about': 'Num of pts in discrete approx to transitory income shock dstn'}
init_idiosyncratic_shocks['_fcts'].update({'tranShkCount': tranShkCount_fcts})
init_idiosyncratic_shocks.update({'tranShkCount_fcts': tranShkCount_fcts})

aXtraMin_fcts = {
    'about': 'Minimum end-of-period "assets above minimum" value'}
init_idiosyncratic_shocks['_fcts'].update({'aXtraMin': aXtraMin_fcts})
init_idiosyncratic_shocks.update({'aXtraMin_fcts': aXtraMin_fcts})

aXtraMax_fcts = {
    'about': 'Maximum end-of-period "assets above minimum" value'}
init_idiosyncratic_shocks['_fcts'].update({'aXtraMax': aXtraMax_fcts})
init_idiosyncratic_shocks.update({'aXtraMax_fcts': aXtraMax_fcts})

aXtraNestFac_fcts = {
    'about': 'Exponential nesting factor when constructing "assets above minimum" grid'}
init_idiosyncratic_shocks['_fcts'].update({'aXtraNestFac': aXtraNestFac_fcts})
init_idiosyncratic_shocks.update({'aXtraNestFac_fcts': aXtraNestFac_fcts})

aXtraCount_fcts = {
    'about': 'Number of points in the grid of "assets above minimum"'}
init_idiosyncratic_shocks['_fcts'].update({'aXtraMax': aXtraCount_fcts})
init_idiosyncratic_shocks.update({'aXtraMax_fcts': aXtraCount_fcts})

aXtraCount_fcts = {
    'about': 'Number of points to include in grid of assets above minimum possible'}
init_idiosyncratic_shocks['_fcts'].update({'aXtraCount': aXtraCount_fcts})
init_idiosyncratic_shocks.update({'aXtraCount_fcts': aXtraCount_fcts})

aXtraExtra_fcts = {
    'about': 'List of other values of "assets above minimum" to add to the grid (e.g., 10000)'}
init_idiosyncratic_shocks['_fcts'].update({'aXtraExtra': aXtraExtra_fcts})
init_idiosyncratic_shocks.update({'aXtraExtra_fcts': aXtraExtra_fcts})

aXtraGrid_fcts = {
    'about': 'Grid of values to add to minimum possible value to obtain actual end-of-period asset grid'}
init_idiosyncratic_shocks['_fcts'].update({'aXtraGrid': aXtraGrid_fcts})
init_idiosyncratic_shocks.update({'aXtraGrid_fcts': aXtraGrid_fcts})

vFuncBool_fcts = {
    'about': 'Whether to calculate the value function during solution'
}
init_idiosyncratic_shocks['_fcts'].update({'vFuncBool': vFuncBool_fcts})
init_idiosyncratic_shocks.update({'vFuncBool_fcts': vFuncBool_fcts})

CubicBool_fcts = {
    'about': 'Use cubic spline interpolation when True, linear interpolation when False'
}
init_idiosyncratic_shocks['_fcts'].update({'CubicBool': CubicBool_fcts})
init_idiosyncratic_shocks.update({'CubicBool_fcts': CubicBool_fcts})


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
        A user-specified starting point (last stage solution) for the iteration,
        to be used in place of the hardwired solution_terminal.  For example, you
        might set a loose tolerance to get a quick `solution_rough,` and
        then set the tolerance lower, or change some approximation parameter,
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
        "MaxKinks"  # PF inf hor with MaxKinks is equiv to fin hor with hor=MaxKinks
    )

    shock_vars_ = ['permShk', 'tranShk']  # The unemployment shock is transitory

    def __init__(self, cycles=1, verbose=1,  quiet=True, solution_startfrom=None, **kwds):
        params = init_idiosyncratic_shocks.copy()  # Get default params
        params.update(kwds)  # Update/overwrite dict params with user-specified

        # Construct and inherit characteristics of a PF model
        # initialized with the same parameters

        PerfForesightConsumerType.__init__(
            self, cycles=cycles, verbose=verbose, quiet=quiet,
            solution_startfrom=solution_startfrom,
            solver=ConsPerfForesightSolver,  # Default if (as usual) no solver supplied
            **params
        )

        # Reduce clutter with local variable (no annoying 'self.' required)
        soln_crnt = self.soln_crnt  # Already created by PerfForesightConsumerType
        Ths = soln_crnt.Ths

        # If precooked terminal answer not provided by user ...
        if not hasattr(self, 'solution_startfrom'):  # .. then init the default
            self.update_income_process()
            self.update_assets_grid()
            self.shock_vars = deepcopy(self.shock_vars_)  # Default before __init__
            # Add parameters NOT already created by PerfForesightConsumerType
            self.parameters.update({'shock_vars': self.shock_vars})
        else:
            # user-provided solution_startfrom should define shock_vars
            self.parameters.update({'shock_vars': solution_startfrom.shock_vars})

        # Add consumer-type specific objects; deepcopy creates own versions
        # - Default interpolation method is piecewise linear
        # - Cubic is smoother, works best if problem has no constraints
        # - User may or may not want to create the value function
        if (not self.soln_crnt.Ths.CubicBool) and (not self.soln_crnt.Ths.vFuncBool):
            solver = ConsIndShockSolverBasic
        else:  # Use the "advanced" solver if either is requested
            solver = ConsIndShockSolver

        # slvr_type will have been set by PF as perfect foresight; reset
        soln_crnt.stge_kind['slvr_type'] = 'ConsIndShockSolver'

#        if soln_crnt.stge_kind['iter_status'] == 'terminal':
#            breakpoint()

        # Attach the corresponding one-stage solver to the agent
        self.solve_one_period = make_one_period_oo_solver(solver)

        # Add solver args so solution soln_crnt knows how it was made
        soln_crnt.parameters_solver = \
            get_solve_one_period_args(self, self.solve_one_period, stge_which=0)

        # one_period_solver should know parameters that generated it
        self.solve_one_period.parameters_model = self.parameters

        # Store setup parameters so we can check for changes
        self.store_model_params(params['prmtv_par'], params['aprox_lim'])

        # Quiet mode: Define model without calculating anything
        # If not quiet, solve one period so we can check conditions

        if not quiet:
            self.check_conditions(verbose=3)  # Check conditions for nature/existence of soln

    def make_solution_for_final_period(self):  # solution[0]=terminal_solution
        # but with extra info required for backward induction
        self.tolerance_orig = deepcopy(self.tolerance)  # preserve true tolerance
        self.tolerance = float('inf')  # tolerance is infinity ...
        self.pseudo_terminal = True  # ... and pseudo_terminal = True
        self.solve()  # ... means that "solve" will stop after one period ...
        self.pseudo_terminal = False  # ... and replace terminal with updated
        self.tolerance = self.tolerance_orig  # which leaves us ready to solve

    def update_income_process(self):
        """
        Updates this agent's income process based on his own attributes.

        Parameters
        ----------
        none

        Returns:
        -----------
        none
        """
        soln_crnt = self.soln_crnt

        (IncShkDstn,
            permShkDstn,
            tranShkDstn,
         ) = self.construct_lognormal_income_process_unemployment()
        soln_crnt.IncShkDstn = self.IncShkDstn = IncShkDstn
        soln_crnt.permShkDstn = self.permShkDstn = permShkDstn
        soln_crnt.tranShkDstn = self.tranShkDstn = tranShkDstn
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
        soln_crnt = self.soln_crnt
        soln_crnt.aXtraGrid = self.aXtraGrid = construct_assets_grid(self)
        self.add_to_time_inv("aXtraGrid")
        self.parameters.update({'aXtraGrid': self.aXtraGrid})

    def update(self):
        """
        Update the characteristics of the problem if any parameters have changed
        since the last update.

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

        solve_par_vals_now = {}
        for par in self.solve_par_vals:
            solve_par_vals_now[par] = getattr(self, par)

        if not solve_par_vals_now == self.solve_par_vals:
            _log.info('Some model parameter has changed since last update.')
            _log.info('Storing new parameters and updating shocks and grid.')
            self.finish_setup_default_term_by_putting_into_soln_crnt()

        # Encourage the practice of sharing with the solver the agent's parameters
        try:
            self.solve_one_period.parameters_model
        except NameError:  # If solver doesn't have such a variable, mildly complain
            _log.info('No parameters inherited.  Please add a line like:')
            _log.info('')
            _log.info('    self.solve_one_period.parameters  = self.parameters')
            _log.info('')
            _log.info('before invoking the self.update() method in the calling AgentType.')
        else:
            if self.verbose == 3:
                _log.info('')
                _log.info('Solver inherited these parameters:')
                _log.info('')
                _log.info(self.solve_one_period.parameters_model.keys())
                _log.info('')
                _log.info('from model_type '+self.solve_one_period.parameters_model['model_type'])
                _log.info('')

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

    def pre_solve(self):  # Before beginning any solution steps
        self.update()

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
                permShkValsNxtRet = np.array(
                    [1.0, 1.0]
                )  # Permanent income is deterministic in retirement (2 states for temp income shocks)
                tranShkValsRet = np.array(
                    [
                        IncUnempRet,
                        (1.0 - UnempPrbRet * IncUnempRet) / (1.0 - UnempPrbRet),
                    ]
                )
                ShkPrbsRet = np.array([UnempPrbRet, 1.0 - UnempPrbRet])
            else:
                permShkValsNxtRet = np.array([1.0])
                tranShkValsRet = np.array([1.0])
                ShkPrbsRet = np.array([1.0])
                IncShkDstnRet = DiscreteApproximationToContinuousDistribution(
                    ShkPrbsRet,
                    [permShkValsNxtRet, tranShkValsRet],
                    seed=self.RNG.randint(0, 2 ** 31 - 1),
                )

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
        permShk = np.zeros(self.mcrlo_AgentCount)  # Initialize shock arrays
        tranShk = np.zeros(self.mcrlo_AgentCount)
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
        self.Emp = np.ones(self.mcrlo_AgentCount, dtype=bool)
        self.Emp[tranShk == self.IncUnemp] = False
        self.shocks['permShk'] = permShk
        self.shocks['tranShk'] = tranShk


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
        self.update()  # Make assets grid, income process, terminal solution

    def pre_solve(self):
        #        AgentType.pre_solve(self)
        self.update_solution_terminal()

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
        Returns an array of size self.mcrlo_AgentCount with self.Rboro or self.Rsave in each entry, based
        on whether self.aNrm >< 0.

        Parameters
        ----------
        None

        Returns
        -------
        Rfree : np.array
             Array of size self.mcrlo_AgentCount with risk free interest rate for each agent.
        """
        Rfree = self.Rboro * np.ones(self.mcrlo_AgentCount)
        Rfree[self.state_prev['aNrm'] > 0] = self.Rsave
        return Rfree


def apply_flat_income_tax(
        IncShkDstn, tax_rate, T_retire, unemployed_indices=None, transitory_index=2
):
    """
    Applies a flat income tax rate to all employed income states during the working
    period of life (those before T_retire).  Time runs forward in this function.

    Parameters
    ----------
    IncShkDstn : [distribution.Distribution]
        The discrete approximation to the income distribution in each time period.
    tax_rate : float
        A flat income tax rate to be applied to all employed income.
    T_retire : int
        The time index after which the agent retires.
    unemployed_indices : [int]
        Indices of transitory shocks that represent unemployment states (no tax).
    transitory_index : int
        The index of each element of IncShkDstn representing transitory shocks.

    Returns
    -------
    IncShkDstn_new : [distribution.Distribution]
        The updated income distributions, after applying the tax.
    """
    unemployed_indices = (
        unemployed_indices if unemployed_indices is not None else list()
    )
    IncShkDstn_new = deepcopy(IncShkDstn)
    i = transitory_index
    for t in range(len(IncShkDstn)):
        if t < T_retire:
            for j in range((IncShkDstn[t][i]).size):
                if j not in unemployed_indices:
                    IncShkDstn_new[t][i][j] = IncShkDstn[t][i][j] * (1 - tax_rate)
    return IncShkDstn_new

# =======================================================
# ================ Other useful functions ===============
# =======================================================


def construct_assets_grid(parameters):
    """
    Constructs the base grid of post-decision states, representing end-of-period
    assets above the absolute minimum.

    All parameters are passed as attributes of the single input parameters.  The
    input can be an instance of a ConsumerType, or a custom Parameters class.

    Parameters
    ----------
    aXtraMin:                  float
        Minimum value for the a-grid
    aXtraMax:                  float
        Maximum value for the a-grid
    aXtraCount:                 int
        Size of the a-grid
    aXtraExtra:                [float]
        Extra values for the a-grid.
    exp_nest:               int
        Level of nesting for the exponentially spaced grid

    Returns
    -------
    aXtraGrid:     np.ndarray
        Base array of values for the post-decision-state grid.
    """
    # Unpack the parameters
    aXtraMin = parameters.aXtraMin
    aXtraMax = parameters.aXtraMax
    aXtraCount = parameters.aXtraCount
    aXtraExtra = parameters.aXtraExtra
    grid_type = "exp_mult"
    exp_nest = parameters.aXtraNestFac

    # Set up post decision state grid:
    aXtraGrid = None
    if grid_type == "linear":
        aXtraGrid = np.linspace(aXtraMin, aXtraMax, aXtraCount)
    elif grid_type == "exp_mult":
        aXtraGrid = make_grid_exp_mult(
            ming=aXtraMin, maxg=aXtraMax, ng=aXtraCount, timestonest=exp_nest
        )
    else:
        raise Exception(
            "grid_type not recognized in __init__."
            + "Please ensure grid_type is 'linear' or 'exp_mult'"
        )
    # Add in additional points for the grid:
    for a in aXtraExtra:
        if a is not None:
            if a not in aXtraGrid:
                j = aXtraGrid.searchsorted(a)
                aXtraGrid = np.insert(aXtraGrid, j, a)
    return aXtraGrid


# Make a dictionary to specify a lifecycle consumer with a finite horizon

# Main calibration characteristics
birth_age = 25
death_age = 90
adjust_infl_to = 1992
# Use income estimates from Cagetti (2003) for High-school graduates
education = "HS"
income_calib = Cagetti_income[education]

# Income specification
income_params = parse_income_spec(
    age_min=birth_age,
    age_max=death_age,
    adjust_infl_to=adjust_infl_to,
    **income_calib,
    SabelhausSong=True
)

# Initial distribution of wealth and permanent income
dist_params = income_wealth_dists_from_scf(
    base_year=adjust_infl_to, age=birth_age, education=education, wave=1995
)

# We need survival probabilities only up to death_age-1, because survival
# probability at death_age is 1.
liv_prb = parse_ssa_life_table(
    female=False, cross_sec=True, year=2004, min_age=birth_age, max_age=death_age
    - 1)

# Parameters related to the number of periods implied by the calibration
time_params = parse_time_params(age_birth=birth_age, age_death=death_age)

# Update all the new parameters
init_lifecycle = copy(init_idiosyncratic_shocks)
init_lifecycle.update(time_params)
init_lifecycle.update(dist_params)
# Note the income specification overrides the mcrlo_pLvlInitMean from the SCF.
init_lifecycle.update(income_params)
init_lifecycle.update({"LivPrb": liv_prb})

# Make a dictionary to specify an infinite consumer with a four period cycle
init_cyclical = copy(init_idiosyncratic_shocks)
init_cyclical['PermGroFac'] = [1.082251, 2.8, 0.3, 1.1]
init_cyclical['permShkStd'] = [0.1, 0.1, 0.1, 0.1]
init_cyclical['tranShkStd'] = [0.1, 0.1, 0.1, 0.1]
init_cyclical['LivPrb'] = 4*[0.98]
init_cyclical['T_cycle'] = 4
