from builtins import range
from builtins import object
from copy import copy, deepcopy
import numpy as np
from scipy.optimize import newton
from HARK import AgentType, NullFunc, MetricObject, make_one_period_oo_solver
from HARK.utilities import warnings  # Because of "patch" to warnings modules
from HARK.interpolation import (
    CubicInterp,
    LowerEnvelope,
    LinearInterp,
    ValueFuncCRRA,
    MargValueFuncCRRA,
    MargMargValueFuncCRRA
)
from HARK.distribution import Lognormal, MeanOneLogNormal, Uniform
from HARK.distribution import (
    DiscreteDistribution,
    add_discrete_outcome_constant_mean,
    calc_expectation,
    combine_indep_dstns,
)
from HARK.utilities import (
    make_grid_exp_mult,
    CRRAutility,
    CRRAutilityP,
    CRRAutilityPP,
    CRRAutilityP_inv,
    CRRAutility_invP,
    CRRAutility_inv,
    CRRAutilityP_invP,
)
from HARK import _log
from HARK import set_verbosity_level
from HARK.Calibration.Income.IncomeTools import parse_income_spec, parse_time_params, Cagetti_income
from HARK.datasets.SCF.WealthIncomeDist.SCFDistTools import income_wealth_dists_from_scf
from HARK.datasets.life_tables.us_ssa.SSATools import parse_ssa_life_table
from builtins import str
"""
Classes to solve canonical consumption-saving models with idiosyncratic shocks
to income.  All models here assume CRRA utility with geometric discounting, no
bequest motive, and income shocks that are fully transitory or fully permanent.

It currently solves three types of models:
   1) A very basic "perfect foresight" consumption-savings model with no uncertainty.
   2) A consumption-saving model with transitory and permanent income shocks.
   3) The model described in (2), but with an interest rate for debt that differs
      from the interest rate for savings.

See NARK https://HARK.githhub.io/Documentation/NARK for information on variable naming conventions.
See HARK documentation for mathematical descriptions of the models being solved.
"""

__all__ = [
    "ConsumerSolution",
    "ConsumerSolutionWithEx",
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
# === Classes that help solve consumption-saving models ===
# =====================================================================


class ConsumerSolution(MetricObject):
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
        tion and other functions are undefined for m < mNrmMin.
    hNrmNow : float
        Human wealth after receiving income this period: PDV of all future
        income, ignoring mortality.
    MPCminNow : float
        Infimum of the marginal propensity to consume this period.
        MPC --> MPCminNow as m --> infinity.
    MPCmaxNow : float
        Supremum of the marginal propensity to consume this period.
        MPC --> MPCmaxNow as m --> mNrmMin.

    """
#    distance_criteria = ["vPfunc"]
    distance_criteria = ["cFunc"]

    def __init__(
            self,
            cFunc=None,
            vFunc=None,
            vPfunc=None,
            vPPfunc=None,
            mNrmMin=None,
            hNrmNow=None,
            MPCminNow=None,
            MPCmaxNow=None,
    ):
        # Generate a url that will locate the documentation
        self.url_doc = "https://hark.readthedocs.io/en/latest/search.html?q=" + \
            self.__class__.__name__+"&check_keywords=yes&area=default#"

        # Change any missing function inputs to NullFunc
        self.cFunc = cFunc if cFunc is not None else NullFunc()
        self.vFunc = vFunc if vFunc is not None else NullFunc()
        self.vPfunc = vPfunc if vPfunc is not None else NullFunc()
        self.vPPfunc = vPPfunc if vPPfunc is not None else NullFunc()
        self.mNrmMin = mNrmMin
        self.hNrmNow = hNrmNow
        self.MPCminNow = MPCminNow
        self.MPCmaxNow = MPCmaxNow
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
            self.vFunc.append(new_solution.vFuncv)
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

        m_init_guess = self.mNrmMin + self.Ex_IncNextNrm
        try:  # Find value where argument is zero
            self.mNrmTrg = newton(
                self.Ex_m_tp1_minus_m_t,
                m_init_guess)
        except:
            self.mNrmTrg = None

        return self.mNrmTrg


# class ConsumerSolutionWithEx(ConsumerSolution):
#     """
#     ConsumerSolution enhanced with functions to calculate various objects that
#     are interesting or useful for understanding the consumer's problem.

#     Parameters
#     ----------
#     cFunc : function
#         The consumption function for this period, defined over market
#         resources: c = cFunc(m).
#     vFunc : function
#         The beginning-of-period value function for this period, defined over
#         market resources: v = vFunc(m).
#     vPfunc : function
#         The beginning-of-period marginal value function for this period,
#         defined over market resources: vP = vPfunc(m).
#     vPPfunc : function
#         The beginning-of-period marginal marginal value function for this
#         period, defined over market resources: vPP = vPPfunc(m).
#     mNrmMin : float
#         The minimum allowable market resources for this period; the consump-
#         tion and other functions are undefined for m < mNrmMin.
#     hNrmNow : float
#         Human wealth after receiving income this period: PDV of all future
#         income, ignoring mortality.
#     MPCminNow : float
#         Infimum of the marginal propensity to consume this period.
#         MPC --> MPCminNow as m --> infinity.
#     MPCmaxNow : float
#         Supremum of the marginal propensity to consume this period.
#         MPC --> MPCmaxNow as m --> mNrmMin.
#     """

    # def mNrmTrg_finder(self):
    #     """
    #     Finds value of (normalized) market resources mNrm at which individual consumer
    #     expects m not to change.

    #     This will exist if the GICNrm holds.

    #     https://econ-ark.github.io/BufferStockTheory#UniqueStablePoints

    #     Parameters
    #     ----------
    #     solution : ConsumerSolution
    #         Solution to this period's problem, which must have attribute cFunc.
    #     Returns
    #     -------
    #         The target value mNrmTrg.
    #     """

    #     # Minimum market resources plus next income is okay starting guess
    #     # Better would be to presere the last value (if it exists)
    #     # and use that as a starting point

    #     m_init_guess = self.mNrmMin + self.Ex_IncNextNrm
    #     try:  # Find value where argument is zero
    #         self.mNrmTrg = newton(
    #             self.Ex_m_tp1_minus_m_t,
    #             m_init_guess)
    #     except:
    #         self.mNrmTrg = None

    #     return self.mNrmTrg


# =====================================================================
# === Classes and functions that solve consumption-saving models ===
# =====================================================================

# The PerfForesight class also incorporates and info that are useful for
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
    MaxKinks : int
        Maximum number of kink points to allow in the consumption function;
        additional points will be thrown out.  Only relevant in infinite
        horizon model with artificial borrowing constraint.
    """

    def __init__(
            self,
            solution_next,
            DiscFac,
            LivPrb,
            CRRA,
            Rfree,
            PermGroFac,
            BoroCnstArt,
            MaxKinks,
    ):
        # Generate a url that will locate the documentation
        self.url_doc = "https://hark.readthedocs.io/en/latest/search.html?q=" + \
            self.__class__.__name__+"&check_keywords=yes&area=default#"

        # url for paper that contains various theoretical results
        self.url_ref = "https://econ-ark.github.io/BufferStockTheory"
        self.solution_next = solution_next
        self.DiscFac = DiscFac
        self.LivPrb = LivPrb
        self.CRRA = CRRA
        self.Rfree = Rfree
        self.PermGroFac = PermGroFac
        self.BoroCnstArt = BoroCnstArt
        self.MaxKinks = MaxKinks
        self.results = {}
        self.conditions = {}

    def solution_add_further_info(self, solution):  # Perfect Foreight version
        """
        Adds to the solution extensive information and references about
        all its elements.

        Parameters
        ----------
        solution: ConsumerSolution
            A solution that has already been augmented with required_calcs

        Returns
        -------
        solution : ConsumerSolution
            Same solution that was provided, augmented with facts and 
            references
        """

        # To make formulae shorter, make local copies of solution variables
        urlroot = self.url_ref+'/#'  # used for references to derivations
#        solution.parameters_model = parameters = self.parameters
        BoroCnstArt = solution.BoroCnstArt = self.BoroCnstArt
        CRRA = solution.CRRA = self.CRRA
        DiscFac = solution.DiscFac = self.DiscFac
        LivNum = solution.LivNum = self.LivPrb
        PermGroNum = PermGroFac = solution.PermGroFac = self.PermGroFac
        Rfree = solution.Rfree = self.Rfree
        DiscFacEff = solution.DiscFacEff = DiscFac * LivNum

        solution.facts = {}
        # The code below is also used in the model with uncertainty
        # which explains why it contains a number of calculations that
        # are trivial or pointless in a perfect foresight model

        uInv_Ex_uInv_PermShk = 1.0
        solution.conditions = {}

        APF_facts = {
            'about': 'Absolute Patience Factor'}
        solution.APF = APF = \
            ((Rfree * DiscFacEff) ** (1.0 / CRRA))
        APF_facts.update({'latexexpr': r'\APF'})
        APF_facts.update({'_unicode_': r'Þ'})
        APF_facts.update({'urlhandle': urlroot+'APF'})
        APF_facts.update({'py___code': '(Rfree*DiscFacEff)**(1/CRRA)'})
        APF_facts.update({'value_now': APF})
        solution.facts.update({'APF': APF_facts})
        solution.APF_facts = APF_facts

        AIC_facts = {'about': 'Absolute Impatience Condition'}
        AIC_facts.update({'latexexpr': r'\AIC'})
        AIC_facts.update({'urlhandle': urlroot+'AIC'})
        AIC_facts.update({'py___code': 'test: APF < 1'})
        solution.facts.update({'AIC': AIC_facts})
        solution.AIC_facts = AIC_facts

        RPF_facts = {
            'about': 'Return Patience Factor'}
        RPF = APF / Rfree
        RPF_facts.update({'latexexpr': r'\RPF'})
        RPF_facts.update({'_unicode_': r'Þ_R'})
        RPF_facts.update({'urlhandle': urlroot+'RPF'})
        RPF_facts.update({'py___code': r'APF/Rfree'})
        RPF_facts.update({'value_now': RPF})
        solution.facts.update({'RPF': RPF_facts})
        solution.RPF_facts = RPF_facts
        solution.RPF = RPF

        RIC_facts = {'about': 'Growth Impatience Condition'}
        RIC_facts.update({'latexexpr': r'\RIC'})
        RIC_facts.update({'urlhandle': urlroot+'RIC'})
        RIC_facts.update({'py___code': 'test: agent.RPF < 1'})
        solution.facts.update({'RIC': RIC_facts})
        solution.RIC_facts = RIC_facts

        GPFRaw_facts = {
            'about': 'Growth Patience Factor'}
        GPFRaw = APF / PermGroNum
        GPFRaw_facts.update({'latexexpr': '\GPFRaw'})
        GPFRaw_facts.update({'urlhandle': urlroot+'GPFRaw'})
        GPFRaw_facts.update({'_unicode_': r'Þ_Γ'})
        GPFRaw_facts.update({'value_now': GPFRaw})
        solution.facts.update({'GPFRaw': GPFRaw_facts})
        solution.GPFRaw_facts = GPFRaw_facts
        solution.GPFRaw = GPFRaw

        GICRaw_facts = {'about': 'Growth Impatience Condition'}
        GICRaw_facts.update({'latexexpr': r'\GICRaw'})
        GICRaw_facts.update({'urlhandle': urlroot+'GICRaw'})
        GICRaw_facts.update({'py___code': 'test: agent.GPFRaw < 1'})
        solution.facts.update({'GICRaw': GICRaw_facts})
        solution.GICRaw_facts = GICRaw_facts

        GPFLiv_facts = {
            'about': 'Mortality-Risk-Adjusted Growth Patience Factor'}
        GPFLiv = APF * LivNum / PermGroNum
        GPFLiv_facts.update({'latexexpr': '\GPFLiv'})
        GPFLiv_facts.update({'urlhandle': urlroot+'GPFLiv'})
        GPFLiv_facts.update({'py___code': 'APF*Liv/PermGroNum'})
        GPFLiv_facts.update({'value_now': GPFLiv})
        solution.facts.update({'GPFLiv': GPFLiv_facts})
        solution.GPFLiv_facts = GPFLiv_facts
        solution.GPFLiv = GPFLiv

        GICLiv_facts = {'about': 'Growth Impatience Condition'}
        GICLiv_facts.update({'latexexpr': r'\GICLiv'})
        GICLiv_facts.update({'urlhandle': urlroot+'GICLiv'})
        GICLiv_facts.update({'py___code': 'test: GPFLiv < 1'})
        solution.facts.update({'GICLiv': GICLiv_facts})
        solution.GICLiv_facts = GICLiv_facts

        PF_RNrm_facts = {
            'about': 'Growth-Normalized Perfect Foresight Return Factor'}
        PF_RNrm = Rfree/PermGroNum
        PF_RNrm_facts.update({'latexexpr': r'\PF_RNrm'})
        PF_RNrm_facts.update({'_unicode_': r'R/Γ'})
        PF_RNrm_facts.update({'py___code': r'Rfree/PermGroNum'})
        PF_RNrm_facts.update({'value_now': PF_RNrm})
        solution.facts.update({'PF_RNrm': PF_RNrm_facts})
        solution.PF_RNrm_facts = PF_RNrm_facts
        solution.PF_RNrm = PF_RNrm

        Inv_PF_RNrm_facts = {
            'about': 'Inverse of Growth-Normalized Perfect Foresight Return Factor'}
        Inv_PF_RNrm = 1/PF_RNrm
        Inv_PF_RNrm_facts.update({'latexexpr': r'\Inv_PF_RNrm'})
        Inv_PF_RNrm_facts.update({'_unicode_': r'Γ/R'})
        Inv_PF_RNrm_facts.update({'py___code': r'PermGroNumInd/Rfree'})
        Inv_PF_RNrm_facts.update({'value_now': Inv_PF_RNrm})
        solution.facts.update({'Inv_PF_RNrm': Inv_PF_RNrm_facts})
        solution.Inv_PF_RNrm_facts = Inv_PF_RNrm_facts
        solution.Inv_PF_RNrm = Inv_PF_RNrm

        FHWF_facts = {
            'about': 'Finite Human Wealth Factor'}
        FHWF = PermGroNum/Rfree
        FHWF_facts.update({'latexexpr': r'\FHWF'})
        FHWF_facts.update({'_unicode_': r'R/Γ'})
        FHWF_facts.update({'urlhandle': urlroot+'FHWF'})
        FHWF_facts.update({'py___code': r'PermGroNumInf/Rfree'})
        FHWF_facts.update({'value_now': FHWF})
        solution.facts.update({'FHWF': FHWF_facts})
        solution.FHWF_facts = FHWF_facts
        solution.FHWF = FHWF

        FHWC_facts = {'about': 'Finite Human Wealth Condition'}
        FHWC_facts.update({'latexexpr': r'\FHWC'})
        FHWC_facts.update({'urlhandle': urlroot+'FHWC'})
        FHWC_facts.update({'py___code': 'test: agent.FHWF < 1'})
        solution.facts.update({'FHWC': FHWC_facts})
        solution.FHWC_facts = FHWC_facts

        hNrmNowInf_facts = {'about':
                            'Human wealth for infinite horizon consumer'}
        hNrmNowInf = float('inf')  # default to infinity
        if FHWF < 1:  # If it is finite, set it to its value
            hNrmNowInf = 1/(1-FHWF)

        solution.hNrmNowInf = hNrmNowInf
        hNrmNowInf_facts = dict({'latexexpr': '1/(1-\FHWF)'})
        hNrmNowInf_facts.update({'value_now': hNrmNowInf})
        hNrmNowInf_facts.update({
            'py___code': '1/(1-FHWF)'})
        solution.facts.update({'hNrmNowInf': hNrmNowInf_facts})
        solution.hNrmNowInf_facts = hNrmNowInf_facts
        # solution.hNrmNowInf = hNrmNowInf

        DiscGPFRawCusp_facts = {
            'about': 'DiscFac s.t. GPFRaw = 1'}
        solution.DiscGPFRawCusp = DiscGPFRawCusp = \
            ((PermGroNum) ** (CRRA)) / (Rfree)
        DiscGPFRawCusp_facts.update({'latexexpr': ''})
        DiscGPFRawCusp_facts.update({'value_now': DiscGPFRawCusp})
        DiscGPFRawCusp_facts.update({
            'py___code': '( PermGroNum                       ** CRRA)/(Rfree)'})
        solution.facts.update({'DiscGPFRawCusp': DiscGPFRawCusp_facts})
        solution.DiscGPFRawCusp_facts = DiscGPFRawCusp_facts

        DiscGPFLivCusp_facts = {
            'about': 'DiscFac s.t. GPFLiv = 1'}
        solution.DiscGPFLivCusp = DiscGPFLivCusp = ((PermGroNum) ** (CRRA)) \
            / (Rfree * LivNum)
        DiscGPFLivCusp_facts.update({'latexexpr': ''})
        DiscGPFLivCusp_facts.update({'value_now': DiscGPFLivCusp})
        DiscGPFLivCusp_facts.update({
            'py___code': '( PermGroNum                       ** CRRA)/(Rfree*LivNum)'})
        solution.facts.update({'DiscGPFLivCusp': DiscGPFLivCusp_facts})
        solution.DiscGPFLivCusp_facts = DiscGPFLivCusp_facts

        FVAF_facts = {'about': 'Finite Value of Autarky Factor'}
        solution.FVAF = FVAF = LivNum * DiscFacEff * uInv_Ex_uInv_PermShk
        FVAF_facts.update({'latexexpr': r'\FVAFPF'})
        FVAF_facts.update({'urlhandle': urlroot+'FVAFPF'})
        solution.facts.update({'FVAF': FVAF_facts})
        solution.FVAF_facts = FVAF_facts

        FVAC_facts = {'about': 'Finite Value of Autarky Condition - Perfect Foresight'}
        FVAC_facts.update({'latexexpr': r'\FVACPF'})
        FVAC_facts.update({'urlhandle': urlroot+'FVACPF'})
        FVAC_facts.update({'py___code': 'test: FVAFPF < 1'})
        solution.facts.update({'FVAC': FVAC_facts})
        solution.FVAC_facts = FVAC_facts

        # Calculate objects whose values are built up recursively from
        # prior period's values

        hNrmNow = (
            (PermGroNum / Rfree) * (1.0 + self.solution_next.hNrmNow)
        )
        hNrmNow = PermGroNum/Rfree
        hNrmNow_facts = {'about': 'Human Wealth Now'}
        hNrmNow_facts.update({'latexexpr': r'\hNrmNow'})
#        hNrmNow_facts.update({'_unicode_': r'R/Γ'})
#        hNrmNow_facts.update({'urlhandle': urlroot+'hNrmNow'})
#        hNrmNow_facts.update({'py___code': r'PermGroNumInf/Rfree'})
#        hNrmNow_facts.update({'value_now': hNrmNow})
        solution.facts.update({'hNrmNow': hNrmNow_facts})
        solution.hNrmNow_facts = hNrmNow_facts
        self.hNrmNow = solution.hNrmNow = hNrmNow

        MPCminNow = 1.0 / (1.0 + RPF / self.solution_next.MPCminNow)
        MPCminNow_facts = {
            'about': 'Minimal MPC as m -> infty'}
        MPCminNow_facts.update({'latexexpr': r''})
        MPCminNow_facts.update({'urlhandle': urlroot+'MPCminNow'})
        MPCminNow_facts.update({'value_now': MPCminNow})
        solution.facts.update({'MPCminNow': MPCminNow_facts})
        solution.MPCminNow_facts = MPCminNow_facts
        self.MPCminNow = solution.MPCminNow = MPCminNow

        MPCmaxNow = 1.0 / (1.0 + (0.0 ** (1.0 / CRRA)) * RPF
                           / self.solution_next.MPCmaxNow)
        MPCmaxNow_facts = {
            'about': 'Maximal MPC in current period as m -> minimum'}
#        MPCmaxNow_facts.update({'latexexpr': r''})
        MPCmaxNow_facts.update({'urlhandle': urlroot+'MPCmaxNow'})
        MPCmaxNow_facts.update({'value_now': MPCmaxNow})
        solution.facts.update({'MPCmaxNow': MPCmaxNow_facts})
        solution.MPCmaxNow_facts = MPCmaxNow_facts
        solution.MPCmaxNow = MPCmaxNow

        # Lower bound of aggregate wealth growth if all inheritances squandered
        cFuncLimitIntercept = MPCminNow * hNrmNow
        cFuncLimitIntercept_facts = {
            'about': 'Vertical intercept of perfect foresight consumption function'}
        cFuncLimitIntercept_facts.update({'latexexpr': '\MPC '})
        cFuncLimitIntercept_facts.update({'urlhandle': ''})
        cFuncLimitIntercept_facts.update({'value_now': cFuncLimitIntercept})
        cFuncLimitIntercept_facts.update({
            'py___code': 'MPCminNow * hNrmNow'})
        solution.facts.update({'cFuncLimitIntercept': cFuncLimitIntercept_facts})
        solution.cFuncLimitIntercept_facts = cFuncLimitIntercept_facts
        solution.cFuncLimitIntercept = cFuncLimitIntercept

        cFuncLimitSlope = MPCminNow
        cFuncLimitSlope_facts = {
            'about': 'Slope of limiting consumption function'}
        cFuncLimitSlope_facts = dict({'latexexpr': '\MPC \hNrmNow'})
        cFuncLimitSlope_facts.update({'urlhandle': ''})
        cFuncLimitSlope_facts.update({'value_now': cFuncLimitSlope})
        cFuncLimitSlope_facts.update({
            'py___code': 'MPCminNow * hNrmNow'})
        solution.facts.update({'cFuncLimitSlope': cFuncLimitSlope_facts})
        solution.cFuncLimitSlope_facts = cFuncLimitSlope_facts
        solution.cFuncLimitSlope = cFuncLimitSlope

        # We are in the perfect foresight model now so these are all 1.0
        Ex_Inv_PermShk = 1.0
        Inv_Ex_PermShk_Inv = 1.0
        Ex_uInv_PermShk = 1.0
        uInv_Ex_uInv_PermShk = 1.0

        # These formulae do not require "live" computation of expectations
        # from a distribution that is on hand.  So, having constructed
        # expected values above, we can use those below.

        # This allows sharing these formulae between the perfect foresight
        # and the non-perfect-foresight models.  They are constructed here
        # and inherited by the descendant model, which augments them with
        # the objects that require live calculation.

        if solution.Inv_PF_RNrm < 1:        # Finite if Rfree > PermGroNumFac
            solution.hNrmNowInf = 1/(1-solution.Inv_PF_RNrm)

        # Given m, value of c where E[mLev_{t+1}/mLev_{t}]=PermGroNumFac
        # Solves for c in equation at url/#balgrostable

        solution.c_where_Ex_PermShk_times_mtp1_minus_mt_eq_0 = (
            lambda m_t:
            m_t * (1 - solution.Inv_PF_RNrm) + solution.Inv_PF_RNrm
        )

        solution.Ex_cLev_tp1_Over_cLev_t_from_mt = (
            lambda m_t:
            solution.Ex_cLev_tp1_Over_pLev_t_from_mt(solution,
                                                     m_t - solution.cFunc(m_t))
            / solution.cFunc(m_t)
        )

#        # E[m_{t+1} pLev_{t+1}/pLev_{t}] as a fn of a_{t}
        solution.Ex_mLev_tp1_Over_pLev_t_from_at = (
            lambda a_t:
                solution.PermGroNumFac *
            (solution.PF_RNrm * a_t + solution.Ex_IncNextNrm)
        )

        # E[m_{t+1} pLev_{t+1}/(m_{t}pLev_{t})] as a fn of m_{t}
        solution.Ex_mLev_tp1_Over_mLev_t_from_at = (
            lambda m_t:
                solution.Ex_mLev_tp1_Over_pLev_t_from_at(solution,
                                                         m_t-solution.cFunc(m_t)
                                                         )/m_t
        )

        return solution

    def set_and_update_values(self, solution_next, IncShkDstn, LivPrb, DiscFac):
        """
        Unpacks some of the inputs (and calculates simple objects based on them),
        storing the results in self for use by other methods.  These include:
        income shocks and probabilities, next period's marginal value function
        (etc), the probability of getting the worst income shock next period,
        the patience factor, human wealth, and the bounding MPCs.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncShkDstn : distribution.DiscreteDistribution
            A DiscreteDistribution with a pmf
            and two point value arrays in X, order:
            permanent shocks, transitory shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.
        DiscFac : float
            Intertemporal discount factor for future utility.

        Returns
        -------
        None
        """
        self.DiscFacEff = DiscFac * LivPrb  # "effective" discount factor
        self.IncShkDstn = IncShkDstn
        self.ShkPrbsNext = IncShkDstn.pmf
        self.PermShkValsNext = IncShkDstn.X[0]
        self.TranShkValsNext = IncShkDstn.X[1]
        self.PermShkMinNext = np.min(self.PermShkValsNext)
        self.TranShkMinNext = np.min(self.TranShkValsNext)
        self.vPfuncNext = solution_next.vPfunc
        self.WorstIncPrb = np.sum(
            self.ShkPrbsNext[
                (self.PermShkValsNext * self.TranShkValsNext)
                == (self.PermShkMinNext * self.TranShkMinNext)
            ]
        )

        if self.CubicBool:
            self.vPPfuncNext = solution_next.vPPfunc

        if self.vFuncBool:
            self.vFuncNext = solution_next.vFunc

        # Update the bounding MPCs and PDV of human wealth:
        self.RPF = ((self.Rfree * self.DiscFacEff) ** (1.0 / self.CRRA)) / self.Rfree
        self.MPCminNow = 1.0 / (1.0 + self.RPF / solution_next.MPCminNow)
        self.Ex_IncNext = np.dot(
            self.ShkPrbsNext, self.TranShkValsNext * self.PermShkValsNext
        )
        self.hNrmNow = (
            self.PermGroFac / self.Rfree * (self.Ex_IncNext + solution_next.hNrmNow)
        )
        self.MPCmaxNow = 1.0 / (
            1.0
            + (self.WorstIncPrb ** (1.0 / self.CRRA))
            * self.RPF
            / solution_next.MPCmaxNow
        )

        self.cFuncLimitIntercept = self.MPCminNow * self.hNrmNow
        self.cFuncLimitSlope = self.MPCminNow

    def def_utility_funcs(self):
        """
        Defines CRRA utility function for this period (and its derivatives),
        saving them as attributes of self for other methods to use.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.u = lambda c: utility(c, gam=self.CRRA)  # utility function
        self.uP = lambda c: utilityP(c, gam=self.CRRA)  # marginal utility function
        self.uPP = lambda c: utilityPP(
            c, gam=self.CRRA
        )  # marginal marginal utility function

    def def_value_funcs(self):
        """
        Defines the value and marginal value functions for this period.
        mNrmMin.  See PerfForesightConsumerType.ipynb
        for a brief explanation and the links below for a fuller treatment.
        https://github.com/llorracc/SolvingMicroDSOPs/#vFuncPF

        Parameters
        ----------
        None

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
        vFuncNvrsSlope = self.MPCminNow ** (-self.CRRA / (1.0 - self.CRRA))
        vFuncNvrs = LinearInterp(
            np.array([self.mNrmMinNow, self.mNrmMinNow + 1.0]),
            np.array([0.0, vFuncNvrsSlope]),
        )
        self.vFunc = ValueFuncCRRA(vFuncNvrs, self.CRRA)
        self.vPfunc = MargValueFuncCRRA(self.cFunc, self.CRRA)

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
        # Use local value of BoroCnstArt to prevent comparing None and float
        if self.BoroCnstArt is None:
            BoroCnstArt = -np.inf
        else:
            BoroCnstArt = self.BoroCnstArt

        # Calculate human wealth this period
        self.hNrmNow = (self.PermGroFac / self.Rfree) * (self.solution_next.hNrmNow + 1.0)

        # Calculate the lower bound of the MPC
        RPF = ((self.Rfree * self.DiscFacEff) ** (1.0 / self.CRRA)) / self.Rfree
        self.MPCminNow = 1.0 / (1.0 + RPF / self.solution_next.MPCminNow)

        # Extract kink points in next period's consumption function;
        # don't take the last one; it only defines extrapolation, is not kink.
        mNrmNext = self.solution_next.cFunc.x_list[:-1]
        cNrmNext = self.solution_next.cFunc.y_list[:-1]

        # Calculate the end-of-period asset values that would reach those kink points
        # next period, then invert the first order condition to get consumption. Then
        # find the endogenous gridpoint (kink point) today that corresponds to each kink
        aNrmNow = (self.PermGroFac / self.Rfree) * (mNrmNext - 1.0)
        cNrmNow = (self.DiscFacEff * self.Rfree) ** (-1.0 / self.CRRA) * (
            self.PermGroFac * cNrmNext
        )
        mNrmNow = aNrmNow + cNrmNow

        # Add an additional point to the list of gridpoints for the extrapolation,
        # using the new value of the lower bound of the MPC.
        mNrmNow = np.append(mNrmNow, mNrmNow[-1] + 1.0)
        cNrmNow = np.append(cNrmNow, cNrmNow[-1] + self.MPCminNow)
        # If the artificial borrowing constraint binds, combine the constrained and
        # unconstrained consumption functions.
        if BoroCnstArt > mNrmNow[0]:
            # Find the highest index where constraint binds
            cNrmCnst = mNrmNow - BoroCnstArt
            CnstBinds = cNrmCnst < cNrmNow
            idx = np.where(CnstBinds)[0][-1]
            if idx < (mNrmNow.size - 1):
                # If it is not the *very last* index, find the the critical level
                # of mNrm where the artificial borrowing contraint begins to bind.
                d0 = cNrmNow[idx] - cNrmCnst[idx]
                d1 = cNrmCnst[idx + 1] - cNrmNow[idx + 1]
                m0 = mNrmNow[idx]
                m1 = mNrmNow[idx + 1]
                alpha = d0 / (d0 + d1)
                mCrit = m0 + alpha * (m1 - m0)
                # Adjust the grids of mNrm and cNrm to account for the borrowing constraint.
                cCrit = mCrit - BoroCnstArt
                mNrmNow = np.concatenate(([BoroCnstArt, mCrit], mNrmNow[(idx + 1):]))
                cNrmNow = np.concatenate(([0.0, cCrit], cNrmNow[(idx + 1):]))
            else:
                # If it *is* the last index, then there are only three points
                # that characterize the c function: the artificial borrowing
                # constraint, the constraint kink, and the extrapolation point.
                mXtra = (cNrmNow[-1] - cNrmCnst[-1]) / (1.0 - self.MPCminNow)
                mCrit = mNrmNow[-1] + mXtra
                cCrit = mCrit - BoroCnstArt
                mNrmNow = np.array([BoroCnstArt, mCrit, mCrit + 1.0])
                cNrmNow = np.array([0.0, cCrit, cCrit + self.MPCminNow])
                # If the mNrm and cNrm grids have become too large, throw out the last
                # kink point, being sure to adjust the extrapolation.
        if mNrmNow.size > self.MaxKinks:
            mNrmNow = np.concatenate((mNrmNow[:-2], [mNrmNow[-3] + 1.0]))
            cNrmNow = np.concatenate((cNrmNow[:-2], [cNrmNow[-3] + self.MPCminNow]))
            # Construct the consumption function as a linear interpolation.
        self.cFunc = LinearInterp(mNrmNow, cNrmNow)
        # Calculate the upper bound of the MPC as the slope of the bottom segment.
        self.MPCmaxNow = (cNrmNow[1] - cNrmNow[0]) / (mNrmNow[1] - mNrmNow[0])

        # Add two attributes to enable calculation of steady state market resources.
        self.Ex_IncNextNrm = 1.0  # Perfect foresight income of 1
        self.mNrmMinNow = mNrmNow[0]  # Relabeling for compatibility with add_mNrmStE

    def add_stable_points(self, solution):
        """
        Checks necessary conditions for the existence of the individual pseudo
        steady state StE and target Trg levels of market resources.
        If the conditions are satisfied, computes and adds the stable points
        to the solution.

        Parameters
        ----------
        solution : ConsumerSolution
            Solution to this period's problem, which must have attribute cFunc.
        Returns
        -------
        solution : ConsumerSolution
            Same solution that was provided, augmented with attributes mNrmStE and
            mNrmTrg, if they exist.

        """
        # This is the version for perfect foresight model; models that
        # inherit from the PF model will replace it with suitable alternatives
        # For the PF model:
        # 0. There is no non-degenerate steady state without constraints
        # 1. There is a non-degenerate SS for constrained PF model if GICRaw holds.
        # Therefore
        # Check if  (GICRaw and BoroCnstArt) and if so compute them both
        APF = (self.Rfree*self.DiscFacEff)**(1/self.CRRA)
        GICRaw = 1 > APF/self.PermGroFac
        if self.BoroCnstArt is not None and GICRaw:
            # Result of borrowing max allowed
            bNrmNxt = -self.BoroCnstArt*self.Rfree/self.PermGroFac
            solution.mNrmStE = self.Ex_IncNextNrm-bNrmNxt
            solution.mNrmTrg = self.Ex_IncNextNrm-bNrmNxt
        else:
            _log.warning("The unconstrained PF model solution is degenerate")
            if GICRaw:
                if self.Rfree > self.PermGroFac:  # impatience drives wealth to minimum
                    solution.mNrmStE = -(1/(1-self.PermGroFac/self.Rfree))
                    solution.mNrmTrg = -(1/(1-self.PermGroFac/self.Rfree))
                else:  # patience drives wealth to infinity
                    _log.warning(
                        "Pathological patience plus infinite human wealth: solution undefined")
                    solution.mNrmStE = float('NaN')
                    solution.mNrmTrg = float('NaN')
            else:
                solution.mNrmStE = float('inf')
                solution.mNrmTrg = float('inf')
        return solution

    def solve(self):
        """
        Solves the one period perfect foresight consumption-saving problem.

        Parameters
        ----------
        None

        Returns
        -------
        solution : ConsumerSolution
            The solution to this period's problem.
        """
        self.def_utility_funcs()
        self.DiscFacEff = self.DiscFac * self.LivPrb  # Effective=pure x LivPrb
        self.make_cFunc_PF()
        self.def_value_funcs()

        solution = ConsumerSolutionWithEx(
            cFunc=self.cFunc,
            vFunc=self.vFunc,
            vPfunc=self.vPfunc,
            mNrmMin=self.mNrmMinNow,
            hNrmNow=self.hNrmNow,
            MPCminNow=self.MPCminNow,
            MPCmaxNow=self.MPCmaxNow,
        )
        # Oddly, though the value and consumption functions were included in the solution,
        # and the inverse utlity function and its derivatives, the baseline setup did not
        # include the utility function itself.  This should be fixed more systematically,
        # but for now what is done below will work
        solution.u = self.u
        solution.uP = self.uP
        solution.uPP = self.uPP

        return solution

    # @staticmethod
    def solver_check_condition(self, name, test, messages, verbose, verbose_messages=None):
        """
        Checks one condition.

        Parameters
        ----------
        name : string
             Name for the condition.

        test : function(self -> boolean)
             A function (of self) which tests the condition

        messages : dict{boolean : string}
            A dictiomary with boolean keys containing values
            for messages to print if the condition is
            true or false.

        verbose_messages : dict{boolean : string}
            (Optional) A dictiomary with boolean keys containing values
            for messages to print if the condition is
            true or false under verbose printing.
        """
        self.conditions[name] = test(self)
        set_verbosity_level((4 - verbose) * 10)
        _log.info(messages[self.conditions[name]].format(self))
        if verbose_messages:
            _log.debug(verbose_messages[self.conditions[name]].format(self))

    # @staticmethod
    def solver_check_condtnnew(self, name, test, messages, verbose, verbose_messages, fact):
        """
        Checks one condition.

        Parameters
        ----------
        name : string
             Name for the condition.

        fact : string
             Name of the factor whose value should be less than 1

        test : function(self -> boolean)
             A function (of self) which tests the condition

        messages : dict{boolean : string}
            A dictiomary with boolean keys containing values
            for messages to print if the condition is
            true or false.

        verbose_messages : dict{boolean : string}
            (Optional) A dictiomary with boolean keys containing values
            for messages to print if the condition is
            true or false under verbose printing.
        """
        self.solution_curr.conditions[name] = test(self)
        set_verbosity_level((4 - verbose) * 10)
        self.solution_curr.conditions[fact] = (
            messages[self.solution_curr.conditions[name]] +
            verbose_messages[self.solution_curr.conditions[name]]).format(self.solution_curr)
        print(self.solution_curr.conditions[fact])
        _log.info((
            messages[self.solution_curr.conditions[name]] +
            verbose_messages[self.solution_curr.conditions[name]]).format(self.solution_curr)
        )

    # @staticmethod
    def solver_check_AIC(self, verbose=None):
        """
        Evaluate and report on the Absolute Impatience Condition
        """
        name = "AIC"
        fact = "APF"

        def test(agent): return agent.solution_curr.APF < 1

        messages = {
            True: "\nThe Absolute Patience Factor for the supplied parameter values, APF={0.APF}, satisfies the Absolute Impatience Condition (AIC), which requires APF < 1: "+self.solution_curr.AIC_facts['urlhandle'],
            False: "\nThe Absolute Patience Factor for the supplied parameter values, APF={0.APF}, violates the Absolute Impatience Condition (AIC), which requires APF < 1: "+self.solution_curr.AIC_facts['urlhandle']
        }
        verbose_messages = {
            True: "  Because the APF < 1,  the absolute amount of consumption is expected to fall over time.  \n",
            False: "  Because the APF > 1, the absolute amount of consumption is expected to grow over time.  \n",
        }
        verbose = self.verbose if verbose is None else verbose
        self.check_condtnnew(name, test, messages, verbose, verbose_messages, fact)

    # @staticmethod
    def solver_check_FVAC(self, verbose=None):
        """
        Evaluate and report on the Finite Value of Autarky Condition
        """
        name = "FVAC"
        fact = "FVAF"

        def test(agent): return agent.solution_curr.FVAF < 1

        messages = {
            True: "\nThe Finite Value of Autarky Factor for the supplied parameter values, FVAF={0.FVAF}, satisfies the Finite Value of Autarky Condition, which requires FVAF < 1: "+self.solution_curr.FVAC_facts['urlhandle'],
            False: "\nThe Finite Value of Autarky Factor for the supplied parameter values, FVAF={0.FVAF}, violates the Finite Value of Autarky Condition, which requires FVAF: "+self.solution_curr.FVAC_facts['urlhandle']
        }
        verbose_messages = {
            True: "  Therefore, a nondegenerate solution exists if the RIC also holds. ("+self.solution_curr.FVAC_facts['urlhandle']+")\n",
            False: "  Therefore, a nondegenerate solution exits if the RIC holds.\n",
        }

        verbose = self.verbose if verbose is None else verbose
        self.check_condtnnew(name, test, messages, verbose, verbose_messages, fact)

    # @staticmethod
    def solver_check_GICRaw(self, verbose=None):
        """
        Evaluate and report on the Growth Impatience Condition
        """
        name = "GICRaw"
        fact = "GPFRaw"

        def test(agent): return agent.solution_curr.GPFRaw < 1

        messages = {
            True: "\nThe Growth Patience Factor for the supplied parameter values, GPF={0.GPFRaw}, satisfies the Growth Impatience Condition (GIC), which requires GPF < 1: "+self.solution_curr.GICRaw_facts['urlhandle'],
            False: "\nThe Growth Patience Factor for the supplied parameter values, GPF={0.GPFRaw}, violates the Growth Impatience Condition (GIC), which requires GPF < 1: "+self.solution_curr.GICRaw_facts['urlhandle'],
        }
        verbose_messages = {
            True: "  Therefore,  for a perfect foresight consumer, the ratio of individual wealth to permanent income is expected to fall indefinitely.    \n",
            False: "  Therefore, for a perfect foresight consumer, the ratio of individual wealth to permanent income is expected to rise toward infinity. \n"
        }

        verbose = self.verbose if verbose is None else verbose
        self.check_condtnnew(name, test, messages, verbose, verbose_messages, fact)

    # @staticmethod
    def solver_check_GICLiv(self, verbose=None):
        name = "GICLiv"
        fact = "GPFLiv"

        def test(agent): return agent.solution_curr.GPFLiv < 1

        messages = {
            True: "\nThe Mortality Adjusted Aggregate Growth Patience Factor for the supplied parameter values, GPFLiv={0.GPFLiv}, satisfies the Mortality Adjusted Aggregate Growth Imatience Condition (GICLiv): "+self.solution_curr.GPFLiv_facts['urlhandle'],
            False: "\nThe Mortality Adjusted Aggregate Growth Patience Factor for the supplied parameter values, GPFLiv={0.GPFLiv}, violates the Mortality Adjusted Aggregate Growth Imatience Condition (GICLiv): "+self.solution_curr.GPFLiv_facts['urlhandle'],
        }
        verbose_messages = {
            True: "  Therefore, a target level of the ratio of aggregate market resources to aggregate permanent income exists ("+self.solution_curr.GPFLiv_facts['urlhandle']+")\n",
            False: "  Therefore, a target ratio of aggregate resources to aggregate permanent income may not exist ("+self.solution_curr.GPFLiv_facts['urlhandle']+")\n",
        }
        verbose = self.verbose if verbose is None else verbose
        self.check_condtnnew(name, test, messages, verbose, verbose_messages, fact)

    # @staticmethod
    def solver_check_RIC(self, verbose=None):
        """
        Evaluate and report on the Return Impatience Condition
        """

        name = "RIC"
        fact = "RPF"

        def test(agent): return agent.solution_curr.RPF < 1

        messages = {
            True: "\nThe Return Patience Factor for the supplied parameter values, RPF= {0.RPF}, satisfies the Return Impatience Condition (RIC), which requires RPF < 1: "+self.solution_curr.RPF_facts['urlhandle'],
            False: "\nThe Return Patience Factor for the supplied parameter values, RPF= {0.RPF}, violates the Return Impatience Condition (RIC), which requires RPF < 1: "+self.solution_curr.RPF_facts['urlhandle'],
        }
        verbose_messages = {
            True: "  Therefore, the limiting consumption function is not c(m)=0 for all m\n",
            False: "  Therefore, if the FHWC is satisfied, the limiting consumption function is c(m)=0 for all m.\n",
        }
        verbose = self.verbose if verbose is None else verbose
        self.check_condtnnew(name, test, messages, verbose, verbose_messages, fact)

    # @staticmethod
    def solver_check_FHWC(self, verbose=None):
        """
        Evaluate and report on the Finite Human Wealth Condition
        """
        name = "FHWC"
        fact = "FHWF"

        def test(agent): return agent.solution_curr.FHWF < 1

        messages = {
            True: "\nThe Finite Human Wealth Factor value for the supplied parameter values, FHWF={0.FHWF}, satisfies the Finite Human Wealth Condition (FHWC), which requires FHWF < 1: "+self.solution_curr.FHWC_facts['urlhandle'],
            False: "\nThe Finite Human Wealth Factor value for the supplied parameter values, FHWF={0.FHWF}, violates the Finite Human Wealth Condition (FHWC), which requires FHWF < 1: "+self.solution_curr.FHWC_facts['urlhandle'],
        }
        verbose_messages = {
            True: "  Therefore, the limiting consumption function is not c(m)=Infinity ("+self.solution_curr.FHWC_facts['urlhandle']+")\n  Human wealth normalized by permanent income is {0.hNrmNowInf}.\n",
            False: "  Therefore, the limiting consumption function is c(m)=Infinity for all m unless the RIC is also violated.\n  If both FHWC and RIC fail and the consumer faces a liquidity constraint, the limiting consumption function is nondegenerate but has a limiting slope of 0. ("+self.solution_curr.FHWC_facts['urlhandle']+")\n",
        }
        verbose = self.verbose if verbose is None else verbose
        self.check_condtnnew(name, test, messages, verbose, verbose_messages, fact)

    # @staticmethod
    def solver_check_condtnsnew(self, verbose=None):
        """
        This method checks whether the instance's type satisfies the
        Absolute Impatience Condition (AIC),
        the Return Impatience Condition (RIC),
        the Finite Human Wealth Condition (FHWC), the perfect foresight
        model's Growth Impatience Condition (GICRaw) and
        Perfect Foresight Finite Value of Autarky Condition (FVACPF). Depending on the configuration of parameter values, some
        combination of these conditions must be satisfied in order for the problem to have
        a nondegenerate solution. To check which conditions are required, in the verbose mode
        a reference to the relevant theoretical literature is made.

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
        self.solution_curr.conditions = {}

        self.solution_curr.violated = False

        # This method only checks for the conditions for infinite horizon models
        # with a 1 period cycle. If these conditions are not met, we exit early.
        if self.cycles != 0 or self.T_cycle > 1:
            return

        verbose = self.verbose if verbose is None else verbose
        self.check_AIC(verbose)
        self.check_FHWC(verbose)
        self.check_RIC(verbose)
        self.check_GICRaw(verbose)
        self.check_GICLiv(verbose)
        self.check_FVAC(verbose)

        if hasattr(self, "BoroCnstArt") and self.solution_curr.BoroCnstArt is not None:
            self.solution_curr.violated = not self.solution_curr.conditions["RIC"]
        else:
            self.solution_curr.violated = not self.solution_curr.conditions[
                "RIC"] or not self.solution_curr.conditions["FHWC"]


###############################################################################
###############################################################################
class ConsIndShockSetup(ConsPerfForesightSolver):
    """
    A superclass for solvers of one period consumption-saving problems with
    constant relative risk aversion utility and permanent and transitory shocks
    to income.  Has methods to set up but not solve the one period problem.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncShkDstn : distribution.Distribution
        A discrete approximation to the income process between the period
        being solved and the one immediately following (in solution_next).
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
        included in the reported solution.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear inter-
        polation.
    """

    def solution_add_further_info_ConsPerfForesightSolver(self, solution):
        super().solution_add_further_info(solution)

    def __init__(
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
    ):
        # Generate a url that will locate the documentation
        self.url_doc = "https://hark.readthedocs.io/en/latest/search.html?q=" + \
            self.__class__.__name__+"&check_keywords=yes&area=default#"

        # url for paper that contains various theoretical results
        self.url_ref = "https://econ-ark.github.io/BufferStockTheory"
        self.solution_next = solution_next
        self.IncShkDstn = IncShkDstn
        self.LivPrb = LivPrb
        self.DiscFac = DiscFac
        self.CRRA = CRRA
        self.Rfree = Rfree
        self.PermGroFac = PermGroFac
        self.BoroCnstArt = BoroCnstArt
        self.aXtraGrid = aXtraGrid
        self.vFuncBool = vFuncBool
        self.CubicBool = CubicBool
        self.facts = {}
        self.def_utility_funcs()

    def def_utility_funcs(self):
        """
        Defines CRRA utility function for this period (and its derivatives,
        and their inverses), saving them as attributes of self for other methods
        to use.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        ConsPerfForesightSolver.def_utility_funcs(self)
        self.uPinv = lambda u: utilityP_inv(u, gam=self.CRRA)
        self.uPinvP = lambda u: utilityP_invP(u, gam=self.CRRA)
        self.uinvP = lambda u: utility_invP(u, gam=self.CRRA)

        if self.vFuncBool:
            self.uinv = lambda u: utility_inv(u, gam=self.CRRA)

    def solution_add_further_info(self, solution):
        # self here is the solver, which knows info about the problem from the agent
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
        solution.urlroot = urlroot = self.url_ref+'/#'

#        solution.parameters_model = self.parameters

        if not hasattr(solution, 'facts'):
            solution.facts = {}

        # The Xref versions are full rank, for direct multiplication by probs
        PermShkValsXref = solution.PermShkValsXref = self.IncShkDstn.X[0]
        TranShkValsXref = solution.TranShkValsXref = self.IncShkDstn.X[1]
        ShkPrbsNext = solution.ShkPrbsNext = self.IncShkDstn.pmf

        solution.Rfree = Rfree = self.Rfree
        solution.DiscFac = DiscFac = self.DiscFac
        PermGroNum = solution.PermGroNum = self.PermGroFac
        solution.LivNum = LivNum = self.LivPrb
        solution.DiscFacEff = DiscFacEff = DiscFac * LivNum
        solution.CRRA = CRRA = self.CRRA
        solution.UnempPrb = UnempPrb = self.IncShkDstn.params['UnempPrb']
        solution.UnempPrbRet = UnempPrbRet = self.IncShkDstn.params['UnempPrbRet']

        # Facts that apply in the perfect foresight case should already have been added
        # Conveniently instantiate the income shocks

        PermShkMinNext = np.min(PermShkValsXref)
        TranShkMinNext = np.min(TranShkValsXref)
        WorstIncPrb = np.sum(
            ShkPrbsNext[
                (PermShkValsXref * TranShkValsXref)
                == (PermShkMinNext * TranShkMinNext)
            ]
        )

        # Many other facts will have been inherited from the perfect foresight
        # model of which this model, as a descendant, has already inherited
        # Here we need compute only those objects whose value changes when
        # the shock distributions are nondegenerate.
        Ex_IncNextNrm_facts = {
            'about': 'Expected income next period'}
        solution.Ex_IncNextNrm = Ex_IncNextNrm = np.dot(
            ShkPrbsNext, TranShkValsXref * PermShkValsXref).item()
        Ex_IncNextNrm_facts.update({'latexexpr': r'\Ex_IncNextNrm'})
        Ex_IncNextNrm_facts.update({'_unicode_': r'R/Γ'})
        Ex_IncNextNrm_facts.update({'urlhandle': urlroot+'ExIncNextNrm'})
        Ex_IncNextNrm_facts.update(
            {'py___code': r'np.dot(ShkPrbsNext,TranShkValsXref*PermShkValsXref)'})
        Ex_IncNextNrm_facts.update({'value_now': Ex_IncNextNrm})
        solution.facts.update({'Ex_IncNextNrm': Ex_IncNextNrm_facts})
        solution.Ex_IncNextNrm_facts = Ex_IncNextNrm_facts


#        Ex_Inv_PermShk = calc_expectation(            PermShkDstn[0], lambda x: 1 / x        )
        Ex_Inv_PermShk_facts = {
            'about': 'Expectation of Inverse of Permanent Shock'}
        solution.Ex_Inv_PermShk = Ex_Inv_PermShk = np.dot(1/PermShkValsXref, ShkPrbsNext)
        Ex_Inv_PermShk_facts.update({'latexexpr': r'\Ex_Inv_PermShk'})
#        Ex_Inv_PermShk_facts.update({'_unicode_': r'R/Γ'})
        Ex_Inv_PermShk_facts.update({'urlhandle': urlroot+'ExInvPermShk'})
        Ex_Inv_PermShk_facts.update({'py___code': r'Rfree/PermGroFacAdj'})
        Ex_Inv_PermShk_facts.update({'value_now': Ex_Inv_PermShk})
        solution.facts.update({'Ex_Inv_PermShk': Ex_Inv_PermShk_facts})
        solution.Ex_Inv_PermShk_facts = Ex_Inv_PermShk_facts
        # solution.Ex_Inv_PermShk = Ex_Inv_PermShk

        Inv_Ex_Inv_PermShk_facts = {
            'about': 'Inverse of Expectation of Inverse of Permanent Shock'}
        solution.Inv_Ex_Inv_PermShk = Inv_Ex_Inv_PermShk = 1/Ex_Inv_PermShk
        Inv_Ex_Inv_PermShk_facts.update(
            {'latexexpr': '\InvExInvPermShk=\left(\Ex[\PermShk^{-1}]\right)^{-1}'})
#        Inv_Ex_Inv_PermShk_facts.update({'_unicode_': r'R/Γ'})
        Inv_Ex_Inv_PermShk_facts.update({'urlhandle': urlroot+'InvExInvPermShk'})
        Inv_Ex_Inv_PermShk_facts.update({'py___code': r'1/Ex_Inv_PermShk'})
        Inv_Ex_Inv_PermShk_facts.update({'value_now': Inv_Ex_Inv_PermShk})
        solution.facts.update({'Inv_Ex_Inv_PermShk': Inv_Ex_Inv_PermShk_facts})
        solution.Inv_Ex_Inv_PermShk_facts = Inv_Ex_Inv_PermShk_facts
        # solution.Inv_Ex_Inv_PermShk = Inv_Ex_Inv_PermShk

        Ex_RNrm_facts = {
            'about': 'Expectation of Stochastic-Growth-Normalized Return'}
        Ex_RNrm = np.dot(solution.PF_RNrm/PermShkValsXref, ShkPrbsNext)
        Ex_RNrm_facts.update({'latexexpr': r'\Ex_RNrm'})
#        Ex_RNrm_facts.update({'_unicode_': r'R/Γ'})
        Ex_RNrm_facts.update({'urlhandle': urlroot+'ExRNrm'})
        Ex_RNrm_facts.update({'py___code': r'Rfree/PermGroFacAdj'})
        Ex_RNrm_facts.update({'value_now': Ex_RNrm})
        solution.facts.update({'Ex_RNrm': Ex_RNrm_facts})
        solution.Ex_RNrm_facts = Ex_RNrm_facts
        solution.Ex_RNrm = Ex_RNrm

        Inv_Ex_RNrm_facts = {
            'about': 'Inverse of Expectation of Stochastic-Growth-Normalized Return'}
        Inv_Ex_RNrm = 1/Ex_RNrm
        Inv_Ex_RNrm_facts.update(
            {'latexexpr': '\InvExInvPermShk=\left(\Ex[\PermShk^{-1}]\right)^{-1}'})
#        Inv_Ex_RNrm_facts.update({'_unicode_': r'R/Γ'})
        Inv_Ex_RNrm_facts.update({'urlhandle': urlroot+'InvExRNrm'})
        Inv_Ex_RNrm_facts.update({'py___code': r'1/Ex_RNrm'})
        Inv_Ex_RNrm_facts.update({'value_now': Inv_Ex_RNrm})
        solution.facts.update({'Inv_Ex_RNrm': Inv_Ex_RNrm_facts})
        solution.Inv_Ex_RNrm_facts = Inv_Ex_RNrm_facts
        solution.Inv_Ex_RNrm = Inv_Ex_RNrm

        Ex_uInv_PermShk_facts = {
            'about': 'Expected Utility for Consuming Permanent Shock'}
        Ex_uInv_PermShk = np.dot(PermShkValsXref ** (1 - CRRA), ShkPrbsNext)
        Ex_uInv_PermShk_facts.update({'latexexpr': r'\Ex_uInv_PermShk'})
        Ex_uInv_PermShk_facts.update({'urlhandle': r'ExuInvPermShk'})
        Ex_uInv_PermShk_facts.update(
            {'py___code': r'np.dot(PermShkValsXref**(1-CRRA),ShkPrbsNext)'})
        Ex_uInv_PermShk_facts.update({'value_now': Ex_uInv_PermShk})
        solution.facts.update({'Ex_uInv_PermShk': Ex_uInv_PermShk_facts})
        solution.Ex_uInv_PermShk_facts = Ex_uInv_PermShk_facts
        solution.Ex_uInv_PermShk = Ex_uInv_PermShk

        uInv_Ex_uInv_PermShk = Ex_uInv_PermShk ** (1 / (1 - CRRA))
        uInv_Ex_uInv_PermShk_facts = {
            'about': 'Inverted Expected Utility for Consuming Permanent Shock'}
        uInv_Ex_uInv_PermShk_facts.update({'latexexpr': r'\uInvExuInvPermShk'})
        uInv_Ex_uInv_PermShk_facts.update({'urlhandle': urlroot+'uInvExuInvPermShk'})
        uInv_Ex_uInv_PermShk_facts.update({'py___code': r'Ex_uInv_PermShk**(1/(1-CRRA))'})
        uInv_Ex_uInv_PermShk_facts.update({'value_now': uInv_Ex_uInv_PermShk})
        solution.facts.update({'uInv_Ex_uInv_PermShk': uInv_Ex_uInv_PermShk_facts})
        solution.uInv_Ex_uInv_PermShk_facts = uInv_Ex_uInv_PermShk_facts
        solution.uInv_Ex_uInv_PermShk = uInv_Ex_uInv_PermShk

        PermGroFacAdj_facts = {
            'about': 'Uncertainty-Adjusted Permanent Income Growth Factor'}
        PermGroFacAdj = PermGroNum * Inv_Ex_Inv_PermShk
        PermGroFacAdj_facts.update({'latexexpr': r'\mathcal{R}\underline{\permShk}'})
        PermGroFacAdj_facts.update({'urlhandle': urlroot+'PermGroFacAdj'})
        PermGroFacAdj_facts.update({'value_now': PermGroFacAdj})
        solution.facts.update({'PermGroFacAdj': PermGroFacAdj_facts})
        solution.PermGroFacAdj_facts = PermGroFacAdj_facts
        solution.PermGroFacAdj = PermGroFacAdj

        GPFNrm_facts = {
            'about': 'Normalized Expected Growth Patience Factor'}
        solution.GPFNrm = GPFNrm = solution.GPFRaw * Ex_Inv_PermShk
        GPFNrm_facts.update({'latexexpr': r'\GPFNrm'})
        GPFNrm_facts.update({'_unicode_': r'Þ_Γ'})
        GPFNrm_facts.update({'urlhandle': urlroot+'GPFNrm'})
        GPFNrm_facts.update({'py___code': 'test: GPFNrm < 1'})
        solution.facts.update({'GPFNrm': GPFNrm_facts})
        solution.GPFNrm_facts = GPFNrm_facts

        GICNrm_facts = {'about': 'Growth Impatience Condition'}
        GICNrm_facts.update({'latexexpr': r'\GICNrm'})
        GICNrm_facts.update({'urlhandle': urlroot+'GICNrm'})
        GICNrm_facts.update({'py___code': 'test: agent.GPFNrm < 1'})
        solution.facts.update({'GICNrm': GICNrm_facts})
        solution.GICNrm_facts = GICNrm_facts

        FVAF_facts = {'about': 'Finite Value of Autarky Factor'}
        solution.FVAF = FVAF = LivNum * DiscFacEff * uInv_Ex_uInv_PermShk
        FVAF_facts.update({'latexexpr': r'\FVAF'})
        FVAF_facts.update({'urlhandle': urlroot+'FVAF'})
        solution.facts.update({'FVAF': FVAF_facts})
        solution.FVAF_facts = FVAF_facts

        FVAC_facts = {'about': 'Finite Value of Autarky Condition'}
        FVAC_facts.update({'latexexpr': r'\FVAC'})
        FVAC_facts.update({'urlhandle': urlroot+'FVAC'})
        FVAC_facts.update({'py___code': 'test: FVAF < 1'})
        solution.facts.update({'FVAC': FVAC_facts})
        solution.FVAC_facts = FVAC_facts

        DiscGPFNrmCusp_facts = {'about':
                                'DiscFac s.t. GPFNrm = 1'}
        solution.DiscGPFNrmCusp = DiscGPFNrmCusp = ((PermGroNum*Inv_Ex_Inv_PermShk)**(CRRA))/Rfree
        DiscGPFNrmCusp_facts.update({'latexexpr': ''})
        DiscGPFNrmCusp_facts.update({'value_now': DiscGPFNrmCusp})
        DiscGPFNrmCusp_facts.update({
            'py___code': '((PermGro * Inv_Ex_Inv_PermShk) ** CRRA)/(Rfree)'})
        solution.facts.update({'DiscGPFNrmCusp': DiscGPFNrmCusp_facts})
        solution.DiscGPFNrmCusp_facts = DiscGPFNrmCusp_facts

        DiscGPFRawCusp_facts = {
            'about': 'DiscFac s.t. GPFRaw = 1'}
        solution.DiscGPFRawCusp = DiscGPFRawCusp = \
            ((PermGroNum) ** (CRRA)) / (Rfree)
        DiscGPFRawCusp_facts.update({'latexexpr': ''})
        DiscGPFRawCusp_facts.update({'value_now': DiscGPFRawCusp})
        DiscGPFRawCusp_facts.update({
            'py___code': '( PermGro                       ** CRRA)/(Rfree)'})
        solution.facts.update({'DiscGPFRawCusp': DiscGPFRawCusp_facts})
        solution.DiscGPFRawCusp_facts = DiscGPFRawCusp_facts

        DiscGPFLivCusp_facts = {
            'about': 'DiscFac s.t. GPFLiv = 1'}
        solution.DiscGPFLivCusp = DiscGPFLivCusp = ((PermGroNum) ** (CRRA)) \
            / (Rfree * LivNum)
        DiscGPFLivCusp_facts.update({'latexexpr': ''})
        DiscGPFLivCusp_facts.update({'value_now': DiscGPFLivCusp})
        DiscGPFLivCusp_facts.update({
            'py___code': '( PermGroNum                       ** CRRA)/(Rfree*LivNum)'})
        solution.facts.update({'DiscGPFLivCusp': DiscGPFLivCusp_facts})
        solution.DiscGPFLivCusp_facts = DiscGPFLivCusp_facts

        # Calculate objects whose values are built up recursively from
        # prior period's values

        Ex_IncNextNrm = np.dot(PermShkValsXref * TranShkValsXref, ShkPrbsNext)
        hNrmNow = (
            (PermGroNum / Rfree) * (Ex_IncNextNrm + self.solution_next.hNrmNow)
        )
        hNrmNow = PermGroNum/Rfree
        hNrmNow_facts = {'about': 'Finite Human Wealth Factor'}
        hNrmNow_facts.update({'latexexpr': r'\hNrmNow'})
        hNrmNow_facts.update({'_unicode_': r'R/Γ'})
        hNrmNow_facts.update({'urlhandle': urlroot+'hNrmNow'})
        hNrmNow_facts.update({'py___code': r'PermGroNumInf/Rfree'})
        hNrmNow_facts.update({'value_now': hNrmNow})
        solution.facts.update({'hNrmNow': hNrmNow_facts})
        solution.hNrmNow_facts = hNrmNow_facts
        self.hNrmNow = solution.hNrmNow = hNrmNow

        MPCminNow = 1.0 / (1.0 + solution.RPF / self.solution_next.MPCminNow)
        MPCminNow_facts = {
            'about': 'Minimal MPC as m -> infty'}
        MPCminNow_facts.update({'latexexpr': r''})
        MPCminNow_facts.update({'urlhandle': urlroot+'MPCminNow'})
        MPCminNow_facts.update({'value_now': MPCminNow})
        solution.facts.update({'MPCminNow': MPCminNow_facts})
        solution.MPCminNow_facts = MPCminNow_facts
        self.MPCminNow = solution.MPCminNow = MPCminNow

        MPCmaxNow = 1.0 / (1.0 + (WorstIncPrb ** (1.0 / CRRA)) * solution.RPF
                           / self.solution_next.MPCmaxNow)
        MPCmaxNow_facts = {
            'about': 'Maximal MPC in current period as m -> minimum'}
        MPCmaxNow_facts.update({'latexexpr': r''})
        MPCmaxNow_facts.update({'urlhandle': urlroot+'MPCmaxNow'})
        MPCmaxNow_facts.update({'value_now': MPCmaxNow})
        solution.facts.update({'MPCmaxNow': MPCmaxNow_facts})
        solution.MPCmaxNow_facts = MPCmaxNow_facts
        solution.MPCmaxNow = MPCmaxNow
#       print('MPCmaxNow matches: ' + str(solution.MPCmaxNow == self.solnew.MPCmaxNow))

        # Lower bound of aggregate wealth growth if all inheritances squandered

        cFuncLimitIntercept = MPCminNow * hNrmNow
        cFuncLimitIntercept_facts = {
            'about': 'Vertical intercept of perfect foresight consumption function'}
        cFuncLimitIntercept_facts.update({'latexexpr': '\MPC '})
        cFuncLimitIntercept_facts.update({'urlhandle': ''})
        cFuncLimitIntercept_facts.update({'value_now': cFuncLimitIntercept})
        cFuncLimitIntercept_facts.update({
            'py___code': 'MPCminNow * hNrmNow'})
        solution.facts.update({'cFuncLimitIntercept': cFuncLimitIntercept_facts})
        solution.cFuncLimitIntercept_facts = cFuncLimitIntercept_facts
        solution.cFuncLimitIntercept = cFuncLimitIntercept

        cFuncLimitSlope = MPCminNow
        cFuncLimitSlope_facts = {
            'about': 'Slope of limiting consumption function'}
        cFuncLimitSlope_facts = dict({'latexexpr': '\MPC \hNrmNow'})
        cFuncLimitSlope_facts.update({'urlhandle': ''})
        cFuncLimitSlope_facts.update({'value_now': cFuncLimitSlope})
        cFuncLimitSlope_facts.update({
            'py___code': 'MPCminNow * hNrmNow'})
        solution.facts.update({'cFuncLimitSlope': cFuncLimitSlope_facts})
        solution.cFuncLimitSlope_facts = cFuncLimitSlope_facts
        solution.cFuncLimitSlope = cFuncLimitSlope

        # # Merge all the parameters
        # # In python 3.9, the syntax is new_dict = dict_a | dict_b
        # solution.params_all = {**self.params_cons_ind_shock_setup_init,
        #                    **params_cons_ind_shock_setup_set_and_update_values}

        # Now that the calculations are done, store results in self.
        # self, here, is the solver
        # goal: agent,  solver, and solution should be standalone
        # this requires the solution to get some info from the solver

        Ex_Inv_PermShk = np.dot(1/PermShkValsXref, ShkPrbsNext)
        Inv_Ex_PermShk_Inv = 1/Ex_Inv_PermShk
        Ex_uInv_PermShk = np.dot(PermShkValsXref ** (1 - CRRA), ShkPrbsNext)
        uInv_Ex_uInv_PermShk = Ex_uInv_PermShk ** (
            1 / (1 - CRRA)
        )

        if solution.Inv_PF_RNrm < 1:        # Finite if Rfree > PermGroNum
            solution.hNrmNowInf = 1/(1-solution.Inv_PF_RNrm)

        # Given m, value of c where E[m_{t+1}]=m_{t}
        # url/#
        solution.c_where_Ex_mtp1_minus_mt_eq_0 = (
            lambda m_t:
            m_t * (1 - solution.Inv_Ex_RNrm) + (solution.Inv_Ex_RNrm)
        )

        # Given m, value of c where E[mLev_{t+1}/mLev_{t}]=PermGroNum
        # Solves for c in equation at url/#balgrostable

        solution.c_where_Ex_PermShk_times_mtp1_minus_mt_eq_0 = (
            lambda m_t:
            m_t * (1 - solution.Inv_PF_RNrm) + solution.Inv_PF_RNrm
        )

        # E[c_{t+1} pLev_{t+1}/pLev_{t}] as a fn of a_{t}
        solution.Ex_cLev_tp1_Over_pLev_t_from_at = (
            lambda a_t:
            np.dot(solution.PermGroNum *
                   solution.PermShkValsXref *
                   solution.cFunc(
                       (solution.PF_RNrm/solution.PermShkValsXref) * a_t
                       + solution.TranShkValsXref
                   ),
                   solution.ShkPrbsNext)
        )

        solution.c_where_Ex_mtp1_minus_mt_eq_0 = c_where_Ex_mtp1_minus_mt_eq_0 = (
            lambda m_t:
            m_t * (1 - 1/solution.Ex_RNrm) + (1/solution.Ex_RNrm)
        )

        # Solve the equation at url/#balgrostable
        solution.c_where_Ex_PermShk_times_mtp1_minus_mt_eq_0 = \
            c_where_Ex_PermShk_times_mtp1_minus_mt_eq_0 = (
                lambda m_t:
                (m_t * (1 - 1/solution.PF_RNrm)) + (1/solution.PF_RNrm)
            )

        # mNrmTrg solves Ex_RNrm*(m - c(m)) + E[inc_next] - m = 0
        Ex_m_tp1_minus_m_t = (
            lambda m_t:
            solution.Ex_RNrm * (m_t - solution.cFunc(m_t))+solution.Ex_IncNextNrm - m_t
        )
        solution.Ex_m_tp1_minus_m_t = Ex_m_tp1_minus_m_t

        solution.Ex_cLev_tp1_Over_pLev_t_from_at = Ex_cLev_tp1_Over_pLev_t_from_at = (
            lambda a_t:
            np.dot(
                solution.PermShkValsXref * solution.PermGroNum * solution.cFunc(
                    (solution.PF_RNrm/solution.PermShkValsXref) * a_t + solution.TranShkValsXref
                ),
                solution.ShkPrbsNext)
        )

        solution.Ex_PermShk_tp1_times_m_tp1_minus_m_t = \
            Ex_PermShk_tp1_times_m_tp1_minus_m_t = (
                lambda m_t: self.PF_RNrm * (m_t - solution.cFunc(m_t)) + 1.0 - m_t
            )
        return solution

    def def_BoroCnst(self, BoroCnstArt):
        """
        Defines the constrained portion of the consumption function as cFuncNowCnst,
        an attribute of self.  Uses the artificial and natural borrowing constraints.

        Parameters
        ----------
        BoroCnstArt : float or None
            Borrowing constraint for the minimum allowable assets to end the
            period with.  If it is less than the natural borrowing constraint,
            then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
            rowing constraint.

        Returns
        -------
        none
        """
        # Calculate the minimum allowable value of money resources in this period
        self.BoroCnstNat = (
            (self.solution_next.mNrmMin - self.TranShkMinNext)
            * (self.PermGroFac * self.PermShkMinNext)
            / self.Rfree
        )

        if BoroCnstArt is None:
            self.mNrmMinNow = self.BoroCnstNat
#            self.solnew.mNrmMinNow = self.solnew.BoroCnstNat
        else:
            self.mNrmMinNow = np.max([self.BoroCnstNat, BoroCnstArt])
#            self.solnew.mNrmMinNow = np.max([self.solnew.BoroCnstNat, BoroCnstArt])
        if self.BoroCnstNat < self.mNrmMinNow:
            self.MPCmaxNowEff = 1.0  # If actually constrained, MPC near limit is 1
#            self.solnew.MPCmaxNowEff = 1.0  # If actually constrained, MPC near limit is 1
        else:
            self.MPCmaxNowEff = self.MPCmaxNow
#            self.solnew.MPCmaxNowEff = self.solnew.MPCmaxNow

        # Define the borrowing constraint (limiting consumption function)
        self.cFuncNowCnst = LinearInterp(
            np.array([self.mNrmMinNow, self.mNrmMinNow + 1]), np.array([0.0, 1.0])
        )

    def prepare_to_solve(self):
        """
        Perform preparatory work before calculating this period's unconstrained consumption
        function.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        print('hi')
        # self.set_and_update_values(
        #     self.solution_next, self.IncShkDstn, self.LivPrb, self.DiscFac
        # )
        self.def_BoroCnst(self.BoroCnstArt)


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
        aNrmNow : np.array
            A 1D array of end-of-period assets; also stored as attribute of self.
        """

        # We define aNrmNow all the way from BoroCnstNat up to max(self.aXtraGrid)
        # even if BoroCnstNat < BoroCnstArt, so we can construct the consumption
        # function as the lower envelope of the (by the artificial borrowing con-
        # straint) unconstrained consumption function, and the artificially con-
        # strained consumption function.
        self.aNrmNow = np.asarray(self.aXtraGrid) + self.BoroCnstNat

        return self.aNrmNow

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
        return self.Rfree / (self.PermGroFac * shocks[0]) \
            * a_Nrm_Val + shocks[1]

    def calc_EndOfPrdvP(self):
        """
        Calculate end-of-period marginal value of assets at each point in aNrmNow.
        Does so by taking a weighted sum of next period marginal values across
        income shocks (in a preconstructed grid self.mNrmNext).

        Parameters
        ----------
        none

        Returns
        -------
        EndOfPrdvP : np.array
            A 1D array of end-of-period marginal value of assets
        """

        def vp_next(shocks, a_Nrm_Val):
            return shocks[0] ** (-self.CRRA) \
                * self.vPfuncNext(self.m_Nrm_tp1(shocks, a_Nrm_Val))

        EndOfPrdvP = (
            self.DiscFacEff
            * self.Rfree
            * self.PermGroFac ** (-self.CRRA)
            * calc_expectation(
                self.IncShkDstn,
                vp_next,
                self.aNrmNow
            )
        )

        return EndOfPrdvP

    def get_interpolation_points_via_EGM(self, EndOfPrdvP, aNrmNow):
        """
        Finds interpolation points (c,m) for the consumption function.

        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal values.
        aNrmNow : np.array
            Array of end-of-period asset values that yield the marginal values
            in EndOfPrdvP.

        Returns
        -------
        c_for_interpolation : np.array
            Consumption points for interpolation.
        m_for_interpolation : np.array
            Corresponding market resource points for interpolation.
        """
        cNrmNow = self.uPinv(EndOfPrdvP)
        mNrmNow = cNrmNow + aNrmNow

        # Limiting consumption is zero as m approaches mNrmMin
        c_for_interpolation = np.insert(cNrmNow, 0, 0.0, axis=-1)
        m_for_interpolation = np.insert(mNrmNow, 0, self.BoroCnstNat, axis=-1)

        # Store these for calcvFunc
        self.cNrmNow = cNrmNow
        self.mNrmNow = mNrmNow

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
        solution_now : ConsumerSolution
            The solution to this period's consumption-saving problem, with a
            consumption function, marginal value function, and minimum m.
        """
        # Construct the unconstrained consumption function
        cFuncNowUnc = interpolator(mNrm, cNrm)

        # Combine the constrained and unconstrained functions into the true consumption function
        # breakpoint()  # LowerEnvelope should only be used when BoroCnstArt is true
        cFuncNow = LowerEnvelope(cFuncNowUnc, self.cFuncNowCnst, nan_bool=False)

        # Make the marginal value function and the marginal marginal value function
        vPfuncNow = MargValueFuncCRRA(cFuncNow, self.CRRA)

        # Pack up the solution and return it
        solution_now = ConsumerSolution(
            cFunc=cFuncNow, vPfunc=vPfuncNow, mNrmMin=self.mNrmMinNow
        )

        return solution_now

    def make_EGM_solution(self, EndOfPrdvP, aNrm, interpolator):
        """
        Given end of period assets and end of period marginal value, construct
        the basic solution for this period.

        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal values.
        aNrm : np.array
            Array of end-of-period asset values that yield the marginal values
            in EndOfPrdvP.

        interpolator : function
            A function that constructs and returns a consumption function.

        Returns
        -------
        solution_now : ConsumerSolution
            The solution to this period's consumption-saving problem, with a
            consumption function, marginal value function, and minimum m.
        """
        cNrm, mNrm = self.get_interpolation_points_via_EGM(EndOfPrdvP, aNrm)
        solution_now = self.use_points_for_interpolation(cNrm, mNrm, interpolator)

        return solution_now

    def solution_add_MPC_bounds_and_human_wealth_PDV(self, solution):
        """
        Take a solution and add human wealth and the bounding MPCs to it.

        Parameters
        ----------
        solution : ConsumerSolution
            The solution to this period's consumption-saving problem.

        Returns:
        ----------
        solution : ConsumerSolution
            The solution to this period's consumption-saving problem, but now
            with human wealth and the bounding MPCs.
        """
        solution.hNrmNow = self.hNrmNow
        solution.MPCminNow = self.MPCminNow
        solution.MPCmaxNow = self.MPCmaxNowEff
#        _log.warning(
#            "add_MPC_bounds_and_human_wealth_PDV is deprecated; its functions have been absorbed by add_results")
        return solution

    def add_mNrmTrg(self, solution):
        """
        Finds value of (normalized) market resources m at which individual consumer
        expects m not to change.

        This will exist if the GICNrm holds.

        https://econ-ark.github.io/BufferStockTheory#UniqueStablePoints

        Parameters
        ----------
        solution : ConsumerSolution
            Solution to this period's problem, which must have attribute cFunc.
        Returns
        -------
        solution : ConsumerSolution
            Same solution that was passed, but now with the attribute mNrmStE.
        """
        PerfFsgt = False

        if type(self) == ConsPerfForesightSolver:
            PerfFsgt = True

        # If no uncertainty, return the degenerate targets for the PF model
        if hasattr(self, "TranShkMinNext"):  # Then it has transitory shocks
            # Handle the degenerate case where shocks are of size zero
            if ((self.TranShkMinNext == 1.0) and (self.PermShkMinNext == 1.0)):
                # But they still might have unemployment risk
                if hasattr(self, "UnempPrb"):
                    if ((self.UnempPrb == 0.0) or (self.IncUnemp == 1.0)):
                        PerfFsgt = True  # No unemployment risk either
                    else:
                        PerfFsgt = False  # The only kind of uncertainty is unemployment

        if PerfFsgt:  # If growth impatient limit be to borrow max possible
            if self.GICRaw:  # max of nat and art boro cnst
                if type(self.BoroCnstArt) == type(None):
                    solution.mNrmStE = -self.hNrmNow
                    solution.mNrmTrg = -self.hNrmNow
                else:
                    bNrmNxt = -self.BoroCnstArt * self.PF_RNrm
                    solution.mNrmStE = bNrmNxt + 1.0
                    solution.mNrmTrg = bNrmNxt + 1.0
            else:  # infinity
                solution.mNrmStE = float('inf')
                solution.mNrmTrg = float('inf')
            return solution

        # First find
        # \bar{\mathcal{R}} = E_t[R/Gamma_{t+1}] = (R/Gamma) E_t[1/psi_{t+1}]
        if type(self) == ConsPerfForesightSolver:
            Ex_PermShkInv = 1.0
        else:
            Ex_PermShkInv = np.dot(1/self.PermShkValsNext, self.ShkPrbsNext)

        c_where_Ex_mtp1_minus_mt_eq_0 = (
            lambda m_t:
            m_t * (1 - 1/solution.Ex_RNrm) + (1/solution.Ex_RNrm)
        )
        solution.c_where_Ex_mtp1_minus_mt_eq_0 = c_where_Ex_mtp1_minus_mt_eq_0

        # Solve the equation at url/#balgrostable
        # for c
        c_where_Ex_PermShk_times_mtp1_minus_mt_eq_0 = (
            lambda m_t:
            (m_t * (1 - 1/solution.PF_RNrm)) + (1/solution.PF_RNrm)
        )
        solution.c_where_Ex_PermShk_times_mtp1_minus_mt_eq_0 = c_where_Ex_PermShk_times_mtp1_minus_mt_eq_0

        # mNrmTrg solves Ex_RNrm*(m - c(m)) + E[inc_next] - m = 0
        Ex_m_tp1_minus_m_t = (
            lambda m_t:
            solution.Ex_RNrm * (m_t - solution.cFunc(m_t))+solution.Ex_IncNextNrm - m_t
        )
        solution.Ex_m_tp1_minus_m_t = Ex_m_tp1_minus_m_t

        # def mNrmTrg_for_solution(self):solution_
        # # Minimum market resources plus next income is okay starting guess
        #     m_init_guess=solution.mNrmMin + solution.Ex_IncNextNrm
        #     try:
        #         mNrmTrg=newton(
        #             solution.Ex_m_tp1_minus_m_t,
        #             m_init_guess)
        #     except:
        #         mNrmTrg=None

        #     return solution

        # url/#
        solution.Ex_cLev_tp1_Over_pLev_t_from_at = (
            lambda a_t:
            np.dot(
                solution.PermShkValsNext * solution.PermGroFac * solution.cFunc(
                    (solution.PF_RNrm/solution.PermShkValsNext) * a_t + solution.TranShkValsNext
                ),
                solution.ShkPrbsNext)
        )

        # Minimum market resources plus next income is okay starting guess
        m_init_guess = solution.mNrmMin + solution.Ex_IncNextNrm
        try:
            mNrmTrg = newton(Ex_m_tp1_minus_m_t, m_init_guess)
        except:
            mNrmTrg = None

        # Add mNrmTrg to the solution and return it
        solution.mNrmTrg = mNrmTrg
        return solution

    def add_mNrmTrg_new(self, solution):
        """
        Finds value of (normalized) market resources m at which individual consumer
        expects m not to change.

        This will exist if the GICNrm holds.

        https://econ-ark.github.io/BufferStockTheory#UniqueStablePoints

        Parameters
        ----------
        solution : ConsumerSolution
            Solution to this period's problem, which must have attribute cFunc.
        Returns
        -------
        solution : ConsumerSolution
            Same solution that was passed, but now with the attribute mNrmStE.
        """

        # Minimum market resources plus next income is okay starting guess
        m_init_guess = solution.mNrmMin + solution.Ex_IncNextNrm
        try:
            mNrmTrg = newton(
                solution.Ex_m_tp1_minus_m_t,
                m_init_guess)
        except:
            mNrmTrg = None

        # Add mNrmTrg to the solution and return it
#        solution.mNrmTrg_new = mNrmTrg
        return solution

    # Making this a # @staticmethod allows us to attach it to the solution
    # @staticmethod
    def mNrmTrg_find(solution):
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
        m_init_guess = solution.mNrmMin + solution.Ex_IncNextNrm
        try:
            mNrmTrg = newton(  # Find value where argument is zero
                solution.Ex_m_tp1_minus_m_t,
                m_init_guess)
        except:
            mNrmTrg = None

        return mNrmTrg

    def add_mNrmStE_new(self, solution):
        """
        Finds market resources ratio at which 'balanced growth' is expected.
        This is the m ratio such that the expected growth rate of the M level
        matches the expected growth rate of permanent income. This value does
        not exist if the Growth Impatience Condition does not hold.

        https://econ-ark.github.io/BufferStockTheory#Unique-Stable-Points

        Parameters
        ----------
        solution : ConsumerSolution
            Solution to this period's problem, which must have attribute cFunc.
        Returns
        -------
        solution : ConsumerSolution
            Same solution that was passed, but now with the attribute mNrmStE
        """

        # If the necessary inputs have not yet been made, make them
        if not hasattr(solution, "Ex_PermShk_tp1_times_m_tp1_minus_m_t"):
            solution = self.add_MPC_bounds_and_human_wealth_PDV(solution)

        if not hasattr(solution, "Ex_IncNextNrm"):
            solution = self.add_MPC_bounds_and_human_wealth_PDV(solution)

        # Minimum market resources plus next income is starting guess
        m_init_guess = self.mNrmMinNow + 1.0
        # newton() finds the point where this is zero
        try:
            mNrmStE = newton(
                solution.Ex_PermShk_tp1_times_m_tp1_minus_m_t,
                m_init_guess
            )
        except:
            mNrmStE = None

        solution.mNrmStE_new = mNrmStE
        return solution

    # Making this a # @staticmethod allows us to attach it to the solution
    # @staticmethod
    def mNrmStE_find(solution):
        """
        Finds market resources ratio at which 'balanced growth' is expected.
        This is the m ratio such that the expected growth rate of the M level
        matches the expected growth rate of permanent income. This value does
        not exist if the Growth Impatience Condition does not hold.

        https://econ-ark.github.io/BufferStockTheory#Unique-Stable-Points

        Parameters
        ----------
        solution : ConsumerSolution
            Solution to this period's problem, which must have attribute cFunc.
        Returns
        -------
        mNrmStE : Value of mNrm for the given ConsumerSolution where
            `Ex_mLev_tp1_over_mLev_t == PermGroFac`

        """
        # Minimum market resources plus next income is starting guess

        m_init_guess = solution.mNrmMin + 1.0
        # newton() finds the point where this is zero
        try:
            mNrmStE = newton(  # Finds the point where argument is zero
                solution.Ex_PermShk_tp1_times_m_tp1_minus_m_t,
                m_init_guess
            )
        except:
            mNrmStE = None
            _log.warning('No value of mNrmStE was found.  Continuing anyway.')

        return mNrmStE

    def add_mNrmStE(self, solution):
        """
        Finds market resources ratio at which 'balanced growth' is expected.
        This is the m ratio such that the expected growth rate of the M level
        matches the expected growth rate of permanent income. This value does
        not exist if the Growth Impatience Condition does not hold.

        https://econ-ark.github.io/BufferStockTheory#Unique-Stable-Points

        Parameters
        ----------
        solution : ConsumerSolution
            Solution to this period's problem, which must have attribute cFunc.
        Returns
        -------
        solution : ConsumerSolution
            Same solution that was passed, but now with the attribute mNrmStE
        """
        # Probably should test whether GICRaw holds and log error if it does not
        # using check_conditions
        # All combinations of c and m that yield E[PermGroFac PermShkVal mNext] = mNow
        # https://econ-ark.github.io/BufferStockTheory/#The-Individual-Steady-State

        # PF_RNrm = self.PF_RNrm

        # If we are working with a model that permits uncertainty but
        # uncertainty has been set to zero, calculate the correct answer
        # by hand because in this degenerate case numerical search will
        # have trouble
        if hasattr(self, "TranShkMinNext"):  # Then it has transitory shocks
            if ((self.TranShkMinNext == 1.0) and (self.PermShkMinNext == 1.0)):
                # but of zero size (and permanent shocks also not there)
                if self.GICRaw:  # max of nat and art boro cnst
                    #                    breakpoint()
                    if type(self.BoroCnstArt) == type(None):
                        solution.mNrmStE = -self.hNrmNow
                        solution.mNrmTrg = -self.hNrmNow
                    else:
                        bNrmNxt = -self.BoroCnstArt * self.PF_RNrm
                        solution.mNrmStE = bNrmNxt + 1.0
                        solution.mNrmTrg = bNrmNxt + 1.0
                else:  # infinity
                    solution.mNrmStE = float('inf')
                    solution.mNrmTrg = float('inf')
                return solution

        Ex_PermShk_tp1_times_m_tp1_minus_m_t = (
            lambda m_t: solution.PF_RNrm * (m_t - solution.cFunc(m_t)) + 1.0 - m_t
        )

        solution.Ex_PermShk_tp1_times_m_tp1_minus_m_t = \
            Ex_PermShk_tp1_times_m_tp1_minus_m_t

        # Minimum market resources plus next income is okay starting guess
        m_init_guess = self.mNrmMinNow + 1.0
        try:
            mNrmStE = newton(
                Ex_PermShk_tp1_times_m_tp1_minus_m_t, m_init_guess
            )
        except:
            mNrmStE = None

        solution.mNrmStE = mNrmStE
        return solution

    def add_stable_points(self, solution):
        """
        Checks necessary conditions for the existence of the individual steady
        state and target levels of market resources (see above).
        If the conditions are satisfied, computes and adds the stable points
        to the solution.

        Parameters
        ----------
        solution : ConsumerSolution
            Solution to this period's problem, which must have attribute cFunc.
        Returns
        -------
        solution : ConsumerSolution
            Same solution that was passed, but now with attributes mNrmStE and
            mNrmTrg, if they exist.

        """
        # Test for the edge case where the model that allows for uncertainty has been
        # called with values of all the uncertainty parameters equal to zero

        solution = self.add_MPC_bounds_and_human_wealth_PDV(solution)

        PerfFsgt = False

        if type(self) == ConsPerfForesightSolver:
            PerfFsgt = True

        # If no uncertainty, return the degenerate targets for the PF model
        if hasattr(self, "TranShkMinNext"):  # Then it has transitory shocks
            # Handle the degenerate case where shocks are of size zero
            if ((self.TranShkMinNext == 1.0) and (self.PermShkMinNext == 1.0)):
                # But they still might have unemployment risk
                if hasattr(self, "UnempPrb"):
                    if ((self.UnempPrb == 0.0) or (self.IncUnemp == 1.0)):
                        PerfFsgt = True  # No unemployment risk either
                    else:
                        PerfFsgt = False  # The only kind of uncertainty is unemployment

        if PerfFsgt:
            if self.GICRaw:  # max of nat and art boro cnst
                if type(self.BoroCnstArt) == type(None):
                    # If growth impatient, limit is to borrow max possible
                    if self.FHWC:  # Finite human wealth
                        solution.mNrmStE_new = -self.hNrmNow
                        solution.mNrmTrg = -self.hNrmNow
                    else:
                        _log.warning("Limiting solution is c(m) = infty")
                        solution.mNrmStE_new = float('NaN')
                        solution.mNrmTrg = float('NaN')
                else:  # Max they can borrow is up to extent of liq constr
                    bNrmNxt = -self.BoroCnstArt * self.PF_RNrm
                    solution.mNrmStE_new = bNrmNxt + 1.0
                    solution.mNrmTrg = bNrmNxt + 1.0
            else:  # infinity
                solution.mNrmStE_new = float('inf')
                solution.mNrmTrg = float('inf')
            return solution

        # 0. Check if GICRaw holds. If so, then mNrmStE will exist. So, compute it.
        # 1. Check if GICNrm holds. If so, then mNrmTrg will exist. So, compute it.

        if self.GICRaw:
            solution = self.add_mNrmStE_new(solution)  # pseudo steady state m, if it exists
        if self.GICNrm:
            solution = self.add_mNrmTrg_new(solution)  # find target m, if it exists

        return solution

    def add_stable_points_to_solution(self):
        """
        Checks necessary conditions for the existence of the individual steady
        state and target levels of market resources (see above).
        If the conditions are satisfied, computes and adds the stable points
        to the solution.

        Parameters
        ----------
        solution : ConsumerSolution
            Solution to this period's problem, which must have attribute cFunc.
        Returns
        -------
        solution : ConsumerSolution
            Same solution that was passed, but now with attributes mNrmStE and
            mNrmTrg, if they exist.

        """
        # Test for the edge case where the model that allows for uncertainty has been
        # called with values of all the uncertainty parameters equal to zero

        PerfFsgt = False

        if type(self) == ConsPerfForesightSolver:
            PerfFsgt = True

        # If no uncertainty, return the degenerate targets for the PF model
        if hasattr(self, "TranShkMinNext"):  # Then it has transitory shocks
            # Handle the degenerate case where shocks are of size zero
            if ((self.TranShkMinNext == 1.0) and (self.PermShkMinNext == 1.0)):
                # But they still might have unemployment risk
                if hasattr(self, "UnempPrb"):
                    if ((self.UnempPrb == 0.0) or (self.IncUnemp == 1.0)):
                        PerfFsgt = True  # No unemployment risk either
                    else:
                        PerfFsgt = False  # The only kind of uncertainty is unemployment

        if PerfFsgt:
            if self.GICRaw:  # max of nat and art boro cnst
                if type(self.BoroCnstArt) == type(None):
                    # If growth impatient, limit is to borrow max possible
                    if self.FHWC:  # Finite human wealth
                        self.mNrmStE_new = -self.hNrmNow
                        self.mNrmTrg = -self.hNrmNow
                    else:
                        _log.warning("Limiting self is c(m) = infty")
                        self.mNrmStE_new = float('NaN')
                        self.mNrmTrg = float('NaN')
                else:  # Max they can borrow is up to extent of liq constr
                    bNrmNxt = -self.BoroCnstArt * self.PF_RNrm
                    self.mNrmStE_new = bNrmNxt + 1.0
                    self.mNrmTrg = bNrmNxt + 1.0
            else:  # infinity
                self.mNrmStE_new = float('inf')
                self.mNrmTrg = float('inf')
            return self

        # 0. Check if GICRaw holds. If so, then mNrmStE will exist. So, compute it.
        # 1. Check if GICNrm holds. If so, then mNrmTrg will exist. So, compute it.

        if self.GICRaw:
            self = self.add_mNrmStE_new(self)  # pseudo steady state m, if it exists
        if self.GICNrm:
            self = self.add_mNrmTrg_new(self)  # find target m, if it exists

    def add_stable_points_old(self, solution):
        """
        Checks necessary conditions for the existence of the individual steady
        state and target levels of market resources (see above).
        If the conditions are satisfied, computes and adds the stable points
        to the solution.

        Parameters
        ----------
        solution : ConsumerSolution
            Solution to this period's problem, which must have attribute cFunc.
        Returns
        -------
        solution : ConsumerSolution
            Same solution that was passed, but now with attributes mNrmStE and
            mNrmTrg, if they exist.

        """

        # 0. Check if GICRaw holds. If so, then mNrmStE will exist. So, compute it.
        # 1. Check if GICNrm holds. If so, then mNrmTrg will exist. So, compute it.

        APF = (self.Rfree*self.DiscFacEff)**(1/self.CRRA)
        self.APF = APF

        Ex_Inv_PermShk = np.dot(1/self.PermShkValsNext, self.ShkPrbsNext)
        Inv_Ex_Inv_PermShk = 1 / Ex_Inv_PermShk
        self.PermGroFacAdj = self.PermGroFac * Inv_Ex_Inv_PermShk

        GPFRaw = APF / self.PermGroFac
        self.GPFRaw = GPFRaw

        GPFNrm = APF / self.PermGroFacAdj
        self.GPFNrm = GPFNrm

        GICRaw = 1 > APF/self.PermGroFac
        self.GICRaw = GICRaw

        GICNrm = 1 > GPFNrm
        self.GICNrm = GICNrm

        PF_RNrm = self.Rfree/self.PermGroFac
        Inv_PF_RNrm = 1 / PF_RNrm

        self.PF_RNrm = PF_RNrm
        self.Inv_PF_RNrm = Inv_PF_RNrm

        self.Ex_RNrm = PF_RNrm * Ex_Inv_PermShk
        self.Inv_Ex_PF_RNrm = Inv_Ex_Inv_PermShk

        if self.GICRaw:
            solution = self.add_mNrmStE(solution)  # find steady state m, if it exists
        if self.GICNrm:
            solution = self.add_mNrmTrg(solution)  # find target m, if it exists

        return solution

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
            mNrm, cNrm, self.cFuncLimitIntercept, self.cFuncLimitSlope
        )
        return cFunc_unconstrained

    def make_cFunc_using_EGM(self):  # Endogenous Gridpoints Method
        """
        Given a grid of end-of-period values of assets a, use the endogenous
        gridpoints method to obtain the corresponding values of consumption,
        and use the dynamic budget constraint to obtain the corresponding value
        of market resources m.

        Parameters
        ----------
        none (relies upon self.aNrmNow existing before invocation)

        Returns
        -------
        solution : ConsumerSolution
            The solution to the single period consumption-saving problem.
        """
        self.aNrmNow = self.prepare_to_calc_EndOfPrdvP()
        EndOfPrdvP = self.calc_EndOfPrdvP()

        # Construct a basic solution for this period
        if self.CubicBool:
            solution = self.make_EGM_solution(
                EndOfPrdvP, self.aNrmNow, interpolator=self.make_cubic_cFunc
            )
        else:
            solution = self.make_EGM_solution(
                EndOfPrdvP, self.aNrmNow, interpolator=self.make_linear_cFunc
            )
        return solution

    def solve(self):
        """
        Solves (one period of) the single period consumption-saving problem using the 
        method of endogenous gridpoints.  Solution includes a consumption function 
        cFunc (using cubic or linear splines), a marginal value function vPfunc, a min-
        imum acceptable level of normalized market resources mNrmMin, normalized
        human wealth hNrmNow, and bounding MPCs MPCminNow and MPCmaxNow.  It might also
        have a value function vFunc and marginal marginal value function vPPfunc.

        Parameters
        ----------
        none

        Returns
        -------
        solution : ConsumerSolution
            The solution to the single period consumption-saving problem.
        """

        solution = self.make_cFunc_using_EGM()

        self.solution_add_further_info_ConsPerfForesightSolver(solution)
#        solution = self.add_MPC_bounds_and_human_wealth_PDV(solution)
        self.solution_add_further_info(solution)

        # Add the value function if requested, as well as the marginal marginal
        # value function if cubic splines were used for interpolation
        if self.vFuncBool:
            solution = self.add_vFunc(solution, EndOfPrdvP)
        if self.CubicBool:
            solution = self.add_vPPfunc(solution)

        return solution

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

        Requires self.aNrmNow to have been computed already.

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
        def vpp_next(shocks, a_Nrm_Val):
            return shocks[0] ** (- self.CRRA - 1.0) \
                * self.vPPfuncNext(self.m_Nrm_tp1(shocks, a_Nrm_Val))

        EndOfPrdvPP = (
            self.DiscFacEff
            * self.Rfree
            * self.Rfree
            * self.PermGroFac ** (-self.CRRA - 1.0)
            * calc_expectation(
                self.IncShkDstn,
                vpp_next,
                self.aNrmNow
            )
        )
        dcda = EndOfPrdvPP / self.uPP(np.array(cNrm_Vec[1:]))
        MPC = dcda / (dcda + 1.0)
        MPC = np.insert(MPC, 0, self.MPCmaxNow)

        cFuncNowUnc = CubicInterp(
            mNrm_Vec, cNrm_Vec, MPC, self.MPCminNow * self.hNrmNow, self.MPCminNow
        )
        return cFuncNowUnc

    def make_EndOfPrdvFunc(self, EndOfPrdvP):
        """
        Construct the end-of-period value function for this period, storing it
        as an attribute of self for use by other methods.

        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal value of assets corresponding to the
            asset values in self.aNrmNow.

        Returns
        -------
        none
        """
        def v_lvl_next(shocks, a_Nrm_Val):
            return (
                shocks[0] ** (1.0 - self.CRRA)
                * self.PermGroFac ** (1.0 - self.CRRA)
            ) * self.vFuncNext(self.m_Nrm_tp1(shocks, a_Nrm_Val))
        EndOfPrdv = self.DiscFacEff * calc_expectation(
            self.IncShkDstn, v_lvl_next, self.aNrmNow
        )
        EndOfPrdvNvrs = self.uinv(
            EndOfPrdv
        )  # value transformed through inverse utility
        EndOfPrdvNvrsP = EndOfPrdvP * self.uinvP(EndOfPrdv)
        EndOfPrdvNvrs = np.insert(EndOfPrdvNvrs, 0, 0.0)
        EndOfPrdvNvrsP = np.insert(
            EndOfPrdvNvrsP, 0, EndOfPrdvNvrsP[0]
        )  # This is a very good approximation, vNvrsPP = 0 at the asset minimum
        aNrm_temp = np.insert(self.aNrmNow, 0, self.BoroCnstNat)
        EndOfPrdvNvrsFunc = CubicInterp(aNrm_temp, EndOfPrdvNvrs, EndOfPrdvNvrsP)
        self.EndOfPrdvFunc = ValueFuncCRRA(EndOfPrdvNvrsFunc, self.CRRA)

    def add_vFunc(self, solution, EndOfPrdvP):
        """
        Creates the value function for this period and adds it to the solution.

        Parameters
        ----------
        solution : ConsumerSolution
            The solution to this single period problem, likely including the
            consumption function, marginal value function, etc.
        EndOfPrdvP : np.array
            Array of end-of-period marginal value of assets corresponding to the
            asset values in self.aNrmNow.

        Returns
        -------
        solution : ConsumerSolution
            The single period solution passed as an input, but now with the
            value function (defined over market resources m) as an attribute.
        """
        self.make_EndOfPrdvFunc(EndOfPrdvP)
        solution.vFunc = self.make_vFunc(solution)
        return solution

    def make_vFunc(self, solution):
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
        vFuncNow : ValueFuncCRRA
            A representation of the value function for this period, defined over
            normalized market resources m: v = vFuncNow(m).
        """
        # Compute expected value and marginal value on a grid of market resources
        mNrm_temp = self.mNrmMinNow + self.aXtraGrid
        cNrmNow = solution.cFunc(mNrm_temp)
        aNrmNow = mNrm_temp - cNrmNow
        vNrmNow = self.u(cNrmNow) + self.EndOfPrdvFunc(aNrmNow)
        vPnow = self.uP(cNrmNow)

        # Construct the beginning-of-period value function
        vNvrs = self.uinv(vNrmNow)  # value transformed through inverse utility
        vNvrsP = vPnow * self.uinvP(vNrmNow)
        mNrm_temp = np.insert(mNrm_temp, 0, self.mNrmMinNow)
        vNvrs = np.insert(vNvrs, 0, 0.0)
        vNvrsP = np.insert(
            vNvrsP, 0, self.MPCmaxNowEff ** (-self.CRRA / (1.0 - self.CRRA))
        )
        MPCminNowNvrs = self.MPCminNow ** (-self.CRRA / (1.0 - self.CRRA))
        vNvrsFuncNow = CubicInterp(
            mNrm_temp, vNvrs, vNvrsP, MPCminNowNvrs * self.hNrmNow, MPCminNowNvrs
        )
        vFuncNow = ValueFuncCRRA(vNvrsFuncNow, self.CRRA)
        return vFuncNow

    def add_vPPfunc(self, solution):
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
        vPPfuncNow = MargMargValueFuncCRRA(solution.cFunc, self.CRRA)
        solution.vPPfunc = vPPfuncNow
        return solution


####################################################################################################
####################################################################################################


class ConsKinkedRsolver(ConsIndShockSolver):
    """
    A class to solve a single period consumption-saving problem where the interest
    rate on debt differs from the interest rate on savings.  Inherits from
    ConsIndShockSolver, with nearly identical inputs and outputs.  The key diff-
    erence is that Rfree is replaced by Rsave (a>0) and Rboro (a<0).  The solver
    can handle Rboro == Rsave, which makes it identical to ConsIndShocksolver, but
    it terminates immediately if Rboro < Rsave, as this has a different solution.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncShkDstn : distribution.Distribution
        A discrete
        approximation to the income process between the period being solved
        and the one immediately following (in solution_next).
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
        included in the reported solution.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear inter-
        polation.
    """

    def __init__(
            self,
            solution_next,
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
        # Generate a url that will locate the documentation
        self.url_doc = "https://hark.readthedocs.io/en/latest/search.html?q=" + \
            self.__class__.__name__+"&check_keywords=yes&area=default#"

        assert (
            Rboro >= Rsave
        ), "Interest factor on debt less than interest factor on savings!"

        # Initialize the solver.  Most of the steps are exactly the same as in
        # the non-kinked-R basic case, so start with that.
        ConsIndShockSolver.__init__(
            self,
            solution_next,
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
        cFuncNowUncKink = super().make_cubic_cFunc(mNrm, cNrm)

        # Change the coeffients at the kinked points.
        cFuncNowUncKink.coeffs[self.i_kink + 1] = [
            cNrm[self.i_kink],
            mNrm[self.i_kink + 1] - mNrm[self.i_kink],
            0,
            0,
        ]

        return cFuncNowUncKink

    def add_stable_points(self, solution):
        """
        TODO:
        Placeholder method for a possible future implementation of stable
        points in the kinked R model. For now it simply serves to override
        ConsIndShock's method, which does not apply here given the multiple
        interest rates.

        Discusson:
        - The target and steady state should exist under the same conditions
          as in ConsIndShock.
        - The ConsIndShock code as it stands can not be directly applied
          because it assumes that R is a constant, and in this model R depends
          on the level of wealth.
        - After allowing for wealth-depending interest rates, the existing
         code might work without modification to add the stable points. If not,
         it should be possible to find these values by checking within three
         distinct intervals:
             - From h_min to the lower kink.
             - From the lower kink to the upper kink
             - From the upper kink to infinity.
        the stable points must be in one of these regions.

        """
        return solution

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
        aNrmNow : np.array
            A 1D array of end-of-period assets; stored as attribute of self.
        """
        KinkBool = (
            self.Rboro > self.Rsave
        )  # Boolean indicating that there is actually a kink.
        # When Rboro == Rsave, this method acts just like it did in IndShock.
        # When Rboro < Rsave, the solver would have terminated when it was called.

        # Make a grid of end-of-period assets, including *two* copies of a=0
        if KinkBool:
            aNrmNow = np.sort(
                np.hstack(
                    (np.asarray(self.aXtraGrid) + self.mNrmMinNow, np.array([0.0, 0.0]))
                )
            )
        else:
            aNrmNow = np.asarray(self.aXtraGrid) + self.mNrmMinNow
            aXtraCount = aNrmNow.size

        # Make tiled versions of the assets grid and income shocks
        ShkCount = self.TranShkValsNext.size
        aNrm_temp = np.tile(aNrmNow, (ShkCount, 1))
        PermShkVals_temp = (np.tile(self.PermShkValsNext, (aXtraCount, 1))).transpose()
        TranShkVals_temp = (np.tile(self.TranShkValsNext, (aXtraCount, 1))).transpose()
        ShkPrbs_temp = (np.tile(self.ShkPrbsNext, (aXtraCount, 1))).transpose()

        # Make a 1D array of the interest factor at each asset gridpoint
        Rfree_vec = self.Rsave * np.ones(aXtraCount)
        if KinkBool:
            self.i_kink = (
                np.sum(aNrmNow <= 0) - 1
            )  # Save the index of the kink point as an attribute
            Rfree_vec[0: self.i_kink] = self.Rboro
            self.Rfree = Rfree_vec
            Rfree_temp = np.tile(Rfree_vec, (ShkCount, 1))

        # Make an array of market resources that we could have next period,
        # considering the grid of assets and the income shocks that could occur
        mNrmNext = (
            Rfree_temp / (self.PermGroFac * PermShkVals_temp) * aNrm_temp
            + TranShkVals_temp
        )

        # Recalculate the minimum MPC and human wealth using the interest factor on saving.
        # This overwrites values from set_and_update_values, which were based on Rboro instead.
        if KinkBool:
            RPFTop = (
                (self.Rsave * self.DiscFacEff) ** (1.0 / self.CRRA)
            ) / self.Rsave
            self.MPCminNow = 1.0 / (1.0 + RPFTop / solution_next.MPCminNow)
            self.hNrmNow = (
                self.PermGroFac
                / self.Rsave
                * (
                    np.dot(
                        self.ShkPrbsNext, self.TranShkValsNext * self.PermShkValsNext
                    )
                    + solution_next.hNrmNow
                )
            )

        # Store some of the constructed arrays for later use and return the assets grid
        self.PermShkVals_temp = PermShkVals_temp
        self.ShkPrbs_temp = ShkPrbs_temp
        self.mNrmNext = mNrmNext
        self.aNrmNow = aNrmNow
        return aNrmNow


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
    'T_cycle': 1,         # Number of periods in the cycle for this agent type
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
    # Optional extra facts about the model and its calibration
}

init_perfect_foresight.update(dict({'facts': {'import': 'init_perfect_foresight'}}))

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
# The simulation stage of the problem requires additional parameterization
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

# Optional more detailed facts about various parameters
CRRA_facts = {}
CRRA_facts.update({'about': 'Coefficient of Relative Risk Aversion'})
CRRA_facts.update({'latexexpr': '\providecommand{\CRRA}{\rho}\CRRA'})
CRRA_facts.update({'_unicode_': 'ρ'})  # \rho is Greek r: relative risk aversion rrr
CRRA_facts.update({'prmtv_par': 'True'})
init_perfect_foresight['prmtv_par'].append('CRRA')
init_perfect_foresight['facts'].update({'CRRA': CRRA_facts})
init_perfect_foresight.update({'CRRA_facts': CRRA_facts})

DiscFac_facts = {}
DiscFac_facts.update({'about': 'Pure time preference rate'})
DiscFac_facts.update({'latexexpr': '\providecommand{\DiscFac}{\beta}\DiscFac'})
DiscFac_facts.update({'_unicode_': 'β'})
DiscFac_facts.update({'prmtv_par': 'True'})
init_perfect_foresight['prmtv_par'].append('DiscFac')
init_perfect_foresight['facts'].update({'DiscFac': DiscFac_facts})
init_perfect_foresight.update({'DiscFac_facts': DiscFac_facts})

LivPrb_facts = {}
LivPrb_facts.update({'about': 'Probability of survival from this period to next'})
LivPrb_facts.update({'latexexpr': '\providecommand{\LivPrb}{\Pi}\LivPrb'})
LivPrb_facts.update({'_unicode_': 'Π'})  # \Pi mnemonic: 'Probability of surival'
LivPrb_facts.update({'prmtv_par': 'True'})
init_perfect_foresight['prmtv_par'].append('LivPrb')
init_perfect_foresight['facts'].update({'LivPrb': LivPrb_facts})
init_perfect_foresight.update({'LivPrb_facts': LivPrb_facts})

Rfree_facts = {}
Rfree_facts.update({'about': 'Risk free interest factor'})
Rfree_facts.update({'latexexpr': '\providecommand{\Rfree}{\mathsf{R}}\Rfree'})
Rfree_facts.update({'_unicode_': 'R'})
Rfree_facts.update({'prmtv_par': 'True'})
init_perfect_foresight['prmtv_par'].append('Rfree')
init_perfect_foresight['facts'].update({'Rfree': Rfree_facts})
init_perfect_foresight.update({'Rfree_facts': Rfree_facts})

PermGroFac_facts = {}
PermGroFac_facts.update({'about': 'Growth factor for permanent income'})
PermGroFac_facts.update({'latexexpr': '\providecommand{\PermGroFac}{\Gamma}\PermGroFac'})
PermGroFac_facts.update({'_unicode_': 'Γ'})  # \Gamma is Greek G for Growth
PermGroFac_facts.update({'prmtv_par': 'True'})
init_perfect_foresight['prmtv_par'].append('PermGroFac')
init_perfect_foresight['facts'].update({'PermGroFac': PermGroFac_facts})
init_perfect_foresight.update({'PermGroFac_facts': PermGroFac_facts})

PermGroFacAgg_facts = {}
PermGroFacAgg_facts.update({'about': 'Growth factor for aggregate permanent income'})
#PermGroFacAgg_facts.update({'latexexpr': '\providecommand{\PermGroFacAgg}{\Gamma}\PermGroFacAgg'})
# PermGroFacAgg_facts.update({'_unicode_': 'Γ'})  # \Gamma is Greek G for Growth
PermGroFacAgg_facts.update({'prmtv_par': 'True'})
init_perfect_foresight['prmtv_par'].append('PermGroFacAgg')
init_perfect_foresight['facts'].update({'PermGroFacAgg': PermGroFacAgg_facts})
init_perfect_foresight.update({'PermGroFacAgg_facts': PermGroFacAgg_facts})

BoroCnstArt_facts = {}
BoroCnstArt_facts.update(
    {'about': 'If not None, maximum proportion of permanent income borrowable'})
BoroCnstArt_facts.update({'latexexpr': r'\providecommand{\BoroCnstArt}{\underline{a}}\BoroCnstArt'})
BoroCnstArt_facts.update({'prmtv_par': 'True'})
init_perfect_foresight['prmtv_par'].append('BoroCnstArt')
init_perfect_foresight['facts'].update({'BoroCnstArt': BoroCnstArt_facts})
init_perfect_foresight.update({'BoroCnstArt_facts': BoroCnstArt_facts})

MaxKinks_facts = {}
MaxKinks_facts.update(
    {'about': 'PF Constrained model solves to period T-MaxKinks,'
     ' where the solution has exactly this many kink points'})
MaxKinks_facts.update({'prmtv_par': 'True'})
init_perfect_foresight['prmtv_par'].append('MaxKinks')
init_perfect_foresight['facts'].update({'MaxKinks': MaxKinks_facts})
init_perfect_foresight.update({'MaxKinks_facts': MaxKinks_facts})

mcrlo_AgentCount_facts = {}
mcrlo_AgentCount_facts.update(
    {'about': 'Number of agents to use in baseline Monte Carlo simulation'})
mcrlo_AgentCount_facts.update(
    {'latexexpr': '\providecommand{\mcrlo_AgentCount}{N}\mcrlo_AgentCount'})
mcrlo_AgentCount_facts.update({'mcrlo_sim': 'True'})
mcrlo_AgentCount_facts.update({'mcrlo_lim': 'infinity'})
# init_perfect_foresight['mcrlo_sim'].append('mcrlo_AgentCount')
init_perfect_foresight['facts'].update({'mcrlo_AgentCount': mcrlo_AgentCount_facts})
init_perfect_foresight.update({'mcrlo_AgentCount_facts': mcrlo_AgentCount_facts})

aNrmInitMean_facts = {}
aNrmInitMean_facts.update(
    {'about': 'Mean initial population value of aNrm'})
aNrmInitMean_facts.update({'mcrlo_sim': 'True'})
aNrmInitMean_facts.update({'mcrlo_lim': 'infinity'})
init_perfect_foresight['mcrlo_sim'].append('aNrmInitMean')
init_perfect_foresight['facts'].update({'aNrmInitMean': aNrmInitMean_facts})
init_perfect_foresight.update({'aNrmInitMean_facts': aNrmInitMean_facts})

aNrmInitStd_facts = {}
aNrmInitStd_facts.update(
    {'about': 'Std dev of initial population value of aNrm'})
aNrmInitStd_facts.update({'mcrlo_sim': 'True'})
init_perfect_foresight['mcrlo_sim'].append('aNrmInitStd')
init_perfect_foresight['facts'].update({'aNrmInitStd': aNrmInitStd_facts})
init_perfect_foresight.update({'aNrmInitStd_facts': aNrmInitStd_facts})

mcrlo_pLvlInitMean_facts = {}
mcrlo_pLvlInitMean_facts.update(
    {'about': 'Mean initial population value of log pLvl'})
mcrlo_pLvlInitMean_facts.update({'mcrlo_sim': 'True'})
init_perfect_foresight['mcrlo_sim'].append('mcrlo_pLvlInitMean')
init_perfect_foresight['facts'].update({'mcrlo_pLvlInitMean': mcrlo_pLvlInitMean_facts})
init_perfect_foresight.update({'mcrlo_pLvlInitMean_facts': mcrlo_pLvlInitMean_facts})

mcrlo_pLvlInitStd_facts = {}
mcrlo_pLvlInitStd_facts.update(
    {'about': 'Mean initial std dev of log ppLvl'})
mcrlo_pLvlInitStd_facts.update({'mcrlo_sim': 'True'})
init_perfect_foresight['mcrlo_sim'].append('mcrlo_pLvlInitStd')
init_perfect_foresight['facts'].update({'mcrlo_pLvlInitStd': mcrlo_pLvlInitStd_facts})
init_perfect_foresight.update({'mcrlo_pLvlInitStd_facts': mcrlo_pLvlInitStd_facts})

T_age_facts = {
    'about': 'Age after which simulated agents are automatically killedl'}
T_age_facts.update({'mcrlo_sim': 'False'})
init_perfect_foresight['facts'].update({'T_age': T_age_facts})
init_perfect_foresight.update({'T_age_facts': T_age_facts})

T_cycle_facts = {
    'about': 'Number of periods in a "cycle" (like, year) for this agent type'}
init_perfect_foresight['facts'].update({'T_cycle': T_cycle_facts})
init_perfect_foresight.update({'T_cycle_facts': T_cycle_facts})


class PerfForesightConsumerType(AgentType):
    """
    A perfect foresight consumer type who has no uncertainty other than mortality.
    The problem is defined by a coefficient of relative risk aversion, intertemporal
    discount factor, interest factor, an artificial borrowing constraint (maybe)
    and time sequences of the permanent income growth rate and survival probability.

    Parameters
    ----------
    cycles : int
        Number of times the sequence of periods should be solved.
    """

    # Define some universal options for all consumer types
    # In the afterlife, value is zero, consumption and all MPC's are infinity
    # This is useful chiefly because when the recursive formulae for backwards
    # computation of various objects are applied to these choices, they generate
    # the characteristics of the terminal value function betlow
    cFunc_afterlife_ = float('inf')
    vFunc_afterlife_ = 0.0
    solution_afterlife_ = ConsumerSolution(
        cFunc=cFunc_afterlife_,
        vFunc=vFunc_afterlife_,
        mNrmMin=0.0,
        hNrmNow=0.0,
        MPCminNow=float('inf'),
        MPCmaxNow=float('inf')
    )

    cFunc_terminal_ = LinearInterp([0.0, 1.0], [0.0, 1.0])  # c=m in terminal period
    vFunc_terminal_ = LinearInterp([0.0, 1.0], [0.0, 0.0])  # This is overwritten
    solution_terminal_ = ConsumerSolution(  # Also overwritten; here to show structure
        cFunc=cFunc_terminal_,
        vFunc=vFunc_terminal_,
        mNrmMin=0.0,
    )
    # This is a better name for the same thing, which should eventually replace cFunc_terminal_
    cFunc_nobequest_ = LinearInterp([0.0, 1.0], [0.0, 1.0])  # c=m in terminal period
    vFunc_nobequest_ = LinearInterp([0.0, 1.0], [0.0, 0.0])  # This is overwritten
    solution_nobequest_ = ConsumerSolution(  # Also overwritten; here to show structure
        cFunc=cFunc_nobequest_,
        vFunc=vFunc_nobequest_,
        mNrmMin=0.0,
        hNrmNow=0.0,
        MPCminNow=1.0,
        MPCmaxNow=1.0,
    )

    time_vary_ = ["LivPrb",  # Age-varying death rates can match data
                  "PermGroFac"]  # Age-varying income growth can match data
    time_inv_ = ["CRRA", "Rfree", "DiscFac", "MaxKinks", "BoroCnstArt"]
    state_vars = ['pLvl',  # Idiosyncratic permanent income
                  'PlvlAgg',  # Aggregate permanent income
                  'bNrm',  # Bank balances at beginning of period (normed)
                  'mNrm',  # Market resources (b + income) normed
                  "aNrm"]  # Assets after all actions (normed)
    shock_vars_ = []

    def __init__(self, cycles=1, verbose=1, quiet=False, **kwds):
        params = init_perfect_foresight.copy()
        params.update(kwds)
        kwds = params

        # Initialize a basic AgentType
        AgentType.__init__(
            self,
            solution_terminal=deepcopy(self.solution_terminal_),
            #            solution_afterlife=deepcopy(self.solution_afterlife_),
            cycles=cycles,
            #            pseudo_terminal=True,
            pseudo_terminal=False,
            ** kwds
        )
        self.conditions = {}
        # Extract the class name
        self.model_type = self.__class__.__name__

        # Generate a url that will locate the documentation
        self.url_doc = "https://hark.readthedocs.io/en/latest/search.html?q=" + \
            self.model_type+"&check_keywords=yes&area=default#"

        # url for reference paper that contains many results
        self.url_ref = "https://econ-ark.github.io/BufferStockTheory"

        # Add consumer-type specific objects, copying to create independent versions
        self.time_vary = deepcopy(self.time_vary_)
        self.time_inv = deepcopy(self.time_inv_)

        # Instantiate parameters that may have been passed
        self.shock_vars = deepcopy(self.shock_vars_)
        self.verbose = verbose
        set_verbosity_level((4 - verbose) * 10)
        self.quiet = quiet

        # Construct the solver
        self.solve_one_period = make_one_period_oo_solver(ConsPerfForesightSolver)

    def add_facts_to_PerfForesightConsumerType_solution(self, solution):
        # self here is the agent, whose self must have attached to it
        # solution_now and solution_next objects.
        # solution_now will be updated.
        """
        Adds to the solution a set of results useful for calculating
        and various diagnostic conditions about the problem, and stable
        points (if they exist).

        Parameters
        ----------
        solution: ConsumerSolution
            Solution a consumer's problem, which must have attribute cFunc.

        Returns
        -------
        solution : ConsumerSolution
            Same solution that was provided, augmented with the factors

        """
        urlroot = self.url_ref+'/#'
        solution.parameters_model = self.parameters

        BoroCnstArt = solution.BoroCnstArt = self.BoroCnstArt
        CRRA = solution.CRRA = self.CRRA
        DiscFac = solution.DiscFac = self.DiscFac
        LivNum = solution.LivNum = self.LivPrb[0]
        PermGroNum = solution.PermGroNum = self.PermGroFac[0]
        Rfree = solution.Rfree = self.Rfree
        DiscFacEff = solution.DiscFacEff = DiscFac * LivNum

        solution.facts = {}
        # First calculate a bunch of things that do not required
        # info about the income shocks

        uInv_Ex_uInv_PermShk = 1.0
        solution.conditions = {}

        APF_facts = {
            'about': 'Absolute Patience Factor'}
        solution.APF = APF = \
            ((Rfree * DiscFacEff) ** (1.0 / CRRA))
        APF_facts.update({'latexexpr': r'\APF'})
        APF_facts.update({'_unicode_': r'Þ'})
        APF_facts.update({'urlhandle': urlroot+'APF'})
        APF_facts.update({'py___code': '(Rfree*DiscFacEff)**(1/CRRA)'})
        APF_facts.update({'value_now': APF})
        solution.facts.update({'APF': APF_facts})
        solution.APF_facts = APF_facts

        AIC_facts = {'about': 'Absolute Impatience Condition'}
        AIC_facts.update({'latexexpr': r'\AIC'})
        AIC_facts.update({'urlhandle': urlroot+'AIC'})
        AIC_facts.update({'py___code': 'test: APF < 1'})
        solution.facts.update({'AIC': AIC_facts})
        solution.AIC_facts = AIC_facts

        RPF_facts = {
            'about': 'Return Patience Factor'}
        RPF = APF / Rfree
        RPF_facts.update({'latexexpr': r'\RPF'})
        RPF_facts.update({'_unicode_': r'Þ_R'})
        RPF_facts.update({'urlhandle': urlroot+'RPF'})
        RPF_facts.update({'py___code': r'APF/Rfree'})
        RPF_facts.update({'value_now': RPF})
        solution.facts.update({'RPF': RPF_facts})
        solution.RPF_facts = RPF_facts
        solution.RPF = RPF

        RIC_facts = {'about': 'Growth Impatience Condition'}
        RIC_facts.update({'latexexpr': r'\RIC'})
        RIC_facts.update({'urlhandle': urlroot+'RIC'})
        RIC_facts.update({'py___code': 'test: agent.RPF < 1'})
        solution.facts.update({'RIC': RIC_facts})
        solution.RIC_facts = RIC_facts

        GPFRaw_facts = {
            'about': 'Growth Patience Factor'}
        GPFRaw = APF / PermGroNum
        GPFRaw_facts.update({'latexexpr': '\GPFRaw'})
        GPFRaw_facts.update({'urlhandle': urlroot+'GPFRaw'})
        GPFRaw_facts.update({'_unicode_': r'Þ_Γ'})
        GPFRaw_facts.update({'value_now': GPFRaw})
        solution.facts.update({'GPFRaw': GPFRaw_facts})
        solution.GPFRaw_facts = GPFRaw_facts
        solution.GPFRaw = GPFRaw

        GICRaw_facts = {'about': 'Growth Impatience Condition'}
        GICRaw_facts.update({'latexexpr': r'\GICRaw'})
        GICRaw_facts.update({'urlhandle': urlroot+'GICRaw'})
        GICRaw_facts.update({'py___code': 'test: agent.GPFRaw < 1'})
        solution.facts.update({'GICRaw': GICRaw_facts})
        solution.GICRaw_facts = GICRaw_facts

        GPFLiv_facts = {
            'about': 'Mortality-Risk-Adjusted Growth Patience Factor'}
        GPFLiv = APF * LivNum / PermGroNum
        GPFLiv_facts.update({'latexexpr': '\GPFLiv'})
        GPFLiv_facts.update({'urlhandle': urlroot+'GPFLiv'})
        GPFLiv_facts.update({'py___code': 'APF*Liv/PermGroNum'})
        GPFLiv_facts.update({'value_now': GPFLiv})
        solution.facts.update({'GPFLiv': GPFLiv_facts})
        solution.GPFLiv_facts = GPFLiv_facts
        solution.GPFLiv = GPFLiv

        GICLiv_facts = {'about': 'Growth Impatience Condition'}
        GICLiv_facts.update({'latexexpr': r'\GICLiv'})
        GICLiv_facts.update({'urlhandle': urlroot+'GICLiv'})
        GICLiv_facts.update({'py___code': 'test: GPFLiv < 1'})
        solution.facts.update({'GICLiv': GICLiv_facts})
        solution.GICLiv_facts = GICLiv_facts

        PF_RNrm_facts = {
            'about': 'Growth-Normalized Perfect Foresight Return Factor'}
        PF_RNrm = Rfree/PermGroNum
        PF_RNrm_facts.update({'latexexpr': r'\PF_RNrm'})
        PF_RNrm_facts.update({'_unicode_': r'R/Γ'})
        PF_RNrm_facts.update({'py___code': r'Rfree/PermGroNum'})
        PF_RNrm_facts.update({'value_now': PF_RNrm})
        solution.facts.update({'PF_RNrm': PF_RNrm_facts})
        solution.PF_RNrm_facts = PF_RNrm_facts
        solution.PF_RNrm = PF_RNrm

        Inv_PF_RNrm_facts = {
            'about': 'Inverse of Growth-Normalized Perfect Foresight Return Factor'}
        Inv_PF_RNrm = 1/PF_RNrm
        Inv_PF_RNrm_facts.update({'latexexpr': r'\Inv_PF_RNrm'})
        Inv_PF_RNrm_facts.update({'_unicode_': r'Γ/R'})
        Inv_PF_RNrm_facts.update({'py___code': r'PermGroNumInd/Rfree'})
        Inv_PF_RNrm_facts.update({'value_now': Inv_PF_RNrm})
        solution.facts.update({'Inv_PF_RNrm': Inv_PF_RNrm_facts})
        solution.Inv_PF_RNrm_facts = Inv_PF_RNrm_facts
        solution.Inv_PF_RNrm = Inv_PF_RNrm

        FHWF_facts = {
            'about': 'Finite Human Wealth Factor'}
        FHWF = PermGroNum/Rfree
        FHWF_facts.update({'latexexpr': r'\FHWF'})
        FHWF_facts.update({'_unicode_': r'R/Γ'})
        FHWF_facts.update({'urlhandle': urlroot+'FHWF'})
        FHWF_facts.update({'py___code': r'PermGroNumInf/Rfree'})
        FHWF_facts.update({'value_now': FHWF})
        solution.facts.update({'FHWF': FHWF_facts})
        solution.FHWF_facts = FHWF_facts
        solution.FHWF = FHWF

        FHWC_facts = {'about': 'Finite Human Wealth Condition'}
        FHWC_facts.update({'latexexpr': r'\FHWC'})
        FHWC_facts.update({'urlhandle': urlroot+'FHWC'})
        FHWC_facts.update({'py___code': 'test: agent.FHWF < 1'})
        solution.facts.update({'FHWC': FHWC_facts})
        solution.FHWC_facts = FHWC_facts

        hNrmNowInf_facts = {'about':
                            'Human wealth for infinite horizon consumer'}
        hNrmNowInf = float('inf')  # default to infinity
        if FHWF < 1:  # If it is finite, set it to its value
            hNrmNowInf = 1/(1-FHWF)

        solution.hNrmNowInf = hNrmNowInf
        hNrmNowInf_facts = dict({'latexexpr': '1/(1-\FHWF)'})
        hNrmNowInf_facts.update({'value_now': hNrmNowInf})
        hNrmNowInf_facts.update({
            'py___code': '1/(1-FHWF)'})
        solution.facts.update({'hNrmNowInf': hNrmNowInf_facts})
        solution.hNrmNowInf_facts = hNrmNowInf_facts
        # solution.hNrmNowInf = hNrmNowInf

        DiscGPFRawCusp_facts = {
            'about': 'DiscFac s.t. GPFRaw = 1'}
        solution.DiscGPFRawCusp = DiscGPFRawCusp = \
            ((PermGroNum) ** (CRRA)) / (Rfree)
        DiscGPFRawCusp_facts.update({'latexexpr': ''})
        DiscGPFRawCusp_facts.update({'value_now': DiscGPFRawCusp})
        DiscGPFRawCusp_facts.update({
            'py___code': '( PermGroNum                       ** CRRA)/(Rfree)'})
        solution.facts.update({'DiscGPFRawCusp': DiscGPFRawCusp_facts})
        solution.DiscGPFRawCusp_facts = DiscGPFRawCusp_facts

        DiscGPFLivCusp_facts = {
            'about': 'DiscFac s.t. GPFLiv = 1'}
        solution.DiscGPFLivCusp = DiscGPFLivCusp = ((PermGroNum) ** (CRRA)) \
            / (Rfree * LivNum)
        DiscGPFLivCusp_facts.update({'latexexpr': ''})
        DiscGPFLivCusp_facts.update({'value_now': DiscGPFLivCusp})
        DiscGPFLivCusp_facts.update({
            'py___code': '( PermGroNum                       ** CRRA)/(Rfree*LivNum)'})
        solution.facts.update({'DiscGPFLivCusp': DiscGPFLivCusp_facts})
        solution.DiscGPFLivCusp_facts = DiscGPFLivCusp_facts

        FVAF_facts = {'about': 'Finite Value of Autarky Factor'}
        solution.FVAF = FVAF = LivNum * DiscFacEff * uInv_Ex_uInv_PermShk
        FVAF_facts.update({'latexexpr': r'\FVAFPF'})
        FVAF_facts.update({'urlhandle': urlroot+'FVAFPF'})
        solution.facts.update({'FVAF': FVAF_facts})
        solution.FVAF_facts = FVAF_facts

        FVAC_facts = {'about': 'Finite Value of Autarky Condition - Perfect Foresight'}
        FVAC_facts.update({'latexexpr': r'\FVACPF'})
        FVAC_facts.update({'urlhandle': urlroot+'FVACPF'})
        FVAC_facts.update({'py___code': 'test: FVAFPF < 1'})
        solution.facts.update({'FVAC': FVAC_facts})
        solution.FVAC_facts = FVAC_facts

        # To reduce "self." clutter in formulae, retrieve local
        # values of useful variables

        # PermGroNum and LivNum (and perhaps other time_vary items) originate
        # as lists but when we are in the one period solver they have becoome
        # scalars; to reduce confusion arising from a variable name being of a
        # different type depending on context, we make a local copy with a
        # different name
        # Call it by a different name to avoid confusion about that

        # Calculate objects whose values are built up recursively from
        # prior period's values

        hNrmNow = (
            (PermGroNum / Rfree) * (1.0 + self.solution_next.hNrmNow)
        )
        hNrmNow = PermGroNum/Rfree
        hNrmNow_facts = {'about': 'Human Wealth Now'}
        hNrmNow_facts.update({'latexexpr': r'\hNrmNow'})
#        hNrmNow_facts.update({'_unicode_': r'R/Γ'})
#        hNrmNow_facts.update({'urlhandle': urlroot+'hNrmNow'})
#        hNrmNow_facts.update({'py___code': r'PermGroNumInf/Rfree'})
#        hNrmNow_facts.update({'value_now': hNrmNow})
        solution.facts.update({'hNrmNow': hNrmNow_facts})
        solution.hNrmNow_facts = hNrmNow_facts
        self.hNrmNow = solution.hNrmNow = hNrmNow

        MPCminNow = 1.0 / (1.0 + RPF / self.solution_next.MPCminNow)
        MPCminNow_facts = {
            'about': 'Minimal MPC as m -> infty'}
        MPCminNow_facts.update({'latexexpr': r''})
        MPCminNow_facts.update({'urlhandle': urlroot+'MPCminNow'})
        MPCminNow_facts.update({'value_now': MPCminNow})
        solution.facts.update({'MPCminNow': MPCminNow_facts})
        solution.MPCminNow_facts = MPCminNow_facts
        self.MPCminNow = solution.MPCminNow = MPCminNow

        MPCmaxNow = 1.0 / (1.0 + (0.0 ** (1.0 / CRRA)) * RPF
                           / self.solution_next.MPCmaxNow)
        MPCmaxNow_facts = {
            'about': 'Maximal MPC in current period as m -> minimum'}
        MPCmaxNow_facts.update({'latexexpr': r''})
        MPCmaxNow_facts.update({'urlhandle': urlroot+'MPCmaxNow'})
        MPCmaxNow_facts.update({'value_now': MPCmaxNow})
        solution.facts.update({'MPCmaxNow': MPCmaxNow_facts})
        solution.MPCmaxNow_facts = MPCmaxNow_facts
        solution.MPCmaxNow = MPCmaxNow

        # Lower bound of aggregate wealth growth if all inheritances squandered
        cFuncLimitIntercept = MPCminNow * hNrmNow
        cFuncLimitIntercept_facts = {
            'about': 'Vertical intercept of perfect foresight consumption function'}
        cFuncLimitIntercept_facts.update({'latexexpr': '\MPC '})
        cFuncLimitIntercept_facts.update({'urlhandle': ''})
        cFuncLimitIntercept_facts.update({'value_now': cFuncLimitIntercept})
        cFuncLimitIntercept_facts.update({
            'py___code': 'MPCminNow * hNrmNow'})
        solution.facts.update({'cFuncLimitIntercept': cFuncLimitIntercept_facts})
        solution.cFuncLimitIntercept_facts = cFuncLimitIntercept_facts
        solution.cFuncLimitIntercept = cFuncLimitIntercept

        cFuncLimitSlope = MPCminNow
        cFuncLimitSlope_facts = {
            'about': 'Slope of limiting consumption function'}
        cFuncLimitSlope_facts = dict({'latexexpr': '\MPC \hNrmNow'})
        cFuncLimitSlope_facts.update({'urlhandle': ''})
        cFuncLimitSlope_facts.update({'value_now': cFuncLimitSlope})
        cFuncLimitSlope_facts.update({
            'py___code': 'MPCminNow * hNrmNow'})
        solution.facts.update({'cFuncLimitSlope': cFuncLimitSlope_facts})
        solution.cFuncLimitSlope_facts = cFuncLimitSlope_facts
        solution.cFuncLimitSlope = cFuncLimitSlope

        # We are in the perfect foresight model now so these are all 1.0
        Ex_Inv_PermShk = 1.0
        Inv_Ex_PermShk_Inv = 1.0
        Ex_uInv_PermShk = 1.0
        uInv_Ex_uInv_PermShk = 1.0

        # These formulae do not require "live" computation of expectations
        # from a distribution that is on hand.  So, having constructed
        # expected values above, we can use those below.

        # This allows sharing these formulae between the perfect foresight
        # and the non-perfect-foresight models.  They are constructed here
        # and inherited by the descendant model, which augments them with
        # the objects that require live calculation.

        if solution.Inv_PF_RNrm < 1:        # Finite if Rfree > PermGroNum
            solution.hNrmNowInf = 1/(1-solution.Inv_PF_RNrm)

        # Given m, value of c where E[mLev_{t+1}/mLev_{t}]=PermGroNum
        # Solves for c in equation at url/#balgrostable

        solution.c_where_Ex_PermShk_times_mtp1_minus_mt_eq_0 = (
            lambda m_t:
            m_t * (1 - solution.Inv_PF_RNrm) + solution.Inv_PF_RNrm
        )

        solution.Ex_cLev_tp1_Over_cLev_t_from_mt = (
            lambda m_t:
            solution.Ex_cLev_tp1_Over_pLev_t_from_mt(solution,
                                                     m_t - solution.cFunc(m_t))
            / solution.cFunc(m_t)
        )

#        # E[m_{t+1} pLev_{t+1}/pLev_{t}] as a fn of a_{t}
        solution.Ex_mLev_tp1_Over_pLev_t_from_at = (
            lambda a_t:
                solution.PermGroNum *
            (solution.PF_RNrm * a_t + solution.Ex_IncNextNrm)
        )

        # E[m_{t+1} pLev_{t+1}/(m_{t}pLev_{t})] as a fn of m_{t}
        solution.Ex_mLev_tp1_Over_mLev_t_from_at = (
            lambda m_t:
                solution.Ex_mLev_tp1_Over_pLev_t_from_at(solution,
                                                         m_t-solution.cFunc(m_t)
                                                         )/m_t
        )

        return solution

    def add_mNrmStE():
        """
        Finds value of (normalized) market resources m at which individual consumer
        expects m not to change.

        This will exist if the GICNrm holds.

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

        # Minimum market resources plus next income is okay starting guess
        m_init_guess = solution.mNrmMin + solution.Ex_IncNextNrm
        try:
            m_t = newton(
                self.Ex_m_tp1_minus_m_t,
                m_init_guess)
        except:
            m_t = None

        # Add mNrmTrg to the solution and return it
        mNrmTrg = m_t

    def pre_solve(self):
        if not hasattr(self, "BoroCnstArt"):  # If BoroCnst specified...
            self.BoroCnstArt = None  # ...assume the user wanted none
            if not hasattr(self, "MaxKinks"):
                if self.cycles > 0:  # If it's not an infinite horizon model...
                    self.MaxKinks = np.inf  # ...there's no need to set MaxKinks
                elif self.BoroCnstArt is None:  # If there's no borrowing cnst...
                    self.MaxKinks = np.inf  # ...there's no need to set MaxKinks
                else:
                    raise (
                        AttributeError(
                            "PerfForesightConsumerType requires MaxKinks when BoroCnstArt is not None, cycles == 0."
                        )
                    )
        # Trick: First time it is run, tolerance will be default
        # Set it to infinity so that it will run for one period
        # This will cause the terminal period to be calculated from the afterlife

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

        if self.TranShkStd < 0:
            raise Exception("TranShkStd is negative with value: " + str(self.TranShkStd))

        if self.PermShkStd < 0:
            raise Exception("PermShkStd is negative with value: " + str(self.PermShkStd))

        if self.IncUnemp < 0:
            raise Exception("IncUnemp is negative with value: " + str(self.IncUnemp))

        if self.IncUnempRet < 0:
            raise Exception("IncUnempRet is negative with value: " + str(self.IncUnempRet))

        if self.CRRA < 1:
            raise Exception("CRRA is less than 1 with value: " + str(self.CRRA))

        return

    def update_solution_terminal(self):
        """
        Update the terminal period solution.  This method is run when a
        new AgentType is created or when CRRA changes.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        # Value in last period is obtained by consuming everything
        self.solution_terminal.cFunc = self.cFunc_nobequest_
        self.solution_terminal.vFunc = ValueFuncCRRA(self.cFunc_nobequest_, self.CRRA)
        self.solution_terminal.vPfunc = MargValueFuncCRRA(self.cFunc_nobequest_, self.CRRA)
        self.solution_terminal.vPPfunc = MargMargValueFuncCRRA(
            self.cFunc_nobequest_, self.CRRA)
        self.solution_terminal.MPCminNow = 1.0
        self.solution_terminal.MPCmaxNow = 1.0
        self.solution_terminal.hNrmNow = 0.0
        self.solution_terminal.mNrmMin = 0.0

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
        self.PermShkAggNow = self.PermGroFacAgg  # Never changes during sim
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
        mcrlo_pLvlInitMeanNow = self.mcrlo_pLvlInitMean + np.log(
            self.state_now['PlvlAgg']
        )  # Account for newer cohorts having higher permanent income
        self.state_now['pLvl'][which_agents] = Lognormal(
            mcrlo_pLvlInitMeanNow,
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
        perfect foresight model, there are no stochastic shocks: PermShkNow = PermGroFac for each
        agent (according to their t_cycle) and TranShkNow = 1.0 for all agents.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        PermGroFac = np.array(self.PermGroFac)
        self.shocks['PermShk'] = PermGroFac[
            self.t_cycle - 1
        ]  # cycle time has already been advanced
        self.shocks['TranShk'] = np.ones(self.mcrlo_AgentCount)

    def get_Rfree(self):
        """
        Returns an array of size self.mcrlo_AgentCount with self.Rfree in every entry.

        Parameters
        ----------
        None

        Returns
        -------
        RfreeNow : np.array
            Array of size self.mcrlo_AgentCount with risk free interest rate for each agent.
        """
        RfreeNow = self.Rfree * np.ones(self.mcrlo_AgentCount)
        return RfreeNow

    def transition(self):
        pLvlPrev = self.state_prev['pLvl']
        aNrmPrev = self.state_prev['aNrm']
        RfreeNow = self.get_Rfree()

        # Calculate new states: normalized market resources and permanent income level
        pLvlNow = pLvlPrev*self.shocks['PermShk']  # Updated permanent income level
        # Updated aggregate permanent productivity level
        PlvlAggNow = self.state_prev['PlvlAgg']*self.PermShkAggNow
        # "Effective" interest factor on normalized assets
        ReffNow = RfreeNow/self.shocks['PermShk']
        bNrmNow = ReffNow*aNrmPrev         # Bank balances before labor income
        mNrmNow = bNrmNow + self.shocks['TranShk']  # Market resources after income

        return pLvlNow, PlvlAggNow, bNrmNow, mNrmNow, None

    def get_controls(self):
        """
        Calculates consumption for each consumer of this type using the consumption functions.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        cNrmNow = np.zeros(self.mcrlo_AgentCount) + np.nan
        MPCnow = np.zeros(self.mcrlo_AgentCount) + np.nan
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            cNrmNow[these], MPCnow[these] = self.solution[t].cFunc.eval_with_derivative(
                self.state_now['mNrm'][these]
            )
            self.controls['cNrm'] = cNrmNow

        # MPCnow is not really a control
        self.MPCnow = MPCnow
        return None

    def get_poststates(self):
        """
        Calculates end-of-period assets for each consumer of this type.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # should this be "Now", or "Prev"?!?
        self.state_now['aNrm'] = self.state_now['mNrm'] - self.controls['cNrm']
        # Useful in some cases to precalculate asset level
        self.state_now['aLvl'] = self.state_now['aNrm'] * self.state_now['pLvl']

        # moves now to prev
        super().get_poststates()

        return None

    def check_condition(self, name, test, messages, verbose, verbose_messages=None):
        """
        Checks one condition.

        Parameters
        ----------
        name : string
             Name for the condition.

        test : function(self -> boolean)
             A function (of self) which tests the condition

        messages : dict{boolean : string}
            A dictiomary with boolean keys containing values
            for messages to print if the condition is
            true or false.

        verbose_messages : dict{boolean : string}
            (Optional) A dictiomary with boolean keys containing values
            for messages to print if the condition is
            true or false under verbose printing.
        """
        self.conditions[name] = test(self)
        set_verbosity_level((4 - verbose) * 10)
        _log.info(messages[self.conditions[name]].format(self))
        if verbose_messages:
            _log.debug(verbose_messages[self.conditions[name]].format(self))

    def check_condtnnew(self, name, test, messages, verbose, verbose_messages, fact):
        """
        Checks one condition.

        Parameters
        ----------
        name : string
             Name for the condition.

        fact : string
             Name of the factor whose value should be less than 1

        test : function(self -> boolean)
             A function (of self) which tests the condition

        messages : dict{boolean : string}
            A dictiomary with boolean keys containing values
            for messages to print if the condition is
            true or false.

        verbose_messages : dict{boolean : string}
            (Optional) A dictiomary with boolean keys containing values
            for messages to print if the condition is
            true or false under verbose printing.
        """
        self.solution_curr.conditions[name] = test(self)
        set_verbosity_level((4 - verbose) * 10)
        self.solution_curr.conditions[fact] = (
            messages[self.solution_curr.conditions[name]] +
            verbose_messages[self.solution_curr.conditions[name]]).format(self.solution_curr)
        print(self.solution_curr.conditions[fact])
        _log.info((
            messages[self.solution_curr.conditions[name]] +
            verbose_messages[self.solution_curr.conditions[name]]).format(self.solution_curr)
        )

    def check_AIC(self, verbose=None):
        """
        Evaluate and report on the Absolute Impatience Condition
        """
        name = "AIC"
        fact = "APF"

        def test(agent): return agent.solution_curr.APF < 1

        messages = {
            True: "\nThe Absolute Patience Factor for the supplied parameter values, APF={0.APF}, satisfies the Absolute Impatience Condition (AIC), which requires APF < 1: "+self.solution_curr.AIC_facts['urlhandle'],
            False: "\nThe Absolute Patience Factor for the supplied parameter values, APF={0.APF}, violates the Absolute Impatience Condition (AIC), which requires APF < 1: "+self.solution_curr.AIC_facts['urlhandle']
        }
        verbose_messages = {
            True: "  Because the APF < 1,  the absolute amount of consumption is expected to fall over time.  \n",
            False: "  Because the APF > 1, the absolute amount of consumption is expected to grow over time.  \n",
        }
        verbose = self.verbose if verbose is None else verbose
        self.check_condtnnew(name, test, messages, verbose, verbose_messages, fact)

    def check_FVAC(self, verbose=None):
        """
        Evaluate and report on the Finite Value of Autarky Condition
        """
        name = "FVAC"
        fact = "FVAF"

        def test(agent): return agent.solution_curr.FVAF < 1

        messages = {
            True: "\nThe Finite Value of Autarky Factor for the supplied parameter values, FVAF={0.FVAF}, satisfies the Finite Value of Autarky Condition, which requires FVAF < 1: "+self.solution_curr.FVAC_facts['urlhandle'],
            False: "\nThe Finite Value of Autarky Factor for the supplied parameter values, FVAF={0.FVAF}, violates the Finite Value of Autarky Condition, which requires FVAF: "+self.solution_curr.FVAC_facts['urlhandle']
        }
        verbose_messages = {
            True: "  Therefore, a nondegenerate solution exists if the RIC also holds. ("+self.solution_curr.FVAC_facts['urlhandle']+")\n",
            False: "  Therefore, a nondegenerate solution exits if the RIC holds.\n",
        }

        verbose = self.verbose if verbose is None else verbose
        self.check_condtnnew(name, test, messages, verbose, verbose_messages, fact)

    def check_GICRaw(self, verbose=None):
        """
        Evaluate and report on the Growth Impatience Condition
        """
        name = "GICRaw"
        fact = "GPFRaw"

        def test(agent): return agent.solution_curr.GPFRaw < 1

        messages = {
            True: "\nThe Growth Patience Factor for the supplied parameter values, GPF={0.GPFRaw}, satisfies the Growth Impatience Condition (GIC), which requires GPF < 1: "+self.solution_curr.GICRaw_facts['urlhandle'],
            False: "\nThe Growth Patience Factor for the supplied parameter values, GPF={0.GPFRaw}, violates the Growth Impatience Condition (GIC), which requires GPF < 1: "+self.solution_curr.GICRaw_facts['urlhandle'],
        }
        verbose_messages = {
            True: "  Therefore,  for a perfect foresight consumer, the ratio of individual wealth to permanent income is expected to fall indefinitely.    \n",
            False: "  Therefore, for a perfect foresight consumer, the ratio of individual wealth to permanent income is expected to rise toward infinity. \n"
        }

        verbose = self.verbose if verbose is None else verbose
        self.check_condtnnew(name, test, messages, verbose, verbose_messages, fact)

    def check_GICLiv(self, verbose=None):
        name = "GICLiv"
        fact = "GPFLiv"

        def test(agent): return agent.solution_curr.GPFLiv < 1

        messages = {
            True: "\nThe Mortality Adjusted Aggregate Growth Patience Factor for the supplied parameter values, GPFLiv={0.GPFLiv}, satisfies the Mortality Adjusted Aggregate Growth Imatience Condition (GICLiv): "+self.solution_curr.GPFLiv_facts['urlhandle'],
            False: "\nThe Mortality Adjusted Aggregate Growth Patience Factor for the supplied parameter values, GPFLiv={0.GPFLiv}, violates the Mortality Adjusted Aggregate Growth Imatience Condition (GICLiv): "+self.solution_curr.GPFLiv_facts['urlhandle'],
        }
        verbose_messages = {
            True: "  Therefore, a target level of the ratio of aggregate market resources to aggregate permanent income exists ("+self.solution_curr.GPFLiv_facts['urlhandle']+")\n",
            False: "  Therefore, a target ratio of aggregate resources to aggregate permanent income may not exist ("+self.solution_curr.GPFLiv_facts['urlhandle']+")\n",
        }
        verbose = self.verbose if verbose is None else verbose
        self.check_condtnnew(name, test, messages, verbose, verbose_messages, fact)

    def check_RIC(self, verbose=None):
        """
        Evaluate and report on the Return Impatience Condition
        """

        name = "RIC"
        fact = "RPF"

        def test(agent): return agent.solution_curr.RPF < 1

        messages = {
            True: "\nThe Return Patience Factor for the supplied parameter values, RPF= {0.RPF}, satisfies the Return Impatience Condition (RIC), which requires RPF < 1: "+self.solution_curr.RPF_facts['urlhandle'],
            False: "\nThe Return Patience Factor for the supplied parameter values, RPF= {0.RPF}, violates the Return Impatience Condition (RIC), which requires RPF < 1: "+self.solution_curr.RPF_facts['urlhandle'],
        }
        verbose_messages = {
            True: "  Therefore, the limiting consumption function is not c(m)=0 for all m\n",
            False: "  Therefore, if the FHWC is satisfied, the limiting consumption function is c(m)=0 for all m.\n",
        }
        verbose = self.verbose if verbose is None else verbose
        self.check_condtnnew(name, test, messages, verbose, verbose_messages, fact)

    def check_FHWC(self, verbose=None):
        """
        Evaluate and report on the Finite Human Wealth Condition
        """
        name = "FHWC"
        fact = "FHWF"

        def test(agent): return agent.solution_curr.FHWF < 1

        messages = {
            True: "\nThe Finite Human Wealth Factor value for the supplied parameter values, FHWF={0.FHWF}, satisfies the Finite Human Wealth Condition (FHWC), which requires FHWF < 1: "+self.solution_curr.FHWC_facts['urlhandle'],
            False: "\nThe Finite Human Wealth Factor value for the supplied parameter values, FHWF={0.FHWF}, violates the Finite Human Wealth Condition (FHWC), which requires FHWF < 1: "+self.solution_curr.FHWC_facts['urlhandle'],
        }
        verbose_messages = {
            True: "  Therefore, the limiting consumption function is not c(m)=Infinity ("+self.solution_curr.FHWC_facts['urlhandle']+")\n  Human wealth normalized by permanent income is {0.hNrmNowInf}.\n",
            False: "  Therefore, the limiting consumption function is c(m)=Infinity for all m unless the RIC is also violated.\n  If both FHWC and RIC fail and the consumer faces a liquidity constraint, the limiting consumption function is nondegenerate but has a limiting slope of 0. ("+self.solution_curr.FHWC_facts['urlhandle']+")\n",
        }
        verbose = self.verbose if verbose is None else verbose
        self.check_condtnnew(name, test, messages, verbose, verbose_messages, fact)

    def check_condtnsnew(self, verbose=None):
        """
        This method checks whether the instance's type satisfies the
        Absolute Impatience Condition (AIC),
        the Return Impatience Condition (RIC),
        the Finite Human Wealth Condition (FHWC), the perfect foresight
        model's Growth Impatience Condition (GICRaw) and
        Perfect Foresight Finite Value of Autarky Condition (FVACPF). Depending on the configuration of parameter values, some
        combination of these conditions must be satisfied in order for the problem to have
        a nondegenerate solution. To check which conditions are required, in the verbose mode
        a reference to the relevant theoretical literature is made.

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
        self.solution_curr.conditions = {}

        self.solution_curr.violated = False

        # This method only checks for the conditions for infinite horizon models
        # with a 1 period cycle. If these conditions are not met, we exit early.
        if self.cycles != 0 or self.T_cycle > 1:
            return

        verbose = self.verbose if verbose is None else verbose
        self.check_AIC(verbose)
        self.check_FHWC(verbose)
        self.check_RIC(verbose)
        self.check_GICRaw(verbose)
        self.check_GICLiv(verbose)
        self.check_FVAC(verbose)

        if hasattr(self, "BoroCnstArt") and self.solution_curr.BoroCnstArt is not None:
            self.solution_curr.violated = not self.solution_curr.conditions["RIC"]
        else:
            self.solution_curr.violated = not self.solution_curr.conditions[
                "RIC"] or not self.solution_curr.conditions["FHWC"]


# Make a dictionary to specify an idiosyncratic income shocks consumer
init_idiosyncratic_shocks = dict(
    init_perfect_foresight,
    **{
        # Income process variables
        "PermShkStd": [0.1],  # Standard deviation of log permanent income shocks
        "TranShkStd": [0.1],  # Standard deviation of log transitory income shocks
        "UnempPrb": 0.05,  # Probability of unemployment while working
        "UnempPrbRet": 0.005,  # Probability of "unemployment" while retired
        "IncUnemp": 0.3,  # Unemployment benefits replacement rate
        "IncUnempRet": 0.0,  # "Unemployment" benefits when retired
        "BoroCnstArt": 0.0,  # Artificial borrowing constraint; imposed minimum level of end-of period assets
        "tax_rate": 0.0,  # Flat income tax rate
        "T_retire": 0,  # Period of retirement (0 --> no retirement)
        # Parameters governing construction of income process
        "PermShkCount": 7,  # Number of points in discrete approximation to permanent income shocks
        "TranShkCount": 7,  # Number of points in discrete approximation to transitory income shocks
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

# Now add parameters that were not part of perfect foresight model
# Primitives
init_idiosyncratic_shocks['prmtv_par'].append('PermShkStd')
init_idiosyncratic_shocks['prmtv_par'].append('TranShkStd')
init_idiosyncratic_shocks['prmtv_par'].append('UnempPrb')
init_idiosyncratic_shocks['prmtv_par'].append('UnempPrbRet')
init_idiosyncratic_shocks['prmtv_par'].append('IncUnempRet')
init_idiosyncratic_shocks['prmtv_par'].append('BoroCnstArt')
init_idiosyncratic_shocks['prmtv_par'].append('tax_rate')
init_idiosyncratic_shocks['prmtv_par'].append('T_retire')

# Approximation parameters and their limits (if any)
# init_idiosyncratic_shocks['aprox_par'].append('PermShkCount')
init_idiosyncratic_shocks['aprox_lim'].update({'PermShkCount': 'infinity'})
# init_idiosyncratic_shocks['aprox_par'].append('TranShkCount')
init_idiosyncratic_shocks['aprox_lim'].update({'TranShkCount': 'infinity'})
# init_idiosyncratic_shocks['aprox_par'].append('aXtraMin')
init_idiosyncratic_shocks['aprox_lim'].update({'aXtraMin': float('0.0')})
# init_idiosyncratic_shocks['aprox_par'].append('aXtraMax')
init_idiosyncratic_shocks['aprox_lim'].update({'aXtraMax': float('inf')})
# init_idiosyncratic_shocks['aprox_par'].append('aXtraNestFac')
init_idiosyncratic_shocks['aprox_lim'].update({'aXtraNestFac': None})
# init_idiosyncratic_shocks['aprox_par'].append('aXtraCount')
init_idiosyncratic_shocks['aprox_lim'].update({'aXtraCount': None})

# # Auxiliary parameters
# init_idiosyncratic_shocks['auxiliary'].append('vFuncBool')
# init_idiosyncratic_shocks['auxiliary'].append('CubicBool')

PermShkStd_facts = {}
PermShkStd_facts.update({'about': 'Standard deviation for lognormal shock to permanent income'})
PermShkStd_facts.update({'latexexpr': '\PermShkStd'})
init_idiosyncratic_shocks['facts'].update({'PermShkStd': PermShkStd_facts})
init_idiosyncratic_shocks.update({'PermShkStd_facts': PermShkStd_facts})

TranShkStd_facts = {}
TranShkStd_facts.update({'about': 'Standard deviation for lognormal shock to permanent income'})
TranShkStd_facts.update({'latexexpr': '\TranShkStd'})
init_idiosyncratic_shocks['facts'].update({'TranShkStd': TranShkStd_facts})
init_idiosyncratic_shocks.update({'TranShkStd_facts': TranShkStd_facts})

UnempPrb_facts = {}
UnempPrb_facts.update({'about': 'Probability of unemployment while working'})
UnempPrb_facts.update({'latexexpr': r'\UnempPrb'})
UnempPrb_facts.update({'_unicode_': '℘'})
init_idiosyncratic_shocks['facts'].update({'UnempPrb': UnempPrb_facts})
init_idiosyncratic_shocks.update({'UnempPrb_facts': UnempPrb_facts})

UnempPrbRet_facts = {}
UnempPrbRet_facts.update({'about': '"unemployment" in retirement = big medical shock'})
UnempPrbRet_facts.update({'latexexpr': r'\UnempPrbRet'})
init_idiosyncratic_shocks['facts'].update({'UnempPrbRet': UnempPrbRet_facts})
init_idiosyncratic_shocks.update({'UnempPrbRet_facts': UnempPrbRet_facts})

IncUnemp_facts = {}
IncUnemp_facts.update({'about': 'Unemployment insurance replacement rate'})
IncUnemp_facts.update({'latexexpr': '\IncUnemp'})
IncUnemp_facts.update({'_unicode_': 'μ'})
init_idiosyncratic_shocks['facts'].update({'IncUnemp': IncUnemp_facts})
init_idiosyncratic_shocks.update({'IncUnemp_facts': IncUnemp_facts})

IncUnempRet_facts = {}
IncUnempRet_facts.update({'about': 'Size of medical shock (frac of perm inc)'})
init_idiosyncratic_shocks['facts'].update({'IncUnempRet': IncUnempRet_facts})
init_idiosyncratic_shocks.update({'IncUnempRet_facts': IncUnempRet_facts})

tax_rate_facts = {}
tax_rate_facts.update({'about': 'Flat income tax rate'})
tax_rate_facts.update({'about': 'Size of medical shock (frac of perm inc)'})
init_idiosyncratic_shocks['facts'].update({'tax_rate': tax_rate_facts})
init_idiosyncratic_shocks.update({'tax_rate_facts': tax_rate_facts})

T_retire_facts = {}
T_retire_facts.update({'about': 'Period of retirement (0 --> no retirement)'})
init_idiosyncratic_shocks['facts'].update({'T_retire': T_retire_facts})
init_idiosyncratic_shocks.update({'T_retire_facts': T_retire_facts})

PermShkCount_facts = {}
PermShkCount_facts.update({'about': 'Num of pts in discrete approx to permanent income shock dstn'})
init_idiosyncratic_shocks['facts'].update({'PermShkCount': PermShkCount_facts})
init_idiosyncratic_shocks.update({'PermShkCount_facts': PermShkCount_facts})

TranShkCount_facts = {}
TranShkCount_facts.update(
    {'about': 'Num of pts in discrete approx to transitory income shock dstn'})
init_idiosyncratic_shocks['facts'].update({'TranShkCount': TranShkCount_facts})
init_idiosyncratic_shocks.update({'TranShkCount_facts': TranShkCount_facts})

aXtraMin_facts = {}
aXtraMin_facts.update(
    {'about': 'Minimum end-of-period "assets above minimum" value'})
init_idiosyncratic_shocks['facts'].update({'aXtraMin': aXtraMin_facts})
init_idiosyncratic_shocks.update({'aXtraMin_facts': aXtraMin_facts})

aXtraMax_facts = {}
aXtraMax_facts.update(
    {'about': 'Maximum end-of-period "assets above minimum" value'})
init_idiosyncratic_shocks['facts'].update({'aXtraMax': aXtraMax_facts})
init_idiosyncratic_shocks.update({'aXtraMax_facts': aXtraMax_facts})

aXtraNestFac_facts = {}
aXtraNestFac_facts.update(
    {'about': 'Exponential nesting factor when constructing "assets above minimum" grid'})
init_idiosyncratic_shocks['facts'].update({'aXtraMax': aXtraNestFac_facts})
init_idiosyncratic_shocks.update({'aXtraMax_facts': aXtraNestFac_facts})

aXtraCount_facts = {}
aXtraCount_facts.update(
    {'about': 'Number of points in the grid of "assets above minimum"'})
init_idiosyncratic_shocks['facts'].update({'aXtraMax': aXtraCount_facts})
init_idiosyncratic_shocks.update({'aXtraMax_facts': aXtraCount_facts})

aXtraCount_facts = {}
aXtraCount_facts.update(
    {'about': 'Number of points to include in grid of assets above minimum possible'})
init_idiosyncratic_shocks['facts'].update({'aXtraCount': aXtraCount_facts})
init_idiosyncratic_shocks.update({'aXtraCount_facts': aXtraCount_facts})

aXtraExtra_facts = {}
aXtraExtra_facts.update(
    {'about': 'List of other values of "assets above minimum" to add to the grid (e.g., 10000)'})
init_idiosyncratic_shocks['facts'].update({'aXtraExtra': aXtraExtra_facts})
init_idiosyncratic_shocks.update({'aXtraExtra_facts': aXtraExtra_facts})

aXtraGrid_facts = {}
aXtraGrid_facts.update(
    {'about': 'Grid of values to add to minimum possible value to obtain actual end-of-period asset grid'})
init_idiosyncratic_shocks['facts'].update({'aXtraGrid': aXtraGrid_facts})
init_idiosyncratic_shocks.update({'aXtraGrid_facts': aXtraGrid_facts})

vFuncBool_facts = {}
vFuncBool_facts.update(
    {'about': 'Whether to calculate the value function during solution'})
init_idiosyncratic_shocks['facts'].update({'vFuncBool': vFuncBool_facts})
init_idiosyncratic_shocks.update({'vFuncBool_facts': vFuncBool_facts})

CubicBool_facts = {}
CubicBool_facts.update(
    {'about': 'Use cubic spline interpolation when True, linear interpolation when False'})
init_idiosyncratic_shocks['facts'].update({'CubicBool': CubicBool_facts})
init_idiosyncratic_shocks.update({'CubicBool_facts': CubicBool_facts})


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
    """

    # Time invariante parameters
    time_inv_ = PerfForesightConsumerType.time_inv_ + [
        "vFuncBool",
        "CubicBool",
    ]
    time_inv_.remove(  # Remove items inherited from PerfForesight but not IndShock
        "MaxKinks"  # PF inf hor with MaxKinks is equiv to fin hor with hor=MaxKinks
    )

    shock_vars_ = ['PermShk', 'TranShk']  # The unemployment shock is transitory

    def __init__(self, cycles=1, verbose=1, quiet=False, **kwds):
        params = init_idiosyncratic_shocks.copy()

        # Update them with any customizations the user has chosen
        params.update(kwds)  # This gets all params, not just those in the dict

        # Inherit characteristics of a perfect foresight model
        PerfForesightConsumerType.__init__(
            self, cycles=cycles, verbose=verbose, quiet=quiet, **params
        )

        # Extract the class name so that we can ...
        self.model_type = self.__class__.__name__

        # ... generate a url that will locate the documentation:
        self.url_doc_model_type = "https://hark.readthedocs.io/en/latest/search.html?q=" + \
            self.model_type+"&check_keywords=yes&area=default#"

        # Define a reference to a paper that contains the main results
        self.url_ref = "https://econ-ark.github.io/BufferStockTheory"

        # Add model_type and doc url to auto-generated self.parameters
        self.parameters.update({"model_type": self.model_type})
        self.parameters.update({"url_doc_model_type": self.url_doc_model_type})
        self.parameters.update({"url_ref": self.url_ref})

        # Add consumer-type specific objects, copying to create independent versions
        # - Default interpolation method is piecewise linear
        # - Cubic is smoother, works well if problem has no constraints
        # - User may or may not want to create the value function
        if (not self.CubicBool) and (not self.vFuncBool):
            solver = ConsIndShockSolverBasic
        else:  # Use the "advanced" solver if either is requested
            solver = ConsIndShockSolver

        # Attach the corresponding one period solver to the agent
        self.solve_one_period = make_one_period_oo_solver(solver)

        # Let it know about all the parameters of the model
        self.solve_one_period.parameters_model = self.parameters

        # Quiet mode: Define model without calculating anything
        # If not quiet, solve one period so we can check conditions
        if not quiet:
            self.update()  # Otherwise, build last period soln with lots of info
            self.tolerance_orig = self.tolerance  # preserve true tolerance
            self.tolerance = float('inf')  # tolerance is infiniy ...
            self.solve()  # ... means that "solve" will stop after one period
            self.tolerance = self.tolerance_orig  # restore original tolerance
            self.check_condtns()  # Check conditions for nature/existence of soln

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
        (
            IncShkDstn,
            PermShkDstn,
            TranShkDstn,
        ) = self.construct_lognormal_income_process_unemployment()
        self.IncShkDstn = IncShkDstn
        self.PermShkDstn = PermShkDstn
        self.TranShkDstn = TranShkDstn
        self.add_to_time_vary("IncShkDstn", "PermShkDstn", "TranShkDstn")

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
        aXtraGrid = construct_assets_grid(self)
        self.aXtraGrid = aXtraGrid
        self.add_to_time_inv("aXtraGrid")

    def update(self):
        """
        Update the income process, the assets grid, and the terminal solution.

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
            _log.info('')
            _log.info('Solver inherited these parameters:')
            _log.info('')
            _log.info(self.solve_one_period.parameters_model.keys())
            _log.info('')
            _log.info('from model_type '+self.solve_one_period.parameters_model['model_type'])
            _log.info('')

        self.update_income_process()
        self.update_assets_grid()
        self.update_solution_terminal()

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

    def calc_bounding_values(self):
        """
        Calculate human wealth plus minimum and maximum MPC in an infinite
        horizon model with only one period repeated indefinitely.  Store results
        as attributes of self.  Human wealth is the present discounted value of
        expected future income after receiving income this period, ignoring mort-
        ality (because your income matters to you only if you are still alive).
        The maximum MPC is the limit of the MPC as m --> mNrmMin.  The
        minimum MPC is the limit of the MPC as m --> infty.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Unpack the income distribution and get average and worst outcomes
        PermShkValsNext = self.IncShkDstn[0][1]
        TranShkValsNext = self.IncShkDstn[0][2]
        ShkPrbsNext = self.IncShkDstn[0][0]
        Ex_IncNextNrm = np.dot(ShkPrbsNext, PermShkValsNext * TranShkValsNext)
        PermShkMinNext = np.min(PermShkValsNext)
        TranShkMinNext = np.min(TranShkValsNext)
        WorstIncNext = PermShkMinNext * TranShkMinNext
        WorstIncPrb = np.sum(
            ShkPrbsNext[(PermShkValsNext * TranShkValsNext) == WorstIncNext]
        )

        # Calculate human wealth and the infinite horizon natural borrowing constraint
        hNrmNow = (Ex_IncNextNrm * self.PermGroFac[0] / self.Rfree) / (
            1.0 - self.PermGroFac[0] / self.Rfree
        )
        temp = self.PermGroFac[0] * PermShkMinNext / self.Rfree
        BoroCnstNat = -TranShkMinNext * temp / (1.0 - temp)

        RPF = (self.DiscFac * self.LivPrb[0] * self.Rfree) ** (
            1.0 / self.CRRA
        ) / self.Rfree
        if BoroCnstNat < self.BoroCnstArt:
            MPCmaxNow = 1.0  # if natural borrowing constraint is overridden by artificial one, MPCmaxNow is 1
        else:
            MPCmaxNow = 1.0 - WorstIncPrb ** (1.0 / self.CRRA) * RPF
            MPCminNow = 1.0 - RPF

        # Store the results as attributes of self
        self.hNrmNow = hNrmNow
        self.MPCminNow = MPCminNow
        self.MPCmaxNow = MPCmaxNow

    def make_euler_error_func(self, mMax=100, approx_inc_dstn=True):
        """
        Creates a "normalized Euler error" function for this instance, mapping
        from market resources to "consumption error per dollar of consumption."
        Stores result in attribute eulerErrorFunc as an interpolated function.
        Has option to use approximate income distribution stored in self.IncShkDstn
        or to use a (temporary) very dense approximation.

        Only works on (one period) infinite horizon models at this time, will
        be generalized later.

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
        # Get the income distribution (or make a very dense one)
        if approx_inc_dstn:
            IncShkDstn = self.IncShkDstn[0]
        else:
            TranShkDstn = MeanOneLogNormal(sigma=self.TranShkStd[0]).approx(
                N=200, tail_N=50, tail_order=1.3, tail_bound=[0.05, 0.95]
            )
            TranShkDstn = add_discrete_outcome_constant_mean(
                TranShkDstn, self.UnempPrb, self.IncUnemp
            )
            PermShkDstn = MeanOneLogNormal(sigma=self.PermShkStd[0]).approx(
                N=200, tail_N=50, tail_order=1.3, tail_bound=[0.05, 0.95]
            )
            IncShkDstn = combine_indep_dstns(PermShkDstn, TranShkDstn)

        # Make a grid of market resources
        mNowMin = self.solution[0].mNrmMin + 10 ** (
            -15
        )  # add tiny bit to get around 0/0 problem
        mNowMax = mMax
        mNowGrid = np.linspace(mNowMin, mNowMax, 1000)

        # Get the consumption function this period and the marginal value function
        # for next period.  Note that this part assumes a one period cycle.
        cFuncNow = self.solution[0].cFunc
        vPfuncNext = self.solution[0].vPfunc

        # Calculate consumption this period at each gridpoint (and assets)
        cNowGrid = cFuncNow(mNowGrid)
        aNowGrid = mNowGrid - cNowGrid

        # Tile the grids for fast computation
        ShkCount = IncShkDstn[0].size
        aCount = aNowGrid.size
        aNowGrid_tiled = np.tile(aNowGrid, (ShkCount, 1))
        PermShkVals_tiled = (np.tile(IncShkDstn[1], (aCount, 1))).transpose()
        TranShkVals_tiled = (np.tile(IncShkDstn[2], (aCount, 1))).transpose()
        ShkPrbs_tiled = (np.tile(IncShkDstn[0], (aCount, 1))).transpose()

        # Calculate marginal value next period for each gridpoint and each shock
        mNextArray = (
            self.Rfree / (self.PermGroFac[0] * PermShkVals_tiled) * aNowGrid_tiled
            + TranShkVals_tiled
        )
        vPnextArray = vPfuncNext(mNextArray)

        # Calculate expected marginal value and implied optimal consumption
        ExvPnextGrid = (
            self.DiscFac
            * self.Rfree
            * self.LivPrb
            * self.PermGroFac[0] ** (-self.CRRA)
            * np.sum(
                PermShkVals_tiled ** (-self.CRRA) * vPnextArray * ShkPrbs_tiled, axis=0
            )
        )
        cOptGrid = ExvPnextGrid ** (
            -1.0 / self.CRRA
        )  # This is the 'Endogenous Gridpoints' step

        # Calculate Euler error and store an interpolated function
        EulerErrorNrmGrid = (cNowGrid - cOptGrid) / cOptGrid
        eulerErrorFunc = LinearInterp(mNowGrid, EulerErrorNrmGrid)
        self.eulerErrorFunc = eulerErrorFunc

    def pre_solve(self):
        self.update_solution_terminal()
        self.solution_next = self.solution_terminal

    def check_GICNrm(self, verbose=None):
        """
        Check Individual Growth Patience Factor.
        """
        name = "GICNrm"
        fact = "GPFNrm"

        def test(agent): return agent.solution_curr.GPFNrm <= 1

        messages = {
            True: "\nThe Normalized Growth Patience Factor GPFNrm for the supplied parameter values, GPFNrm={0.GPFNrm}, satisfies the Normalized Growth Impatience Condition (GICNrm), which requires GICNrm < 1: "+self.solution_curr.GPFNrm_facts['urlhandle']+"\n",
            False: "\nThe Normalized Growth Patience Factor GPFNrm for the supplied parameter values, GPFNrm={0.GPFNrm}, violates the Normalized Growth Impatience Condition (GICNrm), which requires GICNrm < 1: "+self.solution_curr.GPFNrm_facts['urlhandle']+"\n",
        }
        verbose_messages = {
            True: " Therefore, a target level of the individual market resources ratio m exists ("+self.solution_curr.GICNrm_facts['urlhandle']+").\n",
            False: " Therefore, a target ratio of individual market resources to individual permanent income does not exist.  ("+self.solution_curr.GICNrm_facts['urlhandle']+")\n",
        }
        verbose = self.verbose if verbose is None else verbose
#        self.check_condition(name, test, messages, verbose, verbose_messages)
        self.check_condtnnew(name, test, messages, verbose, verbose_messages, fact)

    def check_WRIC(self, verbose=None):
        """
        Evaluate and report on the Weak Return Impatience Condition
        [url]/#WRPF modified to incorporate LivPrb
        """
        self.solution_curr.WRPF = (
            (self.solution_curr.UnempPrb ** (1 / self.solution_curr.CRRA))
            * (self.solution_curr.Rfree * self.solution_curr.DiscFac * self.solution_curr.LivNum) ** (1 / self.solution_curr.CRRA)
            / self.solution_curr.Rfree
        )

        self.solution_curr.WRIC = self.solution_curr.WRPF < 1
        name = "WRIC"
        fact = "WRPF"

        def test(agent): return agent.solution_curr.WRPF <= 1

        WRIC_facts = {'about': 'Weak Return Impatience Condition'}
        WRIC_facts.update({'latexexpr': r'\WRIC'})
        WRIC_facts.update({'urlhandle': self.solution_curr.urlroot+'WRIC'})
        WRIC_facts.update({'py___code': 'test: agent.WRPF < 1'})
        self.solution_curr.WRIC_facts = WRIC_facts

        WRPF_facts = {'about': 'Growth Patience Factor'}
        WRPF_facts.update({'latexexpr': r'\WRPF'})
        WRPF_facts.update({'_unicode_': r'℘ RPF'})
        WRPF_facts.update({'urlhandle': self.solution_curr.urlroot+'WRPF'})
        WRPF_facts.update({'py___code': r'UnempPrb * RPF'})

        messages = {
            True: "\nThe Weak Return Patience Factor value for the supplied parameter values, WRPF={0.WRPF}, satisfies the Weak Return Impatience Condition, which requires WRIF < 1: "+self.solution_curr.WRIC_facts['urlhandle'],
            False: "\nThe Weak Return Patience Factor value for the supplied parameter values, WRPF={0.WRPF}, violates the Weak Return Impatience Condition, which requires WRIF < 1: "+self.solution_curr.WRIC_facts['urlhandle'],
        }

        verbose_messages = {
            True: "  Therefore, a nondegenerate solution exists if the FVAC is also satisfied. ("+self.solution_curr.WRIC_facts['urlhandle']+")\n",
            False: "  Therefore, a nondegenerate solution is not available ("+self.solution_curr.WRIC_facts['urlhandle']+")\n",
        }
        verbose = self.verbose if verbose is None else verbose
#        self.check_condition(name, test, messages, verbose, verbose_messages)
        self.check_condtnnew(name, test, messages, verbose, verbose_messages, fact)

        self.solution_curr.WRPF_facts = WRPF_facts

    def check_FVAC(self, verbose=None):
        """
        Evaluate and report on the Finite Value of Autarky Condition
        """
        name = "FVAC"
        fact = "FVAF"

        def test(agent): return agent.solution_curr.FVAF < 1

        messages = {
            True: "\nThe Finite Value of Autarky Factor for the supplied parameter values, FVAF={0.FVAF}, satisfies the Finite Value of Autarky Condition, which requires FVAF < 1: "+self.solution_curr.FVAC_facts['urlhandle'],
            False: "\nThe Finite Value of Autarky Factor for the supplied parameter values, FVAF={0.FVAF}, violates the Finite Value of Autarky Condition, which requires FVAF < 1: "+self.solution_curr.FVAC_facts['urlhandle']
        }

        verbose_messages = {
            # ; see {0.url}/#Conditions-Under-Which-the-Problem-Defines-a-Contraction-Mapping\n",
            True: "  Therefore, a nondegenerate solution exists if the WRIC also holds. ("+self.solution_curr.FVAC_facts['urlhandle']+")\n",
            # (see {0.url}/#Conditions-Under-Which-the-Problem-Defines-a-Contraction-Mapping\n",
            False: "  Therefore, a nondegenerate solution is not available ("+self.solution_curr.FVAC_facts['urlhandle']+")\n",
        }
        verbose = self.verbose if verbose is None else verbose
#        self.check_condition(name, test, messages, verbose, verbose_messages)
        self.check_condtnnew(name, test, messages, verbose, verbose_messages, fact)

    def check_condtnsnew(self, verbose=None):
        """
        Checks whether the type satisfies the Absolute Impatience Condition (AIC), Weak Return
        Impatience Condition (WRIC), Finite Human Wealth Condition (FHWC) and Finite Value of
        Autarky Condition (FVAC).  When combinations of these conditions are satisfied, the
        solution to the problem exhibits different characteristics.  (For an exposition of the
        conditions, see https://econ-ark.github.io/BufferStockTheory/)

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
        self.conditions = {}

        self.violated = False

        if self.cycles != 0 or self.T_cycle > 1:
            if self.verbose == 3:
                _log.warning("The conditions apply only to infinite horizon models")
            return

        self.check_AIC(verbose)
        self.check_RIC(verbose)
        self.check_FHWC(verbose)
        self.check_GICNrm(verbose)
        self.check_GICRaw(verbose)
        self.check_GICLiv(verbose)
        self.check_WRIC(verbose)
        self.check_FVAC(verbose)

        self.violated = not self.solution_curr.conditions["WRIC"] or not self.solution_curr.conditions["FVAC"]

        if self.violated:
            _log.warning('Under the provided parameter values, the solution is degenerate')
            _log.warning(
                '\n[!] For more information on the conditions, see Tables 3 and 4 in "Theoretical Foundations of Buffer Stock Saving" at '
                + self.url_ref
                + "/#Factors-Defined-And-Compared"
            )

        _log.warning("APF            = %2.6f " %
                     (self.solution_curr.APF) + '  ' + str(self.solution_curr.APF_facts['urlhandle']))
        _log.warning("RPF            = %2.6f " %
                     (self.solution_curr.RPF) + '  ' + str(self.solution_curr.RPF_facts['urlhandle']))
        _log.warning("FHWF           = %2.6f " %
                     (self.solution_curr.FHWF) + '  ' + str(self.solution_curr.FHWF_facts['urlhandle']))
        _log.warning("GPFNrm         = %2.6f " % (
            self.solution_curr.GPFNrm) + '  ' + str(self.solution_curr.GPFNrm_facts['urlhandle']))
        _log.warning("GPFRaw         = %2.6f " % (
            self.solution_curr.GPFRaw) + '  ' + str(self.solution_curr.GPFRaw_facts['urlhandle']))
        _log.warning("GPFLiv         = %2.6f " % (
            self.solution_curr.GPFLiv) + '  ' + str(self.solution_curr.GPFLiv_facts['urlhandle']))
        _log.warning("FVAF           = %2.6f " %
                     (self.solution_curr.FVAF) + '  ' + str(self.solution_curr.FVAF_facts['urlhandle']))
        _log.warning("WRPF           = %2.6f " %
                     (self.solution_curr.WRPF) + '  ' + str(self.solution_curr.WRPF_facts['urlhandle']))
        _log.warning("DiscGPFNrmCusp = %2.6f " % (self.solution_curr.DiscGPFNrmCusp) + '  code:' +
                     self.solution_curr.DiscGPFNrmCusp_facts['py___code'])
        _log.warning("DiscGPFRawCusp = %2.6f " % (self.solution_curr.DiscGPFRawCusp) + '  code:' +
                     self.solution_curr.DiscGPFRawCusp_facts['py___code'])
        _log.warning("DiscGPFLivCusp = %2.6f " % (self.solution_curr.DiscGPFLivCusp) + '  code:' +
                     self.solution_curr.DiscGPFLivCusp_facts['py___code'])

    def construct_lognormal_income_process_unemployment(self):
        """
        Generates a sequence of discrete approximations to the income process for each
        life period, from end of life to beginning of life.  Permanent shocks are mean
        one lognormally distributed with standard deviation PermShkStd[t] during the
        working life, and degenerate at 1 in the retirement period.  Transitory shocks
        are mean one lognormally distributed with a point mass at IncUnemp with
        probability UnempPrb while working; they are mean one with a point mass at
        IncUnempRet with probability UnempPrbRet.  Retirement occurs
        after t=T_retire periods of working.

        Note 1: All time in this function runs forward, from t=0 to t=T

        Note 2: All parameters are passed as attributes of the input parameters.

        Parameters (passed as attributes of the input parameters)
        ----------
        PermShkStd : [float]
            List of standard deviations in log permanent income uncertainty during
            the agent's life.
        PermShkCount : int
            The number of approximation points to be used in the discrete approxima-
            tion to the permanent income shock distribution.
        TranShkStd : [float]
            List of standard deviations in log transitory income uncertainty during
            the agent's life.
        TranShkCount : int
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
            Transitory income received when unemployed.
        IncUnempRet : float
            Transitory income received while "unemployed" when retired.
        T_cycle :  int
            Total number of non-terminal periods in the consumer's sequence of periods.

        Returns
        -------
        IncShkDstn :  [distribution.Distribution]
            A list with T_cycle elements, each of which is a
            discrete approximation to the income process in a period.
        PermShkDstn : [[distribution.Distribution]]
            A list with T_cycle elements, each of which
            a discrete approximation to the permanent income shocks.
        TranShkDstn : [[distribution.Distribution]]
            A list with T_cycle elements, each of which
            a discrete approximation to the transitory income shocks.
        """
        # Unpack the parameters from the input

        PermShkStd = self.PermShkStd
        PermShkCount = self.PermShkCount
        TranShkStd = self.TranShkStd
        TranShkCount = self.TranShkCount
        UnempPrb = self.UnempPrb
        UnempPrbRet = self.UnempPrbRet
        T_retire = self.T_retire
        IncUnemp = self.IncUnemp
        IncUnempRet = self.IncUnempRet
        T_cycle = self.T_cycle

        # make a dictionary of the parameters
        # This is so that, later, we can determine whether any parameters have changed
        params = {
            'PermShkStd':  self.PermShkStd,
            'PermShkCount':  self.PermShkCount,
            'TranShkStd':  self.TranShkStd,
            'TranShkCount':  self.TranShkCount,
            'UnempPrb':  self.UnempPrb,
            'UnempPrbRet':  self.UnempPrbRet,
            'T_retire':  self.T_retire,
            'IncUnemp':  self.IncUnemp,
            'IncUnempRet':  self.IncUnempRet,
            'T_cycle':  self.T_cycle
        }

        # This is so that, later, we can determine whether another distribution object
        # was constructed using the same method or a different method
        constructed_by = {'method': 'construct_lognormal_income_process_unemployment'}

        IncShkDstn = []  # Discrete approximations to income process in each period
        PermShkDstn = []  # Discrete approximations to permanent income shocks
        TranShkDstn = []  # Discrete approximations to transitory income shocks

        # Fill out a simple discrete RV for retirement, with value 1.0 (mean of shocks)
        # in normal times; value 0.0 in "unemployment" times with small prob.
        if T_retire > 0:
            if UnempPrbRet > 0:
                PermShkValsRet = np.array(
                    [1.0, 1.0]
                )  # Permanent income is deterministic in retirement (2 states for temp income shocks)
                TranShkValsRet = np.array(
                    [
                        IncUnempRet,
                        (1.0 - UnempPrbRet * IncUnempRet) / (1.0 - UnempPrbRet),
                    ]
                )
                ShkPrbsRet = np.array([UnempPrbRet, 1.0 - UnempPrbRet])
            else:
                PermShkValsRet = np.array([1.0])
                TranShkValsRet = np.array([1.0])
                ShkPrbsRet = np.array([1.0])
                IncShkDstnRet = DiscreteDistribution(
                    ShkPrbsRet,
                    [PermShkValsRet, TranShkValsRet],
                    seed=self.RNG.randint(0, 2 ** 31 - 1),
                )

        # Loop to fill in the list of IncShkDstn random variables.
        for t in range(T_cycle):  # Iterate over all periods, counting forward
            if T_retire > 0 and t >= T_retire:
                # Then we are in the "retirement period" and add a retirement income object.
                IncShkDstn.append(deepcopy(IncShkDstnRet))
                PermShkDstn.append([np.array([1.0]), np.array([1.0])])
                TranShkDstn.append([ShkPrbsRet, TranShkValsRet])
            else:
                # We are in the "working life" periods.
                TranShkDstn_t = MeanOneLogNormal(sigma=TranShkStd[t]).approx(
                    TranShkCount, tail_N=0
                )
                if UnempPrb > 0:
                    TranShkDstn_t = add_discrete_outcome_constant_mean(
                        TranShkDstn_t, p=UnempPrb, x=IncUnemp
                    )
                    PermShkDstn_t = MeanOneLogNormal(sigma=PermShkStd[t]).approx(
                        PermShkCount, tail_N=0
                    )
                    IncShkDstn.append(
                        combine_indep_dstns(
                            PermShkDstn_t,
                            TranShkDstn_t,
                            seed=self.RNG.randint(0, 2 ** 31 - 1),
                        )
                    )  # mix the independent distributions
                    PermShkDstn.append(PermShkDstn_t)
                    TranShkDstn.append(TranShkDstn_t)

        IncShkDstn[-1].params = params
        IncShkDstn[-1].constructed_by = constructed_by

        return IncShkDstn, PermShkDstn, TranShkDstn

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
        PermShkNow = np.zeros(self.mcrlo_AgentCount)  # Initialize shock arrays
        TranShkNow = np.zeros(self.mcrlo_AgentCount)
        newborn = self.t_age == 0
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            N = np.sum(these)
            if N > 0:
                IncShkDstnNow = self.IncShkDstn[
                    t - 1
                ]  # set current income distribution
                PermGroNow = self.PermGroFac[t - 1]  # and permanent growth factor
                # Get random draws of income shocks from the discrete distribution
                IncShks = IncShkDstnNow.draw(N)

                PermShkNow[these] = (
                    IncShks[0, :] * PermGroNow
                )  # permanent "shock" includes expected growth
                TranShkNow[these] = IncShks[1, :]

        # That procedure used the *last* period in the sequence for newborns, but that's not right
        # Redraw shocks for newborns, using the *first* period in the sequence.  Approximation.
        N = np.sum(newborn)
        if N > 0:
            these = newborn
            IncShkDstnNow = self.IncShkDstn[0]  # set current income distribution
            PermGroNow = self.PermGroFac[0]  # and permanent growth factor

            # Get random draws of income shocks from the discrete distribution
            EventDraws = IncShkDstnNow.draw_events(N)
            PermShkNow[these] = (
                IncShkDstnNow.X[0][EventDraws] * PermGroNow
            )  # permanent "shock" includes expected growth
            TranShkNow[these] = IncShkDstnNow.X[1][EventDraws]
            #        PermShkNow[newborn] = 1.0
        TranShkNow[newborn] = 1.0

        # Store the shocks in self
        self.EmpNow = np.ones(self.mcrlo_AgentCount, dtype=bool)
        self.EmpNow[TranShkNow == self.IncUnemp] = False
        self.shocks['PermShk'] = PermShkNow
        self.shocks['TranShk'] = TranShkNow


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

        # Generate a url that will locate the documentation
        self.url_doc = "https://hark.readthedocs.io/en/latest/search.html?q=" + \
            self.__class__.__name__+"&check_keywords=yes&area=default#"

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
        PermShkValsNext = self.IncShkDstn[0][1]
        TranShkValsNext = self.IncShkDstn[0][2]
        ShkPrbsNext = self.IncShkDstn[0][0]
        Ex_IncNextNrm = calc_expectation(
            self.IncShkDstn,
            lambda trans, perm: trans * perm
        )
        PermShkMinNext = np.min(PermShkValsNext)
        TranShkMinNext = np.min(TranShkValsNext)
        WorstIncNext = PermShkMinNext * TranShkMinNext
        WorstIncPrb = np.sum(
            ShkPrbsNext[(PermShkValsNext * TranShkValsNext) == WorstIncNext]
        )

        # Calculate human wealth and the infinite horizon natural borrowing constraint
        hNrmNow = (Ex_IncNextNrm * self.PermGroFac[0] / self.Rsave) / (
            1.0 - self.PermGroFac[0] / self.Rsave
        )
        temp = self.PermGroFac[0] * PermShkMinNext / self.Rboro
        BoroCnstNat = -TranShkMinNext * temp / (1.0 - temp)

        RPFTop = (self.DiscFac * self.LivPrb[0] * self.Rsave) ** (
            1.0 / self.CRRA
        ) / self.Rsave
        RPFBot = (self.DiscFac * self.LivPrb[0] * self.Rboro) ** (
            1.0 / self.CRRA
        ) / self.Rboro
        if BoroCnstNat < self.BoroCnstArt:
            MPCmaxNow = 1.0  # if natural borrowing constraint is overridden by artificial one, MPCmaxNow is 1
        else:
            MPCmaxNow = 1.0 - WorstIncPrb ** (1.0 / self.CRRA) * RPFBot
            MPCminNow = 1.0 - RPFTop

        # Store the results as attributes of self
        self.hNrmNow = hNrmNow
        self.MPCminNow = MPCminNow
        self.MPCmaxNow = MPCmaxNowMNow
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
        on whether self.aNrmNow >< 0.

        Parameters
        ----------
        None

        Returns
        -------
        RfreeNow : np.array
             Array of size self.mcrlo_AgentCount with risk free interest rate for each agent.
        """
        RfreeNow = self.Rboro * np.ones(self.mcrlo_AgentCount)
        RfreeNow[self.state_prev['aNrm'] > 0] = self.Rsave
        return RfreeNow


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
init_cyclical['PermShkStd'] = [0.1, 0.1, 0.1, 0.1]
init_cyclical['TranShkStd'] = [0.1, 0.1, 0.1, 0.1]
init_cyclical['LivPrb'] = 4*[0.98]
init_cyclical['T_cycle'] = 4
