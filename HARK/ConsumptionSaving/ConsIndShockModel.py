from builtins import (range, object, str)
from copy import copy, deepcopy
import numpy as np
from scipy.optimize import newton
from HARK import AgentType, NullFunc, MetricObject, make_one_period_oo_solver
from HARK.interpolation import (CubicInterp, LowerEnvelope, LinearInterp, ValueFuncCRRA, MargValueFuncCRRA,
                                MargMargValueFuncCRRA)
from HARK.distribution import (DiscreteDistribution, add_discrete_outcome_constant_mean, calc_expectation,
                               combine_indep_dstns, Lognormal, MeanOneLogNormal, Uniform)
from HARK.utilities import (make_grid_exp_mult, CRRAutility, CRRAutilityP, CRRAutilityPP, CRRAutilityP_inv,
                            CRRAutility_invP, CRRAutility_inv, CRRAutilityP_invP, warnings)
# import types  # Needed to allow solver to attach methods to solution
from HARK.core import (_log, set_verbosity_level, core_check_condition, bind_method)
from HARK.Calibration.Income.IncomeTools import parse_income_spec, parse_time_params, Cagetti_income
from HARK.datasets.SCF.WealthIncomeDist.SCFDistTools import income_wealth_dists_from_scf
from HARK.datasets.life_tables.us_ssa.SSATools import parse_ssa_life_table


class TrnsPars():
    def __init__(self):
        self.about = {'TrnsPars': 'Parameters for transition from current to next stage'}


class DolObj(MetricObject):
    def __init__(
            self,
            dsymbls=None,
            dsttes=None,
            dcntrols=None,
            dexpects=None,
            dvlues=None,
            dparms=None,
            drwards=None,
            ddefns=None,
            deqns={'darbitrge': dict(), 'dtrnstn': dict(), 'dvlues': dict(), 'dfelicty': dict(), 'ddirct_rspnse': dict()
                   },
            dcalibrtn=dict({'dparms': dict(), 'dendog': dict()}),
            dexog=dict(),
            ddmain=dict(),
            doptns=dict()
    ):
        self.about = {'DolObj': None}


"""
Classes to solve canonical consumption-saving models with idiosyncratic shocks
to income.  All models here assume CRRA utility with geometric discounting, no
bequest motive, and income shocks that are fully transitory or fully permanent.

It currently solves three types of models:
   1) A very basic "perfect foresight" consumption-savings model with no uncertainty.
      * Features of the model prepare it for convenient inheritance
   2) A consumption-saving model with transitory and permanent income shocks
      * Inherits from PF model
   3) The model described in (2), but with an interest rate for debt that differs
      from the interest rate for savings.

See NARK https://HARK.githhub.io/Documentation/NARK for information on variable naming conventions.
See HARK documentation for mathematical descriptions of the models being solved.
"""

__all__ = [
    "DolObj",
    "TrnsPars",
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
    Represents the solution of a single period/decision-stage of a consumption
    problem.  The solution must include a consumption function and marginal
    value function.  (period/stage will refer to the solution of the
    consumer's Bellman problem; terminology will eventually become stage).

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
    hNrmNow : float
        Human wealth after receiving income this period: PDV of all future
        income, ignoring mortality.
    MPCminNow : float
        Infimum of the marginal propensity to consume this period.
        MPC --> MPCminNow as m --> infinity.
    MPCmaxNow : float
        Supremum of the marginal propensity to consume this period.
        MPC --> MPCmaxNow as m --> mNrmMin.
    stge_kind : dict
        Dictionary with info about the type of this stage
        One built-in entry keeps track of the nature of the stage:
        {'iter_status':'terminal'}: Terminal (last period of existence)
        {'iter_status':'iterator'}: Solution during iteration
        {'iter_status':'finished'}: Solution satisfied stopping requirements
    """
    distance_criteria = ["vPfunc"]
#    distance_criteria = ["mNrmStE"]
#    distance_criteria = ["cFunc"]

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
            stge_kind=None,
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
        self.stge_kind = stge_kind

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

        m_init_guess = self.mNrmMin + self.Ex_IncNextNrm
        try:  # Find value where argument is zero
            self.mNrmTrg = newton(
                self.Ex_m_tp1_minus_m_t,
                m_init_guess)
        except:
            self.mNrmTrg = None

        return self.mNrmTrg

# The PerfForesight class also incorporates calcs and info that are useful for
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
        self.stg_Nxt = solution_next

        self.stg_Now = ConsumerSolution()  # create a blank template to fill in
        self.stg_Now.Nxt = TrnsPars()  # Pars for trans to Nxt stage
        # CDC: These parameters properly belong to the next stage, and
        # so should be retrieved by it rather than provided as inputs

        self.stg_Now.MaxKinks = MaxKinks
        self.stg_Now.Nxt.LivPrb = LivPrb
        self.stg_Now.Nxt.DiscFac = DiscFac
        # As code is currently structured, CRRA not allowed in time_vary
        # because same u is used for EGM, current vFunc, and E[v_{t+1}]
        self.stg_Now.Nxt.CRRA = self.stg_Now.CRRA = CRRA
        self.stg_Now.Nxt.Rfree = Rfree
        self.stg_Now.Nxt.PermGro = PermGroFac
        self.stg_Now.Nxt.BoroCnstArt = BoroCnstArt

        # Old code may expect these things to live at root of self
        # For now, put them there too, but over time weed out
        self.stg_Now.LivPrb = self.stg_Now.Nxt.LivPrb
        self.stg_Now.DiscFac = self.stg_Now.Nxt.DiscFac
        self.stg_Now.CRRA = self.stg_Now.Nxt.CRRA
        self.stg_Now.Rfree = self.stg_Now.Nxt.Rfree
        self.stg_Now.PermGro = self.stg_Now.Nxt.PermGro
        self.stg_Now.BoroCnstArt = self.stg_Now.Nxt.BoroCnstArt

        self.stg_Now.fcts = {}
        self.stg_Now = self.def_utility_funcs(self.stg_Now)

        # Generate a url that will locate the documentation
        # Perfect Foreight version
        self.stg_Now.url_doc = "https://hark.readthedocs.io/en/latest/search.html?q=" + \
            self.stg_Now.__class__.__name__+"&check_keywords=yes&area=default#"

        # url for paper that contains various theoretical results
        self.stg_Now.url_ref = "https://econ-ark.github.io/BufferStockTheory"
        self.stg_Now.urlroot = self.stg_Now.url_ref+'/#'  # used for references to derivations

        # Constructing these allows the use of identical formulae for the perfect
        # foresight model and models with transitory and permanent shocks
        self.stg_Now.Ex_Inv_PermShk = 1.0
        self.stg_Now.Ex_uInv_PermShk = 1.0
        self.stg_Now.uInv_Ex_uInv_PermShk = 1.0

    def add_fcts_to_soln_ConsPerfForesightSolver_20210410(self, stg_Now):
        """
            Adds to the solution extensive information and references about
            its elements.

            Parameters
            ----------
            solution: ConsumerSolution
                A solution that has already been augmented with required_calcs

            Returns
            -------
            solution : ConsumerSolution
                Same solution that was provided, augmented with fcts and
                references
            """
        # Sometimes BoroCnstArt has not been set
        # Define local variables w/o nxt so formulae are less cumbersome
        # if hasattr(self.stg_Now.Nxt, 'BoroCnstArt'):
        #     BoroCnstArt = self.stg_Now.Nxt.BoroCnstArt
        # else:
        #     BoroCnstArt = self.stg_Now.Nxt.BoroCnstArt = None
        CRRA = self.stg_Now.CRRA
        DiscFac = self.stg_Now.Nxt.DiscFac
        LivPrb = self.stg_Now.Nxt.LivPrb
        PermGro = self.stg_Now.Nxt.PermGro = self.stg_Now.Nxt.PermGroFac
        Rfree = self.stg_Now.Nxt.Rfree
        DiscFacEff = stg_Now.Nxt.DiscFacEff \
            = stg_Now.Nxt.DiscFac * stg_Now.Nxt.LivPrb
        MPCminNow = self.stg_Now.MPCminNow
        MPCmaxNow = self.stg_Now.MPCmaxNow
        hNrmNow = self.stg_Now.hNrmNow
        urlroot = self.stg_Now.urlroot
        self.stg_Now.fcts = {}

        APFfcts = {'about': 'Absolute Patience Factor'}
        self.stg_Now.APF = APF = ((Rfree * DiscFacEff) ** (1.0 / CRRA))
        APFfcts.update({'latexexpr': r'\APF'})
        APFfcts.update({'_unicode_': r'Þ'})
        APFfcts.update({'urlhandle': urlroot+'APF'})
        APFfcts.update({'py___code': '(Rfree*DiscFacEff)**(1/CRRA)'})
        APFfcts.update({'value_now': APF})
        stg_Now.fcts.update({'APF': APFfcts})
        stg_Now.APFfcts = APFfcts

        AICfcts = {'about': 'Absolute Impatience Condition'}
        AICfcts.update({'latexexpr': r'\AIC'})
        AICfcts.update({'urlhandle': urlroot+'AIC'})
        AICfcts.update({'py___code': 'test: APF < 1'})
        stg_Now.fcts.update({'AIC': AICfcts})
        stg_Now.AICfcts = AICfcts

        RPFfcts = {'about': 'Return Patience Factor'}
        stg_Now.RPF = RPF = APF / Rfree
        RPFfcts.update({'latexexpr': r'\RPF'})
        RPFfcts.update({'_unicode_': r'Þ_R'})
        RPFfcts.update({'urlhandle': urlroot+'RPF'})
        RPFfcts.update({'py___code': r'APF/Rfree'})
        RPFfcts.update({'value_now': RPF})
        stg_Now.fcts.update({'RPF': RPFfcts})
        stg_Now.RPFfcts = RPFfcts
        stg_Now.RPF = RPF

        RICfcts = {'about': 'Growth Impatience Condition'}
        RICfcts.update({'latexexpr': r'\RIC'})
        RICfcts.update({'urlhandle': urlroot+'RIC'})
        RICfcts.update({'py___code': 'test: RPF < 1'})
        stg_Now.fcts.update({'RIC': RICfcts})
        stg_Now.RICfcts = RICfcts

        GPFRawfcts = {
            'about': 'Growth Patience Factor'}
        GPFRaw = APF / PermGro
        GPFRawfcts.update({'latexexpr': '\GPFRaw'})
        GPFRawfcts.update({'urlhandle': urlroot+'GPFRaw'})
        GPFRawfcts.update({'_unicode_': r'Þ_Γ'})
        GPFRawfcts.update({'value_now': GPFRaw})
        stg_Now.fcts.update({'GPFRaw': GPFRawfcts})
        stg_Now.GPFRawfcts = GPFRawfcts
        stg_Now.GPFRaw = GPFRaw

        GICRawfcts = {'about': 'Growth Impatience Condition'}
        GICRawfcts.update({'latexexpr': r'\GICRaw'})
        GICRawfcts.update({'urlhandle': urlroot+'GICRaw'})
        GICRawfcts.update({'py___code': 'test: GPFRaw < 1'})
        stg_Now.fcts.update({'GICRaw': GICRawfcts})
        stg_Now.GICRawfcts = GICRawfcts

        GPFLivfcts = {
            'about': 'Mortality-Risk-Adjusted Growth Patience Factor'}
        GPFLiv = APF * LivPrb / PermGro
        GPFLivfcts.update({'latexexpr': '\GPFLiv'})
        GPFLivfcts.update({'urlhandle': urlroot+'GPFLiv'})
        GPFLivfcts.update({'py___code': 'APF*Liv/PermGro'})
        GPFLivfcts.update({'value_now': GPFLiv})
        stg_Now.fcts.update({'GPFLiv': GPFLivfcts})
        stg_Now.GPFLivfcts = GPFLivfcts
        stg_Now.GPFLiv = GPFLiv

        GICLivfcts = {'about': 'Growth Impatience Condition'}
        GICLivfcts.update({'latexexpr': r'\GICLiv'})
        GICLivfcts.update({'urlhandle': urlroot+'GICLiv'})
        GICLivfcts.update({'py___code': 'test: GPFLiv < 1'})
        stg_Now.fcts.update({'GICLiv': GICLivfcts})
        stg_Now.GICLivfcts = GICLivfcts

        PF_RNrmfcts = {
            'about': 'Growth-Normalized Perfect Foresight Return Factor'}
        PF_RNrm = Rfree/PermGro
        PF_RNrmfcts.update({'latexexpr': r'\PF_RNrm'})
        PF_RNrmfcts.update({'_unicode_': r'R/Γ'})
        PF_RNrmfcts.update({'py___code': r'Rfree/PermGro'})
        PF_RNrmfcts.update({'value_now': PF_RNrm})
        stg_Now.fcts.update({'PF_RNrm': PF_RNrmfcts})
        stg_Now.PF_RNrmfcts = PF_RNrmfcts
        stg_Now.PF_RNrm = PF_RNrm

        Inv_PF_RNrmfcts = {
            'about': 'Inverse of Growth-Normalized Perfect Foresight Return Factor'}
        Inv_PF_RNrm = 1/PF_RNrm
        Inv_PF_RNrmfcts.update({'latexexpr': r'\Inv_PF_RNrm'})
        Inv_PF_RNrmfcts.update({'_unicode_': r'Γ/R'})
        Inv_PF_RNrmfcts.update({'py___code': r'PermGroInd/Rfree'})
        Inv_PF_RNrmfcts.update({'value_now': Inv_PF_RNrm})
        stg_Now.fcts.update({'Inv_PF_RNrm': Inv_PF_RNrmfcts})
        stg_Now.Inv_PF_RNrmfcts = Inv_PF_RNrmfcts
        stg_Now.Inv_PF_RNrm = Inv_PF_RNrm

        FHWFfcts = {
            'about': 'Finite Human Wealth Factor'}
        FHWF = PermGro/Rfree
        FHWFfcts.update({'latexexpr': r'\FHWF'})
        FHWFfcts.update({'_unicode_': r'R/Γ'})
        FHWFfcts.update({'urlhandle': urlroot+'FHWF'})
        FHWFfcts.update({'py___code': r'PermGroInf/Rfree'})
        FHWFfcts.update({'value_now': FHWF})
        stg_Now.fcts.update({'FHWF': FHWFfcts})
        stg_Now.FHWFfcts = FHWFfcts
        stg_Now.FHWF = FHWF

        FHWCfcts = {'about': 'Finite Human Wealth Condition'}
        FHWCfcts.update({'latexexpr': r'\FHWC'})
        FHWCfcts.update({'urlhandle': urlroot+'FHWC'})
        FHWCfcts.update({'py___code': 'test: FHWF < 1'})
        stg_Now.fcts.update({'FHWC': FHWCfcts})
        stg_Now.FHWCfcts = FHWCfcts

        hNrmNowInffcts = {'about': 'Human wealth for inf hor'}
        hNrmNowInf = float('inf')
        if FHWF < 1:  # If it is finite, set it to its value
            hNrmNowInf = 1/(1-FHWF)
        stg_Now.hNrmNowInf = hNrmNowInf
        hNrmNowInffcts = dict({'latexexpr': '1/(1-\FHWF)'})
        hNrmNowInffcts.update({'value_now': hNrmNowInf})
        hNrmNowInffcts.update({
            'py___code': '1/(1-FHWF)'})
        stg_Now.fcts.update({'hNrmNowInf': hNrmNowInffcts})
        stg_Now.hNrmNowInffcts = hNrmNowInffcts
        # stg_Now.hNrmNowInf = hNrmNowInf

        DiscGPFRawCuspfcts = {
            'about': 'DiscFac s.t. GPFRaw = 1'}
        stg_Now.DiscGPFRawCusp = DiscGPFRawCusp = ((PermGro) ** (CRRA)) / (Rfree)
        DiscGPFRawCuspfcts.update({'latexexpr': ''})
        DiscGPFRawCuspfcts.update({'value_now': DiscGPFRawCusp})
        DiscGPFRawCuspfcts.update({
            'py___code': '( PermGro                       ** CRRA)/(Rfree)'})
        stg_Now.fcts.update({'DiscGPFRawCusp': DiscGPFRawCuspfcts})
        stg_Now.DiscGPFRawCuspfcts = DiscGPFRawCuspfcts

        DiscGPFLivCuspfcts = {
            'about': 'DiscFac s.t. GPFLiv = 1'}
        stg_Now.DiscGPFLivCusp = DiscGPFLivCusp = ((PermGro) ** (CRRA)) \
            / (Rfree * LivPrb)
        DiscGPFLivCuspfcts.update({'latexexpr': ''})
        DiscGPFLivCuspfcts.update({'value_now': DiscGPFLivCusp})
        DiscGPFLivCuspfcts.update({
            'py___code': '( PermGro                       ** CRRA)/(Rfree*LivPrb)'})
        stg_Now.fcts.update({'DiscGPFLivCusp': DiscGPFLivCuspfcts})
        stg_Now.DiscGPFLivCuspfcts = DiscGPFLivCuspfcts

        FVAFfcts = {'about': 'Finite Value of Autarky Factor'}
        stg_Now.FVAF = FVAF = LivPrb * DiscFacEff * stg_Now.uInv_Ex_uInv_PermShk
        FVAFfcts.update({'latexexpr': r'\FVAFPF'})
        FVAFfcts.update({'urlhandle': urlroot+'FVAFPF'})
        stg_Now.fcts.update({'FVAF': FVAFfcts})
        stg_Now.FVAFfcts = FVAFfcts

        FVACfcts = {'about': 'Finite Value of Autarky Condition - Perfect Foresight'}
        FVACfcts.update({'latexexpr': r'\FVACPF'})
        FVACfcts.update({'urlhandle': urlroot+'FVACPF'})
        FVACfcts.update({'py___code': 'test: FVAFPF < 1'})
        stg_Now.fcts.update({'FVAC': FVACfcts})
        stg_Now.FVACfcts = FVACfcts

        # Now add required facts defining bounds
        hNrmNow = (
            (PermGro / Rfree) * (1.0 + self.solution_next.hNrmNow)
        )
        hNrmNowfcts = {'about': 'Human Wealth Now'}
        hNrmNowfcts.update({'latexexpr': r'\hNrmNow'})
        hNrmNowfcts.update({'_unicode_': r'R/Γ'})
        hNrmNowfcts.update({'urlhandle': urlroot+'hNrmNow'})
        hNrmNowfcts.update({'py___code': r'PermGroInf/Rfree'})
        hNrmNowfcts.update({'value_now': hNrmNow})
        stg_Now.hNrmNowfcts = hNrmNowfcts
        stg_Now.fcts.update({'hNrmNow': hNrmNowfcts})
        self.hNrmNow = stg_Now.hNrmNow = hNrmNow

        # Calculate the minimum allowable value of money resources in this period

        stg_Now.BoroCnstNat = (
            (self.solution_next.mNrmMin - min(self.solution_next.TranShkValsNxt))
            * (PermGro * min(self.solution_next.PermShkValsNxt))
            / self.solution_next.Rfree
        )

        if stg_Now.BoroCnstArt is None:
            stg_Now.mNrmMin = stg_Now.BoroCnstNat
        else:  # Artificial is only relevant if tighter than natural
            stg_Now.mNrmMin = np.max([stg_Now.BoroCnstNat, stg_Now.BoroCnstArt])
            # Liquidity constrained consumption function: c(mMin+x) = x
            stg_Now.cFuncNowCnst = LinearInterp(
                np.array([stg_Now.mNrmMin, stg_Now.mNrmMin + 1]
                         ), np.array([0.0, 1.0])
            )

        mNrmMin = stg_Now.mNrmMin
        mNrmMinfcts = {'about': 'Minimum mNrm'}
        mNrmMinfcts.update({'latexexpr': r'\mNrmMin'})
        stg_Now.fcts.update({'mNrmMin': mNrmMinfcts})
        stg_Now.mNrmMinfcts = mNrmMinfcts
        stg_Now.mNrmMin = mNrmMin

        MPCminNow = 1.0 / (1.0 + stg_Now.RPF / self.solution_next.MPCminNow)
        MPCminNowfcts = {
            'about': 'Minimal MPC as m -> infty'}
        MPCminNowfcts.update({'latexexpr': r''})
        MPCminNowfcts.update({'urlhandle': urlroot+'MPCminNow'})
        MPCminNowfcts.update({'value_now': MPCminNow})
        stg_Now.fcts.update({'MPCminNow': MPCminNowfcts})
        stg_Now.MPCminNowfcts = MPCminNowfcts
        stg_Now.MPCminNow = stg_Now.MPCminNow = MPCminNow

        MPCmaxNow = 1.0 / \
            (1.0 + (self.solution_next.WorstIncPrbNxt ** (1.0 / stg_Now.CRRA))
             * self.solution_next.RPF
             / self.solution_next.MPCmaxNow)
        MPCmaxNowfcts = {
            'about': 'Maximal MPC in current period as m -> minimum'}
        MPCmaxNowfcts.update({'latexexpr': r''})
        MPCmaxNowfcts.update({'urlhandle': urlroot+'MPCmaxNow'})
        MPCmaxNowfcts.update({'value_now': MPCmaxNow})
        stg_Now.fcts.update({'MPCmaxNow': MPCmaxNowfcts})
        stg_Now.MPCmaxNowfcts = MPCmaxNowfcts
        stg_Now.MPCmaxNow = MPCmaxNow

        # Lower bound of aggregate wealth growth if all inheritances squandered
        cFuncLimitIntercept = MPCminNow * stg_Now.hNrmNow
        cFuncLimitInterceptfcts = {
            'about': 'Vertical intercept of perfect foresight consumption function'}
        cFuncLimitInterceptfcts.update({'latexexpr': '\MPC '})
        cFuncLimitInterceptfcts.update({'urlhandle': ''})
        cFuncLimitInterceptfcts.update({'value_now': cFuncLimitIntercept})
        cFuncLimitInterceptfcts.update({
            'py___code': 'MPCminNow * hNrmNow'})
        stg_Now.fcts.update({'cFuncLimitIntercept': cFuncLimitInterceptfcts})
        stg_Now.cFuncLimitInterceptfcts = cFuncLimitInterceptfcts
        stg_Now.cFuncLimitIntercept = cFuncLimitIntercept

        cFuncLimitSlope = MPCminNow
        cFuncLimitSlopefcts = {
            'about': 'Slope of limiting consumption function'}
        cFuncLimitSlopefcts = dict({'latexexpr': '\MPC \hNrmNow'})
        cFuncLimitSlopefcts.update({'urlhandle': ''})
        cFuncLimitSlopefcts.update({'value_now': cFuncLimitSlope})
        cFuncLimitSlopefcts.update({
            'py___code': 'MPCminNow * hNrmNow'})
        stg_Now.fcts.update({'cFuncLimitSlope': cFuncLimitSlopefcts})
        stg_Now.cFuncLimitSlopefcts = cFuncLimitSlopefcts
        stg_Now.cFuncLimitSlope = cFuncLimitSlope

        # Fcts that apply in the perfect foresight case should already have been added

        if stg_Now.Inv_PF_RNrm < 1:        # Finite if Rfree > PermGro
            stg_Now.hNrmNowInf = 1/(1-stg_Now.Inv_PF_RNrm)

        # Given m, value of c where E[mLev_{t+1}/mLev_{t}]=PermGro
        # Solves for c in equation at url/#balgrostable

        stg_Now.c_where_Ex_PermShk_times_mtp1_minus_mt_eq_0 = (
            lambda m_t:
            m_t * (1 - stg_Now.Inv_PF_RNrm) + stg_Now.Inv_PF_RNrm
        )

        stg_Now.Ex_cLev_tp1_Over_cLev_t_from_mt = (
            lambda m_t:
            stg_Now.Ex_cLev_tp1_Over_pLev_t_from_mt(stg_Now,
                                                    m_t - stg_Now.cFunc(m_t))
            / stg_Now.cFunc(m_t)
        )

#        # E[m_{t+1} pLev_{t+1}/pLev_{t}] as a fn of a_{t}
        stg_Now.Ex_mLev_tp1_Over_pLev_t_from_at = (
            lambda a_t:
                PermGro *
            (stg_Now.PF_RNrm * a_t + stg_Now.Ex_IncNextNrm)
        )

        # E[m_{t+1} pLev_{t+1}/(m_{t}pLev_{t})] as a fn of m_{t}
        stg_Now.Ex_mLev_tp1_Over_mLev_t_from_at = (
            lambda m_t:
                stg_Now.Ex_mLev_tp1_Over_pLev_t_from_at(stg_Now,
                                                        m_t-stg_Now.cFunc(m_t)
                                                        )/m_t
        )
        hNrmNow = (
            (PermGro / Rfree) * (1.0 + self.solution_next.hNrmNow)
        )
        hNrmNowfcts = {'about': 'Human Wealth Now'}
        hNrmNowfcts.update({'latexexpr': r'\hNrmNow'})
        hNrmNowfcts.update({'_unicode_': r'R/Γ'})
        hNrmNowfcts.update({'urlhandle': urlroot+'hNrmNow'})
        hNrmNowfcts.update({'py___code': r'PermGroInf/Rfree'})
        hNrmNowfcts.update({'value_now': hNrmNow})
        stg_Now.hNrmNowfcts = hNrmNowfcts
        stg_Now.fcts.update({'hNrmNow': hNrmNowfcts})
        self.hNrmNow = stg_Now.hNrmNow = hNrmNow

        # Calculate the minimum allowable value of money resources in this period

        stg_Now.BoroCnstNat = (
            (self.solution_next.mNrmMin - min(self.solution_next.TranShkValsNxt))
            * (PermGro * min(self.solution_next.PermShkValsNxt))
            / self.solution_next.Rfree
        )

        if stg_Now.BoroCnstArt is None:
            stg_Now.mNrmMin = stg_Now.BoroCnstNat
        else:  # Artificial is only relevant if tighter than natural
            stg_Now.mNrmMin = np.max([stg_Now.BoroCnstNat, stg_Now.BoroCnstArt])
            # Liquidity constrained consumption function: c(mMin+x) = x
            stg_Now.cFuncNowCnst = LinearInterp(
                np.array([stg_Now.mNrmMin, stg_Now.mNrmMin + 1]
                         ), np.array([0.0, 1.0])
            )

        mNrmMin = stg_Now.mNrmMin
        mNrmMinfcts = {'about': 'Minimum mNrm'}
        mNrmMinfcts.update({'latexexpr': r'\mNrmMin'})
        stg_Now.fcts.update({'mNrmMin': mNrmMinfcts})
        stg_Now.mNrmMinfcts = mNrmMinfcts
        stg_Now.mNrmMin = mNrmMin

        MPCminNow = 1.0 / (1.0 + stg_Now.RPF / self.solution_next.MPCminNow)
        MPCminNowfcts = {
            'about': 'Minimal MPC as m -> infty'}
        MPCminNowfcts.update({'latexexpr': r''})
        MPCminNowfcts.update({'urlhandle': urlroot+'MPCminNow'})
        MPCminNowfcts.update({'value_now': MPCminNow})
        stg_Now.fcts.update({'MPCminNow': MPCminNowfcts})
        stg_Now.MPCminNowfcts = MPCminNowfcts
        stg_Now.MPCminNow = stg_Now.MPCminNow = MPCminNow

        MPCmaxNow = 1.0 / \
            (1.0 + (self.solution_next.WorstIncPrbNxt ** (1.0 / stg_Now.CRRA))
             * self.solution_next.RPF
             / self.solution_next.MPCmaxNow)
        MPCmaxNowfcts = {
            'about': 'Maximal MPC in current period as m -> minimum'}
        MPCmaxNowfcts.update({'latexexpr': r''})
        MPCmaxNowfcts.update({'urlhandle': urlroot+'MPCmaxNow'})
        MPCmaxNowfcts.update({'value_now': MPCmaxNow})
        stg_Now.fcts.update({'MPCmaxNow': MPCmaxNowfcts})
        stg_Now.MPCmaxNowfcts = MPCmaxNowfcts
        stg_Now.MPCmaxNow = MPCmaxNow

        # Lower bound of aggregate wealth growth if all inheritances squandered
        cFuncLimitIntercept = MPCminNow * stg_Now.hNrmNow
        cFuncLimitInterceptfcts = {
            'about': 'Vertical intercept of perfect foresight consumption function'}
        cFuncLimitInterceptfcts.update({'latexexpr': '\MPC '})
        cFuncLimitInterceptfcts.update({'urlhandle': ''})
        cFuncLimitInterceptfcts.update({'value_now': cFuncLimitIntercept})
        cFuncLimitInterceptfcts.update({
            'py___code': 'MPCminNow * hNrmNow'})
        stg_Now.fcts.update({'cFuncLimitIntercept': cFuncLimitInterceptfcts})
        stg_Now.cFuncLimitInterceptfcts = cFuncLimitInterceptfcts
        stg_Now.cFuncLimitIntercept = cFuncLimitIntercept

        cFuncLimitSlope = MPCminNow
        cFuncLimitSlopefcts = {
            'about': 'Slope of limiting consumption function'}
        cFuncLimitSlopefcts = dict({'latexexpr': '\MPC \hNrmNow'})
        cFuncLimitSlopefcts.update({'urlhandle': ''})
        cFuncLimitSlopefcts.update({'value_now': cFuncLimitSlope})
        cFuncLimitSlopefcts.update({
            'py___code': 'MPCminNow * hNrmNow'})
        stg_Now.fcts.update({'cFuncLimitSlope': cFuncLimitSlopefcts})
        stg_Now.cFuncLimitSlopefcts = cFuncLimitSlopefcts
        stg_Now.cFuncLimitSlope = cFuncLimitSlope

        # Fcts that apply in the perfect foresight case should already have been added

        if stg_Now.Inv_PF_RNrm < 1:        # Finite if Rfree > PermGro
            stg_Now.hNrmNowInf = 1/(1-stg_Now.Inv_PF_RNrm)

        # Given m, value of c where E[mLev_{t+1}/mLev_{t}]=PermGro
        # Solves for c in equation at url/#balgrostable

        stg_Now.c_where_Ex_PermShk_times_mtp1_minus_mt_eq_0 = (
            lambda m_t:
            m_t * (1 - stg_Now.Inv_PF_RNrm) + stg_Now.Inv_PF_RNrm
        )

        stg_Now.Ex_cLev_tp1_Over_cLev_t_from_mt = (
            lambda m_t:
            stg_Now.Ex_cLev_tp1_Over_pLev_t_from_mt(stg_Now,
                                                    m_t - stg_Now.cFunc(m_t))
            / stg_Now.cFunc(m_t)
        )

#        # E[m_{t+1} pLev_{t+1}/pLev_{t}] as a fn of a_{t}
        stg_Now.Ex_mLev_tp1_Over_pLev_t_from_at = (
            lambda a_t:
                PermGro *
            (stg_Now.PF_RNrm * a_t + stg_Now.Ex_IncNextNrm)
        )

        # E[m_{t+1} pLev_{t+1}/(m_{t}pLev_{t})] as a fn of m_{t}
        stg_Now.Ex_mLev_tp1_Over_mLev_t_from_at = (
            lambda m_t:
                stg_Now.Ex_mLev_tp1_Over_pLev_t_from_at(stg_Now,
                                                         m_t-stg_Now.cFunc(m_t)
                                                         )/m_t
        )

        # Below formulae do not require "live" computation of expectations
        # from a distribution that is on hand.  So, having constructed
        # expected values above, we can use them.

        # This allows sharing these formulae between the perfect foresight
        # and the non-perfect-foresight models.  They are constructed here
        # and inherited by the descendant model(s), which augment(s) them with
        # the objects (if any) that require live calculation.

        if stg_Now.Inv_PF_RNrm < 1:        # Finite if Rfree > PermGro
            stg_Now.hNrmNowInf = 1/(1-stg_Now.Inv_PF_RNrm)

        # Given m, value of c where E[mLev_{t+1}/mLev_{t}]=PermGroFac
        # Solves for c in equation at url/#balgrostable

        self.stg_Now.c_where_Ex_PermShk_times_mtp1_minus_mt_eq_0 = (
            lambda m_t:
            m_t * (1 - self.stg_Now.Inv_PF_RNrm) + self.stg_Now.Inv_PF_RNrm
        )

        self.stg_Now.Ex_cLev_tp1_Over_cLev_t_from_mt = (
            lambda m_t:
            self.stg_Now.Ex_cLev_tp1_Over_pLev_t_from_mt(stg_Now,
                                                          m_t - self.stg_Now.cFunc(m_t))
            / self.stg_Now.cFunc(m_t)
        )

    #        # E[m_{t+1} pLev_{t+1}/pLev_{t}] as a fn of a_{t}
        self.stg_Now.Ex_mLev_tp1_Over_pLev_t_from_at = (
            lambda a_t:
                self.stg_Now.PermGroNum *
            (stg_Now.PF_RNrm * a_t + self.stg_Now.Ex_IncNextNrm)
        )

        # E[m_{t+1} pLev_{t+1}/(m_{t}pLev_{t})] as a fn of m_{t}
        self.stg_Now.Ex_mLev_tp1_Over_mLev_t_from_at = (
            lambda m_t:
                self.stg_Now.Ex_mLev_tp1_Over_pLev_t_from_at(stg_Now,
                                                              m_t-stg_Now.cFunc(m_t)
                                                              )/m_t
        )

        return stg_Now

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
        stge.u = lambda c: utility(
            c, gam=stge.CRRA)  # utility function
        stge.uP = lambda c: utilityP(
            c, gam=stge.CRRA)  # marginal utility function
        stge.uPP = lambda c: utilityPP(
            c, gam=stge.CRRA
        )  # marginal marginal utility function
        stge.uPinv = lambda u: utilityP_inv(u, gam=stge.CRRA)
        stge.uPinvP = lambda u: utilityP_invP(u, gam=stge.CRRA)
        stge.uinvP = lambda u: utility_invP(u, gam=stge.CRRA)

        if stge.vFuncBool:
            stge.uinv = lambda u: utility_inv(u, gam=stge.CRRA)

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
        vFuncNvrsSlope = stge.MPCminNow ** (-stge.CRRA / (1.0 - stge.CRRA))
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
n        """
        # Reduce cluttered formulae with local copies

        CRRA = self.stg_Now.CRRA
        Rfree = self.stg_Now.Nxt.Rfree
        PermGro = self.stg_Now.Nxt.PermGro
#        hNrmNow = self.stg_Now.hNrmNow
#        RPF = self.stg_Now.RPF
        MPCminNow = self.stg_Now.MPCminNow
        DiscFacEff = self.stg_Now.Nxt.DiscFacEff
        MaxKinks = self.stg_Now.MaxKinks

        # Use local value of BoroCnstArtNxt to prevent comparing None and float
        if self.stg_Now.Nxt.BoroCnstArt is None:
            BoroCnstArt = -np.inf
        else:
            BoroCnstArt = self.stg_Now.Nxt.BoroCnstArt

        # # Calculate human wealth this period
        # self.hNrmNow = (PermGro / Rfree) * (self.solution_next.hNrmNow + 1.0)

        # # Calculate the lower bound of the MPC
        # RPF = ((Rfree * self.stg_Now.Nxt.DiscFacEff) ** (1.0 / self.stg_Now.CRRA)) / Rfree
        # self.stg_Now.MPCminNow = 1.0 / (1.0 + self.stg_Now.RPF / self.solution_next.MPCminNow)

        # Extract kink points in next period's consumption function;
        # don't take the last one; it only defines extrapolation, is not kink.
        mNrmNext = self.solution_next.cFunc.x_list[:-1]
        cNrmNext = self.solution_next.cFunc.y_list[:-1]

        # Calculate the end-of-period asset values that would reach those kink points
        # next period, then invert the first order condition to get consumption. Then
        # find the endogenous gridpoint (kink point) today that corresponds to each kink
        aNrmNow = (PermGro / Rfree) * (mNrmNext - 1.0)
        cNrmNow = (DiscFacEff * Rfree) ** (-1.0 / CRRA) * (
            PermGro * cNrmNext
        )
        mNrmNow = aNrmNow + cNrmNow

        # Add an additional point to the list of gridpoints for the extrapolation,
        # using the new value of the lower bound of the MPC.
        mNrmNow = np.append(mNrmNow, mNrmNow[-1] + 1.0)
        cNrmNow = np.append(cNrmNow, cNrmNow[-1] + MPCminNow)
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
                mXtra = (cNrmNow[-1] - cNrmCnst[-1]) / (1.0 - MPCminNow)
                mCrit = mNrmNow[-1] + mXtra
                cCrit = mCrit - BoroCnstArt
                mNrmNow = np.array([BoroCnstArt, mCrit, mCrit + 1.0])
                cNrmNow = np.array([0.0, cCrit, cCrit + MPCminNow])
                # If the mNrm and cNrm grids have become too large, throw out the last
                # kink point, being sure to adjust the extrapolation.
        if mNrmNow.size > MaxKinks:
            mNrmNow = np.concatenate((mNrmNow[:-2], [mNrmNow[-3] + 1.0]))
            cNrmNow = np.concatenate((cNrmNow[:-2], [cNrmNow[-3] + MPCminNow]))
            # Construct the consumption function as a linear interpolation.
        self.stg_Now.cFunc = LinearInterp(mNrmNow, cNrmNow)
        # Calculate the upper bound of the MPC as the slope of the bottom segment.
        self.stg_Now.MPCmaxNow = (cNrmNow[1] - cNrmNow[0]) / (mNrmNow[1] - mNrmNow[0])

        # Add two attributes to enable calculation of steady state market resources.
        self.stg_Now.Ex_IncNextNrm = 1.0  # Perfect foresight income of 1
        self.stg_Now.mNrmMin = mNrmNow[0]  # Relabeling for compatibility with add_mNrmStE

    def solve(self):
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
        stg_Now = ConsumerSolution(
            cFunc=self.cFunc,
            vFunc=self.vFunc,
            vPfunc=self.vPfunc,
            mNrmMin=self.mNrmMin,
            hNrmNow=self.hNrmNow,
            MPCminNow=self.MPCminNow,
            MPCmaxNow=self.MPCmaxNow,
        )
        self.stg_Now = self.def_utility_funcs(stg_Now)
        self.stg_Now.DiscFacEff = self.stg_Now.DiscFac * \
            self.stg_Now.Nxt.LivPrb  # Effective=pure x LivPrb
        self.stg_Now.make_cFunc_PF()
        self.stg_Now = self.stg_Now.def_value_funcs(self.stg_Now)

        # # Oddly, though the value and consumption functions were included in the solution,
        # # and the inverse utlity function and its derivatives, the baseline setup did not
        # # include the utility function itself.  This should be fixed more systematically,
        # # but for now what is done below will work
        # stg_Now.u = self.u
        # stg_Now.uP = self.uP
        # stg_Now.uPP = self.uPP

        return stg_Now

    def solver_check_AIC_20210404(self, stge, verbose=None):
        """
        Evaluate and report on the Absolute Impatience Condition
        """
        name = "AIC"
        fact = "APF"

        def test(stge): return stge.APF < 1

        messages = {
            True: "\nThe Absolute Patience Factor for the supplied parameter values, APF={0.APF}, satisfies the Absolute Impatience Condition (AIC), which requires APF < 1: "+stge.AICfcts['urlhandle'],
            False: "\nThe Absolute Patience Factor for the supplied parameter values, APF={0.APF}, violates the Absolute Impatience Condition (AIC), which requires APF < 1: "+stge.AICfcts['urlhandle']
        }
        verbose_messages = {
            True: "  Because the APF < 1,  the absolute amount of consumption is expected to fall over time.  \n",
            False: "  Because the APF > 1, the absolute amount of consumption is expected to grow over time.  \n",
        }
        if not hasattr(self, 'verbose'):
            verbose = 0 if verbose is None else verbose
        else:
            verbose = self.verbose if verbose is None else verbose

        core_check_condition(name, test, messages, verbose,
                             verbose_messages, fact, stge)

    def solver_check_FVAC_20210404(self, stge, verbose=None):
        """
        Evaluate and report on the Finite Value of Autarky Condition
        """
        name = "FVAC"
        fact = "FVAF"

        def test(stge): return stge.FVAF < 1

        messages = {
            True: "\nThe Finite Value of Autarky Factor for the supplied parameter values, FVAF={0.FVAF}, satisfies the Finite Value of Autarky Condition, which requires FVAF < 1: "+stge.FVACfcts['urlhandle'],
            False: "\nThe Finite Value of Autarky Factor for the supplied parameter values, FVAF={0.FVAF}, violates the Finite Value of Autarky Condition, which requires FVAF: "+stge.FVACfcts['urlhandle']
        }
        verbose_messages = {
            True: "  Therefore, a nondegenerate solution exists if the RIC also holds. ("+stge.FVACfcts['urlhandle']+")\n",
            False: "  Therefore, a nondegenerate solution exits if the RIC holds.\n",
        }

        if not hasattr(self, 'verbose'):
            verbose = 0 if verbose is None else verbose
        else:
            verbose = self.verbose if verbose is None else verbose

        core_check_condition(name, test, messages, verbose,
                             verbose_messages, fact, stge)

    def solver_check_GICRaw_20210404(self, stge, verbose=None):
        """
        Evaluate and report on the Growth Impatience Condition
        """
        name = "GICRaw"
        fact = "GPFRaw"

        def test(stge): return stge.GPFRaw < 1

        messages = {
            True: "\nThe Growth Patience Factor for the supplied parameter values, GPF={0.GPFRaw}, satisfies the Growth Impatience Condition (GIC), which requires GPF < 1: "+self.stg_Now.GICRawfcts['urlhandle'],
            False: "\nThe Growth Patience Factor for the supplied parameter values, GPF={0.GPFRaw}, violates the Growth Impatience Condition (GIC), which requires GPF < 1: "+self.stg_Now.GICRawfcts['urlhandle'],
        }
        verbose_messages = {
            True: "  Therefore,  for a perfect foresight consumer, the ratio of individual wealth to permanent income is expected to fall indefinitely.    \n",
            False: "  Therefore, for a perfect foresight consumer, the ratio of individual wealth to permanent income is expected to rise toward infinity. \n"
        }
        if not hasattr(self, 'verbose'):
            verbose = 0 if verbose is None else verbose
        else:
            verbose = self.verbose if verbose is None else verbose

        core_check_condition(name, test, messages, verbose,
                             verbose_messages, fact, stge)

    def solver_check_GICLiv_20210404(self, stge, verbose=None):
        name = "GICLiv"
        fact = "GPFLiv"

        def test(stge): return stge.GPFLiv < 1

        messages = {
            True: "\nThe Mortality Adjusted Aggregate Growth Patience Factor for the supplied parameter values, GPFLiv={0.GPFLiv}, satisfies the Mortality Adjusted Aggregate Growth Imatience Condition (GICLiv): "+self.stg_Now.GPFLivfcts['urlhandle'],
            False: "\nThe Mortality Adjusted Aggregate Growth Patience Factor for the supplied parameter values, GPFLiv={0.GPFLiv}, violates the Mortality Adjusted Aggregate Growth Imatience Condition (GICLiv): "+self.stg_Now.GPFLivfcts['urlhandle'],
        }
        verbose_messages = {
            True: "  Therefore, a target level of the ratio of aggregate market resources to aggregate permanent income exists ("+self.stg_Now.GPFLivfcts['urlhandle']+")\n",
            False: "  Therefore, a target ratio of aggregate resources to aggregate permanent income may not exist ("+self.stg_Now.GPFLivfcts['urlhandle']+")\n",
        }
        if not hasattr(self, 'verbose'):
            verbose = 0 if verbose is None else verbose
        else:
            verbose = self.verbose if verbose is None else verbose

        core_check_condition(name, test, messages, verbose,
                             verbose_messages, fact, stge)

    def solver_check_RIC_20210404(self, stge, verbose=None):
        """
        Evaluate and report on the Return Impatience Condition
        """

        name = "RIC"
        fact = "RPF"

        def test(stge): return stge.RPF < 1

        messages = {
            True: "\nThe Return Patience Factor for the supplied parameter values, RPF= {0.RPF}, satisfies the Return Impatience Condition (RIC), which requires RPF < 1: "+self.stg_Now.RPFfcts['urlhandle'],
            False: "\nThe Return Patience Factor for the supplied parameter values, RPF= {0.RPF}, violates the Return Impatience Condition (RIC), which requires RPF < 1: "+self.stg_Now.RPFfcts['urlhandle'],
        }
        verbose_messages = {
            True: "  Therefore, the limiting consumption function is not c(m)=0 for all m\n",
            False: "  Therefore, if the FHWC is satisfied, the limiting consumption function is c(m)=0 for all m.\n",
        }
        if not hasattr(self, 'verbose'):
            verbose = 0 if verbose is None else verbose
        else:
            verbose = self.verbose if verbose is None else verbose

        core_check_condition(name, test, messages, verbose,
                             verbose_messages, fact, stge)

    def solver_check_FHWC_20210404(self, stge, verbose=None):
        """
        Evaluate and report on the Finite Human Wealth Condition
        """
        name = "FHWC"
        fact = "FHWF"

        def test(stge): return stge.FHWF < 1

        messages = {
            True: "\nThe Finite Human Wealth Factor value for the supplied parameter values, FHWF={0.FHWF}, satisfies the Finite Human Wealth Condition (FHWC), which requires FHWF < 1: "+self.stg_Now.FHWCfcts['urlhandle'],
            False: "\nThe Finite Human Wealth Factor value for the supplied parameter values, FHWF={0.FHWF}, violates the Finite Human Wealth Condition (FHWC), which requires FHWF < 1: "+self.stg_Now.FHWCfcts['urlhandle'],
        }
        verbose_messages = {
            True: "  Therefore, the limiting consumption function is not c(m)=Infinity ("+self.stg_Now.FHWCfcts['urlhandle']+")\n  Human wealth normalized by permanent income is {0.hNrmNowInf}.\n",
            False: "  Therefore, the limiting consumption function is c(m)=Infinity for all m unless the RIC is also violated.\n  If both FHWC and RIC fail and the consumer faces a liquidity constraint, the limiting consumption function is nondegenerate but has a limiting slope of 0. ("+self.stg_Now.FHWCfcts['urlhandle']+")\n",
        }
        if not hasattr(self, 'verbose'):
            verbose = 0 if verbose is None else verbose
        else:
            verbose = self.verbose if verbose is None else verbose

        core_check_condition(name, test, messages, verbose,
                             verbose_messages, fact, stge)

    def solver_check_GICNrm_20210404(self, stge, verbose=None):
        """
        Check Individual Growth Patience Factor.
        """
        name = "GICNrm"
        fact = "GPFNrm"

        def test(stge): return stge.GPFNrm <= 1

        messages = {
            True: "\nThe Normalized Growth Patience Factor GPFNrm for the supplied parameter values, GPFNrm={0.GPFNrm}, satisfies the Normalized Growth Impatience Condition (GICNrm), which requires GICNrm < 1: "+self.stg_Now.GPFNrmfcts['urlhandle']+"\n",
            False: "\nThe Normalized Growth Patience Factor GPFNrm for the supplied parameter values, GPFNrm={0.GPFNrm}, violates the Normalized Growth Impatience Condition (GICNrm), which requires GICNrm < 1: "+self.stg_Now.GPFNrmfcts['urlhandle']+"\n",
        }
        verbose_messages = {
            True: " Therefore, a target level of the individual market resources ratio m exists ("+self.stg_Now.GICNrmfcts['urlhandle']+").\n",
            False: " Therefore, a target ratio of individual market resources to individual permanent income does not exist.  ("+self.stg_Now.GICNrmfcts['urlhandle']+")\n",
        }
        if not hasattr(self, 'verbose'):
            verbose = 0 if verbose is None else verbose
        else:
            verbose = self.verbose if verbose is None else verbose

        core_check_condition(name, test, messages, verbose,
                             verbose_messages, fact, stge)

    def solver_check_WRIC_20210404(self, stge, verbose=None):
        """
        Evaluate and report on the Weak Return Impatience Condition
        [url]/#WRPF modified to incorporate LivPrb
        """
        stge.WRPF = (
            (stge.Nxt.UnempPrb ** (1 / stge.CRRA))
            * (stge.Nxt.Rfree * stge.Nxt.DiscFac * stge.Nxt.LivPrb) ** (1 / stge.CRRA)
            / stge.Nxt.Rfree
        )

        stge.WRIC = stge.WRPF < 1
        name = "WRIC"
        fact = "WRPF"

        def test(stge): return stge.WRPF <= 1

        WRICfcts = {'about': 'Weak Return Impatience Condition'}
        WRICfcts.update({'latexexpr': r'\WRIC'})
        WRICfcts.update({'urlhandle': stge.self.stg_Now.urlroot+'WRIC'})
        WRICfcts.update({'py___code': 'test: WRPF < 1'})
        stge.WRICfcts = WRICfcts

        WRPFfcts = {'about': 'Growth Patience Factor'}
        WRPFfcts.update({'latexexpr': r'\WRPF'})
        WRPFfcts.update({'_unicode_': r'℘ RPF'})
        WRPFfcts.update({'urlhandle': stge.self.stg_Now.urlroot+'WRPF'})
        WRPFfcts.update({'py___code': r'UnempPrb * RPF'})

        messages = {
            True: "\nThe Weak Return Patience Factor value for the supplied parameter values, WRPF={0.WRPF}, satisfies the Weak Return Impatience Condition, which requires WRIF < 1: "+stge.WRICfcts['urlhandle'],
            False: "\nThe Weak Return Patience Factor value for the supplied parameter values, WRPF={0.WRPF}, violates the Weak Return Impatience Condition, which requires WRIF < 1: "+stge.WRICfcts['urlhandle'],
        }

        verbose_messages = {
            True: "  Therefore, a nondegenerate solution exists if the FVAC is also satisfied. ("+stge.WRICfcts['urlhandle']+")\n",
            False: "  Therefore, a nondegenerate solution is not available ("+stge.WRICfcts['urlhandle']+")\n",
        }
        if not hasattr(self, 'verbose'):
            verbose = 0 if verbose is None else verbose
        else:
            verbose = self.verbose if verbose is None else verbose

        core_check_condition(name, test, messages, verbose,
                             verbose_messages, fact, stge)

        stge.WRPFfcts = WRPFfcts

    def solver_check_condtnsnew_20210404(self, stg_Now, verbose=None):
        """
        This method checks whether the instance's type satisfies the
        Absolute Impatience Condition (AIC),
        the Return Impatience Condition (RIC),
        the Finite Human Wealth Condition (FHWC), the perfect foresight
        model's Growth Impatience Condition (GICRaw) and
        Perfect Foresight Finite Value of Autarky Condition (FVACPF). Depending on the configuration of parameter values, some
        combination of these conditions must be satisfied in order for the problem to have
        a nondegenerate stg_Now. To check which conditions are required, in the verbose mode
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
        self.stg_Now.conditions = {}

        self.stg_Now.violated = False

        # This method only checks for the conditions for infinite horizon models
        # with a 1 period cycle. If these conditions are not met, we exit early.
        if self.parameters_model['cycles'] != 0 \
           or self.parameters_model['T_cycle'] > 1:
            return

        if not hasattr(self, 'verbose'):
            verbose = 0 if verbose is None else verbose
        else:
            verbose = self.verbose if verbose is None else verbose

        self.solver_check_AIC_20210404(stg_Now, verbose)
        self.solver_check_FHWC_20210404(stg_Now, verbose)
        self.solver_check_RIC_20210404(stg_Now, verbose)
        self.solver_check_GICRaw_20210404(stg_Now, verbose)
        self.solver_check_GICLiv_20210404(stg_Now, verbose)
        self.solver_check_FVAC_20210404(stg_Now, verbose)

        if hasattr(self.stg_Now.Nxt, "BoroCnstArt") and self.stg_Now.Nxt.BoroCnstArt is not None:
            self.stg_Now.violated = not self.stg_Now.conditions["RIC"]
        else:
            self.stg_Now.violated = not self.stg_Now.conditions[
                "RIC"] or not self.stg_Now.conditions["FHWC"]

            ###############################################################################
###############################################################################


class ConsIndShockSetup(ConsPerfForesightSolver):
    """
    A superclass for solvers of one period consumption-saving problems with
    constant relative risk aversion utility and permanent and transitory shocks
    to income.  Has methods to set up but not solve the one period problem.
    N.B.: Because this is a one stge solver, objects that in the full problem
    are lists because they are allowed to vary at different stages, are scalars
    here because the value that is appropriate for the current stage is the one
    that will be passed.  To memorialize that, the "self" versions of such
    variables will have "Nxt" appended to signalize their status as scalars.

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
        included in the reported stg_Now.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear inter-
        polation.
    """

    # Get the "further info" method from the perfect foresight solver
# def add_fcts_to_soln_ConsPerfForesightSolver(self, stg_Now):
    #        super().add_fcts_to_soln(stg_Now)

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
            PermShkDstn,
            TranShkDstn,
    ):
        # Create an empty solution in which to store the inputs
        self.stg_Now = ConsumerSolution()  # create a blank template to fill in
        self.stg_Now.Nxt = TrnsPars()

        # Store them.  Nxt signfier is to remind that they are no longer lists
        self.stg_Nxt = solution_next

        # All variables below are used for evaluating expressions in subsequent steps
        self.stg_Now.Nxt.IncShkDstn = IncShkDstn
        self.stg_Now.Nxt.LivPrb = LivPrb
        self.stg_Now.Nxt.DiscFac = DiscFac
        self.stg_Now.Nxt.CRRA = CRRA
        self.stg_Now.Nxt.Rfree = Rfree
        self.stg_Now.Nxt.PermGroFac = PermGroFac
        self.stg_Now.Nxt.BoroCnstArt = BoroCnstArt
        self.stg_Now.Nxt.PermShkDstn = PermShkDstn
        self.stg_Now.Nxt.TranShkDstn = TranShkDstn

        # These variables govern objects constructed for the current step
        self.stg_Now.aXtraGrid = aXtraGrid
        self.stg_Now.vFuncBool = vFuncBool
        self.stg_Now.CubicBool = CubicBool

        # Old code may expect these things to live at root of agent
        # For now, put them there too, but over time weed out
        self.stg_Now.IncShkDstn = self.stg_Now.Nxt.IncShkDstn
        self.stg_Now.LivPrb = self.stg_Now.Nxt.LivPrb
        self.stg_Now.DiscFac = self.stg_Now.Nxt.DiscFac
        self.stg_Now.CRRA = self.stg_Now.Nxt.CRRA
        self.stg_Now.Rfree = self.stg_Now.Nxt.Rfree
        self.stg_Now.PermGroFac = self.stg_Now.Nxt.PermGroFac
        self.stg_Now.BoroCnstArt = self.stg_Now.Nxt.BoroCnstArt
        self.stg_Now.PermShkDstn = self.stg_Now.Nxt.PermShkDstn
        self.stg_Now.TranShkDstn = self.stg_Now.Nxt.TranShkDstn

        self.stg_Now.fcts = {}

        self.stg_Now = self.def_utility_funcs(self.stg_Now)

        # Generate a url that will locate the documentation
        self.stg_Now.url_doc = "https://hark.readthedocs.io/en/latest/search.html?q=" + \
            self.stg_Now.__class__.__name__+"&check_keywords=yes&area=default#"

        # url for paper that contains various theoretical results
        self.stg_Now.url_ref = "https://econ-ark.github.io/BufferStockTheory"
        self.stg_Now.urlroot = self.stg_Now.url_ref+'/#'  # used for references to derivations

    def add_fcts_to_soln(self, stge_futr):   # Facts
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
        stg_Now = self.stg_Now

        Rfree = stg_Now.Nxt.Rfree
        DiscFac = stg_Now.Nxt.DiscFac
        PermGro = stg_Now.Nxt.PermGroFac
        LivPrb = stg_Now.Nxt.LivPrb
        DiscFacEff = stg_Now.Nxt.DiscFacEff \
            = stg_Now.Nxt.DiscFac * stg_Now.Nxt.LivPrb
        CRRA = stg_Now.Nxt.CRRA
        UnempPrb = stg_Now.Nxt.IncShkDstn.parameters['UnempPrb']
        UnempPrbRet = stg_Now.Nxt.IncShkDstn.parameters['UnempPrbRet']
        urlroot = self.stg_Now.urlroot

        PermShkValsNxtXref = self.stg_Now.Nxt.PermShkValsXref = stg_Now.Nxt.IncShkDstn.X[0]
        TranShkValsNxtXref = self.stg_Now.Nxt.TranShkValsXref = stg_Now.Nxt.IncShkDstn.X[1]

        self.stg_Now.ShkPrbsNxt = ShkPrbsNxt = self.stg_Now.IncShkPrbsNxt \
            = stg_Now.Nxt.IncShkDstn.pmf

        self.stg_Now.IncShkValsNxt = stg_Now.Nxt.IncShkDstn.X

        self.stg_Now.PermShkPrbsNxt = PermShkPrbsNxt = stg_Now.Nxt.PermShkDstn.pmf
        self.stg_Now.PermShkValsNxt = PermShkValsNxt = stg_Now.Nxt.PermShkDstn.X

        self.stg_Now.TranShkPrbsNxt = TranShkPrbsNxt = stg_Now.Nxt.TranShkDstn.pmf
        self.stg_Now.TranShkValsNxt = TranShkValsNxt = stg_Now.Nxt.TranShkDstn.X

        self.stg_Now.PermShkValsNxtMin = PermShkValsNxtMin = np.min(PermShkValsNxt)
        self.stg_Now.TranShkNxtMin = TranShkNxtMin = np.min(TranShkValsNxt)

        # First calc some things needed for formulae that are needed even in the PF model
        self.stg_Now.WorstIncPrbNxt = np.sum(
            ShkPrbsNxt[
                (PermShkValsNxtXref * TranShkValsNxtXref)
                == (PermShkValsNxtMin * TranShkNxtMin)
            ]
        )

        self.stg_Now.Inv_PermShkValsNxt = Inv_PermShkValsNxt = 1/PermShkValsNxt

        self.stg_Now.Ex_Inv_PermShk = Ex_Inv_PermShk =\
            np.dot(Inv_PermShkValsNxt, PermShkPrbsNxt)

        self.stg_Now.Ex_uInv_PermShk = Ex_uInv_PermShk = \
            np.dot(PermShkValsNxt ** (1 - stg_Now.CRRA), PermShkPrbsNxt)

        self.stg_Now.uInv_Ex_uInv_PermShk = uInv_Ex_uInv_PermShk =\
            Ex_uInv_PermShk ** (1 / (1 - stg_Now.CRRA))

        self.add_fcts_to_soln_ConsPerfForesightSolver_20210410(stg_Now)

        # Retrieve a few things constructed by the PF add_info
        PF_RNrm = self.stg_Now.PF_RNrm

        if not hasattr(stg_Now, 'fcts'):
            stg_Now.fcts = {}

        # Many other fcts will have been inherited from the perfect foresight
        # model of which this model, as a descendant, has already inherited
        # Here we need compute only those objects whose value changes when
        # the shock distributions are nondegenerate.
        Ex_IncNextNrmfcts = {
            'about': 'Expected income next period'}
        stg_Now.Ex_IncNextNrm = Ex_IncNextNrm = np.dot(
            ShkPrbsNxt, TranShkValsNxtXref * PermShkValsNxtXref).item()
        Ex_IncNextNrmfcts.update({'latexexpr': r'\Ex_IncNextNrm'})
        Ex_IncNextNrmfcts.update({'_unicode_': r'R/Γ'})
        Ex_IncNextNrmfcts.update({'urlhandle': urlroot+'ExIncNextNrm'})
        Ex_IncNextNrmfcts.update(
            {'py___code': r'np.dot(ShkPrbsNxt,TranShkValsNxtXref*PermShkValsNxtXref)'})
        Ex_IncNextNrmfcts.update({'value_now': Ex_IncNextNrm})
        stg_Now.fcts.update({'Ex_IncNextNrm': Ex_IncNextNrmfcts})
        stg_Now.Ex_IncNextNrmfcts = Ex_IncNextNrmfcts

#        Ex_Inv_PermShk = calc_expectation(            PermShkDstn[0], lambda x: 1 / x        )
        stg_Now.Ex_Inv_PermShk = self.stg_Now.Ex_Inv_PermShk  # Precalc
        Ex_Inv_PermShkfcts = {
            'about': 'Expectation of Inverse of Permanent Shock'}
        Ex_Inv_PermShkfcts.update({'latexexpr': r'\Ex_Inv_PermShk'})
#        Ex_Inv_PermShkfcts.update({'_unicode_': r'R/Γ'})
        Ex_Inv_PermShkfcts.update({'urlhandle': urlroot+'ExInvPermShk'})
        Ex_Inv_PermShkfcts.update({'py___code': r'Rfree/PermGroFacAdj'})
        Ex_Inv_PermShkfcts.update({'value_now': Ex_Inv_PermShk})
        stg_Now.fcts.update({'Ex_Inv_PermShk': Ex_Inv_PermShkfcts})
        stg_Now.Ex_Inv_PermShkfcts = Ex_Inv_PermShkfcts
        # stg_Now.Ex_Inv_PermShk = Ex_Inv_PermShk

        Inv_Ex_Inv_PermShkfcts = {
            'about': 'Inverse of Expectation of Inverse of Permanent Shock'}
        stg_Now.Inv_Ex_Inv_PermShk = Inv_Ex_Inv_PermShk = 1/Ex_Inv_PermShk
        Inv_Ex_Inv_PermShkfcts.update(
            {'latexexpr': '\InvExInvPermShk=\left(\Ex[\PermShk^{-1}]\right)^{-1}'})
#        Inv_Ex_Inv_PermShkfcts.update({'_unicode_': r'R/Γ'})
        Inv_Ex_Inv_PermShkfcts.update({'urlhandle': urlroot+'InvExInvPermShk'})
        Inv_Ex_Inv_PermShkfcts.update({'py___code': r'1/Ex_Inv_PermShk'})
        Inv_Ex_Inv_PermShkfcts.update({'value_now': Inv_Ex_Inv_PermShk})
        stg_Now.fcts.update({'Inv_Ex_Inv_PermShk': Inv_Ex_Inv_PermShkfcts})
        stg_Now.Inv_Ex_Inv_PermShkfcts = Inv_Ex_Inv_PermShkfcts
        # stg_Now.Inv_Ex_Inv_PermShk = Inv_Ex_Inv_PermShk

        Ex_RNrmfcts = {
            'about': 'Expectation of Stochastic-Growth-Normalized Return'}
        Ex_RNrm = PF_RNrm * Ex_Inv_PermShk
        Ex_RNrmfcts.update({'latexexpr': r'\Ex_RNrm'})
#        Ex_RNrmfcts.update({'_unicode_': r'R/Γ'})
        Ex_RNrmfcts.update({'urlhandle': urlroot+'ExRNrm'})
        Ex_RNrmfcts.update({'py___code': r'Rfree/PermGroFacAdj'})
        Ex_RNrmfcts.update({'value_now': Ex_RNrm})
        stg_Now.fcts.update({'Ex_RNrm': Ex_RNrmfcts})
        stg_Now.Ex_RNrmfcts = Ex_RNrmfcts
        stg_Now.Ex_RNrm = Ex_RNrm

        Inv_Ex_RNrmfcts = {
            'about': 'Inverse of Expectation of Stochastic-Growth-Normalized Return'}
        Inv_Ex_RNrm = 1/Ex_RNrm
        Inv_Ex_RNrmfcts.update(
            {'latexexpr': '\InvExInvPermShk=\left(\Ex[\PermShk^{-1}]\right)^{-1}'})
#        Inv_Ex_RNrmfcts.update({'_unicode_': r'R/Γ'})
        Inv_Ex_RNrmfcts.update({'urlhandle': urlroot+'InvExRNrm'})
        Inv_Ex_RNrmfcts.update({'py___code': r'1/Ex_RNrm'})
        Inv_Ex_RNrmfcts.update({'value_now': Inv_Ex_RNrm})
        stg_Now.fcts.update({'Inv_Ex_RNrm': Inv_Ex_RNrmfcts})
        stg_Now.Inv_Ex_RNrmfcts = Inv_Ex_RNrmfcts
        stg_Now.Inv_Ex_RNrm = Inv_Ex_RNrm

        Ex_uInv_PermShkfcts = {
            'about': 'Expected Utility for Consuming Permanent Shock'}

        Ex_uInv_PermShkfcts.update({'latexexpr': r'\Ex_uInv_PermShk'})
        Ex_uInv_PermShkfcts.update({'urlhandle': r'ExuInvPermShk'})
        Ex_uInv_PermShkfcts.update(
            {'py___code': r'np.dot(PermShkValsNxtXref**(1-CRRA),ShkPrbsNxt)'})
        Ex_uInv_PermShkfcts.update({'value_now': Ex_uInv_PermShk})
        stg_Now.fcts.update({'Ex_uInv_PermShk': Ex_uInv_PermShkfcts})
        stg_Now.Ex_uInv_PermShkfcts = Ex_uInv_PermShkfcts
        stg_Now.Ex_uInv_PermShk = Ex_uInv_PermShk = self.stg_Now.Ex_uInv_PermShk

        uInv_Ex_uInv_PermShk = Ex_uInv_PermShk ** (1 / (1 - CRRA))
        uInv_Ex_uInv_PermShkfcts = {
            'about': 'Inverted Expected Utility for Consuming Permanent Shock'}
        uInv_Ex_uInv_PermShkfcts.update({'latexexpr': r'\uInvExuInvPermShk'})
        uInv_Ex_uInv_PermShkfcts.update({'urlhandle': urlroot+'uInvExuInvPermShk'})
        uInv_Ex_uInv_PermShkfcts.update({'py___code': r'Ex_uInv_PermShk**(1/(1-CRRA))'})
        uInv_Ex_uInv_PermShkfcts.update({'value_now': uInv_Ex_uInv_PermShk})
        stg_Now.fcts.update({'uInv_Ex_uInv_PermShk': uInv_Ex_uInv_PermShkfcts})
        stg_Now.uInv_Ex_uInv_PermShkfcts = uInv_Ex_uInv_PermShkfcts
        self.stg_Now.uInv_Ex_uInv_PermShk = stg_Now.uInv_Ex_uInv_PermShk = uInv_Ex_uInv_PermShk
        PermGroFacAdjfcts = {
            'about': 'Uncertainty-Adjusted Permanent Income Growth Factor'}
        PermGroFacAdj = stg_Now.Nxt.PermGro * Inv_Ex_Inv_PermShk
        PermGroFacAdjfcts.update({'latexexpr': r'\mathcal{R}\underline{\permShk}'})
        PermGroFacAdjfcts.update({'urlhandle': urlroot+'PermGroFacAdj'})
        PermGroFacAdjfcts.update({'value_now': PermGroFacAdj})
        stg_Now.fcts.update({'PermGroFacAdj': PermGroFacAdjfcts})
        stg_Now.PermGroFacAdjfcts = PermGroFacAdjfcts
        stg_Now.PermGroFacAdj = PermGroFacAdj

        GPFNrmfcts = {
            'about': 'Normalized Expected Growth Patience Factor'}
        stg_Now.GPFNrm = GPFNrm = stg_Now.GPFRaw * Ex_Inv_PermShk
        GPFNrmfcts.update({'latexexpr': r'\GPFNrm'})
        GPFNrmfcts.update({'_unicode_': r'Þ_Γ'})
        GPFNrmfcts.update({'urlhandle': urlroot+'GPFNrm'})
        GPFNrmfcts.update({'py___code': 'test: GPFNrm < 1'})
        stg_Now.fcts.update({'GPFNrm': GPFNrmfcts})
        stg_Now.GPFNrmfcts = GPFNrmfcts

        GICNrmfcts = {'about': 'Growth Impatience Condition'}
        GICNrmfcts.update({'latexexpr': r'\GICNrm'})
        GICNrmfcts.update({'urlhandle': urlroot+'GICNrm'})
        GICNrmfcts.update({'py___code': 'test: GPFNrm < 1'})
        stg_Now.fcts.update({'GICNrm': GICNrmfcts})
        stg_Now.GICNrmfcts = GICNrmfcts

        FVACfcts = {'about': 'Finite Value of Autarky Condition'}
        FVACfcts.update({'latexexpr': r'\FVAC'})
        FVACfcts.update({'urlhandle': urlroot+'FVAC'})
        FVACfcts.update({'py___code': 'test: FVAF < 1'})
        stg_Now.fcts.update({'FVAC': FVACfcts})
        stg_Now.FVACfcts = FVACfcts

        DiscGPFNrmCuspfcts = {'about':
                              'DiscFac s.t. GPFNrm = 1'}
        stg_Now.DiscGPFNrmCusp = DiscGPFNrmCusp = (
            (stg_Now.Nxt.PermGro*Inv_Ex_Inv_PermShk)**(CRRA))/Rfree
        DiscGPFNrmCuspfcts.update({'latexexpr': ''})
        DiscGPFNrmCuspfcts.update({'value_now': DiscGPFNrmCusp})
        DiscGPFNrmCuspfcts.update({
            'py___code': '((PermGro * Inv_Ex_Inv_PermShk) ** CRRA)/(Rfree)'})
        stg_Now.fcts.update({'DiscGPFNrmCusp': DiscGPFNrmCuspfcts})
        stg_Now.DiscGPFNrmCuspfcts = DiscGPFNrmCuspfcts

        # # Merge all the parameters
        # # In python 3.9, the syntax is new_dict = dict_a | dict_b
        # stg_Now.params_all = {**self.params_cons_ind_shock_setup_init,
        #                    **params_cons_ind_shock_setup_set_and_update_values}

        # Now that the calculations are done, store results in self.
        # self, here, is the solver
        # goal: agent,  solver, and solution should be standalone
        # this requires the solution to get some info from the solver

        if stg_Now.Inv_PF_RNrm < 1:        # Finite if Rfree > stg_Now.Nxt.PermGro
            stg_Now.hNrmNowInf = 1/(1-stg_Now.Inv_PF_RNrm)

        # Given m, value of c where E[m_{t+1}]=m_{t}
        # url/#
        stg_Now.c_where_Ex_mtp1_minus_mt_eq_0 = (
            lambda m_t:
            m_t * (1 - stg_Now.Inv_Ex_RNrm) + (stg_Now.Inv_Ex_RNrm)
        )

        # Given m, value of c where E[mLev_{t+1}/mLev_{t}]=stg_Now.Nxt.PermGro
        # Solves for c in equation at url/#balgrostable

        stg_Now.c_where_Ex_PermShk_times_mtp1_minus_mt_eq_0 = (
            lambda m_t:
            m_t * (1 - stg_Now.Inv_PF_RNrm) + stg_Now.Inv_PF_RNrm
        )

        # E[c_{t+1} pLev_{t+1}/pLev_{t}] as a fn of a_{t}
        stg_Now.Ex_cLev_tp1_Over_pLev_t_from_at = (
            lambda a_t:
            np.dot(stg_Now.Nxt.PermGro *
                   stg_Now.Nxt.PermShkValsXref *
                   stg_Now.cFunc(
                       (stg_Now.PF_RNrm/stg_Now.Nxt.PermShkValsXref) * a_t
                       + stg_Now.Nxt.TranShkValsXref
                   ),
                   stg_Now.ShkPrbsNxt)
        )

        stg_Now.c_where_Ex_mtp1_minus_mt_eq_0 = c_where_Ex_mtp1_minus_mt_eq_0 = (
            lambda m_t:
            m_t * (1 - 1/stg_Now.Ex_RNrm) + (1/stg_Now.Ex_RNrm)
        )

        # Solve the equation at url/#balgrostable
        stg_Now.c_where_Ex_PermShk_times_mtp1_minus_mt_eq_0 = \
            c_where_Ex_PermShk_times_mtp1_minus_mt_eq_0 = (
                lambda m_t:
                (m_t * (1 - 1/stg_Now.PF_RNrm)) + (1/stg_Now.PF_RNrm)
            )

        # mNrmTrg solves Ex_RNrm*(m - c(m)) + E[inc_next] - m = 0
        Ex_m_tp1_minus_m_t = (
            lambda m_t:
            stg_Now.Ex_RNrm * (m_t - stg_Now.cFunc(m_t)) +
            stg_Now.Ex_IncNextNrm - m_t
        )
        stg_Now.Ex_m_tp1_minus_m_t = Ex_m_tp1_minus_m_t

        stg_Now.Ex_cLev_tp1_Over_pLev_t_from_at = Ex_cLev_tp1_Over_pLev_t_from_at = (
            lambda a_t:
            np.dot(
                stg_Now.Nxt.PermShkValsXref * stg_Now.Nxt.PermGro * stg_Now.cFunc(
                    (stg_Now.PF_RNrm/stg_Now.Nxt.PermShkValsXref) *
                    a_t + stg_Now.Nxt.TranShkValsXref
                ),
                stg_Now.ShkPrbsNxt)
        )

        stg_Now.Ex_PermShk_tp1_times_m_tp1_minus_m_t = \
            Ex_PermShk_tp1_times_m_tp1_minus_m_t = (
                lambda m_t: self.stg_Now.PF_RNrm *
                (m_t - stg_Now.cFunc(m_t)) + 1.0 - m_t
            )

        return stg_Now

    def prepare_to_solve(self):
        """
        Perform preparatory work when the solver is first invoked.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        # self.stg_Now.solver_check_condtnsnew_20210404 = self.solver_check_condtnsnew_20210404
        # self.stg_Now.solver_check_AIC_20210404 = self.solver_check_AIC_20210404
        # self.stg_Now.solver_check_RIC_20210404 = self.solver_check_RIC_20210404
        # self.stg_Now.solver_check_FVAC_20210404 = self.solver_check_FVAC_20210404
        # self.stg_Now.solver_check_GICLiv_20210404 = self.solver_check_GICLiv_20210404
        # self.stg_Now.solver_check_GICRaw_20210404 = self.solver_check_GICRaw_20210404
        # self.stg_Now.solver_check_GICNrm_20210404 = self.solver_check_GICNrm_20210404
        # self.stg_Now.solver_check_FHWC_20210404 = self.solver_check_FHWC_20210404
        # self.stg_Now.solver_check_WRIC_20210404 = self.solver_check_WRIC_20210404

        # Define a few variables that permit the same formulae to be used for
        # versions with and without uncertainty
        # We are in the perfect foresight model now so these are all 1.0

        self.PerfFsgt = (type(self) == ConsPerfForesightSolver)

        # If no uncertainty, return the degenerate targets for the PF model
        if hasattr(self, "TranShkMinNext"):  # Then it has transitory shocks
            # Handle the degenerate case where shocks are of size zero
            if ((self.stg_Now.TranShkMinNext == 1.0) and (self.stg_Now.PermShkMinNext == 1.0)):
                # But they still might have unemployment risk
                if hasattr(self, "UnempPrb"):
                    if ((self.stg_Now.UnempPrb == 0.0) or (self.stg_Now.IncUnemp == 1.0)):
                        self.PerfFsgt = True  # No unemployment risk either
                    else:
                        self.PerfFsgt = False  # The only kind of uncertainty is unemployment

        if self.PerfFsgt:
            self.stg_Now.Ex_Inv_PermShk = 1.0
            self.stg_Now.Ex_uInv_PermShk = 1.0

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
        aNrmNow : np.array
            A 1D array of end-of-period assets; also stored as attribute of self.stg_Now.
        """

        # We define aNrmNow all the way from BoroCnstNat up to max(self.aXtraGrid)
        # even if BoroCnstNat < BoroCnstArt, so we can construct the consumption
        # function as the lower envelope of the (by the artificial borrowing con-
        # straint) unconstrained consumption function, and the artificially con-
        # strained consumption function.
        self.stg_Now.aNrmNow = np.asarray(
            self.stg_Now.aXtraGrid) + self.stg_Now.BoroCnstNat

        return self.stg_Now.aNrmNow

    def calc_EndOfPrdvP(self):
        """
        Calculate end-of-period marginal value of assets at each point in aNrmNow.
        Does so by taking a weighted sum of next period marginal values across
        income shocks (in a preconstructed grid self.stg_Now.mNrmNext).

        Parameters
        ----------
        none

        Returns
        -------
        EndOfPrdvP : np.array
            A 1D array of end-of-period marginal value of assets
        """

        def vp_next(shocks, a_Nrm_Val):
            return shocks[0] ** (-self.stg_Now.CRRA) \
                * self.stg_Nxt.vPfunc(self.m_Nrm_tp1(shocks, a_Nrm_Val))

        EndOfPrdvP = (
            self.stg_Now.Nxt.DiscFac * self.stg_Now.Nxt.LivPrb
            * self.stg_Now.Nxt.Rfree
            * self.stg_Now.Nxt.PermGro ** (-self.stg_Now.CRRA)
            * calc_expectation(
                self.stg_Now.Nxt.IncShkDstn,
                vp_next,
                self.stg_Now.aNrmNow
            )
        )

        return EndOfPrdvP

    def get_source_points_via_EGM(self, EndOfPrdvP, aNrmNow):
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
        cNrmNow = self.stg_Now.uPinv(EndOfPrdvP)
        mNrmNow = cNrmNow + aNrmNow

        # Limiting consumption is zero as m approaches mNrmMin
        c_for_interpolation = np.insert(cNrmNow, 0, 0.0, axis=-1)
        m_for_interpolation = np.insert(mNrmNow, 0, self.stg_Now.BoroCnstNat, axis=-1)

        # Store these for calcvFunc
        self.stg_Now.cNrmNow = cNrmNow
        self.stg_Now.mNrmNow = mNrmNow

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
        cFuncNowUnc = interpolator(mNrm, cNrm)

        # Combine the constrained and unconstrained functions into the true consumption function
        # by choosing the lower of the constrained and unconstrained functions
        # LowerEnvelope should only be used when BoroCnstArt is true
        if self.stg_Now.BoroCnstArt is None:
            cFuncNow = cFuncNowUnc
        else:
            self.stg_Now.cFuncNowCnst = LinearInterp(
                np.array([self.stg_Now.mNrmMin, self.stg_Now.mNrmMin + 1]
                         ), np.array([0.0, 1.0]))
            cFuncNow = LowerEnvelope(cFuncNowUnc, self.stg_Now.cFuncNowCnst, nan_bool=False)

        # Make the marginal value function and the marginal marginal value function
        vPfuncNow = MargValueFuncCRRA(cFuncNow, self.stg_Now.CRRA)

        # Pack up the solution and return it
        solution_interpolating = ConsumerSolution(
            cFunc=cFuncNow,
            vPfunc=vPfuncNow,
            mNrmMin=self.stg_Now.mNrmMin
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
        none (relies upon self.stg_Now.aNrmNow existing before invocation)

        Returns
        -------
        solution : ConsumerSolution
            The solution to the single period consumption-saving problem.
        """
        self.stg_Now.aNrmNow = self.prepare_to_calc_EndOfPrdvP()
        self.stg_Now.EndOfPrdvP = self.calc_EndOfPrdvP()

        # Construct a solution for this period
        if self.stg_Now.CubicBool:
            stg_Now = self.interpolating_EGM_solution(
                self.stg_Now.EndOfPrdvP, self.stg_Now.aNrmNow, interpolator=self.make_cubic_cFunc
            )
        else:
            stg_Now = self.interpolating_EGM_solution(
                self.stg_Now.EndOfPrdvP, self.stg_Now.aNrmNow, interpolator=self.make_linear_cFunc
            )
        return stg_Now

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
            mNrm, cNrm, self.stg_Now.cFuncLimitIntercept, self.stg_Now.cFuncLimitSlope
        )
        return cFunc_unconstrained

    def solve(self):  # make self.stg_Now from self.stg_Nxt
        """
        Solves (one period/stage of) the single period consumption-saving problem using the
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
        if self.stg_Nxt.stge_kind['iter_status'] == 'finished':
            self.stg_Now.stge_kind['iter_status'] = 'finished'
            _log.error("The model has already been solved.  Aborting.")
            return self.stg_Now

        # If this is the first invocation of solve, just flesh out the terminal
        # period solution so it is a proper starting point for iteration
        if self.stg_Nxt.stge_kind['iter_status'] == 'terminal':
            self.Nxt = self.stg_Now.Nxt  # Get parameter values
            self.stg_Now = self.stg_Nxt
            self.stg_Now.Nxt = self.Nxt
            self.stg_Now.stge_kind['iter_status'] = 'iterator'
            self.stg_Now = self.def_utility_funcs(self.stg_Now)
            self.stg_Now = self.def_value_funcs(self.stg_Now)
            self.stg_Now.vPfunc = MargValueFuncCRRA(self.stg_Now.cFunc, self.stg_Now.CRRA)
            self.stg_Now.vPPfunc = MargMargValueFuncCRRA(
                self.stg_Now.cFunc, self.stg_Now.CRRA)
#            self.add_Ex_values(self.stg_Now)
            self.add_fcts_to_soln(self.stg_Nxt)  # Do not iterate MPC and hMin
            return self.stg_Now  # Replaces original "terminal" solution; next stg_Nxt

        self.stg_Now.stge_kind = {'iter_status': 'iterator'}
        # Add a bunch of metadata

        self.add_fcts_to_soln(self.stg_Nxt)
        # self.stg_Now = self.solution_add_MPC_bounds_and_human_wealth_PDV_20210410(self.stg_Now)
        sol_EGM = self.make_sol_using_EGM()  # Need to add test for finished, change stge_kind if so
        self.stg_Now.cFunc = sol_EGM.cFunc
        self.stg_Now.vPfunc = sol_EGM.vPfunc

        # Add the value function if requested, as well as the marginal marginal
        # value function if cubic splines were used for interpolation
        if self.stg_Now.vFuncBool:
            self.stg_Now = self.add_vFunc(self.stg_Now, self.EndOfPrdvP)
        if self.stg_Now.CubicBool:
            self.stg_Now = self.add_vPPfunc(self.stg_Now)

        return self.stg_Now

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
        return self.stg_Now.Nxt.Rfree / (self.stg_Now.Nxt.PermGro * shocks[0]) \
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

        Requires self.stg_Now.aNrmNow to have been computed already.

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
            return shocks[0] ** (- self.stg_Now.CRRA - 1.0) \
                * self.stg_Nxt.vPPfunc(self.m_Nrm_tp1(shocks, a_Nrm_Val))

        EndOfPrdvPP = (
            self.stg_Now.Nxt.DiscFac * self.stg_Now.Nxt.LivPrb
            * self.stg_Now.Nxt.Rfree
            * self.stg_Now.Nxt.Rfree
            * self.stg_Now.Nxt.PermGro ** (-self.stg_Now.CRRA - 1.0)
            * calc_expectation(
                self.stg_Now.Nxt.IncShkDstn,
                vPP_next,
                self.stg_Now.aNrmNow
            )
        )
        dcda = EndOfPrdvPP / self.stg_Now.uPP(np.array(cNrm_Vec[1:]))
        MPC = dcda / (dcda + 1.0)
        MPC = np.insert(MPC, 0, self.stg_Now.MPCmaxNow)

        cFuncNowUnc = CubicInterp(
            mNrm_Vec, cNrm_Vec, MPC, self.stg_Now.MPCminNow *
            self.stg_Now.hNrmNow, self.stg_Now.MPCminNow
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
            asset values in self.stg_Now.aNrmNow.

        Returns
        -------
        none
        """
        def v_Lvl_next(shocks, a_Nrm_Val):
            return (
                shocks[0] ** (1.0 - self.stg_Now.CRRA)
                * self.stg_Now.Nxt.PermGro ** (1.0 - self.stg_Now.CRRA)
            ) * self.stg_Now.vFuncNext(self.stg_Now.m_Nrm_tp1(shocks, a_Nrm_Val))
        EndOfPrdv = self.stg_Now.Nxt.DiscFacEff * calc_expectation(
            self.stg_Now.Nxt.IncShkDstn, v_Lvl_next, self.stg_Now.aNrmNow
        )
        EndOfPrdvNvrs = self.stg_Now.uinv(
            EndOfPrdv
        )  # value transformed through inverse utility
        EndOfPrdvNvrsP = EndOfPrdvP * self.stg_Now.uinvP(EndOfPrdv)
        EndOfPrdvNvrs = np.insert(EndOfPrdvNvrs, 0, 0.0)
        EndOfPrdvNvrsP = np.insert(
            EndOfPrdvNvrsP, 0, EndOfPrdvNvrsP[0]
        )  # This is a very good approximation, vNvrsPP = 0 at the asset minimum
        aNrm_temp = np.insert(self.stg_Now.aNrmNow, 0, self.stg_Now.BoroCnstNat)
        EndOfPrdvNvrsFunc = CubicInterp(aNrm_temp, EndOfPrdvNvrs, EndOfPrdvNvrsP)
        self.stg_Now.EndOfPrdvFunc = ValueFuncCRRA(
            EndOfPrdvNvrsFunc, self.stg_Now.CRRA)

    def add_vFunc(self, stg_Now, EndOfPrdvP):
        """
        Creates the value function for this period and adds it to the stg_Now.

        Parameters
        ----------
        solution : ConsumerSolution
            The solution to this single period problem, likely including the
            consumption function, marginal value function, etc.
        EndOfPrdvP : np.array
            Array of end-of-period marginal value of assets corresponding to the
            asset values in self.stg_Now.aNrmNow.

        Returns
        -------
        solution : ConsumerSolution
            The single period solution passed as an input, but now with the
            value function (defined over market resources m) as an attribute.
        """
        self.make_EndOfPrdvFunc(EndOfPrdvP)
        stg_Now.vFunc = self.make_vFunc(stg_Now)
        return stg_Now

    def make_vFunc(self, stg_Now):
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
        mNrm_temp = self.stg_Now.mNrmMin + self.stg_Now.aXtraGrid
        cNrmNow = stg_Now.cFunc(mNrm_temp)
        aNrmNow = mNrm_temp - cNrmNow
        vNrmNow = self.stg_Now.u(cNrmNow) + self.EndOfPrdvFunc(aNrmNow)
        vPnow = self.uP(cNrmNow)

        # Construct the beginning value function
        vNvrs = self.stg_Now.uinv(vNrmNow)  # value transformed through inverse utility
        vNvrsP = vPnow * self.stg_Now.uinvP(vNrmNow)
        mNrm_temp = np.insert(mNrm_temp, 0, self.stg_Now.mNrmMin)
        vNvrs = np.insert(vNvrs, 0, 0.0)
        vNvrsP = np.insert(
            vNvrsP, 0, self.stg_Now.MPCmaxNowEff ** (-self.stg_Now.CRRA /
                                                  (1.0 - self.stg_Now.CRRA))
        )
        MPCminNowNvrs = self.stg_Now.MPCminNow ** (-self.stg_Now.CRRA /
                                                (1.0 - self.stg_Now.CRRA))
        vNvrsFuncNow = CubicInterp(
            mNrm_temp, vNvrs, vNvrsP, MPCminNowNvrs * self.stg_Now.hNrmNow, MPCminNowNvrs
        )
        vFuncNow = ValueFuncCRRA(vNvrsFuncNow, self.stg_Now.CRRA)
        return vFuncNow

    def add_vPPfunc(self, stg_Now):
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
        vPPfuncNow = MargMargValueFuncCRRA(stg_Now.cFunc, stg_Now.CRRA)
        stg_Now.vPPfunc = vPPfuncNow
        return stg_Now


####################################################################################################
####################################################################################################
class ConsKinkedRsolver(ConsIndShockSolver):
    """
    A class to solve a single period consumption-saving problem where the interest
    rate on debt differs from the interest rate on savings.  Inherits from
    ConsIndShockSolver, with nearly identical inputs and outputs.  The key diff-
    erence is that Rfree is replaced by Rsave (a>0) and Rboro (a<0).  The solver
    can handle Rboro == Rsave, which makes it identical to ConsIndShocksolver, but
    it terminates immediately if Rboro < Rsave, as this has a different stg_Now.

    Parameters
    ----------
    stg_Nxt : ConsumerSolution
        The solution to next period's one period problem.
    IncShkDstn : distribution.Distribution
        A discrete
        approximation to the income process between the period being solved
        and the one immediately following (in stg_Nxt).
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
        included in the reported stg_Now.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear inter-
        polation.
    """

    def __init__(
            self,
            stg_Nxt,
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
            stg_Nxt,
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
        self.Nxt.Rboro = self.Rboro = Rboro
        self.Nxt.Rsave = self.Rsave = Rsave
        self.Nxt.cnstrct = {'vFuncBool', 'IncShkDstn'}

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
            self.Nxt.Rboro > self.Nxt.Rsave
        )  # Boolean indicating that there is actually a kink.
        # When Rboro == Rsave, this method acts just like it did in IndShock.
        # When Rboro < Rsave, the solver would have terminated when it was called.

        # Make a grid of end-of-period assets, including *two* copies of a=0
        if KinkBool:
            aNrmNow = np.sort(
                np.hstack(
                    (np.asarray(self.aXtraGrid) + self.mNrmMin, np.array([0.0, 0.0]))
                )
            )
        else:
            aNrmNow = np.asarray(self.aXtraGrid) + self.mNrmMin
            aXtraCount = aNrmNow.size

        # Make tiled versions of the assets grid and income shocks
        ShkCount = self.TranShkValsNxt.size
        aNrm_temp = np.tile(aNrmNow, (ShkCount, 1))
        PermShkValsNxt_temp = (np.tile(self.Nxt.PermShkVals, (aXtraCount, 1))).transpose()
        TranShkValsNxt_temp = (np.tile(self.TranShkValsNxtNext, (aXtraCount, 1))).transpose()
        ShkPrbs_temp = (np.tile(self.ShkPrbsNxt, (aXtraCount, 1))).transpose()

        # Make a 1D array of the interest factor at each asset gridpoint
        Rfree_vec = self.Nxt.Rsave * np.ones(aXtraCount)
        if KinkBool:
            self.i_kink = (
                np.sum(aNrmNow <= 0) - 1
            )  # Save the index of the kink point as an attribute
            Rfree_vec[0: self.i_kink] = self.Nxt.Rboro
            Rfree = Rfree_vec
            Rfree_temp = np.tile(Rfree_vec, (ShkCount, 1))

        # Make an array of market resources that we could have next period,
        # considering the grid of assets and the income shocks that could occur
        mNrmNext = (
            Rfree_temp / (self.PermGroFac * PermShkValsNxt_temp) * aNrm_temp
            + TranShkValsNxt_temp
        )

        # Recalculate the minimum MPC and human wealth using the interest factor on saving.
        # This overwrites values from set_and_update_values, which were based on Rboro instead.
        if KinkBool:
            RPFTop = (
                (self.Nxt.Rsave * self.DiscFacEff) ** (1.0 / self.CRRA)
            ) / self.Nxt.Rsave
            self.MPCminNow = 1.0 / (1.0 + RPFTop / self.stg_Nxt.MPCminNow)
            self.hNrmNow = (
                self.PermGroFac
                / self.Nxt.Rsave
                * (
                    np.dot(
                        self.ShkPrbsNxt, self.TranShkValsNxtNext * self.Nxt.PermShkVals
                    )
                    + self.stg_Nxt.hNrmNow
                )
            )

        # Store some of the constructed arrays for later use and return the assets grid
        self.Nxt.PermShkVals_temp = PermShkValsNxt_temp
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
    # Optional extra fcts about the model and its calibration
}

init_perfect_foresight.update(dict({'fcts': {'import': 'init_perfect_foresight'}}))

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

# Optional more detailed fcts about various parameters
CRRAfcts = {'about': 'Coefficient of Relative Risk Aversion'}
CRRAfcts.update({'latexexpr': '\providecommand{\CRRA}{\rho}\CRRA'})
CRRAfcts.update({'_unicode_': 'ρ'})  # \rho is Greek r: relative risk aversion rrr
CRRAfcts.update({'prmtv_par': 'True'})
init_perfect_foresight['prmtv_par'].append('CRRA')
init_perfect_foresight['fcts'].update({'CRRA': CRRAfcts})
init_perfect_foresight.update({'CRRAfcts': CRRAfcts})

DiscFacfcts = {'about': 'Pure time preference rate'}
DiscFacfcts.update({'latexexpr': '\providecommand{\DiscFac}{\beta}\DiscFac'})
DiscFacfcts.update({'_unicode_': 'β'})
DiscFacfcts.update({'prmtv_par': 'True'})
init_perfect_foresight['prmtv_par'].append('DiscFac')
init_perfect_foresight['fcts'].update({'DiscFac': DiscFacfcts})
init_perfect_foresight.update({'DiscFacfcts': DiscFacfcts})

LivPrbfcts = {'about': 'Probability of survival from this period to next'}
LivPrbfcts.update({'latexexpr': '\providecommand{\LivPrb}{\Pi}\LivPrb'})
LivPrbfcts.update({'_unicode_': 'Π'})  # \Pi mnemonic: 'Probability of surival'
LivPrbfcts.update({'prmtv_par': 'True'})
init_perfect_foresight['prmtv_par'].append('LivPrb')
init_perfect_foresight['fcts'].update({'LivPrb': LivPrbfcts})
init_perfect_foresight.update({'LivPrbfcts': LivPrbfcts})

Rfreefcts = {'about': 'Risk free interest factor'}
Rfreefcts.update({'latexexpr': '\providecommand{\Rfree}{\mathsf{R}}\Rfree'})
Rfreefcts.update({'_unicode_': 'R'})
Rfreefcts.update({'prmtv_par': 'True'})
init_perfect_foresight['prmtv_par'].append('Rfree')
init_perfect_foresight['fcts'].update({'Rfree': Rfreefcts})
init_perfect_foresight.update({'Rfreefcts': Rfreefcts})

PermGroFacfcts = {'about': 'Growth factor for permanent income'}
PermGroFacfcts.update({'latexexpr': '\providecommand{\PermGroFac}{\Gamma}\PermGroFac'})
PermGroFacfcts.update({'_unicode_': 'Γ'})  # \Gamma is Greek G for Growth
PermGroFacfcts.update({'prmtv_par': 'True'})
init_perfect_foresight['prmtv_par'].append('PermGroFac')
init_perfect_foresight['fcts'].update({'PermGroFac': PermGroFacfcts})
init_perfect_foresight.update({'PermGroFacfcts': PermGroFacfcts})

PermGroFacAggfcts = {'about': 'Growth factor for aggregate permanent income'}
# PermGroFacAggfcts.update({'latexexpr': '\providecommand{\PermGroFacAgg}{\Gamma}\PermGroFacAgg'})
# PermGroFacAggfcts.update({'_unicode_': 'Γ'})  # \Gamma is Greek G for Growth
PermGroFacAggfcts.update({'prmtv_par': 'True'})
init_perfect_foresight['prmtv_par'].append('PermGroFacAgg')
init_perfect_foresight['fcts'].update({'PermGroFacAgg': PermGroFacAggfcts})
init_perfect_foresight.update({'PermGroFacAggfcts': PermGroFacAggfcts})

BoroCnstArtfcts = {'about': 'If not None, maximum proportion of permanent income borrowable'}
BoroCnstArtfcts.update({'latexexpr': r'\providecommand{\BoroCnstArt}{\underline{a}}\BoroCnstArt'})
BoroCnstArtfcts.update({'prmtv_par': 'True'})
init_perfect_foresight['prmtv_par'].append('BoroCnstArt')
init_perfect_foresight['fcts'].update({'BoroCnstArt': BoroCnstArtfcts})
init_perfect_foresight.update({'BoroCnstArtfcts': BoroCnstArtfcts})

MaxKinksfcts = {'about': 'PF Constrained model solves to period T-MaxKinks,'
                ' where the solution has exactly this many kink points'}
MaxKinksfcts.update({'prmtv_par': 'False'})
# init_perfect_foresight['prmtv_par'].append('MaxKinks')
init_perfect_foresight['fcts'].update({'MaxKinks': MaxKinksfcts})
init_perfect_foresight.update({'MaxKinksfcts': MaxKinksfcts})

mcrlo_AgentCountfcts = {'about': 'Number of agents to use in baseline Monte Carlo simulation'}
mcrlo_AgentCountfcts.update(
    {'latexexpr': '\providecommand{\mcrlo_AgentCount}{N}\mcrlo_AgentCount'})
mcrlo_AgentCountfcts.update({'mcrlo_sim': 'True'})
mcrlo_AgentCountfcts.update({'mcrlo_lim': 'infinity'})
# init_perfect_foresight['mcrlo_sim'].append('mcrlo_AgentCount')
init_perfect_foresight['fcts'].update({'mcrlo_AgentCount': mcrlo_AgentCountfcts})
init_perfect_foresight.update({'mcrlo_AgentCountfcts': mcrlo_AgentCountfcts})

aNrmInitMeanfcts = {'about': 'Mean initial population value of aNrm'}
aNrmInitMeanfcts.update({'mcrlo_sim': 'True'})
aNrmInitMeanfcts.update({'mcrlo_lim': 'infinity'})
init_perfect_foresight['mcrlo_sim'].append('aNrmInitMean')
init_perfect_foresight['fcts'].update({'aNrmInitMean': aNrmInitMeanfcts})
init_perfect_foresight.update({'aNrmInitMeanfcts': aNrmInitMeanfcts})

aNrmInitStdfcts = {'about': 'Std dev of initial population value of aNrm'}
aNrmInitStdfcts.update({'mcrlo_sim': 'True'})
init_perfect_foresight['mcrlo_sim'].append('aNrmInitStd')
init_perfect_foresight['fcts'].update({'aNrmInitStd': aNrmInitStdfcts})
init_perfect_foresight.update({'aNrmInitStdfcts': aNrmInitStdfcts})

mcrlo_pLvlInitMeanfcts = {'about': 'Mean initial population value of log pLvl'}
mcrlo_pLvlInitMeanfcts.update({'mcrlo_sim': 'True'})
init_perfect_foresight['mcrlo_sim'].append('mcrlo_pLvlInitMean')
init_perfect_foresight['fcts'].update({'mcrlo_pLvlInitMean': mcrlo_pLvlInitMeanfcts})
init_perfect_foresight.update({'mcrlo_pLvlInitMeanfcts': mcrlo_pLvlInitMeanfcts})

mcrlo_pLvlInitStdfcts = {'about': 'Mean initial std dev of log ppLvl'}
mcrlo_pLvlInitStdfcts.update({'mcrlo_sim': 'True'})
init_perfect_foresight['mcrlo_sim'].append('mcrlo_pLvlInitStd')
init_perfect_foresight['fcts'].update({'mcrlo_pLvlInitStd': mcrlo_pLvlInitStdfcts})
init_perfect_foresight.update({'mcrlo_pLvlInitStdfcts': mcrlo_pLvlInitStdfcts})

T_agefcts = {
    'about': 'Age after which simulated agents are automatically killedl'}
T_agefcts.update({'mcrlo_sim': 'False'})
init_perfect_foresight['fcts'].update({'T_age': T_agefcts})
init_perfect_foresight.update({'T_agefcts': T_agefcts})

T_cyclesfcts = {
    'about': 'Number of periods in a "cycle" (like, a lifetime) for this agent type'}
init_perfect_foresight['fcts'].update({'T_cycle': T_cyclesfcts})
init_perfect_foresight.update({'T_cyclesfcts': T_cyclesfcts})

cyclesfcts = {
    'about': 'Number of times the sequence of periods/stages should be solved'}
init_perfect_foresight['fcts'].update({'cycle': cyclesfcts})
init_perfect_foresight.update({'cyclesfcts': cyclesfcts})


class PerfForesightConsumerType(AgentType):
    """
    A perfect foresight consumer type who has no uncertainty other than mortality.
    The problem is defined by a coefficient of relative risk aversion, intertemporal
    discount factor, interest factor, an artificial borrowing constraint (maybe)
    and time sequences of the permanent income growth rate and survival probability.

    Parameters
    ----------
    cycles : int
        Number of times the sequence of periods/stages should be solved.
    """

    # Define some universal options for all consumer types
    # Use underscores (_) to define useful defaults available to all inheritors
    # These should not be modified -- if used, a deepcopy should
    # be made and that should be modified

    # Consumption function in last period in which everything is consumed
    # Two names for the same thing
    # Define as LinearInterp object because it has derivatives
    cFunc_terminal_nobequest_ = LinearInterp([0.0, 1.0], [0.0, 1.0])
    cFunc_terminal_ = LinearInterp([0.0, 1.0], [0.0, 1.0])

    solution_nobequest_ = ConsumerSolution(  # Can't include vFunc b/c u not yet def
        cFunc=cFunc_terminal_nobequest_,
        mNrmMin=0.0,
        hNrmNow=0.0,
        MPCminNow=1.0,
        MPCmaxNow=1.0,
        stge_kind={'iter_status': 'terminal'}
    )
    solution_nobequest = deepcopy(solution_nobequest_)  # Modifiable copy

    solution_terminal_ = solution_nobequest_         # Default terminal solution
    solution_terminal = deepcopy(solution_terminal_)  # Modifiable copy

    time_vary_ = ["LivPrb",  # Age-varying death rates can match mortality data
                  "PermGroFac"]  # Age-varying income growth can match data
    time_inv_ = ["CRRA", "Rfree", "DiscFac", "MaxKinks", "BoroCnstArt"]
    state_vars = ['pLvl',  # Idiosyncratic permanent income
                  'PlvlAgg',  # Aggregate permanent income
                  'bNrm',  # Bank balances at beginning of period (normed)
                  'mNrm',  # Market resources (b + income) normed
                  "aNrm"]  # Assets after all actions (normed)
    shock_vars_ = []

    def __init__(self, cycles=1,  # Finite horiz
                 verbose=1, quiet=False,
                 solution_startfrom=None,  # Default is no interim solution
                 BoroCnstArt=None,
                 **kwds):
        params = init_perfect_foresight.copy()
        params.update(kwds)
        kwds_all = params
        solution_terminal = deepcopy(self.solution_nobequest)
        if solution_startfrom:  # If user chose other terminal point, use that
            self.solution_startfrom = solution_startfrom
            solution_terminal = self.solution_terminal = solution_startfrom

        AgentType.__init__(
            self,
            solution_terminal,
            cycles=cycles,
            pseudo_terminal=False,
            ** kwds_all
        )

        # Add consumer-type-specific objects; deepcopy creates own versions
        self.time_vary = deepcopy(self.time_vary_)
        self.time_inv = deepcopy(self.time_inv_)

        # Params may have been passed by models that BUILD on PerfForesight
        self.shock_vars = deepcopy(self.shock_vars_)

        self.conditions = {}  # To track check_conditions

        # Extract the class name
        self.model_type = self.__class__.__name__

        # url that will locate the documentation
        self.url_doc = "https://hark.readthedocs.io/en/latest/search.html?q=" + \
            self.model_type+"&check_keywords=yes&area=default#"

        # paper that contains many results
        self.url_ref = "https://econ-ark.github.io/BufferStockTheory"

        # Setup and squirrel away the values initially used
        self.store_pre_iteration_starting_point()

        # Honor optional arguments (if any)
        self.verbose = verbose
        set_verbosity_level((4 - verbose) * 10)
        self.quiet = quiet

        # Construct one-period(/stage) solver (fix needed with staging mod)
        self.solve_one_period = \
            make_one_period_oo_solver(ConsPerfForesightSolver)

        # Store initial model params; later used to test if anything changed
        self.store_model_params(params['prmtv_par'], params['aprox_lim'])

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
            self.solve_penultimate_prd(verbose=0)

#        self.solution[-1].solver_check_condtnsnew_20210404(self.solution[-1], verbose=3)
        self.solution[-1].solver_check_condtnsnew_20210404(self, self.solution[-1], verbose=3)

    def pre_solve(self):  # Do anything necessary to prepare agent to solve
        if not self.BoroCnstArt:
            if hasattr(self, "MaxKinks"):
                if self.MaxKinks:  # True if MaxKinks is anything other than None
                    raise(
                        AttributeError(
                            "Kinks are caused by constraints.  Cannot specify MaxKinks without borrowing constraints!  Ignoring."
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
        new AgentType is created or when primitive or approximating parameters
        change (necessitating a new solution).

        Parameters
        ----------
        none

        Returns
        -------
        none
        """

        # Default income process is perf fore with perm = tran = 1.0
        setattr(self.solution_terminal, 'PermShkValsNxt', np.array([1.0]))
        setattr(self.solution_terminal, 'TranShkValsNxt', np.array([1.0]))
        # Update with actual args
        from HARK.core import get_solve_one_period_args
        solve_dict = get_solve_one_period_args(self, self.solve_one_period, stge_which=0)
        for key in solve_dict:
            setattr(self.solution_terminal, key, solve_dict[key])
            setattr(self.solution_terminal, key+'Nxt', solve_dict[key])
        self.solution_terminal.BoroCnstNat = self.solution_terminal.hNrmNow = 0.0

#        print('Eliminated update_solution_terminal')

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

    def get_Rfree(self):  # -> mcrlo_get_Rfree.
        # CDC: We should have a generic mcrlo_get_all_params
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

    def transition(self):  # -> mcrlo_trnsitn
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
        # should this be "Now", or "Prev"?!?
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

IncShkDstnfcts = {
    'about': 'Income Shock Distribution: .X[0] and .X[1] retrieve shocks, .pmf retrieves probabilities'}
IncShkDstnfcts.update({'py___code': r'construct_lognormal_income_process_unemployment'})
init_idiosyncratic_shocks['fcts'].update({'IncShkDstn': IncShkDstnfcts})
init_idiosyncratic_shocks.update({'IncShkDstnfcts:': IncShkDstnfcts})

PermShkStdfcts = {'about': 'Standard deviation for lognormal shock to permanent income'}
PermShkStdfcts.update({'latexexpr': '\PermShkStd'})
init_idiosyncratic_shocks['fcts'].update({'PermShkStd': PermShkStdfcts})
init_idiosyncratic_shocks.update({'PermShkStdfcts': PermShkStdfcts})

TranShkStdfcts = {'about': 'Standard deviation for lognormal shock to permanent income'}
TranShkStdfcts.update({'latexexpr': '\TranShkStd'})
init_idiosyncratic_shocks['fcts'].update({'TranShkStd': TranShkStdfcts})
init_idiosyncratic_shocks.update({'TranShkStdfcts': TranShkStdfcts})

UnempPrbfcts = {'about': 'Probability of unemployment while working'}
UnempPrbfcts.update({'latexexpr': r'\UnempPrb'})
UnempPrbfcts.update({'_unicode_': '℘'})
init_idiosyncratic_shocks['fcts'].update({'UnempPrb': UnempPrbfcts})
init_idiosyncratic_shocks.update({'UnempPrbfcts': UnempPrbfcts})

UnempPrbRetfcts = {'about': '"unemployment" in retirement = big medical shock'}
UnempPrbRetfcts.update({'latexexpr': r'\UnempPrbRet'})
init_idiosyncratic_shocks['fcts'].update({'UnempPrbRet': UnempPrbRetfcts})
init_idiosyncratic_shocks.update({'UnempPrbRetfcts': UnempPrbRetfcts})

IncUnempfcts = {'about': 'Unemployment insurance replacement rate'}
IncUnempfcts.update({'latexexpr': '\IncUnemp'})
IncUnempfcts.update({'_unicode_': 'μ'})
init_idiosyncratic_shocks['fcts'].update({'IncUnemp': IncUnempfcts})
init_idiosyncratic_shocks.update({'IncUnempfcts': IncUnempfcts})

IncUnempRetfcts = {'about': 'Size of medical shock (frac of perm inc)'}
init_idiosyncratic_shocks['fcts'].update({'IncUnempRet': IncUnempRetfcts})
init_idiosyncratic_shocks.update({'IncUnempRetfcts': IncUnempRetfcts})

tax_ratefcts = {'about': 'Flat income tax rate'}
tax_ratefcts.update({'about': 'Size of medical shock (frac of perm inc)'})
init_idiosyncratic_shocks['fcts'].update({'tax_rate': tax_ratefcts})
init_idiosyncratic_shocks.update({'tax_ratefcts': tax_ratefcts})

T_retirefcts = {'about': 'Period of retirement (0 --> no retirement)'}
init_idiosyncratic_shocks['fcts'].update({'T_retire': T_retirefcts})
init_idiosyncratic_shocks.update({'T_retirefcts': T_retirefcts})

PermShkCountfcts = {'about': 'Num of pts in discrete approx to permanent income shock dstn'}
init_idiosyncratic_shocks['fcts'].update({'PermShkCount': PermShkCountfcts})
init_idiosyncratic_shocks.update({'PermShkCountfcts': PermShkCountfcts})

TranShkCountfcts = {'about': 'Num of pts in discrete approx to transitory income shock dstn'}
init_idiosyncratic_shocks['fcts'].update({'TranShkCount': TranShkCountfcts})
init_idiosyncratic_shocks.update({'TranShkCountfcts': TranShkCountfcts})

aXtraMinfcts = {'about': 'Minimum end-of-period "assets above minimum" value'}
init_idiosyncratic_shocks['fcts'].update({'aXtraMin': aXtraMinfcts})
init_idiosyncratic_shocks.update({'aXtraMinfcts': aXtraMinfcts})

aXtraMaxfcts = {'about': 'Maximum end-of-period "assets above minimum" value'}
init_idiosyncratic_shocks['fcts'].update({'aXtraMax': aXtraMaxfcts})
init_idiosyncratic_shocks.update({'aXtraMaxfcts': aXtraMaxfcts})

aXtraNestFacfcts = {
    'about': 'Exponential nesting factor when constructing "assets above minimum" grid'}
init_idiosyncratic_shocks['fcts'].update({'aXtraNestFac': aXtraNestFacfcts})
init_idiosyncratic_shocks.update({'aXtraNestFacfcts': aXtraNestFacfcts})

aXtraCountfcts = {'about': 'Number of points in the grid of "assets above minimum"'}
init_idiosyncratic_shocks['fcts'].update({'aXtraMax': aXtraCountfcts})
init_idiosyncratic_shocks.update({'aXtraMaxfcts': aXtraCountfcts})

aXtraCountfcts = {'about': 'Number of points to include in grid of assets above minimum possible'}
init_idiosyncratic_shocks['fcts'].update({'aXtraCount': aXtraCountfcts})
init_idiosyncratic_shocks.update({'aXtraCountfcts': aXtraCountfcts})

aXtraExtrafcts = {
    'about': 'List of other values of "assets above minimum" to add to the grid (e.g., 10000)'}
init_idiosyncratic_shocks['fcts'].update({'aXtraExtra': aXtraExtrafcts})
init_idiosyncratic_shocks.update({'aXtraExtrafcts': aXtraExtrafcts})

aXtraGridfcts = {
    'about': 'Grid of values to add to minimum possible value to obtain actual end-of-period asset grid'}
init_idiosyncratic_shocks['fcts'].update({'aXtraGrid': aXtraGridfcts})
init_idiosyncratic_shocks.update({'aXtraGridfcts': aXtraGridfcts})

vFuncBoolfcts = {'about': 'Whether to calculate the value function during solution'}
init_idiosyncratic_shocks['fcts'].update({'vFuncBool': vFuncBoolfcts})
init_idiosyncratic_shocks.update({'vFuncBoolfcts': vFuncBoolfcts})

CubicBoolfcts={
    'about': 'Use cubic spline interpolation when True, linear interpolation when False'}
init_idiosyncratic_shocks['fcts'].update({'CubicBool': CubicBoolfcts})
init_idiosyncratic_shocks.update({'CubicBoolfcts': CubicBoolfcts})


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
        If False, creates a solution object populated by the solution for
        the final period.

    solution_startfrom : stge, optional
        A user-specified starting point for the iteration, to be used in place
        of the hardwired solution_terminal.  For example, you
        might set a loose tolerance to get a quick but rough solution, and
        if that works, set the tolerance lower and restart the solution from
        the interim solution.
    """

    # Time invariant parameters
    time_inv_=PerfForesightConsumerType.time_inv_ + [
        "vFuncBool",
        "CubicBool",
    ]
    time_inv_.remove(  # Unwanted item(s) inherited from PerfForesight
        "MaxKinks"  # PF inf hor with MaxKinks is equiv to fin hor with hor=MaxKinks
    )

    shock_vars_=['PermShk', 'TranShk']  # The unemployment shock is transitory

    def __init__(self, cycles=1, verbose=1,  quiet=True, solution_startfrom=None, **kwds):
        params=init_idiosyncratic_shocks.copy()

        # Update them with any customizations the user has chosen
        params.update(kwds)  # This gets all params, not just those in the dict

        # Inherit characteristics of a perfect foresight model initialized
        # with the same parameters
        PerfForesightConsumerType.__init__(
            self, cycles=cycles, verbose=verbose, quiet=quiet,
            solution_startfrom=solution_startfrom, **params
        )

        # Add the few parameters that are not in the initialization
        self.parameters.update({"cycles": self.cycles})
#        self.parameters.update({"time_inv_": time_inv_})
#        self.parameters.update({"time_vary_": time_vary_})
#        self.parameters.update({"shock_vars_": shock_vars_})

        # Extract the class name so that we can ...
        self.model_type=self.__class__.__name__

        # ... generate a url that will locate the documentation:
        self.url_doc_model_type="https://hark.readthedocs.io/en/latest/search.html?q=" + \
            self.model_type+"&check_keywords=yes&area=default#"

        # Define a reference to a paper that contains the main results
        self.url_ref="https://econ-ark.github.io/BufferStockTheory"

        # Add model_type and doc url to auto-generated self.parameters
        self.parameters.update({"model_type": self.model_type})
        self.parameters.update({"url_doc_model_type": self.url_doc_model_type})
        self.parameters.update({"url_ref": self.url_ref})

        # Add consumer-type specific objects, copying to create independent versions
        # - Default interpolation method is piecewise linear
        # - Cubic is smoother, works well if problem has no constraints
        # - User may or may not want to create the value function
        if (not self.CubicBool) and (not self.vFuncBool):
            solver=ConsIndShockSolverBasic
        else:  # Use the "advanced" solver if either is requested
            solver=ConsIndShockSolver

        # Construct the infrastructure needed to begin the solution process
        self.store_pre_iteration_starting_point()

        # Attach the corresponding one-stage solver to the agent
        self.solve_one_period=make_one_period_oo_solver(solver)

        self.update_solution_terminal()
        self.solution_terminal.url_ref="https://econ-ark.github.io/BufferStockTheory"
        self.solution_terminal.urlroot=self.solution_terminal.url_ref + \
            '/#'  # used for references to derivations

        # Store the initial model parameters so we can check for changes
        self.store_model_params(params['prmtv_par'], params['aprox_lim'])

        # Quiet mode: Define model without calculating anything
        # If not quiet, solve one period so we can check conditions
        if not quiet:
            self.solve_penultimate_prd(self)
            self.check_conditions(verbose)  # Check conditions for nature/existence of soln

    def solve_penultimate_prd(self, verbose):  # Build T-1 with lots of info
        self.update()
        self.tolerance_orig=deepcopy(self.tolerance)  # preserve true tolerance
        self.tolerance=float('inf')  # tolerance is infiniy ...
        self.solve(verbose)  # ... means that "solve" will stop after one period
        # restore original tolerance        self.solver_check_condtnsnew_20210404()  # Check conditions for nature/existence of soln
        self.tolerance=self.tolerance_orig

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
        (IncShkDstn,
            PermShkDstn,
            TranShkDstn,
         )=self.construct_lognormal_income_process_unemployment()
        self.IncShkDstn=IncShkDstn
        self.PermShkDstn=PermShkDstn
        self.TranShkDstn=TranShkDstn
        self.add_to_time_vary("IncShkDstn", "PermShkDstn", "TranShkDstn")
        self.parameters.update

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

    def store_pre_iteration_starting_point(self):
        """
        Construct the income process, the assets grid, and the terminal solution.

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

        self.update_income_process()
        self.update_assets_grid()

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

        solve_par_vals_now = {}
        for par in self.solve_par_vals:
            solve_par_vals_now[par] = getattr(self, par)

        if not solve_par_vals_now == self.solve_par_vals:
            self.store_pre_iteration_starting_point()

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
        PermShkValsNxt = self.IncShkDstn[0][1]
        TranShkValsNxt = self.IncShkDstn[0][2]
        ShkPrbsNxt = self.IncShkDstn[0][0]
        Ex_IncNextNrm = np.dot(ShkPrbsNxt, PermShkValsNxt * TranShkValsNxt)
        PermShkMinNext = np.min(PermShkValsNxt)
        TranShkMinNext = np.min(TranShkValsNxt)
        WorstIncNext = PermShkMinNext * TranShkMinNext
        WorstIncPrb = np.sum(
            ShkPrbsNxt[(PermShkValsNxt * TranShkValsNxt) == WorstIncNext]
        )
        PermGro = self.PermGroFac[0]  # AgentType gets list of growth rates
        LivNxt = self.LivPrb[0]  # and survival rates

        # Calculate human wealth and the infinite horizon natural borrowing constraint
        hNrmNow = (Ex_IncNextNrm * PermGro / self.Rfree) / (
            1.0 - PermGro / self.Rfree
        )
        temp = PermGro * PermShkMinNext / self.Rfree
        BoroCnstNat = -TranShkMinNext * temp / (1.0 - temp)

        RPF = (self.DiscFac * LivNxt * self.Rfree) ** (
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
        PermShkValsNxt_tiled = (np.tile(IncShkDstn[1], (aCount, 1))).transpose()
        TranShkVals_tiled = (np.tile(IncShkDstn[2], (aCount, 1))).transpose()
        ShkPrbs_tiled = (np.tile(IncShkDstn[0], (aCount, 1))).transpose()

        # Calculate marginal value next period for each gridpoint and each shock
        mNextArray = (
            self.Rfree / (self.PermGroFac[0] * PermShkValsNxt_tiled) * aNowGrid_tiled
            + TranShkVals_tiled
        )
        vPnextArray = vPfuncNext(mNextArray)

        # Calculate expected marginal value and implied optimal consumption
        ExvPnextGrid = (
            self.DiscFac
            * self.Rfree
            * self.LivPrb[0]
            * self.PermGroFac[0] ** (-self.CRRA)
            * np.sum(
                PermShkValsNxt_tiled ** (-self.CRRA) * vPnextArray * ShkPrbs_tiled, axis=0
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
        self.update()

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
        parameters = {
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
                PermShkValsNxtRet = np.array(
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
                PermShkValsNxtRet = np.array([1.0])
                TranShkValsRet = np.array([1.0])
                ShkPrbsRet = np.array([1.0])
                IncShkDstnRet = DiscreteApproximationToContinuousDistribution(
                    ShkPrbsRet,
                    [PermShkValsNxtRet, TranShkValsRet],
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

        IncShkDstn[-1].parameters = parameters
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
        PermShkValsNxt = self.IncShkDstn[0][1]
        TranShkValsNxt = self.IncShkDstn[0][2]
        ShkPrbsNxt = self.IncShkDstn[0][0]
        Ex_IncNextNrm = calc_expectation(
            self.IncShkDstn,
            lambda trans, perm: trans * perm
        )
        PermShkMinNext = np.min(PermShkValsNxt)
        TranShkMinNext = np.min(TranShkValsNxt)
        WorstIncNext = PermShkMinNext * TranShkMinNext
        WorstIncPrb = np.sum(
            ShkPrbsNxt[(PermShkValsNxt * TranShkValsNxt) == WorstIncNext]
        )

        # Calculate human wealth and the infinite horizon natural borrowing constraint
        hNrmNow = (Ex_IncNextNrm * self.PermGroFac[0] / self.Rsave) / (
            1.0 - self.PermGroFac[0] / self.Rsave
        )
        temp = self.PermGroFac[0] * PermShkMinNext / self.Rboro
        BoroCnstNat = -TranShkMinNext * temp / (1.0 - temp)

        RPFTop = (self.DiscFac * self.LivPrb * self.Rsave) ** (
            1.0 / self.CRRA
        ) / self.Rsave
        RPFBot = (self.DiscFac * self.LivPrb * self.Rboro) ** (
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
        self.MPCmaxNow = MPCmaxNow
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
