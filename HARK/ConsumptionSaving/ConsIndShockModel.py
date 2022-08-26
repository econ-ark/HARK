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
from copy import copy, deepcopy

import numpy as np
from HARK import (
    AgentType,
    MetricObject,
    NullFunc,
    _log,
    make_one_period_oo_solver,
    set_verbosity_level,
)
from HARK.Calibration.Income.IncomeTools import (
    Cagetti_income,
    parse_income_spec,
    parse_time_params,
)
from HARK.datasets.life_tables.us_ssa.SSATools import parse_ssa_life_table
from HARK.datasets.SCF.WealthIncomeDist.SCFDistTools import income_wealth_dists_from_scf
from HARK.distribution import (
    DiscreteDistribution,
    IndexDistribution,
    Lognormal,
    MeanOneLogNormal,
    Uniform,
    add_discrete_outcome_constant_mean,
    combine_indep_dstns,
    expected,
)
from HARK.interpolation import CubicHermiteInterp as CubicInterp
from HARK.interpolation import (
    CubicInterp,
    LinearInterp,
    LowerEnvelope,
    MargMargValueFuncCRRA,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.utilities import (
    CRRAutility,
    CRRAutility_inv,
    CRRAutility_invP,
    CRRAutilityP,
    CRRAutilityP_inv,
    CRRAutilityP_invP,
    CRRAutilityPP,
    UtilityFuncCRRA,
    make_grid_exp_mult,
)
from scipy.optimize import newton

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
    ):
        # Change any missing function inputs to NullFunc
        self.cFunc = cFunc if cFunc is not None else NullFunc()
        self.vFunc = vFunc if vFunc is not None else NullFunc()
        self.vPfunc = vPfunc if vPfunc is not None else NullFunc()
        # vPFunc = NullFunc() if vPfunc is None else vPfunc
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


# =====================================================================
# === Classes and functions that solve consumption-saving models ===
# =====================================================================


class ConsPerfForesightSolver(MetricObject):
    """
    A class for solving a one period perfect foresight
    consumption-saving problem.
    An instance of this class is created by the function solvePerfForesight
    in each period.

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
        self.solution_next = solution_next
        self.DiscFac = DiscFac
        self.LivPrb = LivPrb
        self.CRRA = CRRA
        self.Rfree = Rfree
        self.PermGroFac = PermGroFac
        self.BoroCnstArt = BoroCnstArt
        self.MaxKinks = MaxKinks

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
        self.u = UtilityFuncCRRA(self.CRRA)

    def def_value_funcs(self):
        """
        Defines the value and marginal value functions for this period.
        Uses the fact that for a perfect foresight CRRA utility problem,
        if the MPC in period t is :math:`\kappa_{t}`, and relative risk
        aversion :math:`\rho`, then the inverse value vFuncNvrs has a
        constant slope of :math:`\kappa_{t}^{-\rho/(1-\rho)}` and
        vFuncNvrs has value of zero at the lower bound of market resources
        mNrmMin.  See PerfForesightConsumerType.ipynb documentation notebook
        for a brief explanation and the links below for a fuller treatment.

        https://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/consumption/PerfForesightCRRA/#vFuncAnalytical
        https://www.econ2.jhu.edu/people/ccarroll/SolvingMicroDSOPs/#vFuncPF

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # See the PerfForesightConsumerType.ipynb documentation notebook for the derivations
        vFuncNvrsSlope = self.MPCmin ** (-self.CRRA / (1.0 - self.CRRA))
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
        # Use a local value of BoroCnstArt to prevent comparing None and float below.
        if self.BoroCnstArt is None:
            BoroCnstArt = -np.inf
        else:
            BoroCnstArt = self.BoroCnstArt

        # Calculate human wealth this period
        self.hNrmNow = (self.PermGroFac / self.Rfree) * (self.solution_next.hNrm + 1.0)

        # Calculate the lower bound of the marginal propensity to consume
        PatFac = ((self.Rfree * self.DiscFacEff) ** (1.0 / self.CRRA)) / self.Rfree
        self.MPCmin = 1.0 / (1.0 + PatFac / self.solution_next.MPCmin)

        # Extract the discrete kink points in next period's consumption function;
        # don't take the last one, as it only defines the extrapolation and is not a kink.
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
        cNrmNow = np.append(cNrmNow, cNrmNow[-1] + self.MPCmin)

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
                mNrmNow = np.concatenate(([BoroCnstArt, mCrit], mNrmNow[(idx + 1) :]))
                cNrmNow = np.concatenate(([0.0, cCrit], cNrmNow[(idx + 1) :]))

            else:
                # If it *is* the very last index, then there are only three points
                # that characterize the consumption function: the artificial borrowing
                # constraint, the constraint kink, and the extrapolation point.
                mXtra = (cNrmNow[-1] - cNrmCnst[-1]) / (1.0 - self.MPCmin)
                mCrit = mNrmNow[-1] + mXtra
                cCrit = mCrit - BoroCnstArt
                mNrmNow = np.array([BoroCnstArt, mCrit, mCrit + 1.0])
                cNrmNow = np.array([0.0, cCrit, cCrit + self.MPCmin])

        # If the mNrm and cNrm grids have become too large, throw out the last
        # kink point, being sure to adjust the extrapolation.
        if mNrmNow.size > self.MaxKinks:
            mNrmNow = np.concatenate((mNrmNow[:-2], [mNrmNow[-3] + 1.0]))
            cNrmNow = np.concatenate((cNrmNow[:-2], [cNrmNow[-3] + self.MPCmin]))

        # Construct the consumption function as a linear interpolation.
        self.cFunc = LinearInterp(mNrmNow, cNrmNow)

        # Calculate the upper bound of the MPC as the slope of the bottom segment.
        self.MPCmax = (cNrmNow[1] - cNrmNow[0]) / (mNrmNow[1] - mNrmNow[0])

        # Add two attributes to enable calculation of steady state market resources.
        self.Ex_IncNext = 1.0  # Perfect foresight income of 1
        # Relabeling for compatibility with add_mNrmStE
        self.mNrmMinNow = mNrmNow[0]

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

        # If no uncertainty, return the degenerate targets for the PF model
        if hasattr(self, "TranShkMinNext"):  # Then it has transitory shocks
            # Handle the degenerate case where shocks are of size zero
            if (self.TranShkMinNext == 1.0) and (self.PermShkMinNext == 1.0):
                # but they are of zero size (and also permanent are zero)
                if self.GICRaw:  # max of nat and art boro cnst
                    if type(self.BoroCnstArt) == type(None):
                        solution.mNrmStE = -self.hNrmNow
                        solution.mNrmTrg = -self.hNrmNow
                    else:
                        bNrmNxt = -self.BoroCnstArt * self.Rfree / self.PermGroFac
                        solution.mNrmStE = bNrmNxt + 1.0
                        solution.mNrmTrg = bNrmNxt + 1.0
                else:  # infinity
                    solution.mNrmStE = float("inf")
                    solution.mNrmTrg = float("inf")
                return solution

        # First find
        # \bar{\mathcal{R}} = E_t[R/Gamma_{t+1}] = R/Gamma E_t[1/psi_{t+1}]
        if type(self) == ConsPerfForesightSolver:
            Ex_PermShkInv = 1.0
        else:
            Ex_PermShkInv = np.dot(1 / self.PermShkValsNext, self.ShkPrbsNext)

        Ex_RNrmFac = (self.Rfree / self.PermGroFac) * Ex_PermShkInv

        # mNrmTrg solves Rcalbar*(m - c(m)) + E[inc_next] = m. Define a
        # rearranged version.
        Ex_m_tp1_minus_m_t = (
            lambda m: Ex_RNrmFac * (m - solution.cFunc(m)) + self.Ex_IncNext - m
        )

        # Minimum market resources plus next income is okay starting guess
        m_init_guess = self.mNrmMinNow + self.Ex_IncNext
        try:
            mNrmTrg = newton(Ex_m_tp1_minus_m_t, m_init_guess)
        except:
            mNrmTrg = None

        # Add mNrmTrg to the solution and return it
        solution.mNrmTrg = mNrmTrg
        return solution

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

        PF_RNrm = self.Rfree / self.PermGroFac
        # If we are working with a model that permits uncertainty but that
        # uncertainty has been set to zero, return the correct answer
        # by hand because in this degenerate case numerical search may
        # have trouble
        if hasattr(self, "TranShkMinNext"):  # Then it has transitory shocks
            if (self.TranShkMinNext == 1.0) and (self.PermShkMinNext == 1.0):
                # but they are of zero size (and permanent shocks also not there)
                if self.GICRaw:  # max of nat and art boro cnst
                    #                    breakpoint()
                    if type(self.BoroCnstArt) == type(None):
                        solution.mNrmStE = -self.hNrmNow
                        solution.mNrmTrg = -self.hNrmNow
                    else:
                        bNrmNxt = -self.BoroCnstArt * self.Rfree / self.PermGroFac
                        solution.mNrmStE = bNrmNxt + 1.0
                        solution.mNrmTrg = bNrmNxt + 1.0
                else:  # infinity
                    solution.mNrmStE = float("inf")
                    solution.mNrmTrg = float("inf")
                return solution

        Ex_PermShk_tp1_times_m_tp1_minus_m_t = (
            lambda mStE: PF_RNrm * (mStE - solution.cFunc(mStE)) + 1.0 - mStE
        )

        # Minimum market resources plus next income is okay starting guess
        m_init_guess = self.mNrmMinNow + self.Ex_IncNext
        try:
            mNrmStE = newton(Ex_PermShk_tp1_times_m_tp1_minus_m_t, m_init_guess)
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
            Same solution that was provided, augmented with attributes mNrmStE and
            mNrmTrg, if they exist.

        """

        # 0. There is no non-degenerate steady state for any unconstrained PF model.
        # 1. There is a non-degenerate SS for constrained PF model if GICRaw holds.
        # Therefore
        # Check if  (GICRaw and BoroCnstArt) and if so compute them both
        thorn = (self.Rfree * self.DiscFacEff) ** (1 / self.CRRA)
        GICRaw = 1 > thorn / self.PermGroFac
        if self.BoroCnstArt is not None and GICRaw:
            solution = self.add_mNrmStE(solution)
            solution = self.add_mNrmTrg(solution)
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

        solution = ConsumerSolution(
            cFunc=self.cFunc,
            vFunc=self.vFunc,
            vPfunc=self.vPfunc,
            mNrmMin=self.mNrmMinNow,
            hNrm=self.hNrmNow,
            MPCmin=self.MPCmin,
            MPCmax=self.MPCmax,
        )

        solution = self.add_stable_points(solution)

        return solution


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
        """
        Constructor for a new solver-setup for problems with income subject to
        permanent and transitory shocks.
        """
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

        self.def_utility_funcs()

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
            A DiscreteDistribution with a pmv
            and two point value arrays in atoms, order:
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
        self.ShkPrbsNext = IncShkDstn.pmv
        self.PermShkValsNext = IncShkDstn.atoms[0]
        self.TranShkValsNext = IncShkDstn.atoms[1]
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
        self.PatFac = ((self.Rfree * self.DiscFacEff) ** (1.0 / self.CRRA)) / self.Rfree
        self.MPCminNow = 1.0 / (1.0 + self.PatFac / solution_next.MPCmin)
        self.Ex_IncNext = np.dot(
            self.ShkPrbsNext, self.TranShkValsNext * self.PermShkValsNext
        )
        self.hNrmNow = (
            self.PermGroFac / self.Rfree * (self.Ex_IncNext + solution_next.hNrm)
        )
        self.MPCmaxNow = 1.0 / (
            1.0
            + (self.WorstIncPrb ** (1.0 / self.CRRA))
            * self.PatFac
            / solution_next.MPCmax
        )

        self.cFuncLimitIntercept = self.MPCminNow * self.hNrmNow
        self.cFuncLimitSlope = self.MPCminNow

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

        # Note: need to be sure to handle BoroCnstArt==None appropriately.
        # In Py2, this would evaluate to 5.0:  np.max([None, 5.0]).
        # However in Py3, this raises a TypeError. Thus here we need to directly
        # address the situation in which BoroCnstArt == None:
        if BoroCnstArt is None:
            self.mNrmMinNow = self.BoroCnstNat
        else:
            self.mNrmMinNow = np.max([self.BoroCnstNat, BoroCnstArt])
        if self.BoroCnstNat < self.mNrmMinNow:
            self.MPCmaxEff = 1.0  # If actually constrained, MPC near limit is 1
        else:
            self.MPCmaxEff = self.MPCmaxNow

        # Define the borrowing constraint (limiting consumption function)
        self.cFuncNowCnst = LinearInterp(
            np.array([self.mNrmMinNow, self.mNrmMinNow + 1]), np.array([0.0, 1.0])
        )

    def prepare_to_solve(self):
        """
        Perform preparatory work before calculating the unconstrained consumption
        function.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        self.set_and_update_values(
            self.solution_next, self.IncShkDstn, self.LivPrb, self.DiscFac
        )
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
        of market resources that the agent could have next period, considering
        the grid of end-of-period assets and the distribution of shocks he might
        experience next period.

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
        # straint) uconstrained consumption function, and the artificially con-
        # strained consumption function.
        self.aNrmNow = np.asarray(self.aXtraGrid) + self.BoroCnstNat

        return self.aNrmNow

    def m_nrm_next(self, shocks, a_nrm, Rfree):
        """
        Computes normalized market resources of the next period
        from income shocks and current normalized market resources.

        Parameters
        ----------
        shocks: [float]
            Permanent and transitory income shock levels.
        a_nrm: float
            Normalized market assets this period

        Returns
        -------
        float
           normalized market resources in the next period
        """
        return Rfree / (self.PermGroFac * shocks[0]) * a_nrm + shocks[1]

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

        def vp_next(shocks, a_nrm, Rfree):
            return shocks[0] ** (-self.CRRA) * self.vPfuncNext(
                self.m_nrm_next(shocks, a_nrm, Rfree)
            )

        EndOfPrdvP = (
            self.DiscFacEff
            * self.Rfree
            * self.PermGroFac ** (-self.CRRA)
            * expected(vp_next, self.IncShkDstn, args=(self.aNrmNow, self.Rfree))
        )

        return EndOfPrdvP

    def get_points_for_interpolation(self, EndOfPrdvP, aNrmNow):
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
        cNrmNow = self.u.inv(EndOfPrdvP, order=(1, 0))
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

    def make_basic_solution(self, EndOfPrdvP, aNrm, interpolator):
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
        cNrm, mNrm = self.get_points_for_interpolation(EndOfPrdvP, aNrm)
        solution_now = self.use_points_for_interpolation(cNrm, mNrm, interpolator)

        return solution_now

    def add_MPC_and_human_wealth(self, solution):
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
        solution.hNrm = self.hNrmNow
        solution.MPCmin = self.MPCminNow
        solution.MPCmax = self.MPCmaxEff
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

        # 0. Check if GICRaw holds. If so, then mNrmStE will exist. So, compute it.
        # 1. Check if GICNrm holds. If so, then mNrmTrg will exist. So, compute it.

        thorn = (self.Rfree * self.DiscFacEff) ** (1 / self.CRRA)

        GPFRaw = thorn / self.PermGroFac
        self.GPFRaw = GPFRaw
        GPFNrm = (
            thorn / self.PermGroFac / np.dot(1 / self.PermShkValsNext, self.ShkPrbsNext)
        )
        self.GPFNrm = GPFNrm
        GICRaw = 1 > thorn / self.PermGroFac
        self.GICRaw = GICRaw
        GICNrm = 1 > GPFNrm
        self.GICNrm = GICNrm

        if GICRaw:
            # find steady state m, if it exists
            solution = self.add_mNrmStE(solution)
        if GICNrm:
            # find target m, if it exists
            solution = self.add_mNrmTrg(solution)

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
        cFuncUnc : LinearInterp
            The unconstrained consumption function for this period.
        """
        cFuncUnc = LinearInterp(
            mNrm, cNrm, self.cFuncLimitIntercept, self.cFuncLimitSlope
        )
        return cFuncUnc

    def solve(self):
        """
        Solves a one period consumption saving problem with risky income.

        Parameters
        ----------
        None

        Returns
        -------
        solution : ConsumerSolution
            The solution to the one period problem.
        """
        aNrmNow = self.prepare_to_calc_EndOfPrdvP()
        EndOfPrdvP = self.calc_EndOfPrdvP()
        solution = self.make_basic_solution(EndOfPrdvP, aNrmNow, self.make_linear_cFunc)
        solution = self.add_MPC_and_human_wealth(solution)
        solution = self.add_stable_points(solution)

        return solution


###############################################################################
###############################################################################


class ConsIndShockSolver(ConsIndShockSolverBasic):
    """
    This class solves a single period of a standard consumption-saving problem.
    It inherits from ConsIndShockSolverBasic, adding the ability to perform cubic
    interpolation and to calculate the value function.
    """

    def make_cubic_cFunc(self, mNrm, cNrm):
        """
        Makes a cubic spline interpolation of the unconstrained consumption
        function for this period.

        Parameters
        ----------
        mNrm : np.array
            Corresponding market resource points for interpolation.
        cNrm : np.array
            Consumption points for interpolation.

        Returns
        -------
        cFuncUnc : CubicInterp
            The unconstrained consumption function for this period.
        """

        def vpp_next(shocks, a_nrm, Rfree):
            return shocks[0] ** (-self.CRRA - 1.0) * self.vPPfuncNext(
                self.m_nrm_next(shocks, a_nrm, Rfree)
            )

        EndOfPrdvPP = (
            self.DiscFacEff
            * self.Rfree
            * self.Rfree
            * self.PermGroFac ** (-self.CRRA - 1.0)
            * expected(vpp_next, self.IncShkDstn, args=(self.aNrmNow, self.Rfree))
        )
        dcda = EndOfPrdvPP / self.u.der(np.array(cNrm[1:]), order=2)
        MPC = dcda / (dcda + 1.0)
        MPC = np.insert(MPC, 0, self.MPCmaxNow)

        cFuncNowUnc = CubicInterp(
            mNrm, cNrm, MPC, self.MPCminNow * self.hNrmNow, self.MPCminNow
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

        def v_lvl_next(shocks, a_nrm, Rfree):
            return (
                shocks[0] ** (1.0 - self.CRRA) * self.PermGroFac ** (1.0 - self.CRRA)
            ) * self.vFuncNext(self.m_nrm_next(shocks, a_nrm, Rfree))

        EndOfPrdv = self.DiscFacEff * expected(
            v_lvl_next, self.IncShkDstn, args=(self.aNrmNow, self.Rfree)
        )
        EndOfPrdvNvrs = self.u.inv(
            EndOfPrdv
        )  # value transformed through inverse utility
        EndOfPrdvNvrsP = EndOfPrdvP * self.u.inv(EndOfPrdv, order=(0, 1))
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
        vPnow = self.u.der(cNrmNow)

        # Construct the beginning-of-period value function
        vNvrs = self.u.inv(vNrmNow)  # value transformed through inverse utility
        vNvrsP = vPnow * self.u.inv(vNrmNow, order=(0, 1))
        mNrm_temp = np.insert(mNrm_temp, 0, self.mNrmMinNow)
        vNvrs = np.insert(vNvrs, 0, 0.0)
        vNvrsP = np.insert(
            vNvrsP, 0, self.MPCmaxEff ** (-self.CRRA / (1.0 - self.CRRA))
        )
        MPCminNvrs = self.MPCminNow ** (-self.CRRA / (1.0 - self.CRRA))
        vNvrsFuncNow = CubicInterp(
            mNrm_temp, vNvrs, vNvrsP, MPCminNvrs * self.hNrmNow, MPCminNvrs
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

    def solve(self):
        """
        Solves the single period consumption-saving problem using the method of
        endogenous gridpoints.  Solution includes a consumption function cFunc
        (using cubic or linear splines), a marginal value function vPfunc, a min-
        imum acceptable level of normalized market resources mNrmMin, normalized
        human wealth hNrm, and bounding MPCs MPCmin and MPCmax.  It might also
        have a value function vFunc and marginal marginal value function vPPfunc.

        Parameters
        ----------
        none

        Returns
        -------
        solution : ConsumerSolution
            The solution to the single period consumption-saving problem.
        """
        # Make arrays of end-of-period assets and end-of-period marginal value
        aNrm = self.prepare_to_calc_EndOfPrdvP()
        EndOfPrdvP = self.calc_EndOfPrdvP()

        # Construct a basic solution for this period
        if self.CubicBool:
            solution = self.make_basic_solution(
                EndOfPrdvP, aNrm, interpolator=self.make_cubic_cFunc
            )
        else:
            solution = self.make_basic_solution(
                EndOfPrdvP, aNrm, interpolator=self.make_linear_cFunc
            )

        solution = self.add_MPC_and_human_wealth(solution)  # add a few things
        solution = self.add_stable_points(solution)

        # Add the value function if requested, as well as the marginal marginal
        # value function if cubic splines were used (to prepare for next period)
        if self.vFuncBool:
            solution = self.add_vFunc(solution, EndOfPrdvP)
        if self.CubicBool:
            solution = self.add_vPPfunc(solution)
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

    def make_cubic_cFunc(self, mNrm, cNrm):
        """
        Makes a cubic spline interpolation that contains the kink of the unconstrained
        consumption function for this period.

        Parameters
        ----------
        mNrm : np.array
            Corresponding market resource points for interpolation.
        cNrm : np.array
            Consumption points for interpolation.

        Returns
        -------
        cFuncUnc : CubicInterp
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
            A 1D array of end-of-period assets; also stored as attribute of self.
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
            Rfree_vec[0 : self.i_kink] = self.Rboro
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
            PatFacTop = (
                (self.Rsave * self.DiscFacEff) ** (1.0 / self.CRRA)
            ) / self.Rsave
            self.MPCminNow = 1.0 / (1.0 + PatFacTop / self.solution_next.MPCmin)
            self.hNrmNow = (
                self.PermGroFac
                / self.Rsave
                * (
                    np.dot(
                        self.ShkPrbsNext, self.TranShkValsNext * self.PermShkValsNext
                    )
                    + self.solution_next.hNrm
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
    "cycles": 1,  # Finite, non-cyclic model
    "CRRA": 2.0,  # Coefficient of relative risk aversion,
    "Rfree": 1.03,  # Interest factor on assets
    "DiscFac": 0.96,  # Intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability
    "PermGroFac": [1.01],  # Permanent income growth factor
    "BoroCnstArt": None,  # Artificial borrowing constraint
    "MaxKinks": 400,  # Maximum number of grid points to allow in cFunc (should be large)
    "AgentCount": 10000,  # Number of agents of this type (only matters for simulation)
    "aNrmInitMean": 0.0,  # Mean of log initial assets (only matters for simulation)
    "aNrmInitStd": 1.0,  # Standard deviation of log initial assets (only for simulation)
    "pLvlInitMean": 0.0,  # Mean of log initial permanent income (only matters for simulation)
    # Standard deviation of log initial permanent income (only matters for simulation)
    "pLvlInitStd": 0.0,
    # Aggregate permanent income growth factor: portion of PermGroFac attributable to aggregate productivity growth (only matters for simulation)
    "PermGroFacAgg": 1.0,
    "T_age": None,  # Age after which simulated agents are automatically killed
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "PerfMITShk": False,
    # Do Perfect Foresight MIT Shock: Forces Newborns to follow solution path of the agent he/she replaced when True
}


class PerfForesightConsumerType(AgentType):
    """
    A perfect foresight consumer type who has no uncertainty other than mortality.
    His problem is defined by a coefficient of relative risk aversion, intertemporal
    discount factor, interest factor, an artificial borrowing constraint (maybe)
    and time sequences of the permanent income growth rate and survival probability.

    Parameters
    ----------

    """

    # Define some universal values for all consumer types
    cFunc_terminal_ = LinearInterp([0.0, 1.0], [0.0, 1.0])  # c=m in terminal period
    vFunc_terminal_ = LinearInterp([0.0, 1.0], [0.0, 0.0])  # This is overwritten
    solution_terminal_ = ConsumerSolution(
        cFunc=cFunc_terminal_,
        vFunc=vFunc_terminal_,
        mNrmMin=0.0,
        hNrm=0.0,
        MPCmin=1.0,
        MPCmax=1.0,
    )
    time_vary_ = ["LivPrb", "PermGroFac"]
    time_inv_ = ["CRRA", "DiscFac", "MaxKinks", "BoroCnstArt"]
    state_vars = ["pLvl", "PlvlAgg", "bNrm", "mNrm", "aNrm", "aLvl"]
    shock_vars_ = []

    def __init__(self, verbose=1, quiet=False, **kwds):
        params = init_perfect_foresight.copy()
        params.update(kwds)
        kwds = params

        # Initialize a basic AgentType
        AgentType.__init__(
            self,
            solution_terminal=deepcopy(self.solution_terminal_),
            pseudo_terminal=False,
            **kwds
        )

        # Add consumer-type specific objects, copying to create independent versions
        self.time_vary = deepcopy(self.time_vary_)
        self.time_inv = deepcopy(self.time_inv_)

        self.shock_vars = deepcopy(self.shock_vars_)
        self.verbose = verbose
        self.quiet = quiet
        self.solve_one_period = make_one_period_oo_solver(ConsPerfForesightSolver)
        set_verbosity_level((4 - verbose) * 10)

        self.update_Rfree()  # update interest rate if time varying

    def pre_solve(self):
        self.update_solution_terminal()  # Solve the terminal period problem

        # Fill in BoroCnstArt and MaxKinks if they're not specified or are irrelevant.
        # If no borrowing constraint specified...
        if not hasattr(self, "BoroCnstArt"):
            self.BoroCnstArt = None  # ...assume the user wanted none

        if not hasattr(self, "MaxKinks"):
            if self.cycles > 0:  # If it's not an infinite horizon model...
                self.MaxKinks = np.inf  # ...there's no need to set MaxKinks
            elif self.BoroCnstArt is None:  # If there's no borrowing constraint...
                self.MaxKinks = np.inf  # ...there's no need to set MaxKinks
            else:
                raise (
                    AttributeError(
                        "PerfForesightConsumerType requires the attribute MaxKinks to be specified when BoroCnstArt is not None and cycles == 0."
                    )
                )

    def check_restrictions(self):
        """
        A method to check that various restrictions are met for the model class.
        """
        if self.DiscFac < 0:
            raise Exception("DiscFac is below zero with value: " + str(self.DiscFac))

        return

    def update_solution_terminal(self):
        """
        Update the terminal period solution.  This method should be run when a
        new AgentType is created or when CRRA changes.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        self.solution_terminal.vFunc = ValueFuncCRRA(self.cFunc_terminal_, self.CRRA)
        self.solution_terminal.vPfunc = MargValueFuncCRRA(
            self.cFunc_terminal_, self.CRRA
        )
        self.solution_terminal.vPPfunc = MargMargValueFuncCRRA(
            self.cFunc_terminal_, self.CRRA
        )

    def update_Rfree(self):
        """
        Determines whether Rfree is time-varying or fixed.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        if isinstance(self.Rfree, list):
            if len(self.Rfree) == self.T_cycle:
                self.add_to_time_vary("Rfree")
            else:
                raise AttributeError(
                    "If Rfree is time-varying, it should have a length of T_cycle!"
                )
        elif isinstance(self.Rfree, (int, float)):
            self.add_to_time_inv("Rfree")
        else:  # temporary fix for MarkovConsumerType
            self.add_to_time_inv("Rfree")

    def unpack_cFunc(self):
        """DEPRECATED: Use solution.unpack('cFunc') instead.
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
        self.PermShkAggNow = self.PermGroFacAgg  # This never changes during simulation
        self.state_now["PlvlAgg"] = 1.0
        AgentType.initialize_sim(self)

    def sim_birth(self, which_agents):
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
        self.state_now["aNrm"][which_agents] = Lognormal(
            mu=self.aNrmInitMean,
            sigma=self.aNrmInitStd,
            seed=self.RNG.randint(0, 2**31 - 1),
        ).draw(N)
        # why is a now variable set here? Because it's an aggregate.
        pLvlInitMeanNow = self.pLvlInitMean + np.log(
            self.state_now["PlvlAgg"]
        )  # Account for newer cohorts having higher permanent income
        self.state_now["pLvl"][which_agents] = Lognormal(
            pLvlInitMeanNow, self.pLvlInitStd, seed=self.RNG.randint(0, 2**31 - 1)
        ).draw(N)
        # How many periods since each agent was born
        self.t_age[which_agents] = 0

        if not hasattr(
            self, "PerfMITShk"
        ):  # If PerfMITShk not specified, let it be False
            self.PerfMITShk = False
        if (
            self.PerfMITShk == False
        ):  # If True, Newborns inherit t_cycle of agent they replaced (i.e. t_cycles are not reset).
            self.t_cycle[
                which_agents
            ] = 0  # Which period of the cycle each agent is currently in

        return None

    def sim_death(self):
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
            self.t_cycle - 1 if self.cycles == 1 else self.t_cycle
        ]  # Time has already advanced, so look back one

        # In finite-horizon problems the previous line gives newborns the
        # survival probability of the last non-terminal period. This is okay,
        # however, since they will be instantly replaced by new newborns if
        # they die.
        # See: https://github.com/econ-ark/HARK/pull/981

        DeathShks = Uniform(seed=self.RNG.randint(0, 2**31 - 1)).draw(
            N=self.AgentCount
        )
        which_agents = DeathShks < DiePrb
        if self.T_age is not None:  # Kill agents that have lived for too many periods
            too_old = self.t_age >= self.T_age
            which_agents = np.logical_or(which_agents, too_old)
        return which_agents

    def get_shocks(self):
        """
        Finds permanent and transitory income "shocks" for each agent this period.  As this is a
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
        self.shocks["PermShk"] = PermGroFac[
            self.t_cycle - 1
        ]  # cycle time has already been advanced
        self.shocks["TranShk"] = np.ones(self.AgentCount)

    def get_Rfree(self):
        """
        Returns an array of size self.AgentCount with self.Rfree in every entry.

        Parameters
        ----------
        None

        Returns
        -------
        RfreeNow : np.array
             Array of size self.AgentCount with risk free interest rate for each agent.
        """
        RfreeNow = np.ones(self.AgentCount)
        if "Rfree" in self.time_inv:
            RfreeNow = RfreeNow * self.Rfree
        elif "Rfree" in self.time_vary:
            for t in range(self.T_cycle):
                these = t == self.t_cycle
                RfreeNow[these] = self.Rfree[t]
        return RfreeNow

    def transition(self):
        pLvlPrev = self.state_prev["pLvl"]
        aNrmPrev = self.state_prev["aNrm"]
        RfreeNow = self.get_Rfree()

        # Calculate new states: normalized market resources and permanent income level
        # Updated permanent income level
        pLvlNow = pLvlPrev * self.shocks["PermShk"]
        # Updated aggregate permanent productivity level
        PlvlAggNow = self.state_prev["PlvlAgg"] * self.PermShkAggNow
        # "Effective" interest factor on normalized assets
        ReffNow = RfreeNow / self.shocks["PermShk"]
        bNrmNow = ReffNow * aNrmPrev  # Bank balances before labor income
        # Market resources after income
        mNrmNow = bNrmNow + self.shocks["TranShk"]

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
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        MPCnow = np.zeros(self.AgentCount) + np.nan
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            cNrmNow[these], MPCnow[these] = self.solution[t].cFunc.eval_with_derivative(
                self.state_now["mNrm"][these]
            )
        self.controls["cNrm"] = cNrmNow

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
        self.state_now["aNrm"] = self.state_now["mNrm"] - self.controls["cNrm"]
        # Useful in some cases to precalculate asset level
        self.state_now["aLvl"] = self.state_now["aNrm"] * self.state_now["pLvl"]

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

    def check_AIC(self, verbose=None):
        """
        Evaluate and report on the Absolute Impatience Condition
        """
        name = "AIC"

        def test(agent):
            return agent.thorn < 1

        messages = {
            True: "The value of the Absolute Patience Factor (APF) for the supplied parameter values satisfies the Absolute Impatience Condition.",
            False: "The given type violates the Absolute Impatience Condition with the supplied parameter values; the APF is {0.thorn}",
        }
        verbose_messages = {
            True: "  Because the APF < 1, the absolute amount of consumption is expected to fall over time.",
            False: "  Because the APF > 1, the absolute amount of consumption is expected to grow over time.",
        }
        verbose = self.verbose if verbose is None else verbose
        self.check_condition(name, test, messages, verbose, verbose_messages)

    def check_GICRaw(self, verbose=None):
        """
        Evaluate and report on the Growth Impatience Condition for the Perfect Foresight model
        """
        name = "GICRaw"

        self.GPFRaw = self.thorn / self.PermGroFac[0]

        def test(agent):
            return agent.GPFRaw < 1

        messages = {
            True: "The value of the Growth Patience Factor for the supplied parameter values satisfies the Perfect Foresight Growth Impatience Condition.",
            False: "The value of the Growth Patience Factor for the supplied parameter values fails the Perfect Foresight Growth Impatience Condition; the GPFRaw is: {0.GPFRaw}",
        }

        verbose_messages = {
            True: "  Therefore, for a perfect foresight consumer, the ratio of individual wealth to permanent income will fall indefinitely.",
            False: "  Therefore, for a perfect foresight consumer, the ratio of individual wealth to permanent income is expected to grow toward infinity.",
        }
        verbose = self.verbose if verbose is None else verbose
        self.check_condition(name, test, messages, verbose, verbose_messages)

    def check_RIC(self, verbose=None):
        """
        Evaluate and report on the Return Impatience Condition
        """

        self.RPF = self.thorn / self.Rfree

        name = "RIC"

        def test(agent):
            return self.RPF < 1

        messages = {
            True: "The value of the Return Patience Factor for the supplied parameter values satisfies the Return Impatience Condition.",
            False: "The value of the Return Patience Factor for the supplied parameter values fails the Return Impatience Condition; the factor is {0.RPF}",
        }

        verbose_messages = {
            True: "  Therefore, the limiting consumption function is not c(m)=0 for all m",
            False: "  Therefore, if the FHWC is satisfied, the limiting consumption function is c(m)=0 for all m.",
        }
        verbose = self.verbose if verbose is None else verbose
        self.check_condition(name, test, messages, verbose, verbose_messages)

    def check_FHWC(self, verbose=None):
        """
        Evaluate and report on the Finite Human Wealth Condition
        """

        self.FHWF = self.PermGroFac[0] / self.Rfree
        self.cNrmPDV = 1.0 / (1.0 - self.thorn / self.Rfree)

        name = "FHWC"

        def test(agent):
            return self.FHWF < 1

        messages = {
            True: "The Finite Human wealth factor value for the supplied parameter values satisfies the Finite Human Wealth Condition.",
            False: "The given type violates the Finite Human Wealth Condition; the Finite Human wealth factor value is {0.FHWF}",
        }

        verbose_messages = {
            True: "  Therefore, the limiting consumption function is not c(m)=Infinity\nand human wealth normalized by permanent income is {0.hNrm}\nand the PDV of future consumption growth is {0.cNrmPDV}",
            False: "  Therefore, the limiting consumption function is c(m)=Infinity for all m unless the RIC is also violated.  If both FHWC and RIC fail and the consumer faces a liquidity constraint, the limiting consumption function is nondegenerate but has a limiting slope of 0.  (https://econ-ark.github.io/BufferStockTheory#PFGICRawHoldsFHWCFailsRICFailsDiscuss)",
        }
        verbose = self.verbose if verbose is None else verbose
        self.check_condition(name, test, messages, verbose)

    def check_conditions(self, verbose=None):
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
        self.conditions = {}

        self.violated = False

        # This method only checks for the conditions for infinite horizon models
        # with a 1 period cycle. If these conditions are not met, we exit early.
        if self.cycles != 0 or self.T_cycle > 1:
            return

        self.thorn = (self.Rfree * self.DiscFac * self.LivPrb[0]) ** (1 / self.CRRA)

        verbose = self.verbose if verbose is None else verbose
        self.check_AIC(verbose)
        self.check_GICRaw(verbose)
        self.check_RIC(verbose)
        self.check_FHWC(verbose)

        if hasattr(self, "BoroCnstArt") and self.BoroCnstArt is not None:
            self.violated = not self.conditions["RIC"]
        else:
            self.violated = not self.conditions["RIC"] or not self.conditions["FHWC"]


# Make a dictionary to specify an idiosyncratic income shocks consumer
init_idiosyncratic_shocks = dict(
    init_perfect_foresight,
    **{  # assets above grid parameters
        "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
        "aXtraMax": 20,  # Maximum end-of-period "assets above minimum" value
        # Exponential nesting factor when constructing "assets above minimum" grid
        "aXtraNestFac": 3,
        "aXtraCount": 48,  # Number of points in the grid of "assets above minimum"
        "aXtraExtra": [
            None
        ],  # Some other value of "assets above minimum" to add to the grid, not used
        # Income process variables
        # Standard deviation of log permanent income shocks
        "PermShkStd": [0.1],
        "PermShkCount": 7,  # Number of points in discrete approximation to permanent income shocks
        # Standard deviation of log transitory income shocks
        "TranShkStd": [0.1],
        "TranShkCount": 7,  # Number of points in discrete approximation to transitory income shocks
        "UnempPrb": 0.05,  # Probability of unemployment while working
        "UnempPrbRet": 0.005,  # Probability of "unemployment" while retired
        "IncUnemp": 0.3,  # Unemployment benefits replacement rate
        "IncUnempRet": 0.0,  # "Unemployment" benefits when retired
        # Artificial borrowing constraint; imposed minimum level of end-of period assets
        "BoroCnstArt": 0.0,
        "tax_rate": 0.0,  # Flat income tax rate
        "T_retire": 0,  # Period of retirement (0 --> no retirement)
        "vFuncBool": False,  # Whether to calculate the value function during solution
        "CubicBool": False,  # Use cubic spline interpolation when True, linear interpolation when False
        "neutral_measure": False,
        # Use permanent income neutral measure (see Harmenberg 2021) during simulations when True.
        "NewbornTransShk": False,  # Whether Newborns have transitory shock. The default is False.
    }
)


class IndShockConsumerType(PerfForesightConsumerType):
    """
    A consumer type with idiosyncratic shocks to permanent and transitory income.
    His problem is defined by a sequence of income distributions, survival prob-
    abilities, and permanent income growth rates, as well as time invariant values
    for risk aversion, discount factor, the interest rate, the grid of end-of-
    period assets, and an artificial borrowing constraint.

    Parameters
    ----------
    cycles : int
        Number of times the sequence of periods should be solved.
    """

    time_inv_ = PerfForesightConsumerType.time_inv_ + [
        "BoroCnstArt",
        "vFuncBool",
        "CubicBool",
    ]
    time_inv_.remove(
        "MaxKinks"
    )  # This is in the PerfForesight model but not ConsIndShock
    shock_vars_ = ["PermShk", "TranShk"]

    def __init__(self, verbose=1, quiet=False, **kwds):
        params = init_idiosyncratic_shocks.copy()
        params.update(kwds)

        # Initialize a basic AgentType
        PerfForesightConsumerType.__init__(self, verbose=verbose, quiet=quiet, **params)

        # Add consumer-type specific objects, copying to create independent versions
        if (not self.CubicBool) and (not self.vFuncBool):
            solver = ConsIndShockSolverBasic
        else:  # Use the "advanced" solver if either is requested
            solver = ConsIndShockSolver
        self.solve_one_period = make_one_period_oo_solver(solver)

        self.update()  # Make assets grid, income process, terminal solution

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
        """
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

    def get_shocks(self):
        """
        Gets permanent and transitory income shocks for this period.  Samples from IncShkDstn for
        each period in the cycle.

        Parameters
        ----------
        NewbornTransShk : boolean, optional
            Whether Newborns have transitory shock. The default is False.

        Returns
        -------
        None
        """
        NewbornTransShk = (
            self.NewbornTransShk
        )  # Whether Newborns have transitory shock. The default is False.

        PermShkNow = np.zeros(self.AgentCount)  # Initialize shock arrays
        TranShkNow = np.zeros(self.AgentCount)
        newborn = self.t_age == 0
        for t in range(self.T_cycle):
            these = t == self.t_cycle

            # temporary, see #1022
            if self.cycles == 1:
                t = t - 1

            N = np.sum(these)
            if N > 0:
                # set current income distribution
                IncShkDstnNow = self.IncShkDstn[t]
                # and permanent growth factor
                PermGroFacNow = self.PermGroFac[t]
                # Get random draws of income shocks from the discrete distribution
                IncShks = IncShkDstnNow.draw(N)

                PermShkNow[these] = (
                    IncShks[0, :] * PermGroFacNow
                )  # permanent "shock" includes expected growth
                TranShkNow[these] = IncShks[1, :]

        # That procedure used the *last* period in the sequence for newborns, but that's not right
        # Redraw shocks for newborns, using the *first* period in the sequence.  Approximation.
        N = np.sum(newborn)
        if N > 0:
            these = newborn
            # set current income distribution
            IncShkDstnNow = self.IncShkDstn[0]
            PermGroFacNow = self.PermGroFac[0]  # and permanent growth factor

            # Get random draws of income shocks from the discrete distribution
            EventDraws = IncShkDstnNow.draw_events(N)
            PermShkNow[these] = (
                IncShkDstnNow.atoms[0][EventDraws] * PermGroFacNow
            )  # permanent "shock" includes expected growth
            TranShkNow[these] = IncShkDstnNow.atoms[1][EventDraws]
        #        PermShkNow[newborn] = 1.0
        #  Whether Newborns have transitory shock. The default is False.
        if not NewbornTransShk:
            TranShkNow[newborn] = 1.0

        # Store the shocks in self
        self.EmpNow = np.ones(self.AgentCount, dtype=bool)
        self.EmpNow[TranShkNow == self.IncUnemp] = False
        self.shocks["PermShk"] = PermShkNow
        self.shocks["TranShk"] = TranShkNow

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
        Ex_IncNext = np.dot(ShkPrbsNext, PermShkValsNext * TranShkValsNext)
        PermShkMinNext = np.min(PermShkValsNext)
        TranShkMinNext = np.min(TranShkValsNext)
        WorstIncNext = PermShkMinNext * TranShkMinNext
        WorstIncPrb = np.sum(
            ShkPrbsNext[(PermShkValsNext * TranShkValsNext) == WorstIncNext]
        )

        # Calculate human wealth and the infinite horizon natural borrowing constraint
        hNrm = (Ex_IncNext * self.PermGroFac[0] / self.Rfree) / (
            1.0 - self.PermGroFac[0] / self.Rfree
        )
        temp = self.PermGroFac[0] * PermShkMinNext / self.Rfree
        BoroCnstNat = -TranShkMinNext * temp / (1.0 - temp)

        PatFac = (self.DiscFac * self.LivPrb[0] * self.Rfree) ** (
            1.0 / self.CRRA
        ) / self.Rfree
        if BoroCnstNat < self.BoroCnstArt:
            MPCmax = 1.0  # if natural borrowing constraint is overridden by artificial one, MPCmax is 1
        else:
            MPCmax = 1.0 - WorstIncPrb ** (1.0 / self.CRRA) * PatFac
        MPCmin = 1.0 - PatFac

        # Store the results as attributes of self
        self.hNrm = hNrm
        self.MPCmin = MPCmin
        self.MPCmax = MPCmax

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
            * self.LivPrb[0]
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
        #        AgentType.pre_solve(self)
        # Update all income process variables to match any attributes that might
        # have been changed since `__init__` or `solve()` was last called.
        #        self.update_income_process()
        self.update_solution_terminal()
        if not self.quiet:
            self.check_conditions(verbose=self.verbose)

    def check_GICNrm(self, verbose=None):
        """
        Check Individual Growth Patience Factor.
        """
        self.GPFNrm = self.thorn / (
            self.PermGroFac[0] * self.InvEx_PermShkInv
        )  # [url]/#GICRawI

        name = "GICRaw"

        def test(agent):
            return agent.GPFNrm <= 1

        messages = {
            True: "\nThe value of the Individual Growth Patience Factor for the supplied parameter values satisfies the Growth Impatience Condition; the value of the GPFNrm is: {0.GPFNrm}",
            False: "\nThe given parameter values violate the Normalized Growth Impatience Condition; the GPFNrm is: {0.GPFNrm}",
        }

        verbose_messages = {
            True: " Therefore, a target level of the individual market resources ratio m exists (see {0.url}/#onetarget for more).\n",
            False: " Therefore, a target ratio of individual market resources to individual permanent income does not exist.  (see {0.url}/#onetarget for more).\n",
        }
        verbose = self.verbose if verbose is None else verbose
        self.check_condition(name, test, messages, verbose, verbose_messages)

    def check_GICAggLivPrb(self, verbose=None):
        name = "GICAggLivPrb"

        def test(agent):
            return agent.GPFAggLivPrb <= 1

        messages = {
            True: "\nThe value of the Mortality Adjusted Aggregate Growth Patience Factor for the supplied parameter values satisfies the Mortality Adjusted Aggregate Growth Imatience Condition; the value of the GPFAggLivPrb is: {0.GPFAggLivPrb}",
            False: "\nThe given parameter values violate the Mortality Adjusted Aggregate Growth Imatience Condition; the GPFAggLivPrb is: {0.GPFAggLivPrb}",
        }

        verbose_messages = {  # (see {0.url}/#WRIC for more).',
            True: "  Therefore, a target level of the ratio of aggregate market resources to aggregate permanent income exists.\n",
            # (see {0.url}/#WRIC for more).'
            False: "  Therefore, a target ratio of aggregate resources to aggregate permanent income may not exist.\n",
        }
        verbose = self.verbose if verbose is None else verbose
        self.check_condition(name, test, messages, verbose, verbose_messages)

    def check_WRIC(self, verbose=None):
        """
        Evaluate and report on the Weak Return Impatience Condition
        [url]/#WRPF modified to incorporate LivPrb
        """
        self.WRPF = (
            (self.UnempPrb ** (1 / self.CRRA))
            * (self.Rfree * self.DiscFac * self.LivPrb[0]) ** (1 / self.CRRA)
            / self.Rfree
        )

        name = "WRIC"

        def test(agent):
            return agent.WRPF <= 1

        messages = {
            True: "\nThe Weak Return Patience Factor value for the supplied parameter values satisfies the Weak Return Impatience Condition; the WRPF is {0.WRPF}.",
            False: "\nThe Weak Return Patience Factor value for the supplied parameter values fails     the Weak Return Impatience Condition; the WRPF is {0.WRPF} (see {0.url}/#WRIC for more).",
        }

        verbose_messages = {
            True: "  Therefore, a nondegenerate solution exists if the FVAC is also satisfied.  (see {0.url}/#WRIC for more) \n",
            False: "  Therefore, a nondegenerate solution is not available (see {0.url}/#WRIC for more). \n",
        }
        verbose = self.verbose if verbose is None else verbose
        self.check_condition(name, test, messages, verbose, verbose_messages)

    def check_FVAC(self, verbose=None):
        """
        Evaluate and report on the Finite Value of Autarky Condition
        Hyperlink to paper: [url]/#Autarky-Value
        """
        EpShkuInv = expected(lambda x: x ** (1 - self.CRRA), self.PermShkDstn[0])[0]

        if self.CRRA != 1.0:
            uInvEpShkuInv = EpShkuInv ** (
                1 / (1 - self.CRRA)
            )  # The term that gives a utility-consequence-adjusted utility growth
        else:
            uInvEpShkuInv = 1.0

        self.uInvEpShkuInv = uInvEpShkuInv

        self.VAF = self.LivPrb[0] * self.DiscFac * self.uInvEpShkuInv

        name = "FVAC"

        def test(agent):
            return agent.VAF <= 1

        messages = {
            True: "\nThe Value of Autarky Factor (VAF) for the supplied parameter values satisfies the Finite Value of Autarky Condition; the VAF is {0.VAF}",
            False: "\nThe Value of Autarky Factor (VAF) for the supplied parameter values fails     the Finite Value of Autarky Condition; the VAF is {0.VAF}",
        }

        verbose_messages = {
            True: "  Therefore, a nondegenerate solution exists if the WRIC also holds; see {0.url}/#Conditions-Under-Which-the-Problem-Defines-a-Contraction-Mapping\n",
            False: "  Therefore, a nondegenerate solution is not available (see {0.url}/#Conditions-Under-Which-the-Problem-Defines-a-Contraction-Mapping\n",
        }
        verbose = self.verbose if verbose is None else verbose
        self.check_condition(name, test, messages, verbose, verbose_messages)

    def check_conditions(self, verbose=None):
        """
        This method checks whether the instance's type satisfies the Absolute Impatience Condition (AIC), Weak Return
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

        # PerfForesightConsumerType.check_conditions(self, verbose=False, verbose_reference=False)
        self.violated = False

        if self.cycles != 0 or self.T_cycle > 1:
            return

        # For theory, see hyperlink targets to expressions in
        # url=https://econ-ark.github.io/BufferStockTheory
        # For example, the hyperlink to the relevant section of the paper
        self.url = "https://econ-ark.github.io/BufferStockTheory"
        # would be referenced below as:
        # [url]/#Uncertainty-Modified-Conditions

        self.Ex_PermShkInv = expected(lambda x: 1 / x, self.PermShkDstn[0])[0]
        # $\Ex_{t}[\psi^{-1}_{t+1}]$ (in first eqn in sec)

        # [url]/#Pat, adjusted to include mortality

        self.InvEx_PermShkInv = (
            1 / self.Ex_PermShkInv
        )  # $\underline{\psi}$ in the paper (\bar{\isp} in private version)
        self.PermGroFacAdj = (
            self.PermGroFac[0] * self.InvEx_PermShkInv
        )  # [url]/#PGroAdj

        self.thorn = ((self.Rfree * self.DiscFac)) ** (1 / self.CRRA)

        # self.Ex_RNrm           = self.Rfree*Ex_PermShkInv/(self.PermGroFac[0]*self.LivPrb[0])
        self.GPFRaw = self.thorn / (self.PermGroFac[0])  # [url]/#GPF
        # Lower bound of aggregate wealth growth if all inheritances squandered

        self.GPFAggLivPrb = self.thorn * self.LivPrb[0] / self.PermGroFac[0]

        self.DiscFacGPFRawMax = ((self.PermGroFac[0]) ** (self.CRRA)) / (
            self.Rfree
        )  # DiscFac at growth impatience knife edge
        self.DiscFacGPFNrmMax = (
            (self.PermGroFac[0] * self.InvEx_PermShkInv) ** (self.CRRA)
        ) / (
            self.Rfree
        )  # DiscFac at growth impatience knife edge
        self.DiscFacGPFAggLivPrbMax = ((self.PermGroFac[0]) ** (self.CRRA)) / (
            self.Rfree * self.LivPrb[0]
        )  # DiscFac at growth impatience knife edge
        verbose = self.verbose if verbose is None else verbose

        #        self.check_GICRaw(verbose)
        self.check_GICNrm(verbose)
        self.check_GICAggLivPrb(verbose)
        self.check_WRIC(verbose)
        self.check_FVAC(verbose)

        self.violated = not self.conditions["WRIC"] or not self.conditions["FVAC"]

        if self.violated:
            _log.warning(
                '\n[!] For more information on the conditions, see Tables 3 and 4 in "Theoretical Foundations of Buffer Stock Saving" at '
                + self.url
                + "/#Factors-Defined-And-Compared"
            )

        _log.warning("GPFRaw                 = %2.6f " % (self.GPFRaw))
        _log.warning("GPFNrm                 = %2.6f " % (self.GPFNrm))
        _log.warning("GPFAggLivPrb           = %2.6f " % (self.GPFAggLivPrb))
        _log.warning("Thorn = APF            = %2.6f " % (self.thorn))
        _log.warning("PermGroFacAdj          = %2.6f " % (self.PermGroFacAdj))
        _log.warning("uInvEpShkuInv          = %2.6f " % (self.uInvEpShkuInv))
        _log.warning("VAF                    = %2.6f " % (self.VAF))
        _log.warning("WRPF                   = %2.6f " % (self.WRPF))
        _log.warning("DiscFacGPFNrmMax       = %2.6f " % (self.DiscFacGPFNrmMax))
        _log.warning("DiscFacGPFAggLivPrbMax = %2.6f " % (self.DiscFacGPFAggLivPrbMax))

    def Ex_Mtp1_over_Ex_Ptp1(self, mNrm):
        cNrm = self.solution[-1].cFunc(mNrm)
        aNrm = mNrm - cNrm
        Ex_Ptp1 = PermGroFac[0]
        Ex_bLev_tp1 = aNrm * self.Rfree
        Ex_Mtp1 = Ex_bLev_tp1
        return Ex_Mtp1 / Ex_Ptp1

    def Ex_mtp1(self, mNrm):
        cNrm = self.solution[-1].cFunc(mNrm)
        aNrm = mNrm - cNrm
        Ex_bNrm_tp1 = aNrm * self.Rfree * self.Ex_PermShkInv / self.PermGroFac[0]
        Ex_Mtp1 = (Ex_bNrm_tp1 + 1) * Ex_Ptp1  # mean TranShk and PermShk are 1
        return Ex_Mtp1 / Ex_Ptp1

    def calc_stable_points(self):
        """
        If the problem is one that satisfies the conditions required for target ratios of different
        variables to permanent income to exist, and has been solved to within the self-defined
        tolerance, this method calculates the target values of market resources, consumption,
        and assets.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        infinite_horizon = cycles_left == 0
        if not infinite_horizon:
            _log.warning(
                "The calc_stable_points method works only for infinite horizon models."
            )
            return

    # = Functions for generating discrete income processes and
    #   simulated income shocks =
    # ========================================================

    def construct_lognormal_income_process_unemployment(self):
        """
        Generates a list of discrete approximations to the income process for each
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
        UnempPrb : float or [float]
            The probability of becoming unemployed during the working period.
        UnempPrbRet : float or None
            The probability of not receiving typical retirement income when retired.
        T_retire : int
            The index value for the final working period in the agent's life.
            If T_retire <= 0 then there is no retirement.
        IncUnemp : float or [float]
            Transitory income received when unemployed.
        IncUnempRet : float or None
            Transitory income received while "unemployed" when retired.
        T_cycle :  int
            Total number of non-terminal periods in the consumer's sequence of periods.

        Returns
        -------
        IncShkDstn :  [distribution.Distribution]
            A list with T_cycle elements, each of which is a
            discrete approximation to the income process in a period.
        PermShkDstn : [[distribution.Distributiony]]
            A list with T_cycle elements, each of which
            a discrete approximation to the permanent income shocks.
        TranShkDstn : [[distribution.Distribution]]
            A list with T_cycle elements, each of which
            a discrete approximation to the transitory income shocks.
        """
        # Unpack the parameters from the input
        T_cycle = self.T_cycle
        PermShkStd = self.PermShkStd
        PermShkCount = self.PermShkCount
        TranShkStd = self.TranShkStd
        TranShkCount = self.TranShkCount
        T_retire = self.T_retire
        UnempPrb = self.UnempPrb
        IncUnemp = self.IncUnemp
        UnempPrbRet = self.UnempPrbRet
        IncUnempRet = self.IncUnempRet

        if T_retire > 0:
            normal_length = T_retire
            retire_length = T_cycle - T_retire
        else:
            normal_length = T_cycle
            retire_length = 0

        if all(
            [
                isinstance(x, (float, int)) or (x is None)
                for x in [UnempPrb, IncUnemp, UnempPrbRet, IncUnempRet]
            ]
        ):

            UnempPrb_list = [UnempPrb] * normal_length + [UnempPrbRet] * retire_length
            IncUnemp_list = [IncUnemp] * normal_length + [IncUnempRet] * retire_length

        elif all([isinstance(x, list) for x in [UnempPrb, IncUnemp]]):

            UnempPrb_list = UnempPrb
            IncUnemp_list = IncUnemp

        else:

            raise Exception(
                "Unemployment must be specified either using floats for UnempPrb,"
                + "IncUnemp, UnempPrbRet, and IncUnempRet, in which case the "
                + "unemployment probability and income change only with retirement, or "
                + "using lists of length T_cycle for UnempPrb and IncUnemp, specifying "
                + "each feature at every age."
            )

        PermShkCount_list = [PermShkCount] * normal_length + [1] * retire_length
        TranShkCount_list = [TranShkCount] * normal_length + [1] * retire_length

        if not hasattr(self, "neutral_measure"):
            self.neutral_measure = False

        neutral_measure_list = [self.neutral_measure] * len(PermShkCount_list)

        IncShkDstn = IndexDistribution(
            engine=BufferStockIncShkDstn,
            conditional={
                "sigma_Perm": PermShkStd,
                "sigma_Tran": TranShkStd,
                "n_approx_Perm": PermShkCount_list,
                "n_approx_Tran": TranShkCount_list,
                "neutral_measure": neutral_measure_list,
                "UnempPrb": UnempPrb_list,
                "IncUnemp": IncUnemp_list,
            },
            RNG=self.RNG,
        )

        PermShkDstn = IndexDistribution(
            engine=LognormPermIncShk,
            conditional={
                "sigma": PermShkStd,
                "n_approx": PermShkCount_list,
                "neutral_measure": neutral_measure_list,
            },
        )

        TranShkDstn = IndexDistribution(
            engine=MixtureTranIncShk,
            conditional={
                "sigma": TranShkStd,
                "UnempPrb": UnempPrb_list,
                "IncUnemp": IncUnemp_list,
                "n_approx": TranShkCount_list,
            },
        )

        return IncShkDstn, PermShkDstn, TranShkDstn


class LognormPermIncShk(DiscreteDistribution):
    """
    A one-period distribution of a multiplicative lognormal permanent income shock.

    Parameters
    ----------
    sigma : float
        Standard deviation of the log-shock.
    n_approx : int
        Number of points to use in the discrete approximation.
    neutral_measure : Bool, optional
        Whether to use Hamenberg's permanent-income-neutral measure. The default is False.
    seed : int, optional
        Random seed. The default is 0.

    Returns
    -------
    PermShkDstn : DiscreteDistribution
        Permanent income shock distribution.

    """

    def __init__(self, sigma, n_approx, neutral_measure=False, seed=0):
        # Construct an auxiliary discretized normal
        logn_approx = MeanOneLogNormal(sigma).approx(
            n_approx if sigma > 0.0 else 1, tail_N=0
        )
        # Change the pmv if necessary
        if neutral_measure:
            logn_approx.pmv = (logn_approx.atoms * logn_approx.pmv).flatten()

        super().__init__(pmv=logn_approx.pmv, atoms=logn_approx.atoms, seed=seed)


class MixtureTranIncShk(DiscreteDistribution):
    """
    A one-period distribution for transitory income shocks that are a mixture
    between a log-normal and a single-value unemployment shock.

    Parameters
    ----------
    sigma : float
        Standard deviation of the log-shock.
    UnempPrb : float
        Probability of the "unemployment" shock.
    IncUnemp : float
        Income shock in the "unemployment" state.
    n_approx : int
        Number of points to use in the discrete approximation.
    seed : int, optional
        Random seed. The default is 0.

    Returns
    -------
    TranShkDstn : DiscreteDistribution
        Transitory income shock distribution.

    """

    def __init__(self, sigma, UnempPrb, IncUnemp, n_approx, seed=0):
        dstn_approx = MeanOneLogNormal(sigma).approx(
            n_approx if sigma > 0.0 else 1, tail_N=0
        )
        if UnempPrb > 0.0:
            dstn_approx = add_discrete_outcome_constant_mean(
                dstn_approx, p=UnempPrb, x=IncUnemp
            )

        super().__init__(pmv=dstn_approx.pmv, atoms=dstn_approx.atoms, seed=seed)


class BufferStockIncShkDstn(DiscreteDistribution):
    """
    A one-period distribution object for the joint distribution of income
    shocks (permanent and transitory), as modeled in the Buffer Stock Theory
    paper:
        - Lognormal, discretized permanent income shocks.
        - Transitory shocks that are a mixture of:
            - A lognormal distribution in normal times.
            - An "unemployment" shock.

    Parameters
    ----------
    sigma_Perm : float
        Standard deviation of the log- permanent shock.
    sigma_Tran : float
        Standard deviation of the log- transitory shock.
    n_approx_Perm : int
        Number of points to use in the discrete approximation of the permanent shock.
    n_approx_Tran : int
        Number of points to use in the discrete approximation of the transitory shock.
    UnempPrb : float
        Probability of the "unemployment" shock.
    IncUnemp : float
        Income shock in the "unemployment" state.
    neutral_measure : Bool, optional
        Whether to use Hamenberg's permanent-income-neutral measure. The default is False.
    seed : int, optional
        Random seed. The default is 0.

    Returns
    -------
    IncShkDstn : DiscreteDistribution
        Income shock distribution.

    """

    def __init__(
        self,
        sigma_Perm,
        sigma_Tran,
        n_approx_Perm,
        n_approx_Tran,
        UnempPrb,
        IncUnemp,
        neutral_measure=False,
        seed=0,
    ):
        perm_dstn = LognormPermIncShk(
            sigma=sigma_Perm, n_approx=n_approx_Perm, neutral_measure=neutral_measure
        )
        tran_dstn = MixtureTranIncShk(
            sigma=sigma_Tran,
            UnempPrb=UnempPrb,
            IncUnemp=IncUnemp,
            n_approx=n_approx_Tran,
        )

        joint_dstn = combine_indep_dstns(perm_dstn, tran_dstn)

        super().__init__(pmv=joint_dstn.pmv, atoms=joint_dstn.atoms, seed=seed)


# Make a dictionary to specify a "kinked R" idiosyncratic shock consumer
init_kinked_R = dict(
    init_idiosyncratic_shocks,
    **{
        "Rboro": 1.20,  # Interest factor on assets when borrowing, a < 0
        "Rsave": 1.02,  # Interest factor on assets when saving, a > 0
        "BoroCnstArt": None,
        # kinked R is a bit silly if borrowing not allowed
        "CubicBool": True,
        # kinked R is now compatible with linear cFunc and cubic cFunc
        "aXtraCount": 48,
        # ...so need lots of extra gridpoints to make up for it
    }
)
del init_kinked_R["Rfree"]  # get rid of constant interest factor


class KinkedRconsumerType(IndShockConsumerType):
    """
    A consumer type that faces idiosyncratic shocks to income and has a different
    interest factor on saving vs borrowing.  Extends IndShockConsumerType, with
    very small changes.  Solver for this class is currently only compatible with
    linear spline interpolation.

    Same parameters as AgentType.


    Parameters
    ----------
    """

    time_inv_ = copy(IndShockConsumerType.time_inv_)
    time_inv_ += ["Rboro", "Rsave"]

    def __init__(self, **kwds):
        params = init_kinked_R.copy()
        params.update(kwds)

        # Initialize a basic AgentType
        PerfForesightConsumerType.__init__(self, **params)

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
        Ex_IncNext = expected(lambda trans, perm: trans * perm, self.IncShkDstn)
        PermShkMinNext = np.min(PermShkValsNext)
        TranShkMinNext = np.min(TranShkValsNext)
        WorstIncNext = PermShkMinNext * TranShkMinNext
        WorstIncPrb = np.sum(
            ShkPrbsNext[(PermShkValsNext * TranShkValsNext) == WorstIncNext]
        )

        # Calculate human wealth and the infinite horizon natural borrowing constraint
        hNrm = (Ex_IncNext * self.PermGroFac[0] / self.Rsave) / (
            1.0 - self.PermGroFac[0] / self.Rsave
        )
        temp = self.PermGroFac[0] * PermShkMinNext / self.Rboro
        BoroCnstNat = -TranShkMinNext * temp / (1.0 - temp)

        PatFacTop = (self.DiscFac * self.LivPrb[0] * self.Rsave) ** (
            1.0 / self.CRRA
        ) / self.Rsave
        PatFacBot = (self.DiscFac * self.LivPrb[0] * self.Rboro) ** (
            1.0 / self.CRRA
        ) / self.Rboro
        if BoroCnstNat < self.BoroCnstArt:
            MPCmax = 1.0  # if natural borrowing constraint is overridden by artificial one, MPCmax is 1
        else:
            MPCmax = 1.0 - WorstIncPrb ** (1.0 / self.CRRA) * PatFacBot
        MPCmin = 1.0 - PatFacTop

        # Store the results as attributes of self
        self.hNrm = hNrm
        self.MPCmin = MPCmin
        self.MPCmax = MPCmax

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
        on whether self.aNrmNow >< 0.

        Parameters
        ----------
        None

        Returns
        -------
        RfreeNow : np.array
             Array of size self.AgentCount with risk free interest rate for each agent.
        """
        RfreeNow = self.Rboro * np.ones(self.AgentCount)
        RfreeNow[self.state_prev["aNrm"] > 0] = self.Rsave
        return RfreeNow

    def check_conditions(self):
        """
        This method checks whether the instance's type satisfies the Absolute Impatience Condition (AIC),
        the Return Impatience Condition (RIC), the Growth Impatience Condition (GICRaw), the Normalized Growth Impatience Condition (GIC-Nrm), the Weak Return
        Impatience Condition (WRIC), the Finite Human Wealth Condition (FHWC) and the Finite Value of
        Autarky Condition (FVAC). To check which conditions are relevant to the model at hand, a
        reference to the relevant theoretical literature is made.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        raise NotImplementedError()


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
    female=False, cross_sec=True, year=2004, min_age=birth_age, max_age=death_age - 1
)

# Parameters related to the number of periods implied by the calibration
time_params = parse_time_params(age_birth=birth_age, age_death=death_age)

# Update all the new parameters
init_lifecycle = copy(init_idiosyncratic_shocks)
init_lifecycle.update(time_params)
init_lifecycle.update(dist_params)
# Note the income specification overrides the pLvlInitMean from the SCF.
init_lifecycle.update(income_params)
init_lifecycle.update({"LivPrb": liv_prb})

# Make a dictionary to specify an infinite consumer with a four period cycle
init_cyclical = copy(init_idiosyncratic_shocks)
init_cyclical["PermGroFac"] = [1.1, 1.082251, 2.8, 0.3]
init_cyclical["PermShkStd"] = [0.1, 0.1, 0.1, 0.1]
init_cyclical["TranShkStd"] = [0.1, 0.1, 0.1, 0.1]
init_cyclical["LivPrb"] = 4 * [0.98]
init_cyclical["T_cycle"] = 4
