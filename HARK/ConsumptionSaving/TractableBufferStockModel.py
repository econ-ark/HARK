"""
Defines and solves the Tractable Buffer Stock model described in lecture notes
for "A Tractable Model of Buffer Stock Saving" (henceforth, TBS) available at
https://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/consumption/TractableBufferStock
The model concerns an agent with constant relative risk aversion utility making
decisions over consumption and saving.  He is subject to only a very particular
sort of risk: the possibility that he will become permanently unemployed until
the day he dies; barring this, his income is certain and grows at a constant rate.

The model has an infinite horizon, but is not solved by backward iteration in a
traditional sense.  Because of the very specific assumptions about risk, it is
possible to find the agent's steady state or target level of market resources
when employed, as well as information about the optimal consumption rule at this
target level.  The full consumption function can then be constructed by "back-
shooting", inverting the Euler equation to find what consumption *must have been*
in the previous period.  The consumption function is thus constructed by repeat-
edly adding "stable arm" points to either end of a growing list until specified
bounds are exceeded.

Despite the non-standard solution method, the iterative process can be embedded
in the HARK framework, as shown below.
"""

from copy import copy

import numpy as np
from scipy.optimize import brentq, newton

from HARK import AgentType, NullFunc
from HARK.distributions import Bernoulli, Lognormal
from HARK.interpolation import LinearInterp, CubicInterp

# Import the HARK library.
from HARK.metric import MetricObject
from HARK.rewards import (
    CRRAutility,
    CRRAutility_inv,
    CRRAutility_invP,
    CRRAutilityP,
    CRRAutilityP_inv,
    CRRAutilityPP,
    CRRAutilityPPP,
    CRRAutilityPPPP,
)

__all__ = ["TractableConsumerSolution", "TractableConsumerType"]

# If you want to run the "tractable" version of cstwMPC, use cstwMPCagent from
# cstwMPC REMARK and have TractableConsumerType inherit from cstwMPCagent rather than AgentType

# Define utility function and its derivatives (plus inverses)
utility = CRRAutility
utilityP = CRRAutilityP
utilityPP = CRRAutilityPP
utilityPPP = CRRAutilityPPP
utilityPPPP = CRRAutilityPPPP
utilityP_inv = CRRAutilityP_inv
utility_invP = CRRAutility_invP
utility_inv = CRRAutility_inv


class TractableConsumerSolution(MetricObject):
    """
    A class representing the solution to a tractable buffer saving problem.
    Attributes include a list of money points mNrm_list, a list of consumption points
    cNrm_list, a list of MPCs MPC_list, a perfect foresight consumption function
    while employed, and a perfect foresight consumption function while unemployed.
    The solution includes a consumption function constructed from the lists.

    Parameters
    ----------
    mNrm_list : [float]
        List of normalized market resources points on the stable arm.
    cNrm_list : [float]
        List of normalized consumption points on the stable arm.
    MPC_list : [float]
        List of marginal propensities to consume on the stable arm, corres-
        ponding to the (mNrm,cNrm) points.
    cFunc_U : function
        The (linear) consumption function when permanently unemployed.
    cFunc : function
        The consumption function when employed.
    """

    def __init__(
        self,
        mNrm_list=None,
        cNrm_list=None,
        MPC_list=None,
        cFunc_U=NullFunc,
        cFunc=NullFunc,
    ):
        self.mNrm_list = mNrm_list if mNrm_list is not None else list()
        self.cNrm_list = cNrm_list if cNrm_list is not None else list()
        self.MPC_list = MPC_list if MPC_list is not None else list()
        self.cFunc_U = cFunc_U
        self.cFunc = cFunc
        self.distance_criteria = ["PointCount"]
        # The distance between two solutions is the difference in the number of
        # stable arm points in each.  This is a very crude measure of distance
        # that captures the notion that the process is over when no points are added.


def find_next_point(
    DiscFac,
    Rfree,
    CRRA,
    PermGroFacCmp,
    UnempPrb,
    Rnrm,
    Beth,
    cNext,
    mNext,
    MPCnext,
    PFMPC,
):
    """
    Calculates what consumption, market resources, and the marginal propensity
    to consume must have been in the previous period given model parameters and
    values of market resources, consumption, and MPC today.

    Parameters
    ----------
    DiscFac : float
        Intertemporal discount factor on future utility.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    PermGroFacCmp : float
        Permanent income growth factor, compensated for the possibility of
        permanent unemployment.
    UnempPrb : float
        Probability of becoming permanently unemployed.
    Rnrm : float
        Interest factor normalized by compensated permanent income growth factor.
    Beth : float
        Composite effective discount factor for reverse shooting solution; defined
        in appendix "Numerical Solution/The Consumption Function" in TBS
        lecture notes
    cNext : float
        Normalized consumption in the succeeding period.
    mNext : float
        Normalized market resources in the succeeding period.
    MPCnext : float
        The marginal propensity to consume in the succeeding period.
    PFMPC : float
        The perfect foresight MPC; also the MPC when permanently unemployed.

    Returns
    -------
    mNow : float
        Normalized market resources this period.
    cNow : float
        Normalized consumption this period.
    MPCnow : float
        Marginal propensity to consume this period.
    """

    def uPP(x):
        return utilityPP(x, rho=CRRA)

    cNow = (
        PermGroFacCmp
        * (DiscFac * Rfree) ** (-1.0 / CRRA)
        * cNext
        * (1 + UnempPrb * ((cNext / (PFMPC * (mNext - 1.0))) ** CRRA - 1.0))
        ** (-1.0 / CRRA)
    )
    mNow = (PermGroFacCmp / Rfree) * (mNext - 1.0) + cNow
    cUNext = PFMPC * (mNow - cNow) * Rnrm
    # See TBS Appendix "E.1 The Consumption Function"
    natural = (
        Beth
        * Rnrm
        * (1.0 / uPP(cNow))
        * ((1.0 - UnempPrb) * uPP(cNext) * MPCnext + UnempPrb * uPP(cUNext) * PFMPC)
    )  # Convenience variable
    MPCnow = natural / (natural + 1)
    return mNow, cNow, MPCnow


def add_to_stable_arm_points(
    solution_next,
    DiscFac,
    Rfree,
    CRRA,
    PermGroFacCmp,
    UnempPrb,
    PFMPC,
    Rnrm,
    Beth,
    mLowerBnd,
    mUpperBnd,
):
    """
    Adds a one point to the bottom and top of the list of stable arm points if
    the bounding levels of mLowerBnd (lower) and mUpperBnd (upper) have not yet
    been met by a stable arm point in mNrm_list.  This acts as the "one period
    solver" / solve_one_period in the tractable buffer stock model.

    Parameters
    ----------
    solution_next : TractableConsumerSolution
        The solution object from the previous iteration of the backshooting
        procedure.  Not the "next period" solution per se.
    DiscFac : float
        Intertemporal discount factor on future utility.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    CRRA : float
        Coefficient of relative risk aversion.
    PermGroFacCmp : float
        Permanent income growth factor, compensated for the possibility of
        permanent unemployment.
    UnempPrb : float
        Probability of becoming permanently unemployed.
    PFMPC : float
        The perfect foresight MPC; also the MPC when permanently unemployed.
    Rnrm : float
        Interest factor normalized by compensated permanent income growth factor.
    Beth : float
        Damned if I know.
    mLowerBnd : float
        Lower bound on market resources for the backshooting process.  If
        min(solution_next.mNrm_list) < mLowerBnd, no new bottom point is found.
    mUpperBnd : float
        Upper bound on market resources for the backshooting process.  If
        max(solution_next.mNrm_list) > mUpperBnd, no new top point is found.

    Returns:
    ---------
    solution_now : TractableConsumerSolution
        A new solution object with new points added to the top and bottom.  If
        no new points were added, then the backshooting process is about to end.
    """
    # Unpack the lists of Euler points
    mNrm_list = copy(solution_next.mNrm_list)
    cNrm_list = copy(solution_next.cNrm_list)
    MPC_list = copy(solution_next.MPC_list)

    # Check whether to add a stable arm point to the top
    mNext = mNrm_list[-1]
    if mNext < mUpperBnd:
        # Get the rest of the data for the previous top point
        cNext = solution_next.cNrm_list[-1]
        MPCNext = solution_next.MPC_list[-1]

        # Calculate employed levels of c, m, and MPC from next period's values
        mNow, cNow, MPCnow = find_next_point(
            DiscFac,
            Rfree,
            CRRA,
            PermGroFacCmp,
            UnempPrb,
            Rnrm,
            Beth,
            cNext,
            mNext,
            MPCNext,
            PFMPC,
        )

        # Add this point to the top of the stable arm list
        mNrm_list.append(mNow)
        cNrm_list.append(cNow)
        MPC_list.append(MPCnow)

    # Check whether to add a stable arm point to the bottom
    mNext = mNrm_list[0]
    if mNext > mLowerBnd:
        # Get the rest of the data for the previous bottom point
        cNext = solution_next.cNrm_list[0]
        MPCNext = solution_next.MPC_list[0]

        # Calculate employed levels of c, m, and MPC from next period's values
        mNow, cNow, MPCnow = find_next_point(
            DiscFac,
            Rfree,
            CRRA,
            PermGroFacCmp,
            UnempPrb,
            Rnrm,
            Beth,
            cNext,
            mNext,
            MPCNext,
            PFMPC,
        )

        # Add this point to the top of the stable arm list
        mNrm_list.insert(0, mNow)
        cNrm_list.insert(0, cNow)
        MPC_list.insert(0, MPCnow)

    # Construct and return this period's solution
    solution_now = TractableConsumerSolution(
        mNrm_list=mNrm_list, cNrm_list=cNrm_list, MPC_list=MPC_list
    )
    solution_now.PointCount = len(mNrm_list)
    return solution_now


###############################################################################

# Define a dictionary for the tractable buffer stock model
init_tractable = {
    "cycles": 0,  # infinite horizon
    "T_cycle": 1,  # only one period repeated indefinitely
    "UnempPrb": 0.00625,  # Probability of becoming permanently unemployed
    "DiscFac": 0.975,  # Intertemporal discount factor
    "Rfree": 1.01,  # Risk-free interest factor on assets
    "PermGroFac": 1.0025,  # Permanent income growth factor (uncompensated)
    "CRRA": 1.0,  # Coefficient of relative risk aversion
    "kLogInitMean": -3.0,  # Mean of initial log normalized assets
    "kLogInitStd": 0.0,  # Standard deviation of initial log normalized assets
}


class TractableConsumerType(AgentType):
    """
    Parameters
    ----------
    Same as AgentType
    """

    time_inv_ = [
        "DiscFac",
        "Rfree",
        "CRRA",
        "PermGroFacCmp",
        "UnempPrb",
        "PFMPC",
        "Rnrm",
        "Beth",
        "mLowerBnd",
        "mUpperBnd",
    ]
    shock_vars_ = ["eState"]
    state_vars = ["bNrm", "mNrm", "aNrm"]
    poststate_vars = ["aNrm", "eState"]  # For simulation
    default_ = {"params": init_tractable, "solver": add_to_stable_arm_points}

    def pre_solve(self):
        """
        Calculates all of the solution objects that can be obtained before con-
        ducting the backshooting routine, including the target levels, the per-
        fect foresight solution, (marginal) consumption at m=0, and the small
        perturbations around the steady state.

        TODO: This should probably all be moved to a constructor function.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        CRRA = self.CRRA
        UnempPrb = self.UnempPrb
        DiscFac = self.DiscFac
        PermGroFac = self.PermGroFac
        Rfree = self.Rfree

        # Define utility functions
        def uPP(x):
            return utilityPP(x, rho=CRRA)

        def uPPP(x):
            return utilityPPP(x, rho=CRRA)

        def uPPPP(x):
            return utilityPPPP(x, rho=CRRA)

        # Define some useful constants from model primitives
        PermGroFacCmp = PermGroFac / (
            1.0 - UnempPrb
        )  # "uncertainty compensated" wage growth factor
        Rnrm = (
            Rfree / PermGroFacCmp
        )  # net interest factor (Rfree normalized by wage growth)
        PFMPC = 1.0 - (Rfree ** (-1.0)) * (Rfree * DiscFac) ** (
            1.0 / CRRA
        )  # MPC for a perfect forsight consumer
        Beth = Rnrm * DiscFac * PermGroFacCmp ** (1.0 - CRRA)

        # Verify that this consumer is impatient
        PatFacGrowth = (Rfree * DiscFac) ** (1.0 / CRRA) / PermGroFacCmp
        PatFacReturn = (Rfree * DiscFac) ** (1.0 / CRRA) / Rfree
        if PatFacReturn >= 1.0:
            raise Exception("Employed consumer not return impatient, cannot solve!")
        if PatFacGrowth >= 1.0:
            raise Exception("Employed consumer not growth impatient, cannot solve!")

        # Find target money and consumption
        # See TBS Appendix "B.2 A Target Always Exists When Human Wealth Is Infinite"
        Pi = (1 + (PatFacGrowth ** (-CRRA) - 1.0) / UnempPrb) ** (1 / CRRA)
        h = 1.0 / (1.0 - PermGroFac / Rfree)
        zeta = Rnrm * PFMPC * Pi  # See TBS Appendix "C The Exact Formula for target m"
        mTarg = 1.0 + (Rfree / (PermGroFacCmp + zeta * PermGroFacCmp - Rfree))
        cTarg = (1.0 - Rnrm ** (-1.0)) * mTarg + Rnrm ** (-1.0)
        mTargU = (mTarg - cTarg) * Rnrm
        cTargU = mTargU * PFMPC
        SSperturbance = mTarg * 0.1

        # Find the MPC, MMPC, and MMMPC at the target
        def mpcTargFixedPointFunc(k):
            return k * uPP(cTarg) - Beth * (
                (1.0 - UnempPrb) * (1.0 - k) * k * Rnrm * uPP(cTarg)
                + PFMPC * UnempPrb * (1.0 - k) * Rnrm * uPP(cTargU)
            )

        MPCtarg = newton(mpcTargFixedPointFunc, 0)

        def mmpcTargFixedPointFunc(kk):
            return (
                kk * uPP(cTarg)
                + MPCtarg**2.0 * uPPP(cTarg)
                - Beth
                * (
                    -(1.0 - UnempPrb) * MPCtarg * kk * Rnrm * uPP(cTarg)
                    + (1.0 - UnempPrb)
                    * (1.0 - MPCtarg) ** 2.0
                    * kk
                    * Rnrm**2.0
                    * uPP(cTarg)
                    - PFMPC * UnempPrb * kk * Rnrm * uPP(cTargU)
                    + (1.0 - UnempPrb)
                    * (1.0 - MPCtarg) ** 2.0
                    * MPCtarg**2.0
                    * Rnrm**2.0
                    * uPPP(cTarg)
                    + PFMPC**2.0
                    * UnempPrb
                    * (1.0 - MPCtarg) ** 2.0
                    * Rnrm**2.0
                    * uPPP(cTargU)
                )
            )

        MMPCtarg = newton(mmpcTargFixedPointFunc, 0)

        def mmmpcTargFixedPointFunc(kkk):
            return (
                kkk * uPP(cTarg)
                + 3 * MPCtarg * MMPCtarg * uPPP(cTarg)
                + MPCtarg**3 * uPPPP(cTarg)
                - Beth
                * (
                    -(1 - UnempPrb) * MPCtarg * kkk * Rnrm * uPP(cTarg)
                    - 3
                    * (1 - UnempPrb)
                    * (1 - MPCtarg)
                    * MMPCtarg**2
                    * Rnrm**2
                    * uPP(cTarg)
                    + (1 - UnempPrb) * (1 - MPCtarg) ** 3 * kkk * Rnrm**3 * uPP(cTarg)
                    - PFMPC * UnempPrb * kkk * Rnrm * uPP(cTargU)
                    - 3
                    * (1 - UnempPrb)
                    * (1 - MPCtarg)
                    * MPCtarg**2
                    * MMPCtarg
                    * Rnrm**2
                    * uPPP(cTarg)
                    + 3
                    * (1 - UnempPrb)
                    * (1 - MPCtarg) ** 3
                    * MPCtarg
                    * MMPCtarg
                    * Rnrm**3
                    * uPPP(cTarg)
                    - 3
                    * PFMPC**2
                    * UnempPrb
                    * (1 - MPCtarg)
                    * MMPCtarg
                    * Rnrm**2
                    * uPPP(cTargU)
                    + (1 - UnempPrb)
                    * (1 - MPCtarg) ** 3
                    * MPCtarg**3
                    * Rnrm**3
                    * uPPPP(cTarg)
                    + PFMPC**3 * UnempPrb * (1 - MPCtarg) ** 3 * Rnrm**3 * uPPPP(cTargU)
                )
            )

        MMMPCtarg = newton(mmmpcTargFixedPointFunc, 0)

        # Find the MPC at m=0
        def f_temp(k):
            return (
                Beth
                * Rnrm
                * UnempPrb
                * (PFMPC * Rnrm * ((1.0 - k) / k)) ** (-CRRA - 1.0)
                * PFMPC
            )

        def mpcAtZeroFixedPointFunc(k):
            return k - f_temp(k) / (1 + f_temp(k))

        # self.MPCmax = newton(mpcAtZeroFixedPointFunc,0.5)
        MPCmax = brentq(
            mpcAtZeroFixedPointFunc, PFMPC, 0.99, xtol=0.00000001, rtol=0.00000001
        )

        # Make the initial list of Euler points: target and perturbation to either side
        mNrm_list = [
            mTarg - SSperturbance,
            mTarg,
            mTarg + SSperturbance,
        ]
        c_perturb_lo = (
            cTarg
            - SSperturbance * MPCtarg
            + 0.5 * SSperturbance**2.0 * MMPCtarg
            - (1.0 / 6.0) * SSperturbance**3.0 * MMMPCtarg
        )
        c_perturb_hi = (
            cTarg
            + SSperturbance * MPCtarg
            + 0.5 * SSperturbance**2.0 * MMPCtarg
            + (1.0 / 6.0) * SSperturbance**3.0 * MMMPCtarg
        )
        cNrm_list = [c_perturb_lo, cTarg, c_perturb_hi]
        MPC_perturb_lo = (
            MPCtarg - SSperturbance * MMPCtarg + 0.5 * SSperturbance**2.0 * MMMPCtarg
        )
        MPC_perturb_hi = (
            MPCtarg + SSperturbance * MMPCtarg + 0.5 * SSperturbance**2.0 * MMMPCtarg
        )
        MPC_list = [MPC_perturb_lo, MPCtarg, MPC_perturb_hi]

        # Set bounds for money (stable arm construction stops when these are exceeded)
        mLowerBnd = 1.0
        mUpperBnd = 2.0 * mTarg

        # Make the terminal period solution
        solution_terminal = TractableConsumerSolution(
            mNrm_list=mNrm_list, cNrm_list=cNrm_list, MPC_list=MPC_list
        )

        # Make two linear steady state functions
        cSSfunc = lambda m: m * ((Rnrm * PFMPC * Pi) / (1.0 + Rnrm * PFMPC * Pi))
        mSSfunc = lambda m: (PermGroFacCmp / Rfree) + (1.0 - PermGroFacCmp / Rfree) * m

        # Put all the parameters into self
        new_params = {
            "PermGroFacCmp": PermGroFacCmp,
            "Rnrm": Rnrm,
            "PFMPC": PFMPC,
            "Beth": Beth,
            "PatFacGrowth": PatFacGrowth,
            "Pi": Pi,
            "h": h,
            "zeta": zeta,
            "mTarg": mTarg,
            "cTarg": cTarg,
            "mTargU": mTargU,
            "cTargU": cTargU,
            "SSperturbance": SSperturbance,
            "MPCtarg": MPCtarg,
            "MMPCtarg": MMPCtarg,
            "MMMPCtarg": MMMPCtarg,
            "MPCmax": MPCmax,
            "mLowerBnd": mLowerBnd,
            "mUpperBnd": mUpperBnd,
            "solution_terminal": solution_terminal,
            "cSSfunc": cSSfunc,
            "mSSfunc": mSSfunc,
        }
        self.assign_parameters(**new_params)

    def post_solve(self):
        """
        This method adds consumption at m=0 to the list of stable arm points,
        then constructs the consumption function as a cubic interpolation over
        those points.  Should be run after the backshooting routine is complete.

        Parameters
        ----------
        none

        Returns
        -------
        none
        """
        # Add bottom point to the stable arm points
        self.solution[0].mNrm_list.insert(0, 0.0)
        self.solution[0].cNrm_list.insert(0, 0.0)
        self.solution[0].MPC_list.insert(0, self.MPCmax)

        # Construct an interpolation of the consumption function from the stable arm points
        self.solution[0].cFunc = CubicInterp(
            self.solution[0].mNrm_list,
            self.solution[0].cNrm_list,
            self.solution[0].MPC_list,
            self.PFMPC * (self.h - 1.0),
            self.PFMPC,
        )
        self.solution[0].cFunc_U = LinearInterp([0.0, 1.0], [0.0, self.PFMPC])

    def sim_birth(self, which_agents):
        """
        Makes new consumers for the given indices.  Initialized variables include aNrm, as
        well as time variables t_age and t_cycle.  Normalized assets are drawn from a lognormal
        distributions given by aLvlInitMean and aLvlInitStd.

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
            self.kLogInitMean,
            sigma=self.kLogInitStd,
            seed=self.RNG.integers(0, 2**31 - 1),
        ).draw(N)
        self.shocks["eState"] = np.zeros(self.AgentCount)  # Initialize shock array
        # Agents are born employed
        self.shocks["eState"][which_agents] = 1.0
        # How many periods since each agent was born
        self.t_age[which_agents] = 0
        self.t_cycle[which_agents] = (
            0  # Which period of the cycle each agent is currently in
        )
        return None

    def sim_death(self):
        """
        Trivial function that returns boolean array of all False, as there is no death.

        Parameters
        ----------
        None

        Returns
        -------
        which_agents : np.array(bool)
            Boolean array of size AgentCount indicating which agents die.
        """
        # Nobody dies in this model
        which_agents = np.zeros(self.AgentCount, dtype=bool)
        return which_agents

    def get_shocks(self):
        """
        Determine which agents switch from employment to unemployment.  All unemployed agents remain
        unemployed until death.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        employed = self.shocks["eState"] == 1.0
        N = int(np.sum(employed))
        newly_unemployed = Bernoulli(
            self.UnempPrb, seed=self.RNG.integers(0, 2**31 - 1)
        ).draw(N)
        self.shocks["eState"][employed] = 1.0 - newly_unemployed

    def transition(self):
        """
        Calculate market resources for all agents this period.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        bNrmNow = self.Rfree * self.state_prev["aNrm"]
        EmpNow = self.shocks["eState"] == 1.0
        bNrmNow[EmpNow] /= self.PermGroFacCmp
        mNrmNow = bNrmNow + self.shocks["eState"]

        return bNrmNow, mNrmNow

    def get_controls(self):
        """
        Calculate consumption for each agent this period.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        employed = self.shocks["eState"] == 1.0
        unemployed = np.logical_not(employed)
        cNrmNow = np.zeros(self.AgentCount)
        cNrmNow[employed] = self.solution[0].cFunc(self.state_now["mNrm"][employed])
        cNrmNow[unemployed] = self.solution[0].cFunc_U(
            self.state_now["mNrm"][unemployed]
        )
        self.controls["cNrm"] = cNrmNow

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
        self.state_now["aNrm"] = self.state_now["mNrm"] - self.controls["cNrm"]
        return None
