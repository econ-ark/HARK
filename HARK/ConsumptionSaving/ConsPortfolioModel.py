"""
This file contains classes and functions for representing, solving, and simulating
agents who must allocate their resources among consumption, saving in a risk-free
asset (with a low return), and saving in a risky asset (with higher average return).
"""
from copy import deepcopy

import numpy as np
from scipy.optimize import minimize_scalar

from HARK import (
    MetricObject,
    NullFunc,
    AgentType,
    make_one_period_oo_solver,
)  # Basic HARK features
from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,  # PortfolioConsumerType inherits from it
    utility,  # CRRA utility function
    utility_inv,  # Inverse CRRA utility function
    utilityP,  # CRRA marginal utility function
    utility_invP,  # Derivative of inverse CRRA utility function
    utilityP_inv,  # Inverse CRRA marginal utility function
    init_idiosyncratic_shocks,  # Baseline dictionary to build on
)
from HARK.ConsumptionSaving.ConsRiskyAssetModel import RiskyAssetConsumerType
from HARK.distribution import calc_expectation
from HARK.interpolation import (
    LinearInterp,  # Piecewise linear interpolation
    CubicInterp,  # Piecewise cubic interpolation
    LinearInterpOnInterp1D,  # Interpolator over 1D interpolations
    BilinearInterp,  # 2D interpolator
    ConstantFunction,  # Interpolator-like class that returns constant value
    IdentityFunction,  # Interpolator-like class that returns one of its arguments
    ValueFuncCRRA,
    MargValueFuncCRRA,
)


# Define a class to represent the single period solution of the portfolio choice problem
class PortfolioSolution(MetricObject):
    """
    A class for representing the single period solution of the portfolio choice model.

    Parameters
    ----------
    cFuncAdj : Interp1D
        Consumption function over normalized market resources when the agent is able
        to adjust their portfolio shares.
    ShareFuncAdj : Interp1D
        Risky share function over normalized market resources when the agent is able
        to adjust their portfolio shares.
    vFuncAdj : ValueFuncCRRA
        Value function over normalized market resources when the agent is able to
        adjust their portfolio shares.
    vPfuncAdj : MargValueFuncCRRA
        Marginal value function over normalized market resources when the agent is able
        to adjust their portfolio shares.
    cFuncFxd : Interp2D
        Consumption function over normalized market resources and risky portfolio share
        when the agent is NOT able to adjust their portfolio shares, so they are fixed.
    ShareFuncFxd : Interp2D
        Risky share function over normalized market resources and risky portfolio share
        when the agent is NOT able to adjust their portfolio shares, so they are fixed.
        This should always be an IdentityFunc, by definition.
    vFuncFxd : ValueFuncCRRA
        Value function over normalized market resources and risky portfolio share when
        the agent is NOT able to adjust their portfolio shares, so they are fixed.
    dvdmFuncFxd : MargValueFuncCRRA
        Marginal value of mNrm function over normalized market resources and risky
        portfolio share when the agent is NOT able to adjust their portfolio shares,
        so they are fixed.
    dvdsFuncFxd : MargValueFuncCRRA
        Marginal value of Share function over normalized market resources and risky
        portfolio share when the agent is NOT able to adjust their portfolio shares,
        so they are fixed.
    aGrid: np.array
        End-of-period-assets grid used to find the solution.
    Share_adj: np.array
        Optimal portfolio share associated with each aGrid point.
    EndOfPrddvda_adj: np.array
        Marginal value of end-of-period resources associated with each aGrid
        point.
    ShareGrid: np.array
        Grid for the portfolio share that is used to solve the model.
    EndOfPrddvda_fxd: np.array
        Marginal value of end-of-period resources associated with each
        (aGrid x sharegrid) combination, for the agent who can not adjust his
        portfolio.
    AdjustPrb: float
        Probability that the agent will be able to adjust his portfolio
        next period.
    """

    distance_criteria = ["vPfuncAdj"]

    def __init__(
        self,
        cFuncAdj=None,
        ShareFuncAdj=None,
        vFuncAdj=None,
        vPfuncAdj=None,
        cFuncFxd=None,
        ShareFuncFxd=None,
        vFuncFxd=None,
        dvdmFuncFxd=None,
        dvdsFuncFxd=None,
        aGrid=None,
        Share_adj=None,
        EndOfPrddvda_adj=None,
        ShareGrid=None,
        EndOfPrddvda_fxd=None,
        EndOfPrddvds_fxd=None,
        AdjPrb=None,
    ):

        # Change any missing function inputs to NullFunc
        if cFuncAdj is None:
            cFuncAdj = NullFunc()
        if cFuncFxd is None:
            cFuncFxd = NullFunc()
        if ShareFuncAdj is None:
            ShareFuncAdj = NullFunc()
        if ShareFuncFxd is None:
            ShareFuncFxd = NullFunc()
        if vFuncAdj is None:
            vFuncAdj = NullFunc()
        if vFuncFxd is None:
            vFuncFxd = NullFunc()
        if vPfuncAdj is None:
            vPfuncAdj = NullFunc()
        if dvdmFuncFxd is None:
            dvdmFuncFxd = NullFunc()
        if dvdsFuncFxd is None:
            dvdsFuncFxd = NullFunc()

        # Set attributes of self
        self.cFuncAdj = cFuncAdj
        self.cFuncFxd = cFuncFxd
        self.ShareFuncAdj = ShareFuncAdj
        self.ShareFuncFxd = ShareFuncFxd
        self.vFuncAdj = vFuncAdj
        self.vFuncFxd = vFuncFxd
        self.vPfuncAdj = vPfuncAdj
        self.dvdmFuncFxd = dvdmFuncFxd
        self.dvdsFuncFxd = dvdsFuncFxd
        self.aGrid = aGrid
        self.Share_adj = Share_adj
        self.EndOfPrddvda_adj = EndOfPrddvda_adj
        self.ShareGrid = ShareGrid
        self.EndOfPrddvda_fxd = EndOfPrddvda_fxd
        self.EndOfPrddvds_fxd = EndOfPrddvds_fxd
        self.AdjPrb = AdjPrb


class PortfolioConsumerType(RiskyAssetConsumerType):
    """
    A consumer type with a portfolio choice. This agent type has log-normal return
    factors. Their problem is defined by a coefficient of relative risk aversion,
    intertemporal discount factor, risk-free interest factor, and time sequences of
    permanent income growth rate, survival probability, and permanent and transitory
    income shock standard deviations (in logs).  The agent may also invest in a risky
    asset, which has a higher average return than the risk-free asset.  He *might*
    have age-varying beliefs about the risky-return; if he does, then "true" values
    of the risky asset's return distribution must also be specified.
    """

    time_inv_ = deepcopy(RiskyAssetConsumerType.time_inv_)
    time_inv_ = time_inv_ + ["AdjustPrb", "DiscreteShareBool", "ApproxShareBool"]

    def __init__(self, verbose=False, quiet=False, **kwds):
        params = init_portfolio.copy()
        params.update(kwds)
        kwds = params

        # Initialize a basic consumer type
        RiskyAssetConsumerType.__init__(self, verbose=verbose, quiet=quiet, **kwds)

        # Set the solver for the portfolio model, and update various constructed attributes
        if self.IndepDstnBool:
            if self.DiscreteShareBool:
                solver = ConsPortfolioDiscreteSolver
            else:
                solver = ConsPortfolioSolver
        else:
            solver = ConsPortfolioJointDistSolver
        self.solve_one_period = make_one_period_oo_solver(solver)

        self.update()

    def pre_solve(self):
        AgentType.pre_solve(self)
        self.update_solution_terminal()

    def update(self):

        RiskyAssetConsumerType.update(self)
        self.update_ShareGrid()
        self.update_ShareLimit()

    def update_solution_terminal(self):
        """
        Solves the terminal period of the portfolio choice problem.  The solution is
        trivial, as usual: consume all market resources, and put nothing in the risky
        asset (because you have nothing anyway).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Consume all market resources: c_T = m_T
        cFuncAdj_terminal = LinearInterp([0.0, 1.0], [0.0, 1.0])
        cFuncFxd_terminal = IdentityFunction(i_dim=0, n_dims=2)

        # Risky share is irrelevant-- no end-of-period assets; set to zero
        ShareFuncAdj_terminal = ConstantFunction(0.0)
        ShareFuncFxd_terminal = IdentityFunction(i_dim=1, n_dims=2)

        # Value function is simply utility from consuming market resources
        vFuncAdj_terminal = ValueFuncCRRA(cFuncAdj_terminal, self.CRRA)
        vFuncFxd_terminal = ValueFuncCRRA(cFuncFxd_terminal, self.CRRA)

        # Marginal value of market resources is marg utility at the consumption function
        vPfuncAdj_terminal = MargValueFuncCRRA(cFuncAdj_terminal, self.CRRA)
        dvdmFuncFxd_terminal = MargValueFuncCRRA(cFuncFxd_terminal, self.CRRA)
        dvdsFuncFxd_terminal = ConstantFunction(
            0.0
        )  # No future, no marg value of Share

        # Construct the terminal period solution
        self.solution_terminal = PortfolioSolution(
            cFuncAdj=cFuncAdj_terminal,
            ShareFuncAdj=ShareFuncAdj_terminal,
            vFuncAdj=vFuncAdj_terminal,
            vPfuncAdj=vPfuncAdj_terminal,
            cFuncFxd=cFuncFxd_terminal,
            ShareFuncFxd=ShareFuncFxd_terminal,
            vFuncFxd=vFuncFxd_terminal,
            dvdmFuncFxd=dvdmFuncFxd_terminal,
            dvdsFuncFxd=dvdsFuncFxd_terminal,
        )

        self.solution_terminal.ShareEndOfPrdFunc = ShareFuncAdj_terminal

    def update_ShareGrid(self):
        """
        Creates the attribute ShareGrid as an evenly spaced grid on [0.,1.], using
        the primitive parameter ShareCount.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.ShareGrid = np.linspace(0.0, 1.0, self.ShareCount)
        self.add_to_time_inv("ShareGrid")

    def update_ShareLimit(self):
        """
        Creates the attribute ShareLimit, representing the limiting lower bound of
        risky portfolio share as mNrm goes to infinity.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if "RiskyDstn" in self.time_vary:
            self.ShareLimit = []
            for t in range(self.T_cycle):
                RiskyDstn = self.RiskyDstn[t]
                temp_f = lambda s: -((1.0 - self.CRRA) ** -1) * np.dot(
                    (self.Rfree + s * (RiskyDstn.X - self.Rfree)) ** (1.0 - self.CRRA),
                    RiskyDstn.pmf,
                )
                SharePF = minimize_scalar(temp_f, bounds=(0.0, 1.0), method="bounded").x
                self.ShareLimit.append(SharePF)
            self.add_to_time_vary("ShareLimit")

        else:
            RiskyDstn = self.RiskyDstn
            temp_f = lambda s: -((1.0 - self.CRRA) ** -1) * np.dot(
                (self.Rfree + s * (RiskyDstn.X - self.Rfree)) ** (1.0 - self.CRRA),
                RiskyDstn.pmf,
            )
            SharePF = minimize_scalar(temp_f, bounds=(0.0, 1.0), method="bounded").x
            self.ShareLimit = SharePF
            self.add_to_time_inv("ShareLimit")

    def get_Rfree(self):
        """
        Calculates realized return factor for each agent, using the attributes Rfree,
        RiskyNow, and ShareNow.  This method is a bit of a misnomer, as the return
        factor is not riskless, but would more accurately be labeled as Rport.  However,
        this method makes the portfolio model compatible with its parent class.

        Parameters
        ----------
        None

        Returns
        -------
        Rport : np.array
            Array of size AgentCount with each simulated agent's realized portfolio
            return factor.  Will be used by get_states() to calculate mNrmNow, where it
            will be mislabeled as "Rfree".
        """
        Rport = (
            self.controls["Share"] * self.shocks["Risky"]
            + (1.0 - self.controls["Share"]) * self.Rfree
        )
        self.Rport = Rport
        return Rport

    def initialize_sim(self):
        """
        Initialize the state of simulation attributes.  Simply calls the same method
        for IndShockConsumerType, then sets the type of AdjustNow to bool.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # these need to be set because "post states",
        # but are a control variable and shock, respectively
        self.controls["Share"] = np.zeros(self.AgentCount)
        RiskyAssetConsumerType.initialize_sim(self)

    def sim_birth(self, which_agents):
        """
        Create new agents to replace ones who have recently died; takes draws of
        initial aNrm and pLvl, as in ConsIndShockModel, then sets Share and Adjust
        to zero as initial values.
        Parameters
        ----------
        which_agents : np.array
            Boolean array of size AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        """
        IndShockConsumerType.sim_birth(self, which_agents)

        self.controls["Share"][which_agents] = 0
        # here a shock is being used as a 'post state'
        self.shocks["Adjust"][which_agents] = False

    def get_controls(self):
        """
        Calculates consumption cNrmNow and risky portfolio share ShareNow using
        the policy functions in the attribute solution.  These are stored as attributes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        ShareNow = np.zeros(self.AgentCount) + np.nan

        # Loop over each period of the cycle, getting controls separately depending on "age"
        for t in range(self.T_cycle):
            these = t == self.t_cycle

            # Get controls for agents who *can* adjust their portfolio share
            those = np.logical_and(these, self.shocks["Adjust"])
            cNrmNow[those] = self.solution[t].cFuncAdj(self.state_now["mNrm"][those])
            ShareNow[those] = self.solution[t].ShareFuncAdj(
                self.state_now["mNrm"][those]
            )

            # Get Controls for agents who *can't* adjust their portfolio share
            those = np.logical_and(these, np.logical_not(self.shocks["Adjust"]))
            cNrmNow[those] = self.solution[t].cFuncFxd(
                self.state_now["mNrm"][those], ShareNow[those]
            )
            ShareNow[those] = self.solution[t].ShareFuncFxd(
                self.state_now["mNrm"][those], ShareNow[those]
            )

        # Store controls as attributes of self
        self.controls["cNrm"] = cNrmNow
        self.controls["Share"] = ShareNow


class SequentialPortfolioConsumerType(PortfolioConsumerType):
    def __init__(self, verbose=False, quiet=False, **kwds):
        params = init_portfolio.copy()
        params.update(kwds)
        kwds = params

        # Initialize a basic consumer type
        PortfolioConsumerType.__init__(self, verbose=verbose, quiet=quiet, **kwds)

        # Set the solver for the portfolio model, and update various constructed attributes
        self.solve_one_period = make_one_period_oo_solver(ConsSequentialPortfolioSolver)


class ConsPortfolioSolver(MetricObject):
    """
    Define an object-oriented one period solver.
    Solve the one period problem for a portfolio-choice consumer.
    This solver is used when the income and risky return shocks
    are independent and the allowed optimal share is continuous.

    Parameters
    ----------
    solution_next : PortfolioSolution
        Solution to next period's problem.
    ShockDstn : [np.array]
        List with four arrays: discrete probabilities, permanent income shocks,
        transitory income shocks, and risky returns.  This is only used if the
        input IndepDstnBool is False, indicating that income and return distributions
        can't be assumed to be independent.
    IncShkDstn : distribution.Distribution
        Discrete distribution of permanent income shocks
        and transitory income shocks.  This is only used if the input IndepDsntBool
        is True, indicating that income and return distributions are independent.
    RiskyDstn : [np.array]
        List with two arrays: discrete probabilities and risky asset returns. This
        is only used if the input IndepDstnBool is True, indicating that income
        and return distributions are independent.
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
        period with.  In this model, it is *required* to be zero.
    aXtraGrid: np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    ShareGrid : np.array
        Array of risky portfolio shares on which to define the interpolation
        of the consumption function when Share is fixed.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    AdjustPrb : float
        Probability that the agent will be able to update his portfolio share.
    DiscreteShareBool : bool
        Indicator for whether risky portfolio share should be optimized on the
        continuous [0,1] interval using the FOC (False), or instead only selected
        from the discrete set of values in ShareGrid (True).  If True, then
        vFuncBool must also be True.
    ShareLimit : float
        Limiting lower bound of risky portfolio share as mNrm approaches infinity.
    IndepDstnBool : bool
        Indicator for whether the income and risky return distributions are in-
        dependent of each other, which can speed up the expectations step.
    """

    def __init__(
        self,
        solution_next,
        ShockDstn,
        IncShkDstn,
        RiskyDstn,
        LivPrb,
        DiscFac,
        CRRA,
        Rfree,
        PermGroFac,
        BoroCnstArt,
        aXtraGrid,
        ShareGrid,
        vFuncBool,
        AdjustPrb,
        DiscreteShareBool,
        ShareLimit,
        IndepDstnBool,
        ApproxShareBool,
    ):
        """
        Constructor for portfolio choice problem solver.
        """

        self.solution_next = solution_next
        self.ShockDstn = ShockDstn
        self.IncShkDstn = IncShkDstn
        self.RiskyDstn = RiskyDstn
        self.LivPrb = LivPrb
        self.DiscFac = DiscFac
        self.CRRA = CRRA
        self.Rfree = Rfree
        self.PermGroFac = PermGroFac
        self.BoroCnstArt = BoroCnstArt
        self.aXtraGrid = aXtraGrid
        self.ShareGrid = ShareGrid
        self.vFuncBool = vFuncBool
        self.AdjustPrb = AdjustPrb
        self.DiscreteShareBool = DiscreteShareBool
        self.ShareLimit = ShareLimit
        self.IndepDstnBool = IndepDstnBool
        self.ApproxShareBool = ApproxShareBool

        # Make sure the individual is liquidity constrained.  Allowing a consumer to
        # borrow *and* invest in an asset with unbounded (negative) returns is a bad mix.
        if BoroCnstArt != 0.0:
            raise ValueError("PortfolioConsumerType must have BoroCnstArt=0.0!")

        # Make sure that if risky portfolio share is optimized only discretely, then
        # the value function is also constructed (else this task would be impossible).
        if DiscreteShareBool and (not vFuncBool):
            raise ValueError(
                "PortfolioConsumerType requires vFuncBool to be True when DiscreteShareBool is True!"
            )

        self.def_utility_funcs()

    def def_utility_funcs(self):
        """
        Define temporary functions for utility and its derivative and inverse
        """

        self.u = lambda x: utility(x, self.CRRA)
        self.uP = lambda x: utilityP(x, self.CRRA)
        self.uPinv = lambda x: utilityP_inv(x, self.CRRA)
        self.uinv = lambda x: utility_inv(x, self.CRRA)
        self.uinvP = lambda x: utility_invP(x, self.CRRA)

    def set_and_update_values(self):
        """
        Unpacks some of the inputs (and calculates simple objects based on them),
        storing the results in self for use by other methods.
        """

        # Unpack next period's solution
        self.vPfuncAdj_next = self.solution_next.vPfuncAdj
        self.dvdmFuncFxd_next = self.solution_next.dvdmFuncFxd
        self.dvdsFuncFxd_next = self.solution_next.dvdsFuncFxd
        self.vFuncAdj_next = self.solution_next.vFuncAdj
        self.vFuncFxd_next = self.solution_next.vFuncFxd

        # Unpack the shock distribution
        TranShks_next = self.IncShkDstn.X[1]

        # Flag for whether the natural borrowing constraint is zero
        self.zero_bound = np.min(TranShks_next) == 0.0

    def prepare_to_solve(self):
        """
        Perform preparatory work.
        """

        self.set_and_update_values()

    def prepare_to_calc_EndOfPrdvP(self):
        """
        Prepare to calculate end-of-period marginal values by creating an array
        of market resources that the agent could have next period, considering
        the grid of end-of-period assets and the distribution of shocks he might
        experience next period.
        """

        # Unpack the shock distribution
        Risky_next = self.RiskyDstn.X
        RiskyMax = np.max(Risky_next)
        RiskyMin = np.min(Risky_next)

        # bNrm represents R*a, balances after asset return shocks but before income.
        # This just uses the highest risky return as a rough shifter for the aXtraGrid.
        if self.zero_bound:
            self.aNrmGrid = self.aXtraGrid
            self.bNrmGrid = np.insert(
                RiskyMax * self.aXtraGrid, 0, RiskyMin * self.aXtraGrid[0]
            )
        else:
            # Add an asset point at exactly zero
            self.aNrmGrid = np.insert(self.aXtraGrid, 0, 0.0)
            self.bNrmGrid = RiskyMax * np.insert(self.aXtraGrid, 0, 0.0)

        # Get grid and shock sizes, for easier indexing
        self.aNrmCount = self.aNrmGrid.size
        self.ShareCount = self.ShareGrid.size

        # Make tiled arrays to calculate future realizations of mNrm and Share when integrating over IncShkDstn
        self.bNrmNext, self.ShareNext = np.meshgrid(
            self.bNrmGrid, self.ShareGrid, indexing="ij"
        )

    def m_nrm_next(self, shocks, b_nrm_next):
        """
        Calculate future realizations of market resources
        """

        return b_nrm_next / (shocks[0] * self.PermGroFac) + shocks[1]

    def calc_EndOfPrdvP(self):
        """
        Calculate end-of-period marginal value of assets and shares at each point
        in aNrm and ShareGrid. Does so by taking expectation of next period marginal
        values across income and risky return shocks.
        """

        def dvdb_dist(shocks, b_nrm, Share_next):
            """
            Evaluate realizations of marginal value of market resources next period
            """

            mNrm_next = self.m_nrm_next(shocks, b_nrm)

            dvdmAdj_next = self.vPfuncAdj_next(mNrm_next)
            if self.AdjustPrb < 1.0:
                dvdmFxd_next = self.dvdmFuncFxd_next(mNrm_next, Share_next)
                # Combine by adjustment probability
                dvdm_next = (
                    self.AdjustPrb * dvdmAdj_next
                    + (1.0 - self.AdjustPrb) * dvdmFxd_next
                )
            else:  # Don't bother evaluating if there's no chance that portfolio share is fixed
                dvdm_next = dvdmAdj_next

            return (shocks[0] * self.PermGroFac) ** (-self.CRRA) * dvdm_next

        def dvds_dist(shocks, b_nrm, Share_next):
            """
            Evaluate realizations of marginal value of risky share next period
            """

            mNrm_next = self.m_nrm_next(shocks, b_nrm)
            # No marginal value of Share if it's a free choice!
            dvdsAdj_next = np.zeros_like(mNrm_next)
            if self.AdjustPrb < 1.0:
                dvdsFxd_next = self.dvdsFuncFxd_next(mNrm_next, Share_next)
                # Combine by adjustment probability
                dvds_next = (
                    self.AdjustPrb * dvdsAdj_next
                    + (1.0 - self.AdjustPrb) * dvdsFxd_next
                )
            else:  # Don't bother evaluating if there's no chance that portfolio share is fixed
                dvds_next = dvdsAdj_next

            return (shocks[0] * self.PermGroFac) ** (1.0 - self.CRRA) * dvds_next

        # Calculate intermediate marginal value of bank balances by taking expectations over income shocks
        dvdb_intermed = calc_expectation(
            self.IncShkDstn, dvdb_dist, self.bNrmNext, self.ShareNext
        )
        # calc_expectation returns one additional "empty" dimension, remove it
        # this line can be deleted when calc_expectation is fixed
        dvdb_intermed = dvdb_intermed[:, :, 0]
        dvdbNvrs_intermed = self.uPinv(dvdb_intermed)
        dvdbNvrsFunc_intermed = BilinearInterp(
            dvdbNvrs_intermed, self.bNrmGrid, self.ShareGrid
        )
        dvdbFunc_intermed = MargValueFuncCRRA(dvdbNvrsFunc_intermed, self.CRRA)

        # Calculate intermediate marginal value of risky portfolio share by taking expectations
        dvds_intermed = calc_expectation(
            self.IncShkDstn, dvds_dist, self.bNrmNext, self.ShareNext
        )
        # calc_expectation returns one additional "empty" dimension, remove it
        # this line can be deleted when calc_expectation is fixed
        dvds_intermed = dvds_intermed[:, :, 0]
        dvdsFunc_intermed = BilinearInterp(dvds_intermed, self.bNrmGrid, self.ShareGrid)

        # Make tiled arrays to calculate future realizations of bNrm and Share when integrating over RiskyDstn
        self.aNrm_tiled, self.ShareNext = np.meshgrid(
            self.aNrmGrid, self.ShareGrid, indexing="ij"
        )

        # Evaluate realizations of value and marginal value after asset returns are realized

        def EndOfPrddvda_dist(shock, a_nrm, Share_next):

            # Calculate future realizations of bank balances bNrm
            Rxs = shock - self.Rfree
            Rport = self.Rfree + Share_next * Rxs
            b_nrm_next = Rport * a_nrm

            return Rport * dvdbFunc_intermed(b_nrm_next, Share_next)

        def EndOfPrddvds_dist(shock, a_nrm, Share_next):

            # Calculate future realizations of bank balances bNrm
            Rxs = shock - self.Rfree
            Rport = self.Rfree + Share_next * Rxs
            b_nrm_next = Rport * a_nrm

            return Rxs * a_nrm * dvdbFunc_intermed(
                b_nrm_next, Share_next
            ) + dvdsFunc_intermed(b_nrm_next, Share_next)

        # Calculate end-of-period marginal value of assets by taking expectations
        self.EndOfPrddvda = (
            self.DiscFac
            * self.LivPrb
            * calc_expectation(
                self.RiskyDstn, EndOfPrddvda_dist, self.aNrm_tiled, self.ShareNext
            )
        )
        # calc_expectation returns one additional "empty" dimension, remove it
        # this line can be deleted when calc_expectation is fixed
        self.EndOfPrddvda = self.EndOfPrddvda[:, :, 0]
        self.EndOfPrddvdaNvrs = self.uPinv(self.EndOfPrddvda)

        # Calculate end-of-period marginal value of risky portfolio share by taking expectations
        self.EndOfPrddvds = (
            self.DiscFac
            * self.LivPrb
            * calc_expectation(
                self.RiskyDstn, EndOfPrddvds_dist, self.aNrm_tiled, self.ShareNext
            )
        )
        # calc_expectation returns one additional "empty" dimension, remove it
        # this line can be deleted when calc_expectation is fixed
        self.EndOfPrddvds = self.EndOfPrddvds[:, :, 0]

    def optimize_share(self):
        """
        Optimization of Share on continuous interval [0,1]
        """

        # For values of aNrm at which the agent wants to put more than 100% into risky asset, constrain them
        FOC_s = self.EndOfPrddvds
        # Initialize to putting everything in safe asset
        self.Share_now = np.zeros_like(self.aNrmGrid)
        self.cNrmAdj_now = np.zeros_like(self.aNrmGrid)
        # If agent wants to put more than 100% into risky asset, he is constrained
        constrained_top = FOC_s[:, -1] > 0.0
        # Likewise if he wants to put less than 0% into risky asset
        constrained_bot = FOC_s[:, 0] < 0.0
        self.Share_now[constrained_top] = 1.0
        if not self.zero_bound:
            # aNrm=0, so there's no way to "optimize" the portfolio
            self.Share_now[0] = 1.0
            # Consumption when aNrm=0 does not depend on Share
            self.cNrmAdj_now[0] = self.EndOfPrddvdaNvrs[0, -1]
            # Mark as constrained so that there is no attempt at optimization
            constrained_top[0] = True

        # Get consumption when share-constrained
        self.cNrmAdj_now[constrained_top] = self.EndOfPrddvdaNvrs[constrained_top, -1]
        self.cNrmAdj_now[constrained_bot] = self.EndOfPrddvdaNvrs[constrained_bot, 0]
        # For each value of aNrm, find the value of Share such that FOC-Share == 0.
        # This loop can probably be eliminated, but it's such a small step that it won't speed things up much.
        crossing = np.logical_and(FOC_s[:, 1:] <= 0.0, FOC_s[:, :-1] >= 0.0)
        for j in range(self.aNrmCount):
            if not (constrained_top[j] or constrained_bot[j]):
                idx = np.argwhere(crossing[j, :])[0][0]
                bot_s = self.ShareGrid[idx]
                top_s = self.ShareGrid[idx + 1]
                bot_f = FOC_s[j, idx]
                top_f = FOC_s[j, idx + 1]
                bot_c = self.EndOfPrddvdaNvrs[j, idx]
                top_c = self.EndOfPrddvdaNvrs[j, idx + 1]
                alpha = 1.0 - top_f / (top_f - bot_f)
                self.Share_now[j] = (1.0 - alpha) * bot_s + alpha * top_s
                self.cNrmAdj_now[j] = (1.0 - alpha) * bot_c + alpha * top_c

    def make_basic_solution(self):
        """
        Given end of period assets and end of period marginal values, construct
        the basic solution for this period.
        """

        # Calculate the endogenous mNrm gridpoints when the agent adjusts his portfolio
        self.mNrmAdj_now = self.aNrmGrid + self.cNrmAdj_now

        # Construct the consumption function when the agent can adjust
        cNrmAdj_now = np.insert(self.cNrmAdj_now, 0, 0.0)
        self.cFuncAdj_now = LinearInterp(
            np.insert(self.mNrmAdj_now, 0, 0.0), cNrmAdj_now
        )

        # Construct the marginal value (of mNrm) function when the agent can adjust
        self.vPfuncAdj_now = MargValueFuncCRRA(self.cFuncAdj_now, self.CRRA)

        # Construct the consumption function when the agent *can't* adjust the risky share, as well
        # as the marginal value of Share function
        cFuncFxd_by_Share = []
        dvdsFuncFxd_by_Share = []
        for j in range(self.ShareCount):
            cNrmFxd_temp = self.EndOfPrddvdaNvrs[:, j]
            mNrmFxd_temp = self.aNrmGrid + cNrmFxd_temp
            cFuncFxd_by_Share.append(
                LinearInterp(
                    np.insert(mNrmFxd_temp, 0, 0.0), np.insert(cNrmFxd_temp, 0, 0.0)
                )
            )
            dvdsFuncFxd_by_Share.append(
                LinearInterp(
                    np.insert(mNrmFxd_temp, 0, 0.0),
                    np.insert(self.EndOfPrddvds[:, j], 0, self.EndOfPrddvds[0, j]),
                )
            )
        self.cFuncFxd_now = LinearInterpOnInterp1D(cFuncFxd_by_Share, self.ShareGrid)
        self.dvdsFuncFxd_now = LinearInterpOnInterp1D(
            dvdsFuncFxd_by_Share, self.ShareGrid
        )

        # The share function when the agent can't adjust his portfolio is trivial
        self.ShareFuncFxd_now = IdentityFunction(i_dim=1, n_dims=2)

        # Construct the marginal value of mNrm function when the agent can't adjust his share
        self.dvdmFuncFxd_now = MargValueFuncCRRA(self.cFuncFxd_now, self.CRRA)

    def make_ShareFuncAdj(self):
        """
        Construct the risky share function when the agent can adjust
        """

        # Share function for mGrid

        if self.zero_bound:
            Share_lower_bound = self.ShareLimit
        else:
            Share_lower_bound = 1.0
        Share_now = np.insert(self.Share_now, 0, Share_lower_bound)
        self.ShareFuncAdj_now = LinearInterp(
            np.insert(self.mNrmAdj_now, 0, 0.0),
            Share_now,
            intercept_limit=self.ShareLimit,
            slope_limit=0.0,
        )

        # Share function on aGrid

        if self.zero_bound:
            aNrm_temp = np.append(0.0, self.aNrmGrid)
            share_temp = np.append(self.ShareLimit, self.Share_now)
        else:
            aNrm_temp = self.aNrmGrid
            share_temp = self.Share_now

        self.ShareEndOfPrdFunc = LinearInterp(
            aNrm_temp, share_temp, intercept_limit=self.ShareLimit, slope_limit=0.0
        )

    def make_share_func_approx(self):
        """
        Alternative share functions from linear approximation.
        """

        # get next period's consumption and share function
        cFunc_next = self.solution_next.cFuncAdj
        sFunc_next = self.solution_next.ShareEndOfPrdFunc

        def premium(shock):
            """
            Used to evaluate mean and variance of equity premium.
            """
            r_diff = shock - self.Rfree

            return r_diff, r_diff ** 2, r_diff ** 3

        prem_mean, prem_sqrd, prem_cube = calc_expectation(self.RiskyDstn, premium)

        def c_nrm_and_deriv(shocks, a_nrm):
            """
            Used to calculate expected consumption and MPC given today's savings,
            assuming that today's risky share is the same as it would be tomorrow
            with that same level of savings.
            """
            p_shk = shocks[0] * self.PermGroFac
            t_shk = shocks[1]
            share = sFunc_next(a_nrm)
            r_diff = shocks[2] - self.Rfree
            r_port = self.Rfree + r_diff * share
            m_nrm_next = a_nrm * r_port / p_shk + t_shk

            c_next, cP_next = cFunc_next.eval_with_derivative(m_nrm_next)

            return c_next, cP_next

        exp_c_values = calc_expectation(self.ShockDstn, c_nrm_and_deriv, self.aNrmGrid)

        exp_c_values = exp_c_values[:, :, 0]
        exp_c_next = exp_c_values[0]
        exp_cP_next = exp_c_values[1]

        MPC = exp_cP_next * self.aNrmGrid / exp_c_next

        # first order approximation
        approx_share = prem_mean / (self.CRRA * MPC * prem_sqrd)

        # clip at 0 and 1, although we know the Share limit we
        # want to see what the approximation would give us
        approx_share = np.clip(approx_share, 0, 1)

        self.ApproxFirstOrderShareFunc = LinearInterp(
            self.aNrmGrid,
            approx_share,
            intercept_limit=self.ShareLimit,
            slope_limit=0.0,
        )

        # second order approximation

        a = -self.CRRA * prem_cube * MPC ** 2 * (-self.CRRA - 1) / 2
        b = -self.CRRA * MPC * prem_sqrd
        c = prem_mean

        temp = np.sqrt(b ** 2 - 4 * a * c)

        roots = np.array([(-b + temp) / (2 * a), (-b - temp) / (2 * a)])
        roots[:, 0] = 1.0
        roots = np.where(
            np.logical_and(roots[0] >= 0, roots[0] <= 1), roots[0], roots[1]
        )
        roots = np.clip(roots, 0, 1)

        self.ApproxSecondOrderShareFunc = LinearInterp(
            self.aNrmGrid, roots, intercept_limit=self.ShareLimit, slope_limit=0.0,
        )

    def add_save_points(self):
        # This is a point at which (a,c,share) have consistent length. Take the
        # snapshot for storing the grid and values in the solution.
        self.save_points = {
            "a": deepcopy(self.aNrmGrid),
            "eop_dvda_adj": self.uP(self.cNrmAdj_now),
            "share_adj": deepcopy(self.Share_now),
            "share_grid": deepcopy(self.ShareGrid),
            "eop_dvda_fxd": self.uP(self.EndOfPrddvda),
            "eop_dvds_fxd": self.EndOfPrddvds,
        }

    def add_vFunc(self):
        """
        Creates the value function for this period and adds it to the solution.
        """

        self.make_EndOfPrdvFunc()
        self.make_vFunc()

    def make_EndOfPrdvFunc(self):
        """
        Construct the end-of-period value function for this period, storing it
        as an attribute of self for use by other methods.
        """

        def v_intermed_dist(shocks, b_nrm, Share_next):
            mNrm_next = self.m_nrm_next(shocks, b_nrm)

            vAdj_next = self.vFuncAdj_next(mNrm_next)
            if self.AdjustPrb < 1.0:
                vFxd_next = self.vFuncFxd_next(mNrm_next, Share_next)
                # Combine by adjustment probability
                v_next = self.AdjustPrb * vAdj_next + (1.0 - self.AdjustPrb) * vFxd_next
            else:  # Don't bother evaluating if there's no chance that portfolio share is fixed
                v_next = vAdj_next

            return (shocks[0] * self.PermGroFac) ** (1.0 - self.CRRA) * v_next

        # Calculate intermediate value by taking expectations over income shocks
        v_intermed = calc_expectation(
            self.IncShkDstn, v_intermed_dist, self.bNrmNext, self.ShareNext
        )
        # calc_expectation returns one additional "empty" dimension, remove it
        # this line can be deleted when calc_expectation is fixed
        v_intermed = v_intermed[:, :, 0]
        vNvrs_intermed = self.uinv(v_intermed)
        vNvrsFunc_intermed = BilinearInterp(
            vNvrs_intermed, self.bNrmGrid, self.ShareGrid
        )
        vFunc_intermed = ValueFuncCRRA(vNvrsFunc_intermed, self.CRRA)

        def EndOfPrdv_dist(shock, a_nrm, Share_next):

            # Calculate future realizations of bank balances bNrm
            Rxs = shock - self.Rfree
            Rport = self.Rfree + Share_next * Rxs
            b_nrm_next = Rport * a_nrm

            return vFunc_intermed(b_nrm_next, Share_next)

        # Calculate end-of-period value by taking expectations
        self.EndOfPrdv = (
            self.DiscFac
            * self.LivPrb
            * calc_expectation(
                self.RiskyDstn, EndOfPrdv_dist, self.aNrm_tiled, self.ShareNext
            )
        )
        # calc_expectation returns one additional "empty" dimension, remove it
        # this line can be deleted when calc_expectation is fixed
        self.EndOfPrdv = self.EndOfPrdv[:, :, 0]
        self.EndOfPrdvNvrs = self.uinv(self.EndOfPrdv)

    def make_vFunc(self):
        """
        Creates the value functions for this period, defined over market
        resources m when agent can adjust his portfolio, and over market
        resources and fixed share when agent can not adjust his portfolio.
        self must have the attribute EndOfPrdvFunc in order to execute.
        """

        # First, make an end-of-period value function over aNrm and Share
        EndOfPrdvNvrsFunc = BilinearInterp(
            self.EndOfPrdvNvrs, self.aNrmGrid, self.ShareGrid
        )
        EndOfPrdvFunc = ValueFuncCRRA(EndOfPrdvNvrsFunc, self.CRRA)

        # Construct the value function when the agent can adjust his portfolio
        mNrm_temp = self.aXtraGrid  # Just use aXtraGrid as our grid of mNrm values
        cNrm_temp = self.cFuncAdj_now(mNrm_temp)
        aNrm_temp = mNrm_temp - cNrm_temp
        Share_temp = self.ShareFuncAdj_now(mNrm_temp)
        v_temp = self.u(cNrm_temp) + EndOfPrdvFunc(aNrm_temp, Share_temp)
        vNvrs_temp = self.uinv(v_temp)
        vNvrsP_temp = self.uP(cNrm_temp) * self.uinvP(v_temp)
        vNvrsFuncAdj = CubicInterp(
            np.insert(mNrm_temp, 0, 0.0),  # x_list
            np.insert(vNvrs_temp, 0, 0.0),  # f_list
            np.insert(vNvrsP_temp, 0, vNvrsP_temp[0]),  # dfdx_list
        )
        # Re-curve the pseudo-inverse value function
        self.vFuncAdj_now = ValueFuncCRRA(vNvrsFuncAdj, self.CRRA)

        # Construct the value function when the agent *can't* adjust his portfolio
        mNrm_temp, Share_temp = np.meshgrid(self.aXtraGrid, self.ShareGrid)
        cNrm_temp = self.cFuncFxd_now(mNrm_temp, Share_temp)
        aNrm_temp = mNrm_temp - cNrm_temp
        v_temp = self.u(cNrm_temp) + EndOfPrdvFunc(aNrm_temp, Share_temp)
        vNvrs_temp = self.uinv(v_temp)
        vNvrsP_temp = self.uP(cNrm_temp) * self.uinvP(v_temp)
        vNvrsFuncFxd_by_Share = []
        for j in range(self.ShareCount):
            vNvrsFuncFxd_by_Share.append(
                CubicInterp(
                    np.insert(mNrm_temp[:, 0], 0, 0.0),  # x_list
                    np.insert(vNvrs_temp[:, j], 0, 0.0),  # f_list
                    np.insert(vNvrsP_temp[:, j], 0, vNvrsP_temp[j, 0]),  # dfdx_list
                )
            )
        vNvrsFuncFxd = LinearInterpOnInterp1D(vNvrsFuncFxd_by_Share, self.ShareGrid)
        self.vFuncFxd_now = ValueFuncCRRA(vNvrsFuncFxd, self.CRRA)

    def make_porfolio_solution(self):

        self.solution = PortfolioSolution(
            cFuncAdj=self.cFuncAdj_now,
            ShareFuncAdj=self.ShareFuncAdj_now,
            vPfuncAdj=self.vPfuncAdj_now,
            vFuncAdj=self.vFuncAdj_now,
            cFuncFxd=self.cFuncFxd_now,
            ShareFuncFxd=self.ShareFuncFxd_now,
            dvdmFuncFxd=self.dvdmFuncFxd_now,
            dvdsFuncFxd=self.dvdsFuncFxd_now,
            vFuncFxd=self.vFuncFxd_now,
            aGrid=self.save_points["a"],
            Share_adj=self.save_points["share_adj"],
            EndOfPrddvda_adj=self.save_points["eop_dvda_adj"],
            ShareGrid=self.save_points["share_grid"],
            EndOfPrddvda_fxd=self.save_points["eop_dvda_fxd"],
            EndOfPrddvds_fxd=self.save_points["eop_dvds_fxd"],
            AdjPrb=self.AdjustPrb,
        )

        if self.ApproxShareBool:
            self.solution.ShareEndOfPrdFunc = self.ShareEndOfPrdFunc
            self.solution.ApproxFirstOrderShareFunc = self.ApproxFirstOrderShareFunc
            self.solution.ApproxSecondOrderShareFunc = self.ApproxSecondOrderShareFunc

    def solve(self):
        """
        Solve the one period problem for a portfolio-choice consumer.

        Returns
        -------
        solution_now : PortfolioSolution
        The solution to the single period consumption-saving with portfolio choice
        problem.  Includes two consumption and risky share functions: one for when
        the agent can adjust his portfolio share (Adj) and when he can't (Fxd).
        """

        # Make arrays of end-of-period assets and end-of-period marginal values
        self.prepare_to_calc_EndOfPrdvP()
        self.calc_EndOfPrdvP()

        # Construct a basic solution for this period
        self.optimize_share()
        self.make_basic_solution()
        self.make_ShareFuncAdj()

        if self.ApproxShareBool:
            self.make_share_func_approx()

        self.add_save_points()

        # Add the value function if requested
        if self.vFuncBool:
            self.add_vFunc()
        else:  # If vFuncBool is False, fill in dummy values
            self.vFuncAdj_now = NullFunc()
            self.vFuncFxd_now = NullFunc()

        self.make_porfolio_solution()

        return self.solution


class ConsPortfolioDiscreteSolver(ConsPortfolioSolver):
    """
    Define an object-oriented one period solver.
    Solve the one period problem for a portfolio-choice consumer.
    This solver is used when the income and risky return shocks
    are independent and the allowed optimal share is discrete
    over a finite set of points in ShareGrid.
    """

    def optimize_share(self):
        """
        Optimization of Share on the discrete set ShareGrid
        """

        opt_idx = np.argmax(self.EndOfPrdv, axis=1)
        # Best portfolio share is one with highest value
        self.Share_now = self.ShareGrid[opt_idx]
        # Take cNrm at that index as well
        self.cNrmAdj_now = self.EndOfPrddvdaNvrs[np.arange(self.aNrmCount), opt_idx]
        if not self.zero_bound:
            # aNrm=0, so there's no way to "optimize" the portfolio
            self.Share_now[0] = 1.0
            # Consumption when aNrm=0 does not depend on Share
            self.cNrmAdj_now[0] = self.EndOfPrddvdaNvrs[0, -1]

    def make_ShareFuncAdj(self):
        """
        Construct the risky share function when the agent can adjust
        """

        mNrmAdj_mid = (self.mNrmAdj_now[1:] + self.mNrmAdj_now[:-1]) / 2
        mNrmAdj_plus = mNrmAdj_mid * (1.0 + 1e-12)
        mNrmAdj_comb = (np.transpose(np.vstack((mNrmAdj_mid, mNrmAdj_plus)))).flatten()
        mNrmAdj_comb = np.append(np.insert(mNrmAdj_comb, 0, 0.0), self.mNrmAdj_now[-1])
        Share_comb = (
            np.transpose(np.vstack((self.Share_now, self.Share_now)))
        ).flatten()
        self.ShareFuncAdj_now = LinearInterp(mNrmAdj_comb, Share_comb)

    def solve(self):
        """
        Solve the one period problem for a portfolio-choice consumer.

        Returns
        -------
        solution_now : PortfolioSolution
        The solution to the single period consumption-saving with portfolio choice
        problem.  Includes two consumption and risky share functions: one for when
        the agent can adjust his portfolio share (Adj) and when he can't (Fxd).
        """

        # Make arrays of end-of-period assets and end-of-period marginal value
        self.prepare_to_calc_EndOfPrdvP()
        self.calc_EndOfPrdvP()

        # Construct a basic solution for this period
        self.make_EndOfPrdvFunc()
        self.optimize_share()
        self.make_basic_solution()
        self.make_ShareFuncAdj()

        self.add_save_points()

        self.make_vFunc()

        self.make_porfolio_solution()

        return self.solution


class ConsPortfolioJointDistSolver(ConsPortfolioDiscreteSolver, ConsPortfolioSolver):
    """
    Define an object-oriented one period solver.
    Solve the one period problem for a portfolio-choice consumer.
    This solver is used when the income and risky return shocks
    are not independent. The optimal share can be continuous or
    discrete.
    """

    def set_and_update_values(self):
        """
        Unpacks some of the inputs (and calculates simple objects based on them),
        storing the results in self for use by other methods.
        """

        # Unpack next period's solution
        self.vPfuncAdj_next = self.solution_next.vPfuncAdj
        self.dvdmFuncFxd_next = self.solution_next.dvdmFuncFxd
        self.dvdsFuncFxd_next = self.solution_next.dvdsFuncFxd
        self.vFuncAdj_next = self.solution_next.vFuncAdj
        self.vFuncFxd_next = self.solution_next.vFuncFxd

        # If the distributions are NOT independent...
        # Unpack the shock distribution
        self.TranShks_next = self.ShockDstn.X[1]
        # Flag for whether the natural borrowing constraint is zero
        self.zero_bound = np.min(self.TranShks_next) == 0.0

    def prepare_to_calc_EndOfPrdvP(self):
        """
        Prepare to calculate end-of-period marginal values by creating an array
        of market resources that the agent could have next period, considering
        the grid of end-of-period assets and the distribution of shocks he might
        experience next period.
        """

        # Make tiled arrays to calculate future realizations of mNrm and Share; dimension order: mNrm, Share, shock
        if self.zero_bound:
            self.aNrmGrid = self.aXtraGrid
        else:
            # Add an asset point at exactly zero
            self.aNrmGrid = np.insert(self.aXtraGrid, 0, 0.0)

        self.aNrmCount = self.aNrmGrid.size
        self.ShareCount = self.ShareGrid.size

        self.aNrm_tiled, self.Share_tiled = np.meshgrid(
            self.aNrmGrid, self.ShareGrid, indexing="ij"
        )

    def r_port(self, shocks, share):
        """
        Calculate future realizations of market resources
        """

        return (1.0 - share) * self.Rfree + share * shocks[2]

    def m_nrm_next(self, shocks, a_nrm, r_port):
        """
        Calculate future realizations of market resources
        """

        return r_port * a_nrm / (shocks[0] * self.PermGroFac) + shocks[1]

    def calc_EndOfPrdvP(self):
        """
        Calculate end-of-period marginal value of assets and shares at each point
        in aNrm and ShareGrid. Does so by taking expectation of next period marginal
        values across income and risky return shocks.
        """

        def dvdm(m_nrm_next, shares):
            """
            Evaluate realizations of marginal value of market resources next period
            """

            dvdmAdj_next = self.vPfuncAdj_next(m_nrm_next)
            if self.AdjustPrb < 1.0:
                dvdmFxd_next = self.dvdmFuncFxd_next(m_nrm_next, shares)
                # Combine by adjustment probability
                dvdm_next = (
                    self.AdjustPrb * dvdmAdj_next
                    + (1.0 - self.AdjustPrb) * dvdmFxd_next
                )
            else:  # Don't bother evaluating if there's no chance that portfolio share is fixed
                dvdm_next = dvdmAdj_next

            return dvdm_next

        def dvds(m_nrm_next, shares):
            """
            Evaluate realizations of marginal value of risky share next period
            """

            # No marginal value of Share if it's a free choice!
            dvdsAdj_next = np.zeros_like(m_nrm_next)
            if self.AdjustPrb < 1.0:
                dvdsFxd_next = self.dvdsFuncFxd_next(m_nrm_next, shares)
                # Combine by adjustment probability
                dvds_next = (
                    self.AdjustPrb * dvdsAdj_next
                    + (1.0 - self.AdjustPrb) * dvdsFxd_next
                )
            else:  # Don't bother evaluating if there's no chance that portfolio share is fixed
                dvds_next = dvdsAdj_next

            return dvds_next

        def EndOfPrddvda_dists(shocks, a_nrm, shares):
            r_port = self.r_port(shocks, shares)
            m_nrm_next = self.m_nrm_next(shocks, a_nrm, r_port)

            return (
                r_port * self.uP(shocks[0] * self.PermGroFac) * dvdm(m_nrm_next, shares)
            )

        def EndOfPrddvds_dist(shocks, a_nrm, shares):
            Rxs = shocks[2] - self.Rfree
            r_port = self.r_port(shocks, shares)
            m_nrm_next = self.m_nrm_next(shocks, a_nrm, r_port)

            return Rxs * a_nrm * self.uP(shocks[0] * self.PermGroFac) * dvdm(
                m_nrm_next, shares
            ) + (shocks[0] * self.PermGroFac) ** (1.0 - self.CRRA) * dvds(
                m_nrm_next, shares
            )

        # Calculate end-of-period marginal value of assets by taking expectations
        self.EndOfPrddvda = (
            self.DiscFac
            * self.LivPrb
            * calc_expectation(
                self.ShockDstn, EndOfPrddvda_dists, self.aNrm_tiled, self.Share_tiled
            )
        )
        self.EndOfPrddvda = self.EndOfPrddvda[:, :, 0]
        self.EndOfPrddvdaNvrs = self.uPinv(self.EndOfPrddvda)

        # Calculate end-of-period marginal value of risky portfolio share by taking expectations
        self.EndOfPrddvds = (
            self.DiscFac
            * self.LivPrb
            * calc_expectation(
                self.ShockDstn, EndOfPrddvds_dist, self.aNrm_tiled, self.Share_tiled
            )
        )
        self.EndOfPrddvds = self.EndOfPrddvds[:, :, 0]

    def make_EndOfPrdvFunc(self):
        """
        Construct the end-of-period value function for this period, storing it
        as an attribute of self for use by other methods.
        """

        def v_dist(shocks, a_nrm, shares):
            r_port = self.r_port(shocks, shares)
            m_nrm_next = self.m_nrm_next(shocks, a_nrm, r_port)

            vAdj_next = self.vFuncAdj_next(m_nrm_next)
            if self.AdjustPrb < 1.0:
                vFxd_next = self.vFuncFxd_next(m_nrm_next, shares)
                v_next = self.AdjustPrb * vAdj_next + (1.0 - self.AdjustPrb) * vFxd_next
            else:  # Don't bother evaluating if there's no chance that portfolio share is fixed
                v_next = vAdj_next

            return (shocks[0] * self.PermGroFac) ** (1.0 - self.CRRA) * v_next

        self.EndOfPrdv = (
            self.DiscFac
            * self.LivPrb
            * calc_expectation(
                self.ShockDstn, v_dist, self.aNrm_tiled, self.Share_tiled
            )
        )
        self.EndOfPrdv = self.EndOfPrdv[:, :, 0]
        self.EndOfPrdvNvrs = self.uinv(self.EndOfPrdv)

    def solve(self):
        """
        Solve the one period problem for a portfolio-choice consumer.

        Returns
        -------
        solution_now : PortfolioSolution
        The solution to the single period consumption-saving with portfolio choice
        problem.  Includes two consumption and risky share functions: one for when
        the agent can adjust his portfolio share (Adj) and when he can't (Fxd).
        """

        # Make arrays of end-of-period assets and end-of-period marginal value
        self.prepare_to_calc_EndOfPrdvP()
        self.calc_EndOfPrdvP()

        if self.DiscreteShareBool:
            self.make_EndOfPrdvFunc()
            ConsPortfolioDiscreteSolver.optimize_share(self)

            # Construct a basic solution for this period
            self.make_basic_solution()
            ConsPortfolioDiscreteSolver.make_ShareFuncAdj(self)
            self.make_vFunc()
        else:
            # Construct a basic solution for this period
            ConsPortfolioSolver.optimize_share(self)
            self.make_basic_solution()
            ConsPortfolioSolver.make_ShareFuncAdj(self)

            # Add the value function if requested
            if self.vFuncBool:
                self.add_vFunc()
            else:  # If vFuncBool is False, fill in dummy values
                self.vFuncAdj_now = NullFunc()
                self.vFuncFxd_now = NullFunc()

        self.add_save_points()

        self.make_porfolio_solution()

        return self.solution


class ConsSequentialPortfolioSolver(ConsPortfolioSolver):
    def add_SequentialShareFuncAdj(self, solution):
        """
        Construct the risky share function as a function of savings when the agent can adjust.
        """

        if self.zero_bound:
            Share_lower_bound = self.ShareLimit
            aNrm_temp = np.insert(self.aNrmGrid, 0, 0.0)
            Share_now = np.insert(self.Share_now, 0, Share_lower_bound)
        else:
            aNrm_temp = self.aNrmGrid  # already includes 0.0
            Share_now = self.Share_now

        self.SequentialShareFuncAdj_now = LinearInterp(
            aNrm_temp, Share_now, intercept_limit=self.ShareLimit, slope_limit=0.0,
        )

        solution.SequentialShareFuncAdj = self.SequentialShareFuncAdj_now

        return solution

    def solve(self):
        solution = ConsPortfolioSolver.solve(self)

        solution = self.add_SequentialShareFuncAdj(solution)

        return solution


# Make a dictionary to specify a portfolio choice consumer type
init_portfolio = init_idiosyncratic_shocks.copy()
init_portfolio["RiskyAvg"] = 1.08  # Average return of the risky asset
init_portfolio["RiskyStd"] = 0.20  # Standard deviation of (log) risky returns
# Number of integration nodes to use in approximation of risky returns
init_portfolio["RiskyCount"] = 5
# Number of discrete points in the risky share approximation
init_portfolio["ShareCount"] = 25
# Probability that the agent can adjust their risky portfolio share each period
init_portfolio["AdjustPrb"] = 1.0
# Flag for whether to optimize risky share on a discrete grid only
init_portfolio["DiscreteShareBool"] = False
# Flat for wether to approximate risky share
init_portfolio["ApproxShareBool"] = False

# Adjust some of the existing parameters in the dictionary
init_portfolio["aXtraMax"] = 100  # Make the grid of assets go much higher...
init_portfolio["aXtraCount"] = 200  # ...and include many more gridpoints...
init_portfolio["aXtraNestFac"] = 1  # ...which aren't so clustered at the bottom
init_portfolio["BoroCnstArt"] = 0.0  # Artificial borrowing constraint must be turned on
init_portfolio["CRRA"] = 5.0  # Results are more interesting with higher risk aversion
init_portfolio["DiscFac"] = 0.90  # And also lower patience
