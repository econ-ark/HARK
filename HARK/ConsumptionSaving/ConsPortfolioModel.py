"""
This file contains classes and functions for representing, solving, and simulating
agents who must allocate their resources among consumption, saving in a risk-free
asset (with a low return), and saving in a risky asset (with higher average return).
"""

from copy import deepcopy

import numpy as np

from HARK import AgentType, NullFunc, make_one_period_oo_solver
from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    init_idiosyncratic_shocks,
    utility,
    utility_inv,
    utility_invP,
    utilityP,
    utilityP_inv,
)
from HARK.ConsumptionSaving.ConsRiskyAssetModel import RiskyAssetConsumerType
from HARK.distribution import expected
from HARK.interpolation import (
    BilinearInterp,
    ConstantFunction,
    CubicInterp,
    IdentityFunction,
    LinearInterp,
    LinearInterpOnInterp1D,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.metric import MetricObject
from HARK.rewards import UtilityFuncCRRA


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
    time_inv_ = time_inv_ + ["AdjustPrb", "DiscreteShareBool"]

    def __init__(self, verbose=False, quiet=False, **kwds):
        params = init_portfolio.copy()
        params.update(kwds)
        kwds = params

        self.PortfolioBool = True

        # Initialize a basic consumer type
        RiskyAssetConsumerType.__init__(self, verbose=verbose, quiet=quiet, **kwds)

        # Set the solver for the portfolio model, and update various constructed attributes
        if self.IndepDstnBool:
            self.solve_one_period = solve_one_period_ConsPortfolio
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
        cFuncAdj_terminal = IdentityFunction()
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


def solve_one_period_ConsPortfolio(
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
    AdjustPrb,
    ShareLimit,
    vFuncBool,
    DiscreteShareBool,
    IndepDstnBool,
):
    """
    Solve one period of a consumption-saving problem with portfolio allocation
    between a riskless and risky asset. This function handles various sub-cases
    or variations on the problem, including the possibility that the agent does
    not necessarily get to update their portfolio share in every period, or that
    they must choose a discrete rather than continuous risky share.

    Parameters
    ----------
    solution_next : PortfolioSolution
        Solution to next period's problem.
    ShockDstn : Distribution
        Joint distribution of permanent income shocks, transitory income shocks,
        and risky returns.  This is only used if the input IndepDstnBool is False,
        indicating that income and return distributions can't be assumed to be
        independent.
    IncShkDstn : Distribution
        Discrete distribution of permanent income shocks and transitory income
        shocks. This is only used if the input IndepDstnBool is True, indicating
        that income and return distributions are independent.
    RiskyDstn : Distribution
       Distribution of risky asset returns. This is only used if the input
       IndepDstnBool is True, indicating that income and return distributions
       are independent.
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
        of the consumption function when Share is fixed. Also used when the
        risky share choice is specified as discrete rather than continuous.
    AdjustPrb : float
        Probability that the agent will be able to update his portfolio share.
    ShareLimit : float
        Limiting lower bound of risky portfolio share as mNrm approaches infinity.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    DiscreteShareBool : bool
        Indicator for whether risky portfolio share should be optimized on the
        continuous [0,1] interval using the FOC (False), or instead only selected
        from the discrete set of values in ShareGrid (True).  If True, then
        vFuncBool must also be True.
    IndepDstnBool : bool
        Indicator for whether the income and risky return distributions are in-
        dependent of each other, which can speed up the expectations step.

    Returns
    -------
    solution_now : PortfolioSolution
        Solution to this period's problem.
    """
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

    # Define the current period utility function and effective discount factor
    uFunc = UtilityFuncCRRA(CRRA)
    DiscFacEff = DiscFac * LivPrb  # "effective" discount factor

    # Unpack next period's solution for easier access
    vPfuncAdj_next = solution_next.vPfuncAdj
    dvdmFuncFxd_next = solution_next.dvdmFuncFxd
    dvdsFuncFxd_next = solution_next.dvdsFuncFxd
    vFuncAdj_next = solution_next.vFuncAdj
    vFuncFxd_next = solution_next.vFuncFxd

    # Set a flag for whether the natural borrowing constraint is zero, which
    # depends on whether the smallest transitory income shock is zero
    BoroCnstNat_iszero = np.min(IncShkDstn.atoms[1]) == 0.0

    # Prepare to calculate end-of-period marginal values by creating an array
    # of market resources that the agent could have next period, considering
    # the grid of end-of-period assets and the distribution of shocks he might
    # experience next period.

    # Unpack the risky return shock distribution
    Risky_next = RiskyDstn.atoms
    RiskyMax = np.max(Risky_next)
    RiskyMin = np.min(Risky_next)

    # bNrm represents R*a, balances after asset return shocks but before income.
    # This just uses the highest risky return as a rough shifter for the aXtraGrid.
    if BoroCnstNat_iszero:
        aNrmGrid = aXtraGrid
        bNrmGrid = np.insert(RiskyMax * aXtraGrid, 0, RiskyMin * aXtraGrid[0])
    else:
        # Add an asset point at exactly zero
        aNrmGrid = np.insert(aXtraGrid, 0, 0.0)
        bNrmGrid = RiskyMax * np.insert(aXtraGrid, 0, 0.0)

    # Get grid and shock sizes, for easier indexing
    aNrmCount = aNrmGrid.size
    ShareCount = ShareGrid.size

    # Make tiled arrays to calculate future realizations of mNrm and Share when integrating over IncShkDstn
    bNrmNext, ShareNext = np.meshgrid(bNrmGrid, ShareGrid, indexing="ij")

    # Define functions that are used internally to evaluate future realizations
    def calc_mNrm_next(S, b):
        """
        Calculate future realizations of market resources mNrm from the income
        shock distribution S and normalized bank balances b.
        """
        return b / (S["PermShk"] * PermGroFac) + S["TranShk"]

    def calc_dvdm_next(S, b, z):
        """
        Evaluate realizations of marginal value of market resources next period,
        based on the income distribution S, values of bank balances bNrm, and
        values of the risky share z.
        """
        mNrm_next = calc_mNrm_next(S, b)
        dvdmAdj_next = vPfuncAdj_next(mNrm_next)

        if AdjustPrb < 1.0:
            # Expand to the same dimensions as mNrm
            Share_next_expanded = z + np.zeros_like(mNrm_next)
            dvdmFxd_next = dvdmFuncFxd_next(mNrm_next, Share_next_expanded)
            # Combine by adjustment probability
            dvdm_next = AdjustPrb * dvdmAdj_next + (1.0 - AdjustPrb) * dvdmFxd_next
        else:  # Don't bother evaluating if there's no chance that portfolio share is fixed
            dvdm_next = dvdmAdj_next

        dvdm_next = (S["PermShk"] * PermGroFac) ** (-CRRA) * dvdm_next
        return dvdm_next

    def calc_dvds_next(S, b, z):
        """
        Evaluate realizations of marginal value of risky share next period, based
        on the income distribution S, values of bank balances bNrm, and values of
        the risky share z.
        """
        mNrm_next = calc_mNrm_next(S, b)

        # No marginal value of Share if it's a free choice!
        dvdsAdj_next = np.zeros_like(mNrm_next)

        if AdjustPrb < 1.0:
            # Expand to the same dimensions as mNrm
            Share_next_expanded = z + np.zeros_like(mNrm_next)
            dvdsFxd_next = dvdsFuncFxd_next(mNrm_next, Share_next_expanded)
            # Combine by adjustment probability
            dvds_next = AdjustPrb * dvdsAdj_next + (1.0 - AdjustPrb) * dvdsFxd_next
        else:  # Don't bother evaluating if there's no chance that portfolio share is fixed
            dvds_next = dvdsAdj_next

        dvds_next = (S["PermShk"] * PermGroFac) ** (1.0 - CRRA) * dvds_next
        return dvds_next

    # Calculate end-of-period marginal value of assets and shares at each point
    # in aNrm and ShareGrid. Does so by taking expectation of next period marginal
    # values across income and risky return shocks.

    # Calculate intermediate marginal value of bank balances by taking expectations over income shocks
    dvdb_intermed = expected(calc_dvdm_next, IncShkDstn, args=(bNrmNext, ShareNext))
    dvdbNvrs_intermed = uFunc.derinv(dvdb_intermed, order=(1, 0))
    dvdbNvrsFunc_intermed = BilinearInterp(dvdbNvrs_intermed, bNrmGrid, ShareGrid)
    dvdbFunc_intermed = MargValueFuncCRRA(dvdbNvrsFunc_intermed, CRRA)

    # Calculate intermediate marginal value of risky portfolio share by taking expectations over income shocks
    dvds_intermed = expected(calc_dvds_next, IncShkDstn, args=(bNrmNext, ShareNext))
    dvdsFunc_intermed = BilinearInterp(dvds_intermed, bNrmGrid, ShareGrid)

    # Make tiled arrays to calculate future realizations of bNrm and Share when integrating over RiskyDstn
    aNrmNow, ShareNext = np.meshgrid(aNrmGrid, ShareGrid, indexing="ij")

    # Define functions for calculating end-of-period marginal value
    def calc_EndOfPrd_dvda(S, a, z):
        """
        Compute end-of-period marginal value of assets at values a, conditional
        on risky asset return S and risky share z.
        """
        # Calculate future realizations of bank balances bNrm
        Rxs = S - Rfree  # Excess returns
        Rport = Rfree + z * Rxs  # Portfolio return
        bNrm_next = Rport * a

        # Ensure shape concordance
        z_rep = z + np.zeros_like(bNrm_next)

        # Calculate and return dvda
        EndOfPrd_dvda = Rport * dvdbFunc_intermed(bNrm_next, z_rep)
        return EndOfPrd_dvda

    def EndOfPrddvds_dist(S, a, z):
        """
        Compute end-of-period marginal value of risky share at values a, conditional
        on risky asset return S and risky share z.
        """
        # Calculate future realizations of bank balances bNrm
        Rxs = S - Rfree  # Excess returns
        Rport = Rfree + z * Rxs  # Portfolio return
        bNrm_next = Rport * a

        # Make the shares match the dimension of b, so that it can be vectorized
        z_rep = z + np.zeros_like(bNrm_next)

        # Calculate and return dvds
        EndOfPrd_dvds = Rxs * a * dvdbFunc_intermed(
            bNrm_next, z_rep
        ) + dvdsFunc_intermed(bNrm_next, z_rep)
        return EndOfPrd_dvds

    # Evaluate realizations of value and marginal value after asset returns are realized

    # Calculate end-of-period marginal value of assets by taking expectations
    EndOfPrd_dvda = DiscFacEff * expected(
        calc_EndOfPrd_dvda, RiskyDstn, args=(aNrmNow, ShareNext)
    )
    EndOfPrd_dvdaNvrs = uFunc.derinv(EndOfPrd_dvda)

    # Calculate end-of-period marginal value of risky portfolio share by taking expectations
    EndOfPrd_dvds = DiscFacEff * expected(
        EndOfPrddvds_dist, RiskyDstn, args=(aNrmNow, ShareNext)
    )

    # Make the end-of-period value function if the value function is requested
    if vFuncBool:

        def calc_v_intermed(S, b, z):
            """
            Calculate "intermediate" value from next period's bank balances, the
            income shocks S, and the risky asset share.
            """
            mNrm_next = calc_mNrm_next(S, b)

            vAdj_next = vFuncAdj_next(mNrm_next)
            if AdjustPrb < 1.0:
                vFxd_next = vFuncFxd_next(mNrm_next, z)
                # Combine by adjustment probability
                v_next = AdjustPrb * vAdj_next + (1.0 - AdjustPrb) * vFxd_next
            else:  # Don't bother evaluating if there's no chance that portfolio share is fixed
                v_next = vAdj_next

            v_intermed = (S["PermShk"] * PermGroFac) ** (1.0 - CRRA) * v_next
            return v_intermed

        # Calculate intermediate value by taking expectations over income shocks
        v_intermed = expected(calc_v_intermed, IncShkDstn, args=(bNrmNext, ShareNext))

        # Construct the "intermediate value function" for this period
        vNvrs_intermed = uFunc.inv(v_intermed)
        vNvrsFunc_intermed = BilinearInterp(vNvrs_intermed, bNrmGrid, ShareGrid)
        vFunc_intermed = ValueFuncCRRA(vNvrsFunc_intermed, CRRA)

        def calc_EndOfPrd_v(S, a, z):
            # Calculate future realizations of bank balances bNrm
            Rxs = S - Rfree
            Rport = Rfree + z * Rxs
            bNrm_next = Rport * a

            # Make an extended share_next of the same dimension as b_nrm so
            # that the function can be vectorized
            z_rep = z + np.zeros_like(bNrm_next)

            EndOfPrd_v = vFunc_intermed(bNrm_next, z_rep)
            return EndOfPrd_v

        # Calculate end-of-period value by taking expectations
        EndOfPrd_v = DiscFacEff * expected(
            calc_EndOfPrd_v, RiskyDstn, args=(aNrmNow, ShareNext)
        )
        EndOfPrd_vNvrs = uFunc.inv(EndOfPrd_v)

        # Now make an end-of-period value function over aNrm and Share
        EndOfPrd_vNvrsFunc = BilinearInterp(EndOfPrd_vNvrs, aNrmGrid, ShareGrid)
        EndOfPrd_vFunc = ValueFuncCRRA(EndOfPrd_vNvrsFunc, CRRA)
        # This will be used later to make the value function for this period

    # Find the optimal risky asset share either by choosing the best value among
    # the discrete grid choices, or by satisfying the FOC with equality (continuous)
    if DiscreteShareBool:
        # If we're restricted to discrete choices, then portfolio share is
        # the one with highest value for each aNrm gridpoint
        opt_idx = np.argmax(EndOfPrd_v, axis=1)
        ShareAdj_now = ShareGrid[opt_idx]

        # Take cNrm at that index as well... and that's it!
        cNrmAdj_now = EndOfPrd_dvdaNvrs[np.arange(aNrmCount), opt_idx]

    else:
        # Now find the optimal (continuous) risky share on [0,1] by solving the first
        # order condition EndOfPrd_dvds == 0.
        FOC_s = EndOfPrd_dvds  # Relabel for convenient typing

        # For each value of aNrm, find the value of Share such that FOC_s == 0
        crossing = np.logical_and(FOC_s[:, 1:] <= 0.0, FOC_s[:, :-1] >= 0.0)
        share_idx = np.argmax(crossing, axis=1)
        # This represents the index of the segment of the share grid where dvds flips
        # from positive to negative, indicating that there's a zero *on* the segment

        # Calculate the fractional distance between those share gridpoints where the
        # zero should be found, assuming a linear function; call it alpha
        a_idx = np.arange(aNrmCount)
        bot_s = ShareGrid[share_idx]
        top_s = ShareGrid[share_idx + 1]
        bot_f = FOC_s[a_idx, share_idx]
        top_f = FOC_s[a_idx, share_idx + 1]
        bot_c = EndOfPrd_dvdaNvrs[a_idx, share_idx]
        top_c = EndOfPrd_dvdaNvrs[a_idx, share_idx + 1]
        alpha = 1.0 - top_f / (top_f - bot_f)

        # Calculate the continuous optimal risky share and optimal consumption
        ShareAdj_now = (1.0 - alpha) * bot_s + alpha * top_s
        cNrmAdj_now = (1.0 - alpha) * bot_c + alpha * top_c

        # If agent wants to put more than 100% into risky asset, he is constrained.
        # Likewise if he wants to put less than 0% into risky asset, he is constrained.
        constrained_top = FOC_s[:, -1] > 0.0
        constrained_bot = FOC_s[:, 0] < 0.0

        # Apply those constraints to both risky share and consumption (but lower
        # constraint should never be relevant)
        ShareAdj_now[constrained_top] = 1.0
        ShareAdj_now[constrained_bot] = 0.0
        cNrmAdj_now[constrained_top] = EndOfPrd_dvdaNvrs[constrained_top, -1]
        cNrmAdj_now[constrained_bot] = EndOfPrd_dvdaNvrs[constrained_bot, 0]

    # When the natural borrowing constraint is *not* zero, then aNrm=0 is in the
    # grid, but there's no way to "optimize" the portfolio if a=0, and consumption
    # can't depend on the risky share if it doesn't meaningfully exist. Apply
    # a small fix to the bottom gridpoint (aNrm=0) when this happens.
    if not BoroCnstNat_iszero:
        ShareAdj_now[0] = 1.0
        cNrmAdj_now[0] = EndOfPrd_dvdaNvrs[0, -1]

    # Construct functions characterizing the solution for this period

    # Calculate the endogenous mNrm gridpoints when the agent adjusts his portfolio,
    # then construct the consumption function when the agent can adjust his share
    mNrmAdj_now = np.insert(aNrmGrid + cNrmAdj_now, 0, 0.0)
    cNrmAdj_now = np.insert(cNrmAdj_now, 0, 0.0)
    cFuncAdj_now = LinearInterp(mNrmAdj_now, cNrmAdj_now)

    # Construct the marginal value (of mNrm) function when the agent can adjust
    vPfuncAdj_now = MargValueFuncCRRA(cFuncAdj_now, CRRA)

    # Construct the consumption function when the agent *can't* adjust the risky
    # share, as well as the marginal value of Share function
    cFuncFxd_by_Share = []
    dvdsFuncFxd_by_Share = []
    for j in range(ShareCount):
        cNrmFxd_temp = np.insert(EndOfPrd_dvdaNvrs[:, j], 0, 0.0)
        mNrmFxd_temp = np.insert(aNrmGrid + cNrmFxd_temp[1:], 0, 0.0)
        dvdsFxd_temp = np.insert(EndOfPrd_dvds[:, j], 0, EndOfPrd_dvds[0, j])
        cFuncFxd_by_Share.append(LinearInterp(mNrmFxd_temp, cNrmFxd_temp))
        dvdsFuncFxd_by_Share.append(LinearInterp(mNrmFxd_temp, dvdsFxd_temp))
    cFuncFxd_now = LinearInterpOnInterp1D(cFuncFxd_by_Share, ShareGrid)
    dvdsFuncFxd_now = LinearInterpOnInterp1D(dvdsFuncFxd_by_Share, ShareGrid)

    # The share function when the agent can't adjust his portfolio is trivial
    ShareFuncFxd_now = IdentityFunction(i_dim=1, n_dims=2)

    # Construct the marginal value of mNrm function when the agent can't adjust his share
    dvdmFuncFxd_now = MargValueFuncCRRA(cFuncFxd_now, CRRA)

    # Construct the optimal risky share function when adjusting is possible.
    # The interpolation method depends on whether the choice is discrete or continuous.
    if DiscreteShareBool:
        # If the share choice is discrete, the "interpolated" share function acts
        # like a step function, with jumps at the midpoints of mNrm gridpoints.
        # Because an actual step function would break our (assumed continuous) linear
        # interpolator, there's a *tiny* region with extremely high slope.
        mNrmAdj_mid = (mNrmAdj_now[2:] + mNrmAdj_now[1:-1]) / 2
        mNrmAdj_plus = mNrmAdj_mid * (1.0 + 1e-12)
        mNrmAdj_comb = (np.transpose(np.vstack((mNrmAdj_mid, mNrmAdj_plus)))).flatten()
        mNrmAdj_comb = np.append(np.insert(mNrmAdj_comb, 0, 0.0), mNrmAdj_now[-1])
        Share_comb = (np.transpose(np.vstack((ShareAdj_now, ShareAdj_now)))).flatten()
        ShareFuncAdj_now = LinearInterp(mNrmAdj_comb, Share_comb)

    else:
        # If the share choice is continuous, just make an ordinary interpolating function
        if BoroCnstNat_iszero:
            Share_lower_bound = ShareLimit
        else:
            Share_lower_bound = 1.0
        ShareAdj_now = np.insert(ShareAdj_now, 0, Share_lower_bound)
        ShareFuncAdj_now = LinearInterp(mNrmAdj_now, ShareAdj_now, ShareLimit, 0.0)

    # This is a point at which (a,c,share) have consistent length. Take the
    # snapshot for storing the grid and values in the solution.
    save_points = {
        "a": deepcopy(aNrmGrid),
        "eop_dvda_adj": uFunc.der(cNrmAdj_now),
        "share_adj": deepcopy(ShareAdj_now),
        "share_grid": deepcopy(ShareGrid),
        "eop_dvda_fxd": uFunc.der(EndOfPrd_dvda),
        "eop_dvds_fxd": EndOfPrd_dvds,
    }

    # Add the value function if requested
    if vFuncBool:
        # Create the value functions for this period, defined over market resources
        # mNrm when agent can adjust his portfolio, and over market resources and
        # fixed share when agent can not adjust his portfolio.

        # Construct the value function when the agent can adjust his portfolio
        mNrm_temp = aXtraGrid  # Just use aXtraGrid as our grid of mNrm values
        cNrm_temp = cFuncAdj_now(mNrm_temp)
        aNrm_temp = mNrm_temp - cNrm_temp
        Share_temp = ShareFuncAdj_now(mNrm_temp)
        v_temp = uFunc(cNrm_temp) + EndOfPrd_vFunc(aNrm_temp, Share_temp)
        vNvrs_temp = uFunc.inv(v_temp)
        vNvrsP_temp = uFunc.der(cNrm_temp) * uFunc.inverse(v_temp, order=(0, 1))
        vNvrsFuncAdj = CubicInterp(
            np.insert(mNrm_temp, 0, 0.0),  # x_list
            np.insert(vNvrs_temp, 0, 0.0),  # f_list
            np.insert(vNvrsP_temp, 0, vNvrsP_temp[0]),  # dfdx_list
        )
        # Re-curve the pseudo-inverse value function
        vFuncAdj_now = ValueFuncCRRA(vNvrsFuncAdj, CRRA)

        # Construct the value function when the agent *can't* adjust his portfolio
        mNrm_temp, Share_temp = np.meshgrid(aXtraGrid, ShareGrid)
        cNrm_temp = cFuncFxd_now(mNrm_temp, Share_temp)
        aNrm_temp = mNrm_temp - cNrm_temp
        v_temp = uFunc(cNrm_temp) + EndOfPrd_vFunc(aNrm_temp, Share_temp)
        vNvrs_temp = uFunc.inv(v_temp)
        vNvrsP_temp = uFunc.der(cNrm_temp) * uFunc.inverse(v_temp, order=(0, 1))
        vNvrsFuncFxd_by_Share = []
        for j in range(ShareCount):
            vNvrsFuncFxd_by_Share.append(
                CubicInterp(
                    np.insert(mNrm_temp[:, 0], 0, 0.0),  # x_list
                    np.insert(vNvrs_temp[:, j], 0, 0.0),  # f_list
                    np.insert(vNvrsP_temp[:, j], 0, vNvrsP_temp[j, 0]),  # dfdx_list
                )
            )
        vNvrsFuncFxd = LinearInterpOnInterp1D(vNvrsFuncFxd_by_Share, ShareGrid)
        vFuncFxd_now = ValueFuncCRRA(vNvrsFuncFxd, CRRA)

    else:  # If vFuncBool is False, fill in dummy values
        vFuncAdj_now = NullFunc()
        vFuncFxd_now = NullFunc()

    # Package and return the solution
    solution_now = PortfolioSolution(
        cFuncAdj=cFuncAdj_now,
        ShareFuncAdj=ShareFuncAdj_now,
        vPfuncAdj=vPfuncAdj_now,
        vFuncAdj=vFuncAdj_now,
        cFuncFxd=cFuncFxd_now,
        ShareFuncFxd=ShareFuncFxd_now,
        dvdmFuncFxd=dvdmFuncFxd_now,
        dvdsFuncFxd=dvdsFuncFxd_now,
        vFuncFxd=vFuncFxd_now,
        AdjPrb=AdjustPrb,
        # WHAT IS THIS STUFF FOR??
        aGrid=save_points["a"],
        Share_adj=save_points["share_adj"],
        EndOfPrddvda_adj=save_points["eop_dvda_adj"],
        ShareGrid=save_points["share_grid"],
        EndOfPrddvda_fxd=save_points["eop_dvda_fxd"],
        EndOfPrddvds_fxd=save_points["eop_dvds_fxd"],
    )
    return solution_now


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
        TranShks_next = self.IncShkDstn.atoms[1]

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
        Risky_next = self.RiskyDstn.atoms
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

        return b_nrm_next / (shocks["PermShk"] * self.PermGroFac) + shocks["TranShk"]

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
                # Expand to the same dimensions as mNrm
                Share_next_expanded = Share_next + np.zeros_like(mNrm_next)
                dvdmFxd_next = self.dvdmFuncFxd_next(mNrm_next, Share_next_expanded)
                # Combine by adjustment probability
                dvdm_next = (
                    self.AdjustPrb * dvdmAdj_next
                    + (1.0 - self.AdjustPrb) * dvdmFxd_next
                )
            else:  # Don't bother evaluating if there's no chance that portfolio share is fixed
                dvdm_next = dvdmAdj_next

            return (shocks["PermShk"] * self.PermGroFac) ** (-self.CRRA) * dvdm_next

        def dvds_dist(shocks, b_nrm, Share_next):
            """
            Evaluate realizations of marginal value of risky share next period
            """

            mNrm_next = self.m_nrm_next(shocks, b_nrm)
            # No marginal value of Share if it's a free choice!
            dvdsAdj_next = np.zeros_like(mNrm_next)
            if self.AdjustPrb < 1.0:
                # Expand to the same dimensions as mNrm
                Share_next_expanded = Share_next + np.zeros_like(mNrm_next)
                dvdsFxd_next = self.dvdsFuncFxd_next(mNrm_next, Share_next_expanded)
                # Combine by adjustment probability
                dvds_next = (
                    self.AdjustPrb * dvdsAdj_next
                    + (1.0 - self.AdjustPrb) * dvdsFxd_next
                )
            else:  # Don't bother evaluating if there's no chance that portfolio share is fixed
                dvds_next = dvdsAdj_next

            return (shocks["PermShk"] * self.PermGroFac) ** (
                1.0 - self.CRRA
            ) * dvds_next

        # Calculate intermediate marginal value of bank balances by taking expectations over income shocks
        dvdb_intermed = self.IncShkDstn.expected(
            dvdb_dist, self.bNrmNext, self.ShareNext
        )

        dvdbNvrs_intermed = self.uPinv(dvdb_intermed)
        dvdbNvrsFunc_intermed = BilinearInterp(
            dvdbNvrs_intermed, self.bNrmGrid, self.ShareGrid
        )
        dvdbFunc_intermed = MargValueFuncCRRA(dvdbNvrsFunc_intermed, self.CRRA)

        # Calculate intermediate marginal value of risky portfolio share by taking expectations
        dvds_intermed = self.IncShkDstn.expected(
            dvds_dist, self.bNrmNext, self.ShareNext
        )

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

            # Ensure shape concordance
            Share_next_rep = Share_next + np.zeros_like(b_nrm_next)

            return Rport * dvdbFunc_intermed(b_nrm_next, Share_next_rep)

        def EndOfPrddvds_dist(shock, a_nrm, Share_next):
            # Calculate future realizations of bank balances bNrm
            Rxs = shock - self.Rfree
            Rport = self.Rfree + Share_next * Rxs
            b_nrm_next = Rport * a_nrm

            # Make the shares match the dimension of b, so that it can be vectorized
            Share_next_expand = Share_next + np.zeros_like(b_nrm_next)

            return Rxs * a_nrm * dvdbFunc_intermed(
                b_nrm_next, Share_next_expand
            ) + dvdsFunc_intermed(b_nrm_next, Share_next_expand)

        # Calculate end-of-period marginal value of assets by taking expectations
        self.EndOfPrddvda = (
            self.DiscFac
            * self.LivPrb
            * self.RiskyDstn.expected(
                EndOfPrddvda_dist, self.aNrm_tiled, self.ShareNext
            )
        )

        self.EndOfPrddvdaNvrs = self.uPinv(self.EndOfPrddvda)

        # Calculate end-of-period marginal value of risky portfolio share by taking expectations
        self.EndOfPrddvds = (
            self.DiscFac
            * self.LivPrb
            * self.RiskyDstn.expected(
                EndOfPrddvds_dist, self.aNrm_tiled, self.ShareNext
            )
        )

    def optimize_share(self):
        """
        Optimization of Share on continuous interval [0,1]
        """

        FOC_s = self.EndOfPrddvds

        # For each value of aNrm, find the value of Share such that FOC-Share == 0.
        crossing = np.logical_and(FOC_s[:, 1:] <= 0.0, FOC_s[:, :-1] >= 0.0)
        share_idx = np.argmax(crossing, axis=1)
        a_idx = np.arange(self.aNrmCount)
        bot_s = self.ShareGrid[share_idx]
        top_s = self.ShareGrid[share_idx + 1]
        bot_f = FOC_s[a_idx, share_idx]
        top_f = FOC_s[a_idx, share_idx + 1]
        bot_c = self.EndOfPrddvdaNvrs[a_idx, share_idx]
        top_c = self.EndOfPrddvdaNvrs[a_idx, share_idx + 1]
        alpha = 1.0 - top_f / (top_f - bot_f)

        self.Share_now = (1.0 - alpha) * bot_s + alpha * top_s
        self.cNrmAdj_now = (1.0 - alpha) * bot_c + alpha * top_c

        # If agent wants to put more than 100% into risky asset, he is constrained
        constrained_top = FOC_s[:, -1] > 0.0
        # Likewise if he wants to put less than 0% into risky asset
        constrained_bot = FOC_s[:, 0] < 0.0

        # For values of aNrm at which the agent wants to put
        # more than 100% into risky asset, constrain them
        self.Share_now[constrained_top] = 1.0
        self.Share_now[constrained_bot] = 0.0

        # Get consumption when share-constrained
        self.cNrmAdj_now[constrained_top] = self.EndOfPrddvdaNvrs[constrained_top, -1]
        self.cNrmAdj_now[constrained_bot] = self.EndOfPrddvdaNvrs[constrained_bot, 0]

        if not self.zero_bound:
            # aNrm=0, so there's no way to "optimize" the portfolio
            self.Share_now[0] = 1.0
            # Consumption when aNrm=0 does not depend on Share
            self.cNrmAdj_now[0] = self.EndOfPrddvdaNvrs[0, -1]

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

            return (shocks["PermShk"] * self.PermGroFac) ** (1.0 - self.CRRA) * v_next

        # Calculate intermediate value by taking expectations over income shocks
        v_intermed = self.IncShkDstn.expected(
            v_intermed_dist, self.bNrmNext, self.ShareNext
        )

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

            # Make an extended share_next of the same dimension as b_nrm so
            # that the function can be vectorized
            Share_next_extended = Share_next + np.zeros_like(b_nrm_next)

            return vFunc_intermed(b_nrm_next, Share_next_extended)

        # Calculate end-of-period value by taking expectations
        self.EndOfPrdv = (
            self.DiscFac
            * self.LivPrb
            * self.RiskyDstn.expected(EndOfPrdv_dist, self.aNrm_tiled, self.ShareNext)
        )

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
        self.TranShks_next = self.ShockDstn.atoms[1]
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

        return (1.0 - share) * self.Rfree + share * shocks["Risky"]

    def m_nrm_next(self, shocks, a_nrm, r_port):
        """
        Calculate future realizations of market resources
        """

        return (
            r_port * a_nrm / (shocks["PermShk"] * self.PermGroFac) + shocks["TranShk"]
        )

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

            # Expand shares to the shape of m so that operations can be vectorized
            shares_expanded = shares + np.zeros_like(m_nrm_next)

            return (
                r_port
                * self.uP(shocks["PermShk"] * self.PermGroFac)
                * dvdm(m_nrm_next, shares_expanded)
            )

        def EndOfPrddvds_dist(shocks, a_nrm, shares):
            Rxs = shocks["Risky"] - self.Rfree
            r_port = self.r_port(shocks, shares)
            m_nrm_next = self.m_nrm_next(shocks, a_nrm, r_port)

            return Rxs * a_nrm * self.uP(shocks["PermShk"] * self.PermGroFac) * dvdm(
                m_nrm_next, shares
            ) + (shocks["PermShk"] * self.PermGroFac) ** (1.0 - self.CRRA) * dvds(
                m_nrm_next, shares
            )

        # Calculate end-of-period marginal value of assets by taking expectations
        self.EndOfPrddvda = (
            self.DiscFac
            * self.LivPrb
            * self.ShockDstn.expected(
                EndOfPrddvda_dists, self.aNrm_tiled, self.Share_tiled
            )
        )

        self.EndOfPrddvdaNvrs = self.uPinv(self.EndOfPrddvda)

        # Calculate end-of-period marginal value of risky portfolio share by taking expectations
        self.EndOfPrddvds = (
            self.DiscFac
            * self.LivPrb
            * self.ShockDstn.expected(
                EndOfPrddvds_dist, self.aNrm_tiled, self.Share_tiled
            )
        )

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

            return (shocks["PermShk"] * self.PermGroFac) ** (1.0 - self.CRRA) * v_next

        self.EndOfPrdv = (
            self.DiscFac
            * self.LivPrb
            * self.ShockDstn.expected(v_dist, self.aNrm_tiled, self.Share_tiled)
        )

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
            aNrm_temp,
            Share_now,
            intercept_limit=self.ShareLimit,
            slope_limit=0.0,
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

# Adjust some of the existing parameters in the dictionary
init_portfolio["aXtraMax"] = 100  # Make the grid of assets go much higher...
init_portfolio["aXtraCount"] = 200  # ...and include many more gridpoints...
# ...which aren't so clustered at the bottom
init_portfolio["aXtraNestFac"] = 1
# Artificial borrowing constraint must be turned on
init_portfolio["BoroCnstArt"] = 0.0
# Results are more interesting with higher risk aversion
init_portfolio["CRRA"] = 5.0
init_portfolio["DiscFac"] = 0.90  # And also lower patience
