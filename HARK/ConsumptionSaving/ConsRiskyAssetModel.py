"""
This file contains a class that adds a risky asset with a log-normal return
factor to IndShockConsumerType.
This class is not a fully specified model and therefore has no solution or
simulation methods. It is meant as a container of methods for dealing with
risky assets that will be useful to models what will inherit from it.
"""

import numpy as np
from scipy.optimize import minimize_scalar

from HARK import NullFunc
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsumerSolution,
    IndShockConsumerType,  # PortfolioConsumerType inherits from it;
    init_idiosyncratic_shocks,  # Baseline dictionary to build on
)
from HARK.distribution import (
    Bernoulli,
    DiscreteDistributionLabeled,
    expected,
    IndexDistribution,
    Lognormal,
    combine_indep_dstns,
)
from HARK.interpolation import (
    ConstantFunction,
    LinearInterp,
    LowerEnvelope,
    CubicInterp,
    MargMargValueFuncCRRA,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.rewards import UtilityFuncCRRA


class IndShockRiskyAssetConsumerType(IndShockConsumerType):
    """
    A consumer type that has access to a risky asset for his savings. The
    risky asset has lognormal returns that are possibly correlated with his
    income shocks.

    There is a friction that prevents the agent from adjusting his portfolio
    at any given period with an exogenously given probability.
    The meaning of "adjusting his portfolio" depends on the particular model.
    """

    time_inv_ = IndShockConsumerType.time_inv_ + ["PortfolioBisect"]
    shock_vars_ = IndShockConsumerType.shock_vars_ + ["Adjust", "Risky"]

    def __init__(self, verbose=False, quiet=False, **kwds):
        params = init_risky_asset.copy()
        params.update(kwds)
        kwds = params

        # Boolean determines whether agent will use portfolio
        # optimization or only has access to risky asset
        if not hasattr(self, "PortfolioBool"):
            self.PortfolioBool = False

        if not hasattr(self, "PortfolioBisect"):
            self.PortfolioBisect = False

        # Boolean determines whether, when simulating a given time period,
        # all agents will draw the same risky return factor (true by default)
        if not hasattr(self, "sim_common_Rrisky"):
            self.sim_common_Rrisky = True

        # Initialize a basic consumer type
        IndShockConsumerType.__init__(self, verbose=verbose, quiet=quiet, **kwds)

        # These method must be overwritten by classes that inherit from
        # RiskyAssetConsumerType
        self.solve_one_period = solve_one_period_ConsIndShockRiskyAsset

    def pre_solve(self):
        self.update_solution_terminal()

        if self.PortfolioBool:
            self.solution_terminal.ShareFunc = ConstantFunction(1.0)

    def update(self):
        IndShockConsumerType.update(self)
        self.update_AdjustDstn()
        self.update_RiskyDstn()
        self.update_ShockDstn()

        if self.PortfolioBool:
            self.update_ShareLimit()
            self.update_ShareGrid()

    def update_RiskyDstn(self):
        """
        Creates the attributes RiskyDstn from the primitive attributes RiskyAvg,
        RiskyStd, and RiskyCount, approximating the (perceived) distribution of
        returns in each period of the cycle.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Determine whether this instance has time-varying risk perceptions
        if (
            (type(self.RiskyAvg) is list)
            and (type(self.RiskyStd) is list)
            and (len(self.RiskyAvg) == len(self.RiskyStd))
            and (len(self.RiskyAvg) == self.T_cycle)
        ):
            self.add_to_time_vary("RiskyAvg", "RiskyStd")
        elif (type(self.RiskyStd) is list) or (type(self.RiskyAvg) is list):
            raise AttributeError(
                "If RiskyAvg is time-varying, then RiskyStd must be as well, and they must both have length of T_cycle!"
            )
        else:
            self.add_to_time_inv("RiskyAvg", "RiskyStd")

        # Generate a discrete approximation to the risky return distribution
        # if its parameters are time-varying
        if "RiskyAvg" in self.time_vary:
            self.RiskyDstn = IndexDistribution(
                Lognormal.from_mean_std,
                {"mean": self.RiskyAvg, "std": self.RiskyStd},
                seed=self.RNG.integers(0, 2**31 - 1),
            ).discretize(self.RiskyCount, method="equiprobable")

            self.add_to_time_vary("RiskyDstn")

        # Generate a discrete approximation to the risky return distribution if
        # its parameters are constant
        else:
            self.RiskyDstn = Lognormal.from_mean_std(
                self.RiskyAvg, self.RiskyStd
            ).discretize(self.RiskyCount, method="equiprobable")
            self.add_to_time_inv("RiskyDstn")

    def update_ShockDstn(self):
        """
        Combine the income shock distribution (over PermShk and TranShk) with the
        risky return distribution (RiskyDstn) to make a new attribute called ShockDstn.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # Create placeholder distributions
        if "RiskyDstn" in self.time_vary:
            dstn_list = [
                combine_indep_dstns(self.IncShkDstn[t], self.RiskyDstn[t])
                for t in range(self.T_cycle)
            ]
        else:
            dstn_list = [
                combine_indep_dstns(self.IncShkDstn[t], self.RiskyDstn)
                for t in range(self.T_cycle)
            ]

        # Names of the variables (hedging for the unlikely case that in
        # some index of IncShkDstn variables are in a switched order)
        names_list = [
            list(self.IncShkDstn[t].variables.keys()) + ["Risky"]
            for t in range(self.T_cycle)
        ]

        conditional = {
            "pmv": [x.pmv for x in dstn_list],
            "atoms": [x.atoms for x in dstn_list],
            "var_names": names_list,
        }

        # Now create the actual distribution using the index and labeled class
        self.ShockDstn = IndexDistribution(
            engine=DiscreteDistributionLabeled,
            conditional=conditional,
        )

        self.add_to_time_vary("ShockDstn")

        # Mark whether the risky returns and income shocks are independent (they are)
        self.IndepDstnBool = True
        self.add_to_time_inv("IndepDstnBool")

    def update_AdjustDstn(self):
        """
        Checks and updates the exogenous probability of the agent being allowed
        to rebalance his portfolio/contribution scheme. It can be time varying.

        Parameters
        ------
        None.

        Returns
        -------
        None.

        """
        if type(self.AdjustPrb) is list and (len(self.AdjustPrb) == self.T_cycle):
            self.add_to_time_vary("AdjustPrb")

            self.AdjustDstn = IndexDistribution(
                Bernoulli, {"p": self.AdjustPrb}, seed=self.RNG.integers(0, 2**31 - 1)
            )

        elif type(self.AdjustPrb) is list:
            raise AttributeError(
                "If AdjustPrb is time-varying, it must have length of T_cycle!"
            )
        else:
            self.add_to_time_inv("AdjustPrb")
            self.AdjustDstn = Bernoulli(
                p=self.AdjustPrb, seed=self.RNG.integers(0, 2**31 - 1)
            )

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
        if "RiskyDstn" in self.time_vary or "Rfree" in self.time_vary:
            self.ShareLimit = []
            for t in range(self.T_cycle):
                if "RiskyDstn" in self.time_vary:
                    RiskyDstn = self.RiskyDstn[t]
                else:
                    RiskyDstn = self.RiskyDstn
                if "Rfree" in self.time_vary:
                    Rfree = self.Rfree[t]
                else:
                    Rfree = self.Rfree

                def temp_f(s):
                    return -((1.0 - self.CRRA) ** -1) * np.dot(
                        (Rfree + s * (RiskyDstn.atoms - Rfree)) ** (1.0 - self.CRRA),
                        RiskyDstn.pmv,
                    )

                SharePF = minimize_scalar(temp_f, bounds=(0.0, 1.0), method="bounded").x
                self.ShareLimit.append(SharePF)
            self.add_to_time_vary("ShareLimit")

        else:
            RiskyDstn = self.RiskyDstn

            def temp_f(s):
                return -((1.0 - self.CRRA) ** -1) * np.dot(
                    (self.Rfree + s * (RiskyDstn.atoms - self.Rfree))
                    ** (1.0 - self.CRRA),
                    RiskyDstn.pmv,
                )

            SharePF = minimize_scalar(temp_f, bounds=(0.0, 1.0), method="bounded").x
            self.ShareLimit = SharePF
            self.add_to_time_inv("ShareLimit")

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

        RfreeNow = super().get_Rfree()

        Rport = (
            self.controls["Share"] * self.shocks["Risky"]
            + (1.0 - self.controls["Share"]) * RfreeNow
        )
        self.Rport = Rport
        return Rport

    def get_Risky(self):
        """
        Draws a new risky return factor.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # How we draw the shocks depends on whether their distribution is time-varying
        if "RiskyDstn" in self.time_vary:
            if self.sim_common_Rrisky:
                raise AttributeError(
                    "If sim_common_Rrisky is True, RiskyDstn cannot be time-varying!"
                )

            else:
                # Make use of the IndexDistribution.draw() method
                self.shocks["Risky"] = self.RiskyDstn.draw(
                    np.maximum(self.t_cycle - 1, 0)
                    if self.cycles == 1
                    else self.t_cycle
                )

        else:
            # Draw either a common economy-wide return, or one for each agent
            if self.sim_common_Rrisky:
                self.shocks["Risky"] = self.RiskyDstn.draw(1)
            else:
                self.shocks["Risky"] = self.RiskyDstn.draw(self.AgentCount)

    def get_Adjust(self):
        """
        Sets the attribute Adjust as a boolean array of size AgentCount, indicating
        whether each agent is able to adjust their risky portfolio share this period.
        Uses the attribute AdjustPrb to draw from a Bernoulli distribution.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if "AdjustPrb" in self.time_vary:
            self.shocks["Adjust"] = self.AdjustDstn.draw(
                np.maximum(self.t_cycle - 1, 0) if self.cycles == 1 else self.t_cycle
            )
        else:
            self.shocks["Adjust"] = self.AdjustDstn.draw(self.AgentCount)

    def initialize_sim(self):
        """
        Initialize the state of simulation attributes.  Simply calls the same
        method for IndShockConsumerType, then initializes the new states/shocks
        Adjust and Share.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.shocks["Adjust"] = np.zeros(self.AgentCount, dtype=bool)
        IndShockConsumerType.initialize_sim(self)

    def get_shocks(self):
        """
        Draw idiosyncratic income shocks, just as for IndShockConsumerType, then draw
        a single common value for the risky asset return.  Also draws whether each
        agent is able to adjust their portfolio this period.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        IndShockConsumerType.get_shocks(self)
        self.get_Risky()
        self.get_Adjust()


# This is to preserve compatibility with old name
RiskyAssetConsumerType = IndShockRiskyAssetConsumerType



####################################################################################################
####################################################################################################


def solve_one_period_ConsIndShockRiskyAsset(
    solution_next,
    IncShkDstn,
    RiskyDstn,
    ShockDstn,
    LivPrb,
    DiscFac,
    CRRA,
    PermGroFac,
    BoroCnstArt,
    aXtraGrid,
    vFuncBool,
    CubicBool,
    IndepDstnBool,
):
    """
    Solves one period of a consumption-saving model with idiosyncratic shocks to
    permanent and transitory income, with one risky asset and CRRA utility.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncShkDstn : Distribution
        Discrete distribution of permanent income shocks and transitory income
        shocks. This is only used if the input IndepDstnBool is True, indicating
        that income and return distributions are independent.
    RiskyDstn : Distribution
       Distribution of risky asset returns. This is only used if the input
       IndepDstnBool is True, indicating that income and return distributions
       are independent.
    ShockDstn : Distribution
        Joint distribution of permanent income shocks, transitory income shocks,
        and risky returns.  This is only used if the input IndepDstnBool is False,
        indicating that income and return distributions can't be assumed to be
        independent.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion.
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
        An indicator for whether the solver should use cubic or linear interpolation.
    IndepDstnBool : bool
        Indicator for whether the income and risky return distributions are in-
        dependent of each other, which can speed up the expectations step.

    Returns
    -------
    solution_now : ConsumerSolution
        Solution to this period's consumption-saving problem with income risk.
    """
    # Do a quick validity check; don't want to allow borrowing with risky returns
    if BoroCnstArt != 0.0:
        raise ValueError("RiskyAssetConsumerType must have BoroCnstArt=0.0!")

    # Define the current period utility function and effective discount factor
    uFunc = UtilityFuncCRRA(CRRA)
    DiscFacEff = DiscFac * LivPrb  # "effective" discount factor

    # Unpack next period's income shock distribution
    ShkPrbsNext = ShockDstn.pmv
    PermShkValsNext = ShockDstn.atoms[0]
    TranShkValsNext = ShockDstn.atoms[1]
    RiskyValsNext = ShockDstn.atoms[2]
    PermShkMinNext = np.min(PermShkValsNext)
    TranShkMinNext = np.min(TranShkValsNext)
    RiskyMinNext = np.min(RiskyValsNext)
    RiskyMaxNext = np.max(RiskyValsNext)

    # Unpack next period's (marginal) value function
    vFuncNext = solution_next.vFunc  # This is None when vFuncBool is False
    vPfuncNext = solution_next.vPfunc
    vPPfuncNext = solution_next.vPPfunc  # This is None when CubicBool is False

    # Perform an alternate calculation of the absolute patience factor when
    # returns are risky
    def calc_Radj(R):
        return R ** (1.0 - CRRA)

    PatFac = (DiscFacEff * expected(calc_Radj, RiskyDstn)) ** (1.0 / CRRA)
    MPCminNow = 1.0 / (1.0 + PatFac / solution_next.MPCmin)

    # Also perform an alternate calculation for human wealth under risky returns
    def calc_hNrm(S):
        Risky = S["Risky"]
        PermShk = S["PermShk"]
        TranShk = S["TranShk"]
        hNrm = (PermGroFac / Risky) * (PermShk * TranShk + solution_next.hNrm)
        return hNrm

    hNrmNow = expected(calc_hNrm, ShockDstn)

    # The above attempts to pin down the limiting consumption function for this
    # model, however it is not clear why it creates bugs, so for now we allow
    # for a linear extrapolation beyond the last asset point
    cFuncLimitIntercept = None
    cFuncLimitSlope = None

    # Calculate the minimum allowable value of market resources in this period
    BoroCnstNat_cand = (
        (solution_next.mNrmMin - TranShkValsNext)
        * (PermGroFac * PermShkValsNext)
        / RiskyValsNext
    )
    BoroCnstNat = np.max(BoroCnstNat_cand)  # Must be at least this

    # Set a flag for whether the natural borrowing constraint is zero, which
    # depends on whether the smallest transitory income shock is zero
    BoroCnstNat_iszero = np.min(IncShkDstn.atoms[1]) == 0.0

    # Set the minimum allowable (normalized) market resources based on the natural
    # and artificial borrowing constraints
    if BoroCnstArt is None:
        mNrmMinNow = BoroCnstNat
    else:
        mNrmMinNow = np.max([BoroCnstNat, BoroCnstArt])

    # The MPCmax code is a bit unusual here, and possibly "harmlessly wrong".
    # The "worst event" should depend on the risky return factor as well as
    # income shocks. However, the natural borrowing constraint is only ever
    # relevant in this model when it's zero, so the MPC at mNrm is only relevant
    # in the case where risky returns don't matter at all (because a=0).

    # Calculate the probability that we get the worst possible income draw
    IncNext = PermShkValsNext * TranShkValsNext
    WorstIncNext = PermShkMinNext * TranShkMinNext
    WorstIncPrb = np.sum(ShkPrbsNext[IncNext == WorstIncNext])
    # WorstIncPrb is the "Weierstrass p" concept: the odds we get the WORST thing

    # Update the upper bounding MPC as market resources approach the lower bound
    temp_fac = (WorstIncPrb ** (1.0 / CRRA)) * PatFac
    MPCmaxNow = 1.0 / (1.0 + temp_fac / solution_next.MPCmax)

    # Set the upper limit of the MPC (at mNrmMinNow) based on whether the natural
    # or artificial borrowing constraint actually binds
    if BoroCnstNat < mNrmMinNow:
        MPCmaxEff = 1.0  # If actually constrained, MPC near limit is 1
    else:
        MPCmaxEff = MPCmaxNow  # Otherwise, it's the MPC calculated above

    # Define the borrowing-constrained consumption function
    cFuncNowCnst = LinearInterp(
        np.array([mNrmMinNow, mNrmMinNow + 1.0]), np.array([0.0, 1.0])
    )

    # Big methodological split here: whether the income and return distributions are independent.
    # Calculation of end-of-period marginal (marginal) value uses different approaches
    if IndepDstnBool:
        # bNrm represents R*a, balances after asset return shocks but before income.
        # This just uses the highest risky return as a rough shifter for the aXtraGrid.
        if BoroCnstNat_iszero:
            bNrmNow = np.insert(
                RiskyMaxNext * aXtraGrid, 0, RiskyMinNext * aXtraGrid[0]
            )
        else:
            # Add a bank balances point at exactly zero
            bNrmNow = RiskyMaxNext * np.insert(aXtraGrid, 0, 0.0)
        aNrmNow = aXtraGrid

        # Define local functions for taking future expectations when the interest
        # factor *is* independent from the income shock distribution. These go
        # from "bank balances" bNrm = R * aNrm to t+1 realizations.
        def calc_mNrmNext(S, b):
            return b / (PermGroFac * S["PermShk"]) + S["TranShk"]

        def calc_vNext(S, b):
            return S["PermShk"] ** (1.0 - CRRA) * vFuncNext(calc_mNrmNext(S, b))

        def calc_vPnext(S, b):
            return S["PermShk"] ** (-CRRA) * vPfuncNext(calc_mNrmNext(S, b))

        def calc_vPPnext(S, b):
            return S["PermShk"] ** (-CRRA - 1.0) * vPPfuncNext(calc_mNrmNext(S, b))

        # Calculate marginal value of bank balances at each gridpoint
        vPfacEff = PermGroFac ** (-CRRA)
        Intermed_vP = vPfacEff * expected(calc_vPnext, IncShkDstn, args=(bNrmNow))
        Intermed_vPnvrs = uFunc.derinv(Intermed_vP, order=(1, 0))
        if BoroCnstNat_iszero:
            Intermed_vPnvrs = np.insert(Intermed_vPnvrs, 0, 0.0)
            bNrm_temp = np.insert(bNrmNow, 0, 0.0)
        else:
            bNrm_temp = bNrmNow.copy()

        # If using cubic spline interpolation, also calculate "intermediate"
        # marginal marginal value of bank balances
        if CubicBool:
            vPPfacEff = PermGroFac ** (-CRRA - 1.0)
            Intermed_vPP = vPPfacEff * expected(
                calc_vPPnext, IncShkDstn, args=(bNrmNow)
            )
            Intermed_vPnvrsP = Intermed_vPP * uFunc.derinv(Intermed_vP, order=(1, 1))
            if BoroCnstNat_iszero:
                Intermed_vPnvrsP = np.insert(Intermed_vPnvrsP, 0, Intermed_vPnvrsP[0])

            # Make a cubic spline intermediate pseudo-inverse marginal value function
            Intermed_vPnvrsFunc = CubicInterp(
                bNrm_temp, Intermed_vPnvrs, Intermed_vPnvrsP
            )
            Intermed_vPPfunc = MargMargValueFuncCRRA(Intermed_vPnvrsFunc, CRRA)
        else:
            # Make a linear interpolation intermediate pseudo-inverse marginal value function
            Intermed_vPnvrsFunc = LinearInterp(bNrm_temp, Intermed_vPnvrs)

        # "Recurve" the intermediate pseudo-inverse marginal value function
        Intermed_vPfunc = MargValueFuncCRRA(Intermed_vPnvrsFunc, CRRA)

        # If the value function is requested, calculate "intermediate" value
        if vFuncBool:
            vFacEff = PermGroFac ** (1.0 - CRRA)
            Intermed_v = vFacEff * expected(calc_vNext, IncShkDstn, args=(bNrmNow))
            Intermed_vNvrs = uFunc.inv(Intermed_v)
            # value transformed through inverse utility
            Intermed_vNvrsP = Intermed_vP * uFunc.derinv(Intermed_v, order=(0, 1))
            if BoroCnstNat_iszero:
                Intermed_vNvrs = np.insert(Intermed_vNvrs, 0, 0.0)
                Intermed_vNvrsP = np.insert(Intermed_vNvrsP, 0, Intermed_vNvrsP[0])
                # This is a very good approximation, vNvrsPP = 0 at the asset minimum

            # Make a cubic spline intermediate pseudo-inverse value function
            Intermed_vNvrsFunc = CubicInterp(bNrm_temp, Intermed_vNvrs, Intermed_vNvrsP)

            # "Recurve" the intermediate pseudo-inverse value function
            Intermed_vFunc = ValueFuncCRRA(Intermed_vNvrsFunc, CRRA)

        # We have "intermediate" (marginal) value functions defined over bNrm,
        # so now we want to take expectations over Risky realizations at each aNrm.

        # Begin by re-defining transition functions for taking expectations, which are all very simple!
        def calc_bNrmNext(R, a):
            return R * a

        def calc_vNext(R, a):
            return Intermed_vFunc(calc_bNrmNext(R, a))

        def calc_vPnext(R, a):
            return R * Intermed_vPfunc(calc_bNrmNext(R, a))

        def calc_vPPnext(R, a):
            return R * R * Intermed_vPPfunc(calc_bNrmNext(R, a))

        # Calculate end-of-period marginal value of assets at each gridpoint
        EndOfPrdvP = DiscFacEff * expected(calc_vPnext, RiskyDstn, args=(aNrmNow))

        # Invert the first order condition to find optimal cNrm from each aNrm gridpoint
        cNrmNow = uFunc.derinv(EndOfPrdvP, order=(1, 0))
        mNrmNow = cNrmNow + aNrmNow  # Endogenous mNrm gridpoints

        # Calculate the MPC at each gridpoint if using cubic spline interpolation
        if CubicBool:
            # Calculate end-of-period marginal marginal value of assets at each gridpoint
            EndOfPrdvPP = DiscFacEff * expected(calc_vPPnext, RiskyDstn, args=(aNrmNow))
            dcda = EndOfPrdvPP / uFunc.der(np.array(cNrmNow), order=2)
            MPC = dcda / (dcda + 1.0)
            MPC_for_interpolation = np.insert(MPC, 0, MPCmaxNow)

        # Limiting consumption is zero as m approaches mNrmMin
        c_for_interpolation = np.insert(cNrmNow, 0, 0.0)
        m_for_interpolation = np.insert(mNrmNow, 0, BoroCnstNat)

        # Construct the end-of-period value function if requested
        if vFuncBool:
            # Calculate end-of-period value, its derivative, and their pseudo-inverse
            EndOfPrdv = DiscFacEff * expected(calc_vNext, RiskyDstn, args=(aNrmNow))
            EndOfPrdvNvrs = uFunc.inv(EndOfPrdv)
            # value transformed through inverse utility
            EndOfPrdvNvrsP = EndOfPrdvP * uFunc.derinv(EndOfPrdv, order=(0, 1))

            # Construct the end-of-period value function
            if BoroCnstNat_iszero:
                EndOfPrdvNvrs = np.insert(EndOfPrdvNvrs, 0, 0.0)
                EndOfPrdvNvrsP = np.insert(EndOfPrdvNvrsP, 0, EndOfPrdvNvrsP[0])
                # This is a very good approximation, vNvrsPP = 0 at the asset minimum
                aNrm_temp = np.insert(aNrmNow, 0, BoroCnstNat)
            else:
                aNrm_temp = aNrmNow.copy()
            EndOfPrd_vNvrsFunc = CubicInterp(aNrm_temp, EndOfPrdvNvrs, EndOfPrdvNvrsP)
            EndOfPrd_vFunc = ValueFuncCRRA(EndOfPrd_vNvrsFunc, CRRA)

    # NON-INDEPENDENT METHOD BEGINS HERE
    else:
        # Construct the assets grid by adjusting aXtra by the natural borrowing constraint
        # aNrmNow = np.asarray(aXtraGrid) + BoroCnstNat
        if BoroCnstNat_iszero:
            aNrmNow = aXtraGrid
        else:
            # Add an asset point at exactly zero
            aNrmNow = np.insert(aXtraGrid, 0, 0.0)

        # Define local functions for taking future expectations when the interest
        # factor is *not* independent from the income shock distribution
        def calc_mNrmNext(S, a):
            return S["Risky"] / (PermGroFac * S["PermShk"]) * a + S["TranShk"]

        def calc_vNext(S, a):
            return S["PermShk"] ** (1.0 - CRRA) * vFuncNext(calc_mNrmNext(S, a))

        def calc_vPnext(S, a):
            return (
                S["Risky"] * S["PermShk"] ** (-CRRA) * vPfuncNext(calc_mNrmNext(S, a))
            )

        def calc_vPPnext(S, a):
            return (
                (S["Risky"] ** 2)
                * S["PermShk"] ** (-CRRA - 1.0)
                * vPPfuncNext(calc_mNrmNext(S, a))
            )

        # Calculate end-of-period marginal value of assets at each gridpoint
        vPfacEff = DiscFacEff * PermGroFac ** (-CRRA)
        EndOfPrdvP = vPfacEff * expected(calc_vPnext, ShockDstn, args=(aNrmNow))

        # Invert the first order condition to find optimal cNrm from each aNrm gridpoint
        cNrmNow = uFunc.derinv(EndOfPrdvP, order=(1, 0))
        mNrmNow = cNrmNow + aNrmNow  # Endogenous mNrm gridpoints

        # Calculate the MPC at each gridpoint if using cubic spline interpolation
        if CubicBool:
            # Calculate end-of-period marginal marginal value of assets at each gridpoint
            vPPfacEff = DiscFacEff * PermGroFac ** (-CRRA - 1.0)
            EndOfPrdvPP = vPPfacEff * expected(calc_vPPnext, ShockDstn, args=(aNrmNow))
            dcda = EndOfPrdvPP / uFunc.der(np.array(cNrmNow), order=2)
            MPC = dcda / (dcda + 1.0)
            MPC_for_interpolation = np.insert(MPC, 0, MPCmaxNow)

        # Limiting consumption is zero as m approaches mNrmMin
        c_for_interpolation = np.insert(cNrmNow, 0, 0.0)
        m_for_interpolation = np.insert(mNrmNow, 0, BoroCnstNat)

        # Construct the end-of-period value function if requested
        if vFuncBool:
            # Calculate end-of-period value, its derivative, and their pseudo-inverse
            vFacEff = DiscFacEff * PermGroFac ** (1.0 - CRRA)
            EndOfPrdv = vFacEff * expected(calc_vNext, ShockDstn, args=(aNrmNow))
            EndOfPrdvNvrs = uFunc.inv(EndOfPrdv)
            # value transformed through inverse utility
            EndOfPrdvNvrsP = EndOfPrdvP * uFunc.derinv(EndOfPrdv, order=(0, 1))

            # Construct the end-of-period value function
            if BoroCnstNat_iszero:
                EndOfPrdvNvrs = np.insert(EndOfPrdvNvrs, 0, 0.0)
                EndOfPrdvNvrsP = np.insert(EndOfPrdvNvrsP, 0, EndOfPrdvNvrsP[0])
                # This is a very good approximation, vNvrsPP = 0 at the asset minimum
                aNrm_temp = np.insert(aNrmNow, 0, BoroCnstNat)
            else:
                aNrm_temp = aNrmNow.copy()
            EndOfPrd_vNvrsFunc = CubicInterp(aNrm_temp, EndOfPrdvNvrs, EndOfPrdvNvrsP)
            EndOfPrd_vFunc = ValueFuncCRRA(EndOfPrd_vNvrsFunc, CRRA)

    # Construct the consumption function; this uses the same method whether the
    # income distribution is independent from the return distribution
    if CubicBool:
        # Construct the unconstrained consumption function as a cubic interpolation
        cFuncNowUnc = CubicInterp(
            m_for_interpolation,
            c_for_interpolation,
            MPC_for_interpolation,
            cFuncLimitIntercept,
            cFuncLimitSlope,
        )
    else:
        # Construct the unconstrained consumption function as a linear interpolation
        cFuncNowUnc = LinearInterp(
            m_for_interpolation,
            c_for_interpolation,
            cFuncLimitIntercept,
            cFuncLimitSlope,
        )

    # Combine the constrained and unconstrained functions into the true consumption function.
    # LowerEnvelope should only be used when BoroCnstArt is True
    cFuncNow = LowerEnvelope(cFuncNowUnc, cFuncNowCnst, nan_bool=False)

    # Make the marginal value function and the marginal marginal value function
    vPfuncNow = MargValueFuncCRRA(cFuncNow, CRRA)

    # Define this period's marginal marginal value function
    if CubicBool:
        vPPfuncNow = MargMargValueFuncCRRA(cFuncNow, CRRA)
    else:
        vPPfuncNow = NullFunc()  # Dummy object

    # Construct this period's value function if requested. This version is set
    # up for the non-independent distributions, need to write a faster version.
    if vFuncBool:
        # Compute expected value and marginal value on a grid of market resources
        mNrm_temp = mNrmMinNow + aXtraGrid
        cNrm_temp = cFuncNow(mNrm_temp)
        aNrm_temp = mNrm_temp - cNrm_temp
        v_temp = uFunc(cNrm_temp) + EndOfPrd_vFunc(aNrm_temp)
        vP_temp = uFunc.der(cNrm_temp)

        # Construct the beginning-of-period value function
        vNvrs_temp = uFunc.inv(v_temp)  # value transformed through inv utility
        vNvrsP_temp = vP_temp * uFunc.derinv(v_temp, order=(0, 1))
        mNrm_temp = np.insert(mNrm_temp, 0, mNrmMinNow)
        vNvrs_temp = np.insert(vNvrs_temp, 0, 0.0)
        vNvrsP_temp = np.insert(vNvrsP_temp, 0, MPCmaxEff ** (-CRRA / (1.0 - CRRA)))
        # MPCminNvrs = MPCminNow ** (-CRRA / (1.0 - CRRA))
        vNvrsFuncNow = CubicInterp(mNrm_temp, vNvrs_temp, vNvrsP_temp)
        vFuncNow = ValueFuncCRRA(vNvrsFuncNow, CRRA)
    else:
        vFuncNow = NullFunc()  # Dummy object

    # Create and return this period's solution
    solution_now = ConsumerSolution(
        cFunc=cFuncNow,
        vFunc=vFuncNow,
        vPfunc=vPfuncNow,
        vPPfunc=vPPfuncNow,
        mNrmMin=mNrmMinNow,
        hNrm=hNrmNow,
        MPCmin=MPCminNow,
        MPCmax=MPCmaxEff,
    )
    return solution_now


# Initial parameter sets

# Base risky asset dictionary

risky_asset_parms = {
    # Risky return factor moments. Based on SP500 real returns from Shiller's
    # "chapter 26" data, which can be found at https://www.econ.yale.edu/~shiller/data.htm
    "RiskyAvg": 1.080370891,
    "RiskyStd": 0.177196585,
    # Number of integration nodes to use in approximation of risky returns
    "RiskyCount": 5,
    # Probability that the agent can adjust their portfolio each period
    "AdjustPrb": 1.0,
    # When simulating the model, should all agents get the same risky return in
    # a given period?
    "sim_common_Rrisky": True,
}

# Make a dictionary to specify a risky asset consumer type
init_risky_asset = init_idiosyncratic_shocks.copy()
init_risky_asset.update(risky_asset_parms)
# Number of discrete points in the risky share approximation
init_risky_asset["ShareCount"] = 25

init_risky_share_fixed = init_risky_asset.copy()
init_risky_share_fixed["RiskyShareFixed"] = [0.0]
