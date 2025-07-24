"""
This file contains tools for creating risky asset return distributions, for use
as inputs to several consumption-saving model solvers.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from HARK.distributions import (
    combine_indep_dstns,
    DiscreteDistributionLabeled,
    IndexDistribution,
    Lognormal,
)


def make_lognormal_RiskyDstn(T_cycle, RiskyAvg, RiskyStd, RiskyCount, RNG):
    r"""
    Creates a discrete approximation of lognormal risky asset returns, either
    as a single distribution or as a lifecycle sequence.

    .. math::
        \begin{align}
        \phi_t &\sim \exp(\mathcal{N}(\textbf{RiskyStd}_{t}^2)) \\
        \mathbb{E}_{t} \left[ \phi_t \right] &= \textbf{RiskyAvg}_{t}\\
        \end{align}

    Parameters
    ----------
    T_cycle : int
        Number of non-terminal periods in this agent's cycle.
    RiskyAvg : float or [float]
        Mean return factor of risky asset. If a single number, it is used for all
        periods. If it is a list, then it represents lifecycle returns (or
        perceptions thereof).
    RiskyStd : float or [float]
        Standard deviation of log returns of the risky asset. Allows the same
        options as RiskyAvg.
    RiskyCount : int
        Number of equiprobable discrete nodes in the risky return distribution.
    RNG : RandomState
        Internal random number generator for the AgentType instance, used to
        generate random seeds.

    Returns
    -------
    RiskyDstn : DiscreteDistribution or [DiscreteDistribution]
        Discretized approximation to lognormal asset returns.
    """
    # Determine whether this instance has time-varying risk perceptions
    if (
        (type(RiskyAvg) is list)
        and (type(RiskyStd) is list)
        and (len(RiskyAvg) == len(RiskyStd))
        and (len(RiskyAvg) == T_cycle)
    ):
        time_varying_RiskyDstn = True
    elif (type(RiskyStd) is list) or (type(RiskyAvg) is list):
        raise AttributeError(
            "If RiskyAvg is time-varying, then RiskyStd must be as well, and they must both have length of T_cycle!"
        )
    else:
        time_varying_RiskyDstn = False

    # Generate a discrete approximation to the risky return distribution
    # if its parameters are time-varying
    if time_varying_RiskyDstn:
        RiskyDstn = IndexDistribution(
            Lognormal,
            {"mean": RiskyAvg, "sigma": RiskyStd},
            seed=RNG.integers(0, 2**31 - 1),
        ).discretize(RiskyCount, method="equiprobable")

    # Generate a discrete approximation to the risky return distribution if
    # its parameters are constant
    else:
        RiskyDstn = Lognormal(mean=RiskyAvg, sigma=RiskyStd).discretize(
            RiskyCount, method="equiprobable"
        )

    return RiskyDstn


def combine_IncShkDstn_and_RiskyDstn(T_cycle, RiskyDstn, IncShkDstn):
    """
    Combine the income shock distribution (over PermShk and TranShk) with the
    risky return distribution (RiskyDstn) to make a new object called ShockDstn.

    Parameters
    ----------
    T_cycle : int
        Number of non-terminal periods in this agent's cycle.
    RiskyDstn : DiscreteDistribution or [DiscreteDistribution]
        Discretized approximation to lognormal asset returns.
    IncShkDstn : [Distribution]
        A discrete approximation to the income process between each period.

    Returns
    -------
    ShockDstn : IndexDistribution
        A combined trivariate discrete distribution of permanent shocks, transitory
        shocks, and risky returns. Has one element per period of the agent's cycle.
    """
    # Create placeholder distributions
    try:
        dstn_list = [
            combine_indep_dstns(IncShkDstn[t], RiskyDstn[t]) for t in range(T_cycle)
        ]
    except:
        dstn_list = [
            combine_indep_dstns(IncShkDstn[t], RiskyDstn) for t in range(T_cycle)
        ]

    # Names of the variables (hedging for the unlikely case that in
    # some index of IncShkDstn variables are in a switched order)
    names_list = [
        list(IncShkDstn[t].variables.keys()) + ["Risky"] for t in range(T_cycle)
    ]

    conditional = {
        "pmv": [x.pmv for x in dstn_list],
        "atoms": [x.atoms for x in dstn_list],
        "var_names": names_list,
    }

    # Now create the actual distribution using the index and labeled class
    ShockDstn = IndexDistribution(
        engine=DiscreteDistributionLabeled,
        conditional=conditional,
    )
    return ShockDstn


def calc_ShareLimit_for_CRRA(T_cycle, RiskyDstn, CRRA, Rfree):
    """
    Calculates the lower bound on the risky asset share as market resources go
    to infinity, given that utility is CRRA.

    Parameters
    ----------
    T_cycle : int
        Number of non-terminal periods in this agent's cycle.
    RiskyDstn : DiscreteDistribution or [DiscreteDistribution]
        Discretized approximation to lognormal asset returns.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Return factor on the risk-free asset.

    Returns
    -------
    ShareLimit : float or [float]
        Lower bound on risky asset share. Can be a single number or a lifecycle sequence.
    """
    RiskyDstn_is_time_varying = hasattr(RiskyDstn, "__getitem__")
    Rfree_is_time_varying = type(Rfree) is list

    # If the risky share lower bound is time varying...
    if RiskyDstn_is_time_varying or Rfree_is_time_varying:
        ShareLimit = []
        for t in range(T_cycle):
            if RiskyDstn_is_time_varying:
                RiskyDstn_t = RiskyDstn[t]
            else:
                RiskyDstn_t = RiskyDstn
            if Rfree_is_time_varying:
                Rfree_t = Rfree[t]
            else:
                Rfree_t = Rfree

            def temp_f(s):
                return -((1.0 - CRRA) ** -1) * np.dot(
                    (Rfree_t + s * (RiskyDstn_t.atoms[0] - Rfree_t)) ** (1.0 - CRRA),
                    RiskyDstn_t.pmv,
                )

            SharePF = minimize_scalar(temp_f, bounds=(0.0, 1.0), method="bounded").x
            ShareLimit.append(SharePF)

    # If the risky share lower bound is not time-varying...
    else:

        def temp_f(s):
            return -((1.0 - CRRA) ** -1) * np.dot(
                (Rfree + s * (RiskyDstn.atoms[0] - Rfree)) ** (1.0 - CRRA),
                RiskyDstn.pmv,
            )

        SharePF = minimize_scalar(
            temp_f, bracket=(0.0, 1.0), method="golden", tol=1e-10
        ).x
        if type(SharePF) is np.array:
            SharePF = SharePF[0]
        ShareLimit = SharePF

    return ShareLimit
