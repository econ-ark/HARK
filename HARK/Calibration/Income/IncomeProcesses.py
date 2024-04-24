"""
This file has various classes and functions for constructing income processes.
"""

from HARK.distribution import (
    add_discrete_outcome_constant_mean,
    combine_indep_dstns,
    DiscreteDistribution,
    DiscreteDistributionLabeled,
    IndexDistribution,
    MeanOneLogNormal,
    TimeVaryingDiscreteDistribution,
)


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
        logn_approx = MeanOneLogNormal(sigma).discretize(
            n_approx if sigma > 0.0 else 1, method="equiprobable", tail_N=0
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
        dstn_approx = MeanOneLogNormal(sigma).discretize(
            n_approx if sigma > 0.0 else 1, method="equiprobable", tail_N=0
        )
        if UnempPrb > 0.0:
            dstn_approx = add_discrete_outcome_constant_mean(
                dstn_approx, p=UnempPrb, x=IncUnemp
            )

        super().__init__(pmv=dstn_approx.pmv, atoms=dstn_approx.atoms, seed=seed)


class BufferStockIncShkDstn(DiscreteDistributionLabeled):
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

        super().__init__(
            name="Joint distribution of permanent and transitory shocks to income",
            var_names=["PermShk", "TranShk"],
            pmv=joint_dstn.pmv,
            atoms=joint_dstn.atoms,
            seed=seed,
        )


###############################################################################


def construct_lognormal_income_process_unemployment(
    T_cycle,
    PermShkStd,
    PermShkCount,
    TranShkStd,
    TranShkCount,
    T_retire,
    UnempPrb,
    IncUnemp,
    UnempPrbRet,
    IncUnempRet,
    RNG,
    neutral_measure=False,
):
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

    Parameters (passed as attributes of the input parameters)
    ---------------------------------------------------------
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
    RNG : np.random.RandomState
        Random number generator for this type.
    neutral_measure : bool
        Indicator for whether the permanent-income-neutral measure should be used.

    Returns
    -------
    IncShkDstn :  [distribution.Distribution]
        A list with T_cycle elements, each of which is a
        discrete approximation to the income process in a period.
    """
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
    neutral_measure_list = [neutral_measure] * len(PermShkCount_list)

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
        RNG=RNG,
        seed=RNG.integers(0, 2**31 - 1),
    )
    return IncShkDstn



def get_PermShkDstn_from_IncShkDstn(IncShkDstn, RNG):
    PermShkDstn = [this.make_univariate(0, seed=RNG.integers(0, 2**31 - 1)) for this in IncShkDstn]
    return TimeVaryingDiscreteDistribution(PermShkDstn, seed=RNG.integers(0, 2**31 - 1))


def get_TranShkDstn_from_IncShkDstn(IncShkDstn, RNG):
    TranShkDstn = [this.make_univariate(1, seed=RNG.integers(0, 2**31 - 1)) for this in IncShkDstn]
    return TimeVaryingDiscreteDistribution(TranShkDstn, seed=RNG.integers(0, 2**31 - 1))
