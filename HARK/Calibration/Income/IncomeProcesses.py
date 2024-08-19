"""
This file has various classes and functions for constructing income processes.
"""

import numpy as np
from HARK.metric import MetricObject
from HARK.distribution import (
    add_discrete_outcome_constant_mean,
    combine_indep_dstns,
    DiscreteDistribution,
    DiscreteDistributionLabeled,
    IndexDistribution,
    MeanOneLogNormal,
    TimeVaryingDiscreteDistribution,
    Lognormal,
    Uniform,
)
from HARK.interpolation import IdentityFunction, LinearInterp
from HARK.utilities import get_percentiles, make_polynomial_params


class BinaryIncShkDstn(DiscreteDistribution):
    """
    A one period income shock distribution (transitory, permanent, or other)
    with only two outcomes. One probability and value are specified, and the
    other is implied to make it a mean one distribution.

    Parameters
    ----------
    shk_prob : float
        Probability of one of the income shock outcomes.
    shk_val : float
        Value of the specified income shock outcome.
    seed : int, optional
        Random seed. The default is 0.

    Returns
    -------
    ShkDstn : DiscreteDistribution
        Binary income shock distribuion.
    """

    def __init__(self, shk_prob, shk_val, seed=0):
        if shk_prob > 1.0 or shk_prob < 0.0:
            raise ValueError("Shock probability must be between 0 and 1!")

        other_prob = 1.0 - shk_prob
        other_val = (1.0 - shk_prob * shk_val) / other_prob
        probs = [shk_prob, other_prob]
        vals = [shk_val, other_val]
        super().__init__(pmv=probs, atoms=vals, seed=seed)


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
        Whether to use Harmenberg's permanent-income-neutral measure. The default is False.
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


class MixtureTranIncShk_HANK(DiscreteDistribution):
    """
    A one-period distribution for transitory income shocks that are a mixture
    between a log-normal and a single-value unemployment shock. This version
    has additional parameters that makes it useful for HANK models.

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
    tax_rate : float
        Flat tax rate on labor income.
    labor : float
        Intensive margin labor supply.
    wage : float
        Wage rate scaling factor.
    seed : int, optional
        Random seed. The default is 0.
    Returns
    -------
    TranShkDstn : DiscreteDistribution
        Transitory income shock distribution.
    """

    def __init__(
        self,
        sigma,
        UnempPrb,
        IncUnemp,
        n_approx,
        wage,
        labor,
        tax_rate,
        seed=0,
    ):
        dstn_approx = MeanOneLogNormal(sigma).discretize(
            n_approx if sigma > 0.0 else 1, method="equiprobable", tail_N=0
        )

        if UnempPrb > 0.0:
            dstn_approx = add_discrete_outcome_constant_mean(
                dstn_approx, p=UnempPrb, x=IncUnemp
            )
        # Rescale the transitory shock values to account for new features
        TranShkMean_temp = (1.0 - tax_rate) * labor * wage
        dstn_approx.atoms *= TranShkMean_temp
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


class IncShkDstn_HANK(DiscreteDistributionLabeled):
    """
    A one-period distribution object for the joint distribution of income
    shocks (permanent and transitory), as modeled in the Buffer Stock Theory
    paper:
    - Lognormal, discretized permanent income shocks.
    - Transitory shocks that are a mixture of:
    - A lognormal distribution in normal times.
    - An "unemployment" shock.

    This version has additional features that make it particularly useful for HANK models.

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
    tax_rate : float
        Flat tax rate on labor income.
    labor : float
        Intensive margin labor supply.
    wage : float
        Wage rate scaling factor.
    neutral_measure : Bool, optional
        Whether to use Harmenberg's permanent-income-neutral measure. The default is False.
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
        tax_rate,
        labor,
        wage,
        neutral_measure=False,
        seed=0,
    ):
        perm_dstn = LognormPermIncShk(
            sigma=sigma_Perm, n_approx=n_approx_Perm, neutral_measure=neutral_measure
        )
        tran_dstn = MixtureTranIncShk_HANK(
            sigma=sigma_Tran,
            UnempPrb=UnempPrb,
            IncUnemp=IncUnemp,
            n_approx=n_approx_Tran,
            wage=wage,
            labor=labor,
            tax_rate=tax_rate,
        )
        joint_dstn = combine_indep_dstns(perm_dstn, tran_dstn)

        super().__init__(
            name="HANK",
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
    r"""
    Generates a list of discrete approximations to the income process for each
    life period, from end of life to beginning of life.  Permanent shocks (:math:`\psi`) are mean
    one lognormally distributed with standard deviation PermShkStd[t] during the
    working life, and degenerate at 1 in the retirement period. Transitory shocks (:math:`\theta`)
    are mean one lognormally distributed with a point mass at IncUnemp with
    probability UnempPrb while working; they are mean one with a point mass at
    IncUnempRet with probability UnempPrbRet.  Retirement occurs
    after t=T_retire periods of working.

    .. math::
        \begin{align*}
        \psi_t &\sim \begin{cases}
        \exp(\mathcal{N}(-\textbf{PermShkStd}_{t}^{2}/2,\textbf{PermShkStd}_{t}^{2})) & \text{if } t \leq t_{\text{retire}}\\
        1 & \text{if } t > t_{\text{retire}}
        \end{cases}\\
        p_{\text{unemp}} & = \begin{cases}
        \textbf{UnempPrb} & \text{if } t \leq t_{\text{retire}} \\
        \textbf{UnempPrbRet} & \text{if } t > t_{\text{retire}} \\
        \end{cases}\\
        &\text{if } p > p_{\text{unemp}} \\
        \theta_t &\sim\begin{cases}
        \exp(\mathcal{N}(-\textbf{PermShkStd}_{t}^{2}/2-\ln(\frac{1-\textbf{IncUnemp }\textbf{UnempPrb}}{1-\textbf{UnempPrb}}),\textbf{PermShkStd}_{t}^{2})) & \text{if } t\leq t_{\text{retire}}\\
        \frac{1-\textbf{UnempPrbRet }\textbf{IncUnempRet}}{1-\textbf{UnempPrbRet}} & \text{if } t > t_{\text{retire}} \\
        \end{cases}\\
        &\text{otherwise}\\
        \theta_t &\sim\begin{cases}
        \textbf{IncUnemp} & \text{if } t\leq t_{\text{retire}}\\
        \textbf{IncUnempRet} & \text{if } t\leq t_{\text{retire}}\\
        \end{cases}\\
        \mathbb{E}[\psi]&=\mathbb{E}[\theta] = 1.\\
        \end{align*}

    All time in this function runs forward, from t=0 to t=T

    Parameters
    ----------
    PermShkStd : [float]
        List of standard deviations in log permanent income uncertainty during
        the agent's life.
    PermShkCount : int
        The number of approximation points to be used in the discrete approximation
        to the permanent income shock distribution.
    TranShkStd : [float]
        List of standard deviations in log transitory income uncertainty during
        the agent's life.
    TranShkCount : int
        The number of approximation points to be used in the discrete approximation
        to the permanent income shock distribution.
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


def construct_HANK_lognormal_income_process_unemployment(
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
    tax_rate,
    labor,
    wage,
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
    IncUnempRet with probability UnempPrbRet.  Retirement occurs after t=T_retire
    periods of working.

    This version of the function incorporates additional flexibility with respect
    to transitory income (continuous labor supply, wage rate, tax rate) and thus
    is useful in HANK models (hence the name!).

    Note 1: All time in this function runs forward, from t=0 to t=T

    Note 2: All parameters are passed as attributes of the input parameters.

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
    tax_rate : [float]
        List of flat tax rates on labor income, age-varying.
    labor : [float]
        List of intensive margin labor supply, age-varying.
    wage : [float]
        List of wage rate scaling factors, age-varying.
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
        engine=IncShkDstn_HANK,
        conditional={
            "sigma_Perm": PermShkStd,
            "sigma_Tran": TranShkStd,
            "n_approx_Perm": PermShkCount_list,
            "n_approx_Tran": TranShkCount_list,
            "neutral_measure": neutral_measure_list,
            "UnempPrb": UnempPrb_list,
            "IncUnemp": IncUnemp_list,
            "wage": wage,
            "tax_rate": tax_rate,
            "labor": labor,
        },
        RNG=RNG,
    )

    return IncShkDstn


###############################################################################


def get_PermShkDstn_from_IncShkDstn(IncShkDstn, RNG):
    PermShkDstn = [
        this.make_univariate(0, seed=RNG.integers(0, 2**31 - 1)) for this in IncShkDstn
    ]
    return TimeVaryingDiscreteDistribution(PermShkDstn, seed=RNG.integers(0, 2**31 - 1))


def get_TranShkDstn_from_IncShkDstn(IncShkDstn, RNG):
    TranShkDstn = [
        this.make_univariate(1, seed=RNG.integers(0, 2**31 - 1)) for this in IncShkDstn
    ]
    return TimeVaryingDiscreteDistribution(TranShkDstn, seed=RNG.integers(0, 2**31 - 1))


def get_TranShkGrid_from_TranShkDstn(T_cycle, TranShkDstn):
    TranShkGrid = [TranShkDstn[t].atoms.flatten() for t in range(T_cycle)]
    return TranShkGrid


def make_polynomial_PermGroFac(T_cycle, PermGroFac_coeffs, age_0=0.0, age_step=1.0):
    """
    Construct the profile of permanent growth factors by age using polynomial coefficients.

    Parameters
    ----------
    T_cycle : int
        Number of non-terminal period's in this agent's cycle.
    PermGroFac_coeffs : [float]
        Arbitrary length list or 1D vector of polynomial coefficients of age on
        permanent income growth factor.
    age_0 : float, optional
        Initial age of agents (when t_age=0), with respect to the polynomial coefficients.
        The default is 0.
    age_step : float, optional
        Age increment in the model, with respect to the polynomial coefficients.
        The default is 1.

    Returns
    -------
    PermGroFac : [float]
        List of permanent income growth factors, one per period.
    """
    PermGroFac = make_polynomial_params(
        PermGroFac_coeffs, T_cycle, offset=0.0, step=1.0
    )
    return PermGroFac.tolist()


def make_polynomial_PermShkStd(T_cycle, PermShkStd_coeffs, age_0=0.0, age_step=1.0):
    """
    Construct the profile of (log) permanent income shock standard deviations by
    age using polynomial coefficients.

    Parameters
    ----------
    T_cycle : int
        Number of non-terminal period's in this agent's cycle.
    PermGroFac_coeffs : [float]
        Arbitrary length list or 1D vector of polynomial coefficients of age on
        (log) permanent income shock standard deviation.
    age_0 : float, optional
        Initial age of agents (when t_age=0), with respect to the polynomial coefficients.
        The default is 0.
    age_step : float, optional
        Age increment in the model, with respect to the polynomial coefficients.
        The default is 1.

    Returns
    -------
    PermShkStd : [float]
        List of (log) permanent income shock standard deviations, one per period.
    """
    PermShkStd = make_polynomial_params(
        PermShkStd_coeffs, T_cycle, offset=0.0, step=1.0
    )
    return PermShkStd.tolist()


def make_polynomial_TranShkStd(T_cycle, TranShkStd_coeffs, age_0=0.0, age_step=1.0):
    """
    Construct the profile of (log) transitory income shock standard deviations by
    age using polynomial coefficients.

    Parameters
    ----------
    T_cycle : int
        Number of non-terminal period's in this agent's cycle.
    PermGroFac_coeffs : [float]
        Arbitrary length list or 1D vector of polynomial coefficients of age on
        (log) transitory income shock standard deviation.
    age_0 : float, optional
        Initial age of agents (when t_age=0), with respect to the polynomial coefficients.
        The default is 0.
    age_step : float, optional
        Age increment in the model, with respect to the polynomial coefficients.
        The default is 1.

    Returns
    -------
    TranShkStd : [float]
        List of (log) permanent income shock standard deviations, one per period.
    """
    TranShkStd = make_polynomial_params(
        TranShkStd_coeffs, T_cycle, offset=0.0, step=1.0
    )
    return TranShkStd.tolist()


class pLvlFuncAR1(MetricObject):
    """
    A class for representing AR1-style persistent income growth functions.

    Parameters
    ----------
    pLogMean : float
        Log persistent income level toward which we are drawn.
    PermGroFac : float
        Autonomous (e.g. life cycle) pLvl growth (does not AR1 decay).
    Corr : float
        Correlation coefficient on log income.
    """

    def __init__(self, pLogMean, PermGroFac, Corr):
        self.pLogMean = pLogMean
        self.LogGroFac = np.log(PermGroFac)
        self.Corr = Corr

    def __call__(self, pLvlNow):
        """
        Returns expected persistent income level next period as a function of
        this period's persistent income level.

        Parameters
        ----------
        pLvlNow : np.array
            Array of current persistent income levels.

        Returns
        -------
        pLvlNext : np.array
            Identically shaped array of next period persistent income levels.
        """
        pLvlNext = np.exp(
            self.Corr * np.log(pLvlNow)
            + (1.0 - self.Corr) * self.pLogMean
            + self.LogGroFac
        )
        return pLvlNext


###############################################################################

# Define income processes that can be used in the ConsGenIncProcess model


def make_trivial_pLvlNextFunc(T_cycle):
    """
    A dummy function that creates default trivial permanent income dynamics:
    none at all! Simply returns a list of IdentityFunctions, one for each period.

    Parameters
    ----------
    T_cycle : int
        Number of non-terminal periods in the agent's problem.

    Returns
    -------
    pLvlNextFunc : [IdentityFunction]
        List of trivial permanent income dynamic functions.
    """
    pLvlNextFunc_basic = IdentityFunction()
    pLvlNextFunc = T_cycle * [pLvlNextFunc_basic]
    return pLvlNextFunc


def make_explicit_perminc_pLvlNextFunc(T_cycle, PermGroFac):
    """
    A function that creates permanent income dynamics as a sequence of linear
    functions, indicating constant expected permanent income growth across
    permanent income levels.

    Parameters
    ----------
    T_cycle : int
        Number of non-terminal periods in the agent's problem.
    PermGroFac : [float]
        List of permanent income growth factors over the agent's problem.

    Returns
    -------
    pLvlNextFunc : [LinearInterp]
        List of linear functions representing constant permanent income growth
        rate, regardless of current permanent income level.
    """
    pLvlNextFunc = []
    for t in range(T_cycle):
        pLvlNextFunc.append(
            LinearInterp(np.array([0.0, 1.0]), np.array([0.0, PermGroFac[t]]))
        )
    return pLvlNextFunc


def make_AR1_style_pLvlNextFunc(T_cycle, pLvlInitMean, PermGroFac, PrstIncCorr):
    """
    A function that creates permanent income dynamics as a sequence of AR1-style
    functions. If cycles=0, the product of PermGroFac across all periods must be
    1.0, otherwise this method is invalid.

    Parameters
    ----------
    T_cycle : int
        Number of non-terminal periods in the agent's problem.
    pLvlInitMean : float
        Mean of log permanent income at initialization.
    PermGroFac : [float]
        List of permanent income growth factors over the agent's problem.
    PrstIncCorr : float
        Correlation coefficient on log permanent income today on log permanent
        income in the succeeding period.

    Returns
    -------
    pLvlNextFunc : [pLvlFuncAR1]
        List of AR1-style persistent income dynamics functions
    """
    pLvlNextFunc = []
    pLogMean = pLvlInitMean  # Initial mean (log) persistent income
    for t in range(T_cycle):
        pLvlNextFunc.append(pLvlFuncAR1(pLogMean, PermGroFac[t], PrstIncCorr))
        pLogMean += np.log(PermGroFac[t])
    return pLvlNextFunc


###############################################################################


def make_basic_pLvlPctiles(
    pLvlPctiles_count,
    pLvlPctiles_bound=[0.001, 0.999],
    pLvlPctiles_tail_count=0,
    pLvlPctiles_tail_order=np.e,
):
    """
    Make a relatively basic specification for pLvlPctiles by choosing the number
    of uniformly spaced nodes in the "body", the percentile boundaries for the
    body, the number of nodes in each tail, and the order/factor by which the
    tail percentiles approach 0 and 1 respectively.

    Parameters
    ----------
    pLvlPctile_count : int
        Number of nodes in the "body" of the percentile set.
    pLvlPctile_bound : [float,float], optional
        Percentile bounds for the "body" of the set. The default is [0.0, 1.0].
    pLvlPctile_tail_count : int, optional
       Number of nodes in each extant tail of the set. The default is 0.
    pLvlPctile_tail_order : float, optional
        Factor by which tail percentiles shrink toward 0 and 1. The default is np.e.

    Returns
    -------
    pLvlPctiles : np.array
        Array of percentiles of pLvl, usually used to construct pLvlGrid using
        the function below.
    """
    bound = pLvlPctiles_bound
    fac = 1.0 / pLvlPctiles_tail_order
    body = np.linspace(bound[0], bound[1], num=pLvlPctiles_count)

    if bound[0] > 0.0:
        lower = []
        val = bound[0]
        for i in range(pLvlPctiles_tail_count):
            val *= fac
            lower.append(val)
        lower.reverse()
        lower = np.array(lower)
    else:
        lower = np.array([])

    if bound[1] < 1.0:
        upper = []
        val = 1.0 - bound[1]
        for i in range(pLvlPctiles_tail_count):
            val *= fac
            upper.append(val)
        upper = 1.0 - np.array(upper)
    else:
        upper = np.array([])

    pLvlPctiles = np.concatenate((lower, body, upper))
    return pLvlPctiles


def make_pLvlGrid_by_simulation(
    cycles,
    T_cycle,
    PermShkDstn,
    pLvlNextFunc,
    LivPrb,
    pLvlInitMean,
    pLvlInitStd,
    pLvlPctiles,
    pLvlExtra=None,
):
    """
    Construct the permanent income grid for each period of the problem by simulation.
    If the model is infinite horizon (cycles=0), an approximation of the long run
    steady state distribution of permanent income is used (by simulating many periods).
    If the model is lifecycle (cycles=1), explicit simulation is used. In either
    case, the input pLvlPctiles is used to choose percentiles from the distribution.

    If the problem is neither infinite horizon nor lifecycle, this method will fail.
    If the problem is infinite horizon, cumprod(PermGroFac) must equal one.

    Parameters
    ----------
    cycles : int
        Number of times the sequence of periods happens for the agent; must be 0 or 1.
    T_cycle : int
        Number of non-terminal periods in the agent's problem.
    PermShkDstn : [distribution]
        List of permanent shock distributions in each period of the problem.
    pLvlNextFunc : [function]
        List of permanent income dynamic functions.
    LivPrb : [float]
        List of survival probabilities by period of the cycle. Only used in infinite
        horizon specifications.
    pLvlInitMean : float
        Mean of log permanent income at initialization.
    pLvlInitStd : float
        Standard deviaition of log permanent income at initialization.
    pLvlPctiles : [float]
        List or array of percentiles (between 0 and 1) of permanent income to
        use for the pLvlGrid.
    pLvlExtra : None or [float], optional
        Additional pLvl values to automatically include in pLvlGrid.

    Returns
    -------
    pLvlGrid : [np.array]
        List of permanent income grids for each period, constructed by simulating
        the permanent income process and extracting specified percentiles.
    """
    LivPrbAll = np.array(LivPrb)
    Agent_N = 100000

    # Simulate the distribution of persistent income levels by t_cycle in a lifecycle model
    if cycles == 1:
        pLvlNow = Lognormal(pLvlInitMean, sigma=pLvlInitStd, seed=31382).draw(Agent_N)
        pLvlGrid = []  # empty list of time-varying persistent income grids
        # Calculate distribution of persistent income in each period of lifecycle
        for t in range(T_cycle):
            if t > 0:
                PermShkNow = PermShkDstn[t - 1].draw(N=Agent_N)
                pLvlNow = pLvlNextFunc[t - 1](pLvlNow) * PermShkNow
            pLvlGrid.append(get_percentiles(pLvlNow, percentiles=pLvlPctiles))

    # Calculate "stationary" distribution in infinite horizon (might vary across periods of cycle)
    elif cycles == 0:
        T_long = (
            1000  # Number of periods to simulate to get to "stationary" distribution
        )
        pLvlNow = Lognormal(mu=pLvlInitMean, sigma=pLvlInitStd, seed=31382).draw(
            Agent_N
        )
        t_cycle = np.zeros(Agent_N, dtype=int)
        for t in range(T_long):
            # Determine who dies and replace them with newborns
            LivPrb = LivPrbAll[t_cycle]
            draws = Uniform(seed=t).draw(Agent_N)
            who_dies = draws > LivPrb
            pLvlNow[who_dies] = Lognormal(
                pLvlInitMean, pLvlInitStd, seed=t + 92615
            ).draw(np.sum(who_dies))
            t_cycle[who_dies] = 0

            for j in range(T_cycle):  # Update persistent income
                these = t_cycle == j
                PermShkTemp = PermShkDstn[j].draw(N=np.sum(these))
                pLvlNow[these] = pLvlNextFunc[j](pLvlNow[these]) * PermShkTemp
            t_cycle = t_cycle + 1
            t_cycle[t_cycle == T_cycle] = 0

        # We now have a "long run stationary distribution", extract percentiles
        pLvlGrid = []  # empty list of time-varying persistent income grids
        for t in range(T_cycle):
            these = t_cycle == t
            pLvlGrid.append(get_percentiles(pLvlNow[these], percentiles=pLvlPctiles))

    # Throw an error if cycles>1
    else:
        assert False, "Can only handle cycles=0 or cycles=1!"

    # Insert any additional requested points into the pLvlGrid
    if pLvlExtra is not None:
        pLvlExtra_alt = np.array(pLvlExtra)
        for t in range(T_cycle):
            pLvlGrid_t = pLvlGrid[t]
            pLvlGrid[t] = np.unique(np.concatenate((pLvlGrid_t, pLvlExtra_alt)))

    return pLvlGrid
