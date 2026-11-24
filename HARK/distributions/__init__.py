import warnings

__all__ = [
    "DiscreteDistribution",
    "DiscreteDistributionLabeled",
    "cdf_invert",
    "Distribution",
    "IndexDistribution",
    "LogNormal",
    "LogNormalMeanOne",
    "LogNormalTruncated",
    "LogNormalTruncatedMeanOne",
    "truncated_normal_variance",
    "Normal",
    "MultivariateNormal",
    "MultivariateLogNormal",
    "Weibull",
    "Bernoulli",
    "approx_beta",
    "make_markov_approx_to_normal",
    "make_markov_approx_to_normal_by_monte_carlo",
    "approx_lognormal_gauss_hermite",
    "calc_expectation",
    "expected_with_loop",
    "calc_lognormal_style_pars_from_normal_pars",
    "calc_normal_style_pars_from_lognormal_pars",
    "combine_indep_dstns",
    "distr_of_function",
    "expected",
    "Uniform",
    "MarkovProcess",
    "add_discrete_outcome",
    "add_discrete_outcome_constant_mean",
    "employed_atoms",
    "restrict_to_atoms",
    "make_tauchen_ar1",
]

from HARK.distributions.base import (
    Distribution,
    IndexDistribution,
    MarkovProcess,
)
from HARK.distributions.continuous import (
    _RENAMED,
    _renamed_message,
    LogNormal,
    LogNormalMeanOne,
    LogNormalTruncated,
    LogNormalTruncatedMeanOne,
    Normal,
    truncated_normal_variance,
    Uniform,
    Weibull,
)
from HARK.distributions.discrete import (
    Bernoulli,
    DiscreteDistribution,
    DiscreteDistributionLabeled,
    cdf_invert,
)
from HARK.distributions.multivariate import MultivariateLogNormal, MultivariateNormal
from HARK.distributions.utils import (
    add_discrete_outcome,
    add_discrete_outcome_constant_mean,
    approx_beta,
    make_markov_approx_to_normal,
    make_markov_approx_to_normal_by_monte_carlo,
    approx_lognormal_gauss_hermite,
    calc_expectation,
    expected_with_loop,
    calc_lognormal_style_pars_from_normal_pars,
    calc_normal_style_pars_from_lognormal_pars,
    combine_indep_dstns,
    distr_of_function,
    employed_atoms,
    expected,
    make_tauchen_ar1,
    restrict_to_atoms,
)


def __getattr__(name):
    # PEP 562: a lognormal class's name before it was renamed is the class itself,
    # with a DeprecationWarning where it is looked up (the table is in
    # HARK.distributions.continuous)
    if name in _RENAMED:
        warnings.warn(
            _renamed_message(name),
            DeprecationWarning,
            stacklevel=2,
            # an import from a package looks the name up twice, once inside the
            # import machinery: both warnings name the line of the import
            skip_file_prefixes=("<frozen importlib",),
        )
        return globals()[_RENAMED[name]]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
