import math
from warnings import warn

import numpy as np
from scipy import stats

from HARK.distributions.base import IndexDistribution
from HARK.distributions.discrete import (
    DiscreteDistribution,
    DiscreteDistributionLabeled,
)
from HARK.distributions.continuous import Normal


# TODO: This function does not generate the limit attribute
def approx_lognormal_gauss_hermite(N, mu=0.0, sigma=1.0, seed=None):
    d = Normal(mu, sigma).discretize(N, method="hermite")
    return DiscreteDistribution(d.pmv, np.exp(d.atoms), seed=seed)


def calc_normal_style_pars_from_lognormal_pars(avg_lognormal, std_lognormal):
    varLognormal = std_lognormal**2
    varNormal = math.log(1 + varLognormal / avg_lognormal**2)
    avgNormal = math.log(avg_lognormal) - varNormal * 0.5
    std_normal = math.sqrt(varNormal)
    return avgNormal, std_normal


def calc_lognormal_style_pars_from_normal_pars(mu_normal, std_normal):
    varNormal = std_normal**2
    avg_lognormal = math.exp(mu_normal + varNormal * 0.5)
    varLognormal = (math.exp(varNormal) - 1) * avg_lognormal**2
    std_lognormal = math.sqrt(varLognormal)
    return avg_lognormal, std_lognormal


def approx_beta(N, a=1.0, b=1.0):
    """
    Calculate a discrete approximation to the beta distribution.  May be quite
    slow, as it uses a rudimentary numeric integration method to generate the
    discrete approximation.

    Parameters
    ----------
    N : int
        Size of discrete space vector to be returned.
    a : float
        First shape parameter (sometimes called alpha).
    b : float
        Second shape parameter (sometimes called beta).

    Returns
    -------
    d : DiscreteDistribution
        Probability associated with each point in array of discrete
        points for discrete probability mass function.
    """
    P = 1000
    vals = np.reshape(stats.beta.ppf(np.linspace(0.0, 1.0, N * P), a, b), (N, P))
    atoms = np.mean(vals, axis=1)
    pmv = np.ones(N) / float(N)
    return DiscreteDistribution(pmv, atoms)


def make_markov_approx_to_normal(x_grid, mu, sigma, K=351, bound=3.5):
    """
    Creates an approximation to a normal distribution with mean mu and standard
    deviation sigma, returning a stochastic vector called p_vec, corresponding
    to values in x_grid.  If a RV is distributed x~N(mu,sigma), then the expectation
    of a continuous function f() is E[f(x)] = numpy.dot(p_vec,f(x_grid)).

    Parameters
    ----------
    x_grid: numpy.array
        A sorted 1D array of floats representing discrete values that a normally
        distributed RV could take on.
    mu: float
        Mean of the normal distribution to be approximated.
    sigma: float
        Standard deviation of the normal distribution to be approximated.
    K: int
        Number of points in the normal distribution to sample.
    bound: float
        Truncation bound of the normal distribution, as +/- bound*sigma.

    Returns
    -------
    p_vec: numpy.array
        A stochastic vector with probability weights for each x in x_grid.
    """
    x_n = x_grid.size  # Number of points in the outcome grid
    lower_bound = -bound  # Lower bound of normal draws to consider, in SD
    upper_bound = bound  # Upper bound of normal draws to consider, in SD
    raw_sample = np.linspace(
        lower_bound, upper_bound, K
    )  # Evenly spaced draws between bounds
    f_weights = stats.norm.pdf(raw_sample)  # Relative probability of each draw
    sample = mu + sigma * raw_sample  # Adjusted bounds, given mean and stdev
    w_vec = np.zeros(x_n)  # A vector of outcome weights

    # Find the relative position of each of the draws
    sample_pos = np.searchsorted(x_grid, sample)
    sample_pos[sample_pos < 1] = 1
    sample_pos[sample_pos > x_n - 1] = x_n - 1

    # Make arrays of the x_grid point directly above and below each draw
    bot = x_grid[sample_pos - 1]
    top = x_grid[sample_pos]
    alpha = (sample - bot) / (top - bot)

    # Keep the weights (alpha) in bounds
    alpha_clipped = np.clip(alpha, 0.0, 1.0)

    # Loop through each x_grid point and add up the probability that each nearby
    # draw contributes to it (accounting for distance)
    for j in range(1, x_n):
        c = sample_pos == j
        w_vec[j - 1] = w_vec[j - 1] + np.dot(f_weights[c], 1.0 - alpha_clipped[c])
        w_vec[j] = w_vec[j] + np.dot(f_weights[c], alpha_clipped[c])

    # Reweight the probabilities so they sum to 1
    W = np.sum(w_vec)
    p_vec = w_vec / W

    # Check for obvious errors, and return p_vec
    assert (
        (np.all(p_vec >= 0.0))
        and (np.all(p_vec <= 1.0))
        and (np.isclose(np.sum(p_vec), 1.0))
    )
    return p_vec


def make_markov_approx_to_normal_by_monte_carlo(x_grid, mu, sigma, N_draws=10000):
    """
    Creates an approximation to a normal distribution with mean mu and standard
    deviation sigma, by Monte Carlo.
    Returns a stochastic vector called p_vec, corresponding
    to values in x_grid.  If a RV is distributed x~N(mu,sigma), then the expectation
    of a continuous function f() is E[f(x)] = numpy.dot(p_vec,f(x_grid)).

    Parameters
    ----------
    x_grid: numpy.array
        A sorted 1D array of floats representing discrete values that a normally
        distributed RV could take on.
    mu: float
        Mean of the normal distribution to be approximated.
    sigma: float
        Standard deviation of the normal distribution to be approximated.
    N_draws: int
        Number of draws to use in Monte Carlo.

    Returns
    -------
    p_vec: numpy.array
        A stochastic vector with probability weights for each x in x_grid.
    """

    # Take random draws from the desired normal distribution
    random_draws = np.random.normal(loc=mu, scale=sigma, size=N_draws)

    # Compute the distance between the draws and points in x_grid
    distance = np.abs(x_grid[:, np.newaxis] - random_draws[np.newaxis, :])

    # Find the indices of the points in x_grid that are closest to the draws
    distance_minimizing_index = np.argmin(distance, axis=0)

    # For each point in x_grid, the approximate probability of that point is the number
    # of Monte Carlo draws that are closest to that point
    p_vec = np.zeros_like(x_grid)
    for p_index, p in enumerate(p_vec):
        p_vec[p_index] = np.sum(distance_minimizing_index == p_index) / N_draws

    # Check for obvious errors, and return p_vec
    assert (
        (np.all(p_vec >= 0.0))
        and (np.all(p_vec <= 1.0))
        and (np.isclose(np.sum(p_vec), 1.0))
    )
    return p_vec


def make_tauchen_ar1(N, sigma=1.0, ar_1=0.9, bound=3.0, inflendpoint=True):
    """
    Function to return a discretized version of an AR1 process.
    See http://www.fperri.net/TEACHING/macrotheory08/numerical.pdf for details

    Parameters
    ----------
    N: int
        Size of discretized grid
    sigma: float
        Standard deviation of the error term
    ar_1: float
        AR1 coefficient
    bound: float
        The highest (lowest) grid point will be bound (-bound) multiplied by the unconditional
        standard deviation of the process
    inflendpoint: Bool
        If True: implement the standard method as in Tauchen (1986):
            assign the probability of jumping to a point outside the grid to the closest endpoint
        If False: implement an alternative method:
            discard the probability of jumping to a point outside the grid, effectively
            reassigning it to the remaining points in proportion to their probability of being reached

    Returns
    -------
    y: np.array
        Grid points on which the discretized process takes values
    trans_matrix: np.array
        Markov transition array for the discretized process
    """
    yN = bound * sigma / ((1 - ar_1**2) ** 0.5)
    y = np.linspace(-yN, yN, N)
    d = y[1] - y[0]
    cuts = (y[1:] + y[:-1]) / 2.0
    if inflendpoint:
        cuts = np.concatenate(([-np.inf], cuts, [np.inf]))
    else:
        cuts = np.concatenate(([y[0] - d / 2], cuts, [y[-1] + d / 2]))
    dist = np.reshape(cuts, (1, N + 1)) - np.reshape(ar_1 * y, (N, 1))
    dist /= sigma
    cdf_array = stats.norm.cdf(dist)
    sf_array = stats.norm.sf(dist)
    trans = cdf_array[:, 1:] - cdf_array[:, :-1]
    trans_alt = sf_array[:, :-1] - sf_array[:, 1:]
    trans_matrix = np.maximum(trans, trans_alt)
    trans_matrix /= np.sum(trans_matrix, axis=1, keepdims=True)
    return y, trans_matrix


# ================================================================================
# ==================== Functions for manipulating discrete distributions =========
# ================================================================================


def add_discrete_outcome_constant_mean(distribution, x, p, sort=False):
    """
    Adds a discrete outcome of x with probability p to an existing distribution,
    holding constant the relative probabilities of other outcomes and overall mean.

    Parameters
    ----------
    distribution : DiscreteDistribution
        A one-dimensional DiscreteDistribution.
    x : float
        The new value to be added to the distribution.
    p : float
        The probability of the discrete outcome x occuring.
    sort: bool
        Whether or not to sort atoms before returning it

    Returns
    -------
    d : DiscreteDistribution
        Probability associated with each point in array of discrete
        points for discrete probability mass function.
    """
    if (
        isinstance(distribution, IndexDistribution)
        and hasattr(distribution, "distributions")
        and distribution.distributions
    ):
        # apply recursively on all the internal distributions
        return IndexDistribution(
            distributions=[
                add_discrete_outcome_constant_mean(d, x, p)
                for d in distribution.distributions
            ],
            seed=distribution.seed,
        )

    else:
        atoms = np.append(x, distribution.atoms * (1 - p * x) / (1 - p))
        pmv = np.append(p, distribution.pmv * (1 - p))

        if sort:
            indices = np.argsort(atoms)
            atoms = atoms[indices]
            pmv = pmv[indices]

        # Update infimum and supremum
        temp_x = np.array(x, ndmin=1)
        try:
            infimum = np.array(
                [
                    np.minimum(temp_x[i], distribution.limit["infimum"][i])
                    for i in range(temp_x.size)
                ]
            )
        except KeyError:
            infimum = np.min(atoms, axis=-1, keepdims=True)
        try:
            supremum = np.array(
                [
                    np.maximum(temp_x[i], distribution.limit["supremum"][i])
                    for i in range(temp_x.size)
                ]
            )
        except KeyError:
            supremum = np.max(atoms, axis=-1, keepdims=True)

        limit = {
            "dist": distribution,
            "method": "add_discrete_outcome_constant_mean",
            "x": x,
            "p": p,
            "infimum": infimum,
            "supremum": supremum,
        }

        return DiscreteDistribution(pmv, atoms, seed=distribution.seed, limit=limit)


def add_discrete_outcome(distribution, x, p, sort=False):
    """
    Adds a discrete outcome of x with probability p to an existing distribution,
    holding constant the relative probabilities of other outcomes.

    Parameters
    ----------
    distribution : DiscreteDistribution
        One-dimensional distribution to which the outcome is to be added.
    x : float
        The new value to be added to the distribution.
    p : float
        The probability of the discrete outcome x occuring.

    Returns
    -------
    d : DiscreteDistribution
        Probability associated with each point in array of discrete
        points for discrete probability mass function.
    """

    atoms = np.append(x, distribution.atoms)
    pmv = np.append(p, distribution.pmv * (1 - p))

    if sort:
        indices = np.argsort(atoms)
        atoms = atoms[indices]
        pmv = pmv[indices]

    # Update infimum and supremum
    temp_x = np.array(x, ndmin=1)
    try:
        infimum = np.array(
            [
                np.minimum(temp_x[i], distribution.limit["infimum"][i])
                for i in range(temp_x.size)
            ]
        )
    except KeyError:
        infimum = np.min(atoms, axis=-1, keepdims=True)
    try:
        supremum = np.array(
            [
                np.maximum(temp_x[i], distribution.limit["supremum"][i])
                for i in range(temp_x.size)
            ]
        )
    except KeyError:
        supremum = np.max(atoms, axis=-1, keepdims=True)

    limit = {
        "dist": distribution,
        "method": "add_discrete_outcome",
        "x": x,
        "p": p,
        "infimum": infimum,
        "supremum": supremum,
    }

    return DiscreteDistribution(pmv, atoms, seed=distribution.seed, limit=limit)


def combine_indep_dstns(*distributions, seed=None):
    """
    Given n independent vector-valued discrete distributions, construct their joint discrete distribution.
    Can take multivariate discrete distributions as inputs.

    Parameters
    ----------
    distributions : DiscreteDistribution
        Arbitrary number of discrete distributions to combine. Their realizations must be
        vector-valued (for each D in distributions, it must be the case that len(D.dim())==1).
    seed : int, optional
        Value to use as the RNG seed for the combined distribution, default is 0.

    Returns
    -------
    A DiscreteDistribution representing the joint distribution of the given
    random variables.
    """
    # Get information on the distributions
    dist_lengths = ()
    dist_dims = ()
    dist_is_labeled = ()
    var_labels = ()
    for dist in distributions:
        if len(dist.dim()) > 1:
            raise NotImplementedError(
                "We currently only support combining vector-valued distributions."
            )

        dist_dims += (dist.dim(),)
        dist_lengths += (len(dist.pmv),)

        labeled = isinstance(dist, DiscreteDistributionLabeled)
        dist_is_labeled += (labeled,)
        if labeled:
            var_labels += tuple(dist.dataset.data_vars.keys())
        else:
            var_labels += tuple([""] * dist.dim()[0])

    dstn_count = len(distributions)

    all_labeled = all(dist_is_labeled)
    labels_are_unique = len(var_labels) == len(set(var_labels))

    # We need the combinations of indices of realizations in all
    # distributions
    inds = np.meshgrid(
        *[np.array(range(l), dtype=int) for l in dist_lengths], indexing="ij"
    )
    inds = [x.flatten() for x in inds]

    atoms_out = []
    P_temp = []
    for i, ind_vec in enumerate(inds):
        atoms_out += [distributions[i].atoms[..., ind_vec]]
        P_temp += [distributions[i].pmv[ind_vec]]

    atoms_out = np.concatenate(atoms_out, axis=0)
    P_temp = np.stack(P_temp, axis=0)
    P_out = np.prod(P_temp, axis=0)

    assert np.isclose(np.sum(P_out), 1), "Probabilities do not sum to 1!"

    # Make the limit dictionary
    infimum = np.concatenate(
        [distributions[i].limit["infimum"] for i in range(dstn_count)]
    )
    supremum = np.concatenate(
        [distributions[i].limit["supremum"] for i in range(dstn_count)]
    )
    limit = {
        "dist": distributions,
        "method": "combine_indep_dstns",
        "infimum": infimum,
        "supremum": supremum,
    }

    if all_labeled and labels_are_unique:
        combined_dstn = DiscreteDistributionLabeled(
            pmv=P_out,
            atoms=atoms_out,
            var_names=var_labels,
            limit=limit,
            seed=seed,
        )
    else:
        if all_labeled and not labels_are_unique:
            warn(
                "There are duplicated labels in the provided distributions. Returning a non-labeled combination"
            )
        combined_dstn = DiscreteDistribution(P_out, atoms_out, limit=limit, seed=seed)

    return combined_dstn


def calc_expectation(dstn, func=lambda x: x, *args):
    """
    Expectation of a function, given an array of configurations of its inputs
    along with a DiscreteDistribution object that specifies the probability
    of each configuration.

    Parameters
    ----------
    dstn : DiscreteDistribution
        The distribution over which the function is to be evaluated.
    func : function
        The function to be evaluated.
        This function should take an array of shape dstn.dim() and return
        either arrays of arbitrary shape or scalars.
        It may also take other arguments \\*args.
    \\*args :
        Other inputs for func, representing the non-stochastic arguments.
        The the expectation is computed at ``f(dstn, *args)``.

    Returns
    -------
    f_exp : np.array or scalar
        The expectation of the function at the queried values.
        Scalar if only one value.
    """

    f_query = [func(dstn.atoms[..., i], *args) for i in range(len(dstn.pmv))]

    f_query = np.stack(f_query, axis=-1)

    # From the numpy.dot documentation:
    # If both a and b are 1-D arrays, it is inner product of vectors (without complex conjugation).
    # If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
    # Thus, if func returns scalars, f_exp will be a scalar and if it returns arrays f_exp
    # will be an array of the same shape.
    f_exp = np.dot(f_query, dstn.pmv)

    return f_exp


def distr_of_function(dstn, func=lambda x: x, *args):
    """
    Finds the distribution of a random variable Y that is a function
    of discrete random variable atoms, Y=f(atoms).

    Parameters
    ----------
    dstn : DiscreteDistribution
        The distribution over which the function is to be evaluated.
    func : function
        The function to be evaluated.
        This function should take an array of shape dstn.dim().
        It may also take other arguments \\*args.
    \\*args :
        Additional non-stochastic arguments for func,
        The function is computed at ``f(dstn, *args)``.

    Returns
    -------
    f_dstn : DiscreteDistribution
        The distribution of func(dstn).
    """
    # Apply function to every event realization
    f_query = [func(dstn.atoms[..., i], *args) for i in range(len(dstn.pmv))]

    # Stack results along their last (new) axis
    f_query = np.stack(f_query, axis=-1)

    f_dstn = DiscreteDistribution(dstn.pmv, f_query)

    return f_dstn


def expected(func=None, dist=None, args=(), **kwargs):
    """
    Expectation of a function, given an array of configurations of its inputs
    along with a DiscreteDistribution(atomsRA) object that specifies the probability
    of each configuration.

    Parameters
    ----------
    func : function
        The function to be evaluated.
        This function should take the full array of distribution values
        and return either arrays of arbitrary shape or scalars.
        It may also take other arguments ``*args``.
        This function differs from the standalone `calc_expectation`
        method in that it uses numpy's vectorization and broadcasting
        rules to avoid costly iteration.
        Note: If you need to use a function that acts on single outcomes
        of the distribution, consier `distribution.calc_expectation`.
    dist : DiscreteDistribution or DiscreteDistributionLabeled
        The distribution over which the function is to be evaluated.
    args : tuple
        Other inputs for func, representing the non-stochastic arguments.
        The expectation is computed at ``f(dstn, *args)``.
    labels : bool
        If True, the function should use labeled indexing instead of integer
        indexing using the distribution's underlying rv coordinates. For example,
        if `dims = ('rv', 'x')` and `coords = {'rv': ['a', 'b'], }`, then
        the function can be `lambda x: x["a"] + x["b"]`.

    Returns
    -------
    f_exp : np.array or scalar
        The expectation of the function at the queried values.
        Scalar if only one value.
    """

    if not isinstance(args, tuple):
        args = (args,)

    if isinstance(dist, DiscreteDistributionLabeled):
        return dist.expected(func, *args, **kwargs)
    elif isinstance(dist, DiscreteDistribution):
        return dist.expected(func, *args)
