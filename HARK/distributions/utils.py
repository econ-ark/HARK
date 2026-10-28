import math
from warnings import warn

import numpy as np
from scipy import stats
from scipy.special import ndtr, ndtri

from HARK.distributions.base import IndexDistribution
from HARK.distributions.discrete import (
    DiscreteDistribution,
    DiscreteDistributionLabeled,
)
from HARK.distributions.continuous import Normal, _normal_mass


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


def _bounds_with_added_atom(distribution, x, atoms, scale=1.0):
    """
    Compute (infimum, supremum) for a distribution after a new atom ``x`` is added.

    The source distribution's own bounds are multiplied by ``scale``, the factor
    its atoms were multiplied by. Falls back to taking min/max over the augmented
    atoms array when the source distribution does not expose
    ``infimum``/``supremum`` in its ``limit`` dict.
    """
    temp_x = np.array(x, ndmin=1)
    try:
        infimum = np.array(
            [
                np.minimum(temp_x[i], scale * distribution.limit["infimum"][i])
                for i in range(temp_x.size)
            ]
        )
    except KeyError:
        infimum = np.min(atoms, axis=-1, keepdims=True)
    try:
        supremum = np.array(
            [
                np.maximum(temp_x[i], scale * distribution.limit["supremum"][i])
                for i in range(temp_x.size)
            ]
        )
    except KeyError:
        supremum = np.max(atoms, axis=-1, keepdims=True)
    return infimum, supremum


def _finalize_with_added_atom(distribution, atoms, x, p, method, sort, scale=1.0):
    """Build the new ``DiscreteDistribution`` after a value ``x`` (probability ``p``)
    has been spliced onto ``distribution``. Shared by ``add_discrete_outcome`` and
    ``add_discrete_outcome_constant_mean``; ``scale`` is the factor applied to the
    original atoms.
    """
    pmv = np.append(p, distribution.pmv * (1 - p))
    if sort:
        indices = np.argsort(atoms)
        atoms = atoms[indices]
        pmv = pmv[indices]
    infimum, supremum = _bounds_with_added_atom(distribution, x, atoms, scale)
    limit = {
        "dist": distribution,
        "method": method,
        "x": x,
        "p": p,
        "infimum": infimum,
        "supremum": supremum,
    }
    return DiscreteDistribution(pmv, atoms, seed=distribution.seed, limit=limit)


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
                add_discrete_outcome_constant_mean(d, x, p, sort=sort)
                for d in distribution.distributions
            ],
            seed=distribution.seed,
        )

    atoms = np.append(x, distribution.atoms * (1 - p * x) / (1 - p))
    return _finalize_with_added_atom(
        distribution,
        atoms,
        x,
        p,
        "add_discrete_outcome_constant_mean",
        sort,
        # In numpy's arithmetic, as the atoms' scale above is, so that np.errstate
        # governs a probability of one here as it does there
        scale=np.float64(1 - p * x) / (1 - p),
    )


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
    return _finalize_with_added_atom(
        distribution, atoms, x, p, "add_discrete_outcome", sort
    )


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


def calc_expectation(dstn, func=None, *args, **kwargs):
    """
    Deprecated alias for :func:`expected_with_loop`.

    ``calc_expectation`` was renamed to ``expected_with_loop`` in HARK
    0.17.2.  This shim preserves import compatibility for downstream code
    written against earlier versions (including frozen reproduction
    archives that cannot be edited), emitting a ``DeprecationWarning``.
    It will be removed in a future release; new code should call
    :func:`expected_with_loop` directly.
    """

    warn(
        "calc_expectation was renamed to expected_with_loop in HARK 0.17.2 "
        "and this alias will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return expected_with_loop(dstn, func, *args, **kwargs)


def expected_with_loop(dstn, func=None, *args, **kwargs):
    """
    Expectation of a function, given an array of configurations of its inputs
    along with a DiscreteDistribution object that specifies the probability
    of each configuration. Computation is performed by looping over each atom
    of the distribution and evaluating function one at a time. This approach is
    broadly compatible with any function, but is slow because of the loop.

    If func is relatively simple, and particularly if it does not involve array
    operations like tiling, reshaping, or logical indexing, consider using expected
    instead of expected_with_loop. That function evaluates all atoms simultaneously,
    avoiding the costly loop, but with reduced compatibility with "complex" operations.

    Parameters
    ----------
    dstn : DiscreteDistribution or DiscreteDistributionLabeled
        The distribution over which the function is to be evaluated.
    func : function or None
        The function to be evaluated. If dstn is a DiscreteDistribution, then this
        function should take an array of shape dstn.dim() and return either arrays
        of arbitrary shape or scalars. If dstn is a DiscreteDistributionLabeled, then
        the function should take an dictionary-like object (like an xr.dataset) and
        index into it with variable names. In either case, the function may also
        take other arguments \\*args. Defaults to identity function.
    \\*args :
        Other inputs for func, representing the non-stochastic arguments.
        The expectation is computed at ``f(dstn, *args, **kwargs)``.
    \\*kwargs :
        Other keyword inputs for func, representing the non-stochastic arguments.
        The expectation is computed at ``f(dstn, *args, **kwargs)``.

    Returns
    -------
    f_exp : np.array or scalar
        The expectation of the function at the queried values.
        Scalar if only one value.
    """
    func = func or (lambda x: x)

    if hasattr(dstn, "dataset"):  # cheap test for DiscreteDistributionLabeled
        f_query = []
        for i in range(len(dstn.pmv)):
            temp_dict = {
                key: float(dstn.variables[key][i]) for key in dstn.variables.keys()
            }
            f_query.append(func(temp_dict, *args, **kwargs))
    else:
        f_query = [
            func(dstn.atoms[..., i], *args, **kwargs) for i in range(len(dstn.pmv))
        ]

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


def expected(func=None, dstn=None, args=(), vectorized=True, **kwargs):
    """
    Compute the expectation of a function, given an array of configurations of its
    inputs along with a DiscreteDistribution object that specifies the probability
    of each configuration.

    If the function you want to evaluate uses complex array operations, such as
    tiling or logical indexing, pass `vectorized=False`. In that case, expectations
    are calculated by looping over each atom in the distribution. Otherwise, this
    function uses array operations and tries to compute all atoms simultaneously.

    Parameters
    ----------
    func : function
        The function to be evaluated. This function should take the full array of
        distribution values and return either arrays of arbitrary shape or scalars.
        It may also take other arguments ``*args``.
    dstn : DiscreteDistribution or DiscreteDistributionLabeled
        The distribution over which the function is to be evaluated.
    args : tuple
        Other inputs for func, representing the non-stochastic arguments.
        The expectation is computed at ``f(dstn, *args)``.
    vectorized : bool, optional
        Whether func is vectorizable (True, default) or requires looped evaluation.
    labels : bool, optional
        For ``DiscreteDistributionLabeled`` only.  If True (default), func
        receives a dict of labeled arrays.  If False, func receives the raw
        numpy atoms array, making DDL compatible with functions written for
        ``DiscreteDistribution``.
    **kwargs :
        Additional keyword arguments forwarded to ``dist.expected()``.
        For ``DiscreteDistributionLabeled``, these are passed through to
        the user-supplied *func* along with the xarray Dataset.

    Returns
    -------
    f_exp : np.array or scalar
        The expectation of the function at the queried values.
        Scalar if only one value.
    """
    if not isinstance(dstn, DiscreteDistribution):
        raise TypeError(
            f"expected(): 'dstn' must be a DiscreteDistribution or "
            f"DiscreteDistributionLabeled, got {type(dstn).__name__}."
        )

    if not isinstance(args, tuple):
        args = (args,)

    if not vectorized:
        loop_kwargs = kwargs.copy()
        loop_kwargs.pop("labels", None)
        return expected_with_loop(dstn, func, *args, **loop_kwargs)

    if args:
        if kwargs:
            return dstn.expected(func, *args, **kwargs)
        return dstn.expected(func, *args)

    if kwargs:
        return dstn.expected(func, **kwargs)
    return dstn.expected(func)


# Standard deviations used for the standard-normal coordinate of a law whose
# support is unbounded on that side: the normal mass beyond eight is 1.2e-15.
Z_UNBOUNDED = 8.0
# The standard-normal coordinate is cut into pieces this wide, so that a few
# Gauss-Legendre nodes a piece resolve the normal density: over [-8, 8], four
# nodes a piece miss the normal mass by 6e-12 and five by 2e-13, where with one
# piece beyond three standard deviations on either side four miss it by 1e-4.
Z_STEP = 1.0
_GAUSS_LEGENDRE = {}


def _gauss_legendre(n):
    """Gauss-Legendre nodes and weights on [-1, 1], cached by their number."""
    if n not in _GAUSS_LEGENDRE:
        _GAUSS_LEGENDRE[n] = np.polynomial.legendre.leggauss(n)
    return _GAUSS_LEGENDRE[n]


def _coordinate(dstn):
    """
    The standard-normal coordinate z in which expected_continuous integrates a
    continuous distribution, as (edges, value, coord, density): E[f(X)] is the
    integral over [edges[0], edges[-1]] of f(value(z)) density(z) dz, taken in
    pieces between the edges, and coord is the inverse of value, which places
    the split points. A law supplies its own z, in which it is the standard
    normal restricted to an interval, through its method normal_coordinate(),
    as the normal and lognormal laws, truncated or not, do; for any other law z
    is its normal score, reached through its quantile function,
    X = F^{-1}(Phi(z)), so the integrand is smooth in z wherever f is. A side
    without a bound is cut at Z_UNBOUNDED, and the edges are at every multiple
    of Z_STEP between the ends.
    """
    if hasattr(dstn, "normal_coordinate"):
        bot, top, value, coord = dstn.normal_coordinate()
    elif hasattr(dstn, "ppf") and hasattr(dstn, "cdf"):
        bot, top = -np.inf, np.inf

        def value(z):
            return dstn.ppf(ndtr(z))

        def coord(x):
            return ndtri(dstn.cdf(x))

    else:
        raise TypeError(
            "expected_continuous() cannot integrate over a "
            f"{type(dstn).__name__}: it takes laws with a quantile function and "
            "CDF, or a normal_coordinate() method, mixtures and discrete "
            "distributions."
        )
    lo, hi = max(float(bot), -Z_UNBOUNDED), min(float(top), Z_UNBOUNDED)
    steps = np.arange(np.floor(lo / Z_STEP) + 1.0, np.ceil(hi / Z_STEP)) * Z_STEP
    edges = [lo] + [z for z in steps if lo < z < hi] + [hi]
    mass = float(_normal_mass(lo, hi))

    def density(z):
        return np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * mass)

    return edges, value, coord, density


def _split_gauss_legendre(edges, n):
    """
    Gauss-Legendre nodes and weights, n a piece, on the pieces between sorted
    edges, with rows in the leading axes and the edges in the last. Pieces of
    zero length are dropped: each row keeps its pieces in order, padded with
    empty ones to the number the row with the most has. Returns the nodes and
    the weights, each of shape rows + (m,).
    """
    edges = np.asarray(edges, dtype=float)
    empty = np.diff(edges, axis=-1) <= 0.0
    order = np.argsort(empty, axis=-1, kind="stable")
    count = max(int(np.max(np.sum(~empty, axis=-1), initial=0)), 1)
    a = np.take_along_axis(edges[..., :-1], order[..., :count], axis=-1)
    b = np.take_along_axis(edges[..., 1:], order[..., :count], axis=-1)
    x, w = _gauss_legendre(n)
    half = 0.5 * (b - a)[..., None]
    nodes = half * x + 0.5 * (b + a)[..., None]
    shape = nodes.shape[:-2] + (-1,)
    return nodes.reshape(shape), (half * w).reshape(shape)


def _quadrature(dstn, cuts, n):
    """
    Nodes and weights for one distribution, with n Gauss-Legendre nodes on each
    piece between its cut points. With cuts of shape B + (k,), the returned
    values and weights have shape B + (m,), with the nodes in the last axis;
    without cuts they are 1D. A discrete distribution gives its atoms and
    probabilities whatever the cuts, and a mixture its components' nodes, with
    their weights multiplied by the components' weights.
    """
    if isinstance(dstn, DiscreteDistribution):
        atoms = np.asarray(dstn.atoms, dtype=float)
        if atoms.shape[0] != 1:
            raise ValueError(
                "expected_continuous() takes univariate distributions: pass the "
                "variables of a joint discrete distribution separately."
            )
        return atoms[0], np.asarray(dstn.pmv, dtype=float)

    if hasattr(dstn, "components") and hasattr(dstn, "weights"):
        parts = [_quadrature(part, cuts, n) for part in dstn.components]
        lead = () if cuts is None else cuts.shape[:-1]
        values = [np.broadcast_to(v, lead + v.shape[-1:]) for v, _ in parts]
        weights = [
            p * np.broadcast_to(w, lead + w.shape[-1:])
            for p, (_, w) in zip(dstn.weights, parts)
        ]
        return np.concatenate(values, axis=-1), np.concatenate(weights, axis=-1)

    fixed, value, coord, density = _coordinate(dstn)
    lo, hi = fixed[0], fixed[-1]
    if cuts is None:
        edges = np.array(fixed)
    else:
        # Cut points outside the support give pieces of zero length, dropped
        inner = np.clip(
            np.nan_to_num(coord(cuts), nan=lo, posinf=hi, neginf=lo), lo, hi
        )
        ends = np.ones(cuts.shape[:-1] + (1,)) * np.array(fixed)
        edges = np.sort(np.concatenate([ends, inner], axis=-1), axis=-1)
    nodes, weights = _split_gauss_legendre(edges, n)
    return value(nodes), weights * density(nodes)


def expected_continuous(
    func=None, dstns=None, args=(), labels=None, splits=None, nodes=4
):
    """
    The expectation of a function over independent distributions, taken against
    continuous laws by Gauss-Legendre quadrature instead of summed over a
    discretization's atoms. The integral over each law can be split at points
    where the function is not smooth, such as a kink, so that no quadrature
    node straddles one.

    The calling convention is expected()'s: func receives the values of the
    distributions, one per variable, and the other arguments. Each ndarray in
    args gets one more trailing axis per distribution, as expected() gives it
    one for the atoms, and the result has the shape that args broadcast to.

    The distributions can be:

    - continuous laws, integrated in a standard-normal coordinate z: the one a
      law supplies through its method normal_coordinate(), as the normal and
      lognormal laws, truncated or not, supply their own, or else the normal
      score reached through its quantile function, X = F^{-1}(Phi(z));
    - mixtures, objects with ``weights`` and ``components``, integrated
      component by component;
    - discrete distributions, summed over their atoms.

    In z, a side without a bound is cut at Z_UNBOUNDED standard deviations, and
    every integral is split at each multiple of Z_STEP, so that a few nodes a
    piece resolve the normal density.

    Parameters
    ----------
    func : callable, optional
        The function, called as ``func(x, *args)``. Without labels, x stacks the
        variables' values in its first axis, as expected() passes atoms; with
        labels, x is a dictionary of the values keyed by them. Defaults to the
        identity, which gives the means.
    dstns : list
        The independent distributions, one per variable; a single distribution
        need not be in a list.
    args : tuple, optional
        Other arguments of func.
    labels : list of str, optional
        Names of the variables, one per distribution.
    splits : list, optional
        For each distribution, None or a function giving the points at which its
        integral is split: ``splits[i](x_0, ..., x_{i-1}, *args)`` receives the
        values of the earlier variables and the arguments, broadcast as func
        receives them, and returns the split points in the last axis of an
        array. Points outside a law's support are ignored. By default nothing
        is split.
    nodes : int or list of int, optional
        Gauss-Legendre nodes a piece, for all distributions or for each one.
        The default is 4, which on the unit steps of the coordinate integrates
        the normal density to 6e-12 and smooth integrands to about 1e-8.

    Returns
    -------
    f_exp : np.array or float
        The expectation of func at the given arguments.
    """
    if func is None:

        def func(x, *args):
            return x

    if not isinstance(args, tuple):
        args = (args,)
    dstns = list(dstns) if isinstance(dstns, (list, tuple)) else [dstns]
    d = len(dstns)
    splits = [None] * d if splits is None else list(splits)
    nodes = [nodes] * d if np.ndim(nodes) == 0 else list(nodes)
    if len(splits) != d or len(nodes) != d:
        raise ValueError("splits and nodes must have one entry per distribution.")
    if labels is not None and len(labels) != d:
        raise ValueError("labels must have one entry per distribution.")

    # One trailing axis per distribution on each ndarray argument
    arrays = [arg for arg in args if isinstance(arg, np.ndarray)]
    lead = np.broadcast_shapes(*[arg.shape for arg in arrays]) if arrays else ()
    args = tuple(
        arg.reshape(arg.shape + (1,) * d) if isinstance(arg, np.ndarray) else arg
        for arg in args
    )

    # Each variable's nodes go in its own axis, after the arguments' axes
    values, weights = [], []
    for i in range(d):
        prefix = lead + tuple(v.shape[len(lead) + j] for j, v in enumerate(values))
        prefix = prefix + (1,) * (d - i)
        cuts = None
        if splits[i] is not None:
            cuts = np.asarray(splits[i](*values, *args), dtype=float)
            cuts = cuts.reshape(cuts.shape or (1,))
            cuts = np.broadcast_to(
                cuts, np.broadcast_shapes(cuts.shape[:-1], prefix) + cuts.shape[-1:]
            )
        v, w = _quadrature(dstns[i], cuts, int(nodes[i]))
        axis = len(lead) + i
        if v.ndim == 1:
            shape = (1,) * axis + (v.size,) + (1,) * (d - i - 1)
            v, w = v.reshape(shape), w.reshape(shape)
        else:
            v, w = np.swapaxes(v, axis, -1)[..., 0], np.swapaxes(w, axis, -1)[..., 0]
        values.append(v)
        weights.append(w)

    if labels is None:
        shape = np.broadcast_shapes(*[v.shape for v in values])
        x = np.stack([np.broadcast_to(v, shape) for v in values])
    else:
        x = dict(zip(labels, values))
    f = np.asarray(func(x, *args))
    weight = weights[0]
    for w in weights[1:]:
        weight = weight * w
    return np.sum(f * weight, axis=tuple(range(-d, 0)))
