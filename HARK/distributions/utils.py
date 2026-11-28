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


def employed_atoms(dstn):
    """
    Which atoms of a joint distribution of permanent and transitory income
    shocks are not the unemployment event.

    The event is the point that add_discrete_outcome_constant_mean added to the
    transitory shock, as IndShock's income constructor records it in the
    transitory component's limit. A distribution that records no such event of
    positive probability has none.

    Parameters
    ----------
    dstn : DiscreteDistribution
        A joint distribution of (PermShk, TranShk), as combine_indep_dstns
        records it.

    Returns
    -------
    keep : np.ndarray or None
        True at each atom whose transitory shock is not the unemployment income,
        or None where the distribution records no unemployment event.
    """
    limit = getattr(dstn, "limit", None) or {}
    dists = limit.get("dist")
    if limit.get("method") != "combine_indep_dstns" or dists is None or len(dists) != 2:
        return None
    tran = getattr(dists[1], "limit", None) or {}
    if tran.get("method") != "add_discrete_outcome_constant_mean":
        return None
    if not float(tran.get("p", 0.0)) > 0.0:
        return None
    keep = np.asarray(dstn.atoms)[1] != float(tran["x"])
    return keep if keep.any() and not keep.all() else None


def restrict_to_atoms(dstn, keep):
    """
    A discrete distribution restricted to the atoms where keep is True, its
    probabilities renormalized. It draws from dstn's own random number
    generator, so its draws continue dstn's stream.

    Parameters
    ----------
    dstn : DiscreteDistribution
        The distribution.
    keep : np.ndarray
        Which of its atoms to keep.

    Returns
    -------
    part : DiscreteDistribution
    """
    pmv = np.asarray(dstn.pmv)[keep]
    part = DiscreteDistribution(
        pmv / pmv.sum(), np.asarray(dstn.atoms)[:, keep], seed=0
    )
    part._rng = dstn._rng
    return part


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
    as the normal and lognormal laws, truncated or not, do. A law that is not
    the standard normal in the z it supplies also gives that z's density under
    it, through its method coordinate_density(), as the jumpless truncated
    lognormal does. For any other law z is its normal score, reached through
    its quantile function, X = F^{-1}(Phi(z)), so the integrand is smooth in z
    wherever f and the law are. A side without a bound is cut at Z_UNBOUNDED, and the edges are at
    every multiple of Z_STEP between the ends and at the law's breakpoints(),
    the values at which it is not smooth.
    """
    own_density = None
    if hasattr(dstn, "normal_coordinate"):
        bot, top, value, coord = dstn.normal_coordinate()
        own_density = getattr(dstn, "coordinate_density", None)
    elif hasattr(dstn, "ppf") and hasattr(dstn, "cdf") and not hasattr(dstn, "pmf"):
        bot, top = -np.inf, np.inf

        def value(z):
            return dstn.ppf(ndtr(z))

        def coord(x):
            return ndtri(dstn.cdf(x))

    else:
        raise TypeError(
            "expected_continuous() cannot integrate over a "
            f"{type(dstn).__name__}: it takes continuous laws with a quantile "
            "function and CDF, or a normal_coordinate() method, mixtures, and "
            "discrete distributions that hold their atoms and probabilities."
        )
    lo, hi = max(float(bot), -Z_UNBOUNDED), min(float(top), Z_UNBOUNDED)
    steps = np.arange(np.floor(lo / Z_STEP) + 1.0, np.ceil(hi / Z_STEP)) * Z_STEP
    edges = [lo] + [z for z in steps if lo < z < hi] + [hi]
    # The law's own breakpoints are edges too, so that no piece holds a step
    # in its density; one within rounding of an edge already there is that edge
    breaks = getattr(dstn, "breakpoints", None)
    if breaks is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            inner = np.ravel(coord(np.asarray(breaks(), dtype=float)))
        tol = 1e-12 * max(1.0, abs(lo), abs(hi))
        for z in inner[np.isfinite(inner)]:
            if lo < z < hi and min(abs(z - e) for e in edges) > tol:
                edges.append(float(z))
        edges.sort()
    if own_density is not None:
        return edges, value, coord, own_density
    mass = float(_normal_mass(lo, hi))

    def density(z):
        return np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * mass)

    return edges, value, coord, density


def _split_gauss_legendre(edges, n):
    """
    Gauss-Legendre nodes and weights, n a piece, on the pieces between sorted
    edges, with rows in the leading axes and the edges in the last. Pieces of
    zero length are dropped: each row keeps its pieces in order, padded with
    empty ones to the number the row with the most has. A padding piece's nodes
    have no weight, and lie inside the row's first piece, not on an edge, where
    an integrand may be infinite or undefined, as at an end of the support.
    Returns the nodes and the weights, each of shape rows + (m,).
    """
    edges = np.asarray(edges, dtype=float)
    empty = np.diff(edges, axis=-1) <= 0.0
    order = np.argsort(empty, axis=-1, kind="stable")
    count = max(int(np.max(np.sum(~empty, axis=-1), initial=0)), 1)
    a = np.take_along_axis(edges[..., :-1], order[..., :count], axis=-1)
    b = np.take_along_axis(edges[..., 1:], order[..., :count], axis=-1)
    pad = b <= a
    at_a, at_b = np.where(pad, a[..., :1], a), np.where(pad, b[..., :1], b)
    x, w = _gauss_legendre(n)
    nodes = 0.5 * (at_b - at_a)[..., None] * x + 0.5 * (at_b + at_a)[..., None]
    weights = np.where(pad, 0.0, 0.5 * (b - a))[..., None] * w
    shape = nodes.shape[:-2] + (-1,)
    return nodes.reshape(shape), weights.reshape(shape)


def _quadrature(dstn, cuts, n):
    """
    Nodes and weights for one distribution, with n Gauss-Legendre nodes on each
    piece between its cut points. With cuts of shape B + (k,), the returned
    values and weights have shape B + (m,), with the nodes in the last axis;
    without cuts they are 1D. A discrete distribution gives its atoms and
    probabilities whatever the cuts, and a mixture its components' nodes, with
    their weights multiplied by the components' weights. A discrete distribution
    is a DiscreteDistribution, or a law that holds its atoms and their
    probabilities as one does (atoms and pmv), as Bernoulli does.
    """
    if isinstance(dstn, DiscreteDistribution) or (
        hasattr(dstn, "atoms") and hasattr(dstn, "pmv")
    ):
        atoms = np.asarray(dstn.atoms, dtype=float)
        pmv = np.asarray(dstn.pmv, dtype=float)
        if atoms.ndim != 2 or atoms.shape[0] != 1 or pmv.shape != atoms.shape[1:]:
            raise ValueError(
                "expected_continuous() takes univariate distributions: pass the "
                "variables of a joint discrete distribution separately."
            )
        return atoms[0], pmv

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
      lognormal laws, truncated or not, supply their own (with z's density
      under the law, through coordinate_density(), where that is not the
      standard normal's), or else the normal score reached through its
      quantile function, X = F^{-1}(Phi(z));
    - mixtures, objects with ``weights`` and ``components``, integrated
      component by component;
    - discrete distributions, summed over their atoms.

    In z, a side without a bound is cut at Z_UNBOUNDED standard deviations, and
    every integral is split at each multiple of Z_STEP, so that a few nodes a
    piece resolve the normal density, and at each of the law's breakpoints(),
    the values at which it is not smooth, so that no piece holds one.

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
            # the means, in the distributions' order, with labels or without
            if isinstance(x, dict):
                return np.stack(np.broadcast_arrays(*x.values()))
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


# =============================================================================
# The tools behind expected_continuous() for laws in general: the ends of a
# law's support and how it holds probability there, laws scaled and mixed, the
# values at which a law's integral is cut, and a rule graded toward a near-pole
# beyond an end of the support
# =============================================================================


def law_bounds(law):
    """
    The ends of a continuous law's support: its infimum and supremum where it
    records them, else its quantile function at zero and one.
    """
    if hasattr(law, "infimum") and hasattr(law, "supremum"):
        return float(np.ravel(law.infimum)[0]), float(np.ravel(law.supremum)[0])
    return float(np.ravel(law.ppf(0.0))[0]), float(np.ravel(law.ppf(1.0))[0])


class _ScaledLaw:
    """
    The law of scale * X, for X distributed as a continuous law: its quantile
    function, CDF, survival function, density, bounds and breakpoints, all
    scaled, and the standard-normal coordinate the law supplies, if it supplies
    one, with its values scaled, and that coordinate's density, if the law
    supplies it.

    Parameters
    ----------
    law : Distribution
        Any law with a quantile function, CDF and density.
    scale : float
        A positive scale.
    """

    def __init__(self, law, scale):
        self.law = law
        self.scale = float(scale)
        lo, hi = law_bounds(law)
        self.infimum = np.array([self.scale * lo])
        self.supremum = np.array([self.scale * hi])

    def ppf(self, q):
        return self.scale * np.asarray(self.law.ppf(q))

    def cdf(self, x):
        return self.law.cdf(np.asarray(x) / self.scale)

    def sf(self, x):
        x = np.asarray(x) / self.scale
        return self.law.sf(x) if hasattr(self.law, "sf") else 1.0 - self.law.cdf(x)

    def pdf(self, x):
        return self.law.pdf(np.asarray(x) / self.scale) / self.scale

    def mean(self):
        return self.scale * float(np.squeeze(self.law.mean()))

    def breakpoints(self):
        # the law's own, scaled, where it has them
        breaks = getattr(self.law, "breakpoints", None)
        if breaks is None:
            return np.array([])
        return self.scale * np.asarray(breaks(), dtype=float)

    @property
    def normal_coordinate(self):
        # the law's own, if it has one (AttributeError otherwise)
        own, scale = self.law.normal_coordinate, self.scale

        def normal_coordinate():
            bot, top, value, coord = own()
            return (
                bot,
                top,
                lambda z: scale * np.asarray(value(z)),
                lambda x: coord(np.asarray(x) / scale),
            )

        return normal_coordinate

    @property
    def coordinate_density(self):
        # the law's own, if it has one (AttributeError otherwise): the scale
        # moves the values, not the coordinate
        return self.law.coordinate_density


class _Mixture:
    """
    A mixture as expected_continuous() reads one: its weights and its
    components, with the support's ends.
    """

    def __init__(self, weights, components):
        self.weights = np.asarray(weights, dtype=float)
        self.components = list(components)
        ends = [
            (np.min(part.atoms), np.max(part.atoms))
            if isinstance(part, DiscreteDistribution)
            else law_bounds(part)
            for weight, part in zip(self.weights, self.components)
            if weight > 0.0
        ]
        self.infimum = np.array([min(lo for lo, _ in ends)])
        self.supremum = np.array([max(hi for _, hi in ends)])


def scaled_law(law, scale):
    """
    The law of scale * X, for X distributed as law: a discrete law's atoms
    scaled, a mixture scaled component by component, and any other law a
    _ScaledLaw, which keeps the coordinate the law supplies.
    """
    if scale == 1.0:
        return law
    if isinstance(law, DiscreteDistribution):
        return DiscreteDistribution(law.pmv, scale * law.atoms)
    if hasattr(law, "components") and hasattr(law, "weights"):
        return _Mixture(
            law.weights, [scaled_law(part, scale) for part in law.components]
        )
    return _ScaledLaw(law, scale)


# How far from a near-pole beyond an end of a law's support, in the law's
# standard-normal coordinate, the graded rule applies: beyond it, a piece one step
# wide sees the pole at least seven half-widths away
GRADE_WITHIN = 3.0


# The step in the logarithm of the distance to the near-pole at which the graded
# part is cut: integrands there grow or fall like that distance's powers, so
# exponentially in its logarithm, and half a step keeps a few nodes a piece as
# accurate as on the unit steps of the law's coordinate
GRADE_STEP = 0.5


# How far out in a law's coordinate step_values gives its steps: beyond three
# standard deviations a kink's bump in another variable's integrand carries too
# little of the law to matter
STEPS_WITHIN = 3.0


def end_order(law, x, upper):
    """
    The power k with which a continuous law's probability approaches a finite end
    x of its support, P(X within h of x) ~ h^k as h falls: 1 for a density
    positive at x, 2 for one that rises linearly from zero there, and so on; None
    where the probability falls faster than any power, as through a tail. Read
    from the density's one-sided limit, f(x + h) ~ h^(k - 1) (f(x - h) at an upper
    end), at distances h falling by factors of ten from the distance that holds
    the law's first thousandth at that end: the estimates settle, their changes
    shrinking, where there is a power, and keep drifting through a tail.
    """
    q = 1.0 - 1e-3 if upper else 1e-3
    h = abs(float(np.ravel(law.ppf(np.array([q])))[0]) - x) * 10.0 ** -np.arange(14)
    h = h[h > 1e-12 * abs(x)]  # well clear of the rounding of x itself
    with np.errstate(all="ignore"):
        f = np.ravel(law.pdf(x + (-1.0 if upper else 1.0) * np.r_[h, 2.0 * h]))
        k = 1.0 + np.log(f[h.size :] / f[: h.size]) / np.log(2.0)
    inside = False  # the widest probes can leave the support
    for i in range(1, h.size):
        if not (np.isfinite(k[i - 1]) and np.isfinite(k[i])):
            if inside:
                return None  # the density vanishes toward the end: no power
            continue
        inside = True
        change, size = abs(k[i] - k[i - 1]), max(1.0, abs(k[i]))
        # an exact power, up to rounding, or changes shrinking toward one
        if change <= 1e-6 * size or (
            i > 1 and change <= 1e-4 * size and change <= 0.5 * abs(k[i - 1] - k[i - 2])
        ):
            return float(np.round(k[i])) if abs(k[i] - np.round(k[i])) < 1e-3 else k[i]
    return None  # still drifting, as through a tail


def step_values(law):
    """
    The values at which a law's integral is cut, as values of the law: its
    coordinate at each multiple of Z_STEP within STEPS_WITHIN of its center, and
    its ends where it holds probability there (end_order); for a mixture, its
    components'; for a discrete law, its atoms. An integral over another
    variable whose integrand is kinked where that variable reaches this law's
    values is cut at them, so that the bump the kink leaves there, as wide as
    this law, is resolved: the continuous expectation cuts the permanent shock's
    integral where next period's kink is reached through the transitory shock.
    """
    if isinstance(law, DiscreteDistribution):
        return np.unique(np.asarray(law.atoms, dtype=float))
    if hasattr(law, "components"):
        return np.unique(np.concatenate([step_values(part) for part in law.components]))
    edges, value, _, _ = _coordinate(law)
    edges = np.asarray(edges, dtype=float)
    if STEPS_WITHIN is not None:
        edges = edges[np.abs(edges) <= STEPS_WITHIN]
    lo, hi = law_bounds(law)
    ends = [
        x
        for x, upper in ((lo, False), (hi, True))
        if np.isfinite(x) and end_order(law, x, upper) is not None
    ]
    return np.unique(np.r_[np.asarray(value(edges), dtype=float), ends])


def graded_quadrature(law, n, pole, cuts=None):
    """
    Nodes and weights against a continuous law for an integrand that may grow
    like the inverse distance to a near-pole beyond an end of the support, at
    the values ``pole`` (an array, NaN where there is none), split at the
    values ``cuts`` (pole.shape + (k,), or None), where the integrand is not
    smooth. In the law's standard-normal coordinate z, with the pole at z_p a
    distance d beyond a finite end: where d < GRADE_WITHIN, the part of the
    support within GRADE_WITHIN of the pole is integrated in u = log|z - z_p|,
    in which the near-pole is smooth, cut at each GRADE_STEP of u, at each half
    of Z_STEP in z and at the cuts; the rest in z, cut at each multiple of Z_STEP
    and at the cuts, as expected_continuous() cuts it; n Gauss-Legendre nodes a
    piece. Where a law's end is finite in its values but not in its coordinate,
    as a law reached through its quantile function has it, the same is done in
    its values, with the part from the end to the law's median graded, at an end
    the law holds probability at (end_order), and cut at the half steps of its
    coordinate within STEPS_WITHIN of its center. Returns the values and the
    weights, each pole.shape + (m,), the weights carrying the law's density, and
    whether each row is graded.
    """
    edges, value, coord, density = _coordinate(law)
    lo, hi = edges[0], edges[-1]
    pole = np.asarray(pole, dtype=float)
    rows = pole.shape
    if cuts is None:
        cuts = np.full(rows + (0,), np.nan)
    cuts = np.asarray(cuts, dtype=float)
    in_z = lo > -Z_UNBOUNDED or hi < Z_UNBOUNDED  # a finite end in z
    with np.errstate(divide="ignore", invalid="ignore"):
        if in_z:
            t_lo, t_hi, tp = lo, hi, coord(pole)
            open_lo, open_hi = lo > -Z_UNBOUNDED, hi < Z_UNBOUNDED
            # the graded part is also cut at the coordinate's half steps
            half = 0.5 * Z_STEP
            steps = half * np.arange(np.ceil(lo / half), np.floor(hi / half) + 1.0)
            t_cuts = np.concatenate(
                [coord(cuts), np.broadcast_to(steps, rows + steps.shape)], axis=-1
            )
        else:
            t_lo, t_hi = law_bounds(law)
            # in values, the graded part is also cut at the half steps of the law's
            # coordinate within STEPS_WITHIN of its center
            half = 0.5 * Z_STEP
            reach = Z_UNBOUNDED if STEPS_WITHIN is None else STEPS_WITHIN
            steps = half * np.arange(
                -np.floor(reach / half), np.floor(reach / half) + 1
            )
            steps = np.asarray(value(steps), dtype=float)
            tp = pole
            t_cuts = np.concatenate(
                [
                    np.broadcast_to(cuts, rows + cuts.shape[-1:]),
                    np.broadcast_to(steps, rows + steps.shape),
                ],
                axis=-1,
            )
            open_lo = (
                bool(np.isfinite(t_lo)) and end_order(law, t_lo, False) is not None
            )
            open_hi = bool(np.isfinite(t_hi)) and end_order(law, t_hi, True) is not None
            median = float(np.ravel(law.ppf(np.array([0.5])))[0])
        # a pole at an end, to within rounding, as at the natural limit itself, is
        # graded too
        tol_lo = 1e-12 * max(1.0, abs(t_lo))
        tol_hi = 1e-12 * max(1.0, abs(t_hi))
        below = open_lo & np.isfinite(tp) & (tp <= t_lo + tol_lo)
        above = open_hi & np.isfinite(tp) & (tp >= t_hi - tol_hi) & ~below
        d = np.where(
            below,
            np.maximum(t_lo - tp, 0.0),
            np.where(above, np.maximum(tp - t_hi, 0.0), np.inf),
        )
        if in_z:
            graded = d < GRADE_WITHIN
            g = np.where(
                below,
                np.minimum(tp + GRADE_WITHIN, t_hi),
                np.maximum(tp - GRADE_WITHIN, t_lo),
            )
            z_g = g
        else:
            graded = np.where(below, d < median - t_lo, d < t_hi - median)
            g = np.full(rows, median)
            z_g = coord(g)
    graded = graded & (below | above)

    # the rest, in z, cut at each step and at the cuts
    z_from = np.where(graded & below, z_g, lo)
    z_to = np.where(graded & above, z_g, hi)
    z_cuts = np.nan_to_num(coord(cuts), nan=lo, posinf=hi, neginf=lo)
    plain = np.concatenate(
        [
            np.broadcast_to(np.asarray(edges, dtype=float), rows + (len(edges),)),
            np.broadcast_to(z_cuts, rows + z_cuts.shape[-1:]),
            z_from[..., None],
            z_to[..., None],
        ],
        axis=-1,
    )
    plain = np.sort(np.clip(plain, z_from[..., None], z_to[..., None]), axis=-1)
    z, wz = _split_gauss_legendre(plain, n)
    x_plain, w_plain = value(z), wz * density(z)
    if not np.any(graded):
        return x_plain, w_plain, graded

    # the graded part, in u = log of the distance to the pole, cut at each step of
    # GRADE_STEP in u and at the cuts
    sign = np.where(above, -1.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        # nearer than this the nodes would round onto the end itself
        floor = 1e-14 * np.maximum(1.0, np.abs(np.where(above, t_hi, t_lo)))
        near = np.maximum(np.where(graded, d, 1.0), floor)
        u_near = np.where(graded, np.log(near), 0.0)
        u_far = np.where(graded, np.log(np.where(graded, sign * (g - tp), 1.0)), 0.0)
        u_cuts = np.log(sign[..., None] * (t_cuts - tp[..., None]))
    # cuts on the pole's side of the end, or none, fall at the graded part's end
    u_cuts = np.where(np.isfinite(u_cuts), u_cuts, u_far[..., None])
    steps = GRADE_STEP * np.arange(
        np.floor(u_near[graded].min() / GRADE_STEP) + 1.0,
        np.ceil(u_far[graded].max() / GRADE_STEP),
    )
    part = np.concatenate(
        [
            u_near[..., None],
            np.broadcast_to(steps, rows + steps.shape),
            np.broadcast_to(u_cuts, rows + u_cuts.shape[-1:]),
            u_far[..., None],
        ],
        axis=-1,
    )
    part = np.sort(np.clip(part, u_near[..., None], u_far[..., None]), axis=-1)
    u, wu = _split_gauss_legendre(part, n)
    e = np.exp(u)
    t = np.where(graded[..., None], tp[..., None] + sign[..., None] * e, t_lo)
    with np.errstate(invalid="ignore"):
        if in_z:
            x_part, w_part = value(t), wu * e * density(t)
        else:
            x_part, w_part = t, wu * e * law.pdf(t)
    w_part = np.where(graded[..., None], w_part, 0.0)
    x_part = np.where(graded[..., None], x_part, x_plain[..., :1])
    return (
        np.concatenate([x_part, x_plain], axis=-1),
        np.concatenate([w_part, w_plain], axis=-1),
        graded,
    )


def _end_mass(law, x, upper):
    """
    The probability a law holds at the finite end x of its support itself, as a
    law with a point mass at its infimum holds it: its
    CDF at a lower end, and what its CDF leaves just below an upper end. Zero
    for a law that has no CDF.
    """
    cdf = getattr(law, "cdf", None)
    if cdf is None:
        return 0.0
    at = np.nextafter(x, -np.inf) if upper else x
    below = float(np.ravel(cdf(np.array([at])))[0])
    return 1.0 - below if upper else below


def law_end(law, upper):
    """
    One end of a law's support, and how the law holds probability there: 0 for a
    mass, as a discrete law has at each end and a law with a point mass
    at an end has there (_end_mass); otherwise the power k with which its probability
    approaches the end, P(X within h of it) ~ h^k (end_order), 1 for a density
    positive there, as a law truncated there has; None where the law approaches
    the end only through a tail, as an untruncated lognormal approaches zero. A
    mixture law (an object with weights and components) holds its end as the
    components with weight that reach it do.
    """
    if isinstance(law, DiscreteDistribution):
        x = np.asarray(law.atoms, dtype=float)
        return (x.max() if upper else x.min()), 0
    if hasattr(law, "components") and hasattr(law, "weights"):
        parts = [
            law_end(part, upper)
            for weight, part in zip(law.weights, law.components)
            if weight > 0.0
        ]
        x = max(p[0] for p in parts) if upper else min(p[0] for p in parts)
        held = [order for at, order in parts if at == x and order is not None]
        return x, (min(held) if held else None)
    lo, hi = law_bounds(law)
    x = hi if upper else lo
    if lo == hi:
        return x, 0  # a point
    if not np.isfinite(x):
        return x, None
    if _end_mass(law, x, upper) > 1e-12:
        return x, 0  # a point mass at the end
    return x, end_order(law, x, upper)


def end_probability(law, upper):
    """
    The probability a law holds at one end of its support itself: the atom's
    there, for a discrete law; a point mass's at the end (_end_mass); zero for a
    density.
    A mixture law holds what its components with weight that reach the end hold.
    """
    if isinstance(law, DiscreteDistribution):
        x = np.asarray(law.atoms, dtype=float).ravel()
        prb = np.asarray(law.pmv, dtype=float).ravel()
        return float(np.sum(prb[x == (x.max() if upper else x.min())]))
    if hasattr(law, "components") and hasattr(law, "weights"):
        parts = [
            (law_end(part, upper)[0], weight * end_probability(part, upper))
            for weight, part in zip(law.weights, law.components)
            if weight > 0.0
        ]
        x = max(at for at, _ in parts) if upper else min(at for at, _ in parts)
        return float(sum(mass for at, mass in parts if at == x))
    lo, hi = law_bounds(law)
    x = hi if upper else lo
    if lo == hi:
        return 1.0  # a point
    if not np.isfinite(x):
        return 0.0
    return max(_end_mass(law, x, upper), 0.0)
