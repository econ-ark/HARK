'''
General purpose  / miscellaneous functions.  Includes functions to approximate
continuous distributions with discrete ones, utility functions (and their
derivatives), manipulation of discrete distributions, and basic plotting tools.
'''

from __future__ import division     # Import Python 3.x division function
from __future__ import print_function
from builtins import str
from builtins import range
from builtins import object
import functools
import warnings
import numpy as np                  # Python's numeric library, abbreviated "np"
import math
# try:
#     import matplotlib.pyplot as plt                 # Python's plotting library
# except ImportError:
#     import sys
#     exception_type, value, traceback = sys.exc_info()
#     raise ImportError('HARK must be used in a graphical environment.', exception_type, value, traceback)
import scipy.stats as stats         # Python's statistics library
from scipy.interpolate import interp1d
from scipy.special import erf, erfc

def memoize(obj):
   '''
   A decorator to (potentially) make functions more efficient.

   With this decorator, functions will "remember" if they have been evaluated with given inputs
   before.  If they have, they will "remember" the outputs that have already been calculated
   for those inputs, rather than calculating them again.
   '''
   cache = obj._cache = {}

   @functools.wraps(obj)
   def memoizer(*args, **kwargs):
       key = str(args) + str(kwargs)
       if key not in cache:
           cache[key] = obj(*args, **kwargs)
       return cache[key]
   return memoizer


# ==============================================================================
# ============== Some basic function tools  ====================================
# ==============================================================================
def getArgNames(function):
    '''
    Returns a list of strings naming all of the arguments for the passed function.

    Parameters
    ----------
    function : function
        A function whose argument names are wanted.

    Returns
    -------
    argNames : [string]
        The names of the arguments of function.
    '''
    argCount = function.__code__.co_argcount
    argNames = function.__code__.co_varnames[:argCount]
    return argNames


class NullFunc(object):
    '''
    A trivial class that acts as a placeholder "do nothing" function.
    '''
    def __call__(self,*args):
        '''
        Returns meaningless output no matter what the input(s) is.  If no input,
        returns None.  Otherwise, returns an array of NaNs (or a single NaN) of
        the same size as the first input.
        '''
        if len(args) == 0:
            return None
        else:
            arg = args[0]
            if hasattr(arg,'shape'):
                return np.zeros_like(arg) + np.nan
            else:
                return np.nan

    def distance(self,other):
        '''
        Trivial distance metric that only cares whether the other object is also
        an instance of NullFunc.  Intentionally does not inherit from HARKobject
        as this might create dependency problems.

        Parameters
        ----------
        other : any
            Any object for comparison to this instance of NullFunc.

        Returns
        -------
        (unnamed) : float
            The distance between self and other.  Returns 0 if other is also a
            NullFunc; otherwise returns an arbitrary high number.
        '''
        try:
            if other.__class__ is self.__class__:
                return 0.0
            else:
                return 1000.0
        except:
            return 10000.0

# ==============================================================================
# ============== Define utility functions        ===============================
# ==============================================================================
def CRRAutility(c, gam):
    '''
    Evaluates constant relative risk aversion (CRRA) utility of consumption c
    given risk aversion parameter gam.

    Parameters
    ----------
    c : float
        Consumption value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Utility

    Tests
    -----
    Test a value which should pass:
    >>> c, gamma = 1.0, 2.0    # Set two values at once with Python syntax
    >>> utility(c=c, gam=gamma)
    -1.0
    '''
    if gam == 1:
        return np.log(c)
    else:
        return( c**(1.0 - gam) / (1.0 - gam) )

def CRRAutilityP(c, gam):
    '''
    Evaluates constant relative risk aversion (CRRA) marginal utility of consumption
    c given risk aversion parameter gam.

    Parameters
    ----------
    c : float
        Consumption value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Marginal utility
    '''
    return( c**-gam )

def CRRAutilityPP(c, gam):
    '''
    Evaluates constant relative risk aversion (CRRA) marginal marginal utility of
    consumption c given risk aversion parameter gam.

    Parameters
    ----------
    c : float
        Consumption value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Marginal marginal utility
    '''
    return( -gam*c**(-gam-1.0) )

def CRRAutilityPPP(c, gam):
    '''
    Evaluates constant relative risk aversion (CRRA) marginal marginal marginal
    utility of consumption c given risk aversion parameter gam.

    Parameters
    ----------
    c : float
        Consumption value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Marginal marginal marginal utility
    '''
    return( (gam+1.0)*gam*c**(-gam-2.0) )

def CRRAutilityPPPP(c, gam):
    '''
    Evaluates constant relative risk aversion (CRRA) marginal marginal marginal
    marginal utility of consumption c given risk aversion parameter gam.

    Parameters
    ----------
    c : float
        Consumption value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Marginal marginal marginal marginal utility
    '''
    return( -(gam+2.0)*(gam+1.0)*gam*c**(-gam-3.0) )

def CRRAutility_inv(u, gam):
    '''
    Evaluates the inverse of the CRRA utility function (with risk aversion para-
    meter gam) at a given utility level u.

    Parameters
    ----------
    u : float
        Utility value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Consumption corresponding to given utility value
    '''
    if gam == 1:
        return np.exp(u)
    else:
        return( ((1.0-gam)*u)**(1/(1.0-gam)) )

def CRRAutilityP_inv(uP, gam):
    '''
    Evaluates the inverse of the CRRA marginal utility function (with risk aversion
    parameter gam) at a given marginal utility level uP.

    Parameters
    ----------
    uP : float
        Marginal utility value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Consumption corresponding to given marginal utility value.
    '''
    return( uP**(-1.0/gam) )

def CRRAutility_invP(u, gam):
    '''
    Evaluates the derivative of the inverse of the CRRA utility function (with
    risk aversion parameter gam) at a given utility level u.

    Parameters
    ----------
    u : float
        Utility value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Marginal consumption corresponding to given utility value
    '''
    if gam == 1:
        return np.exp(u)
    else:
        return( ((1.0-gam)*u)**(gam/(1.0-gam)) )

def CRRAutilityP_invP(uP, gam):
    '''
    Evaluates the derivative of the inverse of the CRRA marginal utility function
    (with risk aversion parameter gam) at a given marginal utility level uP.

    Parameters
    ----------
    uP : float
        Marginal utility value
    gam : float
        Risk aversion

    Returns
    -------
    (unnamed) : float
        Marginal consumption corresponding to given marginal utility value
    '''
    return( (-1.0/gam)*uP**(-1.0/gam-1.0) )


def CARAutility(c, alpha):
    '''
    Evaluates constant absolute risk aversion (CARA) utility of consumption c
    given risk aversion parameter alpha.

    Parameters
    ----------
    c: float
        Consumption value
    alpha: float
        Risk aversion

    Returns
    -------
    (unnamed): float
        Utility
    '''
    return( 1 - np.exp(-alpha*c)/alpha )

def CARAutilityP(c, alpha):
    '''
    Evaluates constant absolute risk aversion (CARA) marginal utility of
    consumption c given risk aversion parameter alpha.

    Parameters
    ----------
    c: float
        Consumption value
    alpha: float
        Risk aversion

    Returns
    -------
    (unnamed): float
        Marginal utility
    '''
    return( np.exp(-alpha*c) )

def CARAutilityPP(c, alpha):
    '''
    Evaluates constant absolute risk aversion (CARA) marginal marginal utility
    of consumption c given risk aversion parameter alpha.

    Parameters
    ----------
    c: float
        Consumption value
    alpha: float
        Risk aversion

    Returns
    -------
    (unnamed): float
        Marginal marginal utility
    '''
    return( -alpha*np.exp(-alpha*c) )

def CARAutilityPPP(c, alpha):
    '''
    Evaluates constant absolute risk aversion (CARA) marginal marginal marginal
    utility of consumption c given risk aversion parameter alpha.

    Parameters
    ----------
    c: float
        Consumption value
    alpha: float
        Risk aversion

    Returns
    -------
    (unnamed): float
        Marginal marginal marginal utility
    '''
    return( alpha**2.0*np.exp(-alpha*c) )

def CARAutility_inv(u, alpha):
    '''
    Evaluates inverse of constant absolute risk aversion (CARA) utility function
    at utility level u given risk aversion parameter alpha.

    Parameters
    ----------
    u: float
        Utility value
    alpha: float
        Risk aversion

    Returns
    -------
    (unnamed): float
        Consumption value corresponding to u
    '''
    return( -1.0/alpha * np.log(alpha*(1-u)) )

def CARAutilityP_inv(u, alpha):
    '''
    Evaluates the inverse of constant absolute risk aversion (CARA) marginal
    utility function at marginal utility uP given risk aversion parameter alpha.

    Parameters
    ----------
    u: float
        Utility value
    alpha: float
        Risk aversion

    Returns
    -------
    (unnamed): float
        Consumption value corresponding to uP
    '''
    return( -1.0/alpha*np.log(u) )

def CARAutility_invP(u, alpha):
    '''
    Evaluates the derivative of inverse of constant absolute risk aversion (CARA)
    utility function at utility level u given risk aversion parameter alpha.

    Parameters
    ----------
    u: float
        Utility value
    alpha: float
        Risk aversion

    Returns
    -------
    (unnamed): float
        Marginal onsumption value corresponding to u
    '''
    return( 1.0/(alpha*(1.0-u)) )


def approxLognormal(N, mu=0.0, sigma=1.0, tail_N=0, tail_bound=[0.02,0.98], tail_order=np.e):
    '''
    Construct a discrete approximation to a lognormal distribution with underlying
    normal distribution N(mu,sigma).  Makes an equiprobable distribution by
    default, but user can optionally request augmented tails with exponentially
    sized point masses.  This can improve solution accuracy in some models.

    Parameters
    ----------
    N: int
        Number of discrete points in the "main part" of the approximation.
    mu: float
        Mean of underlying normal distribution.
    sigma: float
        Standard deviation of underlying normal distribution.
    tail_N: int
        Number of points in each "tail part" of the approximation; 0 = no tail.
    tail_bound: [float]
        CDF boundaries of the tails vs main portion; tail_bound[0] is the lower
        tail bound, tail_bound[1] is the upper tail bound.  Inoperative when
        tail_N = 0.  Can make "one tailed" approximations with 0.0 or 1.0.
    tail_order: float
        Factor by which consecutive point masses in a "tail part" differ in
        probability.  Should be >= 1 for sensible spacing.

    Returns
    -------
    pmf: np.ndarray
        Probabilities for discrete probability mass function.
    X: np.ndarray
        Discrete values in probability mass function.

    Written by Luca Gerotto
    Based on Matab function "setup_workspace.m," from Chris Carroll's
      [Solution Methods for Microeconomic Dynamic Optimization Problems]
      (http://www.econ2.jhu.edu/people/ccarroll/solvingmicrodsops/) toolkit.
    Latest update: 11 February 2017 by Matthew N. White
    '''
    # Find the CDF boundaries of each segment
    if sigma > 0.0:
        if tail_N > 0:
            lo_cut     = tail_bound[0]
            hi_cut     = tail_bound[1]
        else:
            lo_cut     = 0.0
            hi_cut     = 1.0
        inner_size     = hi_cut - lo_cut
        inner_CDF_vals = [lo_cut + x*N**(-1.0)*inner_size for x in range(1, N)]
        if inner_size < 1.0:
            scale      = 1.0/tail_order
            mag        = (1.0-scale**tail_N)/(1.0-scale)
        lower_CDF_vals = [0.0]
        if lo_cut > 0.0:
            for x in range(tail_N-1,-1,-1):
                lower_CDF_vals.append(lower_CDF_vals[-1] + lo_cut*scale**x/mag)
        upper_CDF_vals  = [hi_cut]
        if hi_cut < 1.0:
            for x in range(tail_N):
                upper_CDF_vals.append(upper_CDF_vals[-1] + (1.0-hi_cut)*scale**x/mag)
        CDF_vals       = lower_CDF_vals + inner_CDF_vals + upper_CDF_vals
        temp_cutoffs   = list(stats.lognorm.ppf(CDF_vals[1:-1], s=sigma, loc=0, scale=np.exp(mu)))
        cutoffs        = [0] + temp_cutoffs + [np.inf]
        CDF_vals       = np.array(CDF_vals)

        # Construct the discrete approximation by finding the average value within each segment.
        # This codeblock ignores warnings because it throws a "divide by zero encountered in log"
        # warning due to computing erf(infty) at the tail boundary.  This is irrelevant and
        # apparently freaks new users out.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            K              = CDF_vals.size-1 # number of points in approximation
            pmf            = CDF_vals[1:(K+1)] - CDF_vals[0:K]
            X              = np.zeros(K)
            for i in range(K):
                zBot  = cutoffs[i]
                zTop = cutoffs[i+1]
                tempBot = (mu+sigma**2-np.log(zBot))/(np.sqrt(2)*sigma)
                tempTop = (mu+sigma**2-np.log(zTop))/(np.sqrt(2)*sigma)
                if tempBot <= 4:
                    X[i] = -0.5*np.exp(mu+(sigma**2)*0.5)*(erf(tempTop) - erf(tempBot))/pmf[i]
                else:
                    X[i] = -0.5*np.exp(mu+(sigma**2)*0.5)*(erfc(tempBot) - erfc(tempTop))/pmf[i]

    else:
        pmf = np.ones(N)/N
        X   = np.exp(mu)*np.ones(N)
    return [pmf, X]

@memoize
def approxMeanOneLognormal(N, sigma=1.0, **kwargs):
    '''
    Calculate a discrete approximation to a mean one lognormal distribution.
    Based on function approxLognormal; see that function's documentation for
    further notes.

    Parameters
    ----------
    N : int
        Size of discrete space vector to be returned.
    sigma : float
        standard deviation associated with underlying normal probability distribution.

    Returns
    -------
    X : np.array
        Discrete points for discrete probability mass function.
    pmf : np.array
        Probability associated with each point in X.

    Written by Nathan M. Palmer
    Based on Matab function "setup_shocks.m," from Chris Carroll's
      [Solution Methods for Microeconomic Dynamic Optimization Problems]
      (http://www.econ2.jhu.edu/people/ccarroll/solvingmicrodsops/) toolkit.
    Latest update: 01 May 2015
    '''
    mu_adj = - 0.5*sigma**2;
    pmf,X = approxLognormal(N=N, mu=mu_adj, sigma=sigma, **kwargs)
    return [pmf,X]

def approxNormal(N, mu=0.0, sigma=1.0):
    x, w = np.polynomial.hermite.hermgauss(N)
    # normalize w
    pmf = w*np.pi**-0.5
    # correct x
    X = math.sqrt(2.0)*sigma*x + mu
    return [pmf, X]

def approxLognormalGaussHermite(N, mu=0.0, sigma=1.0):
    pmf, X = approxNormal(N, mu, sigma)
    return pmf, np.exp(X)

def calcNormalStyleParsFromLognormalPars(avgLognormal, stdLognormal):
    varLognormal = stdLognormal**2
    avgNormal = math.log(avgLognormal/math.sqrt(1+varLognormal/avgLognormal**2))
    varNormal = math.sqrt(math.log(1+varLognormal/avgLognormal**2))
    stdNormal = math.sqrt(varNormal)
    return avgNormal, stdNormal

def calcLognormalStyleParsFromNormalPars(muNormal, stdNormal):
    varNormal = stdNormal**2
    avgLognormal = math.exp(muNormal+varNormal*0.5)
    varLognormal = (math.exp(varNormal)-1)*math.exp(2*muNormal+varNormal)
    stdLognormal = math.sqrt(varLognormal)
    return avgLognormal, stdLognormal

def approxBeta(N,a=1.0,b=1.0):
    '''
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
    X : np.array
        Discrete points for discrete probability mass function.
    pmf : np.array
        Probability associated with each point in X.
    '''
    P    = 1000
    vals = np.reshape(stats.beta.ppf(np.linspace(0.0,1.0,N*P),a,b),(N,P))
    X    = np.mean(vals,axis=1)
    pmf  = np.ones(N)/float(N)
    return( [pmf, X] )

def approxUniform(N,bot=0.0,top=1.0):
    '''
    Makes a discrete approximation to a uniform distribution, given its bottom
    and top limits and number of points.

    Parameters
    ----------
    N : int
        The number of points in the discrete approximation
    bot : float
        The bottom of the uniform distribution
    top : float
        The top of the uniform distribution

    Returns
    -------
    (unnamed) : np.array
        An equiprobable discrete approximation to the uniform distribution.
    '''
    pmf = np.ones(N)/float(N)
    center = (top+bot)/2.0
    width = (top-bot)/2.0
    X = center + width*np.linspace(-(N-1.0)/2.0,(N-1.0)/2.0,N)/(N/2.0)
    return [pmf,X]


def makeMarkovApproxToNormal(x_grid,mu,sigma,K=351,bound=3.5):
    '''
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
    '''
    x_n = x_grid.size     # Number of points in the outcome grid
    lower_bound = -bound  # Lower bound of normal draws to consider, in SD
    upper_bound = bound   # Upper bound of normal draws to consider, in SD
    raw_sample = np.linspace(lower_bound,upper_bound,K) # Evenly spaced draws between bounds
    f_weights = stats.norm.pdf(raw_sample) # Relative probability of each draw
    sample = mu + sigma*raw_sample # Adjusted bounds, given mean and stdev
    w_vec = np.zeros(x_n) # A vector of outcome weights

    # Find the relative position of each of the draws
    sample_pos = np.searchsorted(x_grid,sample)
    sample_pos[sample_pos < 1] = 1
    sample_pos[sample_pos > x_n-1] = x_n-1

    # Make arrays of the x_grid point directly above and below each draw
    bot = x_grid[sample_pos-1]
    top = x_grid[sample_pos]
    alpha = (sample-bot)/(top-bot)

    # Keep the weights (alpha) in bounds
    alpha_clipped = np.clip(alpha,0.,1.)

    # Loop through each x_grid point and add up the probability that each nearby
    # draw contributes to it (accounting for distance)
    for j in range(1,x_n):
        c = sample_pos == j
        w_vec[j-1] = w_vec[j-1] + np.dot(f_weights[c],1.0-alpha_clipped[c])
        w_vec[j] = w_vec[j] + np.dot(f_weights[c],alpha_clipped[c])

    # Reweight the probabilities so they sum to 1
    W = np.sum(w_vec)
    p_vec = w_vec/W

    # Check for obvious errors, and return p_vec
    assert (np.all(p_vec>=0.)) and (np.all(p_vec<=1.)) and (np.isclose(np.sum(p_vec),1.))
    return p_vec

def makeMarkovApproxToNormalByMonteCarlo(x_grid,mu,sigma,N_draws = 10000):
    '''
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
    '''

    # Take random draws from the desired normal distribution
    random_draws = np.random.normal(loc = mu, scale = sigma, size = N_draws)

    # Compute the distance between the draws and points in x_grid
    distance = np.abs(x_grid[:,np.newaxis] - random_draws[np.newaxis,:])

    # Find the indices of the points in x_grid that are closest to the draws
    distance_minimizing_index = np.argmin(distance,axis=0)

    # For each point in x_grid, the approximate probability of that point is the number
    # of Monte Carlo draws that are closest to that point
    p_vec = np.zeros_like(x_grid)
    for p_index,p in enumerate(p_vec):
        p_vec[p_index] = np.sum(distance_minimizing_index==p_index) / N_draws

    # Check for obvious errors, and return p_vec
    assert (np.all(p_vec>=0.)) and (np.all(p_vec<=1.)) and (np.isclose(np.sum(p_vec)),1.)
    return p_vec


def makeTauchenAR1(N, sigma=1.0, rho=0.9, bound=3.0):
    '''
    Function to return a discretized version of an AR1 process.
    See http://www.fperri.net/TEACHING/macrotheory08/numerical.pdf for details

    Parameters
    ----------
    N: int
        Size of discretized grid
    sigma: float
        Standard deviation of the error term
    rho: float
        AR1 coefficient
    bound: float
        The highest (lowest) grid point will be bound (-bound) multiplied by the unconditional
        standard deviation of the process

    Returns
    -------
    y: np.array
        Grid points on which the discretized process takes values
    trans_matrix: np.array
        Markov transition array for the discretized process

    Written by Edmund S. Crawley
    Latest update: 27 October 2017
    '''
    yN = bound*sigma/((1-rho**2)**0.5)
    y = np.linspace(-yN,yN,N)
    d = y[1]-y[0]
    trans_matrix = np.ones((N,N))
    for j in range(N):
        for k_1 in range(N-2):
            k=k_1+1
            trans_matrix[j,k] = stats.norm.cdf((y[k] + d/2.0 - rho*y[j])/sigma) - stats.norm.cdf((y[k] - d/2.0 - rho*y[j])/sigma)
        trans_matrix[j,0] = stats.norm.cdf((y[0] + d/2.0 - rho*y[j])/sigma)
        trans_matrix[j,N-1] = 1.0 - stats.norm.cdf((y[N-1] - d/2.0 - rho*y[j])/sigma)

    return y, trans_matrix


# ================================================================================
# ==================== Functions for manipulating discrete distributions =========
# ================================================================================

def addDiscreteOutcomeConstantMean(distribution, x, p, sort = False):
    '''
    Adds a discrete outcome of x with probability p to an existing distribution,
    holding constant the relative probabilities of other outcomes and overall mean.

    Parameters
    ----------
    distribution : [np.array]
        Two element list containing a list of probabilities and a list of outcomes.
    x : float
        The new value to be added to the distribution.
    p : float
        The probability of the discrete outcome x occuring.
    sort: bool
        Whether or not to sort X before returning it

    Returns
    -------
    X : np.array
        Discrete points for discrete probability mass function.
    pmf : np.array
        Probability associated with each point in X.

    Written by Matthew N. White
    Latest update: 08 December 2015 by David Low
    '''
    X   = np.append(x,distribution[1]*(1-p*x)/(1-p))
    pmf = np.append(p,distribution[0]*(1-p))

    if sort:
        indices = np.argsort(X)
        X       = X[indices]
        pmf     = pmf[indices]

    return([pmf,X])

def addDiscreteOutcome(distribution, x, p, sort = False):
    '''
    Adds a discrete outcome of x with probability p to an existing distribution,
    holding constant the relative probabilities of other outcomes.

    Parameters
    ----------
    distribution : [np.array]
        Two element list containing a list of probabilities and a list of outcomes.
    x : float
        The new value to be added to the distribution.
    p : float
        The probability of the discrete outcome x occuring.

    Returns
    -------
    X : np.array
        Discrete points for discrete probability mass function.
    pmf : np.array
        Probability associated with each point in X.

    Written by Matthew N. White
    Latest update: 11 December 2015
    '''

    X   = np.append(x,distribution[1])
    pmf = np.append(p,distribution[0]*(1-p))

    if sort:
        indices = np.argsort(X)
        X       = X[indices]
        pmf     = pmf[indices]

    return([pmf,X])

def combineIndepDstns(*distributions):
    '''
    Given n lists (or tuples) whose elements represent n independent, discrete
    probability spaces (probabilities and values), construct a joint pmf over
    all combinations of these independent points.  Can take multivariate discrete
    distributions as inputs.

    Parameters
    ----------
    distributions : [np.array]
        Arbitrary number of distributions (pmfs).  Each pmf is a list or tuple.
        For each pmf, the first vector is probabilities and all subsequent vectors
        are values.  For each pmf, this should be true:
        len(X_pmf[0]) == len(X_pmf[j]) for j in range(1,len(distributions))

    Returns
    -------
    List of arrays, consisting of:

    P_out: np.array
        Probability associated with each point in X_out.

    X_out: np.array (as many as in *distributions)
        Discrete points for the joint discrete probability mass function.

    Written by Nathan Palmer
    Latest update: 5 July August 2017 by Matthew N White
    '''
    # Very quick and incomplete parameter check:
    for dist in distributions:
        assert len(dist[0]) == len(dist[-1]), "len(dist[0]) != len(dist[-1])"

    # Get information on the distributions
    dist_lengths = ()
    dist_dims = ()
    for dist in distributions:
        dist_lengths += (len(dist[0]),)
        dist_dims += (len(dist)-1,)
    number_of_distributions = len(distributions)

    # Initialize lists we will use
    X_out  = []
    P_temp = []

    # Now loop through the distributions, tiling and flattening as necessary.
    for dd,dist in enumerate(distributions):

        # The shape we want before we tile
        dist_newshape = (1,) * dd + (len(dist[0]),) + \
                        (1,) * (number_of_distributions - dd)

        # The tiling we want to do
        dist_tiles    = dist_lengths[:dd] + (1,) + dist_lengths[dd+1:]

        # Now we are ready to tile.
        # We don't use the np.meshgrid commands, because they do not
        # easily support non-symmetric grids.

        # First deal with probabilities
        Pmesh  = np.tile(dist[0].reshape(dist_newshape),dist_tiles) # Tiling
        flatP  = Pmesh.ravel() # Flatten the tiled arrays
        P_temp += [flatP,] #Add the flattened arrays to the output lists

        # Then loop through each value variable
        for n in range(1,dist_dims[dd]+1):
            Xmesh  = np.tile(dist[n].reshape(dist_newshape),dist_tiles)
            flatX  = Xmesh.ravel()
            X_out  += [flatX,]

    # We're done getting the flattened X_out arrays we wanted.
    # However, we have a bunch of flattened P_temp arrays, and just want one
    # probability array. So get the probability array, P_out, here.
    P_out = np.prod(np.array(P_temp),axis=0)

    assert np.isclose(np.sum(P_out),1),'Probabilities do not sum to 1!'
    return [P_out,] + X_out

# ==============================================================================
# ============== Functions for generating state space grids  ===================
# ==============================================================================
def makeGridExpMult(ming, maxg, ng, timestonest=20):
    '''
    Make a multi-exponentially spaced grid.

    Parameters
    ----------
    ming : float
        Minimum value of the grid
    maxg : float
        Maximum value of the grid
    ng : int
        The number of grid points
    timestonest : int
        the number of times to nest the exponentiation

    Returns
    -------
    points : np.array
        A multi-exponentially spaced grid

    Original Matab code can be found in Chris Carroll's
    [Solution Methods for Microeconomic Dynamic Optimization Problems]
    (http://www.econ2.jhu.edu/people/ccarroll/solvingmicrodsops/) toolkit.
    Latest update: 01 May 2015
    '''
    if timestonest > 0:
        Lming = ming
        Lmaxg = maxg
        for j in range(timestonest):
            Lming = np.log(Lming + 1)
            Lmaxg = np.log(Lmaxg + 1)
        Lgrid = np.linspace(Lming,Lmaxg,ng)
        grid = Lgrid
        for j in range(timestonest):
            grid = np.exp(grid) - 1
    else:
        Lming = np.log(ming)
        Lmaxg = np.log(maxg)
        Lstep = (Lmaxg - Lming)/(ng - 1)
        Lgrid = np.arange(Lming,Lmaxg+0.000001,Lstep)
        grid = np.exp(Lgrid)
    return(grid)


# ==============================================================================
# ============== Uncategorized general functions  ===================
# ==============================================================================
def calcWeightedAvg(data,weights):
    '''
    Generates a weighted average of simulated data.  The Nth row of data is averaged
    and then weighted by the Nth element of weights in an aggregate average.

    Parameters
    ----------
    data : numpy.array
        An array of data with N rows of J floats
    weights : numpy.array
        A length N array of weights for the N rows of data.

    Returns
    -------
    weighted_sum : float
        The weighted sum of the data.
    '''
    data_avg = np.mean(data,axis=1)
    weighted_sum = np.dot(data_avg,weights)
    return weighted_sum

def getPercentiles(data,weights=None,percentiles=[0.5],presorted=False):
    '''
    Calculates the requested percentiles of (weighted) data.  Median by default.

    Parameters
    ----------
    data : numpy.array
        A 1D array of float data.
    weights : np.array
        A weighting vector for the data.
    percentiles : [float]
        A list of percentiles to calculate for the data.  Each element should
        be in (0,1).
    presorted : boolean
        Indicator for whether data has already been sorted.

    Returns
    -------
    pctl_out : numpy.array
        The requested percentiles of the data.
    '''
    if data.size < 2:
        return np.zeros(np.array(percentiles).shape) + np.nan
    
    if weights is None: # Set equiprobable weights if none were passed
        weights = np.ones(data.size)/float(data.size)

    if presorted: # Sort the data if it is not already
        data_sorted = data
        weights_sorted = weights
    else:
        order = np.argsort(data)
        data_sorted = data[order]
        weights_sorted = weights[order]

    cum_dist = np.cumsum(weights_sorted)/np.sum(weights_sorted) # cumulative probability distribution

    # Calculate the requested percentiles by interpolating the data over the
    # cumulative distribution, then evaluating at the percentile values
    inv_CDF = interp1d(cum_dist,data_sorted,bounds_error=False,assume_sorted=True)
    pctl_out = inv_CDF(percentiles)
    return pctl_out

def getLorenzShares(data,weights=None,percentiles=[0.5],presorted=False):
    '''
    Calculates the Lorenz curve at the requested percentiles of (weighted) data.
    Median by default.

    Parameters
    ----------
    data : numpy.array
        A 1D array of float data.
    weights : numpy.array
        A weighting vector for the data.
    percentiles : [float]
        A list of percentiles to calculate for the data.  Each element should
        be in (0,1).
    presorted : boolean
        Indicator for whether data has already been sorted.

    Returns
    -------
    lorenz_out : numpy.array
        The requested Lorenz curve points of the data.
    '''
    if weights is None: # Set equiprobable weights if none were given
        weights = np.ones(data.size)

    if presorted: # Sort the data if it is not already
        data_sorted = data
        weights_sorted = weights
    else:
        order = np.argsort(data)
        data_sorted = data[order]
        weights_sorted = weights[order]

    cum_dist = np.cumsum(weights_sorted)/np.sum(weights_sorted) # cumulative probability distribution
    temp = data_sorted*weights_sorted
    cum_data = np.cumsum(temp)/sum(temp) # cumulative ownership shares

    # Calculate the requested Lorenz shares by interpolating the cumulative ownership
    # shares over the cumulative distribution, then evaluating at requested points
    lorenzFunc = interp1d(cum_dist,cum_data,bounds_error=False,assume_sorted=True)
    lorenz_out = lorenzFunc(percentiles)
    return lorenz_out

def calcSubpopAvg(data,reference,cutoffs,weights=None):
    '''
    Calculates the average of (weighted) data between cutoff percentiles of a
    reference variable.

    Parameters
    ----------
    data : numpy.array
        A 1D array of float data.
    reference : numpy.array
        A 1D array of float data of the same length as data.
    cutoffs : [(float,float)]
        A list of doubles with the lower and upper percentile bounds (should be
        in [0,1]).
    weights : numpy.array
        A weighting vector for the data.

    Returns
    -------
    slice_avg
        The (weighted) average of data that falls within the cutoff percentiles
        of reference.

    '''
    if weights is None: # Set equiprobable weights if none were given
        weights = np.ones(data.size)

    # Sort the data and generate a cumulative distribution
    order = np.argsort(reference)
    data_sorted = data[order]
    weights_sorted = weights[order]
    cum_dist = np.cumsum(weights_sorted)/np.sum(weights_sorted)

    # For each set of cutoffs, calculate the average of data that falls within
    # the cutoff percentiles of reference
    slice_avg = []
    for j in range(len(cutoffs)):
        bot = np.searchsorted(cum_dist,cutoffs[j][0])
        top = np.searchsorted(cum_dist,cutoffs[j][1])
        slice_avg.append(np.sum(data_sorted[bot:top]*weights_sorted[bot:top])/
                         np.sum(weights_sorted[bot:top]))
    return slice_avg

def kernelRegression(x,y,bot=None,top=None,N=500,h=None):
    '''
    Performs a non-parametric Nadaraya-Watson 1D kernel regression on given data
    with optionally specified range, number of points, and kernel bandwidth.

    Parameters
    ----------
    x : np.array
        The independent variable in the kernel regression.
    y : np.array
        The dependent variable in the kernel regression.
    bot : float
        Minimum value of interest in the regression; defaults to min(x).
    top : float
        Maximum value of interest in the regression; defaults to max(y).
    N : int
        Number of points to compute.
    h : float
        The bandwidth of the (Epanechnikov) kernel. To-do: GENERALIZE.

    Returns
    -------
    regression : LinearInterp
        A piecewise locally linear kernel regression: y = f(x).
    '''
    # Fix omitted inputs
    if bot is None:
        bot = np.min(x)
    if top is None:
        top = np.max(x)
    if h is None:
        h = 2.0*(top - bot)/float(N) # This is an arbitrary default

    # Construct a local linear approximation
    x_vec = np.linspace(bot,top,num=N)
    y_vec = np.zeros_like(x_vec) + np.nan
    for j in range(N):
        x_here = x_vec[j]
        weights = epanechnikovKernel(x,x_here,h)
        y_vec[j] = np.dot(weights,y)/np.sum(weights)
    regression = interp1d(x_vec,y_vec,bounds_error=False,assume_sorted=True)
    return regression

def epanechnikovKernel(x,ref_x,h=1.0):
    '''
    The Epanechnikov kernel.

    Parameters
    ----------
    x : np.array
        Values at which to evaluate the kernel
    x_ref : float
        The reference point
    h : float
        Kernel bandwidth

    Returns
    -------
    out : np.array
        Kernel values at each value of x
    '''
    u          = (x-ref_x)/h   # Normalize distance by bandwidth
    these      = np.abs(u) <= 1.0 # Kernel = 0 outside [-1,1]
    out        = np.zeros_like(x)   # Initialize kernel output
    out[these] = 0.75*(1.0-u[these]**2.0) # Evaluate kernel
    return out


# ==============================================================================
# ============== Some basic plotting tools  ====================================
# ==============================================================================

def plotFuncs(functions,bottom,top,N=1000,legend_kwds = None):
    '''
    Plots 1D function(s) over a given range.

    Parameters
    ----------
    functions : [function] or function
        A single function, or a list of functions, to be plotted.
    bottom : float
        The lower limit of the domain to be plotted.
    top : float
        The upper limit of the domain to be plotted.
    N : int
        Number of points in the domain to evaluate.
    legend_kwds: None, or dictionary
        If not None, the keyword dictionary to pass to plt.legend

    Returns
    -------
    none
    '''
    import matplotlib.pyplot as plt
    if type(functions)==list:
        function_list = functions
    else:
        function_list = [functions]

    for function in function_list:
        x = np.linspace(bottom,top,N,endpoint=True)
        y = function(x)
        plt.plot(x,y)
    plt.xlim([bottom, top])
    if legend_kwds is not None:
        plt.legend(**legend_kwds)
    plt.show()

def plotFuncsDer(functions,bottom,top,N=1000,legend_kwds = None):
    '''
    Plots the first derivative of 1D function(s) over a given range.

    Parameters
    ----------
    function : function
        A function or list of functions, the derivatives of which are to be plotted.
    bottom : float
        The lower limit of the domain to be plotted.
    top : float
        The upper limit of the domain to be plotted.
    N : int
        Number of points in the domain to evaluate.
    legend_kwds: None, or dictionary
        If not None, the keyword dictionary to pass to plt.legend

    Returns
    -------
    none
    '''
    import matplotlib.pyplot as plt
    if type(functions)==list:
        function_list = functions
    else:
        function_list = [functions]

    step = (top-bottom)/N
    for function in function_list:
        x = np.arange(bottom,top,step)
        y = function.derivative(x)
        plt.plot(x,y)
    plt.xlim([bottom, top])
    if legend_kwds is not None:
        plt.legend(**legend_kwds)
    plt.show()


def determine_platform():
    """ Untility function to return the platform currenlty in use.

    Returns
    ---------
    pf: str
        'darwin' (MacOS), 'debian'(debian Linux) or 'win' (windows)
    """
    import platform
    pform = platform.platform().lower()
    if 'darwin' in pform:
        pf = 'darwin' # MacOS
    elif 'debian' in pform:
        pf = 'debian' # Probably cloud (MyBinder, CoLab, ...)
    elif 'ubuntu' in pform:
        pf = 'debian' # Probably cloud (MyBinder, CoLab, ...)
    elif 'win' in pform:
        pf = 'win'
    else:
        raise ValueError('Not able to find out the platform.')
    return pf

def test_latex_installation(pf):
    """ Test to check if latex is installed on the machine.

    Parameters
    -----------
    pf: str (platform)
        output of determine_platform()

    Returns
    --------
    bool: Boolean
        True if latex found, else installed in the case of debian
        otherwise ImportError raised to direct user to install latex manually
    """
    # Test whether latex is installed (some of the figures require it)
    from distutils.spawn import find_executable

    latexExists = False

    if find_executable('latex'):
        latexExists = True
        return True

    if not latexExists:
        print('Some of the figures below require a full installation of LaTeX')
        # If running on Mac or Win, user can be assumed to be able to install
        # any missing packages in response to error messages; but not on cloud
        # so load LaTeX by hand (painfully slowly)
        if 'debian' in pf: # CoLab and MyBinder are both ubuntu
            print('Installing LaTeX now; please wait 3-5 minutes')
            from IPython.utils import io
            
            with io.capture_output() as captured: # Hide hideously long output 
                os.system('apt-get update')
                os.system('apt-get install texlive texlive-latex-extra texlive-xetex dvipng')
                latexExists=True
            return True
        else:
            raise ImportError('Please install a full distribution of LaTeX on your computer then rerun. \n \
            A full distribution means textlive, texlive-latex-extras, texlive-xetex, dvipng, and ghostscript')

def in_ipynb():
    """ If the ipython process contains 'terminal' assume not in a notebook.

    Returns
    --------
    bool: Boolean
          True if called from a jupyter notebook, else False
    """
    try:
        if 'terminal' in str(type(get_ipython())):
            return False
        else:
            return True
    except NameError:
        return False


def setup_latex_env_notebook(pf, latexExists):
    """ This is needed for use of the latex_envs notebook extension
    which allows the use of environments in Markdown.

    Parameters
    -----------
    pf: str (platform)
        output of determine_platform()
    """
    import os
    from matplotlib import rc
    import matplotlib.pyplot as plt
    plt.rc('font', family='serif')
    plt.rc('text', usetex=latexExists)
    if latexExists:
        latex_preamble = r'\usepackage{amsmath}\usepackage{amsfonts}\usepackage[T1]{fontenc}'
        latexdefs_path = os.getcwd()+'/latexdefs.tex'
        if os.path.isfile(latexdefs_path):
            latex_preamble = latex_preamble+r'\input{'+latexdefs_path+r'}'
        else: # the required latex_envs package needs this file to exist even if it is empty
            from pathlib import Path
            Path(latexdefs_path).touch()
        plt.rcParams['text.latex.preamble'] = latex_preamble


def make_figs(figure_name, saveFigs, drawFigs, target_dir="Figures"):
    """ Utility function to save figure in multiple formats and display the image.

    Parameters
    ----------
    figure_name: str
                name of the figure
    saveFigs: bool
              True if the figure needs to be written to disk else False
    drawFigs: bool
              True if the figure should be displayed using plt.draw()
    target_dir: str, default = 'Figures/'
              Name of folder to save figures to in the current directory
            
    """
    import matplotlib.pyplot as plt
    if saveFigs:
        import os
        # Where to put any figures that the user wants to save
        my_file_path = os.getcwd() # Find pathname to this file:
        Figures_dir = os.path.join(my_file_path, "{}".format(figure_name)) # LaTeX document assumes figures will be here
        if not os.path.exists(Figures_dir): 
            os.makedirs(Figures_dir)         # If dir does not exist, create it
        # Save the figures in several formats
        print("Saving figure {} in {}".format(figure_name, target_dir))
        plt.savefig(os.path.join(target_dir, '{}.jpg'.format(figure_name))) # For web/html
        plt.savefig(os.path.join(target_dir, '{}.png'.format(figure_name))) # For web/html
        plt.savefig(os.path.join(target_dir, '{}.pdf'.format(figure_name))) # For LaTeX
        plt.savefig(os.path.join(target_dir, '{}.svg'.format(figure_name))) # For html5
    # Make sure it's possible to plot it by checking for GUI
    if drawFigs and find_gui():
        plt.ion()  # Counterintuitively, you want interactive mode on if you don't want to interact
        plt.draw() # Change to false if you want to pause after the figure
        plt.pause(2)

def find_gui():
    """ Quick fix to check if matplotlib is running in a GUI environment.

    Returns
    -------
    bool: Boolean
          True if it's a GUI environment, False if not.
    """
    try:
        import matplotlib.pyplot as plt
    except:
        return False
    if plt.get_backend() == 'Agg':
        return False
    return True
