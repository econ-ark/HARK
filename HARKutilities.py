'''
General purpose  / miscellaneous functions.  Includes functions to approximate
continuous distributions with discrete ones, utility functions (and their
derivatives), manipulation of discrete distributions, and basic plotting tools.
'''

from __future__ import division     # Import Python 3.x division function
import functools
import re                           # Regular expression, for string cleaning
import warnings
import numpy as np                  # Python's numeric library, abbreviated "np"
import pylab as plt                 # Python's plotting library
import scipy.stats as stats         # Python's statistics library
from scipy.interpolate import interp1d
from scipy.special import erf

def _warning(message,category = UserWarning,filename = '',lineno = -1):
    '''
    A "monkeypatch" to warnings, to print pretty-looking warnings. The
    default behavior of the "warnings" module is to print some extra, unusual-
    looking things when the user calls a warning. A common "fix" for this is
    to "monkeypatch" the warnings module. See:
    http://stackoverflow.com/questions/2187269/python-print-only-the-message-on-warnings
    I implement this fix directly below, for all simulation and solution utilities.
    '''
    print(message)
warnings.showwarning = _warning

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


class NullFunc():
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
    
def CRRAutilityP_invP(u, gam):
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
    return( (-1.0/gam)*u**(-1.0/gam-1.0) )
    
        
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
    normal distribution N(exp(mu),sigma).  Makes an equiprobable distribution by
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
    Latest update: 21 April 2016 by Matthew N. White
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
        temp_cutoffs   = list(stats.lognorm.ppf(CDF_vals[1:-1], s=sigma, loc=0, 
                                                scale=np.exp(mu)))
        cutoffs        = [0] + temp_cutoffs + [np.inf]
        CDF_vals       = np.array(CDF_vals)
    
        # Construct the discrete approximation by finding the average value within each segment
        K              = CDF_vals.size-1 # number of points in approximation
        pmf            = CDF_vals[1:(K+1)] - CDF_vals[0:K]
        X              = np.zeros(K)
        for i in range(K):
            zBot  = cutoffs[i]
            zTop = cutoffs[i+1]
            X[i] = (-0.5)*np.exp(mu+(sigma**2)*0.5)*(erf((mu+sigma**2-np.log(zTop))*(
                   (np.sqrt(2)*sigma)**(-1)))-erf((mu+sigma**2-np.log(zBot))*((np.sqrt(2)*sigma)
                   **(-1))))*(pmf[i]**(-1))           
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
    
    # Loop through each x_grid point and add up the probability that each nearby
    # draw contributes to it (accounting for distance)
    for j in range(1,x_n):
        c = sample_pos == j
        w_vec[j-1] = w_vec[j-1] + np.dot(f_weights[c],1.0-alpha[c])
        w_vec[j] = w_vec[j] + np.dot(f_weights[c],alpha[c])
        
    # Reweight the probabilities so they sum to 1, and return
    W = np.sum(w_vec)
    p_vec = w_vec/W
    return p_vec


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
    all combinations of these independent points.

    Parameters
    ----------
    distributions : [np.array]
        Arbitrary number of distributions (pmfs).  Each pmf is a list or tuple.
        For each pmf, the first vector is probabilities and the second is values.
        For each pmf, this should be true: len(X_pmf[0]) = len(X_pmf[1])
 
    Returns
    -------
    List of arrays, consisting of:
    
    P_out: np.array
        Probability associated with each point in X_out.
    
    X_out: np.array (as many as in *distributions)
        Discrete points for the joint discrete probability mass function.

    Written by Nathan Palmer 
    Latest update: 31 August 2015 by David Low
    '''
    # Very quick and incomplete parameter check:
    for dist in distributions:
        assert len(dist[0]) == len(dist[-1]), "len(dist[0]) != len(dist[-1])"

    # Get information on the distributions
    dist_lengths = ()
    for dist in distributions:
        dist_lengths += (len(dist[0]),)
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
        Xmesh  = np.tile(dist[1].reshape(dist_newshape),dist_tiles)
        Pmesh  = np.tile(dist[0].reshape(dist_newshape),dist_tiles)

        # Now flatten the tiled arrays.
        flatX  = Xmesh.ravel()
        flatP  = Pmesh.ravel()
        
        # Add the flattened arrays to the output lists.
        X_out  += [flatX,]
        P_temp += [flatP,]

    # We're done getting the flattened X_out arrays we wanted.
    # However, we have a bunch of flattened P_temp arrays, and just want one 
    # probability array. So get the probability array, P_out, here.
    P_out = np.ones_like(X_out[0])
    for pp in P_temp:
        P_out *= pp

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

def plotFuncs(functions,bottom,top,N=1000):
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
    Returns
    -------
    none
    '''
    if type(functions)==list:
        function_list = functions
    else:
        function_list = [functions]
       
    step = (top-bottom)/N
    for function in function_list:
        x = np.arange(bottom,top,step)
        y = function(x)
        plt.plot(x,y)
    plt.xlim([bottom, top])
    plt.show()



def plotFuncsDer(functions,bottom,top,N=1000):
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
        
    Returns
    -------
    none
    '''
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
    plt.show()


if __name__ == '__main__':       
    print("Sorry, HARKutilities doesn't actually do anything on its own.")
    print("To see some examples of its functions in action, look at any")
    print("of the model modules in /ConsumptionSavingModel.  As these functions")
    print("are the basic building blocks of HARK, you'll find them used")
    print("everywhere! In the future, this module will show examples of each")
    print("function in the module.")