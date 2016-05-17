'''
This module contains a number of utilities, including the code that implements
agent expectations, utility functions (and their derivatives), manipulation of
discrete approximations, and basic plotting tools.
'''

from __future__ import division     # Import Python 3.x division function
import functools
import re                           # Regular expression, for string cleaning
import warnings
import numpy as np                  # Python's numeric library, abbreviated "np"
import pylab as plt                 # Python's plotting library
import scipy.stats as stats         # Python's statistics library
from scipy.integrate import quad, fixed_quad    # quad integration
from scipy.interpolate import interp1d
from scipy.special import erf
from HARKinterpolation import LinearInterp

def _warning(
    message,
    category = UserWarning,
    filename = '',
    lineno = -1):
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
   cache = obj._cache = {}

   @functools.wraps(obj)
   def memoizer(*args, **kwargs):
       key = str(args) + str(kwargs)
       if key not in cache:
           cache[key] = obj(*args, **kwargs)
       return cache[key]
   return memoizer


# ==============================================================================
# ============== Define utility functions        ===============================
# ==============================================================================

def CRRAutility(c, gam):
    """
    Return constant relative risk aversion (CRRA) utility of consumption "c"
    given risk aversion parameter "gamma" ()

    Parameters
    ----------
    c: float
        Consumption value.
    gam: float
        Risk aversion

    Returns
    -------
    u: float
        Utility.

    Tests
    -----
    Test a value which should pass:
    >>> c, gamma = 1.0, 2.0    # Set two values at once with Python syntax
    >>> utility(c=c, gam=gamma)
    -1.0
    """
    if gam == 1:
        return np.log(c)
    else:
        return( c**(1.0 - gam) / (1.0 - gam) )

def CRRAutilityP(c, gam):
    return( c**-gam )

def CRRAutilityPP(c, gam):
    return( -gam*c**(-gam-1.0) )
    
def CRRAutilityPPP(c, gam):
    return( (gam+1.0)*gam*c**(-gam-2.0) )
    
def CRRAutilityPPPP(c, gam):
    return( -(gam+2.0)*(gam+1.0)*gam*c**(-gam-3.0) )

def CRRAutility_inv(u, gam):
    return( ((1.0-gam)*u)**(1/(1.0-gam)) )

def CRRAutilityP_inv(u, gam):
    return( u**(-1.0/gam) )
    
def CRRAutility_invP(u, gam):
    return( ((1.0-gam)*u)**(gam/(1.0-gam)) )
    
        
def CARAutility(c, alpha):
    """
    Return absolute relative risk aversion (ARRA) utility of consumption "c"
    given risk aversion parameter "alpha" ()

    Parameters
    ----------
    c: float
        Consumption value.
    alpha: float
        Risk aversion

    Returns
    -------
    u: float
        Utility.
    """
    return( 1 - np.exp(-alpha*c)/alpha )

def CARAutilityP(c, alpha):
    return( np.exp(-alpha*c) )

def CARAutilityPP(c, alpha):
    return( -alpha*np.exp(-alpha*c) )
    
def CARAutilityPPP(c, alpha):
    return( alpha**2.0*np.exp(-alpha*c) )

def CARAutility_inv(u, alpha):
    return( -1.0/alpha * np.log(alpha*(1-u)) )

def CARAutilityP_inv(u, alpha):
    return( -1.0/alpha*np.log(u) )
    
def CARAutility_invP(u, alpha):
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
      [Solution Methods for Microeconomic Dynamic Optimization Problems](http://www.econ2.jhu.edu/people/ccarroll/solvingmicrodsops/) toolkit.
    Latest update: 21 April 2016 by Matthew N. White
    '''
    # Find the CDF boundaries of each segment
    mu_adj         = mu - 0.5*sigma**2;
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
    temp_cutoffs   = list(stats.lognorm.ppf(CDF_vals[1:-1], s=sigma, loc=0, scale=np.exp(mu_adj)))
    cutoffs        = [0] + temp_cutoffs + [np.inf]
    CDF_vals       = np.array(CDF_vals)

    # Construct the discrete approximation by finding the average value within each segment
    K              = CDF_vals.size-1 # number of points in approximation
    pmf            = CDF_vals[1:(K+1)] - CDF_vals[0:K]
    X              = np.zeros(K)
    for i in range(K):
        zBot  = cutoffs[i]
        zTop = cutoffs[i+1]
        X[i] = (-0.5)*np.exp(mu_adj+(sigma**2)*0.5)*(erf((mu_adj+sigma**2-np.log(zTop))*((np.sqrt(2)*sigma)**(-1)))-erf((mu_adj+sigma**2-np.log(zBot))*((np.sqrt(2)*sigma)**(-1))))*(pmf[i]**(-1));           
    return [pmf, X]


@memoize
def approxMeanOneLognormal(N, sigma):
    '''
    Calculate a discrete approximation to a mean-1 lognormal distribution.
    Based on function calculateLognormalDiscreteApprox; see that function's
    documentation for further notes.

    Parameters
    ----------
    N: int
        Size of discrete space vector to be returned.
    sigma: float
        standard deviation associated with underlying normal probability distribution.

    Returns
    ----------
    X: np.ndarray
        Discrete points for discrete probability mass function.
    pmf: np.ndarray
        Probability associated with each point in X.

    Written by Nathan M. Palmer
    Based on Matab function "setup_shocks.m," from Chris Carroll's
      [Solution Methods for Microeconomic Dynamic Optimization Problems](http://www.econ2.jhu.edu/people/ccarroll/solvingmicrodsops/) toolkit.
    Latest update: 01 May 2015
    '''
    #mu = -0.5*(sigma**2)
    mu=0.0
    return approxLognormal(N=N, mu=mu, sigma=sigma)
    
        
def approxBeta(N,a=1.0,b=1.0):
    '''
    Calculate a discrete approximation to the beta distribution.
    This needs to be made a bit more sophisticated.
    
    Parameters
    ----------
    N : int
        Size of discrete space vector to be returned.
    a : float
        First shape parameter (sometimes called alpha).
    b : float
        Second shape parameter (sometimes called beta).

    Returns
    ----------
    X: np.ndarray
        Discrete points for discrete probability mass function.
    pmf: np.ndarray
        Probability associated with each point in X.
    '''
    P = 1000
    vals = np.reshape(stats.beta.ppf(np.linspace(0.0,1.0,N*P),a,b),(N,P))
    X = np.mean(vals,axis=1)
    pmf = np.ones(N)/float(N)
    return( [pmf, X] )
    
    
def approxUniform(beta,nabla,N):
    '''
    Makes a discrete approximation to a uniform distribution with center beta and
    width 2*nabla, with N points.
    '''
    return beta + nabla*np.linspace(-(N-1.0)/2.0,(N-1.0)/2.0,N)/(N/2.0)


def makeMarkovApproxToNormal(x_grid,mu,sigma,K=351,bound=3.5):
    '''
    Creates an approximation to a normal distribution with mean mu and standard
    deviation sigma, returning a stochastic vector called p_vec, corresponding
    to values in x_grid.  If a RV is distributed x~N(mu,sigma), then the expectation
    of a continuous function f() is E[f(x)] = numpy.dot(p_vec,f(x_grid)).
    
    Parameters:
    -------------
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
        
    Returns:
    -----------
    p_vec: numpy.array
        A stochastic vector with probability weights for each x in x_grid.
    '''
    x_n = x_grid.size
    lower_bound =  -bound
    upper_bound = bound
    raw_sample = np.linspace(lower_bound,upper_bound,K)
    f_weights = stats.norm.pdf(raw_sample)
    sample = mu + sigma*raw_sample
    w_vec = np.zeros(x_n)
    
    sample_pos = np.searchsorted(x_grid,sample)
    sample_pos[sample_pos < 1] = 1
    sample_pos[sample_pos > x_n-1] = x_n-1
    
    bot = x_grid[sample_pos-1]
    top = x_grid[sample_pos]
    alpha = (sample-bot)/(top-bot)
    
    for j in range(1,x_n):
        c = sample_pos == j
        w_vec[j-1] = w_vec[j-1] + np.dot(f_weights[c],1.0-alpha[c])
        w_vec[j] = w_vec[j] + np.dot(f_weights[c],alpha[c])
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
   
    Parameters:
    -----------
    distribution: [np.ndarray]
        Two element list containing a list of outcomes and a list of probabilities.
    x:
        The new value to be added to the distribution.
    p: float
        The probability of the discrete outcome x occuring.
 
    sort: bool
        Whether or not to sort X before returning it
 
    Returns:
    -----------
    A new distribution object
 
    Written by Matthew N. White
    Latest update: 08 December 2015 by David Low
    '''
   
    X = np.append(x,distribution[1]*(1-p*x)/(1-p))
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
    
    Parameters:
    -----------
    distribution: [np.ndarray]
        Two element list containing a list of outcomes and a list of probabilities.
    x:
        The new value to be added to the distribution.
    p: float
        The probability of the discrete outcome x occuring.

    Returns:
    -----------
    A new distribution object

    Written by Matthew N. White
    Latest update: 11 December 2015
    '''
    
    X = np.append(x,distribution[1])
    pmf = np.append(p,distribution[0]*(1-p))
    
    if sort:
        indices = np.argsort(X)
        X       = X[indices]
        pmf     = pmf[indices]

    return([pmf,X])
    


def combineIndepDstns(*distributions):
    '''
    Given n lists (or tuples) whose elements represent n independent, discrete
    probability spaces (points in the space, and the probabilties across these
    points), construct a joint pmf over all combinations of these independent
    points.

    We flatten because some functions handle 1D arrays better than
    multidimentional arrays. This is particularly true for some lambda-functions
    and interpolation functions applied over arrays, which we employ for
    forming conditional expectations over values in a future period.

    Return an exhaustive combination of all points in each discrete vector, 
    with associated probabilites.

    Parameters
    ----------
    distributions: arbitrary number of distributions (pmfs).  Each pmf is 
        a list or tuple.  For each pmf, the first vector
        is values and the second is probabilties.  For each pmf, this
        should be true: len(X_pmf[0]) = len(X_pmf[1])
 
    Returns
    ----------
    List of arrays, consisting of:    
    
    X_out: list of np.ndarrays.
        Discrete points for the joint discrete probability mass function.

    P_out: np.ndarray
        Probability associated with each point in X_out.

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
    """
    Set up an exponentially spaced grid.

    Directly transcribed from the original Matlab code; see notes at end of
    documentation.

    Parameters
    ----------
    ming:   float
        minimum value of the grid
    maxg:   float
        maximum value of the grid
    ng:     int
        the number of grid-points
    timestonest: int
        the number of times to nest the exponentiation

    Returns
    ----------
    points:     np.ndarray
        a grid for search

    Original Matab code can be found in Chris Carroll's
    [Solution Methods for Microeconomic Dynamic Optimization Problems](http://www.econ2.jhu.edu/people/ccarroll/solvingmicrodsops/) toolkit.
    Latest update: 01 May 2015
    """

    if timestonest > 0:
        Lming = ming
        Lmaxg = maxg
        for j in range(timestonest):
            Lming = np.log(Lming + 1)
            Lmaxg = np.log(Lmaxg + 1)
        #Lstep = (Lmaxg - Lming)/(ng - 1)
        #Lgrid = np.arange(Lming,Lmaxg+0.000001,Lstep)
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
    
    Parameters:
    -----------
    data : numpy.array
        An array of data with N rows of J floats
    weights : numpy.array
        A length N array of weights for the N rows of data.
        
    Returns:
    -----------
    weighted_sum : float
        The weighted sum of the data.
    '''    
    data_avg = np.mean(data,axis=1)
    weighted_sum = np.dot(data_avg,weights)
    return weighted_sum
    

def getPercentiles(data,weights=None,percentiles=[0.5],presorted=False):
    '''
    Calculates the requested percentiles of (weighted) data.  Median by default.
    
    Parameters:
    -----------
    data : numpy.array
        A 1D array of float data.
    weights : nd.array
        A weighting vector for the data.
    percentiles : [float]
        A list of percentiles to calculate for the data.  Each element should
        be in (0,1).
    presorted : boolean
        Indicator for whether data has already been sorted.
        
    Returns:
    ----------
    pctl_out : numpy.array
        The requested percentiles of the data.
    '''  
    if weights is None:
        weights = np.ones(data.size)/float(data.size)
    
    if presorted:
        data_sorted = data
        weights_sorted = weights
    else:
        order = np.argsort(data)
        data_sorted = data[order]
        weights_sorted = weights[order]
    cum_dist = np.cumsum(weights_sorted)/np.sum(weights_sorted)
    
    inv_CDF = interp1d(cum_dist,data_sorted,bounds_error=False,assume_sorted=True)
    pctl_out = inv_CDF(percentiles)
    return pctl_out
    

def getLorenzShares(data,weights=None,percentiles=[0.5],presorted=False):
    '''
    Calculates the Lorenz curve at the requested percentiles of (weighted) data.
    Median by default.
    
    Parameters:
    -----------
    data : numpy.array
        A 1D array of float data.
    weights : numpy.array
        A weighting vector for the data.
    percentiles : [float]
        A list of percentiles to calculate for the data.  Each element should
        be in (0,1).
    presorted : boolean
        Indicator for whether data has already been sorted.
        
    Returns:
    ----------
    lorenz_out : numpy.array
        The requested Lorenz curve points of the data.
    '''
    if weights is None:
        weights = np.ones(data.size)
    
    if presorted:
        data_sorted = data
        weights_sorted = weights
    else:
        order = np.argsort(data)
        data_sorted = data[order]
        weights_sorted = weights[order]
    cum_dist = np.cumsum(weights_sorted)/np.sum(weights_sorted)
    temp = data_sorted*weights_sorted
    cum_data = np.cumsum(temp)/sum(temp)
      
    lorenzFunc = interp1d(cum_dist,cum_data,bounds_error=False,assume_sorted=True)
    lorenz_out = lorenzFunc(percentiles)
    return lorenz_out
    

def calcSubpopAvg(data,reference,cutoffs,weights=None):
    '''
    Calculates the average of (weighted) data between cutoff percentiles of a
    reference variable.
    
    Parameters:
    -----------
    data : numpy.array
        A 1D array of float data.
    reference : numpy.array
        A 1D array of float data of the same length as data.
    cutoffs : (float,float)
        A double with the lower and upper percentile bounds (should be in [0,1])
    weights : numpy.array
        A weighting vector for the data.
        
    Returns:
    ----------
    slice_avg
        The (weighted) average of data that falls within the cutoff percentiles
        of reference.
    
    '''
    if weights is None:
        weights = np.ones(data.size)
    order = np.argsort(reference)
    data_sorted = data[order]
    weights_sorted = weights[order]
    cum_dist = np.cumsum(weights_sorted)/np.sum(weights_sorted)
    slice_avg = []
    for j in range(len(cutoffs)):
        bot = np.searchsorted(cum_dist,cutoffs[j][0])
        top = np.searchsorted(cum_dist,cutoffs[j][1])
        slice_avg.append(np.sum(data_sorted[bot:top]*weights_sorted[bot:top])/np.sum(weights_sorted[bot:top]))
    return slice_avg
    
    
    
def kernelRegression(x,y,bot=None,top=None,N=500,h=None):
    '''
    Performs a non-parametric Nadaraya-Watson 1D kernel regression on given data
    with optionally specified range, number of points, and kernel bandwidth.
    
    Parameters:
    ------------
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
        
    Returns:
    ----------
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
    #print(x_vec)
    #print(y_vec)
    regression = LinearInterp(x_vec,y_vec)
    return regression
    
    
    
    
def epanechnikovKernel(x,ref_x,h=1.0):    
    u = (x-ref_x)/h
    these = np.abs(u) <= 1.0
    out = np.zeros_like(x)
    out[these] = 0.75*(1.0-u[these]**2.0)
    return out

    


def getArgNames(function):
    '''
    Returns a list of strings naming all of the arguments for the passed function.
    '''
    argCount = function.__code__.co_argcount
    argNames = function.__code__.co_varnames[:argCount]
    return argNames
    
'''
A trivial function that takes a single array and returns an array of NaNs of the
same size.  A generic default function that does nothing.
'''
NullFunc = lambda x : np.zeros(x.size) + np.nan  


# ==============================================================================
# ============== Some basic plotting tools  ====================================
# ==============================================================================

def plotFunc(Function,bottom,top,N=1000):
    step = (top-bottom)/N
    x = np.arange(bottom,top,step)
    y = Function(x)
    plt.plot(x,y)
    plt.xlim([bottom, top])
    plt.show()


def plotFuncDer(Function,bottom,top,N=1000):
    step = (top-bottom)/N
    x = np.arange(bottom,top,step)
    y = Function.derivative(x)
    plt.plot(x,y)
    plt.xlim([bottom, top])
    plt.show()

def plotFuncs(FunctionList,bottom,top,N=1000):
    step = (top-bottom)/N
    for Function in FunctionList:
        x = np.arange(bottom,top,step)
        y = Function(x)
        plt.plot(x,y)
    plt.xlim([bottom, top])
    plt.show()

