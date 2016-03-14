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



# ================================================================================
# = Functions for generating discrete approximations to continuous distributions =
# ================================================================================
def calculateLognormalDiscreteApprox(N, mu, sigma):
    '''
    Calculate a discrete approximation to the lognormal distribution, as
    described in Carroll (2012 a,b). N is the number of non-overlapping,
    equiprobably partitions of the support to create. Mu and sigma are the
    mean and standard deviation of the underlying normal distribution.

    See the full documentation for indications as to practical limitations
    on the numerical approximation accuracy.

    Parameters
    ----------
    N: float
        Size of discrete space vector to be returned.
    mu: float
        mu associated with underlying normal probability distribution.
    sigma: float
        standard deviation associated with underlying normal probability distribution.

    Returns
    -------
    X: np.ndarray
        Discrete points for discrete probability mass function.
    pmf: np.ndarray
        Probability associated with each point in X.

    References
    ----------
    Carroll, C. D. (2012a). Theoretical Foundations of Buffer Stock Saving.
    Manuscript, Department of Economics, Johns Hopkins University, 2012.
    http://www.econ2.jhu.edu/people/ccarroll/papers/BufferStockTheory/

    Carroll, C. D. (2012b). Solution methods for microeconomic dynamic stochastic
    optimization problems. Manuscript, Department of Economics, Johns Hopkins
    University, 2012. http://www.econ.jhu.edu/people/ccarroll/solvingmicrodsops

    Written by Nathan M. Palmer
    Based on Matab function "setup_shocks.m," from Chris Carroll (2012),
      [Solution Methods for Microeconomic Dynamic Optimization Problems](http://www.econ2.jhu.edu/people/ccarroll/solvingmicrodsops/) toolkit.
    Latest update: 31 Jan 2016
    --
    '''
    if sigma > 0.0:
        # Note: "return" convention will be: probs always come first, then values.
        distrib = stats.lognorm(sigma, 0, np.exp(mu))  # rv = generic_distrib(<shape(s)>, loc=0, scale=1)
    
        # ------ Set up discrete approx cutoffs ------
        probs_cutoffs = np.arange(N+1.0)/float(N)   # Includes 0 and 1
        state_cutoffs = distrib.ppf(probs_cutoffs)  # Inverse cdf applied to get the
            # "cuttoff" values in the support. Note that final value will be inf.
    
        # Set pmf:
        pmf = np.repeat(1.0/N, N)
        pmf_alt = []
    
        # Find the E[X|bin] values:
        F = lambda x: x*distrib.pdf(x)
        Ebins = []
    
        epsabs = 1e-10  # These are used to set the absolute and relative error
        epsrel = 1e-10
    
        # Integrate: get conditional means over partitions.
        for i, (x0, x1) in enumerate(zip(state_cutoffs[:-1], state_cutoffs[1:])):
    
            OUTPUT = quad(F, x0, x1, epsabs=epsabs, epsrel=epsrel, limit=100, full_output=1)
    
            # Check for errors and unpack the appropriate values:
            if len(OUTPUT) == 3:
                cond_mean = OUTPUT[0]
                abserr = OUTPUT[1]
                infodict = OUTPUT[2]
    
                # Check to see if there appears to be an error:
                if abserr > cond_mean:
                    warnings.warn("WARNING: abserr > cond_mean. You may likely need to set abserr=0, although this only solves the problem which arises when the true integral is very close to zero.")
            else:
                # Assume all other values indicate error. Raise exception.
                cond_mean = OUTPUT[0]
                abserr = OUTPUT[1]
                infodict = OUTPUT[2]
                errormessage = OUTPUT[3]
    
                # Do string cleaning to address issues which emerged during doctest.
                errormessage = re.sub('\s+', ' ', errormessage)
    
                # Now use the error message in raising a user exception:
                raise Exception(errormessage)
    
            # Following two lines purely for error-testing:
            pmf_alt.append(distrib.cdf(x1) - distrib.cdf(x0))
            assert np.isclose(pmf[i], pmf_alt[i]), "In discrete approximation: np.isclose(pmf[i], pmf_alt[i]) is not close!"
    
            # Save the conditional distribution:
            Ebins.append(cond_mean)
    
        X = np.array(Ebins) / pmf
        
    else: # No work to be done if there is no variance
        pmf = np.array([1.0])
        X = np.array([np.exp(mu)])

    return( [pmf, X] )


@memoize
def calculateMeanOneLognormalDiscreteApprox(N, sigma):
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
    mu = -0.5*(sigma**2)
    return calculateLognormalDiscreteApprox(N=N, mu=mu, sigma=sigma)
    
    
    
def calculateBetaDiscreteApprox(N,a=1.0,b=1.0):
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
    


def createFlatStateSpaceFromIndepDiscreteProbs(*distributions):
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


def makeUniformDiscreteDistribution(beta,nabla,N):
    '''
    Makes a discrete approximation to a uniform distribution with center beta and
    width 2*nabla, with N points.
    '''
    return beta + nabla*np.linspace(-(N-1.0)/2.0,(N-1.0)/2.0,N)/(N/2.0)

# ==============================================================================
# ============== Functions for generating state space grids  ===================
# ==============================================================================
def setupGridsExpMult(ming, maxg, ng, timestonest=20):
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

def calculateMeanSacrificeValue(vinv,vhat,xvals,xprobs):
    '''
    Given the inverse of the optimal value function, a value function for an
    approximate policy, and a discrete distribution, return the expected
    sacrifice value.

    Parameters
    ----------
    vinv: univariate real-valued function: vinv:R -> R
        Inverse of the optimal value function.
    vhat: univariate real-valued function: vinv:R -> R
        Value function for approximate policy for which we will find the
        expected sacrifice value.
    xvals: np.ndarray
        Discrete points for discrete probability mass function.
    xprobs: np.ndarray
        Probability associated with each point in xvals.

    Returns
    ----------
    eps: float
        Mean sacrifice value.

    Written by Nathan M. Palmer
    Latest update: 20 March 2015
    '''
    eps = lambda x: x - vinv(vhat(x))
    return np.dot(eps(xvals), xprobs)


def weightedAverageSimData(data,weights):
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
    

def extractPercentiles(data,weights=None,percentiles=[0.5],presorted=False):
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
    

def getLorenzPercentiles(data,weights=None,percentiles=[0.5],presorted=False):
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
    

def avgDataSlice(data,reference,cutoffs,weights=None):
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

