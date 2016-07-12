'''
A collection of functions used for generating simulated data and shocks.
'''

from __future__ import division
import warnings                             # A library for runtime warnings
import numpy as np                          # Numerical Python

def drawMeanOneLognormal(sigma, N, seed=0):
    '''
    Generate arrays of mean one lognormal draws. The sigma input can be a number
    or list-like.  If a number, output is a length N array of draws from the
    lognormal distribution with standard deviation sigma. If a list, output is
    a length T list whose t-th entry is a length N array of draws from the
    lognormal with standard deviation sigma[t].

    Parameters
    ----------
    sigma : float or [float]
        One or more standard deviations. Number of elements T in sigma
        determines number of rows of output.
    N : int
        Number of draws in each row.
    seed : int
        Seed for random number generator.

    Returns:
    ------------
    draws : np.array or [np.array]
        T-length list of arrays of mean one lognormal draws each of size N, or
        a single array of size N (if sigma is a scalar).
    '''
    # Set up the RNG
    RNG = np.random.RandomState(seed)
    
    if type(sigma) == float: # Return a single array of length N
        mu = -0.5*sigma**2
        draws = RNG.lognormal(mean=mu, sigma=sigma, size=N)
    else: # Set up empty list to populate, then loop and populate list with draws
        draws=[]
        for sig in sigma:            
            mu = -0.5*(sig**2)
            draws.append(RNG.lognormal(mean=mu, sigma=sig, size=N))            
    return draws
    
def drawNormal(mu, sigma, N, seed=0):
    '''
    Generate arrays of normal draws.  The mu and sigma inputs can be numbers or
    list-likes.  If a number, output is a length N array of draws from the normal
    distribution with mean mu and standard deviation sigma. If a list, output is
    a length T list whose t-th entry is a length N array with draws from the
    normal distribution with mean mu[t] and standard deviation sigma[t].

    Parameters
    ----------
    mu : float or [float]
        One or more means.  Number of elements T in mu determines number of rows
        of output.
    sigma : float or [float]
        One or more standard deviations. Number of elements T in sigma
        determines number of rows of output.
    N : int
        Number of draws in each row.
    seed : int
        Seed for random number generator.

    Returns
    -------
    draws : np.array or [np.array]
        T-length list of arrays of normal draws each of size N, or a single array
        of size N (if sigma is a scalar).
    '''
    # Set up the RNG
    RNG = np.random.RandomState(seed)
   
    if type(sigma) == float: # Return a single array of length N
        draws = sigma*RNG.randn(N) + mu
    else: # Set up empty list to populate, then loop and populate list with draws
        draws=[]
        for t in range(len(sigma)):
            draws.append(sigma[t]*RNG.randn(N) + mu[t])            
    return draws
    
def drawWeibull(scale, shape, N, seed=0):
    '''
    Generate arrays of Weibull draws.  The scale and shape inputs can be 
    numbers or list-likes.  If a number, output is a length N array of draws from
    the Weibull distribution with the given scale and shape. If a list, output
    is a length T list whose t-th entry is a length N array with draws from the
    Weibull distribution with scale scale[t] and shape shape[t].
    
    Note: When shape=1, the Weibull distribution is simply the exponential dist.
    
    Mean: scale*Gamma(1 + 1/shape)

    Parameters
    ----------
    scale : float or [float]
        One or more scales.  Number of elements T in scale determines number of
        rows of output.
    shape : float or [float]
        One or more shape parameters. Number of elements T in scale
        determines number of rows of output.
    N : int
        Number of draws in each row.
    seed : int
        Seed for random number generator.

    Returns:
    ------------
    draws : np.array or [np.array]
        T-length list of arrays of Weibull draws each of size N, or a single
        array of size N (if sigma is a scalar).
    '''
    # Set up the RNG
    RNG = np.random.RandomState(seed)
    
    if scale == 1:
        scale = float(scale)
    if type(scale) == float: # Return a single array of length N
        draws = scale*(-np.log(1.0-RNG.rand(N)))**(1.0/shape)
    else: # Set up empty list to populate, then loop and populate list with draws
        draws=[]
        for t in range(len(scale)):
            draws.append(scale[t]*(-np.log(1.0-RNG.rand(N)))**(1.0/shape[t]))            
    return draws   
        
def drawUniform(bot, top, N, seed=0):
    '''
    Generate arrays of uniform draws.  The bot and top inputs can be numbers or
    list-likes.  If a number, output is a length N array of draws from the
    uniform distribution on [bot,top]. If a list, output is a length T list
    whose t-th entry is a length N array with draws from the uniform distribution
    on [bot[t],top[t]].

    Parameters
    ----------
    bot : float or [float]
        One or more bottom values.  Number of elements T in mu determines number
        of rows of output.
    top : float or [float]
        One or more top values. Number of elements T in top determines number of
        rows of output.
    N : int
        Number of draws in each row.
    seed : int
        Seed for random number generator.

    Returns
    -------
    draws : np.array or [np.array]
        T-length list of arrays of uniform draws each of size N, or a single
        array of size N (if sigma is a scalar).
    '''
    # Set up the RNG
    RNG = np.random.RandomState(seed)
   
    if type(bot) == float or type(bot) == int: # Return a single array of size N
        draws = bot + (top - bot)*RNG.rand(N)
    else: # Set up empty list to populate, then loop and populate list with draws
        draws=[]
        for t in range(len(bot)):
            draws.append(bot[t] + (top[t] - bot[t])*RNG.rand(N))            
    return draws
    
def drawBernoulli(p,N,seed=0):
    '''
    Generates arrays of booleans drawn from a simple Bernoulli distribution.
    The input p can be a float or a list-like of floats; its length T determines
    the number of entries in the output.  The t-th entry of the output is an
    array of N booleans which are True with probability p[t] and False otherwise.
    
    Arguments
    ---------
    p : float or [float]
        Probability or probabilities of the event occurring (True).
    N : int
        Number of draws in each row.
    seed : int
        Seed for random number generator.

    Returns
    -------
    draws : np.array or [np.array]
        T-length list of arrays of Bernoulli draws each of size N, or a single
        array of size N (if sigma is a scalar).
    '''   
    # Set up the RNG
    RNG = np.random.RandomState(seed)

    if type(p) == float:# Return a single array of size N
        draws = RNG.uniform(size=N) < p
    else: # Set up empty list to populate, then loop and populate list with draws:
        draws=[]
        for t in range(len(p)):
            draws.append(RNG.uniform(size=N) < p[t])
    return draws
        
def drawDiscrete(P,X,N,seed=0):
    '''
    Simulates N draws from a discrete distribution with probabilities P and outcomes X.
    
    Parameters
    ----------
    P : [float]
        A list of probabilities of outcomes.
    X : [float]
        A list of discrete outcomes.
    N : int
        Number of draws to simulate.
        
    Returns
    -------
    draws : np.array
        An array draws from the discrete distribution; each element is a value in X.
    '''
    
    # Set up the RNG
    RNG = np.random.RandomState(seed)
    base_draws = RNG.uniform(size=N)
    
    # Generate a cumulative distribution
    cum_dist = np.cumsum(P)
    
    # Convert the basic uniform draws into discrete draws
    indices = cum_dist.searchsorted(base_draws)
    draws = np.asarray(X)[indices]
    return draws
    