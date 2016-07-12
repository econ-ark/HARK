'''
Functions for generating simulated data and shocks.
'''

from __future__ import division
import warnings                             # A library for runtime warnings
import numpy as np                          # Numerical Python

def drawMeanOneLognormal(N, sigma=1.0, seed=0):
    '''
    Generate arrays of mean one lognormal draws. The sigma input can be a number
    or list-like.  If a number, output is a length N array of draws from the
    lognormal distribution with standard deviation sigma. If a list, output is
    a length T list whose t-th entry is a length N array of draws from the
    lognormal with standard deviation sigma[t].

    Parameters
    ----------
    N : int
        Number of draws in each row.
    sigma : float or [float]
        One or more standard deviations. Number of elements T in sigma
        determines number of rows of output.
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
    
def drawNormal(N, mu=0.0, sigma=1.0, seed=0):
    '''
    Generate arrays of normal draws.  The mu and sigma inputs can be numbers or
    list-likes.  If a number, output is a length N array of draws from the normal
    distribution with mean mu and standard deviation sigma. If a list, output is
    a length T list whose t-th entry is a length N array with draws from the
    normal distribution with mean mu[t] and standard deviation sigma[t].

    Parameters
    ----------
    N : int
        Number of draws in each row.
    mu : float or [float]
        One or more means.  Number of elements T in mu determines number of rows
        of output.
    sigma : float or [float]
        One or more standard deviations. Number of elements T in sigma
        determines number of rows of output.
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
    
def drawWeibull(N, scale=1.0, shape=1.0,  seed=0):
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
    N : int
        Number of draws in each row.
    scale : float or [float]
        One or more scales.  Number of elements T in scale determines number of
        rows of output.
    shape : float or [float]
        One or more shape parameters. Number of elements T in scale
        determines number of rows of output.
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
        
def drawUniform(N, bot=0.0, top=1.0, seed=0):
    '''
    Generate arrays of uniform draws.  The bot and top inputs can be numbers or
    list-likes.  If a number, output is a length N array of draws from the
    uniform distribution on [bot,top]. If a list, output is a length T list
    whose t-th entry is a length N array with draws from the uniform distribution
    on [bot[t],top[t]].

    Parameters
    ----------
    N : int
        Number of draws in each row.
    bot : float or [float]
        One or more bottom values.  Number of elements T in mu determines number
        of rows of output.
    top : float or [float]
        One or more top values. Number of elements T in top determines number of
        rows of output.
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
    
def drawBernoulli(N,p=0.5,seed=0):
    '''
    Generates arrays of booleans drawn from a simple Bernoulli distribution.
    The input p can be a float or a list-like of floats; its length T determines
    the number of entries in the output.  The t-th entry of the output is an
    array of N booleans which are True with probability p[t] and False otherwise.
    
    Arguments
    ---------
    N : int
        Number of draws in each row.
    p : float or [float]
        Probability or probabilities of the event occurring (True).    
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
        
def drawDiscrete(N,P=[1.0],X=[0.0],exact_match=False,seed=0):
    '''
    Simulates N draws from a discrete distribution with probabilities P and outcomes X.
    
    Parameters
    ----------
    P : np.array
        A list of probabilities of outcomes.
    X : np.array
        A list of discrete outcomes.
    N : int
        Number of draws to simulate.
    exact_match : boolean
        Whether the draws should "exactly" match the discrete distribution (as
        closely as possible given finite draws).  When True, returned draws are
        a random permutation of the N-length list that best fits the discrete
        distribution.  When False (default), each draw is independent from the
        others and the result could deviate from the input.
    seed : int
        Seed for random number generator.
        
    Returns
    -------
    draws : np.array
        An array draws from the discrete distribution; each element is a value in X.
    '''   
    # Set up the RNG
    RNG = np.random.RandomState(seed)
    
    if exact_match:
        events = np.arange(P.size) # just a list of integers
        cutoffs = np.round(np.cumsum(P)*N) # cutoff points between discrete outcomes
        top = 0
        # Make a list of event indices that closely matches the discrete distribution
        event_list        = []
        for j in range(events.size):
            bot = top
            top = cutoffs[j]
            event_list += (top-bot)*[events[j]]
        # Randomly permute the event indices and store the corresponding results
        event_draws = RNG.permutation(event_list)
        draws = X[event_draws]
    else:
        # Generate a cumulative distribution
        base_draws = RNG.uniform(size=N)
        cum_dist = np.cumsum(P)
        
        # Convert the basic uniform draws into discrete draws
        indices = cum_dist.searchsorted(base_draws)
        draws = np.asarray(X)[indices]
    return draws
    
    
if __name__ == '__main__':       
    print("Sorry, HARKsimulation doesn't actually do anything on its own.")
    print("To see some examples of its functions in action, look at any")
    print("of the model modules in /ConsumptionSavingModel.  In the future, running")
    print("this module will show examples of each function in the module.")
    