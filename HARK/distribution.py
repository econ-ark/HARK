from HARK.utilities import memoize
import math
import numpy as np
from scipy.special import erf, erfc
import scipy.stats as stats
import warnings

### CONTINUOUS DISTRIBUTIONS

class Lognormal():
    """
    A Lognormal distribution
    """
    mu = None
    sigma = None

    def __init__(self, mu = 0.0, sigma = 1.0):
        '''
        Initialize the distribution.

        Parameters
        ----------
         mu : float or [float]
            One or more means.  Number of elements T in mu determines number
            of rows of output.
        sigma : float or [float]
            One or more standard deviations. Number of elements T in sigma
            determines number of rows of output.
        '''
        self.mu = mu
        self.sigma = sigma

    def draw(self, N, seed=0):
        '''
        Generate arrays of lognormal draws. The sigma input can be a number
        or list-like.  If a number, output is a length N array of draws from the
        lognormal distribution with standard deviation sigma. If a list, output is
        a length T list whose t-th entry is a length N array of draws from the
        lognormal with standard deviation sigma[t].

        Parameters
        ----------
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

        if isinstance(self.sigma,float): # Return a single array of length N
            if self.sigma == 0:
                draws = np.exp(self.mu)*np.ones(N)
            else:
                draws = RNG.lognormal(mean=self.mu,
                                      sigma=self.sigma,
                                      size=N)
        else: # Set up empty list to populate, then loop and populate list with draws
            draws=[]
            for j in range(len(self.sigma)):
                if self.sigma[j] == 0:
                    draws.append(np.exp(self.mu[j])*np.ones(N))
                else:
                    draws.append(RNG.lognormal(mean=self.mu[j],
                                               sigma=self.sigma[j],
                                               size=N))
        return draws

class Normal():
    """
    A Normal distribution.
    """
    mu = None
    sigma = None

    def __init__(self, mu = 0.0, sigma = 1.0):
        '''
        Initialize the distribution.

        Parameters
        ----------
         mu : float or [float]
            One or more means.  Number of elements T in mu determines number
            of rows of output.
        sigma : float or [float]
            One or more standard deviations. Number of elements T in sigma
            determines number of rows of output.
        '''
        self.mu = mu
        self.sigma = sigma

    def draw(self, N, seed=0):
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

        if isinstance(self.sigma,float): # Return a single array of length N
            draws = self.sigma*RNG.randn(N) + self.mu
        else: # Set up empty list to populate, then loop and populate list with draws
            draws=[]
            for t in range(len(sigma)):
                draws.append(self.sigma[t]*RNG.randn(N) + self.mu[t])
        return draws

    def approx(self, N):
        """
        Returns a discrete approximation of this distribution.
        """
        x, w = np.polynomial.hermite.hermgauss(N)
        # normalize w
        pmf = w*np.pi**-0.5
        # correct x
        X = math.sqrt(2.0)*self.sigma*x + self.mu
        return DiscreteDistribution(pmf, X)

class Weibull():
    '''
    A Weibull distribution
    '''

    scale = None
    shape = None

    def __init__(self, scale=1.0, shape=1.0):
        '''
        Initialize the distribution.

        Parameters
        ----------
        scale : float or [float]
            One or more scales.  Number of elements T in scale determines number of
        rows of output.
        shape : float or [float]
            One or more shape parameters. Number of elements T in scale
            determines number of rows of output.
        '''
        self.scale = scale
        self.shape = shape

    def draw(self, N, seed=0):
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

        if self.scale == 1:
            scale = float(self.scale)
        if isinstance(self.scale,float): # Return a single array of length N
            draws = self.scale*(-np.log(1.0-RNG.rand(N)))**(1.0/self.shape)
        else: # Set up empty list to populate, then loop and populate list with draws
            draws=[]
            for t in range(len(self.scale)):
                draws.append(self.scale[t]*(-np.log(1.0-RNG.rand(N)))**(1.0/self.shape[t]))
        return draws

class Uniform():
    """
    A Uniform distribution.
    """

    bot = None
    top = None

    def __init__(self, bot = 0.0, top = 1.0):
        '''
        Initialize the distribution.

        Parameters
        ----------
        bot : float or [float]
            One or more bottom values.  Number of elements T in mu determines number
            of rows of output.
        top : float or [float]
            One or more top values. Number of elements T in top determines number of
            rows of output.
        '''
        self.bot = bot
        self.top = top

    def draw(self, N, seed=0):
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

        if isinstance(self.bot,float) or isinstance(self.bot,int): # Return a single array of size N
            draws = self.bot + (self.top - self.bot)*RNG.rand(N)
        else: # Set up empty list to populate, then loop and populate list with draws
            draws=[]
            for t in range(len(bot)):
                draws.append(self.bot[t] + (self.top[t] - self.bot[t])*RNG.rand(N))
        return draws

    def approx(self, N):
        '''
        Makes a discrete approximation to this uniform distribution.

        Parameters
        ----------
        N : int
            The number of points in the discrete approximation

        Returns
        -------
        d : DiscreteDistribution
            Probability associated with each point in array of discrete
            points for discrete probability mass function.
        '''
        pmf = np.ones(N)/float(N)
        center = (self.top+self.bot)/2.0
        width = (self.top-self.bot)/2.0
        X = center + width*np.linspace(-(N-1.0)/2.0,(N-1.0)/2.0,N)/(N/2.0)
        return DiscreteDistribution(pmf,X)

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
    mu = -0.5*sigma**2

    return Lognormal(mu,sigma).draw(N,seed=seed)


### DISCRETE DISTRIBUTIONS

class Bernoulli():
    """
    A Bernoulli distribution.
    """

    p = None

    def __init__(self, p = 0.5):
        '''
        Initialize the distribution.

        Parameters
        ----------
        p : float or [float]
            Probability or probabilities of the event occurring (True).
        '''
        self.p = p

    def draw(self, N, seed = 0):
        '''
        Generates arrays of booleans drawn from a simple Bernoulli distribution.
        The input p can be a float or a list-like of floats; its length T determines
        the number of entries in the output.  The t-th entry of the output is an
        array of N booleans which are True with probability p[t] and False otherwise.

        Arguments
        ---------
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

        if isinstance(self.p,float):# Return a single array of size N
            draws = RNG.uniform(size=N) < self.p
        else: # Set up empty list to populate, then loop and populate list with draws:
            draws=[]
            for t in range(len(self.p)):
                draws.append(RNG.uniform(size=N) < self.p[t])
        return draws



class DiscreteDistribution():
    """
    A representation of a discrete probability distribution.

    """
    pmf = None
    X = None

    def __init__(self, pmf, X):
        '''
        Initialize a discrete distribution.

        Parameters
        ----------
        pmf : np.array
            An array of floats representing a probability mass function. 
        X : np.array or [np.array]
            Discrete point values for each probability mass.
            May be multivariate (list of arrays).
        
        '''
        self.pmf = pmf
        self.X = X

        # Very quick and incomplete parameter check:
        # TODO: Check that pmf adn X arrays have same length.

    def dim(self):
        if isinstance(self.X, list):
            return len(self.X)
        else:
            return 1

    def draw_events(self, n, seed=0):
        '''
        Draws N 'events' from the distribution PMF.
        These events are indices into X.
        '''
        # Set up the RNG
        RNG = np.random.RandomState(seed)
        # Generate a cumulative distribution
        base_draws = RNG.uniform(size=n)
        cum_dist = np.cumsum(self.pmf)

        # Convert the basic uniform draws into discrete draws
        indices = cum_dist.searchsorted(base_draws)

        return indices

    def drawDiscrete(self, N,X=None,exact_match=False,seed=0):
        '''
        Simulates N draws from a discrete distribution with probabilities P and outcomes X.

        Parameters
        ----------
        N : int
            Number of draws to simulate.
        X : None, int, or np.array
            If None, then use this distribution's X for point values.
            If an int, then the index of X for the point values.
            If an np.array, use the array for the point values.
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
        if X is None:
            X = self.X
        elif isinstance(X,int):
            X = self.X[Xdim]
        else:
            X = X
        
        # Set up the RNG
        RNG = np.random.RandomState(seed)

        if exact_match:
            events = np.arange(self.pmf.size) # just a list of integers
            cutoffs = np.round(np.cumsum(self.pmf)*N).astype(int) # cutoff points between discrete outcomes
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
            indices = self.draw_events(N, seed=seed)
            draws = np.asarray(X)[indices]

        return draws

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
    d : DiscreteDistribution
        Probability associated with each point in array of discrete
        points for discrete probability mass function.
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
    return DiscreteDistribution(pmf, X)

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
    d : DiscreteDistribution
        Probability associated with each point in array of discrete
        points for discrete probability mass function.
    '''
    mu_adj = - 0.5*sigma**2;
    dist = approxLognormal(N=N, mu=mu_adj, sigma=sigma, **kwargs)
    return dist


def approxLognormalGaussHermite(N, mu=0.0, sigma=1.0):
    d = Normal(mu, sigma).approx(N)
    return DiscreteDistribution(d.pmf, np.exp(d.X))


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
    d : DiscreteDistribution
        Probability associated with each point in array of discrete
        points for discrete probability mass function.
    '''
    P    = 1000
    vals = np.reshape(stats.beta.ppf(np.linspace(0.0,1.0,N*P),a,b),(N,P))
    X    = np.mean(vals,axis=1)
    pmf  = np.ones(N)/float(N)
    return DiscreteDistribution(pmf, X)


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
    d : DiscreteDistribution
        Probability associated with each point in array of discrete
        points for discrete probability mass function.
    '''
    X   = np.append(x,distribution.X*(1-p*x)/(1-p))
    pmf = np.append(p,distribution.pmf*(1-p))

    if sort:
        indices = np.argsort(X)
        X       = X[indices]
        pmf     = pmf[indices]

    return DiscreteDistribution(pmf,X)

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
    d : DiscreteDistribution
        Probability associated with each point in array of discrete
        points for discrete probability mass function.
    '''

    X   = np.append(x,distribution.X)
    pmf = np.append(p,distribution.pmf*(1-p))

    if sort:
        indices = np.argsort(X)
        X       = X[indices]
        pmf     = pmf[indices]

    return DiscreteDistribution(pmf,X)

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
    A DiscreteDistribution, consisting of:

    P_out: np.array
        Probability associated with each point in X_out.

    X_out: np.array (as many as in *distributions)
        Discrete points for the joint discrete probability mass function.
    '''
    # Get information on the distributions
    dist_lengths = ()
    dist_dims = ()
    for dist in distributions:
        dist_lengths += (len(dist.pmf),)
        dist_dims += (dist.dim(),)
    number_of_distributions = len(distributions)

    # Initialize lists we will use
    X_out  = []
    P_temp = []

    # Now loop through the distributions, tiling and flattening as necessary.
    for dd,dist in enumerate(distributions):

        # The shape we want before we tile
        dist_newshape = (1,) * dd + (len(dist.pmf),) + \
                        (1,) * (number_of_distributions - dd)

        # The tiling we want to do
        dist_tiles    = dist_lengths[:dd] + (1,) + dist_lengths[dd+1:]

        # Now we are ready to tile.
        # We don't use the np.meshgrid commands, because they do not
        # easily support non-symmetric grids.

        # First deal with probabilities
        Pmesh  = np.tile(dist.pmf.reshape(dist_newshape),dist_tiles) # Tiling
        flatP  = Pmesh.ravel() # Flatten the tiled arrays
        P_temp += [flatP,] #Add the flattened arrays to the output lists

        # Then loop through each value variable
        for n in range(dist_dims[dd]):
            if dist.dim() > 1:
                Xmesh = np.tile(dist.X[n].reshape(dist_newshape),
                                 dist_tiles)
            else:
                Xmesh = np.tile(dist.X.reshape(dist_newshape),
                                dist_tiles)
            flatX  = Xmesh.ravel()
            X_out  += [flatX,]

    # We're done getting the flattened X_out arrays we wanted.
    # However, we have a bunch of flattened P_temp arrays, and just want one
    # probability array. So get the probability array, P_out, here.
    P_out = np.prod(np.array(P_temp),axis=0)

    assert np.isclose(np.sum(P_out),1),'Probabilities do not sum to 1!'
    return DiscreteDistribution(P_out, X_out)
