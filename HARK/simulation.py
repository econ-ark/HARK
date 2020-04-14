'''
Functions for generating simulated data and shocks.
'''

from __future__ import division
import warnings                             # A library for runtime warnings
import numpy as np                          # Numerical Python

class Lognormal():
    mu = None
    sigma = None

    def __init__(self, mu = 0.0, sigma = 1.0):
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
        mu : float or [float]
            One or more means.  Number of elements T in mu determines number
            of rows of output.
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
    mu = None
    sigma = None

    def __init__(self, mu = 0.0, sigma = 1.0):
        self.mu = 0.0
        self.sigma = 1.0

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

        if isinstance(self.sigma,float): # Return a single array of length N
            draws = self.sigma*RNG.randn(N) + self.mu
        else: # Set up empty list to populate, then loop and populate list with draws
            draws=[]
            for t in range(len(sigma)):
                draws.append(self.sigma[t]*RNG.randn(N) + self.mu[t])
        return draws

class Weibull():

    scale = None
    shape = None

    def __init__(self, scale=1.0, shape=1.0):
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

    bot = None
    top = None

    def __init__(self, bot = 0.0, top = 1.0):
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

        if isinstance(self.bot,float) or isinstance(self.bot,int): # Return a single array of size N
            draws = self.bot + (self.top - self.bot)*RNG.rand(N)
        else: # Set up empty list to populate, then loop and populate list with draws
            draws=[]
            for t in range(len(bot)):
                draws.append(self.bot[t] + (self.top[t] - self.bot[t])*RNG.rand(N))
        return draws

class Bernoulli():

    p = None

    def __init__(self, p = 0.5):
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

        if isinstance(self.p,float):# Return a single array of size N
            draws = RNG.uniform(size=N) < self.p
        else: # Set up empty list to populate, then loop and populate list with draws:
            draws=[]
            for t in range(len(self.p)):
                draws.append(RNG.uniform(size=N) < self.p[t])
        return draws


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
