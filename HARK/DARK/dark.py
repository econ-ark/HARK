import numpy
from collections import namedtuple

from HARK import Solution, AgentType

class RustSolution(Solution):
    '''
    A class representing the solution of a single period of a discrete choice
    problem in the style of Rust (1987,1988). The solution must include an
    integrated value function and can also include the choice probabilities.
    These are only constructed when using Newton's method to solve the model.
    '''
    # distance_criteria are set to both V and P here since P is often the
    # object of interest ultimately
    distance_criteria = ['V']

    def __init__(self, V, P):
        '''
        The constructor for a new RustSolution object.

        Parameters
        ----------
        V : integer
            The number of states in the model.
        P :
            Vector of transition probabilities
        utility :
            Cost function that calculates cost given milage and cost parameters
        Returns
        -------
        None
        '''

        self.V = V
        self.P = P

def rustUtility(m, d, RC, c):
    return RC*(d-1) + c*m*(2-d)

class RustAgent(AgentType):
    '''
    A class representing the types of agents in the style of Rust (1987,1988).
    Agents make a binary decision over capital (bus engines) replacement or
    maintenance. The firm pays a maintenance cost that depends on the milage since
    last replacement each period, as well as a replacement cost if the engine is
    replaced.
    '''

    def __init__(self,
                 # number of discrete states
                 Nm=175,
                 # Rust's preferred linear specification
                 costFunc=rustUtility,
                 # The estimated parameters for Nm = 175
                 Fp=numpy.array([0.2, 0.3, 0.4, 0.1]),
                 # Discount factor
                 DiscFac=0.9,
                 # Initial guess; is this unharky?
                 V0=None,
                 # Replacement cost (so negative)
                 RC=-6.0,
                 # Maintenance cost/miles driven (so negative)
                 c=-0.005,
                 sigma=1.0,
                 # Method, can be 'Newton' or 'VFI'
                 method='Newton',
                 tolerance = 1e-12,
                 cycles=0,
                 **kwds):
        '''
        Instantiate a new RustAgent object with model parameters and solution
        method.

        Parameters
        ----------
        Nm : integer
            number of discrete states in the milage process
        costFunc : function
            cost function as function of milage, replacement decision and
            parameters
        Fp : numpy.array
            transition probabilities used to construct transition matrices
        DiscFac : number
            discount factor used to discount future utility
        V0 : numpy.array or None
            initial integrated value function guess
        RC : number
            replacement cost (negative)
        c : number
            maintenance cost paramter
        sigma : number
            scale parameter for additive extreme value type I shock to utility
        cycles : integer

        tolerance : number

        method : string
            switch to toggle solution by value function iteration ('VFI') or
            a simple Newton's method ('Newton')

        Returns
        -------
        new instance of RustAgent
        '''
        # Initialize a basic AgentType
        AgentType.__init__(self, tolerance=tolerance, cycles=cycles, **kwds)

        self.c = c
        self.RC = RC
        self.time_inv = ['states', 'costFunc', 'DiscFac', 'F', 'Vs', 'method',
                         'sigma']
        self.time_vary = []

        self.method = method
        self.sigma = sigma

        # Choice specific value functions in rows
        self.Vs = numpy.zeros((2, Nm))

        # Cost function as state and decisions only
        self.costFunc = lambda m, d: costFunc(m, d, RC, c)
        # Discount factor
        self.DiscFac = DiscFac

        # Linearly spaced discrete states from 0 to 450 thousand miles
        self.states = numpy.linspace(0, 450, Nm)

        # Handle no initial guess for integrated value function
        if V0 is None:
            self.V0 = numpy.zeros(Nm)
        else:
            self.V0 = V0

        # Calculate transition matrices give...
        # ... that the agent keeps and maintains the bus or ...
        FKeep = numpy.zeros((Nm, Nm))
        for i in range(Nm-len(Fp)+1):
            FKeep[i][i:i+len(Fp)] = Fp
        for i in range(Nm-len(Fp)+1, Nm):
            FKeep[i][i:i+len(Fp)-1]=Fp[0:Nm-i]
            FKeep[i, -1] = FKeep[i, -1] + Fp[Nm-i:].sum()

        # ... the agents replaces the bus and resets the state
        FReplace = numpy.zeros((Nm, Nm))
        FReplace[:] = FKeep[0]

        # Collect in ndarray to avoid variable clutter
        self.F = numpy.array([FKeep, FReplace])

        # Update the initial guess
        self.preSolve = self.updateSolutionTerminal

        self.solveOnePeriod = solve_rust_period

    def updateSolutionTerminal(self):
        '''
        Update the terminal period solution.  This method should be run when a
        new AgentType is created or when CRRA changes.
        Parameters
        ----------
        none
        Returns
        -------
        none
        '''
        updateVs(self.Vs, self.states, self.costFunc,
                 self.DiscFac, self.F, self.V0)
        V, P = discreteEnvelope(self.Vs, self.sigma)
        self.solution_terminal = RustSolution(V, P)

def solve_rust_period(solution_next, states, costFunc, DiscFac, F, Vs, method, sigma):
    if method == 'VFI':
        V, P = vfi(solution_next.V, states, costFunc, DiscFac, F, Vs, sigma)
    elif method == 'Newton':
        V, P = newton(solution_next, states, costFunc, DiscFac, F, Vs, sigma)

    return RustSolution(V, P)


def vfi(V, states, costFunc, DiscFac, F, Vs, sigma):
    updateVs(Vs, states, costFunc, DiscFac, F, V)
    V, P = discreteEnvelope(Vs, sigma)
    return V, P

def newton(solution_next, states, costFunc, DiscFac, F, Vs, sigma):
    Vold = solution_next.V
    V, P = vfi(Vold, states, costFunc, DiscFac, F, Vs, sigma)

    # G = (I-B)(V) where B is the Bellman-operator
    Gderiv = numpy.eye(len(V))
    for i in range(Vs.shape[0]):
        Gderiv = Gderiv - (DiscFac*P[i])*F[i,:,:]

    V = Vold - numpy.linalg.solve(Gderiv, Vold-V)
    V, P = vfi(V, states, costFunc, DiscFac, F, Vs, sigma)
    return V, P

def updateVs(Vs, states, costFunc, DiscFac, F, V):
    for i in range(Vs.shape[0]):
        Vs[i] = costFunc(states, i+1) + DiscFac*numpy.dot(F[i,:,:], V)

def conditionalChoiceProbabilities(Vs, sigma):
    if sigma == 0.0:
        # Is there a way to fuse these?
        P = numpy.argmax(Vs, axis=0)
        return P

    maxV = Vs.max()
    P = numpy.divide(numpy.exp(Vs/sigma), numpy.sum(numpy.exp(Vs/sigma), axis=0))
    return P

def discreteEnvelope(Vs, sigma):
    '''
    Returns the final optimal value and policies given the choice specific value
    functions Vs. Policies are degenerate if sigma == 0.0.
    Parameters
    ----------
    Vs : [numpy.array]
        A numpy.array that holds choice specific values at common grid points.
    sigma : float
        A number that controls the variance of the taste shocks
    Returns
    -------
    V : [numpy.array]
        A numpy.array that holds the integrated value function.
    P : [numpy.array]
        A numpy.array that holds the discrete choice probabilities
    '''
    if sigma == 0.0:
        # Is there a way to fuse these?
        Pflat = numpy.argmax(Vs, axis=0)
        P = numpy.zeros(Vs.shape)
        for i in range(Vs.shape[0]):
            P[i][Pflat==i] = 1
        V = numpy.amax(Vs, axis=0)
        return V, P

    maxV = Vs.max()
    P = numpy.divide(numpy.exp((Vs-Vs[0])/sigma), numpy.sum(numpy.exp((Vs-Vs[0])/sigma), axis=0))
    # GUARD THIS USING numpy error level whatever
    sumexp = numpy.sum(numpy.exp((Vs-maxV)/sigma), axis=0)
    V = numpy.log(sumexp)
    V = maxV + sigma*V
    return V, P
