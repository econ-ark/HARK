import numpy
from collections import namedtuple

from HARK import Solution, AgentType

class RustSolution(Solution):
    '''
    A class representing the solution of a single period of a discrete choice
    problem. The solution must include an integrated value function and can
    also include the choice probabilities. These are only constructed when
    using Newton's method to solve the model.
    '''
    distance_criteria = ['vPfunc']

    def __init__(self, V, P = None):
        '''
        The constructor for a new RustSolution object.

        Parameters
        ----------
        Nm : integer
            The number of states in the model.
        Fp :
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

class DARK(AgentType):
    def __init__(self,
                 Nm=175,
                 costFunc=rustUtility,
                 Fp=numpy.array([0.2, 0.3, 0.4, 0.1]),
                 DiscFac=0.99,
                 V0=None,
                 RC=-10.0,
                 c=-0.0025,
                 **kwds):
        AgentType.__init__(self,
                           **kwds)

        self.time_inv = ['states', 'costFunc', 'DiscFac', 'F', 'Vs']
        self.time_vary = []

        self.Vs = numpy.zeros((2, Nm))
        self.costFunc = lambda m, d: costFunc(m, d, RC, c)
        self.c = c

        self.states = numpy.linspace(0,450, Nm)

        self.DiscFac = DiscFac

        if V0 is None:
            self.V0 = numpy.zeros(Nm)
        else:
            self.V0 = V0
        FKeep = numpy.zeros((Nm, Nm))
        for i in range(Nm-len(Fp)+1):
            FKeep[i][i:i+len(Fp)] = Fp

        for i in range(Nm-len(Fp)+1, Nm):
            FKeep[i][i:i+len(Fp)-1]=Fp[0:Nm-i]
            FKeep[i, -1] = FKeep[i, -1] + Fp[Nm-i:].sum()

        FReplace = numpy.zeros((Nm, Nm))
        FReplace[:] = FKeep[0]

        self.F = numpy.array([FKeep, FReplace])

        self.preSolve = self.updateSolutionTerminal

        self.solution_terminal = RustSolution(V0)

        self.solveOnePeriod = bellman

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
        self.solution_terminal.V = self.V0


def bellman(solution_next, states, costFunc, DiscFac, F, Vs):
    for i in range(Vs.shape[0]):
        Vs[i] = costFunc(states, i+1) + DiscFac*numpy.dot(F[i,:,:], solution_next.V)
    V, P = discreteEnvelope(Vs, 1.0)
    return RustSolution(V, P)

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
        P = numpy.argmax(Vs, axis=0)
        V = numpy.amax(Vs, axis=0)
        return V, P

    maxV = Vs.max()
    P = numpy.divide(numpy.exp(Vs/sigma), numpy.sum(numpy.exp(Vs/sigma), axis=0))
    # GUARD THIS USING numpy error level whatever
    sumexp = numpy.sum(numpy.exp((Vs-maxV)/sigma), axis=0)
    V = numpy.log(sumexp)
    V = maxV + sigma*V
    return V, P
