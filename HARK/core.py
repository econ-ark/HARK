'''
High-level functions and classes for solving a wide variety of economic models.
The "core" of HARK is a framework for "microeconomic" and "macroeconomic"
models.  A micro model concerns the dynamic optimization problem for some type
of agents, where agents take the inputs to their problem as exogenous.  A macro
model adds an additional layer, endogenizing some of the inputs to the micro
problem by finding a general equilibrium dynamic rule.
'''
from __future__ import print_function, division
from __future__ import absolute_import

from builtins import str
from builtins import range
from builtins import object
import sys
import os
from distutils.dir_util import copy_tree
from .utilities import getArgNames, NullFunc
from copy import copy, deepcopy
import numpy as np
from time import time
from .parallel import multiThreadCommands, multiThreadCommandsFake


def distanceMetric(thing_A, thing_B):
    '''
    A "universal distance" metric that can be used as a default in many settings.

    Parameters
    ----------
    thing_A : object
        A generic object.
    thing_B : object
        Another generic object.

    Returns:
    ------------
    distance : float
        The "distance" between thing_A and thing_B.
    '''
    # Get the types of the two inputs
    typeA = type(thing_A)
    typeB = type(thing_B)

    if typeA is list and typeB is list:
        lenA = len(thing_A)  # If both inputs are lists, then the distance between
        lenB = len(thing_B)  # them is the maximum distance between corresponding
        if lenA == lenB:  # elements in the lists.  If they differ in length,
            distance_temp = []  # the distance is the difference in lengths.
            for n in range(lenA):
                distance_temp.append(distanceMetric(thing_A[n], thing_B[n]))
            distance = max(distance_temp)
        else:
            distance = float(abs(lenA - lenB))
    # If both inputs are numbers, return their difference
    elif isinstance(thing_A, (int, float)) and isinstance(thing_B, (int, float)):
        distance = float(abs(thing_A - thing_B))
    # If both inputs are array-like, return the maximum absolute difference b/w
    # corresponding elements (if same shape); return largest difference in dimensions
    # if shapes do not align.
    elif hasattr(thing_A, 'shape') and hasattr(thing_B, 'shape'):
        if thing_A.shape == thing_B.shape:
            distance = np.max(abs(thing_A - thing_B))
        else:
            # Flatten arrays so they have the same dimensions
            distance = np.max(abs(thing_A.flatten().shape[0] - thing_B.flatten().shape[0]))
    # If none of the above cases, but the objects are of the same class, call
    # the distance method of one on the other
    elif thing_A.__class__.__name__ == thing_B.__class__.__name__:
        if thing_A.__class__.__name__ == 'function':
            distance = 0.0
        else:
            distance = thing_A.distance(thing_B)
    else:  # Failsafe: the inputs are very far apart
        distance = 1000.0
    return distance


class HARKobject(object):
    '''
    A superclass for object classes in HARK.  Comes with two useful methods:
    a generic/universal distance method and an attribute assignment method.
    '''
    def distance(self, other):
        '''
        A generic distance method, which requires the existence of an attribute
        called distance_criteria, giving a list of strings naming the attributes
        to be considered by the distance metric.

        Parameters
        ----------
        other : object
            Another object to compare this instance to.

        Returns
        -------
        (unnamed) : float
            The distance between this object and another, using the "universal
            distance" metric.
        '''
        distance_list = [0.0]
        for attr_name in self.distance_criteria:
            try:
                obj_A = getattr(self, attr_name)
                obj_B = getattr(other, attr_name)
                distance_list.append(distanceMetric(obj_A, obj_B))
            except AttributeError:
                distance_list.append(1000.0)  # if either object lacks attribute, they are not the same
        return max(distance_list)

    def assignParameters(self, **kwds):
        '''
        Assign an arbitrary number of attributes to this agent.

        Parameters
        ----------
        **kwds : keyword arguments
            Any number of keyword arguments of the form key=value.  Each value
            will be assigned to the attribute named in self.

        Returns
        -------
        none
        '''
        for key in kwds:
            setattr(self, key, kwds[key])

    def __call__(self, **kwds):
        '''
        Assign an arbitrary number of attributes to this agent, as a convenience.
        See assignParameters.
        '''
        self.assignParameters(**kwds)

    def getAvg(self, varname, **kwds):
        '''
        Calculates the average of an attribute of this instance.  Returns NaN if no such attribute.

        Parameters
        ----------
        varname : string
            The name of the attribute whose average is to be calculated.  This attribute must be an
            np.array or other class compatible with np.mean.

        Returns
        -------
        avg : float or np.array
            The average of this attribute.  Might be an array if the axis keyword is passed.
        '''
        if hasattr(self, varname):
            return np.mean(getattr(self, varname), **kwds)
        else:
            return np.nan


class Solution(HARKobject):
    '''
    A superclass for representing the "solution" to a single period problem in a
    dynamic microeconomic model.

    NOTE: This can be deprecated now that HARKobject exists, but this requires
    replacing each instance of Solution with HARKobject in the other modules.
    '''


class AgentType(HARKobject):
    '''
    A superclass for economic agents in the HARK framework. Each model should
    specify its own subclass of AgentType, inheriting its methods and overwriting 
    as necessary.  Critically, every subclass of AgentType should define class-
    specific static values of the attributes time_vary and time_inv as lists of
    strings.  Each element of time_vary is the name of a field in AgentSubType
    that varies over time in the model.  Each element of time_inv is the name of
    a field in AgentSubType that is constant over time in the model.
    '''
    def __init__(self, solution_terminal=None, cycles=1, time_flow=True, pseudo_terminal=True,
                 tolerance=0.000001, seed=0, **kwds):
        '''
        Initialize an instance of AgentType by setting attributes.

        Parameters
        ----------
        solution_terminal : Solution
            A representation of the solution to the terminal period problem of
            this AgentType instance, or an initial guess of the solution if this
            is an infinite horizon problem.
        cycles : int
            The number of times the sequence of periods is experienced by this
            AgentType in their "lifetime".  cycles=1 corresponds to a lifecycle
            model, with a certain sequence of one period problems experienced
            once before terminating.  cycles=0 corresponds to an infinite horizon
            model, with a sequence of one period problems repeating indefinitely.
        time_flow : boolean
            Whether time is currently "flowing" forward(True) or backward(False) for this
            instance.  Used to flip between solving (using backward iteration)
            and simulating (etc).
        pseudo_terminal : boolean
            Indicates whether solution_terminal isn't actually part of the
            solution to the problem (as a known solution to the terminal period
            problem), but instead represents a "scrap value"-style termination.
            When True, solution_terminal is not included in the solution; when
            false, solution_terminal is the last element of the solution.
        tolerance : float
            Maximum acceptable "distance" between successive solutions to the
            one period problem in an infinite horizon (cycles=0) model in order
            for the solution to be considered as having "converged".  Inoperative
            when cycles>0.
        seed : int
            A seed for this instance's random number generator.

        Returns
        -------
        None
        '''
        if solution_terminal is None:
            solution_terminal = NullFunc()
        self.solution_terminal  = solution_terminal # NOQA
        self.cycles             = cycles # NOQA
        self.time_flow          = time_flow # NOQA
        self.pseudo_terminal    = pseudo_terminal # NOQA
        self.solveOnePeriod     = NullFunc() # NOQA
        self.tolerance          = tolerance # NOQA
        self.seed               = seed # NOQA
        self.track_vars         = [] # NOQA
        self.poststate_vars     = [] # NOQA
        self.read_shocks        = False # NOQA
        self.assignParameters(**kwds) # NOQA
        self.resetRNG() # NOQA

    def timeReport(self):
        '''
        Report to the user the direction that time is currently "flowing" for
        this instance.  Only exists as a reminder of how time_flow works.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        if self.time_flow:
            print('Time varying objects are listed in ordinary chronological order.')
        else:
            print('Time varying objects are listed in reverse chronological order.')

    def timeFlip(self):
        '''
        Reverse the flow of time for this instance.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        for name in self.time_vary:
            self.__dict__[name].reverse()
        self.time_flow = not self.time_flow

    def timeFwd(self):
        '''
        Make time flow forward for this instance.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        if not self.time_flow:
            self.timeFlip()

    def timeRev(self):
        '''
        Make time flow backward for this instance.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        if self.time_flow:
            self.timeFlip()

    def addToTimeVary(self, *params):
        '''
        Adds any number of parameters to time_vary for this instance.

        Parameters
        ----------
        params : string
            Any number of strings naming attributes to be added to time_vary

        Returns
        -------
        None
        '''
        for param in params:
            if param not in self.time_vary:
                self.time_vary.append(param)

    def addToTimeInv(self, *params):
        '''
        Adds any number of parameters to time_inv for this instance.

        Parameters
        ----------
        params : string
            Any number of strings naming attributes to be added to time_inv

        Returns
        -------
        None
        '''
        for param in params:
            if param not in self.time_inv:
                self.time_inv.append(param)

    def delFromTimeVary(self, *params):
        '''
        Removes any number of parameters from time_vary for this instance.

        Parameters
        ----------
        params : string
            Any number of strings naming attributes to be removed from time_vary

        Returns
        -------
        None
        '''
        for param in params:
            if param in self.time_vary:
                self.time_vary.remove(param)

    def delFromTimeInv(self, *params):
        '''
        Removes any number of parameters from time_inv for this instance.

        Parameters
        ----------
        params : string
            Any number of strings naming attributes to be removed from time_inv

        Returns
        -------
        None
        '''
        for param in params:
            if param in self.time_inv:
                self.time_inv.remove(param)

    def solve(self, verbose=False):
        '''
        Solve the model for this instance of an agent type by backward induction.
        Loops through the sequence of one period problems, passing the solution
        from period t+1 to the problem for period t.

        Parameters
        ----------
        verbose : boolean
            If True, solution progress is printed to screen.

        Returns
        -------
        none
        '''

        # Ignore floating point "errors". Numpy calls it "errors", but really it's excep-
        # tions with well-defined answers such as 1.0/0.0 that is np.inf, -1.0/0.0 that is
        # -np.inf, np.inf/np.inf is np.nan and so on.
        with np.errstate(divide='ignore', over='ignore', under='ignore', invalid='ignore'):
            self.preSolve()  # Do pre-solution stuff
            self.solution = solveAgent(self, verbose)  # Solve the model by backward induction
            if self.time_flow:  # Put the solution in chronological order if this instance's time flow runs that way
                self.solution.reverse()
            self.addToTimeVary('solution')  # Add solution to the list of time-varying attributes
            self.postSolve()  # Do post-solution stuff

    def resetRNG(self):
        '''
        Reset the random number generator for this type.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        self.RNG = np.random.RandomState(self.seed)

    def checkElementsOfTimeVaryAreLists(self):
        """
        A method to check that elements of time_vary are lists.
        """
        for param in self.time_vary:
            assert type(getattr(self, param)) == list, param + ' is not a list, but should be' + \
                                                   ' because it is in time_vary'

    def checkRestrictions(self):
        """
        A method to check that various restrictions are met for the model class.
        """
        return

    def preSolve(self):
        '''
        A method that is run immediately before the model is solved, to check inputs or to prepare
        the terminal solution, perhaps.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        self.checkRestrictions()
        self.checkElementsOfTimeVaryAreLists()
        return None

    def postSolve(self):
        '''
        A method that is run immediately after the model is solved, to finalize
        the solution in some way.  Does nothing here.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        return None

    def initializeSim(self):
        '''
        Prepares this AgentType for a new simulation.  Resets the internal random number generator,
        makes initial states for all agents (using simBirth), clears histories of tracked variables.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        if not hasattr(self, 'T_sim'):
            raise Exception('To initialize simulation variables it is necessary to first ' +
                            'set the attribute T_sim to the largest number of observations ' +
                            'you plan to simulate for each agent including re-births.')
        elif self.T_sim <= 0:
            raise Exception('T_sim represents the largest number of observations ' +
                            'that can be simulated for an agent, and must be a positive number.')

        self.resetRNG()
        self.t_sim = 0
        all_agents = np.ones(self.AgentCount, dtype=bool)
        blank_array = np.zeros(self.AgentCount)
        for var_name in self.poststate_vars:
            setattr(self, var_name, copy(blank_array))
            # exec('self.' + var_name + ' = copy(blank_array)')
        self.t_age = np.zeros(self.AgentCount, dtype=int)    # Number of periods since agent entry
        self.t_cycle = np.zeros(self.AgentCount, dtype=int)  # Which cycle period each agent is on
        self.simBirth(all_agents)
        self.clearHistory()
        return None

    def simOnePeriod(self):
        '''
        Simulates one period for this type.  Calls the methods getMortality(), getShocks() or
        readShocks, getStates(), getControls(), and getPostStates().  These should be defined for
        AgentType subclasses, except getMortality (define its components simDeath and simBirth
        instead) and readShocks.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        if not hasattr(self, 'solution'):
            raise Exception('Model instance does not have a solution stored. To simulate, it is necessary'
                            ' to run the `solve()` method of the class first.')

        self.getMortality()  # Replace some agents with "newborns"
        if self.read_shocks:  # If shock histories have been pre-specified, use those
            self.readShocks()
        else:                # Otherwise, draw shocks as usual according to subclass-specific method
            self.getShocks()
        self.getStates()  # Determine each agent's state at decision time
        self.getControls()   # Determine each agent's choice or control variables based on states
        self.getPostStates()  # Determine each agent's post-decision / end-of-period states using states and controls

        # Advance time for all agents
        self.t_age = self.t_age + 1  # Age all consumers by one period
        self.t_cycle = self.t_cycle + 1  # Age all consumers within their cycle
        self.t_cycle[self.t_cycle == self.T_cycle] = 0  # Resetting to zero for those who have reached the end

    def makeShockHistory(self):
        '''
        Makes a pre-specified history of shocks for the simulation.  Shock variables should be named
        in self.shock_vars, a list of strings that is subclass-specific.  This method runs a subset
        of the standard simulation loop by simulating only mortality and shocks; each variable named
        in shock_vars is stored in a T_sim x AgentCount array in an attribute of self named X_hist.
        Automatically sets self.read_shocks to True so that these pre-specified shocks are used for
        all subsequent calls to simulate().

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        # Make sure time is flowing forward and re-initialize the simulation
        orig_time = self.time_flow
        self.timeFwd()
        self.initializeSim()

        # Make blank history arrays for each shock variable (and mortality)
        for var_name in self.shock_vars:
            setattr(self, var_name+'_hist', np.zeros((self.T_sim, self.AgentCount)) + np.nan)
        setattr(self, 'who_dies_hist', np.zeros((self.T_sim, self.AgentCount), dtype=bool))

        # Make and store the history of shocks for each period
        for t in range(self.T_sim):
            self.getMortality()
            self.who_dies_hist[t,:] = self.who_dies
            self.getShocks()
            for var_name in self.shock_vars:
                exec('self.' + var_name + '_hist[t,:] = self.' + var_name)
            self.t_sim += 1
            self.t_age = self.t_age + 1  # Age all consumers by one period
            self.t_cycle = self.t_cycle + 1  # Age all consumers within their cycle
            self.t_cycle[self.t_cycle == self.T_cycle] = 0  # Resetting to zero for those who have reached the end

        # Restore the flow of time and flag that shocks can be read rather than simulated
        self.read_shocks = True
        if not orig_time:
            self.timeRev()

    def getMortality(self):
        '''
        Simulates mortality or agent turnover according to some model-specific rules named simDeath
        and simBirth (methods of an AgentType subclass).  simDeath takes no arguments and returns
        a Boolean array of size AgentCount, indicating which agents of this type have "died" and
        must be replaced.  simBirth takes such a Boolean array as an argument and generates initial
        post-decision states for those agent indices.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        if self.read_shocks:
            who_dies = self.who_dies_hist[self.t_sim,:]
        else:
            who_dies = self.simDeath()
        self.simBirth(who_dies)
        self.who_dies = who_dies
        return None

    def simDeath(self):
        '''
        Determines which agents in the current population "die" or should be replaced.  Takes no
        inputs, returns a Boolean array of size self.AgentCount, which has True for agents who die
        and False for those that survive. Returns all False by default, must be overwritten by a
        subclass to have replacement events.

        Parameters
        ----------
        None

        Returns
        -------
        who_dies : np.array
            Boolean array of size self.AgentCount indicating which agents die and are replaced.
        '''
        who_dies = np.zeros(self.AgentCount, dtype=bool)
        return who_dies

    def simBirth(self, which_agents):
        '''
        Makes new agents for the simulation.  Takes a boolean array as an input, indicating which
        agent indices are to be "born".  Does nothing by default, must be overwritten by a subclass.

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        '''
        print('AgentType subclass must define method simBirth!')
        return None

    def getShocks(self):
        '''
        Gets values of shock variables for the current period.  Does nothing by default, but can
        be overwritten by subclasses of AgentType.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        return None

    def readShocks(self):
        '''
        Reads values of shock variables for the current period from history arrays.  For each var-
        iable X named in self.shock_vars, this attribute of self is set to self.X_hist[self.t_sim,:].

        This method is only ever called if self.read_shocks is True.  This can be achieved by using
        the method makeShockHistory() (or manually after storing a "handcrafted" shock history).

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        for var_name in self.shock_vars:
            setattr(self, var_name, getattr(self, var_name + '_hist')[self.t_sim, :])

    def getStates(self):
        '''
        Gets values of state variables for the current period, probably by using post-decision states
        from last period, current period shocks, and maybe market-level events.  Does nothing by
        default, but can be overwritten by subclasses of AgentType.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        return None

    def getControls(self):
        '''
        Gets values of control variables for the current period, probably by using current states.
        Does nothing by default, but can be overwritten by subclasses of AgentType.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        return None

    def getPostStates(self):
        '''
        Gets values of post-decision state variables for the current period, probably by current
        states and controls and maybe market-level events or shock variables.  Does nothing by
        default, but can be overwritten by subclasses of AgentType.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        return None

    def simulate(self, sim_periods=None):
        '''
        Simulates this agent type for a given number of periods. Defaults to self.T_sim if no input.
        Records histories of attributes named in self.track_vars in attributes named varname_hist.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        if not hasattr(self, 't_sim'):
            raise Exception('It seems that the simulation variables were not initialize before calling ' +
                            'simulate(). Call initializeSim() to initialize the variables before calling simulate() again.')

        if not hasattr(self, 'T_sim'):
            raise Exception('This agent type instance must have the attribute T_sim set to a positive integer.' +
                             'Set T_sim to match the largest dataset you might simulate, and run this agent\'s' +
                             'initalizeSim() method before running simulate() again.')

        if sim_periods is not None and self.T_sim < sim_periods:
            raise Exception('To simulate, sim_periods has to be larger than the maximum data set size ' +
                             'T_sim. Either increase the attribute T_sim of this agent type instance ' +
                             'and call the initializeSim() method again, or set sim_periods <= T_sim.')


        # Ignore floating point "errors". Numpy calls it "errors", but really it's excep-
        # tions with well-defined answers such as 1.0/0.0 that is np.inf, -1.0/0.0 that is
        # -np.inf, np.inf/np.inf is np.nan and so on.
        with np.errstate(divide='ignore', over='ignore', under='ignore', invalid='ignore'):
            orig_time = self.time_flow
            self.timeFwd()
            if sim_periods is None:
                sim_periods = self.T_sim

            for t in range(sim_periods):
                self.simOnePeriod()
                for var_name in self.track_vars:
                    exec('self.' + var_name + '_hist[self.t_sim,:] = self.' + var_name)
                self.t_sim += 1

            if not orig_time:
                self.timeRev()

    def clearHistory(self):
        '''
        Clears the histories of the attributes named in self.track_vars.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        for var_name in self.track_vars:
            exec('self.' + var_name + '_hist = np.zeros((self.T_sim,self.AgentCount)) + np.nan')


def solveAgent(agent, verbose):
    '''
    Solve the dynamic model for one agent type.  This function iterates on "cycles"
    of an agent's model either a given number of times or until solution convergence
    if an infinite horizon model is used (with agent.cycles = 0).

    Parameters
    ----------
    agent : AgentType
        The microeconomic AgentType whose dynamic problem is to be solved.
    verbose : boolean
        If True, solution progress is printed to screen (when cycles != 1).

    Returns
    -------
    solution : [Solution]
        A list of solutions to the one period problems that the agent will
        encounter in his "lifetime".  Returns in reverse chronological order.
    '''
    # Record the flow of time when the Agent began the process, and make sure time is flowing backwards
    original_time_flow = agent.time_flow
    agent.timeRev()

    # Check to see whether this is an (in)finite horizon problem
    cycles_left      = agent.cycles # NOQA
    infinite_horizon = cycles_left == 0 # NOQA
    # Initialize the solution, which includes the terminal solution if it's not a pseudo-terminal period
    solution = []
    if not agent.pseudo_terminal:
        solution.append(deepcopy(agent.solution_terminal))


    # Initialize the process, then loop over cycles
    solution_last    = agent.solution_terminal # NOQA
    go               = True # NOQA
    completed_cycles = 0 # NOQA
    max_cycles       = 5000 # NOQA  - escape clause
    if verbose:
        t_last = time()
    while go:
        # Solve a cycle of the model, recording it if horizon is finite
        solution_cycle = solveOneCycle(agent, solution_last)
        if not infinite_horizon:
            solution += solution_cycle

        # Check for termination: identical solutions across cycle iterations or run out of cycles
        solution_now = solution_cycle[-1]
        if infinite_horizon:
            if completed_cycles > 0:
                solution_distance = solution_now.distance(solution_last)
                agent.solution_distance = solution_distance  # Add these attributes so users can 
                agent.completed_cycles  = completed_cycles   # query them to see if solution is ready
                go = (solution_distance > agent.tolerance and completed_cycles < max_cycles)
            else:  # Assume solution does not converge after only one cycle
                solution_distance = 100.0
                go = True
        else:
            cycles_left += -1
            go = cycles_left > 0

        # Update the "last period solution"
        solution_last = solution_now
        completed_cycles += 1

        # Display progress if requested
        if verbose:
            t_now = time()
            if infinite_horizon:
                print('Finished cycle #' + str(completed_cycles) + ' in ' + str(t_now-t_last) +
                      ' seconds, solution distance = ' + str(solution_distance))
            else:
                print('Finished cycle #' + str(completed_cycles) + ' of ' + str(agent.cycles) +
                      ' in ' + str(t_now-t_last) + ' seconds.')
            t_last = t_now

    # Record the last cycle if horizon is infinite (solution is still empty!)
    if infinite_horizon:
        solution = solution_cycle  # PseudoTerminal=False impossible for infinite horizon

    # Restore the direction of time to its original orientation, then return the solution
    if original_time_flow:
        agent.timeFwd()
    return solution


def solveOneCycle(agent, solution_last):
    '''
    Solve one "cycle" of the dynamic model for one agent type.  This function
    iterates over the periods within an agent's cycle, updating the time-varying
    parameters and passing them to the single period solver(s).

    Parameters
    ----------
    agent : AgentType
        The microeconomic AgentType whose dynamic problem is to be solved.
    solution_last : Solution
        A representation of the solution of the period that comes after the
        end of the sequence of one period problems.  This might be the term-
        inal period solution, a "pseudo terminal" solution, or simply the
        solution to the earliest period from the succeeding cycle.

    Returns
    -------
    solution_cycle : [Solution]
        A list of one period solutions for one "cycle" of the AgentType's
        microeconomic model.  Returns in reverse chronological order.
    '''
    # Calculate number of periods per cycle, defaults to 1 if all variables are time invariant
    if len(agent.time_vary) > 0:
        # name = agent.time_vary[0]
        # T = len(eval('agent.' + name))
        T = len(agent.__dict__[agent.time_vary[0]])
    else:
        T = 1

    # Construct a dictionary to be passed to the solver
    # time_inv_string = ''
    # for name in agent.time_inv:
    #     time_inv_string += ' \'' + name + '\' : agent.' + name + ','
    # time_vary_string = ''
    # for name in agent.time_vary:
    #     time_vary_string += ' \'' + name + '\' : None,'
    # solve_dict = eval('{' + time_inv_string + time_vary_string + '}')
    solve_dict = {parameter: agent.__dict__[parameter] for parameter in agent.time_inv}
    solve_dict.update({parameter: None for parameter in agent.time_vary})

    # Initialize the solution for this cycle, then iterate on periods
    solution_cycle = []
    solution_next = solution_last
    for t in range(T):
        # Update which single period solver to use (if it depends on time)
        if hasattr(agent.solveOnePeriod, "__getitem__"):
            solveOnePeriod = agent.solveOnePeriod[t]
        else:
            solveOnePeriod = agent.solveOnePeriod

        these_args = getArgNames(solveOnePeriod)

        # Update time-varying single period inputs
        for name in agent.time_vary:
            if name in these_args:
                # solve_dict[name] = eval('agent.' + name + '[t]')
                solve_dict[name] = agent.__dict__[name][t]
        solve_dict['solution_next'] = solution_next

        # Make a temporary dictionary for this period
        temp_dict = {name: solve_dict[name] for name in these_args}

        # Solve one period, add it to the solution, and move to the next period
        solution_t = solveOnePeriod(**temp_dict)
        solution_cycle.append(solution_t)
        solution_next = solution_t

    # Return the list of per-period solutions
    return solution_cycle


# ========================================================================
# ========================================================================

class Market(HARKobject):
    '''
    A superclass to represent a central clearinghouse of information.  Used for
    dynamic general equilibrium models to solve the "macroeconomic" model as a
    layer on top of the "microeconomic" models of one or more AgentTypes.
    '''
    def __init__(self, agents=[], sow_vars=[], reap_vars=[], const_vars=[], track_vars=[], dyn_vars=[],
                 millRule=None, calcDynamics=None, act_T=1000, tolerance=0.000001,**kwds):
        '''
        Make a new instance of the Market class.

        Parameters
        ----------
        agents : [AgentType]
            A list of all the AgentTypes in this market.
        sow_vars : [string]
            Names of variables generated by the "aggregate market process" that should
            be "sown" to the agents in the market.  Aggregate state, etc.
        reap_vars : [string]
            Names of variables to be collected ("reaped") from agents in the market
            to be used in the "aggregate market process".
        const_vars : [string]
            Names of attributes of the Market instance that are used in the "aggregate
            market process" but do not come from agents-- they are constant or simply
            parameters inherent to the process.
        track_vars : [string]
            Names of variables generated by the "aggregate market process" that should
            be tracked as a "history" so that a new dynamic rule can be calculated.
            This is often a subset of sow_vars.
        dyn_vars : [string]
            Names of variables that constitute a "dynamic rule".
        millRule : function
            A function that takes inputs named in reap_vars and returns an object
            with attributes named in sow_vars.  The "aggregate market process" that
            transforms individual agent actions/states/data into aggregate data to
            be sent back to agents.
        calcDynamics : function
            A function that takes inputs named in track_vars and returns an object
            with attributes named in dyn_vars.  Looks at histories of aggregate
            variables and generates a new "dynamic rule" for agents to believe and
            act on.
        act_T : int
            The number of times that the "aggregate market process" should be run
            in order to generate a history of aggregate variables.
        tolerance: float
            Minimum acceptable distance between "dynamic rules" to consider the
            Market solution process converged.  Distance is a user-defined metric.

        Returns
        -------
        None
    '''
        self.agents     = agents # NOQA
        self.reap_vars  = reap_vars # NOQA
        self.sow_vars   = sow_vars # NOQA
        self.const_vars = const_vars # NOQA
        self.track_vars = track_vars # NOQA
        self.dyn_vars   = dyn_vars # NOQA
        if millRule is not None:  # To prevent overwriting of method-based millRules
            self.millRule = millRule
        if calcDynamics is not None:  # Ditto for calcDynamics
            self.calcDynamics = calcDynamics
        self.act_T     = act_T # NOQA
        self.tolerance = tolerance # NOQA
        self.max_loops = 1000 # NOQA
        self.assignParameters(**kwds)

        self.print_parallel_error_once = True
        # Print the error associated with calling the parallel method
        # "solveAgents" one time. If set to false, the error will never
        # print. See "solveAgents" for why this prints once or never.

    def solveAgents(self):
        '''
        Solves the microeconomic problem for all AgentTypes in this market.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        # for this_type in self.agents:
        # this_type.solve()
        try:
            multiThreadCommands(self.agents, ['solve()'])
        except Exception as err:
            if self.print_parallel_error_once:
                # Set flag to False so this is only printed once.
                self.print_parallel_error_once = False
                print("**** WARNING: could not execute multiThreadCommands in HARK.core.Market.solveAgents() ",
                      "so using the serial version instead. This will likely be slower. "
                      "The multiTreadCommands() functions failed with the following error:", '\n',
                      sys.exc_info()[0], ':', err)  # sys.exc_info()[0])
            multiThreadCommandsFake(self.agents, ['solve()'])

    def solve(self):
        '''
        "Solves" the market by finding a "dynamic rule" that governs the aggregate
        market state such that when agents believe in these dynamics, their actions
        collectively generate the same dynamic rule.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        go = True
        max_loops = self.max_loops  # Failsafe against infinite solution loop
        completed_loops = 0
        old_dynamics = None

        while go:  # Loop until the dynamic process converges or we hit the loop cap
            self.solveAgents()  # Solve each AgentType's micro problem
            self.makeHistory()  # "Run" the model while tracking aggregate variables
            new_dynamics = self.updateDynamics()  # Find a new aggregate dynamic rule

            # Check to see if the dynamic rule has converged (if this is not the first loop)
            if completed_loops > 0:
                distance = new_dynamics.distance(old_dynamics)
            else:
                distance = 1000000.0

            # Move to the next loop if the terminal conditions are not met
            old_dynamics = new_dynamics
            completed_loops += 1
            go = distance >= self.tolerance and completed_loops < max_loops

        self.dynamics = new_dynamics  # Store the final dynamic rule in self

    def reap(self):
        '''
        Collects attributes named in reap_vars from each AgentType in the market,
        storing them in respectively named attributes of self.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        for var_name in self.reap_vars:
            harvest = []
            for this_type in self.agents:
                harvest.append(getattr(this_type, var_name))
            setattr(self, var_name, harvest)

    def sow(self):
        '''
        Distributes attrributes named in sow_vars from self to each AgentType
        in the market, storing them in respectively named attributes.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        for var_name in self.sow_vars:
            this_seed = getattr(self, var_name)
            for this_type in self.agents:
                setattr(this_type, var_name, this_seed)

    def mill(self):
        '''
        Processes the variables collected from agents using the function millRule,
        storing the results in attributes named in aggr_sow.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        # Make a dictionary of inputs for the millRule
        reap_vars_string = ''
        for name in self.reap_vars:
            reap_vars_string += ' \'' + name + '\' : self.' + name + ','
        const_vars_string = ''
        for name in self.const_vars:
            const_vars_string += ' \'' + name + '\' : self.' + name + ','
        mill_dict = eval('{' + reap_vars_string + const_vars_string + '}')

        # Run the millRule and store its output in self
        product = self.millRule(**mill_dict)
        for j in range(len(self.sow_vars)):
            this_var = self.sow_vars[j]
            this_product = getattr(product, this_var)
            setattr(self, this_var, this_product)

    def cultivate(self):
        '''
        Has each AgentType in agents perform their marketAction method, using
        variables sown from the market (and maybe also "private" variables).
        The marketAction method should store new results in attributes named in
        reap_vars to be reaped later.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        for this_type in self.agents:
            this_type.marketAction()

    def reset(self):
        '''
        Reset the state of the market (attributes in sow_vars, etc) to some
        user-defined initial state, and erase the histories of tracked variables.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        for var_name in self.track_vars:  # Reset the history of tracked variables
            setattr(self, var_name + '_hist', [])
        for var_name in self.sow_vars:  # Set the sow variables to their initial levels
            initial_val = getattr(self, var_name + '_init')
            setattr(self, var_name, initial_val)
        for this_type in self.agents:  # Reset each AgentType in the market
            this_type.reset()

    def store(self):
        '''
        Record the current value of each variable X named in track_vars in an
        attribute named X_hist.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        for var_name in self.track_vars:
            value_now = getattr(self, var_name)
            getattr(self, var_name + '_hist').append(value_now)

    def makeHistory(self):
        '''
        Runs a loop of sow-->cultivate-->reap-->mill act_T times, tracking the
        evolution of variables X named in track_vars in attributes named X_hist.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        self.reset()  # Initialize the state of the market
        for t in range(self.act_T):
            self.sow()  # Distribute aggregated information/state to agents
            self.cultivate()  # Agents take action
            self.reap()  # Collect individual data from agents
            self.mill()  # Process individual data into aggregate data
            self.store()  # Record variables of interest

    def updateDynamics(self):
        '''
        Calculates a new "aggregate dynamic rule" using the history of variables
        named in track_vars, and distributes this rule to AgentTypes in agents.

        Parameters
        ----------
        none

        Returns
        -------
        dynamics : instance
            The new "aggregate dynamic rule" that agents believe in and act on.
            Should have attributes named in dyn_vars.
        '''
        # Make a dictionary of inputs for the dynamics calculator
        history_vars_string = ''
        arg_names = list(getArgNames(self.calcDynamics))
        if 'self' in arg_names:
            arg_names.remove('self')
        for name in arg_names:
            history_vars_string += ' \'' + name + '\' : self.' + name + '_hist,'
        update_dict = eval('{' + history_vars_string + '}')

        # Calculate a new dynamic rule and distribute it to the agents in agent_list
        dynamics = self.calcDynamics(**update_dict)  # User-defined dynamics calculator
        for var_name in self.dyn_vars:
            this_obj = getattr(dynamics, var_name)
            for this_type in self.agents:
                setattr(this_type, var_name, this_obj)
        return dynamics


# ------------------------------------------------------------------------------
# Code to copy entire modules to a local directory
# ------------------------------------------------------------------------------

#  Define a function to run the copying:
def copy_module(target_path, my_directory_full_path, my_module):
    '''
    Helper function for copy_module_to_local(). Provides the actual copy
    functionality, with highly cautious safeguards against copying over
    important things.

    Parameters
    ----------
    target_path : string
        String, file path to target location

    my_directory_full_path: string
        String, full pathname to this file's directory

    my_module : string
        String, name of the module to copy

    Returns
    -------
    none
    '''

    if target_path == 'q' or target_path == 'Q':
        print("Goodbye!")
        return
    elif target_path == os.path.expanduser("~") or os.path.normpath(target_path) == os.path.expanduser("~"):
        print("You have indicated that the target location is " + target_path +
              " -- that is, you want to wipe out your home directory with the contents of " + my_module +
              ". My programming does not allow me to do that.\n\nGoodbye!")
        return
    elif os.path.exists(target_path):
        print("There is already a file or directory at the location " + target_path +
              ". For safety reasons this code does not overwrite existing files.\n Please remove the file at "
              + target_path +
              " and try again.")
        return
    else:
        user_input = input("""You have indicated you want to copy module:\n    """ + my_module
                           + """\nto:\n    """ + target_path + """\nIs that correct? Please indicate: y / [n]\n\n""")
        if user_input == 'y' or user_input == 'Y':
            # print("copy_tree(",my_directory_full_path,",", target_path,")")
            copy_tree(my_directory_full_path, target_path)
        else:
            print("Goodbye!")
            return


def print_helper():

    my_directory_full_path = os.path.dirname(os.path.realpath(__file__))

    print(my_directory_full_path)


def copy_module_to_local(full_module_name):
    '''
    This function contains simple code to copy a submodule to a location on
    your hard drive, as specified by you. The purpose of this code is to provide
    users with a simple way to access a *copy* of code that usually sits deep in
    the Econ-ARK package structure, for purposes of tinkering and experimenting
    directly. This is meant to be a simple way to explore HARK code. To interact
    with the codebase under active development, please refer to the documentation
    under github.com/econ-ark/HARK/

    To execute, do the following on the Python command line:

        from HARK.core import copy_module_to_local
        copy_module_to_local("FULL-HARK-MODULE-NAME-HERE")

    For example, if you want SolvingMicroDSOPs you would enter

        from HARK.core import copy_module_to_local
        copy_module_to_local("HARK.SolvingMicroDSOPs")

    '''

    # Find a default directory -- user home directory:
    home_directory_RAW = os.path.expanduser("~")
    # Thanks to https://stackoverflow.com/a/4028943

    # Find the directory of the HARK.core module:
    # my_directory_full_path = os.path.dirname(os.path.realpath(__file__))
    hark_core_directory_full_path = os.path.dirname(os.path.realpath(__file__))
    # From https://stackoverflow.com/a/5137509
    # Important note from that answer:
    # (Note that the incantation above won't work if you've already used os.chdir()
    # to change your current working directory,
    # since the value of the __file__ constant is relative to the current working directory and is not changed by an
    #  os.chdir() call.)
    #
    # NOTE: for this specific file that I am testing, the path should be:
    # '/home/npalmer/anaconda3/envs/py3fresh/lib/python3.6/site-packages/HARK/SolvingMicroDSOPs/---example-file---

    # Split out the name of the module. Break if proper format is not followed:
    all_module_names_list = full_module_name.split('.')  # Assume put in at correct format
    if all_module_names_list[0] != "HARK":
        print("\nWarning: the module name does not start with 'HARK'. Instead it is: '"
              + all_module_names_list[0]+"' --please format the full namespace of the module you want. \n"
              "For example, 'HARK.SolvingMicroDSOPs'")
        print("\nGoodbye!")
        return

    # Construct the pathname to the module to copy:
    my_directory_full_path = hark_core_directory_full_path
    for a_directory_name in all_module_names_list[1:]:
        my_directory_full_path = os.path.join(my_directory_full_path, a_directory_name)

    head_path, my_module = os.path.split(my_directory_full_path)

    home_directory_with_module = os.path.join(home_directory_RAW, my_module)

    print("\n\n\nmy_directory_full_path:", my_directory_full_path, '\n\n\n')

    # Interact with the user:
    #     - Ask the user for the target place to copy the directory
    #         - Offer use "q/y/other" option
    #     - Check if there is something there already
    #     - If so, ask if should replace
    #     - If not, just copy there
    #     - Quit

    target_path = input("""You have invoked the 'replicate' process for the current module:\n    """ +
                        my_module + """\nThe default copy location is your home directory:\n    """ +
                        home_directory_with_module + """\nPlease enter one of the three options in single quotes below, excluding the quotes:

        'q' or return/enter to quit the process
        'y' to accept the default home directory: """+home_directory_with_module+"""
        'n' to specify your own pathname\n\n""")

    if target_path == 'n' or target_path == 'N':
        target_path = input("""Please enter the full pathname to your target directory location: """)

        # Clean up:
        target_path = os.path.expanduser(target_path)
        target_path = os.path.expandvars(target_path)
        target_path = os.path.normpath(target_path)

        # Check to see if they included the module name; if not add it here:
        temp_head, temp_tail = os.path.split(target_path)
        if temp_tail != my_module:
            target_path = os.path.join(target_path, my_module)

    elif target_path == 'y' or target_path == 'Y':
        # Just using the default path:
        target_path = home_directory_with_module
    else:
        # Assume "quit"
        return

    if target_path != 'q' and target_path != 'Q' or target_path == '':
        # Run the copy command:
        copy_module(target_path, my_directory_full_path, my_module)

    return

    if target_path != 'q' and target_path != 'Q' or target_path == '':
        # Run the copy command:
        copy_module(target_path, my_directory_full_path, my_module)

    return


def main():
    print("Sorry, HARK.core doesn't actually do anything on its own.")
    print("To see some examples of its frameworks in action, try running a model module.")
    print("Several interesting model modules can be found in /ConsumptionSavingModel.")
    print('For an extraordinarily simple model that demonstrates the "microeconomic" and')
    print('"macroeconomic" frameworks, see /FashionVictim/FashionVictimModel.')


if __name__ == '__main__':
    main()
