'''
High-level functions and classes for solving a wide variety of economic models.
The "core" of HARK is a framework for "microeconomic" and "macroeconomic"
models.  A micro model concerns the dynamic optimization problem for some type
of agents, where agents take the inputs to their problem as exogenous.  A macro
model adds an additional layer, endogenizing some of the inputs to the micro
problem by finding a general equilibrium dynamic rule.
'''

from HARKutilities import getArgNames, NullFunc
from copy import deepcopy
import numpy as np

 
def distanceMetric(thing_A,thing_B):
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
        lenA = len(thing_A) # If both inputs are lists, then the distance between
        lenB = len(thing_B) # them is the maximum distance between corresponding
        if lenA == lenB:    # elements in the lists.  If they differ in length,
            distance_temp = [] # the distance is the difference in lengths.
            for n in range(lenA):
                distance_temp.append(distanceMetric(thing_A[n],thing_B[n]))
            distance = max(distance_temp)
        else:
            distance = float(abs(lenA - lenB))
    # If both inputs are numbers, return their difference        
    elif (typeA is int or typeB is float) and (typeB is int or typeB is float):
        distance = float(abs(thing_A - thing_B)) 
    # If both inputs are array-like, return the maximum absolute difference b/w
    # corresponding elements (if same shape); return largest difference in dimensions
    # if shapes do not align.
    elif hasattr(thing_A,'shape') and hasattr(thing_B,'shape'):
        if thing_A.shape == thing_B.shape: 
            distance = np.max(abs(thing_A - thing_B))
        else:
            distance = np.max(abs(thing_A.shape - thing_B.shape))
    # If none of the above cases, but the objects are of the same class, call
    # the distance method of one on the other
    elif thing_A.__class__.__name__ is thing_B.__class__.__name__:
        distance = thing_A.distance(thing_B)
    else: # Failsafe: the inputs are very far apart
        distance = 1000.0    
    return distance
        
class HARKobject():
    '''
    A superclass for object classes in HARK.  Comes with two useful methods:
    a generic/universal distance method and an attribute assignment method.
    '''
    def distance(self,other):
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
                obj_A = getattr(self,attr_name)
                obj_B = getattr(other,attr_name)
                distance_list.append(distanceMetric(obj_A,obj_B))
            except:
                distance_list.append(1000.0) # if either object lacks attribute, they are not the same
        return max(distance_list)
        
    def assignParameters(self,**kwds):
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
            setattr(self,key,kwds[key])
            
    def __call__(self,**kwds):
        '''
        Assign an arbitrary number of attributes to this agent, as a convenience.
        See assignParameters.
        '''
        self.assignParameters(**kwds)
        
class Solution(HARKobject):
    '''
    A superclass for representing the "solution" to a single period problem in a
    dynamic microeconomic model.
    
    NOTE: This can be deprecated now that HARKobject exists, but this requires
    replacing each instance of Solution with HARKobject in the other modules.
    '''    
    
class AgentType(HARKobject):
    '''
    A superclass for economic agents in the HARK framework.  Each model should
    specify its own subclass of AgentType, inheriting its methods and overwriting
    as necessary.  Critically, every subclass of AgentType should define class-
    specific static values of the attributes time_vary and time_inv as lists of
    strings.  Each element of time_vary is the name of a field in AgentSubType
    that varies over time in the model.  Each element of time_inv is the name of
    a field in AgentSubType that is constant over time in the model. The string
    'solveOnePeriod' should appear in exactly one of these lists, depending on
    whether the same solution method is used in all periods of the model.
    '''    
    def __init__(self,solution_terminal=None,cycles=1,time_flow=False,pseudo_terminal=True,
                 tolerance=0.000001,seed=0,**kwds):
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
            Whether time is currently "flowing" forward or backward for this
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
        self.solution_terminal  = solution_terminal
        self.cycles             = cycles
        self.time_flow          = time_flow
        self.pseudo_terminal    = pseudo_terminal
        self.solveOnePeriod     = NullFunc()
        self.tolerance          = tolerance
        self.seed               = seed
        self.assignParameters(**kwds)
        self.resetRNG()

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
            exec('self.' + name + '.reverse()')
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
            
    def addToTimeVary(self,*params):
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
                
    def addToTimeInv(self,*params):
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
                
    def delFromTimeVary(self,*params):
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
                
    def delFromTimeInv(self,*params):
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

    def solve(self):
        '''
        Solve the model for this instance of an agent type by backward induction.
        Loops through the sequence of one period problems, passing the solution
        to period t+1 to the problem for period t.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        none
        '''
        self.preSolve() # Do pre-solution stuff
        self.solution = solveAgent(self) # Solve the model by backward induction
        if self.time_flow: # Put the solution in chronological order if this instance's time flow runs that way
            self.solution.reverse()
        self.addToTimeVary('solution') # Add solution to the list of time-varying attributes
        self.postSolve() # Do post-solution stuff
        
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

    def isSameThing(self,solutionA,solutionB):
        '''
        Compare two solutions to see if they are the "same."  The model-specific
        solution class must have a method called distance, which takes another
        solution object as an input and returns the "distance" between the solutions.
        This method is used to test for convergence in infinite horizon problems.
        
        Parameters
        ----------
        solutionA : Solution
            The solution to a one period problem in the model.
            
        solutionB : Solution
            Another solution to (the same) one period problem in the model.
        
        Returns
        -------
        (unnamed) : boolean
            True if the solutions are within a tolerable distance of each other.
        '''
        solution_distance = solutionA.distance(solutionB)
        return(solution_distance <= self.tolerance)
            
    def preSolve(self):
        '''
        A method that is run immediately before the model is solved, to prepare
        the terminal solution, perhaps.  Does nothing here.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        none
        '''
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
        

def solveAgent(agent):
    '''
    Solve the dynamic model for one agent type.  This function iterates on "cycles"
    of an agent's model either a given number of times or until solution convergence
    if an infinite horizon model is used (with agent.cycles = 0).
    
    Parameters
    ----------
    agent : AgentType
        The microeconomic AgentType whose dynamic problem is to be solved.
        
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
    cycles_left      = agent.cycles
    infinite_horizon = cycles_left == 0
    
    # Initialize the solution, which includes the terminal solution if it's not a pseudo-terminal period
    solution = []
    if not agent.pseudo_terminal:
        solution.append(deepcopy(agent.solution_terminal))

    # Initialize the process, then loop over cycles
    solution_last    = agent.solution_terminal
    go               = True
    completed_cycles = 0
    max_cycles       = 5000 # escape clause
    while go:
        # Solve a cycle of the model, recording it if horizon is finite
        solution_cycle = solveOneCycle(agent,solution_last)
        if not infinite_horizon:
            solution += solution_cycle

        # Check for termination: identical solutions across cycle iterations or run out of cycles     
        solution_now = solution_cycle[-1]
        if infinite_horizon:
            if completed_cycles > 0:
                go = (not agent.isSameThing(solution_now,solution_last)) and \
                     (completed_cycles < max_cycles)
            else: # Assume solution does not converge after only one cycle
                go = True
        else:
            cycles_left += -1
            go = cycles_left > 0

        # Update the "last period solution"
        solution_last = solution_now
        completed_cycles += 1

    # Record the last cycle if horizon is infinite (solution is still empty!)
    if infinite_horizon:
        solution = solution_cycle # PseudoTerminal=False impossible for infinite horizon

    # Restore the direction of time to its original orientation, then return the solution
    if original_time_flow:
        agent.timeFwd()
    return solution


def solveOneCycle(agent,solution_last):
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
        name = agent.time_vary[0]
        T    = len(eval('agent.' + name))
    else:
        T = 1

    # Check whether the same solution method is used in all periods
    always_same_solver = 'solveOnePeriod' not in agent.time_vary
    if always_same_solver:
        solveOnePeriod = agent.solveOnePeriod
        these_args     = getArgNames(solveOnePeriod)

    # Construct a dictionary to be passed to the solver
    time_inv_string = ''
    for name in agent.time_inv:
        time_inv_string += ' \'' + name + '\' : agent.' +name + ','
    time_vary_string = ''
    for name in agent.time_vary:
        time_vary_string += ' \'' + name + '\' : None,'
    solve_dict = eval('{' + time_inv_string + time_vary_string + '}')

    # Initialize the solution for this cycle, then iterate on periods
    solution_cycle = []
    solution_next  = solution_last
    for t in range(T):
        # Update which single period solver to use (if it depends on time)
        if not always_same_solver:
            solveOnePeriod = agent.solveOnePeriod[t]
            these_args = getArgNames(solveOnePeriod)

        # Update time-varying single period inputs
        for name in agent.time_vary:
            if name in these_args:
                solve_dict[name] = eval('agent.' + name + '[t]')
        solve_dict['solution_next'] = solution_next
        
        # Make a temporary dictionary for this period
        temp_dict = {name: solve_dict[name] for name in these_args}

        # Solve one period, add it to the solution, and move to the next period
        solution_t = solveOnePeriod(**temp_dict)
        solution_cycle.append(solution_t)
        solution_next = solution_t

    # Return the list of per-period solutions
    return solution_cycle


#========================================================================
#========================================================================    
      
class Market(HARKobject):
    '''
    A superclass to represent a central clearinghouse of information.  Used for
    dynamic general equilibrium models to solve the "macroeconomic" model as a
    layer on top of the "microeconomic" models of one or more AgentTypes.
    '''   
    def __init__(self,agents=[],sow_vars=[],reap_vars=[],const_vars=[],track_vars=[],dyn_vars=[],
                 millRule=None,calcDynamics=None,act_T=1000,tolerance=0.000001):
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
        self.agents      = agents
        self.reap_vars  = reap_vars
        self.sow_vars   = sow_vars
        self.const_vars = const_vars
        self.track_vars = track_vars
        self.dyn_vars   = dyn_vars
        if millRule is not None: # To prevent overwriting of method-based millRules
            self.millRule = millRule
        if calcDynamics is not None: # Ditto for calcDynamics
            self.calcDynamics = calcDynamics
        self.act_T = act_T
        self.tolerance = tolerance
    
    def solve(self):
        '''
        "Solves" the market by finding a "dynamic rule" that governs the aggregate
        market state such that when agents believe in these dynamics, their actions
        collectively generate the same dynamic rule.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        none
        '''
        go              = True
        max_loops       = 1000 # Failsafe against infinite solution loop
        completed_loops = 0
        old_dynamics    = None

        while go: # Loop until the dynamic process converges or we hit the loop cap
            for this_type in self.agents:
                this_type.solve()  # Solve each AgentType's micro problem
            self.makeHistory()     # "Run" the model while tracking aggregate variables
            new_dynamics = self.updateDynamics() # Find a new aggregate dynamic rule
            
            # Check to see if the dynamic rule has converged (if this is not the first loop)
            if completed_loops > 0:
                distance = new_dynamics.distance(old_dynamics)
            else:
                distance = 1000000.0
            
            # Move to the next loop if the terminal conditions are not met
            old_dynamics     = new_dynamics
            completed_loops += 1
            go               = distance >= self.tolerance and completed_loops < max_loops
            
        self.dynamics = new_dynamics # Store the final dynamic rule in self
            
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
                harvest.append(getattr(this_type,var_name))
            setattr(self,var_name,harvest)
            
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
            this_seed = getattr(self,var_name)
            for this_type in self.agents:
                setattr(this_type,var_name,this_seed)
                    
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
            this_product = getattr(product,this_var)
            setattr(self,this_var,this_product)
        
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
        for var_name in self.track_vars: # Reset the history of tracked variables
            setattr(self,var_name + '_hist',[])
        for var_name in self.sow_vars: # Set the sow variables to their initial levels
            initial_val = getattr(self,var_name + '_init')
            setattr(self,var_name,initial_val)
        for this_type in self.agents: # Reset each AgentType in the market
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
            value_now = getattr(self,var_name)
            getattr(self,var_name + '_hist').append(value_now)
        
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
        self.reset() # Initialize the state of the market
        for t in range(self.act_T):
            self.sow()       # Distribute aggregated information/state to agents
            self.cultivate() # Agents take action
            self.reap()      # Collect individual data from agents
            self.mill()      # Process individual data into aggregate data
            self.store()     # Record variables of interest
            
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
        for name in self.track_vars:
            history_vars_string += ' \'' + name + '\' : self.' + name + '_hist,'
        update_dict = eval('{' + history_vars_string + '}')
        
        # Calculate a new dynamic rule and distribute it to the agents in agent_list
        dynamics = self.calcDynamics(**update_dict) # User-defined dynamics calculator
        for var_name in self.dyn_vars:
            this_obj = getattr(dynamics,var_name)
            for this_type in self.agents:
                setattr(this_type,var_name,this_obj)
        return dynamics
        
if __name__ == '__main__':
    print("Sorry, HARKcore doesn't actually do anything on its own.")
    print("To see some examples of its frameworks in action, try running a model module.")
    print("Several interesting model modules can be found in /ConsumptionSavingModel.")
    print('For an extraordinarily simple model that demonstrates the "microeconomic" and')
    print('"macroeconomic" frameworks, see /FashionVictim/FashionVictimModel.')
