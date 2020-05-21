'''
Generic model simulation.

This module contains classes and functions that will
simulate any model that is passed to it as a configuration
object.

The configuration object is intended to match, as much as
possible, the mathematical specification of the model.

The simulator computes results over time based on the
model specification. This can involve:
 - Saving time-varying state
 - Using a decision rule to choose control variables
 - Sampling from exogenous shock distributions
 - Determining the resolution order based on element-wise
   transition functions
'''

from __future__ import division
import warnings                             # A library for runtime warnings
from HARK import AgentType
from HARK.distribution import MeanOneLogNormal
import networkx as nx
import numpy as np                          # Numerical Python

#####
# A Sample Configuration
#
# A template demonstrating the generic configuration object.
#
#####
sample_configuration = {
    'params' : {
        'G' : MeanOneLogNormal(), # income growth SHOCK
        'R' : 1.1  # rate of return
    },
    'states' : {
        'b' : lambda a_, R: a_ * R,
        'm' : lambda p_, b : p_ + b, # within-period assets
    },
    'controls' : {
        'c' : lambda m : m / 3 # consumption.
        # The function is the decision rule.
    },
    'post_states' : { # These values match the initial states below
        'a' : lambda m, c : m - c, #market assets
        'p' : lambda p_, G: p_ * G # income
    },
    'initial_states' : { # Starting values for each agent
        'p_' : 1,
        'a_' : 1
    }
}

class GenericModel(AgentType):
    '''
    A partially implemented AgentType class that is generic.
    An instance is configured with:
       - state variables
       - control variables
       - shocks
       - transition functions

    This class will be oriented towards simulation first.
    Later versions may support generic solvers.
    '''
    params = {}
    states = {}
    controls = {}
    post_states = {}
    initial_states = {}

    def __init__(self, configuration):
        '''
        Initialize an instance of AgentType by setting attributes.

        Parameters
        ----------
        configuration : dictionary
            A dictionary representing a model.
            Must contain 'params', 'states', 'controls', 
            'post_states', and 'initial_states' as keys,
            with dictionary values.
            See sample_configuration for example.
        '''
        self.params = configuration['params']
        self.states = configuration['states']
        self.controls = configuration['controls']
        self.post_states = configuration['post_states']
        self.initial_states = configuration['initial_states']

    ####
    # Kludges for compatibility
    AgentCount = 1
    read_shocks = False
    seed = 0
    solution = "No solution"
    T_cycle = 0
    T_sim = 200
    poststate_vars = []
    track_vars = []
    who_dies_hist = []
    ####


    # initializeSim

    def simBirth(self, which_agents):
        '''
        New consumer.
        TODO: handle multiple consumers?
        '''

        if which_agents[0]:
            self.agent = SimulatedAgent()

            self.agent.states = self.initial_states.copy()
            self.agent.controls = self.controls.copy()
            self.agent.post_states = self.post_states.copy()

    def getStates(self):
        
        self.agent.states.update({
            p : evaluate(self.params[p])
            for p
            in self.params})

        for variable in simulation_order(self.states):
            if variable in self.states:
                self.agent.update_state(
                    variable,
                    call_function_in_scope(
                        self.states[variable],
                        self.agent.states
                    )
                )

    def getControls(self):
        for variable in simulation_order(self.controls):
            self.agent.update_control(
                variable,
                call_function_in_scope(
                    self.controls[variable],
                    self.agent.states
                )
            )

    def getPostStates(self):
        context = self.agent.states.copy()
        context.update(self.agent.controls)
        
        for variable in simulation_order(self.post_states):    
            self.agent.update_post_state(
                variable,
                call_function_in_scope(
                    self.post_states[variable],
                    context
                )
            )

class SimulatedAgent():
    '''
    NOT an AgentType.
    Rather, something that stores a particular agent's
    state, age, etc. in a simulation.
    '''
    
    def __init__(self):
        self.history = {}
        self.states = {}
        self.controls = {}
        self.post_states = {}

        
    def update_history(self, variable, value):
        if variable not in self.history:
            self.history[variable] = []

        self.history[variable].append(value)
    
    def update_state(self, variable, value):
        self.states[variable] = value
        self.update_history(variable, value)

    def update_control(self, variable, value):
        self.controls[variable] = value
        self.update_history(variable, value)

    def update_post_state(self, variable, value):
        self.post_states[variable] = value
        self.states[decrement(variable)] = value
        self.update_history(variable, value)


def call_function_in_scope(function, context):
    '''
    A helper function.
    Calls FUNCTION with arguments bound, by name,
    to the values in dictionary, CONTEXT.
    '''
    return function(
        *[
            context[name]
            for name
            in function.__code__.co_varnames
        ])

def decrement(var_name):
    '''
    Gets the variable name for this state variable,
    but for the previous time step.

    Parameters
    ----------
    var_name : str
        A variable name

    Returns
    -------
    decremented_var_name : str
        A 'decremented' version of this variable.
    '''
    return var_name + '_'

def evaluate(process):
    '''
    Evalautes the value of a process.

    Parameters
    ----------
    process : int or Distribution
        A process.
        If an int, returns value from the int.
        If a Distribution, draws one value from the distribution

    Returns
    -------
    value : float
        The value of the process at this time.
    '''
    ## clean this up; should be a type check
    ## for Distribution, or Distribution should
    ## have a cleaner interface (like __call__)
    if hasattr(process, 'draw'):
        return process.draw(1)[0]
    else:
        return process


def simulation_order(transitions):
    '''
    Given a dictionary of model (state) variables 
    and transtion functions,
    return a list representing the order in which the
    new variable values must be computed.

    Parameters
    ----------
    transitions : {str : function}
        A dictionary of transition functions.
    Returns
    -------
    order : [str]
        A list of strs, indicating the reverse topological
        order of variables with respect to their transition
        functions
    '''
    parents = {
        v : transitions[v].__code__.co_varnames
        for v
        in transitions
    }
    
    order = list(nx.topological_sort(nx.DiGraph(parents).reverse()))

    return [s for s in order if s in transitions]
