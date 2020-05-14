'''
Generic simulation.
'''

from __future__ import division
import warnings                             # A library for runtime warnings
from HARK import AgentType
from HARK.distribution import MeanOneLogNormal
import networkx as nx
import numpy as np                          # Numerical Python


#####
# A Sample Configuration
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
    'post_states' : {
        'a' : lambda m, c : m - c, #market assets
        'p' : lambda p_, G: p_ * G # income
    },
    'initial_states' : {
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
    
    #   simBirth
    #     - sets initial values for agents
    # simeOnePeriod methods
    #   getMortality
    #   getShocks

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
    NOT and AgentType.
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
    '''
    return var_name + '_'

def evaluate(process):
    ## clean this up; should be a type check
    ## for Distribution, or Distribution should
    ## have a cleaner interface (like __call__)
    if hasattr(process, 'draw'):
        return process.draw(1)[0]
    else:
        return process


def simulation_order(transitions):
    parents = {
        v : transitions[v].__code__.co_varnames
        for v
        in transitions
    }
    
    order = list(nx.topological_sort(nx.DiGraph(parents).reverse()))

    return [s for s in order if s in transitions]


################
# TESTING CODE
# To be moved into examples and/or test suite
# once stabilized.
################

#generic_model_test = GenericModel(sample_configuration)
#generic_model_test.initializeSim()
#generic_model_test.simulate()

#print('a_ : ', generic_model_test.agent.history['a'])
#print('c : ', generic_model_test.agent.history['c'])
#print('p : ', generic_model_test.agent.history['p'])
