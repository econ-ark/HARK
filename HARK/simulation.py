'''
Currently empty. Will be used for future simulation handling code.
'''

from __future__ import division
import warnings                             # A library for runtime warnings
from HARK import AgentType
import networkx as nx
import numpy as np                          # Numerical Python


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

    params = {
        'G' : 1.1, # income growth factor
        'R' : 1.1  # rate of return
        }

    states = {
        'b' : lambda a_, R: a_ * R,
        'm' : lambda p_, b : p_ + b, # within-period assets
        }

    controls = {
        'c' : lambda m : m / 3 # consumption.
        # The function is the decision rule.
        }
    post_states = {
        'a' : lambda m, c : m - c, #market assets
        'p' : lambda p_, G: p_ * G # income
    }
    
    initial_states = {
        'p_' : 1,
        'a_' : 1
        }


    ####
    # Kludges for compatibility

    T_sim = 10
    T_cycle = 0
    AgentCount = 1
    solution = "No solution"
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
        self.agent.states.update(self.params)

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
    history = {}
    states = {}
    controls = {}
    post_states = {}

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

def decrement(var_name):
    '''
    Gets the variable name for this state variable,
    but for the previous time step.
    '''
    return var_name + '_'

def simulation_order(transitions):
    parents = {
        v : transitions[v].__code__.co_varnames
        for v
        in transitions
    }
    
    order = list(nx.topological_sort(nx.DiGraph(parents).reverse()))

    return [s for s in order if s in transitions]

def call_function_in_scope(function, context):
    return function(
        *[
            context[name]
            for name
            in function.__code__.co_varnames
        ])


################
# TESTING CODE
# To be moved into examples and/or test suite
# once stabilized.
################

generic_model_test = GenericModel()
generic_model_test.initializeSim()
generic_model_test.simulate()

print('a_ : ', generic_model_test.agent.history['a'])
print('c : ', generic_model_test.agent.history['c'])
print('p : ', generic_model_test.agent.history['p'])
