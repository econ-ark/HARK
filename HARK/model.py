"""
Tools for crafting models.
"""

from dataclasses import dataclass, field
from HARK.distribution import Distribution
from HARK.simulation.monte_carlo import simulate_dynamics
from inspect import signature
from typing import List


class Aggregate:
    """
    Used to designate a shock as an aggregate shock.
    If so designated, draws from the shock will be scalar rather
    than array valued.
    """

    def __init__(self, dist: Distribution):
        self.dist = dist


class Control:
    """
    Used to designate a variabel that is a control variable.

    Parameters
    ----------
    args : list of str
        The labels of the variables that are in the information set of this control.
    """

    def __init__(self, args):
        pass


@dataclass
class DBlock:
    """
    Represents a 'block' of model behavior.
    It prioritizes a representation of the dynamics of the block.
    Control variables are designated by the appropriate dynamic rule.

    Parameters
    ----------
    ...
    """

    name: str = ""
    description: str = ""
    shocks: dict = field(default_factory=dict)
    dynamics: dict = field(default_factory=dict)
    reward: dict = field(default_factory=dict)

    def get_shocks(self):
        return self.shocks

    def get_dynamics(self):
        return self.dynamics

    def get_vars(self):
        return list(self.shocks.keys()) + list(self.dynamics.keys())

    def transition(self, pre, dr):
        """
        Returns variable values given previous values and decision rule for all controls.
        """
        return simulate_dynamics(self.dynamics, pre, dr)

    def reward(self, vals):
        """
        Computes the reward for a given set of variable values
        """
        vals = {}

        for varn in self.reward:
            feq = self.reward[varn]
            vals[varn] = feq(*[vals[var] for var in signature(feq).parameters])

        return vals

    def state_action_value_function_from_continuation(self, continuation):
        def state_action_value(pre, dr):
            vals = self.transition(pre, dr)
            r = list(self.reward(vals))[0] # a hack; to be improved
            cv = continuation(*[vals[var] for var in signature(continuation).parameters])

            return r + cv

        return state_action_value

    def decision_value_function(self, dr, continuation):
        savf = self.state_action_value_function_from_continuation(continuation)

        def decision_value_function(pre):
            return savf(pre, dr)

        return decision_value_function

    def arrival_value_function(self, dr, continuation):
        """
        Value of arrival states, prior to shocks, given a decision rule and continuation.
        """
        def arrival_value(arrv):
            dvf = self.decision_value_function(dr, continuation)

            ##TOD: Take expectation over shocks!!!
            return EXPECTATION(dvf, shock_vals, arrv)



@dataclass
class RBlock:
    """
    A recursive block.

    Parameters
    ----------
    ...
    """

    name: str = ""
    description: str = ""
    blocks: List[DBlock] = field(default_factory=list)

    def get_shocks(self):
        ### TODO: Bug in here is causing AttributeError: 'set' object has no attribute 'draw'

        super_shocks = {}  # uses set to avoid duplicates

        for b in self.blocks:
            for k, v in b.get_shocks().items():  # use d.iteritems() in python 2
                super_shocks[k] = v

        return super_shocks

    def get_dynamics(self):
        super_dyn = {}  # uses set to avoid duplicates

        for b in self.blocks:
            for k, v in b.get_dynamics().items():  # use d.iteritems() in python 2
                super_dyn[k] = v

        return super_dyn

    def get_vars(self):
        return list(self.get_shocks().keys()) + list(self.get_dynamics().keys())
