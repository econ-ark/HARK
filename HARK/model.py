"""
Tools for crafting models.
"""

from dataclasses import dataclass, field
from HARK.distribution import Distribution


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
    blocks: list[DBlock] = field(default_factory=list)

    def get_shocks(self):
        ### TODO: Bug in here is causing AttributeError: 'set' object has no attribute 'draw'

        super_shocks = {}  # uses set to avoid duplicates

        for b in self.blocks:
            for k, v in b.shocks.items():  # use d.iteritems() in python 2
                super_shocks[k] = v

        return super_shocks

    def get_dynamics(self):
        super_dyn = {}  # uses set to avoid duplicates

        for b in self.blocks:
            for k, v in b.dynamics.items():  # use d.iteritems() in python 2
                super_dyn[k] = v

        return super_dyn

    def get_vars(self):
        return list(self.get_shocks().keys()) + list(self.get_dynamics().keys())
