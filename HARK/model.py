"""
Tools for crafting models.
"""

from dataclasses import dataclass, field
from HARK.distribution import (
    Distribution,
    DiscreteDistributionLabeled,
    combine_indep_dstns,
    expected,
)
from inspect import signature
import numpy as np
from typing import Any, Callable, Mapping, List, Union


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


def discretized_shock_dstn(shocks, disc_params):
    """
    Discretizes a collection of independent shocks and combines
    them into one DiscreteDistributionLabeled.

    Shocks are discretized only if they have a corresponding
    element of disc_params defined.

    Parameters
    -----------
    shocks: dict of Distribution
        A dictionary of Distributions, representing independent exogenous shocks.

    disc_params: dict of dict
        A dictionary of dictionaries with arguments to Distribution.discretize.
        Keys of this dictionary should be shared with the shocks argument.
    """
    dshocks = {}

    for shockn in shocks:
        if shockn == "live":  # hacky hack
            pass
        elif shockn in disc_params:
            dshocks[shockn] = DiscreteDistributionLabeled.from_unlabeled(
                shocks[shockn].discretize(**disc_params[shockn]), var_names=[shockn]
            )
        else:
            # assume already discrete
            dshocks[shockn] = DiscreteDistributionLabeled.from_unlabeled(
                shocks[shockn], var_names=[shockn]
            )

    all_shock_dstn = combine_indep_dstns(*dshocks.values())

    return all_shock_dstn


def simulate_dynamics(
    dynamics: Mapping[str, Union[Callable, Control]],
    pre: Mapping[str, Any],
    dr: Mapping[str, Callable],
):
    """
    From the beginning-of-period state (pre), follow the dynamics,
    including any decision rules, to compute the end-of-period state.

    Parameters
    ------------

    dynamics: Mapping[str, Callable]
        Maps variable names to functions from variables to values.
        Can include Controls
        ## TODO: Make collection of equations into a named type


    pre : Mapping[str, Any]
        Bound values for all variables that must be known before beginning the period's dynamics.


    dr : Mapping[str, Callable]
        Decision rules for all the Control variables in the dynamics.
    """
    vals = pre.copy()

    for varn in dynamics:
        # Using the fact that Python dictionaries are ordered

        feq = dynamics[varn]

        if isinstance(feq, Control):
            # This tests if the decision rule is age varying.
            # If it is, this will be a vector with the decision rule for each agent.
            if isinstance(dr[varn], np.ndarray):
                ## Now we have to loop through each agent, and apply the decision rule.
                ## This is quite slow.
                for i in range(dr[varn].size):
                    vals_i = {
                        var: vals[var][i]
                        if isinstance(vals[var], np.ndarray)
                        else vals[var]
                        for var in vals
                    }
                    vals[varn][i] = dr[varn][i](
                        *[vals_i[var] for var in signature(dr[varn][i]).parameters]
                    )
            else:
                vals[varn] = dr[varn](
                    *[vals[var] for var in signature(dr[varn]).parameters]
                )  # TODO: test for signature match with Control
        else:
            vals[varn] = feq(*[vals[var] for var in signature(feq).parameters])

    return vals


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

    def get_controls(self):
        """
        TODO: Repeated in RBlock. Move to higher order class.
        """
        dyn = self.get_dynamics()

        return [varn for varn in dyn if isinstance(dyn[varn], Control)]

    def transition(self, pre, dr):
        """
        Returns variable values given previous values and decision rule for all controls.
        """
        return simulate_dynamics(self.dynamics, pre, dr)

    def calc_reward(self, vals):
        """
        Computes the reward for a given set of variable values
        """
        rvals = {}

        for varn in self.reward:
            feq = self.reward[varn]
            rvals[varn] = feq(*[vals[var] for var in signature(feq).parameters])

        return rvals

    def get_state_rule_value_function_from_continuation(self, continuation):
        """
        Given a continuation value function, returns a state-rule value
        function: the value for each state and decision rule.
        This value includes both the reward for executing the rule
        'this period', and the continuation value of the resulting states.
        """

        def state_rule_value_function(pre, dr):
            vals = self.transition(pre, dr)
            r = list(self.calc_reward(vals).values())[0]  # a hack; to be improved
            cv = continuation(
                *[vals[var] for var in signature(continuation).parameters]
            )

            return r + cv

        return state_rule_value_function

    def get_decision_value_function(self, dr, continuation):
        """
        Given a decision rule and a continuation value function,
        return a function for the value at the decision step/tac,
        after the shock have been realized.
        """
        srvf = self.get_state_rule_value_function_from_continuation(continuation)

        def decision_value_function(shpre):
            return srvf(shpre, dr)

        return decision_value_function

    def get_arrival_value_function(self, disc_params, dr, continuation):
        """
        Returns an arrival value function, which is the value of the states
        upon arrival into the block.

        This involves taking an expectation over shocks (which must
        first be discretized), a decision rule, and a continuation
        value function.)
        """

        def arrival_value_function(arvs):
            dvf = self.get_decision_value_function(dr, continuation)

            ds = discretized_shock_dstn(self.shocks, disc_params)

            arvs_args = [arvs[avn] for avn in arvs]

            def mod_dvf(shock_value_array):
                shockvs = {
                    shn: shock_value_array[shn]
                    for i, shn in enumerate(list(ds.variables.keys()))
                }

                dvf_args = {}
                dvf_args.update(arvs)
                dvf_args.update(shockvs)

                return dvf(dvf_args)

            return expected(func=mod_dvf, dist=ds)

        return arrival_value_function


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

    def get_controls(self):
        dyn = self.get_dynamics()

        return [varn for varn in dyn if isinstance(dyn[varn], Control)]

    def get_dynamics(self):
        super_dyn = {}  # uses set to avoid duplicates

        for b in self.blocks:
            for k, v in b.get_dynamics().items():  # use d.iteritems() in python 2
                super_dyn[k] = v

        return super_dyn

    def get_vars(self):
        return list(self.get_shocks().keys()) + list(self.get_dynamics().keys())
