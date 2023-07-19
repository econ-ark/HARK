from dataclasses import dataclass, field
from typing import Optional, Union
from warnings import warn

import numpy as np
import xarray as xr
from yaml import YAMLObject

rng = np.random.default_rng()


@dataclass
class Variable(YAMLObject):
    name: str  # The name of the variable, required
    domain: Union[list, tuple]  # The domain of the state, required
    is_discrete: bool = False  # Whether the state is continuous or discrete, optional
    attrs: dict = field(default_factory=dict)  # Additional attributes, optional
    yaml_tag: str = "!Variable"  # The YAML tag for the variable, required

    def __post_init__(self):
        for key in ["long_name", "short_name", "latex_repr"]:
            self.attrs.setdefault(key, self.name)


@dataclass
class VariableSpace(YAMLObject):
    variables: list[Variable]
    yaml_tag: str = "!VariableSpace"


@dataclass
class State(Variable):
    yaml_tag: str = "!State"

    def assign_values(self, values):
        return make_state_array(values, self.name, self.attrs)

    def discretize(self, min, max, N, method):
        # linear for now
        self.assign_values(np.linspace(min, max, N))


@dataclass
class StateSpace(VariableSpace):
    states: list[State]
    yaml_tag: str = "!StateSpace"

    def __post_init__(self):
        self.states = xr.merge(self.states)


@dataclass
class Action(Variable):
    """
    Class for representing actions. Actions are variables that are chosen by the agent.
    Can also be called a choice, control, decision, or a policy.

    Args:
        Variable (_type_): _description_
    """

    is_optimal: bool = True
    yaml_tag: str = "!Action"

    def discretize(self, *args, **kwargs):
        warn("Actions cannot be discretized.")


@dataclass
class ActionSpace:
    actions: list[Action]
    yaml_tag: str = "!ActionSpace"

    def __post_init__(self):
        self.actions = xr.merge(self.actions)


@dataclass
class Shock(Variable):
    """
    Class for representing shocks. Shocks are variables that are not
    chosen by the agent.
    Can also be called a random variable, or a state variable.

    Args:
        Variable (_type_): _description_
    """

    yaml_tag: str = "!Shock"


@dataclass
class ShockSpace:
    shocks: list[Shock]
    yaml_tag: str = "!ShockSpace"

    def __post_init__(self):
        self.shocks = xr.merge(self.shocks)


@dataclass
class Auxilliary(Variable):
    yaml_tag: str = "!Auxilliary"


def make_state_array(
    values: np.ndarray,
    name: Optional[str] = None,
    attrs: Optional[dict] = None,
) -> xr.Dataset:
    """
    Function to create a state with given values, name and attrs.

    Parameters:
    values (np.ndarray): The values for the state.
    name (str, optional): The name of the state. Defaults to 'state'.
    attrs (dict, optional): The attrs for the state. Defaults to None.

    Returns:
    State: An xarray DataArray representing the state.
    """
    # Use a default name only when no name is provided
    name = name or f"state{rng.integers(0, 100)}"
    attrs = attrs or {}

    return xr.Dataset(
        {
            name: xr.DataArray(
                values,
                name=name,
                dims=(name,),
                attrs=attrs,
            )
        }
    )


def make_states_array(
    values: Union[np.ndarray, list],
    names: Optional[list[str]] = None,
    attrs: Optional[list[dict]] = None,
) -> xr.Dataset:
    """
    Function to create states with given values, names and attrs.

    Parameters:
    values (Union[np.ndarray, States]): The values for the states.
    names (list[str], optional): The names of the states. Defaults to None.
    attrs (list[dict], optional): The attrs for the states. Defaults to None.

    Returns:
    States: An arrayray Dataset representing the states.
    """
    if isinstance(values, list):
        values_len = len(values)
    elif isinstance(values, np.ndarray):
        values_len = values.shape[0]

    # Use default names and attrs only when they are not provided
    names = names or [f"state{rng.integers(0, 100)}" for _ in range(values_len)]
    attrs = attrs or [{}] * values_len

    states = [
        make_state_array(value, name, attr)
        for value, name, attr in zip(values, names, attrs)
    ]

    return xr.merge(states)
