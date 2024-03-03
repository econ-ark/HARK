from dataclasses import dataclass, field
from typing import Mapping, Optional, Union
from warnings import warn

import numpy as np
import xarray as xr
from yaml import SafeLoader, YAMLObject

rng = np.random.default_rng()


@dataclass
class Variable(YAMLObject):
    """
    Abstract class for representing variables. Variables are the building blocks
    of models. They can be parameters, states, actions, shocks, or auxiliaries.
    """

    name: str  # The name of the variable, required
    attrs: dict = field(default_factory=dict, kw_only=True)
    short_name: str = field(default=None, kw_only=True)
    long_name: str = field(default=None, kw_only=True)
    latex_repr: str = field(default=None, kw_only=True)
    yaml_tag: str = field(default="!Variable", kw_only=False)
    yaml_loader = SafeLoader

    def __post_init__(self):
        for key in ["long_name", "short_name", "latex_repr"]:
            self.attrs.setdefault(key, None)
        self.name = self.name.strip()
        if not self.name:
            raise ValueError("Empty variable name")

    @classmethod
    def from_yaml(cls, loader, node):
        fields = loader.construct_mapping(node, deep=True)
        return cls(**fields)

    def __repr__(self):
        """
        String representation of the variable.

        Returns:
        str: The string representation of the variable.
        """
        return f"{self.__class__.__name__}({self.name})"


@dataclass
class VariableSpace(YAMLObject):
    """
    Abstract class for representing a collection of variables.
    """

    variables: list[Variable]
    yaml_tag: str = field(default="!VariableSpace", kw_only=True)
    yaml_loader = SafeLoader

    def __post_init__(self):
        """
        Save the variables in a dictionary for easy access.
        """
        self.variables = {var.name: var for var in self.variables}

    @classmethod
    def from_yaml(cls, loader, node):
        fields = loader.construct_mapping(node, deep=True)
        return cls(**fields)


@dataclass(kw_only=True)
class Parameter(Variable):
    """
    A `Parameter` is a variable that has a fixed value.
    """

    value: Union[int, float] = 0
    yaml_tag: str = "!Parameter"
    yaml_loader = SafeLoader

    def __repr__(self):
        """
        String representation of the parameter.

        Returns:
        str: The string representation of the parameter.
        """
        return f"{self.__class__.__name__}({self.name}, {self.value})"


@dataclass
class Parameters(VariableSpace):
    """
    A `Parameters` is a collection of parameters.
    """

    yaml_tag: str = "!Parameters"


@dataclass
class Auxiliary(Variable):
    """
    Class for representing auxiliaries. Auxiliaries are abstract variables that
    have an array structure but are not states, actions, or shocks. They may
    include information like domain, measure (discrete or continuous), etc.
    """

    array: Union[list, np.ndarray, xr.DataArray] = None
    domain: Union[list, tuple] = field(default=None, kw_only=True)
    is_discrete: bool = field(default=False, kw_only=True)
    yaml_tag: str = field(default="!Auxiliary", kw_only=True)


@dataclass
class AuxiliarySpace(VariableSpace):
    """
    A `AuxiliarySpace` is a collection of auxiliary variables.
    """

    yaml_tag: str = "!AuxiliarySpace"


@dataclass(kw_only=True)
class State(Auxiliary):
    """
    Class for representing a state variable.
    """

    yaml_tag: str = "!State"

    def assign_values(self, values):
        return make_state_array(values, self.name, self.attrs)

    def discretize(self, min, max, N, method):
        # linear for now
        self.assign_values(np.linspace(min, max, N))


@dataclass(kw_only=True)
class StateSpace(AuxiliarySpace):
    states: Mapping[str, State] = field(init=False)
    yaml_tag: str = "!StateSpace"

    def __post_init__(self):
        super().__post_init__()
        self.states = self.variables


@dataclass(kw_only=True)
class PostState(State):
    yaml_tag: str = "!PostState"


@dataclass(kw_only=True)
class PostStateSpace(StateSpace):
    post_states: Mapping[str, State] = field(init=False)
    yaml_tag: str = "!PostStateSpace"

    def __post_init__(self):
        super().__post_init__()
        self.post_states = self.variables


@dataclass(kw_only=True)
class Action(Auxiliary):
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


@dataclass(kw_only=True)
class ActionSpace(AuxiliarySpace):
    actions: Mapping[str, State] = field(init=False)
    yaml_tag: str = "!ActionSpace"

    def __post_init__(self):
        super().__post_init__()
        self.actions = self.variables


@dataclass(kw_only=True)
class Shock(Variable):
    """
    Class for representing shocks. Shocks are variables that are not
    chosen by the agent.
    Can also be called a random variable, or a state variable.

    Args:
        Variable (_type_): _description_
    """

    yaml_tag: str = "!Shock"


@dataclass(kw_only=True)
class ShockSpace(VariableSpace):
    shocks: list[Shock]
    yaml_tag: str = "!ShockSpace"

    def __post_init__(self):
        super().__post_init__()
        self.shocks = self.variables


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
    States: An xarray Dataset representing the states.
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

    return xr.merge([states])
