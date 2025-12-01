"""
A module with classes and functions for automated simulation of HARK.AgentType
models from a human- and machine-readable model specification.
"""

from dataclasses import dataclass, field
from copy import copy, deepcopy
import numpy as np
from numba import njit
from sympy.utilities.lambdify import lambdify
from sympy import symbols, IndexedBase
from typing import Callable
from HARK.utilities import NullFunc, make_exponential_grid
from HARK.distributions import Distribution
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from itertools import product
import importlib.resources
import yaml

# Prevent pre-commit from removing sympy
x = symbols("x")
del x
y = IndexedBase("y")
del y


@dataclass(kw_only=True)
class ModelEvent:
    """
    Class for representing "events" that happen to agents in the course of their
    model. These might be statements of dynamics, realization of a random shock,
    or the evaluation of a function (potentially a control or other solution-
    based object). This is a superclass for types of events defined below.

    Parameters
    ----------
    description : str
        Text description of this model event.
    statement : str
        The line of the model statement that this event corresponds to.
    parameters : dict
        Dictionary of objects that are static / universal within this event.
    assigns : list[str]
        List of names of variables that this event assigns values for.
    needs : list[str]
        List of names of variables that this event requires to be run.
    data : dict
        Dictionary of current variable values within this event.
    common : bool
        Indicator for whether the variables assigned in this event are commonly
        held across all agents, rather than idiosyncratic.
    N : int
        Number of agents currently in this event.
    """

    statement: str = field(default="")
    parameters: dict = field(default_factory=dict)
    description: str = field(default="")
    assigns: list[str] = field(default_factory=list, repr=False)
    needs: list = field(default_factory=list, repr=False)
    data: dict = field(default_factory=dict, repr=False)
    common: bool = field(default=False, repr=False)
    N: int = field(default=1, repr=False)

    def run(self):
        """
        This method should be filled in by each subclass.
        """
        pass  # pragma: nocover

    def reset(self):
        self.data = {}

    def assign(self, output):
        if len(self.assigns) > 1:
            assert len(self.assigns) == len(output)
            for j in range(len(self.assigns)):
                var = self.assigns[j]
                if type(output[j]) is not np.ndarray:
                    output[j] = np.array([output[j]])
                self.data[var] = output[j]
        else:
            var = self.assigns[0]
            if type(output) is not np.ndarray:
                output = np.array([output])
            self.data[var] = output

    def expand_information(self, origins, probs, atoms, which=None):
        """
        This method is only called internally when a RandomEvent or MarkovEvent
        runs its quasi_run() method. It expands the set of of "probability blobs"
        by applying a random realization event. All extant blobs for which the
        shock applies are replicated for each atom in the random event, with the
        probability mass divided among the replicates.

        Parameters
        ----------
        origins : np.array
            Array that tracks which arrival state space node each blob originated
            from. This is expanded into origins_new, which is returned.
        probs : np.array
            Vector of probabilities of each of the random possibilities.
        atoms : [np.array]
            List of arrays with realization values for the distribution. Each
            array corresponds to one variable that is assigned by this event.
        which : np.array or None
            If given, a Boolean array indicating which of the pre-existing blobs
            is affected by the given probabilities and atoms. By default, all
            blobs are assumed to be affected.

        Returns
        -------
        origins_new : np.array
            Expanded boolean array of indicating the arrival state space node that
            each blob originated from.
        """
        K = probs.size
        N = self.N
        if which is None:
            which = np.ones(N, dtype=bool)
        other = np.logical_not(which)
        M = np.sum(which)  # how many blobs are we affecting?
        MX = N - M  # how many blobs are we not affecting?

        # Update probabilities of outcomes
        pmv_old = np.reshape(self.data["pmv_"][which], (M, 1))
        pmv_new = (pmv_old * np.reshape(probs, (1, K))).flatten()
        self.data["pmv_"] = np.concatenate((self.data["pmv_"][other], pmv_new))

        # Replicate the pre-existing data for each atom
        for var in self.data.keys():
            if (var == "pmv_") or (var in self.assigns):
                continue  # don't double expand pmv, and don't touch assigned variables
            data_old = np.reshape(self.data[var][which], (M, 1))
            data_new = np.tile(data_old, (1, K)).flatten()
            self.data[var] = np.concatenate((self.data[var][other], data_new))

        # If any of the assigned variables don't exist yet, add dummy versions
        # of them. This section exists so that the code works with "partial events"
        # on both the first pass and subsequent passes.
        for j in range(len(self.assigns)):
            var = self.assigns[j]
            if var in self.data.keys():
                continue
            self.data[var] = np.zeros(N, dtype=atoms[j].dtype)
            # Zeros are just dummy values

        # Add the new random variables to the simulation data. This generates
        # replicates for the affected blobs and leaves the others untouched,
        # still with their dummy values. They will be altered on later passes.
        for j in range(len(self.assigns)):
            var = self.assigns[j]
            data_new = np.tile(np.reshape(atoms[j], (1, K)), (M, 1)).flatten()
            self.data[var] = np.concatenate((self.data[var][other], data_new))

        # Expand the origins array to account for the new replicates
        origins_new = np.tile(np.reshape(origins[which], (M, 1)), (1, K)).flatten()
        origins_new = np.concatenate((origins[other], origins_new))
        self.N = MX + M * K

        # Send the new origins array back to the calling process
        return origins_new

    def add_idiosyncratic_bernoulli_info(self, origins, probs):
        """
        Special method for adding Bernoulli outcomes to the information set when
        probabilities are idiosyncratic to each agent. All extant blobs are duplicated
        with the appropriate probability

        Parameters
        ----------
        origins : np.array
            Array that tracks which arrival state space node each blob originated
            from. This is expanded into origins_new, which is returned.
        probs : np.array
            Vector of probabilities of drawing True for each blob.

        Returns
        -------
        origins_new : np.array
            Expanded boolean array of indicating the arrival state space node that
            each blob originated from.
        """
        N = self.N

        # # Update probabilities of outcomes, replicating each one
        pmv_old = np.reshape(self.data["pmv_"], (N, 1))
        P = np.reshape(probs, (N, 1))
        PX = np.concatenate([1.0 - P, P], axis=1)
        pmv_new = (pmv_old * PX).flatten()
        self.data["pmv_"] = pmv_new

        # Replicate the pre-existing data for each atom
        for var in self.data.keys():
            if (var == "pmv_") or (var in self.assigns):
                continue  # don't double expand pmv, and don't touch assigned variables
            data_old = np.reshape(self.data[var], (N, 1))
            data_new = np.tile(data_old, (1, 2)).flatten()
            self.data[var] = data_new

        # Add the (one and only) new random variable to the simulation data
        var = self.assigns[0]
        data_new = np.tile(np.array([[0, 1]]), (N, 1)).flatten()
        self.data[var] = data_new

        # Expand the origins array to account for the new replicates
        origins_new = np.tile(np.reshape(origins, (N, 1)), (1, 2)).flatten()
        self.N = N * 2

        # Send the new origins array back to the calling process
        return origins_new


@dataclass(kw_only=True)
class DynamicEvent(ModelEvent):
    """
    Class for representing model dynamics for an agent, consisting of an expression
    to be evaluated and variables to which the results are assigned.

    Parameters
    ----------
    expr : Callable
        Function or expression to be evaluated for the assigned variables.
    args : list[str]
        Ordered list of argument names for the expression.
    """

    expr: Callable = field(default_factory=NullFunc, repr=False)
    args: list[str] = field(default_factory=list, repr=False)

    def evaluate(self):
        temp_dict = self.data.copy()
        temp_dict.update(self.parameters)
        args = (temp_dict[arg] for arg in self.args)
        out = self.expr(*args)
        return out

    def run(self):
        self.assign(self.evaluate())

    def quasi_run(self, origins, norm=None):
        self.run()
        return origins


@dataclass(kw_only=True)
class RandomEvent(ModelEvent):
    """
    Class for representing the realization of random variables for an agent,
    consisting of a shock distribution and variables to which the results are assigned.

    Parameters
    ----------
    dstn : Distribution
        Distribution of one or more random variables that are drawn from during
        this event and assigned to the corresponding variables.
    """

    dstn: Distribution = field(default_factory=Distribution, repr=False)

    def reset(self):
        self.dstn.reset()
        ModelEvent.reset(self)

    def draw(self):
        out = np.empty((len(self.assigns), self.N))
        if not self.common:
            out[:, :] = self.dstn.draw(self.N)
        else:
            out[:, :] = self.dstn.draw(1)
        if len(self.assigns) == 1:
            out = out.flatten()
        return out

    def run(self):
        self.assign(self.draw())

    def quasi_run(self, origins, norm=None):
        # Get distribution
        atoms = self.dstn.atoms
        probs = self.dstn.pmv.copy()

        # Apply Harmenberg normalization if applicable
        try:
            harm_idx = self.assigns.index(norm)
            probs *= atoms[harm_idx]
        except:
            pass

        # Expand the set of simulated blobs
        origins_new = self.expand_information(origins, probs, atoms)
        return origins_new


@dataclass(kw_only=True)
class RandomIndexedEvent(RandomEvent):
    """
    Class for representing the realization of random variables for an agent,
    consisting of a list of shock distributions, an index for the list, and the
    variables to which the results are assigned.

    Parameters
    ----------
    dstn : [Distribution]
        List of distributions of one or more random variables that are drawn
        from during this event and assigned to the corresponding variables.
    index : str
        Name of the index that is used to choose a distribution for each agent.
    """

    index: str = field(default="", repr=False)
    dstn: list[Distribution] = field(default_factory=list, repr=False)

    def draw(self):
        idx = self.data[self.index]
        K = len(self.assigns)
        out = np.empty((K, self.N))
        out.fill(np.nan)

        if self.common:
            k = idx[0]  # this will behave badly if index is not itself common
            out[:, :] = self.dstn[k].draw(1)
            return out

        for k in range(len(self.dstn)):
            these = idx == k
            if not np.any(these):
                continue
            out[:, these] = self.dstn[k].draw(np.sum(these))
        if K == 1:
            out = out.flatten()
        return out

    def reset(self):
        for k in range(len(self.dstn)):
            self.dstn[k].reset()
        ModelEvent.reset(self)

    def quasi_run(self, origins, norm=None):
        origins_new = origins.copy()
        J = len(self.dstn)

        for j in range(J):
            idx = self.data[self.index]
            these = idx == j

            # Get distribution
            atoms = self.dstn[j].atoms
            probs = self.dstn[j].pmv.copy()

            # Apply Harmenberg normalization if applicable
            try:
                harm_idx = self.assigns.index(norm)
                probs *= atoms[harm_idx]
            except:
                pass

            # Expand the set of simulated blobs
            origins_new = self.expand_information(
                origins_new, probs, atoms, which=these
            )

        # Return the altered origins array
        return origins_new


@dataclass(kw_only=True)
class MarkovEvent(ModelEvent):
    """
    Class for representing the realization of a Markov draw for an agent, in which
    a Markov probabilities (array, vector, or a single float) is used to determine
    the realization of some discrete outcome. If the probabilities are a 2D array,
    it represents a Markov matrix (rows sum to 1), and there must be an index; if
    the probabilities are a vector, it should be a stochastic vector; if it's a
    single float, it represents a Bernoulli probability.
    """

    probs: str = field(default="", repr=False)
    index: str = field(default="", repr=False)
    N: int = field(default=1, repr=False)
    seed: int = field(default=0, repr=False)
    # seed is overwritten when each period is created

    def __post_init__(self):
        self.reset_rng()

    def reset(self):
        self.reset_rng()
        ModelEvent.reset(self)

    def reset_rng(self):
        self.RNG = np.random.RandomState(self.seed)

    def draw(self):
        # Initialize the output
        out = -np.ones(self.N, dtype=int)
        if self.probs in self.parameters:
            probs = self.parameters[self.probs]
            probs_are_param = True
        else:
            probs = self.data[self.probs]
            probs_are_param = False

        # Make the base draw(s)
        if self.common:
            X = self.RNG.rand(1)
        else:
            X = self.RNG.rand(self.N)

        if self.index:  # it's a Markov matrix
            idx = self.data[self.index]
            J = probs.shape[0]
            for j in range(J):
                these = idx == j
                if not np.any(these):
                    continue
                P = np.cumsum(probs[j, :])
                if self.common:
                    out[:] = np.searchsorted(P, X[0])  # only one value of X!
                else:
                    out[these] = np.searchsorted(P, X[these])
            return out

        if (isinstance(probs, np.ndarray)) and (
            probs_are_param
        ):  # it's a stochastic vector
            P = np.cumsum(probs)
            if self.common:
                out[:] = np.searchsorted(P, X[0])
                return out
            else:
                return np.searchsorted(P, X)

        # Otherwise, this is just a Bernoulli RV
        P = probs
        if self.common:
            out[:] = X < P
            return out
        else:
            return X < P  # basic Bernoulli

    def run(self):
        self.assign(self.draw())

    def quasi_run(self, origins, norm=None):
        if self.probs in self.parameters:
            probs = self.parameters[self.probs]
            probs_are_param = True
        else:
            probs = self.data[self.probs]
            probs_are_param = False

        # If it's a Markov matrix:
        if self.index:
            K = probs.shape[0]
            atoms = np.array([np.arange(probs.shape[1], dtype=int)])
            origins_new = origins.copy()
            for k in range(K):
                idx = self.data[self.index]
                these = idx == k
                probs_temp = probs[k, :]
                origins_new = self.expand_information(
                    origins_new, probs_temp, atoms, which=these
                )
            return origins_new

        # If it's a stochastic vector:
        if (isinstance(probs, np.ndarray)) and (probs_are_param):
            atoms = np.array([np.arange(probs.shape[0], dtype=int)])
            origins_new = self.expand_information(origins, probs, atoms)
            return origins_new

        # Otherwise, this is just a Bernoulli RV, but it might have idiosyncratic probability
        if probs_are_param:
            P = probs
            atoms = np.array([[False, True]])
            origins_new = self.expand_information(origins, np.array([1 - P, P]), atoms)
            return origins_new

        # Final case: probability is idiosyncratic Bernoulli
        origins_new = self.add_idiosyncratic_bernoulli_info(origins, probs)
        return origins_new


@dataclass(kw_only=True)
class EvaluationEvent(ModelEvent):
    """
    Class for representing the evaluation of a model function. This might be from
    the solution of the model (like a policy function or decision rule) or just
    a non-algebraic function used in the model. This looks a lot like DynamicEvent.

    Parameters
    ----------
    func : Callable
        Model function that is evaluated in this event, with the output assigned
        to the appropriate variables.
    """

    func: Callable = field(default_factory=NullFunc, repr=False)
    arguments: list[str] = field(default_factory=list, repr=False)

    def evaluate(self):
        temp_dict = self.data.copy()
        temp_dict.update(self.parameters)
        args_temp = (temp_dict[arg] for arg in self.arguments)
        out = self.func(*args_temp)
        return out

    def run(self):
        self.assign(self.evaluate())

    def quasi_run(self, origins, norm=None):
        self.run()
        return origins


@dataclass(kw_only=True)
class SimBlock:
    """
    Class for representing a "block" of a simulated model, which might be a whole
    period or a "stage" within a period.

    Parameters
    ----------
    description : str
        Textual description of what happens in this simulated block.
    statement : str
        Verbatim model statement that was used to create this block.
    content : dict
        Dictionary of objects that are constant / universal within the block.
        This includes both traditional numeric parameters as well as functions.
    arrival : list[str]
        List of inbound states: information available at the *start* of the block.
    events: list[ModelEvent]
        Ordered list of events that happen during the block.
    data: dict
        Dictionary that stores current variable values.
    N : int
        Number of idiosyncratic agents in this block.
    """

    statement: str = field(default="", repr=False)
    content: dict = field(default_factory=dict)
    description: str = field(default="", repr=False)
    arrival: list[str] = field(default_factory=list, repr=False)
    events: list[ModelEvent] = field(default_factory=list, repr=False)
    data: dict = field(default_factory=dict, repr=False)
    N: int = field(default=1, repr=False)

    def run(self):
        """
        Run this simulated block by running each of its events in order.
        """
        for j in range(len(self.events)):
            event = self.events[j]
            for k in range(len(event.assigns)):
                var = event.assigns[k]
                if var in event.data.keys():
                    del event.data[var]
            for k in range(len(event.needs)):
                var = event.needs[k]
                event.data[var] = self.data[var]
            event.N = self.N
            event.run()
            for k in range(len(event.assigns)):
                var = event.assigns[k]
                self.data[var] = event.data[var]

    def reset(self):
        """
        Reset the simulated block by resetting each of its events.
        """
        self.data = {}
        for j in range(len(self.events)):
            self.events[j].reset()

    def distribute_content(self):
        """
        Fill in parameters, functions, and distributions to each event.
        """
        for event in self.events:
            for param in event.parameters.keys():
                try:
                    event.parameters[param] = self.content[param]
                except:
                    raise ValueError(
                        "Could not distribute the parameter called " + param + "!"
                    )
            if (type(event) is RandomEvent) or (type(event) is RandomIndexedEvent):
                try:
                    event.dstn = self.content[event._dstn_name]
                except:
                    raise ValueError(
                        "Could not find a distribution called " + event._dstn_name + "!"
                    )
            if type(event) is EvaluationEvent:
                try:
                    event.func = self.content[event._func_name]
                except:
                    raise ValueError(
                        "Could not find a function called " + event._func_name + "!"
                    )

    def make_transition_matrices(self, grid_specs, twist=None, norm=None):
        """
        Construct a transition matrix for this block, moving from a discretized
        grid of arrival variables to a discretized grid of end-of-block variables.
        User specifies how the grids of pre-states should be built. Output is
        stored in attributes of self as follows:

        - matrices : A dictionary of arrays that cast from the arrival state space
                     to the grid of outcome variables. Doing np.dot(dstn, matrices[var])
                     will yield the discretized distribution of that outcome variable.
        - grids : A dictionary of discretized grids for outcome variables. Doing
                  np.dot(np.dot(dstn, matrices[var]), grids[var]) yields the *average*
                  of that outcome in the population.

        Parameters
        ----------
        grid_specs : dict
            Dictionary of dictionaries of grid specifications. For now, these have
            at most a minimum value, a maximum value, a number of nodes, and a poly-
            nomial order. They are equispaced if a min and max are specified, and
            polynomially spaced with the specified order > 0 if provided. Otherwise,
            they are set at 0,..,N if only N is provided.
        twist : dict or None
            Mapping from end-of-period (continuation) variables to successor's
            arrival variables. When this is specified, additional output is created
            for the "full period" arrival-to-arrival transition matrix.
        norm : str or None
            Name of the shock variable by which to normalize for Harmenberg
            aggregation. By default, no normalization happens.

        Returns
        -------
        None
        """
        # Initialize dictionaries of input and output grids
        arrival_N = len(self.arrival)
        completed = arrival_N * [False]
        grids_in = {}
        grids_out = {}
        if arrival_N == 0:  # should only be for initializer block
            dummy_grid = np.array([0])
            grids_in["_dummy"] = dummy_grid

        # Construct a grid for each requested variable
        continuous_grid_out_bool = []
        grid_orders = {}
        for var in grid_specs.keys():
            spec = grid_specs[var]
            try:
                idx = self.arrival.index(var)
                completed[idx] = True
                is_arrival = True
            except:
                is_arrival = False
            if ("min" in spec) and ("max" in spec):
                Q = spec["order"] if "order" in spec else 1.0
                bot = spec["min"]
                top = spec["max"]
                N = spec["N"]
                new_grid = make_exponential_grid(bot, top, N, Q)
                is_cont = True
                grid_orders[var] = Q
            elif "N" in spec:
                new_grid = np.arange(spec["N"], dtype=int)
                is_cont = False
                grid_orders[var] = -1
            else:
                new_grid = None  # could not make grid, construct later
                is_cont = False
                grid_orders[var] = None

            if is_arrival:
                grids_in[var] = new_grid
            else:
                grids_out[var] = new_grid
                continuous_grid_out_bool.append(is_cont)

        # Verify that specifications were passed for all arrival variables
        for j in range(len(self.arrival)):
            if not completed[j]:
                raise ValueError(
                    "No grid specification was provided for " + self.arrival[var] + "!"
                )

        # If an intertemporal twist was specified, make result grids for continuation variables.
        # This overrides any grids for these variables that were explicitly specified
        if twist is not None:
            for cont_var in twist.keys():
                arr_var = twist[cont_var]
                if cont_var not in list(grids_out.keys()):
                    is_cont = grids_in[arr_var].dtype is np.dtype(np.float64)
                    continuous_grid_out_bool.append(is_cont)
                grids_out[cont_var] = copy(grids_in[arr_var])
                grid_orders[cont_var] = grid_orders[arr_var]
        grid_out_is_continuous = np.array(continuous_grid_out_bool)

        # Make meshes of all the arrival grids, which will be the initial simulation data
        if arrival_N > 0:
            state_meshes = np.meshgrid(
                *[grids_in[k] for k in self.arrival], indexing="ij"
            )
        else:  # this only happens in the initializer block
            state_meshes = [dummy_grid.copy()]
        state_init = {
            self.arrival[k]: state_meshes[k].flatten() for k in range(arrival_N)
        }
        N_orig = state_meshes[0].size
        self.N = N_orig
        mesh_tuples = [
            [state_init[self.arrival[k]][n] for k in range(arrival_N)]
            for n in range(self.N)
        ]

        # Make the initial vector of probability masses
        state_init["pmv_"] = np.ones(self.N)

        # Initialize the array of arrival states
        origin_array = np.arange(self.N, dtype=int)

        # Reset the block's state and give it the initial state data
        self.reset()
        self.data.update(state_init)

        # Loop through each event in order and quasi-simulate it
        for j in range(len(self.events)):
            event = self.events[j]
            event.data = self.data  # Give event *all* data directly
            event.N = self.N
            origin_array = event.quasi_run(origin_array, norm=norm)
            self.N = self.data["pmv_"].size

        # Add survival to output if mortality is in the model
        if "dead" in self.data.keys():
            grids_out["dead"] = None

        # Get continuation variable names, making sure they're in the same order
        # as named by the arrival variables. This should maybe be done in the
        # simulator when it's initialized.
        if twist is not None:
            cont_vars_orig = list(twist.keys())
            temp_dict = {twist[var]: var for var in cont_vars_orig}
            cont_vars = []
            for var in self.arrival:
                cont_vars.append(temp_dict[var])
            if "dead" in self.data.keys():
                cont_vars.append("dead")
                grid_out_is_continuous = np.concatenate(
                    (grid_out_is_continuous, [False])
                )
        else:
            cont_vars = list(grids_out.keys())  # all outcomes are arrival vars
        D = len(cont_vars)

        # Now project the final results onto the output or result grids
        N = self.N
        J = state_meshes[0].size
        matrices_out = {}
        cont_idx = {}
        cont_alpha = {}
        cont_M = {}
        cont_discrete = {}
        k = 0
        for var in grids_out.keys():
            if var not in self.data.keys():
                raise ValueError(
                    "Variable " + var + " does not exist but a grid was specified!"
                )
            grid = grids_out[var]
            vals = self.data[var]
            pmv = self.data["pmv_"]
            M = grid.size if grid is not None else 0

            # Semi-hacky fix to deal with omitted arrival variables
            if (M == 1) and (vals.dtype is np.dtype(np.float64)):
                grid = grid.astype(float)
                grids_out[var] = grid
                grid_out_is_continuous[k] = True

            if grid_out_is_continuous[k]:
                # Split the final values among discrete gridpoints on the interior.
                # NB: This will only work properly if the grid is equispaced
                if M > 1:
                    Q = grid_orders[var]
                    if var in cont_vars:
                        trans_matrix, cont_idx[var], cont_alpha[var] = (
                            aggregate_blobs_onto_polynomial_grid_alt(
                                vals, pmv, origin_array, grid, J, Q
                            )
                        )
                        cont_M[var] = M
                        cont_discrete[var] = False
                    else:
                        trans_matrix = aggregate_blobs_onto_polynomial_grid(
                            vals, pmv, origin_array, grid, J, Q
                        )
                else:  # Skip if the grid is a dummy with only one value.
                    trans_matrix = np.ones((J, M))
                    if var in cont_vars:
                        cont_idx[var] = np.zeros(N, dtype=int)
                        cont_alpha[var] = np.zeros(N)
                        cont_M[var] = M
                        cont_discrete[var] = False

            else:  # Grid is discrete, can use simpler method
                if grid is None:
                    M = np.max(vals.astype(int))
                    if var == "dead":
                        M = 2
                    grid = np.arange(M, dtype=int)
                    grids_out[var] = grid
                M = grid.size
                vals = vals.astype(int)
                trans_matrix = aggregate_blobs_onto_discrete_grid(
                    vals, pmv, origin_array, M, J
                )
                if var in cont_vars:
                    cont_idx[var] = vals
                    cont_alpha[var] = np.zeros(N)
                    cont_M[var] = M
                    cont_discrete[var] = True

            # Store the transition matrix for this variable
            matrices_out[var] = trans_matrix
            k += 1

        # Construct an overall transition matrix from arrival to continuation variables.
        # If this is the initializer block, the "arrival" variable is just the initial
        # dummy state, and the "continuation" variables are actually the arrival variables
        # for ordinary blocks/periods.

        # Count the number of non-trivial dimensions. A continuation dimension
        # is non-trivial if it is both continuous and has more than one grid node.
        C = 0
        shape = [N_orig]
        trivial = []
        for var in cont_vars:
            shape.append(cont_M[var])
            if (not cont_discrete[var]) and (cont_M[var] > 1):
                C += 1
                trivial.append(False)
            else:
                trivial.append(True)
        trivial = np.array(trivial)

        # Make a binary array of offsets from the base index
        bin_array_base = np.array(list(product([0, 1], repeat=C)))
        bin_array = np.empty((2**C, D), dtype=int)
        some_zeros = np.zeros(2**C, dtype=int)
        c = 0
        for d in range(D):
            bin_array[:, d] = some_zeros if trivial[d] else bin_array_base[:, c]
            c += not trivial[d]

        # Make a vector of dimensional offsets from the base index
        dim_offsets = np.ones(D, dtype=int)
        for d in range(D - 1):
            dim_offsets[d] = np.prod(shape[(d + 2) :])
        dim_offsets_X = np.tile(dim_offsets, (2**C, 1))
        offsets = np.sum(bin_array * dim_offsets_X, axis=1)

        # Make combined arrays of indices and alphas
        index_array = np.empty((N, D), dtype=int)
        alpha_array = np.empty((N, D, 2))
        for d in range(D):
            var = cont_vars[d]
            index_array[:, d] = cont_idx[var]
            alpha_array[:, d, 0] = 1.0 - cont_alpha[var]
            alpha_array[:, d, 1] = cont_alpha[var]
        idx_array = np.dot(index_array, dim_offsets)

        # Make the master transition array
        blank = np.zeros(np.array((N_orig, np.prod(shape[1:]))))
        master_trans_array_X = calc_overall_trans_probs(
            blank, idx_array, alpha_array, bin_array, offsets, pmv, origin_array
        )

        # Condition on survival if relevant
        if "dead" in self.data.keys():
            master_trans_array_X = np.reshape(master_trans_array_X, (N_orig, N_orig, 2))
            survival_probs = np.reshape(matrices_out["dead"][:, 0], [N_orig, 1])
            master_trans_array_X = master_trans_array_X[..., 0] / survival_probs

        # Reshape the transition matrix depending on what kind of block this is
        if arrival_N == 0:
            # If this is the initializer block, the "transition" matrix is really
            # just the initial distribution of states at model birth; flatten it.
            master_init_array = master_trans_array_X.flatten()
        else:
            # In an ordinary period, reshape the transition array so it's square.
            master_trans_array = np.reshape(master_trans_array_X, (N_orig, N_orig))

        # Store the results as attributes of self
        grids = {}
        grids.update(grids_in)
        grids.update(grids_out)
        self.grids = grids
        self.matrices = matrices_out
        self.mesh = mesh_tuples
        if twist is not None:
            self.trans_array = master_trans_array
        if arrival_N == 0:
            self.init_dstn = master_init_array


@dataclass(kw_only=True)
class AgentSimulator:
    """
    A class for representing an entire simulator structure for an AgentType.
    It includes a sequence of SimBlocks representing periods of the model, which
    could be built from the information on an AgentType instance.

    Parameters
    ----------
    name : str
        Short name of this model.s
    description : str
        Textual description of what happens in this simulated block.
    statement : str
        Verbatim model statement that was used to create this simulator.
    comments : dict
        Dictionary of comments or descriptions for various model objects.
    parameters : list[str]
        List of parameter names used in the model.
    distributions : list[str]
        List of distribution names used in the model.
    functions : list[str]
        List of function names used in the model.
    common: list[str]
        Names of variables that are common across idiosyncratic agents.
    types: dict
        Dictionary of data types for all variables in the model.
    N_agents: int
        Number of idiosyncratic agents in this simulation.
    T_total: int
        Total number of periods in these agents' model.
    T_sim: int
        Maximum number of periods that will be simulated, determining the size
        of the history arrays.
    T_age: int
        Period after which to automatically terminate an agent if they would
        survive past this period.
    stop_dead : bool
        Whether simulated agents who draw dead=True should actually cease acting.
        Default is True. Setting to False allows "cohort-style" simulation that
        will generate many agents that survive to old ages. In most cases, T_sim
        should not exceed T_age, unless the user really does want multiple succ-
        essive cohorts to be born and fully simulated.
    replace_dead : bool
        Whether simulated agents who are marked as dead should be replaced with
        newborns (default True) or simply cease acting without replacement (False).
        The latter option is useful for models with state-dependent mortality,
        to allow "cohort-style" simulation with the correct distribution of states
        for survivors at each age. Setting to False has no effect if stop_dead is True.
    periods: list[SimBlock]
        Ordered list of simulation blocks, each representing a period.
    twist : dict
        Dictionary that maps period t-1 variables to period t variables, as a
        relabeling "between" periods.
    initializer : SimBlock
        A special simulated block that should have *no* arrival variables, because
        it represents the initialization of "newborn" agents.
    data : dict
        Dictionary that holds *current* values of model variables.
    track_vars : list[str]
        List of names of variables whose history should be tracked in the simulation.
    history : dict
        Dictionary that holds the histories of tracked variables.
    """

    name: str = field(default="")
    description: str = field(default="")
    statement: str = field(default="", repr=False)
    comments: dict = field(default_factory=dict, repr=False)
    parameters: list[str] = field(default_factory=list, repr=False)
    distributions: list[str] = field(default_factory=list, repr=False)
    functions: list[str] = field(default_factory=list, repr=False)
    common: list[str] = field(default_factory=list, repr=False)
    types: dict = field(default_factory=dict, repr=False)
    N_agents: int = field(default=1)
    T_total: int = field(default=1, repr=False)
    T_sim: int = field(default=1)
    T_age: int = field(default=0, repr=False)
    stop_dead: bool = field(default=True)
    replace_dead: bool = field(default=True)
    periods: list[SimBlock] = field(default_factory=list, repr=False)
    twist: dict = field(default_factory=dict, repr=False)
    data: dict = field(default_factory=dict, repr=False)
    initializer: field(default_factory=SimBlock, repr=False)
    track_vars: list[str] = field(default_factory=list, repr=False)
    history: dict = field(default_factory=dict, repr=False)

    def simulate(self, T=None):
        """
        Simulates the model for T periods, including replacing dead agents as
        warranted and storing tracked variables in the history. If T is not
        specified, the agents are simulated for the entire T_sim periods.
        This is the primary user-facing simulation method.
        """
        if T is None:
            T = self.T_sim - self.t_sim  # All remaining simulated periods
        if (T + self.t_sim) > self.T_sim:
            raise ValueError("Can't simulate more than T_sim periods!")

        # Execute the simulation loop for T periods
        for t in range(T):
            # Do the ordinary work for simulating a period
            self.sim_one_period()

            # Mark agents who have reached maximum allowable age
            if "dead" in self.data.keys() and self.T_age > 0:
                too_old = self.t_age == self.T_age
                self.data["dead"][too_old] = True

            # Record tracked variables and advance age
            self.store_tracked_vars()
            self.advance_age()

            # Handle death and replacement depending on simulation style
            if "dead" in self.data.keys() and self.stop_dead:
                self.mark_dead_agents()
            self.t_sim += 1

    def reset(self):
        """
        Completely reset this simulator back to its original state so that it
        can be run from scratch. This should allow it to generate the same results
        every single time the simulator is run (if nothing changes).
        """
        N = self.N_agents
        T = self.T_sim
        self.t_sim = 0  # Time index for the simulation

        # Reset the variable data and history arrays
        self.clear_data()
        self.history = {}
        for var in self.track_vars:
            self.history[var] = np.empty((T, N), dtype=self.types[var])

        # Reset all of the blocks / periods
        self.initializer.reset()
        for t in range(len(self.periods)):
            self.periods[t].reset()

        # Specify all agents as "newborns" assigned to the initializer block
        self.t_seq_bool_array = np.zeros((self.T_total, N), dtype=bool)
        self.t_age = -np.ones(N, dtype=int)

    def clear_data(self, skip=None):
        """
        Reset all current data arrays back to blank, other than those designated
        to be skipped, if any.

        Parameters
        ----------
        skip : [str] or None
            Names of variables *not* to be cleared from data. Default is None.

        Returns
        -------
        None
        """
        if skip is None:
            skip = []
        N = self.N_agents
        # self.data = {}
        for var in self.types.keys():
            if var in skip:
                continue
            this_type = self.types[var]
            if this_type is float:
                self.data[var] = np.full((N,), np.nan)
            elif this_type is bool:
                self.data[var] = np.zeros((N,), dtype=bool)
            elif this_type is int:
                self.data[var] = np.zeros((N,), dtype=np.int32)
            elif this_type is complex:
                self.data[var] = np.full((N,), np.nan, dtype=complex)
            else:
                raise ValueError(
                    "Type "
                    + str(this_type)
                    + " of variable "
                    + var
                    + " was not recognized!"
                )

    def mark_dead_agents(self):
        """
        Looks at the special data field "dead" and marks those agents for replacement.
        If no variable called "dead" has been defined, this is skipped.
        """
        who_died = self.data["dead"]
        self.t_seq_bool_array[:, who_died] = False
        self.t_age[who_died] = -1

    def create_newborns(self):
        """
        Calls the initializer to generate newborns where needed.
        """
        # Skip this step if there are no newborns
        newborns = self.t_age == -1
        if not np.any(newborns):
            return

        # Generate initial arrival variables
        N = np.sum(newborns)
        self.initializer.data = {}  # by definition
        self.initializer.N = N
        self.initializer.run()

        # Set the initial arrival data for newborns and clear other variables
        init_arrival = self.periods[0].arrival
        for var in self.types:
            self.data[var][newborns] = (
                self.initializer.data[var]
                if var in init_arrival
                else np.empty(N, dtype=self.types[var])
            )

        # Set newborns' period to 0
        self.t_age[newborns] = 0
        self.t_seq_bool_array[0, newborns] = True

    def store_tracked_vars(self):
        """
        Record current values of requested variables in the history dictionary.
        """
        for var in self.track_vars:
            self.history[var][self.t_sim, :] = self.data[var]

    def advance_age(self):
        """
        Increments age for all agents, altering t_age and t_age_bool. Agents in
        the last period of the sequence will be assigned to the initial period.
        In a lifecycle model, those agents should be marked as dead and replaced
        in short order.
        """
        alive = self.t_age >= 0  # Don't age the dead
        self.t_age[alive] += 1
        X = self.t_seq_bool_array  # For shorter typing on next line
        self.t_seq_bool_array[:, alive] = np.concatenate(
            (X[-1:, alive], X[:-1, alive]), axis=0
        )

    def sim_one_period(self):
        """
        Simulates one period of the model by advancing all agents one period.
        This includes creating newborns, but it does NOT include eliminating
        dead agents nor storing tracked results in the history. This method
        should usually not be called by a user, instead using simulate(1) if
        you want to run the model for exactly one period.
        """
        # Use the "twist" information to advance last period's end-of-period
        # information/values to be the arrival variables for this period. Then, for
        # each variable other than those brought in with the twist, wipe it clean.
        keepers = []
        for var_tm1 in self.twist:
            var_t = self.twist[var_tm1]
            keepers.append(var_t)
            self.data[var_t] = self.data[var_tm1].copy()
        self.clear_data(skip=keepers)

        # Create newborns first so the arrival vars exist. This should be done in
        # the first simulated period (t_sim=0) or if decedents should be replaced.
        if self.replace_dead or self.t_sim == 0:
            self.create_newborns()

        # Loop through ages and run the model on the appropriately aged agents
        for t in range(self.T_total):
            these = self.t_seq_bool_array[t, :]
            if not np.any(these):
                continue  # Skip any "empty ages"
            this_period = self.periods[t]

            data_temp = {var: self.data[var][these] for var in this_period.arrival}
            this_period.data = data_temp
            this_period.N = np.sum(these)
            this_period.run()

            # Extract all of the variables from this period and write it to data
            for var in this_period.data.keys():
                self.data[var][these] = this_period.data[var]

        # Put time information into the data dictionary
        self.data["t_age"] = self.t_age.copy()
        self.data["t_seq"] = np.argmax(self.t_seq_bool_array, axis=0).astype(int)

    def make_transition_matrices(
        self, grid_specs, norm=None, fake_news_timing=False, for_t=None
    ):
        """
        Build Markov-style transition matrices for each period of the model, as
        well as the initial distribution of arrival variables for newborns.
        Stores results to the attributes of self as follows:

        - trans_arrays : List of Markov matrices for transitioning from the arrival
                         state space in period t to the arrival state space in t+1.
                         This transition includes death (and replacement).
        - newborn_dstn : Stochastic vector as a NumPy array, representing the distribution
                         of arrival states for "newborns" who were just initialized.
        - state_grids : Nested list of tuples representing the arrival state space for
                        each period. Each element corresponds to the discretized arrival
                        state space point with the same index in trans_arrays (and
                        newborn_dstn). Arrival states are ordered within a tuple in the
                        same order as the model file. Linked from period[t].mesh.
        - outcome_arrays : List of dictionaries of arrays that cast from the arrival
                           state space to the grid of outcome variables, for each period.
                           Doing np.dot(state_dstn, outcome_arrays[t][var]) will yield
                           the discretized distribution of that outcome variable. Linked
                           from periods[t].matrices.
        - outcome_grids : List of dictionaries of discretized outcomes in each period.
                          Keys are names of outcome variables, and entries are vectors
                          of discretized values that the outcome variable can take on.
                          Doing np.dot(np.dot(state_dstn, outcome_arrays[var]), outcome_grids[var])
                          yields the *average* of that outcome in the population. Linked
                          from periods[t].grids.

        Parameters
        ----------
        grid_specs : dict
            Dictionary of dictionaries with specifications for discretized grids
            of all variables of interest. If any arrival variables are omitted,
            they will be given a default trivial grid with one node at 0. This
            should only be done if that arrival variable is closely tied to the
            Harmenberg normalizing variable; see below. A grid specification must
            include a number of gridpoints N, and should also include a min and
            max if the variable is continuous. If the variable is discrete, the
            grid values are assumed to be 0,..,N.
        norm : str or None
            Name of the variable for which Harmenberg normalization should be
            applied, if any. This should be a variable that is directly drawn
            from a distribution, not a "downstream" variable.
        fake_news_timing : bool
            Indicator for whether this call is part of the "fake news" algorithm
            for constructing sequence space Jacobians (SSJs). This should only
            ever be set to True in that situation, which affects how mortality
            is handled between periods. In short, the simulator usually assumes
            that "newborns" start with t_seq=0, but during the fake news algorithm,
            that is not the case.
        for_t : list or None
            Optional list of time indices for which the matrices should be built.
            When not specified, all periods are constructed. The most common use
            for this arg is during the "fake news" algorithm for lifecycle models.

        Returns
        -------
        None
        """
        # Sort grid specifications into those needed by the initializer vs those
        # used by other blocks (ordinary periods)
        arrival = self.periods[0].arrival
        arrival_N = len(arrival)
        check_bool = np.zeros(arrival_N, dtype=bool)
        grid_specs_init_orig = {}
        grid_specs_other = {}
        for name in grid_specs.keys():
            if name in arrival:
                idx = arrival.index(name)
                check_bool[idx] = True
                grid_specs_init_orig[name] = copy(grid_specs[name])
            grid_specs_other[name] = copy(grid_specs[name])

        # Build the dictionary of arrival variables, making sure it's in the
        # same order as named self.arrival. For any arrival grids that are
        # not specified, make a dummy specification.
        grid_specs_init = {}
        for n in range(arrival_N):
            name = arrival[n]
            if check_bool[n]:
                grid_specs_init[name] = grid_specs_init_orig[name]
                continue
            dummy_grid_spec = {"N": 1}
            grid_specs_init[name] = dummy_grid_spec
            grid_specs_other[name] = dummy_grid_spec

        # Make the initial state distribution for newborns
        self.initializer.make_transition_matrices(grid_specs_init)
        self.newborn_dstn = self.initializer.init_dstn
        K = self.newborn_dstn.size

        # Make the period-by-period transition matrices
        these_t = range(len(self.periods)) if for_t is None else for_t
        for t in these_t:
            block = self.periods[t]
            block.make_transition_matrices(
                grid_specs_other, twist=self.twist, norm=norm
            )
            block.reset()

        # Extract the master transition matrices into a single list
        p2p_trans_arrays = [block.trans_array for block in self.periods]

        # Apply agent replacement to the last period of the model, representing
        # newborns filling in for decedents. This will usually only do anything
        # at all in "one period infinite horizon" models. If this is part of the
        # fake news algorithm for constructing SSJs, then replace decedents with
        # newborns in *all* periods, because model timing is funny in this case.
        if fake_news_timing:
            T_set = np.arange(len(self.periods)).tolist()
        else:
            T_set = [-1]
        newborn_dstn = np.reshape(self.newborn_dstn, (1, K))
        for t in T_set:
            if "dead" not in self.periods[t].matrices.keys():
                continue
            death_prbs = self.periods[t].matrices["dead"][:, 1]
            p2p_trans_arrays[t] *= np.tile(np.reshape(1 - death_prbs, (K, 1)), (1, K))
            p2p_trans_arrays[t] += np.reshape(death_prbs, (K, 1)) * newborn_dstn

        # Store the transition arrays as attributes of self
        self.trans_arrays = p2p_trans_arrays

        # Build and store lists of state meshes, outcome arrays, and outcome grids
        self.state_grids = [self.periods[t].mesh for t in range(len(self.periods))]
        self.outcome_grids = [self.periods[t].grids for t in range(len(self.periods))]
        self.outcome_arrays = [
            self.periods[t].matrices for t in range(len(self.periods))
        ]

    def find_steady_state(self):
        """
        Calculates the steady state distribution of arrival states for a "one period
        infinite horizon" model, storing the result to the attribute steady_state_dstn.
        Should only be run after make_transition_matrices(), and only if T_total = 1
        and the model is infinite horizon.
        """
        if self.T_total != 1:
            raise ValueError(
                "This method currently only works with one period infinite horizon problems."
            )

        # Find the eigenvector associated with the largest eigenvalue of the
        # infinite horizon transition matrix. The largest eigenvalue *should*
        # be 1 for any Markov matrix, but double check to be sure.
        trans_T = csr_matrix(self.trans_arrays[0].transpose())
        v, V = eigs(trans_T, k=1)
        if not np.isclose(v[0], 1.0):
            raise ValueError(
                "The largest eigenvalue of the transition matrix isn't close to 1!"
            )

        # Normalize that eigenvector and make sure its real, then store it
        D = V[:, 0]
        SS_dstn = (D / np.sum(D)).real
        self.steady_state_dstn = SS_dstn

    def get_long_run_average(self, var):
        """
        Calculate and return the long run / steady state population average of
        one named variable. Should only be run after find_steady_state().

        Parameters
        ----------
        var : str
            Name of the variable for which to calculate the long run average.

        Returns
        -------
        var_mean : float
            Long run / steady state population average of the variable.
        """
        if not hasattr(self, "steady_state_dstn"):
            raise ValueError("This method can only be run after find_steady_state()!")

        dstn = self.steady_state_dstn
        array = self.outcome_arrays[0][var]
        grid = self.outcome_grids[0][var]

        var_dstn = np.dot(dstn, array)
        var_mean = np.dot(var_dstn, grid)
        return var_mean

    def simulate_cohort_by_grids(
        self,
        outcomes,
        T_max=None,
        calc_dstn=False,
        calc_avg=True,
        from_dstn=None,
    ):
        """
        Generate a simulated "cohort style" history for this type of agents using
        discretized grid methods. Can only be run after running make_transition_matrices().
        Starting from the distribution of states at birth, the population is moved
        forward in time via the transition matrices, and the distribution and/or
        average of specified outcomes are stored in the dictionary attributes
        history_dstn and history_avg respectively.

        Parameters
        ----------
        outcomes : str or [str]
            Names of one or more outcome variables to be tracked during the grid
            simulation. Each named variable should have an outcome grid specified
            when make_transition_matrices() was called, whether explicitly or
            implicitly. The existence of these grids is checked as a first step.
        T_max : int or None
            If specified, the number of periods of the model to actually generate
            output for. If not specified, all periods are run.
        calc_dstn : bool
            Whether outcome distributions should be stored in the dictionary
            attribute history_dstn. The default is False.
        calc_avg : bool
            Whether outcome averages should be stored in the dictionary attribute
            history_avg. The default is True.
        from_dstn : np.array or None
            Optional initial distribution of arrival states. If not specified, the
            newborn distribution in the initializer is assumed to be used.

        Returns
        -------
        None
        """
        # First, verify that newborn and transition matrices exist for all periods
        if not hasattr(self, "newborn_dstn"):
            raise ValueError(
                "The newborn state distribution does not exist; make_transition_matrices() must be run before grid simulations!"
            )
        if T_max is None:
            T_max = self.T_total
        T_max = np.minimum(T_max, self.T_total)
        if not hasattr(self, "trans_arrays"):
            raise ValueError(
                "The transition arrays do not exist; make_transition_matrices() must be run before grid simulations!"
            )
        if len(self.trans_arrays) < T_max:
            raise ValueError(
                "There are somehow fewer elements of trans_array than there should be!"
            )
        if not (calc_dstn or calc_avg):
            return  # No work actually requested, we're done here

        # Initialize generated output as requested
        if isinstance(outcomes, str):
            outcomes = [outcomes]
        if calc_dstn:
            history_dstn = {}
            for name in outcomes:  # List will be concatenated to array at end
                history_dstn[name] = []  # if all distributions are same size
        if calc_avg:
            history_avg = {}
            for name in outcomes:
                history_avg[name] = np.empty(T_max)

        # Initialize the state distribution
        current_dstn = (
            self.newborn_dstn.copy() if from_dstn is None else from_dstn.copy()
        )
        state_dstn_by_age = []

        # Loop over requested periods of this agent type's model
        for t in range(T_max):
            state_dstn_by_age.append(current_dstn)

            # Calculate outcome distributions and averages as requested
            for name in outcomes:
                this_outcome = self.periods[t].matrices[name].transpose()
                this_dstn = np.dot(this_outcome, current_dstn)
                if calc_dstn:
                    history_dstn[name].append(this_dstn)
                if calc_avg:
                    this_grid = self.periods[t].grids[name]
                    history_avg[name][t] = np.dot(this_dstn, this_grid)

            # Advance the distribution to the next period
            current_dstn = np.dot(self.trans_arrays[t].transpose(), current_dstn)

        # Reshape the distribution histories if possible
        if calc_dstn:
            for name in outcomes:
                dstn_sizes = np.array([dstn.size for dstn in history_dstn[name]])
                if np.all(dstn_sizes == dstn_sizes[0]):
                    history_dstn[name] = np.stack(history_dstn[name], axis=1)

        # Store results as attributes of self
        self.state_dstn_by_age = state_dstn_by_age
        if calc_dstn:
            self.history_dstn = history_dstn
        if calc_avg:
            self.history_avg = history_avg

    def describe_model(self, display=True):
        """
        Convenience method that prints model information to screen.
        """
        # Make a twist statement
        twist_statement = ""
        for var_tm1 in self.twist.keys():
            var_t = self.twist[var_tm1]
            new_line = var_tm1 + "[t-1] <---> " + var_t + "[t]\n"
            twist_statement += new_line

        # Assemble the overall model statement
        output = ""
        output += "----------------------------------\n"
        output += "%%%%% INITIALIZATION AT BIRTH %%%%\n"
        output += "----------------------------------\n"
        output += self.initializer.statement
        output += "----------------------------------\n"
        output += "%%%% DYNAMICS WITHIN PERIOD t %%%%\n"
        output += "----------------------------------\n"
        output += self.statement
        output += "----------------------------------\n"
        output += "%%%%%%% RELABELING / TWIST %%%%%%%\n"
        output += "----------------------------------\n"
        output += twist_statement
        output += "-----------------------------------"

        # Return or print the output
        if display:
            print(output)
            return
        else:
            return output

    def describe_symbols(self, display=True):
        """
        Convenience method that prints symbol information to screen.
        """
        # Get names and types
        symbols_lines = []
        comments = []
        for key in self.comments.keys():
            comments.append(self.comments[key])

            # Get type of object
            if key in self.types.keys():
                this_type = str(self.types[key].__name__)
            elif key in self.distributions:
                this_type = "dstn"
            elif key in self.parameters:
                this_type = "param"
            elif key in self.functions:
                this_type = "func"

            # Add tags
            if key in self.common:
                this_type += ", common"
            # if key in self.solution:
            #    this_type += ', solution'
            this_line = key + " (" + this_type + ")"
            symbols_lines.append(this_line)

        # Add comments, aligned
        symbols_text = ""
        longest = np.max([len(this) for this in symbols_lines])
        for j in range(len(symbols_lines)):
            line = symbols_lines[j]
            comment = comments[j]
            L = len(line)
            pad = (longest + 1) - L
            symbols_text += line + pad * " " + ": " + comment + "\n"

        # Return or print the output
        output = symbols_text
        if display:
            print(output)
            return
        else:
            return output

    def describe(self, symbols=True, model=True, display=True):
        """
        Convenience method for showing all information about the model.
        """
        # Asssemble the requested output
        output = self.name + ": " + self.description + "\n"
        if symbols or model:
            output += "\n"
        if symbols:
            output += "----------------------------------\n"
            output += "%%%%%%%%%%%%% SYMBOLS %%%%%%%%%%%%\n"
            output += "----------------------------------\n"
            output += self.describe_symbols(display=False)
        if model:
            output += self.describe_model(display=False)
        if symbols and not model:
            output += "----------------------------------"

        # Return or print the output
        if display:
            print(output)
            return
        else:
            return output


def make_simulator_from_agent(agent, stop_dead=True, replace_dead=True, common=None):
    """
    Build an AgentSimulator instance based on an AgentType instance. The AgentType
    should have its model attribute defined so that it can be parsed and translated
    into the simulator structure. The names of objects in the model statement
    should correspond to attributes of the AgentType.

    Parameters
    ----------
    agent : AgentType
        Agents for whom a new simulator is to be constructed.
    stop_dead : bool
        Whether simulated agents who draw dead=True should actually cease acting.
        Default is True. Setting to False allows "cohort-style" simulation that
        will generate many agents that survive to old ages. In most cases, T_sim
        should not exceed T_age, unless the user really does want multiple succ-
        essive cohorts to be born and fully simulated.
    replace_dead : bool
        Whether simulated agents who are marked as dead should be replaced with
        newborns (default True) or simply cease acting without replacement (False).
        The latter option is useful for models with state-dependent mortality,
        to allow "cohort-style" simulation with the correct distribution of states
        for survivors at each age. Setting False has no effect if stop_dead is True.
    common : [str] or None
        List of random variables that should be treated as commonly shared across
        all agents, rather than idiosyncratically drawn. If this is provided, it
        will override the model defaults.

    Returns
    -------
    new_simulator : AgentSimulator
        A simulator structure based on the agents.
    """
    # Read the model statement into a dictionary, and get names of attributes
    if hasattr(agent, "model_statement"):  # look for a custom model statement
        model_statement = copy(agent.model_statement)
    else:  # otherwise use the default model file
        with importlib.resources.open_text("HARK.models", agent.model_file) as f:
            model_statement = f.read()
            f.close()
    model = yaml.safe_load(model_statement)
    time_vary = agent.time_vary
    time_inv = agent.time_inv
    cycles = agent.cycles
    T_age = agent.T_age
    comments = {}
    RNG = agent.RNG  # this is only for generating seeds for MarkovEvents

    # Extract basic fields from the model
    try:
        model_name = model["name"]
    except:
        model_name = "DEFAULT_NAME"
    try:
        description = model["description"]
    except:
        description = "(no description provided)"
    try:
        variables = model["symbols"]["variables"]
    except:
        variables = []
    try:
        twist = model["twist"]
    except:
        twist = {}
    if common is None:
        try:
            common = model["symbols"]["common"]
        except:
            common = []

    # Extract arrival variable names that were explicitly listed
    try:
        arrival = model["symbols"]["arrival"]
    except:
        arrival = []

    # Make a dictionary of declared data types and add comments
    types = {}
    for var_line in variables:  # Loop through declared variables
        var_name, var_type, flags, desc = parse_declaration_for_parts(var_line)
        if var_type is not None:
            try:
                var_type = eval(var_type)
            except:
                raise ValueError(
                    "Couldn't understand type "
                    + var_type
                    + " for declared variable "
                    + var_name
                    + "!"
                )
        else:
            var_type = float
        types[var_name] = var_type
        comments[var_name] = desc
        if ("arrival" in flags) and (var_name not in arrival):
            arrival.append(var_name)
        if ("common" in flags) and (var_name not in common):
            common.append(var_name)

    # Make a blank "template" period with structure but no data
    template_period, information, offset, solution, block_comments = (
        make_template_block(model, arrival, common)
    )
    comments.update(block_comments)

    # Make the agent initializer, without parameter values (etc)
    initializer, init_info = make_initializer(model, arrival, common)

    # Extract basic fields from the template period and model
    statement = template_period.statement
    content = template_period.content

    # Get the names of parameters, functions, and distributions
    parameters = []
    functions = []
    distributions = []
    for key in information.keys():
        val = information[key]
        if val is None:
            parameters.append(key)
        elif type(val) is NullFunc:
            functions.append(key)
        elif type(val) is Distribution:
            distributions.append(key)

    # Loop through variables that appear in the model block but were undeclared
    for var in information.keys():
        if var in types.keys():
            continue
        this = information[var]
        if (this is None) or (type(this) is Distribution) or (type(this) is NullFunc):
            continue
        types[var] = float
        comments[var] = ""
    if "dead" in types.keys():
        types["dead"] = bool
        comments["dead"] = "whether agent died this period"
    types["t_seq"] = int
    types["t_age"] = int
    comments["t_seq"] = "which period of the sequence the agent is on"
    comments["t_age"] = "how many periods the agent has already lived for"

    # Make a dictionary for the initializer and distribute information
    init_dict = {}
    for name in init_info.keys():
        try:
            init_dict[name] = getattr(agent, name)
        except:
            raise ValueError(
                "Couldn't get a value for initializer object " + name + "!"
            )
    initializer.content = init_dict
    initializer.distribute_content()

    # Make a dictionary of time-invariant parameters
    time_inv_dict = {}
    for name in content:
        if name in time_inv:
            try:
                time_inv_dict[name] = getattr(agent, name)
            except:
                raise ValueError(
                    "Couldn't get a value for time-invariant object " + name + "!"
                )

    # Create a list of periods, pulling appropriate data from the agent for each one
    T_seq = len(agent.solution)  # Number of periods in the solution sequence
    periods = []
    T_cycle = agent.T_cycle
    t_cycle = 0
    for t in range(T_seq):
        # Make a fresh copy of the template period
        new_period = deepcopy(template_period)

        # Make sure each period's events have unique seeds; this is only for MarkovEvents
        for event in new_period.events:
            if hasattr(event, "seed"):
                event.seed = RNG.integers(0, 2**31 - 1)

        # Make the parameter dictionary for this period
        new_param_dict = deepcopy(time_inv_dict)
        for name in content:
            if name in solution:
                new_param_dict[name] = getattr(agent.solution[t], name)
            elif name in time_vary:
                s = (t_cycle - 1) if name in offset else t_cycle
                try:
                    new_param_dict[name] = getattr(agent, name)[s]
                except:
                    raise ValueError(
                        "Couldn't get a value for time-varying object "
                        + name
                        + " at time index "
                        + str(s)
                        + "!"
                    )
            elif name in time_inv:
                continue
            else:
                raise ValueError(
                    "The object called "
                    + name
                    + " is not named in time_inv nor time_vary!"
                )

        # Fill in content for this period, then add it to the list
        new_period.content = new_param_dict
        new_period.distribute_content()
        periods.append(new_period)

        # Advance time according to the cycle
        t_cycle += 1
        if t_cycle == T_cycle:
            t_cycle = 0

    # Calculate maximum age
    if T_age is None:
        T_age = 0
    if cycles > 0:
        T_age_max = T_seq - 1
        T_age = np.minimum(T_age_max, T_age)
    try:
        T_sim = agent.T_sim
    except:
        T_sim = 0  # very boring default!

    # Make and return the new simulator
    new_simulator = AgentSimulator(
        name=model_name,
        description=description,
        statement=statement,
        comments=comments,
        parameters=parameters,
        functions=functions,
        distributions=distributions,
        common=common,
        types=types,
        N_agents=agent.AgentCount,
        T_total=T_seq,
        T_sim=T_sim,
        T_age=T_age,
        stop_dead=stop_dead,
        replace_dead=replace_dead,
        periods=periods,
        twist=twist,
        initializer=initializer,
        track_vars=agent.track_vars,
    )
    new_simulator.solution = solution  # this is for use by SSJ constructor
    return new_simulator


def make_template_block(model, arrival=None, common=None):
    """
    Construct a new SimBlock object as a "template" of the model block. It has
    events and reference information, but no values filled in.

    Parameters
    ----------
    model : dict
        Dictionary with model block information, probably read in as a yaml.
    arrival : [str] or None
        List of arrival variables that were flagged or explicitly listed.
    common : [str] or None
        List of variables that are common or shared across all agents, rather
        than idiosyncratically drawn.

    Returns
    -------
    template_block : SimBlock
        A "template" of this model block, with no parameters (etc) on it.
    info : dict
        Dictionary of model objects that were referenced within the block. Keys
        are object names and entries reveal what kind of object they are:
        - None --> parameter
        - 0 --> outcome/data variable (including arrival variables)
        - NullFunc --> function
        - Distribution --> distribution
    offset : [str]
        List of object names that are offset in time by one period.
    solution : [str]
        List of object names that are part of the model solution.
    comments : dict
        Dictionary of comments included with declared functions, distributions,
        and parameters.
    """
    if arrival is None:
        arrival = []
    if common is None:
        common = []

    # Extract explicitly listed metadata
    try:
        name = model["name"]
    except:
        name = "DEFAULT_NAME"
    try:
        offset = model["symbols"]["offset"]
    except:
        offset = []
    try:
        solution = model["symbols"]["solution"]
    except:
        solution = []

    # Extract parameters, functions, and distributions
    comments = {}
    parameters = {}
    if "parameters" in model["symbols"].keys():
        param_lines = model["symbols"]["parameters"]
        for line in param_lines:
            param_name, datatype, flags, desc = parse_declaration_for_parts(line)
            parameters[param_name] = None
            comments[param_name] = desc
            # TODO: what to do with parameter types?
            if ("offset" in flags) and (param_name not in offset):
                offset.append(param_name)
            if ("solution" in flags) and (param_name not in solution):
                solution.append(param_name)

    functions = {}
    if "functions" in model["symbols"].keys():
        func_lines = model["symbols"]["functions"]
        for line in func_lines:
            func_name, datatype, flags, desc = parse_declaration_for_parts(line)
            if (datatype is not None) and (datatype != "func"):
                raise ValueError(
                    func_name
                    + " was declared as a function, but given a different datatype!"
                )
            functions[func_name] = NullFunc()
            comments[func_name] = desc
            if ("offset" in flags) and (func_name not in offset):
                offset.append(func_name)
            if ("solution" in flags) and (func_name not in solution):
                solution.append(func_name)

    distributions = {}
    if "distributions" in model["symbols"].keys():
        dstn_lines = model["symbols"]["distributions"]
        for line in dstn_lines:
            dstn_name, datatype, flags, desc = parse_declaration_for_parts(line)
            if (datatype is not None) and (datatype != "dstn"):
                raise ValueError(
                    dstn_name
                    + " was declared as a distribution, but given a different datatype!"
                )
            distributions[dstn_name] = Distribution()
            comments[dstn_name] = desc
            if ("offset" in flags) and (dstn_name not in offset):
                offset.append(dstn_name)
            if ("solution" in flags) and (dstn_name not in solution):
                solution.append(dstn_name)

    # Combine those dictionaries into a single "information" dictionary, which
    # represents objects available *at that point* in the dynamic block
    content = parameters.copy()
    content.update(functions)
    content.update(distributions)
    info = deepcopy(content)
    for var in arrival:
        info[var] = 0  # Mark as a state variable

    # Parse the model dynamics
    dynamics = format_block_statement(model["dynamics"])

    # Make the list of ordered events
    events = []
    names_used_in_dynamics = []
    for line in dynamics:
        # Make the new event and add it to the list
        new_event, names_used = make_new_event(line, info)
        events.append(new_event)
        names_used_in_dynamics += names_used

        # Add newly assigned variables to the information set
        for var in new_event.assigns:
            if var in info.keys():
                raise ValueError(var + " is assigned, but already exists!")
            info[var] = 0

        # If any assigned variables are common, mark the event as common
        for var in new_event.assigns:
            if var in common:
                new_event.common = True
                break  # No need to check further

    # Remove content that is never referenced within the dynamics
    delete_these = []
    for name in content.keys():
        if name not in names_used_in_dynamics:
            delete_these.append(name)
    for name in delete_these:
        del content[name]

    # Make a single string model statement
    statement = ""
    longest = np.max([len(event.statement) for event in events])
    for event in events:
        this_statement = event.statement
        L = len(this_statement)
        pad = (longest + 1) - L
        statement += this_statement + pad * " " + ": " + event.description + "\n"

    # Make a description for the template block
    if name is None:
        description = "template block for unnamed block"
    else:
        description = "template block for " + name

    # Make and return the new SimBlock
    template_block = SimBlock(
        description=description,
        arrival=arrival,
        content=content,
        statement=statement,
        events=events,
    )
    return template_block, info, offset, solution, comments


def make_initializer(model, arrival=None, common=None):
    """
    Construct a new SimBlock object to be the agent initializer, based on the
    model dictionary. It has structure and events, but no parameters (etc).

    Parameters
    ----------
    model : dict
        Dictionary with model initializer information, probably read in as a yaml.
    arrival : [str]
        List of arrival variables that were flagged or explicitly listed.

    Returns
    -------
    initializer : SimBlock
        A "template" of this model block, with no parameters (etc) on it.
    init_requires : dict
        Dictionary of model objects that are needed by the initializer to run.
        Keys are object names and entries reveal what kind of object they are:
        - None --> parameter
        - 0 --> outcome variable (these should include all arrival variables)
        - NullFunc --> function
        - Distribution --> distribution
    """
    if arrival is None:
        arrival = []
    if common is None:
        common = []
    try:
        name = model["name"]
    except:
        name = "DEFAULT_NAME"

    # Extract parameters, functions, and distributions
    parameters = {}
    if "parameters" in model["symbols"].keys():
        param_lines = model["symbols"]["parameters"]
        for line in param_lines:
            param_name, datatype, flags, desc = parse_declaration_for_parts(line)
            parameters[param_name] = None

    functions = {}
    if "functions" in model["symbols"].keys():
        func_lines = model["symbols"]["functions"]
        for line in func_lines:
            func_name, datatype, flags, desc = parse_declaration_for_parts(line)
            if (datatype is not None) and (datatype != "func"):
                raise ValueError(
                    func_name
                    + " was declared as a function, but given a different datatype!"
                )
            functions[func_name] = NullFunc()

    distributions = {}
    if "distributions" in model["symbols"].keys():
        dstn_lines = model["symbols"]["distributions"]
        for line in dstn_lines:
            dstn_name, datatype, flags, desc = parse_declaration_for_parts(line)
            if (datatype is not None) and (datatype != "dstn"):
                raise ValueError(
                    dstn_name
                    + " was declared as a distribution, but given a different datatype!"
                )
            distributions[dstn_name] = Distribution()

    # Combine those dictionaries into a single "information" dictionary
    content = parameters.copy()
    content.update(functions)
    content.update(distributions)
    info = deepcopy(content)

    # Parse the initialization routine
    initialize = format_block_statement(model["initialize"])

    # Make the list of ordered events
    events = []
    names_used_in_initialize = []  # this doesn't actually get used
    for line in initialize:
        # Make the new event and add it to the list
        new_event, names_used = make_new_event(line, info)
        events.append(new_event)
        names_used_in_initialize += names_used

        # Add newly assigned variables to the information set
        for var in new_event.assigns:
            if var in info.keys():
                raise ValueError(var + " is assigned, but already exists!")
            info[var] = 0

        # If any assigned variables are common, mark the event as common
        for var in new_event.assigns:
            if var in common:
                new_event.common = True
                break  # No need to check further

    # Verify that all arrival variables were created in the initializer
    for var in arrival:
        if var not in info.keys():
            raise ValueError(
                "The arrival variable " + var + " was not set in the initialize block!"
            )

    # Make a blank dictionary with information the initializer needs
    init_requires = {}
    for event in events:
        for var in event.parameters.keys():
            if var not in init_requires.keys():
                try:
                    init_requires[var] = parameters[var]
                except:
                    raise ValueError(
                        var
                        + " was referenced in initialize, but not declared as a parameter!"
                    )
        if type(event) is RandomEvent:
            try:
                dstn_name = event._dstn_name
                init_requires[dstn_name] = distributions[dstn_name]
            except:
                raise ValueError(
                    dstn_name
                    + " was referenced in initialize, but not declared as a distribution!"
                )
        if type(event) is EvaluationEvent:
            try:
                func_name = event._func_name
                init_requires[dstn_name] = functions[func_name]
            except:
                raise ValueError(
                    func_name
                    + " was referenced in initialize, but not declared as a function!"
                )

    # Make a single string initializer statement
    statement = ""
    longest = np.max([len(event.statement) for event in events])
    for event in events:
        this_statement = event.statement
        L = len(this_statement)
        pad = (longest + 1) - L
        statement += this_statement + pad * " " + ": " + event.description + "\n"

    # Make and return the new SimBlock
    initializer = SimBlock(
        description="agent initializer for " + name,
        content=init_requires,
        statement=statement,
        events=events,
    )
    return initializer, init_requires


def make_new_event(statement, info):
    """
    Makes a "blank" version of a model event based on a statement line. Determines
    which objects are needed vs assigned vs parameters / information from context.

    Parameters
    ----------
    statement : str
        One line of a model statement, which will be turned into an event.
    info : dict
        Empty dictionary of model information that already exists. Consists of
        arrival variables, already assigned variables, parameters, and functions.
        Typing of each is based on the kind of "empty" object.

    Returns
    -------
    new_event : ModelEvent
        A new model event with values and information missing, but structure set.
    names_used : [str]
        List of names of objects used in this expression.
    """
    # First determine what kind of event this is
    has_eq = "=" in statement
    has_tld = "~" in statement
    has_amp = "@" in statement
    has_brc = ("{" in statement) and ("}" in statement)
    has_brk = ("[" in statement) and ("]" in statement)
    event_type = None
    if has_eq:
        if has_tld:
            raise ValueError("A statement line can't have both an = and a ~!")
        if has_amp:
            event_type = EvaluationEvent
        else:
            event_type = DynamicEvent
    if has_tld:
        if has_brc:
            event_type = MarkovEvent
        elif has_brk:
            event_type = RandomIndexedEvent
        else:
            event_type = RandomEvent
    if event_type is None:
        raise ValueError("Statement line was not any valid type!")

    # Now make and return an appropriate event for that type
    if event_type is DynamicEvent:
        event_maker = make_new_dynamic
    if event_type is RandomEvent:
        event_maker = make_new_random
    if event_type is RandomIndexedEvent:
        event_maker = make_new_random_indexed
    if event_type is MarkovEvent:
        event_maker = make_new_markov
    if event_type is EvaluationEvent:
        event_maker = make_new_evaluation

    new_event, names_used = event_maker(statement, info)
    return new_event, names_used


def make_new_dynamic(statement, info):
    """
    Construct a new instance of DynamicEvent based on the given model statement
    line and a blank dictionary of parameters. The statement should already be
    verified to be a valid dynamic statement: it has an = but no ~ or @.

    Parameters
    ----------
    statement : str
        One line dynamics statement, which will be turned into a DynamicEvent.
    info : dict
        Empty dictionary of available information.

    Returns
    -------
    new_dynamic : DynamicEvent
        A new dynamic event with values and information missing, but structure set.
    names_used : [str]
        List of names of objects used in this expression.
    """
    # Cut the statement up into its LHS, RHS, and description
    lhs, rhs, description = parse_line_for_parts(statement, "=")

    # Parse the LHS (assignment) to get assigned variables
    assigns = parse_assignment(lhs)

    # Parse the RHS (dynamic statement) to extract object names used
    obj_names, is_indexed = extract_var_names_from_expr(rhs)

    # Allocate each variable to needed dynamic variables or parameters
    needs = []
    parameters = {}
    for j in range(len(obj_names)):
        var = obj_names[j]
        if var not in info.keys():
            raise ValueError(
                var + " is used in a dynamic expression, but does not (yet) exist!"
            )
        val = info[var]
        if type(val) is NullFunc:
            raise ValueError(
                var + " is used in a dynamic expression, but it's a function!"
            )
        if type(val) is Distribution:
            raise ValueError(
                var + " is used in a dynamic expression, but it's a distribution!"
            )
        if val is None:
            parameters[var] = None
        else:
            needs.append(var)

    # Declare a SymPy symbol for each variable used; these are temporary
    _args = []
    for j in range(len(obj_names)):
        _var = obj_names[j]
        if is_indexed[j]:
            exec(_var + " = IndexedBase('" + _var + "')")
        else:
            exec(_var + " = symbols('" + _var + "')")
        _args.append(symbols(_var))

    # Make a SymPy expression, then lambdify it
    sympy_expr = symbols(rhs)
    expr = lambdify(_args, sympy_expr)

    # Make an overall list of object names referenced in this event
    names_used = assigns + obj_names

    # Make and return the new dynamic event
    new_dynamic = DynamicEvent(
        description=description,
        statement=lhs + " = " + rhs,
        assigns=assigns,
        needs=needs,
        parameters=parameters,
        expr=expr,
        args=obj_names,
    )
    return new_dynamic, names_used


def make_new_random(statement, info):
    """
    Make a new random variable realization event based on the given model statement
    line and a blank dictionary of parameters. The statement should already be
    verified to be a valid random statement: it has a ~ but no = or [].

    Parameters
    ----------
    statement : str
        One line of the model statement, which will be turned into a random event.
    info : dict
        Empty dictionary of available information.

    Returns
    -------
    new_random : RandomEvent
        A new random event with values and information missing, but structure set.
    names_used : [str]
        List of names of objects used in this expression.
    """
    # Cut the statement up into its LHS, RHS, and description
    lhs, rhs, description = parse_line_for_parts(statement, "~")

    # Parse the LHS (assignment) to get assigned variables
    assigns = parse_assignment(lhs)

    # Verify that the RHS is actually a distribution
    if type(info[rhs]) is not Distribution:
        raise ValueError(
            rhs + " was treated as a distribution, but not declared as one!"
        )

    # Make an overall list of object names referenced in this event
    names_used = assigns + [rhs]

    # Make and return the new random event
    new_random = RandomEvent(
        description=description,
        statement=lhs + " ~ " + rhs,
        assigns=assigns,
        needs=[],
        parameters={},
        dstn=info[rhs],
    )
    new_random._dstn_name = rhs
    return new_random, names_used


def make_new_random_indexed(statement, info):
    """
    Make a new indexed random variable realization event based on the given model
    statement line and a blank dictionary of parameters. The statement should
    already be verified to be a valid random statement: it has a ~ and [].

    Parameters
    ----------
    statement : str
        One line of the model statement, which will be turned into a random event.
    info : dict
        Empty dictionary of available information.

    Returns
    -------
    new_random_indexed : RandomEvent
        A new random indexed event with values and information missing, but structure set.
    names_used : [str]
        List of names of objects used in this expression.
    """
    # Cut the statement up into its LHS, RHS, and description
    lhs, rhs, description = parse_line_for_parts(statement, "~")

    # Parse the LHS (assignment) to get assigned variables
    assigns = parse_assignment(lhs)

    # Split the RHS into the distribution and the index
    dstn, index = parse_random_indexed(rhs)

    # Verify that the RHS is actually a distribution
    if type(info[dstn]) is not Distribution:
        raise ValueError(
            dstn + " was treated as a distribution, but not declared as one!"
        )

    # Make an overall list of object names referenced in this event
    names_used = assigns + [dstn, index]

    # Make and return the new random indexed event
    new_random_indexed = RandomIndexedEvent(
        description=description,
        statement=lhs + " ~ " + rhs,
        assigns=assigns,
        needs=[index],
        parameters={},
        index=index,
    )
    new_random_indexed._dstn_name = dstn
    return new_random_indexed, names_used


def make_new_markov(statement, info):
    """
    Make a new Markov-type event based on the given model statement line and a
    blank dictionary of parameters. The statement should already be verified to
    be a valid Markov statement: it has a ~ and {} and maybe (). This can represent
    a Markov matrix transition event, a draw from a discrete index, or just a
    Bernoulli random variable. If a Bernoulli event, the "probabilties" can be
    idiosyncratic data.

    Parameters
    ----------
    statement : str
        One line of the model statement, which will be turned into a random event.
    info : dict
        Empty dictionary of available information.

    Returns
    -------
    new_markov : MarkovEvent
        A new Markov draw event with values and information missing, but structure set.
    names_used : [str]
        List of names of objects used in this expression.
    """
    # Cut the statement up into its LHS, RHS, and description
    lhs, rhs, description = parse_line_for_parts(statement, "~")

    # Parse the LHS (assignment) to get assigned variables
    assigns = parse_assignment(lhs)

    # Parse the RHS (Markov statement) for the array and index
    probs, index = parse_markov(rhs)
    if index is None:
        needs = []
    else:
        needs = [index]

    # Determine whether probs is an idiosyncratic variable or a parameter, and
    # set up the event to grab it appropriately
    if info[probs] is None:
        parameters = {probs: None}
    else:
        needs += [probs]
        parameters = {}

    # Make an overall list of object names referenced in this event
    names_used = assigns + needs + [probs]

    # Make and return the new Markov event
    new_markov = MarkovEvent(
        description=description,
        statement=lhs + " ~ " + rhs,
        assigns=assigns,
        needs=needs,
        parameters=parameters,
        probs=probs,
        index=index,
    )
    return new_markov, names_used


def make_new_evaluation(statement, info):
    """
    Make a new function evaluation event based the given model statement line
    and a blank dictionary of parameters. The statement should already be verified
    to be a valid evaluation statement: it has an @ and an = but no ~.

    Parameters
    ----------
    statement : str
        One line of the model statement, which will be turned into an eval event.
    info : dict
        Empty dictionary of available information.

    Returns
    -------
    new_evaluation : EvaluationEvent
        A new evaluation event with values and information missing, but structure set.
    names_used : [str]
        List of names of objects used in this expression.
    """
    # Cut the statement up into its LHS, RHS, and description
    lhs, rhs, description = parse_line_for_parts(statement, "=")

    # Parse the LHS (assignment) to get assigned variables
    assigns = parse_assignment(lhs)

    # Parse the RHS (evaluation) for the function and its arguments
    func, arguments = parse_evaluation(rhs)

    # Allocate each variable to needed dynamic variables or parameters
    needs = []
    parameters = {}
    for j in range(len(arguments)):
        var = arguments[j]
        if var not in info.keys():
            raise ValueError(
                var + " is used in an evaluation statement, but does not (yet) exist!"
            )
        val = info[var]
        if type(val) is NullFunc:
            raise ValueError(
                var
                + " is used as an argument an evaluation statement, but it's a function!"
            )
        if type(val) is Distribution:
            raise ValueError(
                var + " is used in an evaluation statement, but it's a distribution!"
            )
        if val is None:
            parameters[var] = None
        else:
            needs.append(var)

    # Make an overall list of object names referenced in this event
    names_used = assigns + arguments + [func]

    # Make and return the new evaluation event
    new_evaluation = EvaluationEvent(
        description=description,
        statement=lhs + " = " + rhs,
        assigns=assigns,
        needs=needs,
        parameters=parameters,
        arguments=arguments,
        func=info[func],
    )
    new_evaluation._func_name = func
    return new_evaluation, names_used


def look_for_char_and_remove(phrase, symb):
    """
    Check whether a symbol appears in a string, and remove it if it does.

    Parameters
    ----------
    phrase : str
        String to be searched for a symbol.
    symb : char
        Single character to be searched for.

    Returns
    -------
    out : str
        Possibly shortened input phrase.
    found : bool
        Whether the symbol was found and removed.
    """
    found = symb in phrase
    out = phrase.replace(symb, "")
    return out, found


def parse_declaration_for_parts(line):
    """
    Split a declaration line from a model file into the object's name, its datatype,
    any metadata flags, and any provided comment or description.

    Parameters
    ----------
    line : str
        Line of to be parsed into the object name, object type, and a comment or description.

    Returns
    -------
    name : str
        Name of the object.
    datatype : str or None
        Provided datatype string, in parentheses, if any.
    flags : [str]
        List of metadata flags that were detected. These include ! for a variable
        that is in arrival, * for any non-variable that's part of the solution,
        + for any object that is offset in time, and & for a common random variable.

    desc : str
        Comment or description, after //, if any.
    """
    flags = []
    check_for_flags = {"offset": "+", "arrival": "!", "solution": "*", "common": "&"}

    # First, separate off the comment or description, if any
    slashes = line.find("\\")
    desc = "" if slashes == -1 else line[(slashes + 2) :].strip()
    rem = line if slashes == -1 else line[:slashes].strip()

    # Now look for bracketing parentheses declaring a datatype
    lp = rem.find("(")
    if lp > -1:
        rp = rem.find(")")
        if rp == -1:
            raise ValueError("Unclosed parentheses on object declaration line!")
        datatype = rem[(lp + 1) : rp].strip()
        leftover = rem[:lp].strip()
    else:
        datatype = None
        leftover = rem

    # What's left over should be the object name plus any flags
    for key in check_for_flags.keys():
        symb = check_for_flags[key]
        leftover, found = look_for_char_and_remove(leftover, symb)
        if found:
            flags.append(key)

    # Remove any remaining spaces, and that *should* be the name
    name = leftover.replace(" ", "")
    # TODO: Check for valid name formatting based on characters.

    return name, datatype, flags, desc


def parse_line_for_parts(statement, symb):
    """
    Split one line of a model statement into its LHS, RHS, and description. The
    description is everything following \\, while the LHS and RHS are determined
    by a special symbol.

    Parameters
    ----------
    statement : str
        One line of a model statement, which will be parsed for its parts.
    symb : char
        The character that represents the divide between LHS and RHS

    Returns
    -------
    lhs : str
        The left-hand (assignment) side of the expression.
    rhs : str
        The right-hand (evaluation) side of the expression.
    desc : str
        The provided description of the expression.
    """
    eq = statement.find(symb)
    lhs = statement[:eq].replace(" ", "")
    not_lhs = statement[(eq + 1) :]
    comment = not_lhs.find("\\")
    desc = "" if comment == -1 else not_lhs[(comment + 2) :].strip()
    rhs = not_lhs if comment == -1 else not_lhs[:comment]
    rhs = rhs.replace(" ", "")
    return lhs, rhs, desc


def parse_assignment(lhs):
    """
    Get ordered list of assigned variables from the LHS of a model line.

    Parameters
    ----------
    lhs : str
        Left-hand side of a model expression

    Returns
    -------
    assigns : List[str]
        List of variable names that are assigned in this model line.
    """
    if lhs[0] == "(":
        if not lhs[-1] == ")":
            raise ValueError("Parentheses on assignment was not closed!")
        assigns = []
        pos = 0
        while pos != -1:
            pos += 1
            end = lhs.find(",", pos)
            var = lhs[pos:end]
            if var != "":
                assigns.append(var)
            pos = end
    else:
        assigns = [lhs]
    return assigns


def extract_var_names_from_expr(expression):
    """
    Parse the RHS of a dynamic model statement to get variable names used in it.

    Parameters
    ----------
    expression : str
        RHS of a model statement to be parsed for variable names.

    Returns
    -------
    var_names : List[str]
        List of variable names used in the expression. These *should* be dynamic
        variables and parameters, but not functions.
    indexed : List[bool]
        Indicators for whether each variable seems to be used with indexing.
    """
    var_names = []
    indexed = []
    math_symbols = "+-/*^%.(),[]{}<>"
    digits = "01234567890"
    cur = ""
    for j in range(len(expression)):
        c = expression[j]
        if (c in math_symbols) or ((c in digits) and cur == ""):
            if cur == "":
                continue
            if cur in var_names:
                cur = ""
                continue
            var_names.append(cur)
            if c == "[":
                indexed.append(True)
            else:
                indexed.append(False)
            cur = ""
        else:
            cur += c
    if cur != "" and cur not in var_names:
        var_names.append(cur)
        indexed.append(False)  # final symbol couldn't possibly be indexed
    return var_names, indexed


def parse_evaluation(expression):
    """
    Separate a function evaluation expression into the function that is called
    and the variable inputs that are passed to it.

    Parameters
    ----------
    expression : str
        RHS of a function evaluation model statement, which will be parsed for
        the function and its inputs.

    Returns
    -------
    func_name : str
        Name of the function that will be called in this event.
    arg_names : List[str]
        List of arguments of the function.
    """
    # Get the name of the function: what's to the left of the @
    amp = expression.find("@")
    func_name = expression[:amp]

    # Check for parentheses formatting
    rem = expression[(amp + 1) :]
    if not rem[0] == "(":
        raise ValueError(
            "The @ in a function evaluation statement must be followed by (!"
        )
    if not rem[-1] == ")":
        raise ValueError("A function evaluation statement must end in )!")
    rem = rem[1:-1]

    # Parse what's inside the parentheses for argument names
    arg_names = []
    pos = 0
    go = True
    while go:
        end = rem.find(",", pos)
        if end > -1:
            arg = rem[pos:end]
        else:
            arg = rem[pos:]
            go = False
        if arg != "":
            arg_names.append(arg)
        pos = end + 1

    return func_name, arg_names


def parse_markov(expression):
    """
    Separate a Markov draw declaration into the array of probabilities and the
    index for idiosyncratic values.

    Parameters
    ----------
    expression : str
        RHS of a function evaluation model statement, which will be parsed for
        the probabilities name and index name.

    Returns
    -------
    probs : str
        Name of the probabilities object in this statement.
    index : str
        Name of the indexing variable in this statement.
    """
    # Get the name of the probabilitie
    lb = expression.find("{")  # this *should* be 0
    rb = expression.find("}")
    if lb == -1 or rb == -1 or rb < (lb + 2):
        raise ValueError("A Markov assignment must have an {array}!")
    probs = expression[(lb + 1) : rb]

    # Get the name of the index, if any
    x = rb + 1
    lp = expression.find("(", x)
    rp = expression.find(")", x)
    if lp == -1 and rp == -1:  # no index present at all
        return probs, None
    if lp == -1 or rp == -1 or rp < (lp + 2):
        raise ValueError("Improper Markov formatting: should be {probs}(index)!")
    index = expression[(lp + 1) : rp]

    return probs, index


def parse_random_indexed(expression):
    """
    Separate an indexed random variable assignment into the distribution and
    the index for it.

    Parameters
    ----------
    expression : str
        RHS of a function evaluation model statement, which will be parsed for
        the distribution name and index name.

    Returns
    -------
    dstn : str
        Name of the distribution in this statement.
    index : str
        Name of the indexing variable in this statement.
    """
    # Get the name of the index
    lb = expression.find("[")
    rb = expression.find("]")
    if lb == -1 or rb == -1 or rb < (lb + 2):
        raise ValueError("An indexed random variable assignment must have an [index]!")
    index = expression[(lb + 1) : rb]

    # Get the name of the distribution
    dstn = expression[:lb]

    return dstn, index


def format_block_statement(statement):
    """
    Ensure that a string stagement of a model block (maybe a period, maybe an
    initializer) is formatted as a list of strings, one statement per entry.

    Parameters
    ----------
    statement : str
        A model statement, which might be for a block or an initializer. The
        statement might be formatted as a list or as a single string.

    Returns
    -------
    block_statements: [str]
        A list of model statements, one per entry.
    """
    if type(statement) is str:
        if statement.find("\n") > -1:
            block_statements = []
            pos = 0
            end = statement.find("\n", pos)
            while end > -1:
                new_line = statement[pos:end]
                block_statements.append(new_line)
                pos = end + 1
                end = statement.find("\n", pos)
        else:
            block_statements = [statement.copy()]
    if type(statement) is list:
        for line in statement:
            if type(line) is not str:
                raise ValueError("The model statement somehow includes a non-string!")
        block_statements = statement.copy()
    return block_statements


@njit
def aggregate_blobs_onto_polynomial_grid(
    vals, pmv, origins, grid, J, Q
):  # pragma: no cover
    """
    Numba-compatible helper function for casting "probability blobs" onto a discretized
    grid of outcome values, based on their origin in the arrival state space. This
    version is for non-continuation variables, returning only the probability array
    mapping from arrival states to the outcome variable.
    """
    bot = grid[0]
    top = grid[-1]
    M = grid.size
    Mm1 = M - 1
    N = pmv.size
    scale = 1.0 / (top - bot)
    order = 1.0 / Q
    diffs = grid[1:] - grid[:-1]

    probs = np.zeros((J, M))

    for n in range(N):
        x = vals[n]
        jj = origins[n]
        p = pmv[n]
        if (x > bot) and (x < top):
            ii = int(np.floor(((x - bot) * scale) ** order * Mm1))
            temp = (x - grid[ii]) / diffs[ii]
            probs[jj, ii] += (1.0 - temp) * p
            probs[jj, ii + 1] += temp * p
        elif x <= bot:
            probs[jj, 0] += p
        else:
            probs[jj, -1] += p
    return probs


@njit
def aggregate_blobs_onto_polynomial_grid_alt(
    vals, pmv, origins, grid, J, Q
):  # pragma: no cover
    """
    Numba-compatible helper function for casting "probability blobs" onto a discretized
    grid of outcome values, based on their origin in the arrival state space. This
    version is for ncontinuation variables, returning the probability array mapping
    from arrival states to the outcome variable, the index in the outcome variable grid
    for each blob, and the alpha weighting between gridpoints.
    """
    bot = grid[0]
    top = grid[-1]
    M = grid.size
    Mm1 = M - 1
    N = pmv.size
    scale = 1.0 / (top - bot)
    order = 1.0 / Q
    diffs = grid[1:] - grid[:-1]

    probs = np.zeros((J, M))
    idx = np.empty(N, dtype=np.dtype(np.int32))
    alpha = np.empty(N)

    for n in range(N):
        x = vals[n]
        jj = origins[n]
        p = pmv[n]
        if (x > bot) and (x < top):
            ii = int(np.floor(((x - bot) * scale) ** order * Mm1))
            temp = (x - grid[ii]) / diffs[ii]
            probs[jj, ii] += (1.0 - temp) * p
            probs[jj, ii + 1] += temp * p
            alpha[n] = temp
            idx[n] = ii
        elif x <= bot:
            probs[jj, 0] += p
            alpha[n] = 0.0
            idx[n] = 0
        else:
            probs[jj, -1] += p
            alpha[n] = 1.0
            idx[n] = M - 2
    return probs, idx, alpha


@njit
def aggregate_blobs_onto_discrete_grid(vals, pmv, origins, M, J):  # pragma: no cover
    """
    Numba-compatible helper function for allocating "probability blobs" to a grid
    over a discrete state-- the state itself is truly discrete.
    """
    out = np.zeros((J, M))
    N = pmv.size
    for n in range(N):
        ii = vals[n]
        jj = origins[n]
        p = pmv[n]
        out[jj, ii] += p
    return out


@njit
def calc_overall_trans_probs(
    out, idx, alpha, binary, offset, pmv, origins
):  # pragma: no cover
    """
    Numba-compatible helper function for combining transition probabilities from
    the arrival state space to *multiple* continuation variables into a single
    unified transition matrix.
    """
    N = alpha.shape[0]
    B = binary.shape[0]
    D = binary.shape[1]
    for n in range(N):
        ii = origins[n]
        jj_base = idx[n]
        p = pmv[n]
        for b in range(B):
            adj = offset[b]
            P = p
            for d in range(D):
                k = binary[b, d]
                P *= alpha[n, d, k]
            jj = jj_base + adj
            out[ii, jj] += P
    return out
