from dataclasses import dataclass
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import xarray as xr

from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsIndShockSetup,
    IndShockConsumerType,
    init_perfect_foresight,
)
from HARK.ConsumptionSaving.ConsPortfolioModel import (
    PortfolioConsumerType,
    init_portfolio,
)
from HARK.ConsumptionSaving.ConsRiskyAssetModel import (
    FixedPortfolioShareRiskyAssetConsumerType,
    RiskyAssetConsumerType,
    init_risky_asset,
    init_risky_share_fixed,
)
from HARK.core import MetricObject, make_one_period_oo_solver
from HARK.distribution import DiscreteDistributionLabeled
from HARK.rewards import UtilityFuncCRRA


class ValueFuncCRRALabeled(MetricObject):
    """
    Class to allow for value function interpolation using xarray.
    """

    def __init__(self, dataset: xr.Dataset, CRRA: float):
        """
        Initialize a value function.

        Parameters
        ----------
        dataset : xr.Dataset
            Underlying dataset that should include a  variable named
            "v_inv" that is the inverse of the value function.

        CRRA : float
            Coefficient of relative risk aversion.
        """

        self.dataset = dataset
        self.CRRA = CRRA
        self.u = UtilityFuncCRRA(CRRA)

    def __call__(self, state: Mapping[str, np.ndarray]) -> xr.Dataset:
        """
        Interpolate inverse value function then invert to get value function at given state.

        Parameters
        ----------
        state : Mapping[str, np.ndarray]
            State to evaluate value function at.

        Returns
        -------
        result : xr.Dataset
        """

        state_dict = self._validate_state(state)

        result = self.u(
            self.dataset["v_inv"].interp(
                state_dict,
                assume_sorted=True,
                kwargs={"fill_value": "extrapolate"},
            )
        )

        result.name = "v"
        result.attrs = self.dataset["v"].attrs

        return result

    def derivative(self, state):
        """
        Interpolate inverse marginal value function then invert to get marginal value function at given state.

        Parameters
        ----------
        state : Mapping[str, np.ndarray]
            State to evaluate marginal value function at.

        Returns
        -------
        result : xr.Dataset
        """

        state_dict = self._validate_state(state)

        result = self.u.der(
            self.dataset["v_der_inv"].interp(
                state_dict,
                assume_sorted=True,
                kwargs={"fill_value": "extrapolate"},
            )
        )

        result.name = "v_der"
        result.attrs = self.dataset["v"].attrs

        return result

    def evaluate(self, state):
        """
        Interpolate all data variables in the dataset.

        Parameters
        ----------
        state : Mapping[str, np.ndarray]
            State to evaluate all data variables at.

        Returns
        -------
        result : xr.Dataset
        """

        state_dict = self._validate_state(state)

        result = self.dataset.interp(
            state_dict,
            kwargs={"fill_value": None},
        )
        result.attrs = self.dataset["v"].attrs

        return result

    def _validate_state(self, state):
        """
        Allowed states are either a dict or an xr.Dataset.
        This methods keeps only the coordinates of the dataset
        if they are both in the dataset and the input state.

        Parameters
        ----------
        state : Mapping[str, np.ndarray]
            State to validate.

        Returns
        -------
        state_dict : dict
        """

        if isinstance(state, (xr.Dataset, dict)):
            state_dict = {}
            for coords in self.dataset.coords.keys():
                state_dict[coords] = state[coords]
        else:
            raise ValueError("state must be a dict or xr.Dataset")

        return state_dict


class ConsumerSolutionLabeled(MetricObject):
    """
    Class to allow for solution interpolation using xarray.
    Represents a solution object for labeled models.
    """

    def __init__(
        self,
        value: ValueFuncCRRALabeled,
        policy: xr.Dataset,
        continuation: ValueFuncCRRALabeled,
        attrs=None,
    ):
        """
        Consumer Solution for labeled models.

        Parameters
        ----------
        value : ValueFuncCRRALabeled
            Value function and marginal value function.
        policy : xr.Dataset
            Policy function.
        continuation : ValueFuncCRRALabeled
            Continuation value function and marginal value function.
        attrs : _type_, optional
            Attributes of the solution. The default is None.
        """

        if attrs is None:
            attrs = dict()

        self.value = value  # value function
        self.policy = policy  # policy function
        self.continuation = continuation  # continuation function

        self.attrs = attrs

    def distance(self, other: "ConsumerSolutionLabeled"):
        """
        Compute the distance between two solutions.

        Parameters
        ----------
        other : ConsumerSolutionLabeled
            Other solution to compare to.

        Returns
        -------
        float
            Distance between the two solutions.
        """

        # TODO: is there a faster way to compare two xr.Datasets?

        value = self.value.dataset
        other_value = other.value.dataset.interp_like(value)

        return np.max(np.abs(value - other_value).to_array())


class PerfForesightLabeledType(IndShockConsumerType):
    """
    A labeled perfect foresight consumer type. This class is a subclass of
    IndShockConsumerType, and inherits all of its methods and attributes.

    Perfect foresight consumers have no uncertainty about income or interest
    rates, and so the only state variable is market resources m.
    """

    def __init__(self, verbose=1, quiet=False, **kwds):
        """
        Initialize a new instance of a perfect foresight consumer type.
        """

        params = init_perfect_foresight.copy()
        params.update(kwds)

        # Initialize a basic AgentType
        IndShockConsumerType.__init__(self, verbose=verbose, quiet=quiet, **params)

        self.solve_one_period = make_one_period_oo_solver(
            ConsPerfForesightLabeledSolver
        )

    def update_solution_terminal(self):
        """
        Update the terminal solution of the model by creating a terminal
        value function and terminal marginal value function along with
        a terminal policy function.
        """

        u = UtilityFuncCRRA(self.CRRA)

        mNrm = xr.DataArray(
            np.append(0.0, self.aXtraGrid),
            name="mNrm",
            dims=("mNrm"),
            attrs={"long_name": "cash_on_hand"},
        )
        state = xr.Dataset({"mNrm": mNrm})  # only one state var in this model

        # optimal decision is to consume everything in the last period
        cNrm = xr.DataArray(
            mNrm,
            name="cNrm",
            dims=state.dims,
            coords=state.coords,
            attrs={"long_name": "consumption"},
        )

        v = u(cNrm)
        v.name = "v"
        v.attrs = {"long_name": "value function"}

        v_der = u.der(cNrm)
        v_der.name = "v_der"
        v_der.attrs = {"long_name": "marginal value function"}

        v_inv = cNrm.copy()
        v_inv.name = "v_inv"
        v_inv.attrs = {"long_name": "inverse value function"}

        v_der_inv = cNrm.copy()
        v_der_inv.name = "v_der_inv"
        v_der_inv.attrs = {"long_name": "inverse marginal value function"}

        dataset = xr.Dataset(
            {
                "cNrm": cNrm,
                "v": v,
                "v_der": v_der,
                "v_inv": v_inv,
                "v_der_inv": v_der_inv,
            }
        )

        vfunc = ValueFuncCRRALabeled(
            dataset[["v", "v_der", "v_inv", "v_der_inv"]], self.CRRA
        )

        self.solution_terminal = ConsumerSolutionLabeled(
            value=vfunc,
            policy=dataset[["cNrm"]],
            continuation=None,
            attrs={"m_nrm_min": 0.0},  # minimum normalized market resources
        )


class ConsPerfForesightLabeledSolver(ConsIndShockSetup):
    """
    Solver for PerfForeshightLabeledType.
    """

    def create_params_namespace(self):
        """
        Create a namespace for parameters.
        """

        self.params = SimpleNamespace(
            Discount=self.DiscFac * self.LivPrb,
            CRRA=self.CRRA,
            Rfree=self.Rfree,
            PermGroFac=self.PermGroFac,
        )

    def calculate_borrowing_constraint(self):
        """
        Calculate the minimum allowable value of money resources in this period.
        """

        self.BoroCnstNat = (
            self.solution_next.attrs["m_nrm_min"] - 1
        ) / self.params.Rfree

    def define_boundary_constraint(self):
        """
        If the natural borrowing constraint is a binding constraint,
        then we can not evaluate the value function at that point,
        so we must fill out the data by hand.
        """

        if self.BoroCnstArt is None or self.BoroCnstArt <= self.BoroCnstNat:
            self.m_nrm_min = self.BoroCnstNat
            self.nat_boro_cnst = True  # natural borrowing constraint is binding

            self.borocnst = xr.Dataset(
                coords={"mNrm": self.m_nrm_min, "aNrm": self.m_nrm_min},
                data_vars={
                    "cNrm": 0.0,
                    "v": -np.inf,
                    "v_inv": 0.0,
                    "reward": -np.inf,
                    "marginal_reward": np.inf,
                    "v_der": np.inf,
                    "v_der_inv": 0.0,
                },
            )

        elif self.BoroCnstArt > self.BoroCnstNat:
            self.m_nrm_min = self.BoroCnstArt
            self.nat_boro_cnst = False  # artificial borrowing constraint is binding

            self.borocnst = xr.Dataset(
                coords={"mNrm": self.m_nrm_min, "aNrm": self.m_nrm_min},
                data_vars={"cNrm": 0.0},
            )

    def create_post_state(self):
        """
        Create the post state variable, which in this case is
        the normalized assets saved this period.
        """

        if self.nat_boro_cnst:
            # don't include natural borrowing constraint
            a_grid = self.aXtraGrid + self.m_nrm_min
        else:
            # include artificial borrowing constraint
            a_grid = np.append(0.0, self.aXtraGrid) + self.m_nrm_min

        aVec = xr.DataArray(
            a_grid,
            name="aNrm",
            dims=("aNrm"),
            attrs={"long_name": "savings", "state": True},
        )
        post_state = xr.Dataset({"aNrm": aVec})

        self.post_state = post_state

    def state_transition(self, state=None, action=None, params=None):
        """
        State to post_state transition.

        Parameters
        ----------
        state : xr.Dataset
            State variables.
        action : xr.Dataset
            Action variables.
        params : SimpleNamespace
            Parameters.

        Returns
        -------
        post_state : xr.Dataset
            Post state variables.
        """

        post_state = {}  # pytree
        post_state["aNrm"] = state["mNrm"] - action["cNrm"]
        return post_state

    def post_state_transition(self, post_state=None, params=None):
        """
        Post_state to next_state transition.

        Parameters
        ----------
        post_state : xr.Dataset
            Post state variables.
        params : SimpleNamespace
            Parameters.

        Returns
        -------
        next_state : xr.Dataset
            Next period's state variables.
        """

        next_state = {}  # pytree
        next_state["mNrm"] = post_state["aNrm"] * params.Rfree / params.PermGroFac + 1
        return next_state

    def reverse_transition(self, post_state=None, action=None, params=None):
        """
        State from post state and actions.

        Parameters
        ----------
        post_state : xr.Dataset
            Post state variables.
        action : xr.Dataset
            Action variables.
        params : SimpleNamespace

        Returns
        -------
        state : xr.Dataset
            State variables.
        """

        state = {}  # pytree
        state["mNrm"] = post_state["aNrm"] + action["cNrm"]

        return state

    def egm_transition(self, post_state=None, continuation=None, params=None):
        """
        Actions from post state using the endogenous grid method.

        Parameters
        ----------
        post_state : xr.Dataset
            Post state variables.
        continuation : ValueFuncCRRALabeled
            Continuation value function, next period's value function.
        params : SimpleNamespace

        Returns
        -------
        action : xr.Dataset
            Action variables.
        """

        action = {}  # pytree
        action["cNrm"] = self.u.derinv(
            params.Discount * continuation.derivative(post_state)
        )

        return action

    def value_transition(self, action=None, state=None, continuation=None, params=None):
        """
        Value of action given state and continuation

        Parameters
        ----------
        action : xr.Dataset
            Action variables.
        state : xr.Dataset
            State variables.
        continuation : ValueFuncCRRALabeled
            Continuation value function, next period's value function.
        params : SimpleNamespace
            Parameters

        Returns
        -------
        variables : xr.Dataset
            Value, marginal value, reward, marginal reward, and contributions.
        """

        variables = {}  # pytree
        post_state = self.state_transition(state, action, params)
        variables.update(post_state)

        variables["reward"] = self.u(action["cNrm"])
        variables["v"] = variables["reward"] + params.Discount * continuation(
            post_state
        )
        variables["v_inv"] = self.u.inv(variables["v"])

        variables["marginal_reward"] = self.u.der(action["cNrm"])
        variables["v_der"] = variables["marginal_reward"]
        variables["v_der_inv"] = action["cNrm"]

        # for estimagic purposes
        variables["contributions"] = variables["v"]
        variables["value"] = np.sum(variables["v"])

        return variables

    def continuation_transition(self, post_state=None, value_next=None, params=None):
        """
        Continuation value function of post state.

        Parameters
        ----------
        post_state : xr.Dataset
            Post state variables.
        value_next : ValueFuncCRRALabeled
            Next period's value function.
        params : SimpleNamespace
            Parameters.

        Returns
        -------
        variables : xr.Dataset
            Value, marginal value, inverse value, and inverse marginal value.
        """

        variables = {}  # pytree
        next_state = self.post_state_transition(post_state, params)
        variables.update(next_state)
        variables["v"] = params.PermGroFac ** (1 - params.CRRA) * value_next(next_state)
        variables["v_der"] = (
            params.Rfree
            * params.PermGroFac ** (-params.CRRA)
            * value_next.derivative(next_state)
        )

        variables["v_inv"] = self.u.inv(variables["v"])
        variables["v_der_inv"] = self.u.derinv(variables["v_der"])

        # for estimagic purposes
        variables["contributions"] = variables["v"]
        variables["value"] = np.sum(variables["v"])

        return variables

    def prepare_to_solve(self):
        """
        Prepare to solve the model by creating the parameters namespace,
        calculating the borrowing constraint, defining the boundary constraint,
        and creating the post state.
        """

        self.create_params_namespace()
        self.calculate_borrowing_constraint()
        self.define_boundary_constraint()
        self.create_post_state()

    def create_continuation_function(self):
        """
        Create the continuation function, or the value function
        of every possible post state.

        Returns
        -------
        wfunc : ValueFuncCRRALabeled
            Continuation function.
        """

        # unpack next period's solution
        vfunc_next = self.solution_next.value

        v_end = self.continuation_transition(self.post_state, vfunc_next, self.params)
        # need to drop m because it's next period's m
        v_end = xr.Dataset(v_end).drop(["mNrm"])
        borocnst = self.borocnst.drop(["mNrm"]).expand_dims("aNrm")
        if self.nat_boro_cnst:
            v_end = xr.merge([borocnst, v_end])

        wfunc = ValueFuncCRRALabeled(v_end, self.params.CRRA)

        return wfunc

    def endogenous_grid_method(self):
        """
        Solve the model using the endogenous grid method, which consists of
        solving the model backwards in time using the following steps:

        1. Create the continuation function, or the value function of every
            possible post state.
        2. Get the optimal actions/decisions from the endogenous grid transition.
        3. Get the state from the actions and post state using the reverse transition.
        4. EGM requires swapping dimensions; make actions and state functions of state.
        5. Merge the actions and state into a single dataset.
        6. If the natural borrowing constraint is not used, concatenate the
            borrowing constraint to the dataset.
        7. Create the value function from the variables in the dataset.
        8. Create the policy function from the variables in the dataset.
        9. Create the solution from the value and policy functions.
        """
        wfunc = self.create_continuation_function()

        # get optimal actions/decisions from egm
        acted = self.egm_transition(self.post_state, wfunc, self.params)
        # get state from actions and post_state
        state = self.reverse_transition(self.post_state, acted, self.params)

        # egm requires swap dimensions; make actions and state functions of state
        action = xr.Dataset(acted).swap_dims({"aNrm": "mNrm"})
        state = xr.Dataset(state).swap_dims({"aNrm": "mNrm"})

        egm_dataset = xr.merge([action, state])

        if not self.nat_boro_cnst:
            egm_dataset = xr.concat([self.borocnst, egm_dataset], dim="mNrm")

        values = self.value_transition(egm_dataset, egm_dataset, wfunc, self.params)
        egm_dataset.update(values)

        if self.nat_boro_cnst:
            egm_dataset = xr.concat(
                [self.borocnst, egm_dataset], dim="mNrm", combine_attrs="no_conflicts"
            )

        egm_dataset = egm_dataset.drop("aNrm")

        vfunc = ValueFuncCRRALabeled(
            egm_dataset[["v", "v_der", "v_inv", "v_der_inv"]], self.params.CRRA
        )
        pfunc = egm_dataset[["cNrm"]]

        self.solution = ConsumerSolutionLabeled(
            value=vfunc,
            policy=pfunc,
            continuation=wfunc,
            attrs={"m_nrm_min": self.m_nrm_min, "dataset": egm_dataset},
        )

    def solve(self):
        """
        Solve the model by endogenous grid method.
        """

        self.endogenous_grid_method()

        return self.solution


class IndShockLabeledType(PerfForesightLabeledType):
    """
    A labeled version of IndShockConsumerType. This class inherits from
    PerfForesightLabeledType and adds income uncertainty.
    """

    def __init__(self, verbose=1, quiet=False, **kwds):
        """
        Initialize an instance of IndShockLabeledType.
        """

        params = init_perfect_foresight.copy()
        params.update(kwds)

        # Initialize a basic AgentType
        PerfForesightLabeledType.__init__(self, verbose=verbose, quiet=quiet, **params)

        self.solve_one_period = make_one_period_oo_solver(ConsIndShockLabeledSolver)

        self.update_labeled_type()

    def update_labeled_type(self):
        """
        Update the labeled type by creating labeled versions
        of the distributions.
        """

        self.update_distributions()

    def update_distributions(self):
        """
        Create labeled versions of the distributions.
        """

        IncShkDstn = []

        for i in range(len(self.IncShkDstn.dstns)):
            IncShkDstn.append(
                DiscreteDistributionLabeled.from_unlabeled(
                    self.IncShkDstn[i],
                    name="Distribution of Shocks to Income",
                    var_names=["perm", "tran"],
                )
            )

        self.IncShkDstn = IncShkDstn


class ConsIndShockLabeledSolver(ConsPerfForesightLabeledSolver):
    """
    Solver for IndShockLabeledType.
    """

    def calculate_borrowing_constraint(self):
        """
        Calculate the minimum allowable value of money resources in this period.
        This is different from the perfect foresight natural borrowing constraint
        because of the presence of income uncertainty.
        """

        PermShkMinNext = np.min(self.IncShkDstn.atoms[0])
        TranShkMinNext = np.min(self.IncShkDstn.atoms[1])

        self.BoroCnstNat = (
            (self.solution_next.attrs["m_nrm_min"] - TranShkMinNext)
            * (self.params.PermGroFac * PermShkMinNext)
            / self.params.Rfree
        )

    def post_state_transition(self, post_state=None, shocks=None, params=None):
        """
        Post state to next state transition now depends on income shocks.

        Parameters
        ----------
        post_state : dict
            Post state variables.
        shocks : dict
            Shocks to income.
        params : dict
            Parameters.

        Returns
        -------
        next_state : dict
            Next period's state variables.
        """

        next_state = {}  # pytree
        next_state["mNrm"] = (
            post_state["aNrm"] * params.Rfree / (params.PermGroFac * shocks["perm"])
            + shocks["tran"]
        )
        return next_state

    def continuation_transition(
        self, shocks=None, post_state=None, v_next=None, params=None
    ):
        """
        Continuation value function of post state.

        Parameters
        ----------
        shocks : dict
            Shocks to income.
        post_state : dict
            Post state variables.
        v_next : ValueFuncCRRALabeled
            Next period's value function.
        params : dict
            Parameters.

        Returns
        -------
        variables : dict
            Continuation value function and its derivative.
        """

        variables = {}  # pytree
        next_state = self.post_state_transition(post_state, shocks, params)
        variables.update(next_state)

        variables["psi"] = params.PermGroFac * shocks["perm"]

        variables["v"] = variables["psi"] ** (1 - params.CRRA) * v_next(next_state)

        variables["v_der"] = (
            params.Rfree
            * variables["psi"] ** (-params.CRRA)
            * v_next.derivative(next_state)
        )

        # for estimagic purposes

        variables["contributions"] = variables["v"]
        variables["value"] = np.sum(variables["v"])

        return variables

    def create_continuation_function(self):
        """
        Create the continuation function. Because of the income uncertainty
        in this model, we need to integrate over the income shocks to get the
        continuation value function. Depending on the natural borrowing constraint,
        we may also have to append the minimum allowable value of money resources.

        Returns
        -------
        wfunc : ValueFuncCRRALabeled
            Continuation value function.
        """

        # unpack next period's solution
        vfunc_next = self.solution_next.value

        v_end = self.IncShkDstn.expected(
            func=self.continuation_transition,
            post_state=self.post_state,
            v_next=vfunc_next,
            params=self.params,
        )

        v_end["v_inv"] = self.u.inv(v_end["v"])
        v_end["v_der_inv"] = self.u.derinv(v_end["v_der"])

        borocnst = self.borocnst.drop(["mNrm"]).expand_dims("aNrm")
        if self.nat_boro_cnst:
            v_end = xr.merge([borocnst, v_end])

        # need to drop m because it's next period's m
        # v_end = xr.Dataset(v_end).drop(["mNrm"])
        wfunc = ValueFuncCRRALabeled(v_end, self.params.CRRA)

        return wfunc


class RiskyAssetLabeledType(IndShockLabeledType, RiskyAssetConsumerType):
    """
    A labeled RiskyAssetConsumerType. This class is a subclass of
    RiskyAssetConsumerType, and inherits all of its methods and attributes.

    Risky asset consumers can only save on a risky asset that
    pays a stochastic return.
    """

    def __init__(self, verbose=1, quiet=False, **kwds):
        """
        Initialize a labeled RiskyAssetConsumerType.
        """

        params = init_risky_asset.copy()
        params.update(kwds)

        # Initialize a basic AgentType
        RiskyAssetConsumerType.__init__(self, verbose=verbose, quiet=quiet, **params)

        self.solve_one_period = make_one_period_oo_solver(ConsRiskyAssetLabeledSolver)

        self.update_labeled_type()

    def update_distributions(self):
        """
        Update the labeled distributions including the Risky distribution.
        """

        super().update_distributions()

        self.RiskyDstn = DiscreteDistributionLabeled.from_unlabeled(
            self.RiskyDstn,
            name="Distribution of Risky Asset Returns",
            var_names=["risky"],
        )

        ShockDstn = []

        for i in range(len(self.ShockDstn.dstns)):
            ShockDstn.append(
                DiscreteDistributionLabeled.from_unlabeled(
                    self.ShockDstn[i],
                    name="Distribution of Shocks to Income and Risky Asset Returns",
                    var_names=["perm", "tran", "risky"],
                )
            )

        self.ShockDstn = ShockDstn


@dataclass
class ConsRiskyAssetLabeledSolver(ConsIndShockLabeledSolver):
    """
    Solver for an agent that can save in an asset that has a risky return.
    """

    solution_next: ConsumerSolutionLabeled  # solution to next period's problem
    ShockDstn: DiscreteDistributionLabeled  #  distribution of shocks to income and returns
    LivPrb: float  # survival probability
    DiscFac: float  # intertemporal discount factor
    CRRA: float  # coefficient of relative risk aversion
    Rfree: float  # interest factor on assets
    PermGroFac: float  # permanent income growth factor
    BoroCnstArt: float  # artificial borrowing constraint
    aXtraGrid: np.ndarray  # grid of end-of-period assets

    def __post_init__(self):
        """
        Define utility functions.
        """

        self.def_utility_funcs()

    def calculate_borrowing_constraint(self):
        """
        Calculate the borrowing constraint by enforcing a 0.0 artificial borrowing
        constraint and setting the shocks to income to come from the shock distribution.
        """
        self.BoroCnstArt = 0.0
        self.IncShkDstn = self.ShockDstn
        return super().calculate_borrowing_constraint()

    def post_state_transition(self, post_state=None, shocks=None, params=None):
        """
        Post_state to next_state transition with risky asset return.

        Parameters
        ----------
        post_state : dict
            Post-state variables.
        shocks : dict
            Shocks to income and risky asset return.
        params : dict
            Parameters of the model.

        Returns
        -------
        next_state : dict
            Next period's state variables.
        """

        next_state = {}  # pytree
        next_state["mNrm"] = (
            post_state["aNrm"] * shocks["risky"] / (params.PermGroFac * shocks["perm"])
            + shocks["tran"]
        )
        return next_state

    def continuation_transition(
        self, shocks=None, post_state=None, v_next=None, params=None
    ):
        """
        Continuation value function of post_state with risky asset return.

        Parameters
        ----------
        shocks : dict
            Shocks to income and risky asset return.
        post_state : dict
            Post-state variables.
        v_next : function
            Value function of next period.
        params : dict
            Parameters of the model.

        Returns
        -------
        variables : dict
            Variables of the continuation value function.
        """

        variables = {}  # pytree
        next_state = self.post_state_transition(post_state, shocks, params)
        variables.update(next_state)

        variables["psi"] = params.PermGroFac * shocks["perm"]

        variables["v"] = variables["psi"] ** (1 - params.CRRA) * v_next(next_state)

        variables["v_der"] = (
            shocks["risky"]
            * variables["psi"] ** (-params.CRRA)
            * v_next.derivative(next_state)
        )

        # for estimagic purposes

        variables["contributions"] = variables["v"]
        variables["value"] = np.sum(variables["v"])

        return variables

    def create_continuation_function(self):
        """
        Create the continuation value function taking expectation
        over the shock distribution which includes shocks to income and
        the risky asset return.

        Returns
        -------
        wfunc : ValueFuncCRRALabeled
            Continuation value function.
        """
        # unpack next period's solution
        vfunc_next = self.solution_next.value

        v_end = self.ShockDstn.expected(
            func=self.continuation_transition,
            post_state=self.post_state,
            v_next=vfunc_next,
            params=self.params,
        )

        v_end["v_inv"] = self.u.inv(v_end["v"])
        v_end["v_der_inv"] = self.u.derinv(v_end["v_der"])

        borocnst = self.borocnst.drop(["mNrm"]).expand_dims("aNrm")
        if self.nat_boro_cnst:
            v_end = xr.merge([borocnst, v_end])

        v_end = v_end.transpose("aNrm", ...)

        # need to drop m because it's next period's m
        # v_end = xr.Dataset(v_end).drop(["mNrm"])
        wfunc = ValueFuncCRRALabeled(v_end, self.params.CRRA)

        return wfunc


class FixedPortfolioLabeledType(
    RiskyAssetLabeledType, FixedPortfolioShareRiskyAssetConsumerType
):
    """
    A labeled FixedPortfolioShareRiskyAssetConsumerType. This class is a subclass of
    FixedPortfolioShareRiskyAssetConsumerType, and inherits all of its methods and attributes.

    Fixed portfolio share consumers can save on a risk-free and
    risky asset at a fixed proportion.
    """

    def __init__(self, verbose=1, quiet=False, **kwds):
        """
        Initialize a new instance of FixedPortfolioLabeledType.
        """

        params = init_risky_share_fixed.copy()
        params.update(kwds)

        # Initialize a basic AgentType
        FixedPortfolioShareRiskyAssetConsumerType.__init__(
            self, verbose=verbose, quiet=quiet, **params
        )

        self.solve_one_period = make_one_period_oo_solver(
            ConsFixedPortfolioLabeledSolver
        )

        self.update_labeled_type()


@dataclass
class ConsFixedPortfolioLabeledSolver(ConsRiskyAssetLabeledSolver):
    """
    Solver for an agent that can save in a risk-free and risky asset
    at a fixed proportion.
    """

    RiskyShareFixed: float  # share of risky assets in portfolio

    def create_params_namespace(self):
        """
        Create a namespace for parameters.
        """

        self.params = SimpleNamespace(
            Discount=self.DiscFac * self.LivPrb,
            CRRA=self.CRRA,
            Rfree=self.Rfree,
            PermGroFac=self.PermGroFac,
            RiskyShareFixed=self.RiskyShareFixed,
        )

    def post_state_transition(self, post_state=None, shocks=None, params=None):
        """
        Post_state to next_state transition with fixed portfolio share.

        Parameters
        ----------
        post_state : dict
            Post-state variables.
        shocks : dict
            Shocks to income and risky asset return.
        params : dict
            Parameters of the model.

        Returns
        -------
        next_state : dict
            Next period's state variables.
        """

        next_state = {}  # pytree
        next_state["rDiff"] = params.Rfree - shocks["risky"]
        next_state["rPort"] = (
            params.Rfree + next_state["rDiff"] * params.RiskyShareFixed
        )
        next_state["mNrm"] = (
            post_state["aNrm"]
            * next_state["rPort"]
            / (params.PermGroFac * shocks["perm"])
            + shocks["tran"]
        )
        return next_state

    def continuation_transition(
        self, shocks=None, post_state=None, v_next=None, params=None
    ):
        """
        Continuation value function of post_state with fixed portfolio share.

        Parameters
        ----------
        shocks : dict
            Shocks to income and risky asset return.
        post_state : dict
            Post-state variables.
        v_next : ValueFuncCRRALabeled
            Continuation value function.
        params : dict
            Parameters of the model.

        Returns
        -------
        variables : dict
            Variables of the model.
        """

        variables = {}  # pytree
        next_state = self.post_state_transition(post_state, shocks, params)
        variables.update(next_state)

        variables["psi"] = params.PermGroFac * shocks["perm"]

        variables["v"] = variables["psi"] ** (1 - params.CRRA) * v_next(next_state)

        variables["v_der"] = (
            next_state["rPort"]
            * variables["psi"] ** (-params.CRRA)
            * v_next.derivative(next_state)
        )

        # for estimagic purposes

        variables["contributions"] = variables["v"]
        variables["value"] = np.sum(variables["v"])

        return variables


class PortfolioLabeledType(FixedPortfolioLabeledType, PortfolioConsumerType):
    """
    A labeled PortfolioConsumerType. This class is a subclass of
    PortfolioConsumerType, and inherits all of its methods and attributes.

    Portfolio consumers can save on a risk-free and
    risky asset at an optimal proportion.
    """

    def __init__(self, verbose=1, quiet=False, **kwds):
        """
        Initialize a new instance of PortfolioLabeledType.
        """
        params = init_portfolio.copy()
        params.update(kwds)

        self.RiskyShareFixed = [1.0]

        # Initialize a basic AgentType
        PortfolioConsumerType.__init__(self, verbose=verbose, quiet=quiet, **params)

        self.solve_one_period = make_one_period_oo_solver(ConsPortfolioLabeledSolver)

        self.update_labeled_type()


@dataclass
class ConsPortfolioLabeledSolver(ConsFixedPortfolioLabeledSolver):
    """
    Solver for an agent that can save in a risk-free and risky asset
    at an optimal proportion.
    """

    ShareGrid: np.ndarray  # grid of risky shares

    def create_post_state(self):
        """
        Create post-state variables by adding risky share, called
        stigma, to the post-state variables.
        """

        super().create_post_state()

        self.post_state["stigma"] = xr.DataArray(
            self.ShareGrid, dims=["stigma"], attrs={"long_name": "risky share"}
        )

    def post_state_transition(self, post_state=None, shocks=None, params=None):
        """
        Post_state to next_state transition with optimal portfolio share.

        Parameters
        ----------
        post_state : dict
            Post-state variables.
        shocks : dict
            Shocks to income and risky asset return.
        params : dict
            Parameters of the model.

        Returns
        -------
        next_state : dict
            Next period's state variables.
        """

        next_state = {}  # pytree
        next_state["rDiff"] = shocks["risky"] - params.Rfree
        next_state["rPort"] = params.Rfree + next_state["rDiff"] * post_state["stigma"]
        next_state["mNrm"] = (
            post_state["aNrm"]
            * next_state["rPort"]
            / (params.PermGroFac * shocks["perm"])
            + shocks["tran"]
        )
        return next_state

    def continuation_transition(
        self, shocks=None, post_state=None, v_next=None, params=None
    ):
        """
        Continuation value function of post_state with optimal portfolio share.

        Parameters
        ----------
        shocks : dict
            Shocks to income and risky asset return.
        post_state : dict
            Post-state variables.
        v_next : ValueFuncCRRALabeled
            Continuation value function.
        params : dict
            Parameters of the model.

        Returns
        -------
        variables : dict
            Variables of the model.
        """

        variables = {}  # pytree
        next_state = self.post_state_transition(post_state, shocks, params)
        variables.update(next_state)

        variables["psi"] = params.PermGroFac * shocks["perm"]

        variables["v"] = variables["psi"] ** (1 - params.CRRA) * v_next(next_state)

        variables["v_der"] = variables["psi"] ** (-params.CRRA) * v_next.derivative(
            next_state
        )

        variables["dvda"] = next_state["rPort"] * variables["v_der"]
        variables["dvds"] = (
            next_state["rDiff"] * post_state["aNrm"] * variables["v_der"]
        )

        # for estimagic purposes

        variables["contributions"] = variables["v"]
        variables["value"] = np.sum(variables["v"])

        return variables

    def create_continuation_function(self):
        """
        Create continuation function with optimal portfolio share.
        The continuation function is a function of the post-state before
        the growth period, but only a function of assets in the
        allocation period.

        Therefore, the first continuation function is a function of
        assets and stigma. Given this, the agent makes an optimal
        choice of risky share of portfolio, and the second continuation
        function is a function of assets only.

        Returns
        -------
        wfunc : ValueFuncCRRALabeled
            Continuation value function.
        """

        wfunc = super().create_continuation_function()

        dvds = wfunc.dataset["dvds"].values

        # For each value of aNrm, find the value of Share such that FOC-Share == 0.
        crossing = np.logical_and(dvds[:, 1:] <= 0.0, dvds[:, :-1] >= 0.0)
        share_idx = np.argmax(crossing, axis=1)
        a_idx = np.arange(self.post_state["aNrm"].size)
        bot_s = self.ShareGrid[share_idx]
        top_s = self.ShareGrid[share_idx + 1]
        bot_f = dvds[a_idx, share_idx]
        top_f = dvds[a_idx, share_idx + 1]
        alpha = 1.0 - top_f / (top_f - bot_f)
        opt_share = (1.0 - alpha) * bot_s + alpha * top_s

        # If agent wants to put more than 100% into risky asset, he is constrained
        # For values of aNrm at which the agent wants to put
        # more than 100% into risky asset, constrain them
        opt_share[dvds[:, -1] > 0.0] = 1.0
        # Likewise if he wants to put less than 0% into risky asset
        opt_share[dvds[:, 0] < 0.0] = 0.0

        if not self.nat_boro_cnst:
            # aNrm=0, so there's no way to "optimize" the portfolio
            opt_share[0] = 1.0

        opt_share = xr.DataArray(
            opt_share,
            coords={"aNrm": self.post_state["aNrm"].values},
            dims=["aNrm"],
            attrs={"long_name": "optimal risky share"},
        )

        v_end = wfunc.evaluate({"aNrm": self.post_state["aNrm"], "stigma": opt_share})

        v_end = v_end.reset_coords(names="stigma")

        wfunc = ValueFuncCRRALabeled(v_end, self.params.CRRA)

        self.post_state = self.post_state.drop("stigma")

        return wfunc
