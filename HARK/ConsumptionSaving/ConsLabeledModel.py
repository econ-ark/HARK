from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import xarray as xr
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsIndShockSetup,
    IndShockConsumerType,
    init_perfect_foresight,
)
from HARK.ConsumptionSaving.ConsRiskyAssetModel import (
    RiskyAssetConsumerType,
    init_risky_asset,
)
from HARK.core import MetricObject, make_one_period_oo_solver
from HARK.distribution import DiscreteDistributionLabeled
from HARK.utilities import CRRAutility, CRRAutility_inv, CRRAutilityP, CRRAutilityP_inv


class ValueFuncCRRALabeled(MetricObject):
    def __init__(self, dataset: xr.Dataset, CRRA: float):

        self.dataset = dataset
        self.CRRA = CRRA

    def __call__(self, state):
        """
        Interpolate inverse falue function then invert to get value function at given state.
        """

        if isinstance(state, xr.Dataset):
            state_dict = state.coords
        else:
            state_dict = {}
            for coords in self.dataset.coords.keys():
                state_dict[coords] = state[coords]

        result = CRRAutility(
            self.dataset["v_inv"].interp(
                state_dict,
                assume_sorted=True,
                kwargs={"fill_value": "extrapolate"},
            ),
            self.CRRA,
        )

        result.name = "v"
        result.attrs = self.dataset["v"].attrs

        return result

    def derivative(self, state):
        """
        Interpolate inverse marginal value function then invert to get marginal value function at given state.
        """

        if isinstance(state, xr.Dataset):
            state_dict = state.coords
        else:
            state_dict = {}
            for coords in self.dataset.coords.keys():
                state_dict[coords] = state[coords]

        result = CRRAutilityP(
            self.dataset["v_der_inv"].interp(
                state_dict,
                assume_sorted=True,
                kwargs={"fill_value": "extrapolate"},
            ),
            self.CRRA,
        )

        result.name = "v_der"
        result.attrs = self.dataset["v"].attrs

        return result

    def evaluate(self, state):
        """
        Interpolate all data variables in the dataset.
        """

        if isinstance(state, xr.Dataset):
            state_dict = state.coords
        else:
            state_dict = {}
            for coords in self.dataset.coords.keys():
                state_dict[coords] = state[coords]

        result = self.dataset.interp(
            state_dict,
            kwargs={"fill_value": "extrapolate"},
        )
        result.attrs = self.dataset["v"].attrs

        return result


class ConsumerSolutionLabeled(MetricObject):
    def __init__(
        self,
        value: ValueFuncCRRALabeled,
        policy: xr.Dataset,
        continuation: ValueFuncCRRALabeled,
        attrs=None,
    ):

        if attrs is None:
            attrs = dict()

        self.value = value  # value function
        self.policy = policy  # policy function
        self.continuation = continuation  # continuation function

        self.attrs = attrs

    def distance(self, other):

        value = self.value.dataset
        other_value = other.value.dataset.interp_like(value)

        return np.max(np.abs(value - other_value).to_array())


class PerfForesightLabeledType(IndShockConsumerType):
    def __init__(self, verbose=1, quiet=False, **kwds):
        params = init_perfect_foresight.copy()
        params.update(kwds)

        # Initialize a basic AgentType
        IndShockConsumerType.__init__(self, verbose=verbose, quiet=quiet, **params)

        self.solve_one_period = make_one_period_oo_solver(
            ConsPerfForesightLabeledSolver
        )

    def update_solution_terminal(self):

        u = lambda c: CRRAutility(c, self.CRRA)  # utility
        uP = lambda c: CRRAutilityP(c, self.CRRA)  # marginal utility

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

        v_der = uP(cNrm)
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
            attrs={"m_nrm_min": 0.0},
        )


class ConsPerfForesightLabeledSolver(ConsIndShockSetup):
    def def_utility_funcs(self):

        self.u = lambda c: CRRAutility(c, self.CRRA)
        self.uP = lambda c: CRRAutilityP(c, self.CRRA)
        self.u_inv = lambda u: CRRAutility_inv(u, self.CRRA)
        self.uP_inv = lambda uP: CRRAutilityP_inv(uP, self.CRRA)

    def create_params_namespace(self):

        self.params = SimpleNamespace(
            Discount=self.DiscFac * self.LivPrb,
            CRRA=self.CRRA,
            Rfree=self.Rfree,
            PermGroFac=self.PermGroFac,
        )

    def define_boundary_constraint(self):

        # Calculate the minimum allowable value of money resources in this period
        self.BoroCnstNat = (
            self.solution_next.attrs["m_nrm_min"] - 1
        ) / self.params.Rfree

        # if natural borrowing constraint is binding constraint, then we can not
        # evaluate the value function at that point, so we must fill out the data
        if self.BoroCnstArt is None or self.BoroCnstArt <= self.BoroCnstNat:
            self.m_nrm_min = self.BoroCnstNat
            self.nat_boro_cnst = True

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
            self.nat_boro_cnst = False

            self.borocnst = xr.Dataset(
                coords={"mNrm": self.m_nrm_min, "aNrm": self.m_nrm_min},
                data_vars={"cNrm": 0.0},
            )

    def create_post_state(self):

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

    def state_transition(self, s=None, a=None, params=None):
        """
        state to post_state transition
        """
        ps = {}  # pytree
        ps["aNrm"] = s["mNrm"] - a["cNrm"]
        return ps

    def post_state_transition(self, ps=None, params=None):
        """
        post_state to next_state transition
        """
        ns = {}  # pytree
        ns["mNrm"] = ps["aNrm"] * params.Rfree / params.PermGroFac + 1
        return ns

    def reverse_transition(self, ps=None, a=None, params=None):
        """state from post_state and actions"""
        s = {}  # pytree
        s["mNrm"] = ps["aNrm"] + a["cNrm"]

        return s

    def egm_transition(self, ps=None, continuation=None, params=None):
        """actions from post_state"""

        a = {}  # pytree
        a["cNrm"] = self.uP_inv(params.Discount * continuation.derivative(ps))

        return a

    def value_transition(self, action=None, state=None, continuation=None, params=None):
        """
        value of action given state and continuation
        """
        variables = {}  # pytree
        ps = self.state_transition(state, action, params)
        variables.update(ps)

        variables["reward"] = self.u(action["cNrm"])
        variables["v"] = variables["reward"] + params.Discount * continuation(ps)
        variables["v_inv"] = self.u_inv(variables["v"])

        variables["marginal_reward"] = self.uP(action["cNrm"])
        variables["v_der"] = variables["marginal_reward"]
        variables["v_der_inv"] = action["cNrm"]

        # for estimagic purposes
        variables["contributions"] = variables["v"]
        variables["value"] = np.sum(variables["v"])

        return variables

    def continuation_transition(self, post_state=None, value_next=None, params=None):
        """
        continuation value function of post_state
        """
        variables = {}  # pytree
        ns = self.post_state_transition(post_state, params)
        variables.update(ns)
        variables["v"] = params.PermGroFac ** (1 - params.CRRA) * value_next(ns)
        variables["v_der"] = (
            params.Rfree
            * params.PermGroFac ** (-params.CRRA)
            * value_next.derivative(ns)
        )

        variables["v_inv"] = self.u_inv(variables["v"])
        variables["v_der_inv"] = self.uP_inv(variables["v_der"])

        # for estimagic purposes
        variables["contributions"] = variables["v"]
        variables["value"] = np.sum(variables["v"])

        return variables

    def prepare_to_solve(self):
        self.create_params_namespace()
        self.define_boundary_constraint()
        self.create_post_state()

    def create_continuation_function(self):

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
            attrs={"m_nrm_min": self.m_nrm_min},
        )

    def solve(self):

        self.endogenous_grid_method()

        return self.solution


class IndShockLabeledType(PerfForesightLabeledType):
    def __init__(self, verbose=1, quiet=False, **kwds):
        params = init_perfect_foresight.copy()
        params.update(kwds)

        # Initialize a basic AgentType
        PerfForesightLabeledType.__init__(self, verbose=verbose, quiet=quiet, **params)

        self.solve_one_period = make_one_period_oo_solver(ConsIndShockLabeledSolver)

        self.update_labeled_type()

    def update_labeled_type(self):

        self.update_distributions()

    def update_distributions(self):

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
    def define_boundary_constraint(self):

        PermShkMinNext = np.min(self.IncShkDstn.atoms[0])
        TranShkMinNext = np.min(self.IncShkDstn.atoms[1])

        # Calculate the minimum allowable value of money resources in this period
        self.BoroCnstNat = (
            (self.solution_next.attrs["m_nrm_min"] - TranShkMinNext)
            * (self.params.PermGroFac * PermShkMinNext)
            / self.params.Rfree
        )

        if self.BoroCnstArt is None or self.BoroCnstArt <= self.BoroCnstNat:
            self.m_nrm_min = self.BoroCnstNat
            self.nat_boro_cnst = True

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
            self.nat_boro_cnst = False

            self.borocnst = xr.Dataset(
                coords={"mNrm": self.m_nrm_min, "aNrm": self.m_nrm_min},
                data_vars={"cNrm": 0.0},
            )

    def post_state_transition(self, ps=None, shocks=None, params=None):
        """
        post_state to next_state transition
        """
        ns = {}  # pytree
        ns["mNrm"] = (
            ps["aNrm"] * params.Rfree / (params.PermGroFac * shocks["perm"])
            + shocks["tran"]
        )
        return ns

    def continuation_transition(self, shocks=None, ps=None, v_next=None, params=None):
        """
        continuation value function of post_state
        """
        variables = {}  # pytree
        ns = self.post_state_transition(ps, shocks, params)
        variables.update(ns)

        variables["psi"] = params.PermGroFac * shocks["perm"]

        variables["v"] = variables["psi"] ** (1 - params.CRRA) * v_next(ns)

        variables["v_der"] = (
            params.Rfree * variables["psi"] ** (-params.CRRA) * v_next.derivative(ns)
        )

        # for estimagic purposes

        variables["contributions"] = variables["v"]
        variables["value"] = np.sum(variables["v"])

        return variables

    def create_continuation_function(self):

        # unpack next period's solution
        vfunc_next = self.solution_next.value

        v_end = self.IncShkDstn.expected(
            func=self.continuation_transition,
            ps=self.post_state,
            v_next=vfunc_next,
            params=self.params,
        )

        v_end["v_inv"] = self.u_inv(v_end["v"])
        v_end["v_der_inv"] = self.uP_inv(v_end["v_der"])

        borocnst = self.borocnst.drop(["mNrm"]).expand_dims("aNrm")
        if self.nat_boro_cnst:
            v_end = xr.merge([borocnst, v_end])

        # need to drop m because it's next period's m
        # v_end = xr.Dataset(v_end).drop(["mNrm"])
        wfunc = ValueFuncCRRALabeled(v_end, self.params.CRRA)

        return wfunc


class RiskyAssetLabeledType(IndShockLabeledType, RiskyAssetConsumerType):
    def __init__(self, verbose=1, quiet=False, **kwds):
        params = init_risky_asset.copy()
        params.update(kwds)

        # Initialize a basic AgentType
        RiskyAssetConsumerType.__init__(self, verbose=verbose, quiet=quiet, **params)

        self.solve_one_period = make_one_period_oo_solver(ConsRiskyAssetLabeledSolver)

        self.update_labeled_type()

    def update_distributions(self):

        super().update_distributions()

        self.RiskyDstn = DiscreteDistributionLabeled.from_unlabeled(
            self.RiskyDstn,
            name="Distribution of Risky Asset Returns",
            var_names=["risky"],
        )

        ShockDstn = []

        for i in range(len(self.ShockDstn)):

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

    solution_next: ConsumerSolutionLabeled
    ShockDstn: DiscreteDistributionLabeled
    LivPrb: float
    DiscFac: float
    CRRA: float
    Rfree: float
    PermGroFac: float
    BoroCnstArt: float
    aXtraGrid: np.array

    def __post_init__(self):
        self.def_utility_funcs()

    def define_boundary_constraint(self):
        self.BoroCnstArt = 0.0
        self.IncShkDstn = self.ShockDstn
        return super().define_boundary_constraint()

    def post_state_transition(self, ps=None, shocks=None, params=None):
        """
        post_state to next_state transition
        """
        ns = {}  # pytree
        ns["mNrm"] = (
            ps["aNrm"] * shocks["risky"] / (params.PermGroFac * shocks["perm"])
            + shocks["tran"]
        )
        return ns

    def continuation_transition(self, shocks=None, ps=None, v_next=None, params=None):
        """
        continuation value function of post_state
        """
        variables = {}  # pytree
        ns = self.post_state_transition(ps, shocks, params)
        variables.update(ns)

        variables["psi"] = params.PermGroFac * shocks["perm"]

        variables["v"] = variables["psi"] ** (1 - params.CRRA) * v_next(ns)

        variables["v_der"] = (
            shocks["risky"] * variables["psi"] ** (-params.CRRA) * v_next.derivative(ns)
        )

        # for estimagic purposes

        variables["contributions"] = variables["v"]
        variables["value"] = np.sum(variables["v"])

        return variables

    def create_continuation_function(self):

        # unpack next period's solution
        vfunc_next = self.solution_next.value

        v_end = self.ShockDstn.expected(
            func=self.continuation_transition,
            ps=self.post_state,
            v_next=vfunc_next,
            params=self.params,
        )

        v_end["v_inv"] = self.u_inv(v_end["v"])
        v_end["v_der_inv"] = self.uP_inv(v_end["v_der"])

        borocnst = self.borocnst.drop(["mNrm"]).expand_dims("aNrm")
        if self.nat_boro_cnst:
            v_end = xr.merge([borocnst, v_end])

        # need to drop m because it's next period's m
        # v_end = xr.Dataset(v_end).drop(["mNrm"])
        wfunc = ValueFuncCRRALabeled(v_end, self.params.CRRA)

        return wfunc
