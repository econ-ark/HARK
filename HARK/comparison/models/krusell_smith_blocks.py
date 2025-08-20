"""
Block model definitions for Krusell-Smith (1998) model.

This module defines the Krusell-Smith model using HARK's block modeling framework,
allowing for modular representation and solution of the heterogeneous agent
macroeconomic model with aggregate shocks.
"""

from HARK.distributions import Bernoulli, MarkovProcess, MeanOneLogNormal
from HARK.model import Control, Aggregate, DBlock, RBlock
import numpy as np

# Default calibration for Krusell-Smith model
ks_calibration = {
    # Individual preferences
    "DiscFac": 0.99,  # Discount factor
    "CRRA": 1.0,  # Risk aversion
    "LbrInd": 1.0,  # Labor supply (normalized)
    # Production function
    "CapShare": 0.36,  # Capital share (α)
    "DeprFac": 0.025,  # Depreciation rate (δ)
    # Aggregate productivity states
    "ProdB": 0.99,  # Productivity in bad state
    "ProdG": 1.01,  # Productivity in good state
    # Employment states by aggregate state
    "UrateB": 0.10,  # Unemployment rate in bad aggregate state
    "UrateG": 0.04,  # Unemployment rate in good aggregate state
    # Transition dynamics
    "DurMeanB": 8.0,  # Mean duration of bad aggregate state (quarters)
    "DurMeanG": 8.0,  # Mean duration of good aggregate state
    "SpellMeanB": 2.5,  # Mean unemployment spell in bad state
    "SpellMeanG": 1.5,  # Mean unemployment spell in good state
    # Additional calibration
    "BoroCnstArt": 0.0,  # Borrowing constraint
    "LivPrb": 1.0,  # Survival probability (infinite horizon)
}


def create_aggregate_transition_matrix(calib):
    """Create aggregate state transition matrix from duration parameters."""
    # Probability of staying in bad state
    prob_stay_B = 1 - 1 / calib["DurMeanB"]
    # Probability of staying in good state
    prob_stay_G = 1 - 1 / calib["DurMeanG"]

    return np.array(
        [
            [prob_stay_B, 1 - prob_stay_B],  # From bad state
            [1 - prob_stay_G, prob_stay_G],  # From good state
        ]
    )


def create_individual_transition_matrix(calib, agg_state):
    """Create individual employment transition matrix conditional on aggregate state."""
    if agg_state == 0:  # Bad aggregate state
        urate = calib["UrateB"]
        spell_mean = calib["SpellMeanB"]
    else:  # Good aggregate state
        urate = calib["UrateG"]
        spell_mean = calib["SpellMeanG"]

    # Probability of staying unemployed
    prob_stay_unemployed = 1 - 1 / spell_mean
    # Probability of becoming unemployed (from steady state)
    prob_become_unemployed = urate * (1 - prob_stay_unemployed) / (1 - urate)

    return np.array(
        [
            [prob_stay_unemployed, 1 - prob_stay_unemployed],  # From unemployed
            [prob_become_unemployed, 1 - prob_become_unemployed],  # From employed
        ]
    )


# Individual consumption-saving block for Krusell-Smith
individual_ks_block = DBlock(
    name="individual_ks",
    shocks={
        # Aggregate productivity shock (Markov process)
        "agg_state": (
            MarkovProcess,
            {
                "transition_matrix": lambda: create_aggregate_transition_matrix(
                    ks_calibration
                ),
                "seed": 0,
            },
        ),
        # Individual employment shock (conditional on aggregate state)
        "emp_state": (
            MarkovProcess,
            {
                "transition_matrix": lambda agg_state: create_individual_transition_matrix(
                    ks_calibration, agg_state
                ),
                "seed": 1,
            },
        ),
    },
    dynamics={
        # Labor income
        "labor_income": lambda emp_state, agg_state, ProdB, ProdG, LbrInd: (
            LbrInd * (ProdB if agg_state == 0 else ProdG) * emp_state
        ),
        # Interest rate (determined in equilibrium)
        "R": lambda K_agg, L_agg, agg_state, CapShare, DeprFac, ProdB, ProdG: (
            CapShare
            * (ProdB if agg_state == 0 else ProdG)
            * (K_agg / L_agg) ** (CapShare - 1)
            + 1
            - DeprFac
        ),
        # Wage rate
        "w": lambda K_agg, L_agg, agg_state, CapShare, ProdB, ProdG: (
            (1 - CapShare)
            * (ProdB if agg_state == 0 else ProdG)
            * (K_agg / L_agg) ** CapShare
        ),
        # Cash on hand
        "m": lambda a_lag, R, labor_income: a_lag * R + labor_income,
        # Consumption choice
        "c": Control(["m", "K_agg", "agg_state"]),
        # Asset choice
        "a": "m - c",
    },
    reward={
        "u": lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA) if CRRA != 1 else np.log(c)
    },
)

# Aggregate variables block (simplified for now)
aggregate_ks_block = DBlock(
    name="aggregate_ks",
    shocks={
        # Placeholder aggregate shocks - these would be computed from individual variables
        "K_agg": Aggregate(
            MeanOneLogNormal(1)
        ),  # Will be computed from individual assets
        "L_agg": Aggregate(MeanOneLogNormal(1)),  # Will be computed from employment
    },
    dynamics={
        # Capital-labor ratio
        "KL_ratio": lambda K_agg, L_agg: K_agg / L_agg,
    },
)

# Equilibrium/forecasting block (bounded rationality)
equilibrium_ks_block = DBlock(
    name="equilibrium_ks",
    dynamics={
        # Simple linear forecasting rule: log(K') = a0 + a1*log(K) + a2*agg_state
        # Agents use this rule to forecast next period's aggregate capital
        "K_forecast": lambda K_agg, agg_state, forecast_coeffs: np.exp(
            forecast_coeffs[0]
            + forecast_coeffs[1] * np.log(K_agg)
            + forecast_coeffs[2] * agg_state
        ),
        # Forecast accuracy (for updating coefficients)
        "forecast_error": lambda K_agg, K_forecast_lag: np.log(K_agg)
        - np.log(K_forecast_lag)
        if K_forecast_lag is not None
        else 0.0,
    },
)

# Combined Krusell-Smith model
krusell_smith_model = RBlock(
    name="krusell_smith_model",
    blocks=[individual_ks_block, aggregate_ks_block, equilibrium_ks_block],
)

# Alternative simplified version for testing
simple_ks_individual = DBlock(
    name="simple_ks_individual",
    shocks={
        "agg_state": (Bernoulli, {"p": 0.5}),  # Simple 2-state aggregate shock
        "emp_state": (Bernoulli, {"p": 0.95}),  # Simple employment shock
    },
    dynamics={
        "income": lambda emp_state, agg_state: emp_state
        * (0.99 if agg_state == 0 else 1.01),
        "m": lambda a_lag, income, R: a_lag * R + income,
        "c": Control(["m"]),
        "a": "m - c",
    },
    reward={"u": lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA)},
)

simple_ks_aggregate = DBlock(
    name="simple_ks_aggregate",
    shocks={
        "K_agg": Aggregate(MeanOneLogNormal(1)),  # Simplified aggregate capital
    },
    dynamics={
        "R": lambda K_agg, CapShare, DeprFac: CapShare * K_agg ** (CapShare - 1)
        + 1
        - DeprFac,
    },
)

simple_krusell_smith_model = RBlock(
    name="simple_krusell_smith_model",
    blocks=[simple_ks_individual, simple_ks_aggregate],
)
