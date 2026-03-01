"""
Block model definitions for Aiyagari (1994) model.

This module defines the Aiyagari model using HARK's block modeling framework,
allowing for modular representation and solution of the heterogeneous agent
macroeconomic model with idiosyncratic income risk.
"""

from HARK.distributions import Bernoulli, MarkovProcess, MeanOneLogNormal, Lognormal
from HARK.model import Control, Aggregate, DBlock, RBlock
import numpy as np

# Default calibration for Aiyagari model
aiyagari_calibration = {
    # Individual preferences
    "DiscFac": 0.96,  # Discount factor (β)
    "CRRA": 1.0,  # Risk aversion (σ)
    # Production function
    "CapShare": 0.36,  # Capital share (α)
    "DeprFac": 0.08,  # Depreciation rate (δ)
    # Income process (Tauchen discretization)
    "IncUnemp": 0.0,  # Income when unemployed
    "IncEmp": 1.0,  # Income when employed
    "UnempPrb": 0.07,  # Probability of unemployment
    "DurMeanUnemp": 2.5,  # Mean duration of unemployment
    # Log income process parameters
    "PermShkStd": 0.1,  # Standard deviation of permanent income shocks
    "TranShkStd": 0.1,  # Standard deviation of transitory income shocks
    "TranShkCount": 7,  # Number of transitory shock points
    "PermShkCount": 7,  # Number of permanent shock points
    # Asset grid and borrowing constraint
    "aXtraMin": 0.001,  # Minimum end-of-period assets
    "aXtraMax": 20,  # Maximum end-of-period assets
    "aXtraCount": 48,  # Number of asset grid points
    "BoroCnstArt": 0.0,  # Borrowing constraint
    # Additional parameters
    "LivPrb": 1.0,  # Survival probability (infinite horizon)
    "PermGroFac": 1.0,  # Permanent income growth factor
}


def create_employment_transition_matrix(calib):
    """Create employment transition matrix from unemployment parameters."""
    unemp_prob = calib["UnempPrb"]
    dur_mean = calib["DurMeanUnemp"]

    # Probability of staying unemployed
    prob_stay_unemployed = 1 - 1 / dur_mean
    # Probability of becoming unemployed (from steady state)
    prob_become_unemployed = unemp_prob * (1 - prob_stay_unemployed) / (1 - unemp_prob)

    return np.array(
        [
            [prob_stay_unemployed, 1 - prob_stay_unemployed],  # From unemployed
            [prob_become_unemployed, 1 - prob_become_unemployed],  # From employed
        ]
    )


# Individual consumption-saving block for Aiyagari
individual_aiyagari_block = DBlock(
    name="individual_aiyagari",
    shocks={
        # Employment shock (Markov process)
        "emp_state": (
            MarkovProcess,
            {
                "transition_matrix": lambda: create_employment_transition_matrix(
                    aiyagari_calibration
                ),
                "seed": 0,
            },
        ),
        # Permanent income shock
        "perm_shk": (
            Lognormal,
            {
                "mu": -0.5 * aiyagari_calibration["PermShkStd"] ** 2,  # Mean-preserving
                "sigma": aiyagari_calibration["PermShkStd"],
            },
        ),
        # Transitory income shock
        "tran_shk": (
            Lognormal,
            {
                "mu": -0.5 * aiyagari_calibration["TranShkStd"] ** 2,  # Mean-preserving
                "sigma": aiyagari_calibration["TranShkStd"],
            },
        ),
    },
    dynamics={
        # Labor income (employment × shocks)
        "labor_income": lambda emp_state, perm_shk, tran_shk, IncUnemp, IncEmp: (
            (IncEmp if emp_state == 1 else IncUnemp) * perm_shk * tran_shk
        ),
        # Interest rate (determined in equilibrium)
        "R": lambda r: 1 + r,
        # Cash on hand
        "m": lambda a_lag, R, labor_income: a_lag * R + labor_income,
        # Consumption choice
        "c": Control(
            ["m", "r"]
        ),  # Consumption depends on cash-on-hand and interest rate
        # Asset choice
        "a": lambda m, c: m - c,
    },
    reward={
        "u": lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA) if CRRA != 1 else np.log(c)
    },
)

# Aggregate/equilibrium block for Aiyagari
aggregate_aiyagari_block = DBlock(
    name="aggregate_aiyagari",
    shocks={
        # These would be computed from individual variables in full implementation
        "K_agg": Aggregate(MeanOneLogNormal(1)),  # Aggregate capital
        "L_agg": Aggregate(MeanOneLogNormal(1)),  # Aggregate labor
    },
    dynamics={
        # Interest rate from firm's FOC
        "r": lambda K_agg, L_agg, CapShare, DeprFac: (
            CapShare * (K_agg / L_agg) ** (CapShare - 1) - DeprFac
        ),
        # Wage rate from firm's FOC
        "w": lambda K_agg, L_agg, CapShare: (
            (1 - CapShare) * (K_agg / L_agg) ** CapShare
        ),
        # Capital-labor ratio
        "KL_ratio": lambda K_agg, L_agg: K_agg / L_agg,
        # Market clearing condition (should be zero in equilibrium)
        "excess_demand": lambda K_agg, a_demand: a_demand - K_agg,
    },
)

# Combined Aiyagari model
aiyagari_model = RBlock(
    name="aiyagari_model", blocks=[individual_aiyagari_block, aggregate_aiyagari_block]
)

# Simplified version for testing
simple_aiyagari_individual = DBlock(
    name="simple_aiyagari_individual",
    shocks={
        "emp_state": (Bernoulli, {"p": 0.93}),  # Simple employment shock
        "income_shk": (MeanOneLogNormal, {"sigma": 0.2}),  # Simple income shock
    },
    dynamics={
        "income": lambda emp_state, income_shk: emp_state * income_shk,
        "m": lambda a_lag, income, R: a_lag * R + income,
        "c": Control(["m"]),
        "a": lambda m, c: m - c,
    },
    reward={"u": lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA)},
)

simple_aiyagari_aggregate = DBlock(
    name="simple_aiyagari_aggregate",
    shocks={
        "K_agg": Aggregate(MeanOneLogNormal(1)),  # Simplified aggregate capital
    },
    dynamics={
        "r": lambda K_agg, CapShare, DeprFac: CapShare * K_agg ** (CapShare - 1)
        - DeprFac,
        "R": lambda r: 1 + r,
    },
)

simple_aiyagari_model = RBlock(
    name="simple_aiyagari_model",
    blocks=[simple_aiyagari_individual, simple_aiyagari_aggregate],
)
