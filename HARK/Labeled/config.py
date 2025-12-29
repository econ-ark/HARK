"""
Configuration management for labeled consumption-saving models.

This module provides immutable configuration objects for model parameters,
replacing the module-level dict mutation pattern with a more robust approach.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType_aXtraGrid_default,
    init_idiosyncratic_shocks,
    init_perfect_foresight,
)
from HARK.ConsumptionSaving.ConsPortfolioModel import init_portfolio
from HARK.ConsumptionSaving.ConsRiskyAssetModel import (
    IndShockRiskyAssetConsumerType_constructor_default,
    init_risky_asset,
)
from HARK.utilities import make_assets_grid

from .factories import (
    make_labeled_inc_shk_dstn,
    make_labeled_risky_dstn,
    make_labeled_shock_dstn,
    make_solution_terminal_labeled,
)

__all__ = [
    "ModelConfig",
    "PERF_FORESIGHT_CONFIG",
    "IND_SHOCK_CONFIG",
    "RISKY_ASSET_CONFIG",
    "PORTFOLIO_CONFIG",
    "get_config",
]


@dataclass(frozen=True)
class ModelConfig:
    """
    Immutable configuration for a labeled consumption model.

    This class represents a complete configuration including base parameters
    and constructor functions. Configurations can inherit from parent configs
    through the parent field.

    Parameters
    ----------
    base_params : dict
        Model-specific parameter overrides.
    constructors : dict
        Constructor functions for computed parameters.
    parent : ModelConfig, optional
        Parent configuration to inherit from.
    """

    base_params: dict[str, Any] = field(default_factory=dict)
    constructors: dict[str, Callable] = field(default_factory=dict)
    parent: ModelConfig | None = None

    def build_params(self) -> dict[str, Any]:
        """
        Build complete parameter dictionary by merging with parent chain.

        Returns
        -------
        dict[str, Any]
            Complete parameter dictionary with constructors.
        """
        if self.parent is not None:
            params = self.parent.build_params()
        else:
            params = {}

        params.update(deepcopy(self.base_params))

        # Merge constructors
        if "constructors" not in params:
            params["constructors"] = {}

        params["constructors"] = {**params.get("constructors", {}), **self.constructors}

        return params


# =============================================================================
# Default Configurations
# =============================================================================

# Perfect Foresight Labeled
_pf_base_params = deepcopy(init_idiosyncratic_shocks)
_pf_base_params.update(init_perfect_foresight)
_pf_base_params.update(IndShockConsumerType_aXtraGrid_default)

_pf_constructors = deepcopy(init_idiosyncratic_shocks.get("constructors", {}))
_pf_constructors["solution_terminal"] = make_solution_terminal_labeled
_pf_constructors["aXtraGrid"] = make_assets_grid

PERF_FORESIGHT_CONFIG = ModelConfig(
    base_params=_pf_base_params,
    constructors=_pf_constructors,
)

# IndShock Labeled
_ind_shock_constructors = {
    "IncShkDstn": make_labeled_inc_shk_dstn,
}

IND_SHOCK_CONFIG = ModelConfig(
    base_params={},
    constructors=_ind_shock_constructors,
    parent=PERF_FORESIGHT_CONFIG,
)

# Risky Asset Labeled
_risky_constructors = deepcopy(IndShockRiskyAssetConsumerType_constructor_default)
_risky_constructors["IncShkDstn"] = make_labeled_inc_shk_dstn
_risky_constructors["RiskyDstn"] = make_labeled_risky_dstn
_risky_constructors["ShockDstn"] = make_labeled_shock_dstn
_risky_constructors["solution_terminal"] = make_solution_terminal_labeled
if "solve_one_period" in _risky_constructors:
    del _risky_constructors["solve_one_period"]

_risky_base = deepcopy(init_risky_asset)
# Remove solve_one_period from base_params constructors to avoid conflict with our solver
if "constructors" in _risky_base and "solve_one_period" in _risky_base["constructors"]:
    del _risky_base["constructors"]["solve_one_period"]

RISKY_ASSET_CONFIG = ModelConfig(
    base_params=_risky_base,
    constructors=_risky_constructors,
)

# Portfolio Labeled
_portfolio_constructors = {
    "IncShkDstn": make_labeled_inc_shk_dstn,
    "RiskyDstn": make_labeled_risky_dstn,
    "ShockDstn": make_labeled_shock_dstn,
    "solution_terminal": make_solution_terminal_labeled,
}

_portfolio_base = deepcopy(init_portfolio)
_portfolio_base["RiskyShareFixed"] = [0.0]

PORTFOLIO_CONFIG = ModelConfig(
    base_params=_portfolio_base,
    constructors=_portfolio_constructors,
)


def get_config(model_type: str) -> ModelConfig:
    """
    Get configuration for a named model type.

    Parameters
    ----------
    model_type : str
        One of 'perfect_foresight', 'ind_shock', 'risky_asset', 'portfolio'.

    Returns
    -------
    ModelConfig
        Configuration for the specified model type.

    Raises
    ------
    ValueError
        If model_type is not recognized.
    """
    configs = {
        "perfect_foresight": PERF_FORESIGHT_CONFIG,
        "ind_shock": IND_SHOCK_CONFIG,
        "risky_asset": RISKY_ASSET_CONFIG,
        "portfolio": PORTFOLIO_CONFIG,
    }

    if model_type not in configs:
        raise ValueError(
            f"Unknown model type '{model_type}'. Available: {list(configs.keys())}"
        )

    return configs[model_type]
