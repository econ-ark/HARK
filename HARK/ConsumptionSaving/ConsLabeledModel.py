"""
ConsLabeledModel - Labeled consumption-saving models using xarray.

This module provides consumption-saving models that use xarray's labeled
data structures for clear variable naming and easier manipulation of results.

The implementation has been refactored into the labeled/ subpackage.
This file provides backwards-compatible imports.

Classes
-------
ValueFuncCRRALabeled
    Value function interpolation using xarray.
ConsumerSolutionLabeled
    Solution container for labeled models.
ConsPerfForesightLabeledSolver
    Solver for perfect foresight model.
ConsIndShockLabeledSolver
    Solver for model with income shocks.
ConsRiskyAssetLabeledSolver
    Solver for model with risky asset.
ConsFixedPortfolioLabeledSolver
    Solver for model with fixed portfolio allocation.
ConsPortfolioLabeledSolver
    Solver for model with optimal portfolio choice.
PerfForesightLabeledType
    Agent type for perfect foresight.
IndShockLabeledType
    Agent type with income shocks.
RiskyAssetLabeledType
    Agent type with risky asset.
PortfolioLabeledType
    Agent type with portfolio choice.

Functions
---------
make_solution_terminal_labeled
    Create terminal period solution.
make_labeled_inc_shk_dstn
    Create labeled income shock distribution.
make_labeled_risky_dstn
    Create labeled risky return distribution.
make_labeled_shock_dstn
    Create labeled joint shock distribution.
"""

# Re-export everything from the labeled subpackage
from HARK.Labeled import (
    # Solution classes
    ConsumerSolutionLabeled,
    ValueFuncCRRALabeled,
    # Solvers
    ConsFixedPortfolioLabeledSolver,
    ConsIndShockLabeledSolver,
    ConsPerfForesightLabeledSolver,
    ConsPortfolioLabeledSolver,
    ConsRiskyAssetLabeledSolver,
    # Agent types
    IndShockLabeledType,
    PerfForesightLabeledType,
    PortfolioLabeledType,
    RiskyAssetLabeledType,
    # Factories
    make_labeled_inc_shk_dstn,
    make_labeled_risky_dstn,
    make_labeled_shock_dstn,
    make_solution_terminal_labeled,
    # Config objects
    IND_SHOCK_CONFIG,
    PERF_FORESIGHT_CONFIG,
    PORTFOLIO_CONFIG,
    RISKY_ASSET_CONFIG,
)

# Backwards compatibility: Build config dicts from new config objects
init_perf_foresight_labeled = PERF_FORESIGHT_CONFIG.build_params()
init_ind_shock_labeled = IND_SHOCK_CONFIG.build_params()
init_risky_asset_labeled = RISKY_ASSET_CONFIG.build_params()
init_portfolio_labeled = PORTFOLIO_CONFIG.build_params()

# Also export the constructor dicts for backwards compatibility
PF_labeled_constructor_dict = init_perf_foresight_labeled.get("constructors", {})
ind_shock_labeled_constructor_dict = init_ind_shock_labeled.get("constructors", {})
risky_asset_labeled_constructor_dict = init_risky_asset_labeled.get("constructors", {})
init_portfolio_labeled_constructors = init_portfolio_labeled.get("constructors", {})

__all__ = [
    # Solution classes
    "ValueFuncCRRALabeled",
    "ConsumerSolutionLabeled",
    # Solvers
    "ConsPerfForesightLabeledSolver",
    "ConsIndShockLabeledSolver",
    "ConsRiskyAssetLabeledSolver",
    "ConsFixedPortfolioLabeledSolver",
    "ConsPortfolioLabeledSolver",
    # Agent types
    "PerfForesightLabeledType",
    "IndShockLabeledType",
    "RiskyAssetLabeledType",
    "PortfolioLabeledType",
    # Factories
    "make_solution_terminal_labeled",
    "make_labeled_inc_shk_dstn",
    "make_labeled_risky_dstn",
    "make_labeled_shock_dstn",
    # Config dicts (backwards compatibility)
    "init_perf_foresight_labeled",
    "init_ind_shock_labeled",
    "init_risky_asset_labeled",
    "init_portfolio_labeled",
    "PF_labeled_constructor_dict",
    "ind_shock_labeled_constructor_dict",
    "risky_asset_labeled_constructor_dict",
    "init_portfolio_labeled_constructors",
]
