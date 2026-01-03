"""
Labeled consumption-saving models using xarray.

This package provides consumption-saving models that use xarray's labeled
data structures for clear variable naming and easier manipulation of results.

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

from .agents import (
    IndShockLabeledType,
    PerfForesightLabeledType,
    PortfolioLabeledType,
    RiskyAssetLabeledType,
)
from .config import (
    IND_SHOCK_CONFIG,
    PERF_FORESIGHT_CONFIG,
    PORTFOLIO_CONFIG,
    RISKY_ASSET_CONFIG,
    ModelConfig,
    get_config,
)
from .factories import (
    make_labeled_inc_shk_dstn,
    make_labeled_risky_dstn,
    make_labeled_shock_dstn,
    make_solution_terminal_labeled,
)
from .solution import ConsumerSolutionLabeled, ValueFuncCRRALabeled
from .solvers import (
    BaseLabeledSolver,
    ConsFixedPortfolioLabeledSolver,
    ConsIndShockLabeledSolver,
    ConsPerfForesightLabeledSolver,
    ConsPortfolioLabeledSolver,
    ConsRiskyAssetLabeledSolver,
)
from .transitions import (
    FixedPortfolioTransitions,
    IndShockTransitions,
    PerfectForesightTransitions,
    PortfolioTransitions,
    RiskyAssetTransitions,
    Transitions,
)

__all__ = [
    # Solution classes
    "ValueFuncCRRALabeled",
    "ConsumerSolutionLabeled",
    # Solvers
    "BaseLabeledSolver",
    "ConsPerfForesightLabeledSolver",
    "ConsIndShockLabeledSolver",
    "ConsRiskyAssetLabeledSolver",
    "ConsFixedPortfolioLabeledSolver",
    "ConsPortfolioLabeledSolver",
    # Transitions
    "Transitions",
    "PerfectForesightTransitions",
    "IndShockTransitions",
    "RiskyAssetTransitions",
    "FixedPortfolioTransitions",
    "PortfolioTransitions",
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
    # Config
    "ModelConfig",
    "PERF_FORESIGHT_CONFIG",
    "IND_SHOCK_CONFIG",
    "RISKY_ASSET_CONFIG",
    "PORTFOLIO_CONFIG",
    "get_config",
]
