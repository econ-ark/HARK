"""
Agent types for labeled consumption-saving models.

These classes combine the labeled solvers with HARK's agent framework,
providing complete agent types that can be instantiated and solved.
"""

from __future__ import annotations

from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.ConsumptionSaving.ConsPortfolioModel import PortfolioConsumerType
from HARK.ConsumptionSaving.ConsRiskyAssetModel import RiskyAssetConsumerType
from HARK.core import make_one_period_oo_solver

from .config import (
    IND_SHOCK_CONFIG,
    PERF_FORESIGHT_CONFIG,
    PORTFOLIO_CONFIG,
    RISKY_ASSET_CONFIG,
)
from .solvers import (
    ConsIndShockLabeledSolver,
    ConsPerfForesightLabeledSolver,
    ConsPortfolioLabeledSolver,
    ConsRiskyAssetLabeledSolver,
)

__all__ = [
    "PerfForesightLabeledType",
    "IndShockLabeledType",
    "RiskyAssetLabeledType",
    "PortfolioLabeledType",
]


class PerfForesightLabeledType(IndShockConsumerType):
    """
    A labeled perfect foresight consumer type.

    This agent has no uncertainty about income or interest rates.
    The only state variable is normalized market resources (m).

    Uses labeled xarray data structures for solutions, enabling
    clear variable naming and easier manipulation of results.
    """

    default_ = {
        "params": PERF_FORESIGHT_CONFIG.build_params(),
        "solver": make_one_period_oo_solver(ConsPerfForesightLabeledSolver),
        "model": "ConsPerfForesight.yaml",
    }

    def post_solve(self) -> None:
        """Skip post-solve processing (no stable points to calculate)."""
        pass


class IndShockLabeledType(PerfForesightLabeledType):
    """
    A labeled consumer type with idiosyncratic income shocks.

    This agent faces permanent and transitory shocks to income.
    Inherits from PerfForesightLabeledType and adds income uncertainty.

    Uses labeled xarray data structures for solutions.
    """

    default_ = {
        "params": IND_SHOCK_CONFIG.build_params(),
        "solver": make_one_period_oo_solver(ConsIndShockLabeledSolver),
        "model": "ConsIndShock.yaml",
    }


class RiskyAssetLabeledType(IndShockLabeledType, RiskyAssetConsumerType):
    """
    A labeled consumer type with risky asset investment.

    This agent can only save in a risky asset that pays a stochastic return.
    There is no risk-free savings option.

    Uses labeled xarray data structures for solutions.
    """

    default_ = {
        "params": RISKY_ASSET_CONFIG.build_params(),
        "solver": make_one_period_oo_solver(ConsRiskyAssetLabeledSolver),
        "model": "ConsRiskyAsset.yaml",
    }


class PortfolioLabeledType(PortfolioConsumerType):
    """
    A labeled consumer type with optimal portfolio choice.

    This agent can save in both a risk-free asset and a risky asset,
    choosing the optimal allocation between them each period.

    Uses labeled xarray data structures for solutions.

    Note
    ----
    Unlike other labeled types, this class inherits directly from
    `PortfolioConsumerType` rather than from `IndShockLabeledType`.
    This is because `PortfolioConsumerType` provides essential portfolio-
    specific functionality (share grids, portfolio optimization methods)
    that would be difficult to replicate. The labeled solver handles
    the xarray integration while the parent class handles agent lifecycle.
    """

    default_ = {
        "params": PORTFOLIO_CONFIG.build_params(),
        "solver": make_one_period_oo_solver(ConsPortfolioLabeledSolver),
        "model": "ConsPortfolio.yaml",
    }

    def post_solve(self) -> None:
        """Skip post-solve processing."""
        pass
