from HARK.distributions import Bernoulli, Lognormal, MeanOneLogNormal
from HARK.model import Control, DBlock, RBlock

"""
Blocks for consumption saving problem (not normalized)
in the style of Carroll's "Solution Methods for Solving
Microeconomic Dynamic Stochastic Optimization Problems"
"""

calibration = {
    "DiscFac": 0.96,
    "CRRA": 2.0,
    "R": 1.03,  # note: this can be overriden by the portfolio dynamics
    "Rfree": 1.03,
    "EqP": 0.02,
    "LivPrb": 0.98,
    "PermGroFac": 1.01,
    "BoroCnstArt": None,
    "TranShkStd": 0.1,
    "RiskyStd": 0.1,
}

consumption_block = DBlock(
    **{
        "name": "consumption",
        "shocks": {
            "live": (Bernoulli, {"p": "LivPrb"}),  # Move to tick or mortality block?
            "theta": (MeanOneLogNormal, {"sigma": "TranShkStd"}),
        },
        "dynamics": {
            "b": lambda k, R: k * R,
            "y": lambda p, theta: p * theta,
            "m": lambda b, y: b + y,
            "c": Control(["m"]),
            "p": lambda PermGroFac, p: PermGroFac * p,
            "a": lambda m, c: m - c,
        },
        "reward": {"u": lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA)},
    }
)

consumption_block_normalized = DBlock(
    **{
        "name": "consumption normalized",
        "shocks": {
            "live": (Bernoulli, {"p": "LivPrb"}),  # Move to tick or mortality block?
            "theta": (MeanOneLogNormal, {"sigma": "TranShkStd"}),
        },
        "dynamics": {
            "b": lambda k, R, PermGroFac: k * R / PermGroFac,
            "m": lambda b, theta: b + theta,
            "c": Control(["m"]),
            "a": "m - c",
        },
        "reward": {"u": lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA)},
    }
)

portfolio_block = DBlock(
    **{
        "name": "portfolio",
        "shocks": {
            "risky_return": (Lognormal, {"mean": "Rfree + EqP", "std": "RiskyStd"})
        },
        "dynamics": {
            "stigma": Control(["a"]),
            "R": lambda stigma, Rfree, risky_return: Rfree
            + (risky_return - Rfree) * stigma,
        },
    }
)

tick_block = DBlock(
    **{
        "name": "tick",
        "dynamics": {
            "k": lambda a: a,
        },
    }
)

cons_problem = RBlock(blocks=[consumption_block, tick_block])
cons_portfolio_problem = RBlock(blocks=[consumption_block, portfolio_block, tick_block])
