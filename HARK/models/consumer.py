from HARK.distribution import Bernoulli, Lognormal, MeanOneLogNormal
from HARK.model import Control, DBlock, RBlock

"""
Blocks for consumption saving problem (not normalized)
in the style of Carroll's "Solution Methods for Solving
Microeconomic Dynamic Stochastic Optimization Problems"
"""

# TODO: Include these in calibration, then construct shocks
LivPrb = 0.98
TranShkStd = 0.1
RiskyStd = 0.1

calibration = {
    "DiscFac": 0.96,
    "CRRA": 2.0,
    "Rfree": 1.03,
    "EqP": 0.02,
    "LivPrb": LivPrb,
    "PermGroFac": 1.01,
    "BoroCnstArt": None,
}

consumption_block = DBlock(
    **{
        "name": "consumption",
        "shocks": {
            "live": Bernoulli(p=LivPrb),  # Move to tick or mortality block?
            "theta": MeanOneLogNormal(sigma=TranShkStd),
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

portfolio_block = DBlock(
    **{
        "name": "portfolio",
        "shocks": {
            "risky_return": Lognormal(
                mu=calibration["Rfree"] + calibration["EqP"], sigma=RiskyStd
            )
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

merged_block = RBlock(blocks=[consumption_block, portfolio_block, tick_block])
