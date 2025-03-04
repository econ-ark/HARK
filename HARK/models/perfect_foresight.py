from HARK.distributions import Bernoulli
from HARK.model import Control, DBlock

calibration = {
    "DiscFac": 0.96,
    "CRRA": 2.0,
    "Rfree": 1.03,
    "LivPrb": 0.98,
    "PermGroFac": 1.01,
    "BoroCnstArt": None,
}

block = DBlock(
    **{
        "name": "consumption",
        "shocks": {
            "live": Bernoulli(p=calibration["LivPrb"]),
        },
        "dynamics": {
            "y": lambda p: p,
            "m": lambda Rfree, a, y: Rfree * a + y,
            "c": Control(["m"]),
            "p": lambda PermGroFac, p: PermGroFac * p,
            "a": lambda m, c: m - c,
        },
        "reward": {"u": lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA)},
    }
)
