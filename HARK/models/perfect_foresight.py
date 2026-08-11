from HARK.distributions import Bernoulli
from HARK.model import Control, DBlock

# This way of distributing parameters across the scope is clunky
# Can be handled better if parsed from a YAML file, probably
# But it would be better to have a more graceful Python version as well.
LivPrb = 0.98

calibration = {
    "DiscFac": 0.96,
    "CRRA": 2.0,
    "Rfree": 1.03,
    "LivPrb": LivPrb,
    "PermGroFac": 1.01,
    "BoroCnstArt": None,
}

block = DBlock(
    **{
        "name": "consumption",
        "shocks": {
            "live": Bernoulli(p=LivPrb),
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
