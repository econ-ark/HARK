"""
A model file for a Fisher 2-period consumption problem.
"""

from HARK.model import Control

# This way of distributing parameters across the scope is clunky
# Can be handled better if parsed from a YAML file, probably
# But it would be better to have a more graceful Python version as well.
CRRA = (2.0,)

model = {
    "shocks": {},
    "parameters": {
        "DiscFac": 0.96,
        "CRRA": CRRA,
        "Rfree": 1.03,
        "y": [1.0, 1.0],
        "BoroCnstArt": None,
    },
    "dynamics": {
        "m": lambda Rfree, a, y: Rfree * a + y,
        "c": Control(["m"]),
        "a": lambda m, c: m - c,
    },
    "reward": {"u": lambda c: c ** (1 - CRRA) / (1 - CRRA)},
}
