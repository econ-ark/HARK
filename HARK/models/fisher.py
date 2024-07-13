"""
A model file for a Fisher 2-period consumption problem.
"""

from HARK.model import Control, DBlock


calibration = {
    "DiscFac": 0.96,
    "CRRA": (2.0,),
    "Rfree": 1.03,
    "y": [1.0, 1.0],
    "BoroCnstArt": None,
}

block = DBlock(
    **{
        "shocks": {},
        "dynamics": {
            "m": lambda Rfree, a, y: Rfree * a + y,
            "c": Control(["m"]),
            "a": lambda m, c: m - c,
        },
        "reward": {"u": lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA)},
    }
)
