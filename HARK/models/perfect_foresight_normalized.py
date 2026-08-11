from HARK.distributions import Bernoulli
from HARK.model import Control, DBlock

# This way of distributing parameters across the scope is clunky
# Can be handled better if parsed from a YAML file, probably
# But it would be better to have a more graceful Python version as well.
CRRA = (2.0,)
LivPrb = 0.98

calibration = {
    "DiscFac": 0.96,
    "CRRA": CRRA,
    "Rfree": 1.03,
    "LivPrb": LivPrb,
    "PermGroFac": 1.01,
    "BoroCnstArt": None,
}

block = DBlock(
    **{
        "shocks": {
            "live": Bernoulli(p=LivPrb),
        },
        "dynamics": {
            "p": lambda PermGroFac, p: PermGroFac * p,
            "r_eff": lambda Rfree, PermGroFac: Rfree / PermGroFac,
            "b_nrm": lambda r_eff, a_nrm: r_eff * a_nrm,
            "m_nrm": lambda b_nrm: b_nrm + 1,
            "c_nrm": Control(["m_nrm"]),
            "a_nrm": lambda m_nrm, c_nrm: m_nrm - c_nrm,
        },
        "reward": {"u": lambda c: c ** (1 - CRRA) / (1 - CRRA)},
    }
)
