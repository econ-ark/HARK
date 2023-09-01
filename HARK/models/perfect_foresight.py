from HARK.distribution import Bernoulli
from HARK.model import Control

# This way of distributing parameters across the scope is clunky
# Can be handled better if parsed from a YAML file, probably
# But it would be better to have a more graceful Python version as well.
CRRA = 2.0,
LivPrb = 0.98

model = {
    'shocks' : {
        'live' : Bernoulli(p=LivPrb),
    },
    'parameters' : {
        'DiscFac' : 0.96,
        'CRRA' : CRRA,
        'Rfree' : 1.03,
        'LivPrb' : LivPrb,
        'PermGroFac' : 1.01,
        'BoroCnstArt' : None,
    },
    'dynamics' : {
        'm' : lambda Rfree, a, y : Rfree * a + y,
        'c' : Control(['m']),
        'y' : lambda p : p,
        'p' : lambda PermGroFac, p: PermGroFac * p,
        'a' : lambda m, c : m - c
    },
    'reward' : {
        'u' : lambda c : c ** (1 - CRRA) / (1 - CRRA)
    }
}