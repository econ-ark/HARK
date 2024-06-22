from HARK.distribution import Bernoulli, Lognormal, MeanOneLogNormal
from HARK.model import Control

"""
This is an example of what a full configuration looks like for:

(a) a true model

(b) approximations made to it to enable efficient computation

(c) additional parameters needed to solve and simulate the model.

This file shows the model configuraiton in Python.

Another file will show a model configuration in YAML, which gets parsed into
a data structure much like this one.
"""

model_config = {
    "calibration": {
        "DiscFac": 0.96,
        "CRRA": 2.0,
        "R": 1.03,  # note: this can be overriden by the portfolio dynamics
        "Rfree": 1.03,
        "EqP": 0.02,
        "LivPrb": 0.98,
        "PermGroFac": 1.01,
        "BoroCnstArt": None,
        "TranShkStd": 0.1,
    },
    "agent": {
        "size": 10,  # the population size. Is this only for the simulation?
        "blocks": [
            {
                "name": "consumption normalized",
                "shocks": {
                    "live": [Bernoulli, {"p": "LivPrb"}],
                    "theta": [MeanOneLogNormal, {"sigma": "TranShkStd"}],
                },
                "dynamics": {
                    "b": lambda k, R, PermGroFac: k * R / PermGroFac,
                    "m": lambda b, theta: b + theta,
                    "c": Control(["m"]),
                    "a": lambda m, c: m - c,
                },
                "reward": {"u": lambda c, CRRA: c ** (1 - CRRA) / (1 - CRRA)},
            },
            {
                "name": "portfolio",
                "shocks": {
                    "risky_return": [
                        Lognormal.from_mean_std,
                        {"mean": "Rfree + EqP", "std": 0.1},
                    ]
                },
                "dynamics": {
                    "stigma": Control(["a"]),
                    "R": lambda stigma, Rfree, risky_return: Rfree
                    + (risky_return - Rfree) * stigma,
                },
            },
            {
                "name": "tick",
                "dynamics": {
                    "k": lambda a: a,
                },
            },
        ],
    },
    "approximation": {
        "theta": {"N": 5},
        "risky_return": {"N": 5},
    },
    "workflows": [
        {"action": "solve", "algorithm": "vbi"},
        {
            "action": "simulate",
            "initialization": {  # initial values # type: ignore
                "k": Lognormal(-6, 0),
                "R": 1.03,
            },
            "population": 10,  # ten agents
            "T": 20,  # total number of simulated periods
        },
    ],
}
