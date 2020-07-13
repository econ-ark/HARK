# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: collapsed,code_folding
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from HARK.ConsumptionSaving.PerfForesightNumbaModel import (
    PerfForesightConsumerTypeNumba,
)

PerfForesightDict = {
    # Parameters actually used in the solution method
    "CRRA": 2.0,  # Coefficient of relative risk aversion
    "Rfree": 1.03,  # Interest factor on assets
    "DiscFac": 0.96,  # Default intertemporal discount factor
    "LivPrb": [0.98],  # Survival probability
    "PermGroFac": [1.01],  # Permanent income growth factor
    # Parameters that characterize the nature of time
    "T_cycle": 1,  # Number of periods in the cycle for this agent type
    "cycles": 0,  # Number of times the cycle occurs (0 --> infinitely repeated)
}

PFexample = PerfForesightConsumerTypeNumba(**PerfForesightDict)
PFexample.solve()
