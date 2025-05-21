"""Specifies the full set of calibrated values required to estimate the EstimatingMicroDSOPs
model.  The empirical data is stored in a separate csv file and is loaded in setup_scf_data.
"""

# Discount Factor of 1.0 always
# income uncertainty doubles at retirement
# only estimate CRRA, Bequest params
from __future__ import annotations

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
from HARK.Calibration.Income.IncomeTools import Cagetti_income, parse_income_spec
from HARK.Calibration.life_tables.us_ssa.SSATools import parse_ssa_life_table
from HARK.distribution import DiscreteDistribution

# ---------------------------------------------------------------------------------
# - Define all of the model parameters for EstimatingMicroDSOPs and ConsumerExamples -
# ---------------------------------------------------------------------------------

# Assets grid
exp_nest = 1  # Number of times to "exponentially nest" when constructing a_grid
aXtraMin = 0.001  # Minimum end-of-period "assets above minimum" value
aXtraMax = 100  # Maximum end-of-period "assets above minimum" value
aXtraCount = 20  # Number of points in the grid of "assets above minimum"

# Artificial borrowing constraint
BoroCnstArt = 0.0  # imposed minimum level of end-of period assets
Rfree = 1.03  # Interest factor on assets

# Use cubic spline interpolation when True, linear interpolation when False
CubicBool = False
vFuncBool = False  # Whether to calculate the value function during solution

# Income process parameters
# Number of points in discrete approximation to permanent income shocks
PermShkCount = 7
# Number of points in discrete approximation to transitory income shocks
TranShkCount = 7
UnempPrb = 0.05  # Probability of unemployment while working
UnempPrbRet = 0.005  # Probability of "unemployment" while retired
IncUnemp = 0.3  # Unemployment benefits replacement rate
IncUnempRet = 0.0  # "Unemployment" benefits when retired
ss_variances = False  # Use the Sabelhaus-Song variance profiles
education = "College"  # Education level for income process

# Population age parameters
final_age = 120  # Age at which the problem ends (die with certainty)
retirement_age = 65  # Age at which the consumer retires
initial_age = 25  # Age at which the consumer enters the model
final_age_data = 95  # Age at which the data ends
age_interval = 5  # Interval between age groups

# Three point discrete distribution of initial w
init_w_to_y = np.array([0.17, 0.5, 0.83])
# Equiprobable discrete distribution of initial w
prob_w_to_y = np.array([0.33333, 0.33333, 0.33334])
num_agents = 10000  # Number of agents to simulate

# Bootstrap options
bootstrap_size = 50  # Number of re-estimations to do during bootstrap
seed = 1132023  # Just an integer to seed the estimation

params_to_estimate = ["CRRA"]

# Initial guess of the coefficient of relative risk aversion during estimation (rho)
init_CRRA = 2.0
# Bounds for rho; if violated, objective function returns "penalty value"
bounds_CRRA = [1.1, 20.0]

# Initial guess of the adjustment to the discount factor during estimation (beth)
init_DiscFac = 1.0
# Bounds for beth; if violated, objective function returns "penalty value"
bounds_DiscFac = [0.5, 1.1]

init_WealthShare = 0.05  # Initial guess of the wealth share parameter
bounds_WealthShare = [0.01, 0.99]  # Bounds for the wealth share parameter

init_WealthShift = 0.0  # Initial guess of the wealth shift parameter
bounds_WealthShift = [0.0, 100.0]  # Bounds for the wealth shift parameter

init_BeqFac = 1.0  # Initial guess of the bequest factor
bounds_BeqFac = [0.0, 100.0]  # Bounds for the bequest factor

init_BeqShift = 0.0  # Initial guess of the bequest shift parameter
bounds_BeqShift = [0.0, 70.0]  # Bounds for the bequest shift parameter

######################################################################
# Constructed parameters
######################################################################

# Total number of periods in the model
terminal_t = final_age - initial_age
retirement_t = retirement_age - initial_age - 1

# Income
income_spec = Cagetti_income[education]
# Replace retirement age
income_spec["age_ret"] = retirement_age
inc_calib = parse_income_spec(
    age_min=initial_age,
    age_max=final_age,
    **income_spec,
    SabelhausSong=ss_variances,
)

inc_calib["PermGroFac"][retirement_age - initial_age] = 0.9389

# use permgrofac = 0.9389 at retirement

# Age groups for the estimation: calculate average wealth-to-permanent income ratio
# for consumers within each of these age groups, compare actual to simulated data

age_groups = [
    list(range(start, start + age_interval))
    for start in range(initial_age + 1, final_age_data + 1, age_interval)
]

# generate labels as (25,30], (30,35], ...
age_labels = [f"({group[0]-1},{group[-1]}]" for group in age_groups]

# Generate mappings between the real ages in the groups and the indices of simulated data
age_mapping = dict(zip(age_labels, map(np.array, age_groups)))
sim_mapping = {
    label: np.array(group) - initial_age for label, group in zip(age_labels, age_groups)
}

remove_ages_from_scf = np.arange(
    retirement_age - age_interval + 1,
    retirement_age + age_interval + 1,
)  # remove retirement ages 61-70
remove_ages_from_snp = np.arange(
    retirement_age + age_interval + 1,
)  # only match ages 71 and older


# Survival probabilities over the lifecycle
liv_prb = parse_ssa_life_table(
    female=False,
    min_age=initial_age,
    max_age=final_age - 1,
    cohort=1960,
)

aNrmInit = DiscreteDistribution(
    prob_w_to_y,
    init_w_to_y,
    seed=seed,
).draw(N=num_agents)

bootstrap_options = {
    "bootstrap_size": bootstrap_size,
    "seed": seed,
}

minimize_options = {
    "algorithm": "tranquilo_ls",
    "multistart": True,
    "error_handling": "continue",
    "algo_options": {
        "convergence.absolute_params_tolerance": 1e-6,
        "convergence.absolute_criterion_tolerance": 1e-6,
        "stopping.max_iterations": 100,
        "stopping.max_criterion_evaluations": 200,
        "n_cores": 12,
    },
    "numdiff_options": {"n_cores": 12},
}

# -----------------------------------------------------------------------------
# -- Set up the dictionary "container" for making a basic lifecycle type ------
# -----------------------------------------------------------------------------

# Dictionary that can be passed to ConsumerType to instantiate
init_calibration = {
    "CRRA": 9.206775856414323,
    "DiscFac": 1.0,
    "BeqFac": 23.05054873023735, 
    "BeqShift": 45.64298427855443,
    "BeqCRRA": 9.206775856414323, 
    "Rfree": Rfree,
    "PermGroFac": inc_calib["PermGroFac"],
    "PermGroFacAgg": 1.0,
    "BoroCnstArt": BoroCnstArt,
    "PermShkStd": inc_calib["PermShkStd"][: retirement_t + 1]
    + [inc_calib["PermShkStd"][retirement_t]] * (terminal_t - retirement_t - 1),
    "PermShkCount": PermShkCount,
    "TranShkStd": inc_calib["TranShkStd"][: retirement_t + 1]
    + [inc_calib["TranShkStd"][retirement_t]] * (terminal_t - retirement_t - 1),
    "TranShkCount": TranShkCount,
    "T_cycle": terminal_t,
    "UnempPrb": UnempPrb,
    "UnempPrbRet": UnempPrbRet,
    "T_retire": retirement_t,
    "T_age": terminal_t,
    "IncUnemp": IncUnemp,
    "IncUnempRet": IncUnempRet,
    "aXtraMin": aXtraMin,
    "aXtraMax": aXtraMax,
    "aXtraCount": aXtraCount,
    "aXtraNestFac": exp_nest,
    "LivPrb": liv_prb,
    "AgentCount": num_agents,
    "seed": seed,
    "tax_rate": 0.0,
    "vFuncBool": vFuncBool,
    "CubicBool": CubicBool,
    "aNrmInit": aNrmInit,
    "neutral_measure": True,  # Harmemberg
    "sim_common_Rrisky": False,  # idiosyncratic risky return
    "WealthShift": init_WealthShift,
    "ChiFromOmega_N": 501,  # Number of gridpoints in chi-from-omega function
    "ChiFromOmega_bound": 15,  # Highest gridpoint to use for it
}

Eq_prem = 0.03
RiskyStd = 0.20

init_calibration["RiskyAvg"] = Rfree + Eq_prem
init_calibration["RiskyStd"] = RiskyStd

# from Mateo's JMP for College Educated
ElnR_nom = 0.020
VlnR = 0.424**2

TrueElnR_nom = 0.085
TrueVlnR = 0.170**2

logInflation = 0.024
logRfree_nom = 0.043
Rfree_real = np.exp(logRfree_nom - logInflation)  # 1.019

ElnR_real = ElnR_nom - logInflation
TrueElnR_real = TrueElnR_nom - logInflation


init_subjective_stock = {
    "Rfree": Rfree_real,  # from Mateo's JMP
    "RiskyAvg": np.exp(ElnR_real + 0.5 * VlnR),
    "RiskyStd": np.sqrt(np.exp(2 * ElnR_real + VlnR) * (np.exp(VlnR) - 1)),
    "RiskyAvgTrue": np.exp(TrueElnR_real + 0.5 * TrueVlnR),
    "RiskyStdTrue": np.sqrt(
        np.exp(2 * TrueElnR_real + TrueVlnR) * (np.exp(TrueVlnR) - 1),
    ),
}

# from Tao's JMP
init_subjective_labor = {
    "TranShkStd": [0.03] * (retirement_t + 1)
    + [0.03 * np.sqrt(2)] * (terminal_t - retirement_t - 1),
    "PermShkStd": [0.03] * (retirement_t + 1)
    + [0.03 * np.sqrt(2)] * (terminal_t - retirement_t - 1),
}
