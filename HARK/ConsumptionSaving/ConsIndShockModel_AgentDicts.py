# -*- coding: utf-8 -*-
from copy import copy, deepcopy
from HARK.datasets.life_tables.us_ssa.SSATools import parse_ssa_life_table
from HARK.datasets.SCF.WealthIncomeDist.SCFDistTools \
    import income_wealth_dists_from_scf
from HARK.Calibration.Income.IncomeTools \
    import (parse_income_spec, parse_time_params, Cagetti_income)

"""
Define parameters for various Consumer AgentTypes
"""

init_perfect_foresight_plus = {}

# The info below is optional at present but may become mandatory as the toolkit evolves
# 'Primitives' define the 'true' model that we think of ourselves as trying to solve
# (the limit as approximation error reaches zero)

init_perfect_foresight_plus.update(
    {'prmtv_par': ['CRRA', 'Rfree', 'DiscFac', 'LivPrb', 'PermGroFac', 'BoroCnstArt', 'PermGroFacAgg', 'T_cycle', 'cycles']})
# Approximation parameters define the precision of the approximation
# Limiting values for approximation parameters: values such that, as all such parameters approach their limits,
# the approximation gets arbitrarily close to the 'true' model
init_perfect_foresight_plus.update(  # In principle, kinks exist all the way to infinity
    {'aprox_par': {'MaxKinks': '500'}})
init_perfect_foresight_plus.update(  # In principle, kinks exist all the way to infinity
    {'aprox_lim': {'MaxKinks': float('inf')}})

# The simulation stge of the problem requires additional parameterization
init_perfect_foresight_plus.update(  # The 'primitives' for the simulation
    {'prmtv_sim': ['aNrmInitMean', 'aNrmInitStd', 'pLvlInitMean', 'pLvlInitStd']})
init_perfect_foresight_plus.update({  # Approximation parameters for monte carlo sims
    'sim_mcrlo': ['AgentCount', 'T_age']
})
init_perfect_foresight_plus.update({  # Limiting values that define 'true' simulation
    'sim_mcrlo_lim': {
        'AgentCount': 'infinity',
        'T_age': 'infinity'
    }
})

# Optional more detailed _fcts about various parameters
CRRA_fcts = {
    'about': 'Coefficient of Relative Risk Aversion'}
CRRA_fcts.update({'latexexpr': r'\providecommand{\CRRA}{\rho}\CRRA'})
CRRA_fcts.update({'_unicode_': 'ρ'})  # \rho is Greek r: relative risk aversion rrr
CRRA_fcts.update({'prmtv_par': 'True'})
init_perfect_foresight_plus['prmtv_par'].append('CRRA')
# init_perfect_foresight_plus['_fcts'].update({'CRRA': CRRA_fcts})
init_perfect_foresight_plus.update({'CRRA_fcts': CRRA_fcts})

DiscFac_fcts = {
    'about': 'Pure time preference rate'}
DiscFac_fcts.update({'latexexpr': r'\providecommand{\DiscFac}{\beta}\DiscFac'})
DiscFac_fcts.update({'_unicode_': 'β'})
DiscFac_fcts.update({'prmtv_par': 'True'})
init_perfect_foresight_plus['prmtv_par'].append('DiscFac')
# init_perfect_foresight_plus['_fcts'].update({'DiscFac': DiscFac_fcts})
init_perfect_foresight_plus.update({'DiscFac_fcts': DiscFac_fcts})

LivPrb_fcts = {
    'about': 'Probability of survival from this period to next'}
LivPrb_fcts.update({'latexexpr': r'\providecommand{\LivPrb}{\Pi}\LivPrb'})
LivPrb_fcts.update({'_unicode_': 'Π'})  # \Pi mnemonic: 'Probability of surival'
LivPrb_fcts.update({'prmtv_par': 'True'})
init_perfect_foresight_plus['prmtv_par'].append('LivPrb')
# init_perfect_foresight_plus['_fcts'].update({'LivPrb': LivPrb_fcts})
init_perfect_foresight_plus.update({'LivPrb_fcts': LivPrb_fcts})

Rfree_fcts = {
    'about': 'Risk free interest factor'}
Rfree_fcts.update({'latexexpr': r'\providecommand{\Rfree}{\mathsf{R}}\Rfree'})
Rfree_fcts.update({'_unicode_': 'R'})
Rfree_fcts.update({'prmtv_par': 'True'})
init_perfect_foresight_plus['prmtv_par'].append('Rfree')
# init_perfect_foresight_plus['_fcts'].update({'Rfree': Rfree_fcts})
init_perfect_foresight_plus.update({'Rfree_fcts': Rfree_fcts})

PermGroFac_fcts = {
    'about': 'Growth factor for permanent income'}
PermGroFac_fcts.update({'latexexpr': r'\providecommand{\PermGroFac}{\Gamma}\PermGroFac'})
PermGroFac_fcts.update({'_unicode_': 'Γ'})  # \Gamma is Greek G for Growth
PermGroFac_fcts.update({'prmtv_par': 'True'})
init_perfect_foresight_plus['prmtv_par'].append('PermGroFac')
# init_perfect_foresight_plus['_fcts'].update({'PermGroFac': PermGroFac_fcts})
init_perfect_foresight_plus.update({'PermGroFac_fcts': PermGroFac_fcts})

PermGroFacAgg_fcts = {
    'about': 'Growth factor for aggregate permanent income'}
# PermGroFacAgg_fcts.update({'latexexpr': r'\providecommand{\PermGroFacAgg}{\Gamma}\PermGroFacAgg'})
# PermGroFacAgg_fcts.update({'_unicode_': 'Γ'})  # \Gamma is Greek G for Growth
PermGroFacAgg_fcts.update({'prmtv_par': 'True'})
init_perfect_foresight_plus['prmtv_par'].append('PermGroFacAgg')
# init_perfect_foresight_plus['_fcts'].update({'PermGroFacAgg': PermGroFacAgg_fcts})
init_perfect_foresight_plus.update({'PermGroFacAgg_fcts': PermGroFacAgg_fcts})

BoroCnstArt_fcts = {
    'about': 'If not None, maximum future income borrowable (normalized by current permanent income)'}
BoroCnstArt_fcts.update({'latexexpr': r'\providecommand{\BoroCnstArt}{\underline{a}}\BoroCnstArt'})
BoroCnstArt_fcts.update({'prmtv_par': 'True'})
init_perfect_foresight_plus['prmtv_par'].append('BoroCnstArt')
# init_perfect_foresight_plus['_fcts'].update({'BoroCnstArt': BoroCnstArt_fcts})
init_perfect_foresight_plus.update({'BoroCnstArt_fcts': BoroCnstArt_fcts})

MaxKinks_fcts = {
    'about': 'PF Constrained model solves to period T-MaxKinks,'
    ' where the solution has exactly this many kink points'}
MaxKinks_fcts.update({'prmtv_par': 'False'})
# init_perfect_foresight_plus['prmtv_par'].append('MaxKinks')
# init_perfect_foresight_plus['_fcts'].update({'MaxKinks': MaxKinks_fcts})
init_perfect_foresight_plus.update({'MaxKinks_fcts': MaxKinks_fcts})

AgentCount_fcts = {
    'about': 'Number of agents to use in baseline Monte Carlo simulation'}
AgentCount_fcts.update(
    {'latexexpr': r'\providecommand{\AgentCount}{N}\AgentCount'})
AgentCount_fcts.update({'sim_mcrlo': 'True'})
AgentCount_fcts.update({'sim_mcrlo_lim': 'infinity'})
# init_perfect_foresight_plus['sim_mcrlo'].append('AgentCount')
# init_perfect_foresight_plus['_fcts'].update({'AgentCount': AgentCount_fcts})
init_perfect_foresight_plus.update({'AgentCount_fcts': AgentCount_fcts})

aNrmInitMean_fcts = {
    'about': 'Mean initial population value of aNrm'}
aNrmInitMean_fcts.update({'sim_mcrlo': 'True'})
aNrmInitMean_fcts.update({'sim_mcrlo_lim': 'infinity'})
init_perfect_foresight_plus['sim_mcrlo'].append('aNrmInitMean')
# init_perfect_foresight_plus['_fcts'].update({'aNrmInitMean': aNrmInitMean_fcts})
init_perfect_foresight_plus.update({'aNrmInitMean_fcts': aNrmInitMean_fcts})

aNrmInitStd_fcts = {
    'about': 'Std dev of initial population value of aNrm'}
aNrmInitStd_fcts.update({'sim_mcrlo': 'True'})
init_perfect_foresight_plus['sim_mcrlo'].append('aNrmInitStd')
# init_perfect_foresight_plus['_fcts'].update({'aNrmInitStd': aNrmInitStd_fcts})
init_perfect_foresight_plus.update({'aNrmInitStd_fcts': aNrmInitStd_fcts})

pLvlInitMean_fcts = {
    'about': 'Mean initial population value of log pLvl'}
pLvlInitMean_fcts.update({'sim_mcrlo': 'True'})
init_perfect_foresight_plus['sim_mcrlo'].append('pLvlInitMean')
# init_perfect_foresight_plus['_fcts'].update({'pLvlInitMean': pLvlInitMean_fcts})
init_perfect_foresight_plus.update({'pLvlInitMean_fcts': pLvlInitMean_fcts})

pLvlInitStd_fcts = {
    'about': 'Mean initial std dev of log ppLvl'}
pLvlInitStd_fcts.update({'sim_mcrlo': 'True'})
init_perfect_foresight_plus['sim_mcrlo'].append('pLvlInitStd')
# init_perfect_foresight_plus['_fcts'].update({'pLvlInitStd': pLvlInitStd_fcts})
init_perfect_foresight_plus.update({'pLvlInitStd_fcts': pLvlInitStd_fcts})

T_age_fcts = {
    'about': 'Age after which simulated agents are automatically killedl'}
T_age_fcts.update({'sim_mcrlo': 'False'})
# init_perfect_foresight_plus['_fcts'].update({'T_age': T_age_fcts})
init_perfect_foresight_plus.update({'T_age_fcts': T_age_fcts})

T_cycle_fcts = {
    'about': 'Number of periods in a "cycle" (like, a lifetime) for this agent type'}
# init_perfect_foresight_plus['_fcts'].update({'T_cycle': T_cycle_fcts})
init_perfect_foresight_plus.update({'T_cycle_fcts': T_cycle_fcts})

cycles_fcts = {
    'about': 'Number of times the sequence of periods/stages should be solved'}
# init_perfect_foresight_plus['_fcts'].update({'cycle': cycles_fcts})
init_perfect_foresight_plus.update({'cycles_fcts': cycles_fcts})
cycles_fcts.update({'prmtv_par': 'True'})
init_perfect_foresight_plus['prmtv_par'].append('cycles')


# Make a dictionary to specify a perfect foresight consumer type
init_perfect_foresight = {
    'CRRA': 2.0,          # Coefficient of relative risk aversion,
    'Rfree': 1.03,        # Interest factor on assets
    'DiscFac': 0.96,      # Intertemporal discount factor
    'LivPrb': [0.98],     # Survival probability
    'PermGroFac': [1.01],  # Permanent income growth factor
    'BoroCnstArt': None,  # Artificial borrowing constraint
    'T_cycle': 1,         # Num of periods in a finite horizon cycle (like, a life cycle)
    'PermGroFacAgg': 1.0,  # Aggregate income growth factor (multiplies individual)
    'MaxKinks': None,      # Maximum number of grid points to allow in cFunc
    'AgentCount': 10000,  # Number of agents of this type (only matters for simulation)
    'aNrmInitMean': 0.0,  # Mean of log initial assets (only matters for simulation)
    'aNrmInitStd': 1.0,  # Standard deviation of log initial assets (only for simulation)
    'pLvlInitMean': 0.0,  # Mean of log initial permanent income (only matters for simulation)
    # Standard deviation of log initial permanent income (only matters for simulation)
    'pLvlInitStd': 0.0,
    # Aggregate permanent income growth factor: portion of PermGroFac attributable to aggregate productivity growth (only matters for simulation)
    'T_age': None,       # Age after which simulated agents are automatically killed
    # Optional extra _fcts about the model and its calibration
    **init_perfect_foresight_plus
}

# The info above is necessary and sufficient for defining the consumer
# The info below is supplemental
# Some of it is required for further purposes

# Parameters required for a (future) matrix-based discretization of the problem
init_idiosyncratic_shocks_plus = {}
init_idiosyncratic_shocks_plus.update({
    'matrx_par': {
        'mcrlo_aXtraCount': '100',
        'mcrlo_aXtraMin': 0.001,
        'mcrlo_aXtraMax': 20,
    }})
init_idiosyncratic_shocks_plus.update({
    'matrx_lim': {
        'mcrlo_aXtraCount': float('inf'),
        'mcrlo_aXtraMin': float('inf'),
        'mcrlo_aXtraMax': float('inf')
    }})
init_idiosyncratic_shocks_plus['prmtv_par'] = []
init_idiosyncratic_shocks_plus['aprox_lim'] = {}

#  add parameters that were not part of perfect foresight model
# Primitives
init_idiosyncratic_shocks_plus['prmtv_par'].append('permShkStd')
init_idiosyncratic_shocks_plus['prmtv_par'].append('tranShkStd')
init_idiosyncratic_shocks_plus['prmtv_par'].append('UnempPrb')
init_idiosyncratic_shocks_plus['prmtv_par'].append('UnempPrbRet')
init_idiosyncratic_shocks_plus['prmtv_par'].append('IncUnempRet')
init_idiosyncratic_shocks_plus['prmtv_par'].append('BoroCnstArt')
init_idiosyncratic_shocks_plus['prmtv_par'].append('tax_rate')
init_idiosyncratic_shocks_plus['prmtv_par'].append('T_retire')

# Approximation parameters and their limits (if any)
# init_idiosyncratic_shocks_plus['aprox_par'].append('permShkCount')
init_idiosyncratic_shocks_plus['aprox_lim'].update({'permShkCount': 'infinity'})
# init_idiosyncratic_shocks_plus['aprox_par'].append('tranShkCount')
init_idiosyncratic_shocks_plus['aprox_lim'].update({'tranShkCount': 'infinity'})
# init_idiosyncratic_shocks_plus['aprox_par'].append('aXtraMin')
init_idiosyncratic_shocks_plus['aprox_lim'].update({'aXtraMin': float('0.0')})
# init_idiosyncratic_shocks_plus['aprox_par'].append('aXtraMax')
init_idiosyncratic_shocks_plus['aprox_lim'].update({'aXtraMax': float('inf')})
# init_idiosyncratic_shocks_plus['aprox_par'].append('aXtraNestFac')
init_idiosyncratic_shocks_plus['aprox_lim'].update({'aXtraNestFac': None})
# init_idiosyncratic_shocks_plus['aprox_par'].append('aXtraCount')
init_idiosyncratic_shocks_plus['aprox_lim'].update({'aXtraCount': None})

IncShkDstn_fcts = {
    'about': 'Income Shock Distribution: .X[0] and .X[1] retrieve shocks, .pmf retrieves probabilities'}
IncShkDstn_fcts.update({'py___code': r'construct_lognormal_income_process_unemployment'})
# init_idiosyncratic_shocks_plus['_fcts'].update({'IncShkDstn': IncShkDstn_fcts})
init_idiosyncratic_shocks_plus.update({'IncShkDstn_fcts': IncShkDstn_fcts})

permShkStd_fcts = {
    'about': 'Standard deviation for lognormal shock to permanent income'}
permShkStd_fcts.update({'latexexpr': r'\permShkStd'})
# init_idiosyncratic_shocks_plus['_fcts'].update({'permShkStd': permShkStd_fcts})
init_idiosyncratic_shocks_plus.update({'permShkStd_fcts': permShkStd_fcts})

tranShkStd_fcts = {
    'about': 'Standard deviation for lognormal shock to permanent income'}
tranShkStd_fcts.update({'latexexpr': r'\tranShkStd'})
# init_idiosyncratic_shocks_plus['_fcts'].update({'tranShkStd': tranShkStd_fcts})
init_idiosyncratic_shocks_plus.update({'tranShkStd_fcts': tranShkStd_fcts})

UnempPrb_fcts = {
    'about': 'Probability of unemployment while working'}
UnempPrb_fcts.update({'latexexpr': r'\UnempPrb'})
UnempPrb_fcts.update({'_unicode_': '℘'})
# init_idiosyncratic_shocks_plus['_fcts'].update({'UnempPrb': UnempPrb_fcts})
init_idiosyncratic_shocks_plus.update({'UnempPrb_fcts': UnempPrb_fcts})

UnempPrbRet_fcts = {
    'about': '"unemployment" in retirement = big medical shock'}
UnempPrbRet_fcts.update({'latexexpr': r'\UnempPrbRet'})
# init_idiosyncratic_shocks_plus['_fcts'].update({'UnempPrbRet': UnempPrbRet_fcts})
init_idiosyncratic_shocks_plus.update({'UnempPrbRet_fcts': UnempPrbRet_fcts})

IncUnemp_fcts = {
    'about': 'Unemployment insurance replacement rate'}
IncUnemp_fcts.update({'latexexpr': r'\IncUnemp'})
IncUnemp_fcts.update({'_unicode_': 'μ'})
# init_idiosyncratic_shocks_plus['_fcts'].update({'IncUnemp': IncUnemp_fcts})
init_idiosyncratic_shocks_plus.update({'IncUnemp_fcts': IncUnemp_fcts})

IncUnempRet_fcts = {
    'about': 'Size of medical shock (frac of perm inc)'}
# init_idiosyncratic_shocks_plus['_fcts'].update({'IncUnempRet': IncUnempRet_fcts})
init_idiosyncratic_shocks_plus.update({'IncUnempRet_fcts': IncUnempRet_fcts})

tax_rate_fcts = {
    'about': 'Flat income tax rate'}
tax_rate_fcts.update({
    'about': 'Size of medical shock (frac of perm inc)'})
# init_idiosyncratic_shocks_plus['_fcts'].update({'tax_rate': tax_rate_fcts})
init_idiosyncratic_shocks_plus.update({'tax_rate_fcts': tax_rate_fcts})

T_retire_fcts = {
    'about': 'Period of retirement (0 --> no retirement)'}
# init_idiosyncratic_shocks_plus['_fcts'].update({'T_retire': T_retire_fcts})
init_idiosyncratic_shocks_plus.update({'T_retire_fcts': T_retire_fcts})

permShkCount_fcts = {
    'about': 'Num of pts in discrete approx to permanent income shock dstn'}
# init_idiosyncratic_shocks_plus['_fcts'].update({'permShkCount': permShkCount_fcts})
init_idiosyncratic_shocks_plus.update({'permShkCount_fcts': permShkCount_fcts})

tranShkCount_fcts = {
    'about': 'Num of pts in discrete approx to transitory income shock dstn'}
# init_idiosyncratic_shocks_plus['_fcts'].update({'tranShkCount': tranShkCount_fcts})
init_idiosyncratic_shocks_plus.update({'tranShkCount_fcts': tranShkCount_fcts})

aXtraMin_fcts = {
    'about': 'Minimum end-of-period "assets above minimum" value'}
# init_idiosyncratic_shocks_plus['_fcts'].update({'aXtraMin': aXtraMin_fcts})
init_idiosyncratic_shocks_plus.update({'aXtraMin_fcts': aXtraMin_fcts})

aXtraMax_fcts = {
    'about': 'Maximum end-of-period "assets above minimum" value'}
# init_idiosyncratic_shocks_plus['_fcts'].update({'aXtraMax': aXtraMax_fcts})
init_idiosyncratic_shocks_plus.update({'aXtraMax_fcts': aXtraMax_fcts})

aXtraNestFac_fcts = {
    'about': 'Exponential nesting factor when constructing "assets above minimum" grid'}
# init_idiosyncratic_shocks_plus['_fcts'].update({'aXtraNestFac': aXtraNestFac_fcts})
init_idiosyncratic_shocks_plus.update({'aXtraNestFac_fcts': aXtraNestFac_fcts})

aXtraCount_fcts = {
    'about': 'Number of points in the grid of "assets above minimum"'}
# init_idiosyncratic_shocks_plus['_fcts'].update({'aXtraMax': aXtraCount_fcts})
init_idiosyncratic_shocks_plus.update({'aXtraMax_fcts': aXtraCount_fcts})

aXtraCount_fcts = {
    'about': 'Number of points to include in grid of assets above minimum possible'}
# init_idiosyncratic_shocks_plus['_fcts'].update({'aXtraCount': aXtraCount_fcts})
init_idiosyncratic_shocks_plus.update({'aXtraCount_fcts': aXtraCount_fcts})

aXtraExtra_fcts = {
    'about': 'List of other values of "assets above minimum" to add to the grid (e.g., 10000)'}
# init_idiosyncratic_shocks_plus['_fcts'].update({'aXtraExtra': aXtraExtra_fcts})
init_idiosyncratic_shocks_plus.update({'aXtraExtra_fcts': aXtraExtra_fcts})

aXtraGrid_fcts = {
    'about': 'Grid of values to add to minimum possible value to obtain actual end-of-period asset grid'}
# init_idiosyncratic_shocks_plus['_fcts'].update({'aXtraGrid': aXtraGrid_fcts})
init_idiosyncratic_shocks_plus.update({'aXtraGrid_fcts': aXtraGrid_fcts})

vFuncBool_fcts = {
    'about': 'Whether to calculate the value function during solution'
}
# init_idiosyncratic_shocks_plus['_fcts'].update({'vFuncBool': vFuncBool_fcts})
init_idiosyncratic_shocks_plus.update({'vFuncBool_fcts': vFuncBool_fcts})

CubicBool_fcts = {
    'about': 'Use cubic spline interpolation when True, linear interpolation when False'
}
# init_idiosyncratic_shocks_plus['_fcts'].update({'CubicBool': CubicBool_fcts})
init_idiosyncratic_shocks_plus.update({'CubicBool_fcts': CubicBool_fcts})

# Make a dictionary to specify an idiosyncratic income shocks consumer
init_idiosyncratic_shocks = dict(
    init_perfect_foresight,
    **{
        # Income process variables
        "permShkStd": [0.1],  # Standard deviation of log permanent income shocks
        "tranShkStd": [0.1],  # Standard deviation of log transitory income shocks
        "UnempPrb": 0.05,  # Probability of unemployment while working
        "UnempPrbRet": 0.005,  # Probability of "unemployment" while retired
        "IncUnemp": 0.3,  # Unemployment benefits replacement rate
        "IncUnempRet": 0.0,  # "Unemployment" benefits when retired
        "BoroCnstArt": 0.0,  # Artificial borrowing constraint; imposed minimum level of end-of period assets
        "tax_rate": 0.0,  # Flat income tax rate
        "T_retire": 0,  # Period of retirement (0 --> no retirement)
        # Parameters governing construction of income process
        "permShkCount": 7,  # Number of points in discrete approximation to permanent income shocks
        "tranShkCount": 7,  # Number of points in discrete approximation to transitory income shocks
        # parameters governing construction of grid of assets above min value
        "aXtraMin": 0.001,  # Minimum end-of-period "assets above minimum" value
        "aXtraMax": 20,     # Maximum end-of-period "assets above minimum" value
        "aXtraNestFac": 3,  # Exponential nesting factor when constructing "assets above minimum" grid
        "aXtraCount": 48,   # Number of points in the grid of "assets above minimum"
        # list other values of "assets above minimum" to add to the grid (e.g., 10000)
        "aXtraExtra": [None],
        "vFuncBool": False,  # Whether to calculate the value function during solution
        "CubicBool": False,  # Use cubic spline interpolation when True, linear interpolation when False
        **init_idiosyncratic_shocks_plus
    }
)

init_idiosycratic_shocks_no_Rfree = deepcopy(init_idiosyncratic_shocks)
del init_idiosycratic_shocks_no_Rfree["Rfree"]  # get rid of constant interest factor

# Make a dictionary to specify a "kinked R" idiosyncratic shock consumer
init_kinked_R = dict(
    init_idiosycratic_shocks_no_Rfree,
    **{
        "Rboro": 1.20,  # Interest factor on assets when borrowing, a < 0
        "Rsave": 1.02,  # Interest factor on assets when saving, a > 0
        "BoroCnstArt": None,  # kinked R is a bit silly if borrowing not allowed
        "CubicBool": True,  # kinked R is now compatible with linear cFunc and cubic cFunc
        "aXtraCount": 48,  # ...so need lots of extra gridpoints to make up for it
    }
)

# Main calibration characteristics
birth_age = 25
death_age = 90
adjust_infl_to = 1992
# Use income estimates from Cagetti (2003) for High-school graduates
education = "HS"
income_calib = Cagetti_income[education]

# Income specification
income_params = parse_income_spec(
    age_min=birth_age,
    age_max=death_age,
    adjust_infl_to=adjust_infl_to,
    **income_calib,
    SabelhausSong=True
)

# Initial distribution of wealth and permanent income
dist_params = income_wealth_dists_from_scf(
    base_year=adjust_infl_to, age=birth_age, education=education, wave=1995
)

# We need survival probabilities only up to death_age-1, because survival
# probability at death_age is 1.
liv_prb = parse_ssa_life_table(
    female=False, cross_sec=True, year=2004, min_age=birth_age, max_age=death_age
    - 1)

# Parameters related to the number of periods implied by the calibration
time_params = parse_time_params(age_birth=birth_age, age_death=death_age)

# Update all the new parameters
init_lifecycle = deepcopy(init_idiosyncratic_shocks)
init_lifecycle.update(time_params)
init_lifecycle.update(dist_params)
# Note the income specification overrides the pLvlInitMean from the SCF.
init_lifecycle.update(income_params)
init_lifecycle.update({"LivPrb": liv_prb})


# Make a dictionary to specify an infinite consumer with a four period cycle
init_cyclical = copy(init_idiosyncratic_shocks_plus)
init_cyclical['PermGroFac'] = [1.082251, 2.8, 0.3, 1.1]
init_cyclical['permShkStd'] = [0.1, 0.1, 0.1, 0.1]
init_cyclical['tranShkStd'] = [0.1, 0.1, 0.1, 0.1]
init_cyclical['LivPrb'] = 4*[0.98]
init_cyclical['T_cycle'] = 4
