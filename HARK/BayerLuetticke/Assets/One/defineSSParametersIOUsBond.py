# -*- coding: utf-8 -*-
'''
Parameters for one asset IOU
'''

## Parameters
# Household Parameters
parbeta        = 0.99     # Discount factor
parxi          = 2.          # CRRA
pargamma       = 1.          # Inverse Frisch elasticity

# Income Process
parrhoH        = 0.98    # Persistence of productivity
parsigmaH      = 0.06    # STD of productivity shocks
mparin         = 0.00075  # Prob. to become entrepreneur
mparout        = 0.0625   # Prob. to become worker again

# Firm Side Parameters
pareta         = 20.
parmu          = (pareta-1)/pareta       # Markup 5%
paralpha       = (2./3.)/parmu  # Labor share 2/3

# Phillips Curve
parprob_priceadj = 3/4. # average price duration of 4 quarters = 1/(1-par.prob_priceadj)
parkappa         = (1.-parprob_priceadj)*(1.-parprob_priceadj*parbeta)/parprob_priceadj  # Phillips-curve parameter (from Calvo prob.)

# Central Bank Policy
partheta_pi    = 1.5 # Reaction to inflation
parrho_R       = 0.8  # Inertia

# Fiscal Policy
pargamma_b = 0.5
pargamma_pi = 0

# Tax Schedule
partau         = 1   # Net income after tax 
parBtoY        = 0   # Gov debt to output ratio
#partau         = 0.9   # Net income after tax 
#parBtoY        = 0.25   # Gov debt to output ratio
#parBtoY        = 2   # Gov debt to output ratio

## Returns
parPI          = 1.          # Gross inflation
parRB          = 1.          # Market clearing interest rate to be solved for
parborrwedge   = 1.0**0.25 - 1. # Wedge on borrowing

## Grids
# Idiosyncratic States
mparnm         = 500 # integer
mparnh         = 4 # integer
mpartauchen    = 'importance'

gridK = 45 # keep fixed at SS of economy with bonds and capital
gridm_min_art = -100 
#gridm_min_art = 0 # to make the asset as a gov bond
gridm_max_art = 2000 

## Numerical Parameters
mparcrit    = 10**(-11)

# Make a dictionary to specify a HANK model with one asset IOUs
par_one_asset_IOUsBond = { 'beta': parbeta, 'xi': parxi, 'gamma': pargamma,
                     'rhoH': parrhoH, 'sigmaH': parsigmaH, 'eta': pareta, 'mu': parmu,
                     'alpha': paralpha,  'prob_priceadj': parprob_priceadj, 'kappa': parkappa,
                     'theta_pi': partheta_pi, 'rho_R': parrho_R, 'tau': partau, 'PI': parPI,
                     'RB': parRB, 'borrwedge': parborrwedge, 'gamma_b': pargamma_b, 'gamma_pi': pargamma_pi, 'BtoY': parBtoY }
mpar_one_asset_IOUsBond = { 'in': mparin, 'out': mparout, 'nm': mparnm, 'nh': mparnh,
                      'tauchen': mpartauchen, 'crit': mparcrit }
grid_one_asset_IOUsBond = { 'K': gridK, 'm_min': gridm_min_art, 'm_max': gridm_max_art}

parm_one_asset_IOUsBond = {'par': par_one_asset_IOUsBond, 'mpar': mpar_one_asset_IOUsBond, 'grid': grid_one_asset_IOUsBond}

