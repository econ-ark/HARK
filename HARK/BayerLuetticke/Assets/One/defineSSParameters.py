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
parrhoH        = 0.979    # Persistence of productivity
parsigmaH      = 0.059    # STD of productivity shocks
mparin         = 0.00046  # Prob. to become entrepreneur
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
parrho_R       = 0.5  # Inertia

# Tax Schedule
partau         = 1.   # Net income after tax 

## Returns
parPI          = 1.          # Gross inflation
parRB          = 1.          # Market clearing interest rate to be solved for
parborrwedge   = 1.0**0.25 - 1. # Wedge on borrowing

## Grids
# Idiosyncratic States
mparnm         = 100 # integer
mparnh         = 4 # integer
mpartauchen    = 'importance'

gridK = 3.2701 # keep fixed at SS of economy with bonds and capital

## Numerical Parameters
mparcrit    = 10**(-11)

# Make a dictionary to specify a HANK model with one asset IOUs
par_one_asset_IOU = { 'beta': parbeta, 'xi': parxi, 'gamma': pargamma,
                     'rhoH': parrhoH, 'sigmaH': parsigmaH, 'eta': pareta, 'mu': parmu,
                     'alpha': paralpha,  'prob_priceadj': parprob_priceadj, 'kappa': parkappa,
                     'theta_pi': partheta_pi, 'rho_R': parrho_R, 'tau': partau, 'PI': parPI,
                     'RB': parRB, 'borrwedge': parborrwedge }
mpar_one_asset_IOU = { 'in': mparin, 'out': mparout, 'nm': mparnm, 'nh': mparnh,
                      'tauchen': mpartauchen, 'crit': mparcrit }
grid_one_asset_IOU = { 'K': gridK }

parm_one_asset_IOU = {'par': par_one_asset_IOU, 'mpar': mpar_one_asset_IOU, 'grid': grid_one_asset_IOU}

