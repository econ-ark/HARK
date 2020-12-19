import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from HARK.distribution import Uniform
from importlib import reload
figs_dir = '../../Figures/'

# Import configurable parameters, and keep them updated through reload
import parameter_config
reload(parameter_config)
from parameter_config import *

###############################################################################

# Size of simulations
AgentCountTotal = 1000000 # Total simulated population
T_sim = 13              # Number of quarters to simulate in counterfactuals

# Basic lifecycle length parameters (don't touch these)
init_age = 24
working_T = 41*4        # Number of working periods
retired_T = 55*4        # Number of retired periods
T_cycle = working_T + retired_T

# Define the distribution of the discount factor for each eduation level
DiscFacCount = 7
DiscFacDstnD = Uniform(DiscFacMeanD-DiscFacSpread, DiscFacMeanD+DiscFacSpread).approx(DiscFacCount)
DiscFacDstnH = Uniform(DiscFacMeanH-DiscFacSpread, DiscFacMeanH+DiscFacSpread).approx(DiscFacCount)
DiscFacDstnC = Uniform(DiscFacMeanC-DiscFacSpread, DiscFacMeanC+DiscFacSpread).approx(DiscFacCount)
DiscFacDstns = [DiscFacDstnD, DiscFacDstnH, DiscFacDstnC]

# Define permanent income growth rates for each education level (from Cagetti 2003)
PermGroRte_d_ann = [5.2522391e-002,  5.0039782e-002,  4.7586132e-002,  4.5162424e-002,  4.2769638e-002,  4.0408757e-002,  3.8080763e-002,  3.5786635e-002,  3.3527358e-002,  3.1303911e-002,  2.9117277e-002,  2.6968437e-002,  2.4858374e-002, 2.2788068e-002,  2.0758501e-002,  1.8770655e-002,  1.6825511e-002,  1.4924052e-002,  1.3067258e-002,  1.1256112e-002, 9.4915947e-003,  7.7746883e-003,  6.1063742e-003,  4.4876340e-003,  2.9194495e-003,  1.4028022e-003, -6.1326258e-005, -1.4719542e-003, -2.8280999e-003, -4.1287819e-003, -5.3730185e-003, -6.5598280e-003, -7.6882288e-003, -8.7572392e-003, -9.7658777e-003, -1.0713163e-002, -1.1598112e-002, -1.2419745e-002, -1.3177079e-002, -1.3869133e-002, -4.3985368e-001, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003]
PermGroRte_h_ann = [4.1102173e-002,  4.1194381e-002,  4.1117402e-002,  4.0878307e-002,  4.0484168e-002,  3.9942056e-002,  3.9259042e-002,  3.8442198e-002,  3.7498596e-002,  3.6435308e-002,  3.5259403e-002,  3.3977955e-002,  3.2598035e-002,  3.1126713e-002,  2.9571062e-002,  2.7938153e-002,  2.6235058e-002,  2.4468848e-002,  2.2646594e-002,  2.0775369e-002,  1.8862243e-002,  1.6914288e-002,  1.4938576e-002,  1.2942178e-002,  1.0932165e-002,  8.9156095e-003,  6.8995825e-003,  4.8911556e-003,  2.8974003e-003,  9.2538802e-004, -1.0178097e-003, -2.9251214e-003, -4.7894755e-003, -6.6038005e-003, -8.3610250e-003, -1.0054077e-002, -1.1675886e-002, -1.3219380e-002, -1.4677487e-002, -1.6043137e-002, -5.5864350e-001, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002]
PermGroRte_c_ann = [3.9375106e-002,  3.9030288e-002,  3.8601230e-002,  3.8091011e-002,  3.7502710e-002,  3.6839406e-002,  3.6104179e-002,  3.5300107e-002,  3.4430270e-002,  3.3497746e-002,  3.2505614e-002,  3.1456953e-002,  3.0354843e-002,  2.9202363e-002,  2.8002591e-002,  2.6758606e-002,  2.5473489e-002,  2.4150316e-002,  2.2792168e-002,  2.1402124e-002,  1.9983263e-002,  1.8538663e-002,  1.7071404e-002,  1.5584565e-002,  1.4081224e-002,  1.2564462e-002,  1.1037356e-002,  9.5029859e-003,  7.9644308e-003,  6.4247695e-003,  4.8870812e-003,  3.3544449e-003,  1.8299396e-003,  3.1664424e-004, -1.1823620e-003, -2.6640003e-003, -4.1251914e-003, -5.5628564e-003, -6.9739162e-003, -8.3552918e-003, -6.8938447e-001, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004]
PermGroRte_d_ann += 31*[PermGroRte_d_ann[-1]] # Add 31 years of the same permanent income growth rate to the end of the sequence
PermGroRte_h_ann += 31*[PermGroRte_h_ann[-1]]
PermGroRte_c_ann += 31*[PermGroRte_c_ann[-1]]
PermGroRte_d_retire = PermGroRte_d_ann[40]     # Store the big shock to permanent income at retirement
PermGroRte_h_retire = PermGroRte_h_ann[40]
PermGroRte_c_retire = PermGroRte_c_ann[40]
PermGroRte_d_ann[40] = PermGroRte_d_ann[39]   # Overwrite the "retirement drop" with the adjacent growth rate
PermGroRte_h_ann[40] = PermGroRte_h_ann[39]
PermGroRte_c_ann[40] = PermGroRte_c_ann[39]
PermGroFac_d = []
PermGroFac_h = []
PermGroFac_c = []
for j in range(len(PermGroRte_d_ann)):         # Make sequences of quarterly permanent income growth factors from annual permanent income growth rates
    PermGroFac_d += 4*[(1 + PermGroRte_d_ann[j])**0.25]
    PermGroFac_h += 4*[(1 + PermGroRte_h_ann[j])**0.25]
    PermGroFac_c += 4*[(1 + PermGroRte_c_ann[j])**0.25]
PermGroFac_d[working_T-1] = 1 + PermGroRte_d_retire  # Put the big shock at retirement back into the sequence
PermGroFac_h[working_T-1] = 1 + PermGroRte_h_retire
PermGroFac_c[working_T-1] = 1 + PermGroRte_c_retire
SkillRot = 0.00125
for t in range(T_cycle):
    PermGroFac_d[t] = PermGroFac_d[t]*np.ones(6)
    PermGroFac_d[t][1:3] = PermGroFac_d[t][0] - SkillRot
    PermGroFac_d[t][3:6] = PermGroFac_d[t][0] - SkillRot
    PermGroFac_h[t] = PermGroFac_h[t]*np.ones(6)
    PermGroFac_h[t][1:3] = PermGroFac_h[t][0] - SkillRot
    PermGroFac_h[t][3:6] = PermGroFac_h[t][0] - SkillRot
    PermGroFac_c[t] = PermGroFac_c[t]*np.ones(6)
    PermGroFac_c[t][1:3] = PermGroFac_c[t][0] - SkillRot
    PermGroFac_c[t][3:6] = PermGroFac_c[t][0] - SkillRot
PermGroFac_d_small = [PermGroFac_d[t][:2] for t in range(T_cycle)]
PermGroFac_h_small = [PermGroFac_h[t][:2] for t in range(T_cycle)]
PermGroFac_c_small = [PermGroFac_c[t][:2] for t in range(T_cycle)]

# Define the paths of permanent and transitory shocks (from Sabelhaus and Song)
TranShkStd = (np.concatenate((np.linspace(0.1,0.12,17), 0.12*np.ones(17), np.linspace(0.12,0.075,61), np.linspace(0.074,0.007,68), np.zeros(retired_T+1)))*4)**0.5
TranShkStd = np.ndarray.tolist(TranShkStd)
PermShkStd = np.concatenate((((0.00011342*(np.linspace(24,64.75,working_T-1)-47)**2 + 0.01)/(11.0/4.0))**0.5,np.zeros(retired_T+1)))
PermShkStd[123:162] = PermShkStd[122] # Don't extrapolate
PermShkStd = np.ndarray.tolist(PermShkStd)

# Import survival probabilities from SSA data
data_location = os.path.dirname(os.path.abspath(__file__))
f = open(data_location + '/' + 'USactuarial.txt','r')
actuarial_reader = csv.reader(f,delimiter='\t')
raw_actuarial = list(actuarial_reader)
base_death_probs = []
for j in range(len(raw_actuarial)):
    base_death_probs += [float(raw_actuarial[j][4])] # This effectively assumes that everyone is a white woman
f.close()

# Import adjustments for education and apply them to the base mortality rates
f = open(data_location + '/' + 'EducMortAdj.txt','r')
adjustment_reader = csv.reader(f,delimiter=' ')
raw_adjustments = list(adjustment_reader)
d_death_probs = []
h_death_probs = []
c_death_probs = []
for j in range(76):
    d_death_probs += [base_death_probs[j + init_age]*float(raw_adjustments[j][1])]
    h_death_probs += [base_death_probs[j + init_age]*float(raw_adjustments[j][2])]
    c_death_probs += [base_death_probs[j + init_age]*float(raw_adjustments[j][3])]
for j in range(76,96):
    d_death_probs += [base_death_probs[j + init_age]*float(raw_adjustments[75][1])]
    h_death_probs += [base_death_probs[j + init_age]*float(raw_adjustments[75][2])]
    c_death_probs += [base_death_probs[j + init_age]*float(raw_adjustments[75][3])]
LivPrb_d = []
LivPrb_h = []
LivPrb_c = []
for j in range(len(d_death_probs)): # Convert annual mortality rates to quarterly survival rates
    LivPrb_d += 4*[(1 - d_death_probs[j])**0.25]
    LivPrb_h += 4*[(1 - h_death_probs[j])**0.25]
    LivPrb_c += 4*[(1 - c_death_probs[j])**0.25]
LivPrb_d_small = [LivPrb_d[t]*np.ones(2) for t in range(T_cycle)]
LivPrb_h_small = [LivPrb_h[t]*np.ones(2) for t in range(T_cycle)]
LivPrb_c_small = [LivPrb_c[t]*np.ones(2) for t in range(T_cycle)]
LivPrb_d = [LivPrb_d[t]*np.ones(6) for t in range(T_cycle)]
LivPrb_h = [LivPrb_h[t]*np.ones(6) for t in range(T_cycle)]
LivPrb_c = [LivPrb_c[t]*np.ones(6) for t in range(T_cycle)]


def makeMrkvArray(Urate, Uspell, Dspell, Lspell, Dexit=0):
    '''
    Make an age-varying list of Markov transition matrices given the unemployment
    rate (in normal times), the average length of an unemployment spell, the average
    length of a deep unemployment spell, and the average length of a lockdown.
    Parameters
    ----------
    Urate: float
        Erogodic unemployment rate
    Uspell: float
        Expected length of unemployment spell
    Dspell: float
        Expected length of deep unemployment spell
    Lspell: float
        Expected length of pandemic
    Dexit: float
        How likely to leave deep unemployment when pandemic ends
        0 => No correlation between exiting deep unemployment and pandemic ending
        1 => Certain to exit deep unemployment when pandemic ends
    '''
    U_persist = 1.-1./Uspell
    E_persist = 1.-Urate*(1.-U_persist)/(1.-Urate)
    D_persist = 1.-1./Dspell
    u = U_persist
    e = E_persist
    d = D_persist
    r = 1-1./Lspell
    dr = r*(1-Dexit)
    
    MrkvArray_working = np.array([[e, 1-e, 0.0, 0.0, 0.0, 0.0],
                                  [1-u, u, 0.0, 0.0, 0.0, 0.0],
                                  [0.0, 1-d, d, 0.0, 0.0, 0.0],
                                  [e*(1-r), (1-e)*(1-r), 0.0, e*r, (1-e)*r, 0.0],
                                  [(1-u)*(1-r), u*(1-r), 0.0, (1-u)*r, u*r, 0.0],
                                  [0.0, (1-d)*(1-dr), d*(1-dr), 0.0, (1-d)*dr, d*dr]
                                  ])
    MrkvArray_retired = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                  [(1-r), 0.0, 0.0, r, 0.0, 0.0],
                                  [(1-r), 0.0, 0.0, r, 0.0, 0.0],
                                  [(1-r), 0.0, 0.0, r, 0.0, 0.0]
                                  ])
    MrkvArray = (working_T-1)*[MrkvArray_working] + (retired_T+1)*[MrkvArray_retired]
    return MrkvArray


# Make Markov transition arrays among discrete states in each period of the lifecycle (ACTUAL / SIMULATION)
MrkvArray_real = makeMrkvArray(Urate_normal, Uspell, Dspell_real, Lspell_real)

# Make Markov transition arrays among discrete states in each period of the lifecycle (PERCEIVED / SIMULATION)
MrkvArray_pcvd = makeMrkvArray(Urate_normal, Uspell, Dspell_pcvd, Lspell_pcvd)

# Make a two state Markov array ("small") that is only used when generating the initial distribution of states
U_logit_base = np.log(1./(1.-Urate_normal)-1.)
U_persist = 1.-1./Uspell
E_persist = 1.-Urate_normal*(1.-U_persist)/(1.-Urate_normal)
u = U_persist
e = E_persist
MrkvArray_working_small = np.array([[e, 1-e],
                                    [1-u, u]
                                    ])
MrkvArray_retired_small = np.array([[1., 0.],
                                    [1., 0.]
                                    ])
MrkvArray_small = (working_T-1)*[MrkvArray_working_small] + (retired_T+1)*[MrkvArray_retired_small]

# Define a parameter dictionaries for three education levels
init_dropout = {"cycles" : 1,
                "T_cycle": T_cycle,
                "T_retire": working_T-1,
                'T_sim': T_cycle,
                'T_age': T_cycle+1,
                'AgentCount': 10000,
                "PermGroFacAgg": PermGroFacAgg,
                "PopGroFac": PopGroFac,
                "CRRA": CRRA,
                "DiscFac": 0.98, # This will be overwritten at type construction
                "Rfree_big" : np.array(6*[1.01]),
                "PermGroFac_big": PermGroFac_d,
                "LivPrb_big": LivPrb_d,
                "uPfac_big" : np.array(3*[1.0] + 3*[uPfac_L]),
                "MrkvArray_big" : MrkvArray_pcvd,
                "Rfree" : np.array(2*[1.01]),
                "PermGroFac": PermGroFac_d_small,
                "LivPrb": LivPrb_d_small,
                "uPfac" : np.array(2*[1.0]),
                "MrkvArray" : MrkvArray_small, # Yes, this is intentional
                "MrkvArray_pcvd" : MrkvArray_small, # Yes, this is intentional
                "MrkvArray_sim" : MrkvArray_real,
                "BoroCnstArt": 0.0,
                "PermShkStd": PermShkStd,
                "PermShkCount": PermShkCount,
                "TranShkStd": TranShkStd,
                "TranShkCount": TranShkCount,
                "UnempPrb": 0.0, # Unemployment is modeled as a Markov state
                "UnempPrbRet": UnempPrbRet,
                "IncUnemp": IncUnemp,
                "IncUnempRet": IncUnempRet,
                "aXtraMin": aXtraMin,
                "aXtraMax": aXtraMax,
                "aXtraCount": aXtraCount,
                "aXtraExtra": aXtraExtra,
                "aXtraNestFac": aXtraNestFac,
                "CubicBool": False,
                "vFuncBool": False,
                'aNrmInitMean': np.log(0.00001), # Initial assets are zero
                'aNrmInitStd': 0.0,
                'pLvlInitMean': pLvlInitMeanD,
                'pLvlInitStd': pLvlInitStd,
                "MrkvPrbsInit" : np.array([1-Urate_normal, Urate_normal] + 4*[0.0]),
                'Urate' : Urate_normal,
                'Uspell' : Uspell,
                'L_shared' : L_shared,
                'Lspell_pcvd' : Lspell_pcvd,
                'Lspell_real' : Lspell_real,
                'Dspell_pcvd' : Dspell_pcvd,
                'Dspell_real' : Dspell_real,
                'EducType': 0,
                'UpdatePrb': UpdatePrb,
                'track_vars' : []
                }
if L_shared:
    init_dropout['T_lockdown'] = int(Lspell_real)

adj_highschool = {
        "PermGroFac" : PermGroFac_h_small,
        "LivPrb" : LivPrb_h_small,
        "PermGroFac_big" : PermGroFac_h,
        "LivPrb_big" : LivPrb_h,
        'pLvlInitMean' : pLvlInitMeanH,
        'EducType' : 1}
init_highschool = init_dropout.copy()
init_highschool.update(adj_highschool)

adj_college = {
        "PermGroFac" : PermGroFac_c_small,
        "LivPrb" : LivPrb_c_small,
        "PermGroFac_big" : PermGroFac_c,
        "LivPrb_big" : LivPrb_c,
        'pLvlInitMean' : pLvlInitMeanC,
        'EducType' : 2}
init_college = init_dropout.copy()
init_college.update(adj_college)

# Define a dictionary to represent the baseline scenario
base_dict = {
        'PanShock' : False,
        'StimMax'  : 0.,
        'StimCut0' : None,
        'StimCut1' : None,
        'BonusUnemp' : 0.,
        'BonusDeep'  : 0.,
        'T_ahead'  : 0,
        'UnempD'   : U_logit_base,
        'UnempH'   : U_logit_base,
        'UnempC'   : U_logit_base,
        'UnempP'   : 0.,
        'UnempA1'  : 0.,
        'UnempA2'  : 0.,
        'DeepD'    : -np.inf,
        'DeepH'    : -np.inf,
        'DeepC'    : -np.inf,
        'DeepP'    : 0.,
        'DeepA1'   : 0.,
        'DeepA2'   : 0.,
        'Dspell_pcvd' : Dspell_pcvd, # These five parameters don't do anything in baseline scenario
        'Dspell_real' : Dspell_real,
        'Lspell_pcvd' : Lspell_pcvd,
        'Lspell_real' : Lspell_real,
        'L_shared'    : L_shared
        }

# Define a dictionary to mutate baseline for the pandemic
pandemic_changes = {
        'PanShock' : True,
        'UnempD'   : UnempD + Unemp0,
        'UnempH'   : UnempH + Unemp0,
        'UnempC'   : UnempC + Unemp0,
        'UnempP'   : UnempP,
        'UnempA1'  : UnempA1,
        'UnempA2'  : UnempA2,
        'DeepD'    : DeepD + Deep0,
        'DeepH'    : DeepH + Deep0,
        'DeepC'    : DeepC + Deep0,
        'DeepP'    : DeepP,
        'DeepA1'   : DeepA1,
        'DeepA2'   : DeepA2,
        }

# Define a dictionary to mutate baseline for the fiscal stimulus
stimulus_changes = {
        'StimMax'  : StimMax,
        'StimCut0' : StimCut0,
        'StimCut1' : StimCut1,
        'BonusUnemp' : BonusUnemp,
        'BonusDeep'  : BonusDeep,
        'T_ahead'  : T_ahead,
        }

# Define a dictionary to mutate baseline for a deep unemployment pandemic
deep_pandemic_changes = {
        'PanShock' : True,
        'UnempD'   : UnempD + DeepPanAdj1,
        'UnempH'   : UnempH + DeepPanAdj1,
        'UnempC'   : UnempC + DeepPanAdj1,
        'UnempP'   : UnempP + DeepPanAdj3,
        'UnempA1'  : UnempA1,
        'UnempA2'  : UnempA2,
        'DeepD'    : DeepD + DeepPanAdj2,
        'DeepH'    : DeepH + DeepPanAdj2,
        'DeepC'    : DeepC + DeepPanAdj2,
        'DeepP'    : DeepP,
        'DeepA1'   : DeepA1,
        'DeepA2'   : DeepA2,
        }

