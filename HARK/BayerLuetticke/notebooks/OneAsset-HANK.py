# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
#
# # A One Asset HANK Model 
#
# This notebook solves a New Keynesian model in which there is only a single liquid asset.  This is the second model described in <cite data-cite="6202365/ECL3ZAR7"></cite>.  For a detailed description of their solution method, see the companion two-asset HANK model notebook.

# %% {"code_folding": [0]}
# Setup
from __future__ import print_function

# This is a jupytext paired notebook that autogenerates a corresponding .py file
# which can be executed from a terminal command line via "ipython [name].py"
# But a terminal does not permit inline figures, so we need to test jupyter vs terminal
# Google "how can I check if code is executed in the ipython notebook"

def in_ipynb():
    try:
        if str(type(get_ipython())) == "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>":
            return True
        else:
            return False
    except NameError:
        return False

# Determine whether to make the figures inline (for spyder or jupyter)
# vs whatever is the automatic setting that will apply if run from the terminal
if in_ipynb():
    # %matplotlib inline generates a syntax error when run from the shell
    # so do this instead
    get_ipython().run_line_magic('matplotlib', 'inline') 
else:
    get_ipython().run_line_magic('matplotlib', 'auto') 
    
# The tools for navigating the filesystem
import sys
import os

# Find pathname to this file:
my_file_path = os.path.dirname(os.path.abspath("OneAsset-HANK.ipynb"))

# Relative directory for pickled code
code_dir = os.path.join(my_file_path, "../Assets/One/") 

sys.path.insert(0, code_dir)
sys.path.insert(0, my_file_path)

# %% {"code_folding": [0]}
# Ignore system warnings while running the notebook
import warnings
warnings.filterwarnings('ignore')

# Load Stationary equilibrium (StE) object EX2SS

import pickle
os.chdir(code_dir) # Go to the directory with pickled code

## EX2SS.p is the information in the stationary equilibrium (20: the number of illiquid and liquid weath grids )
EX2SS=pickle.load(open("EX2SS.p", "rb"))

# %%
from HARK.BayerLuetticke.Assets.One.FluctuationsOneAssetIOUsBond import FluctuationsOneAssetIOUs, SGU_solver, plot_IRF

# %% {"code_folding": []}
# Uncertainty Shock
    
EX2SS['par']['aggrshock'] = 'Uncertainty'
EX2SS['par']['rhoS'] = 0.84    # Persistence of variance
EX2SS['par']['sigmaS'] = 0.54    # STD of variance shocks

EX2SR=FluctuationsOneAssetIOUs(**EX2SS)
SR=EX2SR.StateReduc()

SGUresult=SGU_solver(SR['Xss'],SR['Yss'],SR['Gamma_state'],SR['Gamma_control'],SR['InvGamma'],SR['Copula'],
                         SR['par'],SR['mpar'],SR['grid'],SR['targets'],SR['P_H'],SR['aggrshock'],SR['oc'])

plot_IRF(SR['mpar'],SR['par'],SGUresult['gx'],SGUresult['hx'],SR['joint_distr'],
             SR['Gamma_state'],SR['grid'],SR['targets'],SR['os'],SR['oc'],SR['Output'])

# %% {"code_folding": [0]}
# Monetary Policy Shock

EX2SS['par']['aggrshock'] = 'MP'
EX2SS['par']['rhoS'] = 0.0      # Persistence of variance
EX2SS['par']['sigmaS'] = 0.001    # STD of variance shocks

EX2SR=FluctuationsOneAssetIOUs(**EX2SS)
SR=EX2SR.StateReduc()

SGUresult=SGU_solver(SR['Xss'],SR['Yss'],SR['Gamma_state'],SR['Gamma_control'],SR['InvGamma'],SR['Copula'],
                         SR['par'],SR['mpar'],SR['grid'],SR['targets'],SR['P_H'],SR['aggrshock'],SR['oc'])

plot_IRF(SR['mpar'],SR['par'],SGUresult['gx'],SGUresult['hx'],SR['joint_distr'],
             SR['Gamma_state'],SR['grid'],SR['targets'],SR['os'],SR['oc'],SR['Output'])

# %%
# Productivity Shock

EX2SS['par']['aggrshock'] = 'TFP'
EX2SS['par']['rhoS'] = 0.95
EX2SS['par']['sigmaS'] = 0.0075

EX2SR=FluctuationsOneAssetIOUs(**EX2SS)
SR=EX2SR.StateReduc()

SGUresult = SGU_solver(SR['Xss'],SR['Yss'],SR['Gamma_state'],SR['Gamma_control'],SR['InvGamma'],SR['Copula'],
                         SR['par'],SR['mpar'],SR['grid'],SR['targets'],SR['P_H'],SR['aggrshock'],SR['oc'])

plot_IRF(SR['mpar'],SR['par'],SGUresult['gx'],SGUresult['hx'],SR['joint_distr'],
             SR['Gamma_state'],SR['grid'],SR['targets'],SR['os'],SR['oc'],SR['Output'])

# %%
