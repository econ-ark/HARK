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
# # Dimensionality Reduction in [Bayer and Luetticke (2018)](https://cepr.org/active/publications/discussion_papers/dp.php?dpno=13071)
#
# [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/econ-ark/HARK/BayerLuetticke?filepath=HARK%2FBayerLuetticke%2FDCT-Copula-Illustration.ipynb)
#
# This companion to the [main notebook](TwoAsset.ipynb) explains in more detail how the authors reduce the dimensionality of their problem
#
# - Based on original slides by Christian Bayer and Ralph Luetticke 
# - Original Jupyter notebook by Seungcheol Lee 
# - Further edits by Chris Carroll, Tao Wang 
#

# %% [markdown]
# ### Preliminaries
#
# In Steady-state Equilibrium (StE) in the model, in any given period, a consumer in state $s$ (which comprises liquid assets $m$, illiquid assets $k$, and human capital $\newcommand{hLev}{h}\hLev$) has two key choices:
# 1. To adjust ('a') or not adjust ('n') their holdings of illiquid assets $k$
# 1. Contingent on that choice, decide the level of consumption, yielding consumption functions:
#     * $c_n(s)$ - nonadjusters
#     * $c_a(s)$ - adjusters
#
# The usual envelope theorem applies here, so marginal value wrt the liquid asset equals marginal utility with respect to consumption:
# $[\frac{d v}{d m} = \frac{d u}{d c}]$.
# In practice, the authors solve their problem using the marginal value of money $\texttt{Vm} = dv/dm$, but because the marginal utility function is invertible it is trivial to recover $\texttt{c}$ from $(u^{\prime})^{-1}(\texttt{Vm} )$.  The consumption function is therefore computed from the $\texttt{Vm}$ function

# %% {"code_folding": [0]}
# Setup stuff

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
my_file_path = os.path.dirname(os.path.abspath("TwoAsset.ipynb"))

# Relative directory for pickled code
code_dir = os.path.join(my_file_path, "../Assets/Two") 

sys.path.insert(0, code_dir)
sys.path.insert(0, my_file_path)

# %% {"code_folding": []}
# Load precalculated Stationary Equilibrium (StE) object EX3SS

import pickle
os.chdir(code_dir) # Go to the directory with pickled code

## EX3SS_20.p is the information in the stationary equilibrium 
## (20: the number of illiquid and liquid weath gridpoints)
### The comments above are original, but it seems that there are 30 not 20 points now

EX3SS=pickle.load(open("EX3SS_20.p", "rb"))

# %% [markdown]
# ### Dimensions
#
# The imported StE solution to the problem represents the functions at a set of gridpoints of
#    * liquid assets ($n_m$ points), illiquid assets ($n_k$), and human capital ($n_h$)
#       * In the code these are $\{\texttt{nm,nk,nh}\}$
#
# So even if the grids are fairly sparse for each state variable, the total number of combinations of the idiosyncratic state gridpoints is large: $n = n_m \times n_k \times n_h$.  So, e.g., $\bar{c}$ is a set of size $n$ containing the level of consumption at each possible _combination_ of gridpoints.
#
# In the "real" micro problem, it would almost never happen that a continuous variable like $m$ would end up being exactly equal to one of the prespecified gridpoints. But the functions need to be evaluated at such non-grid points.  This is addressed by linear interpolation.  That is, if, say, the grid had $m_{8} = 40$ and $m_{9} = 50$ then and a consumer ended up with $m = 45$ then the approximation is that $\tilde{c}(45) = 0.5 \bar{c}_{8} + 0.5 \bar{c}_{9}$.
#

# %% {"code_folding": [0]}
# Show dimensions of the consumer's problem (state space)

print('c_n is of dimension: ' + str(EX3SS['mutil_c_n'].shape))
print('c_a is of dimension: ' + str(EX3SS['mutil_c_a'].shape))

print('Vk is of dimension:' + str(EX3SS['Vk'].shape))
print('Vm is of dimension:' + str(EX3SS['Vm'].shape))

print('For convenience, these are all constructed from the same exogenous grids:')
print(str(len(EX3SS['grid']['m']))+' gridpoints for liquid assets;')
print(str(len(EX3SS['grid']['k']))+' gridpoints for illiquid assets;')
print(str(len(EX3SS['grid']['h']))+' gridpoints for individual productivity.')
print('')
print('Therefore, the joint distribution is of size: ')
print(str(EX3SS['mpar']['nm'])+
      ' * '+str(EX3SS['mpar']['nk'])+
      ' * '+str(EX3SS['mpar']['nh'])+
      ' = '+ str(EX3SS['mpar']['nm']*EX3SS['mpar']['nk']*EX3SS['mpar']['nh']))


# %% [markdown]
# ### Dimension Reduction
#
# The authors use different dimensionality reduction methods for the consumer's problem and the distribution across idiosyncratic states

# %% [markdown]
# #### Representing the consumer's problem with Basis Functions
#
# The idea is to find an efficient "compressed" representation of our functions (e.g., the consumption function), which BL do using tools originally developed for image compression.  The analogy to image compression is that nearby pixels are likely to have identical or very similar colors, so we need only to find an efficient way to represent how the colors _change_ from one pixel to nearby ones.  Similarly, consumption at a given point $s_{i}$ is likely to be close to consumption point at another point $s_{j}$ that is "close" in the state space (similar wealth, income, etc), so a function that captures that similarity efficiently can preserve most of the information without keeping all of the points.
#
# Like linear interpolation, the [DCT transformation](https://en.wikipedia.org/wiki/Discrete_cosine_transform) is a method of representing a continuous function using a finite set of numbers. It uses a set of independent [basis functions](https://en.wikipedia.org/wiki/Basis_function) to do this.
#
# But it turns out that some of those basis functions are much more important than others in representing the steady-state functions. Dimension reduction is accomplished by basically ignoring all basis functions that make "small enough" contributions to the representation of the function.  
#
# ##### When might this go wrong?
#
# Suppose the consumption function changes in a recession in ways that change behavior radically at some states.  Like, suppose unemployment almost never happens in steady state, but it can happen in temporary recessions.  Suppose further that, even for employed people, in a recession, _worries_ about unemployment cause many of them to prudently withdraw some of their illiquid assets -- behavior opposite of what people in the same state would be doing during expansions.  In that case, the basis functions that represented the steady state function would have had no incentive to be able to represent well the part of the space that is never seen in steady state, so any functions that might help do so might well have been dropped in the dimension reduction stage.
#
# On the whole, it seems unlikely that this kind of thing is a major problem, because the vast majority of the variation that people experience is idiosyncratic.  There is always unemployment, for example; it just moves up and down a bit with aggregate shocks, but since the experience of unemployment is in fact well represented in the steady state the method should have no trouble capturing it.
#
# Where the method might have more trouble is in representing economies in which there are multiple equilibria in which behavior is quite different.

# %% [markdown]
# #### For the distribution of agents across states: Copula
#
# The other tool the authors use is the ["copula"](https://en.wikipedia.org/wiki/Copula_(probability_theory)), which allows us to represent the distribution of people across idiosyncratic states efficiently
#
# The copula is computed from the joint distribution of states in StE and will be used to transform the [marginal distributions](https://en.wikipedia.org/wiki/Marginal_distribution) back to joint distributions.  (For an illustration of how the assumptions used when modeling asset price distributions using copulas can fail see [Salmon](https://www.wired.com/2009/02/wp-quant/))
#
#    * A copula is a representation of the joint distribution expressed using a mapping between the uniform joint CDF and the marginal distributions of the variables
#    
#    * The crucial assumption is that what aggregate shocks do is to squeeze or distort the steady state distribution, but leave the rank structure of the distribution the same
#       * An example of when this might not hold is the following.  Suppose that in expansions, the people at the top of the distribution of illiquid assets (the top 1 percent, say) are also at the top 1 percent of liquid assets. But in recessions the bottom 99 percent get angry at the top 1 percent of illiquid asset holders and confiscate part of their liquid assets (the illiquid assets can't be confiscated quickly because they are illiquid). Now the people in the top 99 percent of illiquid assets might be in the _bottom_ 1 percent of liquid assets.
#    
# - In this case we just need to represent how the mapping from ranks into levels of assets
#
# - This reduces the number of points for which we need to track transitions from $3600 = 30 \times 30 \times 4$ to $64 = 30+30+4$.  Or the total number of points we need to contemplate goes from $3600^2 \approx 13 $million to $64^2=4096$.  

# %% {"code_folding": [0]}
# Get some specs about the copula, which is precomputed in the EX3SS object

print('The copula consists of two parts: gridpoints and values at those gridpoints:'+ \
      '\n gridpoints have dimensionality of '+str(EX3SS['Copula']['grid'].shape) + \
      '\n where the first element is total number of gridpoints' + \
      '\n and the second element is number of idiosyncratic state variables' + \
      '\n whose values also are of dimension of '+str(EX3SS['Copula']['value'].shape[0]) + \
      '\n each entry of which is the probability that all three of the'
      '\n state variables are below the corresponding point.')


# %% {"code_folding": []}
## Import necessary libraries

#import sys 
#sys.path.insert(0,'../')

#import numpy as np
#from numpy.linalg import matrix_rank
#import scipy as sc
#from scipy.stats import norm 
#from scipy.interpolate import interp1d, interp2d, griddata, RegularGridInterpolator, interpn
#import multiprocessing as mp
#from multiprocessing import Pool, cpu_count, Process
#from math import ceil
#import math as mt
#from scipy import sparse as sp  # used to work with sparse matrices
#from math import log, cos, pi, sqrt
#from SharedFunc3 import Transition, ExTransitions, GenWeight, MakeGridkm, Tauchen, Fastroot
#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
#import scipy.io #scipy input and output

#from __future__ import print_function
import time
import scipy.fftpack as sf  # scipy discrete fourier transforms
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from matplotlib import lines
import seaborn as sns
import copy as cp
from scipy import linalg   #linear algebra 

from HARK.BayerLuetticke.Assets.Two.FluctuationsTwoAsset import FluctuationsTwoAsset


# %% {"code_folding": [0]}
## State reduction and discrete cosine transformation

class StateReduc_Dct:
    
    def __init__(self, par, mpar, grid, Output, targets, Vm, Vk, 
                 joint_distr, Copula, c_n_guess, c_a_guess, psi_guess,
                 m_n_star, m_a_star, cap_a_star, mutil_c_n, mutil_c_a,mutil_c, P_H):
         
        self.par = par         # Parameters of the theoretical model
        self.mpar = mpar       # Parameters of the numerical representation
        self.grid = grid       # Discrete grid
        self.Output = Output   # Results of the calculations
        self.targets = targets # Like, debt-to-GDP ratio or other desiderata
        self.Vm = Vm           # Marginal value from liquid cash-on-hand
        self.Vk = Vk           # Marginal value of capital
        self.joint_distr = joint_distr # Multidimensional histogram
        self.Copula = Copula   # Encodes rank marginal correlation of joint distribution
        self.mutil_c = mutil_c # Marginal utility of consumption
        self.P_H = P_H         # Transition matrix for macro states (not including distribution)
        
        
    def StateReduc(self):
        """
        input
        -----
        self: dict, stored results from a StE 
        
        output
        ------
        Newly generated
        ===============
        X_ss: ndarray, stacked states, including  
        Y_ss:  ndarray, controls 
        Gamma_state: ndarray, marginal distributions of individual states 
        grid: ndarray, discrete grids
        targets: ndarray, debt-to-GDP ratio or other desiderata
        P_H: transition probability of
        indexMUdct: ndarray, indices selected after dct operation on marginal utility of consumption
        indexVKdct: ndarray, indices selected after dct operation on marginal value of capital
        State: ndarray, dimension equal to reduced states
        State_m: ndarray, dimension equal to reduced states
        Contr:  ndarray, dimension equal to reduced controls
        Contr_m: ndarray, dimension equal to reduced controls
        
        Passed down from the input
        ==========================
        Copula: dict, grids and values
        joint_distr: ndarray, nk x nm x nh
        Output: dict, outputs from the model 
        par: dict, parameters of the theoretical model
        mpar:dict, parameters of the numerical representation
        aggrshock: string, type of aggregate shock used to purturb the StE 
        """
        
        # Inverse of CRRA on x for utility and marginal utility
        invutil = lambda x : ((1-self.par['xi'])*x)**(1./(1-self.par['xi'])) 
        invmutil = lambda x : (1./x)**(1./self.par['xi'])                    
            
        # X=States
        # Marg dist of liquid assets summing over pty and illiquid assets k
        Xss=np.asmatrix(np.concatenate((np.sum(np.sum(self.joint_distr.copy(),axis=1),axis =1),  
                       np.transpose(np.sum(np.sum(self.joint_distr.copy(),axis=0),axis=1)),# marg dist k
                       np.sum(np.sum(self.joint_distr.copy(),axis=1),axis=0), # marg dist pty (\approx income)
                       [np.log(self.par['RB'])],[ 0.]))).T # Given the constant interest rate
        
        # Y="controls" (according to this literature's odd terminology)
        # c = invmarg(marg(c)), so first bit gets consumption policy function
        Yss=np.asmatrix(np.concatenate((invmutil(self.mutil_c.copy().flatten(order = 'F')),\
                                        invmutil(self.Vk.copy().flatten(order = 'F')),
                      [np.log(self.par['Q'])], # Question: Price of the illiquid asset, right?
                                        [ np.log(self.par['PI'])], # Inflation
                                        [ np.log(self.Output)],    
                      [np.log(self.par['G'])], # Gov spending
                                        [np.log(self.par['W'])], # Wage
                                        [np.log(self.par['R'])], # Nominal R
                                        [np.log(self.par['PROFITS'])], 
                      [np.log(self.par['N'])], # Hours worked
                                        [np.log(self.targets['T'])], # Taxes
                                        [np.log(self.grid['K'])],    # Kapital
                      [np.log(self.targets['B'])]))).T # Government debt
        
        # Mapping for Histogram
        # Gamma_state matrix reduced set of states
        #   nm = number of gridpoints for liquid assets
        #   nk = number of gridpoints for illiquid assets
        #   nh = number of gridpoints for human capital (pty)
        Gamma_state = np.zeros( # Create zero matrix of size [nm + nk + nh,nm + nk + nh - 4]
            (self.mpar['nm']+self.mpar['nk']+self.mpar['nh'],
             self.mpar['nm']+self.mpar['nk']+self.mpar['nh'] - 4)) 
            # Question: Why 4? 4 = 3+1, 3: sum to 1 for m, k, h and 1: for entrepreneurs 

        # Impose adding-up conditions: 
        # In each of the block matrices, probabilities must add to 1
        
        for j in range(self.mpar['nm']-1): # np.squeeze reduces one-dimensional matrix to vector
            Gamma_state[0:self.mpar['nm'],j] = -np.squeeze(Xss[0:self.mpar['nm']])
            Gamma_state[j,j]=1. - Xss[j]   #   
            Gamma_state[j,j]=Gamma_state[j,j] - np.sum(Gamma_state[0:self.mpar['nm'],j])
        bb = self.mpar['nm'] # Question: bb='bottom base'? because bb shorter to type than self.mpar['nm'] everywhere

        for j in range(self.mpar['nk']-1):
            Gamma_state[bb+np.arange(0,self.mpar['nk'],1), bb+j-1] = -np.squeeze(Xss[bb+np.arange(0,self.mpar['nk'],1)])
            Gamma_state[bb+j,bb-1+j] = 1. - Xss[bb+j] 
            Gamma_state[bb+j,bb-1+j] = (Gamma_state[bb+j,bb-1+j] - 
                                        np.sum(Gamma_state[bb+np.arange(0,self.mpar['nk']),bb-1+j]))
        bb = self.mpar['nm'] + self.mpar['nk']

        for j in range(self.mpar['nh']-2): 
            # Question: Why -2?  1 for h sum to 1 and 1 for entrepreneur  Some other symmetry/adding-up condition.
            Gamma_state[bb+np.arange(0,self.mpar['nh']-1,1), bb+j-2] = -np.squeeze(Xss[bb+np.arange(0,self.mpar['nh']-1,1)])
            Gamma_state[bb+j,bb-2+j] = 1. - Xss[bb+j]
            Gamma_state[bb+j,bb-2+j] = Gamma_state[bb+j,bb-2+j] - np.sum(Gamma_state[bb+np.arange(0,self.mpar['nh']-1,1),bb-2+j])

        # Number of other state variables not including the gridded -- here, just the interest rate 
        self.mpar['os'] = len(Xss) - (self.mpar['nm']+self.mpar['nk']+self.mpar['nh'])
        # For each gridpoint there are two "regular" controls: consumption and illiquid saving
        # Counts the number of "other" controls (PROFITS, Q, etc)
        self.mpar['oc'] = len(Yss) - 2*(self.mpar['nm']*self.mpar['nk']*self.mpar['nh'])
        
        aggrshock = self.par['aggrshock']
        accuracy = self.par['accuracy']
       
        # Do the dct on the steady state marginal utility
        # Returns an array of indices for the used basis vectors
        indexMUdct = self.do_dct(invmutil(self.mutil_c.copy().flatten(order='F')),
                                 self.mpar,accuracy)

        # Do the dct on the steady state marginal value of capital
        # Returns an array of indices for the used basis vectors
        indexVKdct = self.do_dct(invmutil(self.Vk.copy()),self.mpar,accuracy)
                
        # Calculate the numbers of states and controls
        aux = np.shape(Gamma_state)
        self.mpar['numstates'] = np.int64(aux[1] + self.mpar['os'])
        self.mpar['numcontrols'] = np.int64(len(indexMUdct) + 
                                            len(indexVKdct) + 
                                            self.mpar['oc'])
        
        # Size of the reduced matrices to be used in the Fsys
        # Set to zero because in steady state they are zero
        State = np.zeros((self.mpar['numstates'],1))
        State_m = State
        Contr = np.zeros((self.mpar['numcontrols'],1))
        Contr_m = Contr
        
        return {'Xss': Xss, 'Yss':Yss, 'Gamma_state': Gamma_state, 
                'par':self.par, 'mpar':self.mpar, 'aggrshock':aggrshock,
                'Copula':self.Copula,'grid':self.grid,'targets':self.targets,'P_H':self.P_H, 
                'joint_distr': self.joint_distr, 'Output': self.Output, 'indexMUdct':indexMUdct, 'indexVKdct':indexVKdct,
                'State':State, 'State_m':State_m, 'Contr':Contr, 'Contr_m':Contr_m}

    # Discrete cosine transformation magic happens here
    # sf is scipy.fftpack tool
    def do_dct(self, obj, mpar, level):
        """
        input
        -----
        obj: ndarray nm x nk x nh  
             dimension of states before dct 
        mpar: dict
            parameters in the numerical representaion of the model, e.g. nm, nk and nh
        level: float 
               accuracy level for dct 
        output
        ------
        index_reduced: ndarray n_dct x 1 
                       an array of indices that select the needed grids after dct
                   
        """
        obj = np.reshape(obj.copy(),(mpar['nm'],mpar['nk'],mpar['nh']),order='F')
        X1 = sf.dct(obj,norm='ortho',axis=0)    # dct is operated along three dimensions axis=0/1/2
        X2 = sf.dct(X1.copy(),norm='ortho',axis=1)
        X3 = sf.dct(X2.copy(),norm='ortho',axis=2)

        # Pick the coefficients that are big
        XX = X3.flatten(order='F') 
        ind = np.argsort(abs(XX.copy()))[::-1]
        #  i will 
        i = 1    
        # Sort from smallest (=best) to biggest (=worst)
        # and count how many are 'good enough to keep'
        while linalg.norm(XX[ind[:i]].copy())/linalg.norm(XX) < level:
              i += 1    
        
        needed = i # Question:Isn't this counting the ones that are NOT needed?
        
        index_reduced = np.sort(ind[:i]) # Retrieve the good 
        
        return index_reduced

# %% {"code_folding": [0]}
## Choose an aggregate shock to perturb(one of three shocks: MP, TFP, Uncertainty)

EX3SS['par']['aggrshock']           = 'MP'
EX3SS['par']['rhoS']    = 0.0      # Persistence of variance
EX3SS['par']['sigmaS']  = 0.001    # STD of variance shocks

#EX3SS['par']['aggrshock']           = 'TFP'
#EX3SS['par']['rhoS']    = 0.95
#EX3SS['par']['sigmaS']  = 0.0075
    
#EX3SS['par']['aggrshock']           = 'Uncertainty'
#EX3SS['par']['rhoS']    = 0.84    # Persistence of variance
#EX3SS['par']['sigmaS']  = 0.54    # STD of variance shocks

# %% {"code_folding": [0]}
## Choose an accuracy of approximation with DCT
### Determines number of basis functions chosen -- enough to match this accuracy
### EX3SS is precomputed steady-state pulled in above
EX3SS['par']['accuracy'] = 0.99999 

# %% {"code_folding": []}
## Implement state reduction and DCT
### Do state reduction on steady state
EX3SR=FluctuationsTwoAsset(**EX3SS)   # Takes StE result as input and get ready to invoke state reduction operation
SR=EX3SR.StateReduc()           # StateReduc is operated 

# %% {"code_folding": [0]}
# Measuring the effectiveness of the state reduction

print('What are the results from the state reduction?')
#print('Newly added attributes after the operation include \n'+str(set(SR.keys())-set(EX3SS.keys())))

print('\n')

print('To achieve an accuracy of '+str(EX3SS['par']['accuracy'])+'\n') 

print('The dimension of the policy functions is reduced to '+str(SR['indexMUdct'].shape[0]) \
      +' from '+str(EX3SS['mpar']['nm']*EX3SS['mpar']['nk']*EX3SS['mpar']['nh'])
      )
print('The dimension of the marginal value functions is reduced to '+str(SR['indexVKdct'].shape[0]) \
      + ' from ' + str(EX3SS['Vk'].shape))
print('The total number of control variables is '+str(SR['Contr'].shape[0])+'='+str(SR['indexMUdct'].shape[0]) + \
      '+'+str(SR['indexVKdct'].shape[0])+'+ # of other macro controls')
print('\n')
print('The copula represents the joint distribution with a vector of size '+str(SR['Gamma_state'].shape) )
print('The dimension of states including exogenous state, is ' +str(SR['Xss'].shape[0]))

print('It simply stacks all grids of different\
      \n state variables regardless of their joint distributions.\
      \n This is due to the assumption that the rank order remains the same.')
print('The total number of state variables is '+str(SR['State'].shape[0]) + '='+\
     str(SR['Gamma_state'].shape[1])+'+ the number of macro states (like the interest rate)')


# %% [markdown]
# ### Graphical Illustration
#
# #### Policy/value functions
#
# Taking the consumption function as an example, we plot consumption by adjusters and non-adjusters over a range of $k$ and $m$ that encompasses 100 as well 90 percent of the mass of the distribution function,respectively.  
#
# We plot the functions for the each of the 4 values of the wage $h$.
#

# %% {"code_folding": [0]}
## Graphical illustration

xi = EX3SS['par']['xi']
invmutil = lambda x : (1./x)**(1./xi)  

### convert marginal utilities back to consumption function
mut_StE  =  EX3SS['mutil_c']
mut_n_StE = EX3SS['mutil_c_n']    # marginal utility of non-adjusters
mut_a_StE = EX3SS['mutil_c_a']   # marginal utility of adjusters 

c_StE = invmutil(mut_StE)
cn_StE = invmutil(mut_n_StE)
ca_StE = invmutil(mut_a_StE)


### grid values 
dim_StE = mut_StE.shape
mgrid = EX3SS['grid']['m']
kgrid = EX3SS['grid']['k']
hgrid = EX3SS['grid']['h']


# %% {"code_folding": [0]}
## Define some functions to be used next

def dct3d(x):
    x0=sf.dct(x.copy(),axis=0,norm='ortho')
    x1=sf.dct(x0.copy(),axis=1,norm='ortho')
    x2=sf.dct(x1.copy(),axis=2,norm='ortho')
    return x2

def idct3d(x):
    x2 = sf.idct(x.copy(),axis=2,norm='ortho')
    x1 = sf.idct(x2.copy(),axis=1,norm='ortho')
    x0 = sf.idct(x1.copy(),axis=0,norm='ortho') 
    return x0

def DCTApprox(fullgrids,dct_index):
    dim=fullgrids.shape
    dctcoefs = dct3d(fullgrids)
    dctcoefs_rdc = np.zeros(dim)
    dctcoefs_rdc[dct_index]=dctcoefs[dct_index]
    approxgrids = idct3d(dctcoefs_rdc)
    return approxgrids

# %% [markdown]
# Depending on the accuracy level, the DCT operation choses the necessary number of basis functions used to approximate consumption function at the full grids. This is illustrated in the p31-p34 in this [slides](https://www.dropbox.com/s/46fdxh0aphazm71/presentation_method.pdf?dl=0). We show this for both 1-dimensional (m or k) or 2-dimenstional grids (m and k) in the following. 

# %% {"code_folding": []}
## 2D graph of consumption function: c(m) fixing k and h


## list of accuracy levels  
Accuracy_BL    = 0.99999 # From BL
Accuracy_Less0 = 0.999
Accuracy_Less1 = 0.99
Accuracy_Less2 = 0.95

acc_lst = np.array([Accuracy_BL,Accuracy_Less0,Accuracy_Less1,Accuracy_Less2])

## c(m) fixing k and h
fig = plt.figure(figsize=(8,8))
fig.suptitle('c at full grids and c approximated by DCT in different accuracy levels' 
             '\n non-adjusters, fixing k and h',
             fontsize=(13))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

for idx in range(len(acc_lst)):
    EX3SS_cp =cp.deepcopy(EX3SS) 
    EX3SS_cp['par']['accuracy'] = acc_lst[idx]
    EX3SR_cp=FluctuationsTwoAsset(**EX3SS_cp)   # Takes StE result as input and get ready to invoke state reduction operation
    SR_cp=EX3SR_cp.StateReduc()
    mut_rdc_idx_flt_cp = SR_cp['indexMUdct']
    mut_rdc_idx_cp = np.unravel_index(mut_rdc_idx_flt_cp,dim_StE,order='F')
    nb_bf_cp = len(mut_rdc_idx_cp[0])
    print(str(nb_bf_cp) +" basis functions used.")
    c_n_approx_cp = DCTApprox(cn_StE,mut_rdc_idx_cp)
    c_a_approx_cp = DCTApprox(ca_StE,mut_rdc_idx_cp)
    cn_diff_cp = c_n_approx_cp-cn_StE
    
    # choose the fix grid of h and k
    hgrid_fix=2  # fix level of h as an example 
    kgrid_fix=10  # fix level of k as an example
    
    # get the corresponding c function approximated by dct
    cVec = c_a_approx_cp[:,kgrid_fix,hgrid_fix]
    
    ## plots 
    ax = fig.add_subplot(2,2,idx+1)
    ax.plot(mgrid,cVec,label='c approximated by DCT')
    ax.plot(mgrid,ca_StE[:,kgrid_fix,hgrid_fix],'--',label='c at full grids')
    ax.plot(mgrid,cVec,'r*')
    ax.set_xlabel('m',fontsize=13)
    ax.set_ylabel(r'$c(m)$',fontsize=13)
    ax.set_title(r'accuracy=${}$'.format(acc_lst[idx]))
    ax.legend(loc=0)

# %% {"code_folding": []}
## 2D graph of consumption function: c(k) fixing m and h

fig = plt.figure(figsize=(8,8))
fig.suptitle('c at full grids and c approximated by DCT in different accuracy levels' 
             '\n non-adjusters, fixing m and h',
             fontsize=(13))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

for idx in range(len(acc_lst)):
    EX3SS_cp =cp.deepcopy(EX3SS)
    EX3SS_cp['par']['accuracy'] = acc_lst[idx]
    EX3SR_cp=FluctuationsTwoAsset(**EX3SS_cp)   # Takes StE result as input and get ready to invoke state reduction operation
    SR_cp=EX3SR_cp.StateReduc()
    mut_rdc_idx_flt_cp= SR_cp['indexMUdct']
    mut_rdc_idx_cp = np.unravel_index(mut_rdc_idx_flt_cp,dim_StE,order='F')
    nb_bf_cp = len(mut_rdc_idx_cp[0])
    print(str(nb_bf_cp) +" basis functions used.")
    c_n_approx_cp = DCTApprox(cn_StE,mut_rdc_idx_cp)
    c_a_approx_cp = DCTApprox(ca_StE,mut_rdc_idx_cp)
    cn_diff_cp = c_n_approx_cp-cn_StE
    
    # choose the fix grid of h and m 
    hgrid_fix=2  # fix level of h as an example 
    mgrid_fix=10  # fix level of k as an example
    
    # get the corresponding c function approximated by dct
    cVec = c_n_approx_cp[mgrid_fix,:,hgrid_fix]

    ## plots 
    ax = fig.add_subplot(2,2,idx+1)
    ax.plot(kgrid,cVec,label='c approximated by DCT')
    ax.plot(kgrid,cn_StE[mgrid_fix,:,hgrid_fix],'--',label='c at full grids')
    ax.plot(kgrid,cVec,'r*')
    ax.set_xlabel('k',fontsize=13)
    ax.set_ylabel(r'$c(k)$',fontsize=13)
    ax.set_title(r'accuracy=${}$'.format(acc_lst[idx]))
    ax.legend(loc=0)

# %%
## Set the population density for plotting graphs 

print('Input: plot the graph for bottom x (0-1) of the distribution.')
mass_pct = float(input())

print('Input:choose the accuracy level for DCT, i.e. 0.99999 in the basline of Bayer and Luetticke')
Accuracy_BS = float(input()) ## baseline accuracy level 

# %% {"code_folding": [0]}
# Restore the solution corresponding to the original BL accuracy

EX3SS['par']['accuracy'] = Accuracy_BS
EX3SR=StateReduc_Dct(**EX3SS)   # Takes StE result as input and get ready to invoke state reduction operation
SR=EX3SR.StateReduc()           # StateReduc is operated 

## meshgrids for plots

mmgrid,kkgrid = np.meshgrid(mgrid,kgrid)

## indexMUdct is one dimension, needs to be unraveled to 3 dimensions
mut_rdc_idx_flt = SR['indexMUdct']
mut_rdc_idx = np.unravel_index(mut_rdc_idx_flt,dim_StE,order='F')

## Note: the following chunk of codes can be used to recover the indices of grids selected by DCT. not used here.
#nb_dct = len(mut_StE.flatten()) 
#mut_rdc_bool = np.zeros(nb_dct)     # boolean array of 30 x 30 x 4  
#for i in range(nb_dct):
#    mut_rdc_bool[i]=i in list(SR['indexMUdct'])
#mut_rdc_bool_3d = (mut_rdc_bool==1).reshape(dim_StE)
#mut_rdc_mask_3d = (mut_rdc_bool).reshape(dim_StE)

## For BL accuracy level, get dct compressed c functions at all grids 

c_n_approx = DCTApprox(cn_StE,mut_rdc_idx)
c_a_approx = DCTApprox(ca_StE,mut_rdc_idx)


# Get the joint distribution calculated elsewhere

joint_distr =  EX3SS['joint_distr']


# %% {"code_folding": [0]}
## Functions used to plot consumption functions at the trimmed grids

def WhereToTrim2d(joint_distr,mass_pct):
    """
    parameters
    -----------
    marginal1: marginal pdf in the 1st dimension
    marginal2: marginal pdf in the 2nd dimension
    mass_pct: bottom percentile to keep 
    
    returns
    ----------
    trim1_idx: idx for trimming in the 1s dimension
    trim2_idx: idx for trimming in the 1s dimension
    """
    
    marginal1 = joint_distr.sum(axis=0)
    marginal2 = joint_distr.sum(axis=1)
    ## this can handle cases where the joint_distr itself is a marginal distr from 3d, 
    ##   i.e. marginal.cumsum().max() =\= 1 
    trim1_idx = (np.abs(marginal1.cumsum()-mass_pct*marginal1.cumsum().max())).argmin() 
    trim2_idx = (np.abs(marginal2.cumsum()-mass_pct*marginal2.cumsum().max())).argmin()
    return trim1_idx,trim2_idx

def TrimMesh2d(grids1,grids2,trim1_idx,trim2_idx,drop=True):
    if drop ==True:
        grids_trim1 = grids1.copy()
        grids_trim2 = grids2.copy()
        grids_trim1=grids_trim1[:trim1_idx]
        grids_trim2=grids_trim2[:trim2_idx]
        grids1_trimmesh, grids2_trimmesh = np.meshgrid(grids_trim1,grids_trim2)
    else:
        grids_trim1 = grids1.copy()
        grids_trim2 = grids2.copy()
        grids_trim1[trim1_idx:]=np.nan
        grids_trim2[trim2_idx:]=np.nan
        grids1_trimmesh, grids2_trimmesh = np.meshgrid(grids_trim1,grids_trim2)
        
    return grids1_trimmesh,grids2_trimmesh


# %% {"code_folding": []}
## Other configurations for plotting 

distr_min = 0
distr_max = np.nanmax(joint_distr)
fontsize_lg = 13 

# %% {"code_folding": []}
# For non-adjusters: 3D surface plots of consumption function at full grids and approximated by DCT
##    at all grids and grids after dct first for non-adjusters and then for adjusters


fig = plt.figure(figsize=(14,14))
fig.suptitle('Consumption of non-adjusters at grid points of m and k \n where ' +str(int(mass_pct*100))+ ' % of the agents are distributed \n (for each h)',
             fontsize=(fontsize_lg))
for hgrid_id in range(EX3SS['mpar']['nh']):
    
    ## get the grids and distr for fixed h
    hgrid_fix = hgrid_id    
    distr_fix = joint_distr[:,:,hgrid_fix]
    c_n_approx_fix = c_n_approx[:,:,hgrid_fix]
    c_n_StE_fix = cn_StE[:,:,hgrid_fix]
    
    ## additions to the above cell
    ## for each h grid, take the 90% mass of m and k as the maximum of the m and k axis 
    mk_marginal = joint_distr[:,:,hgrid_fix]
    mmax_idx, kmax_idx = WhereToTrim2d(mk_marginal,mass_pct)
    mmax, kmax = mgrid[mmax_idx],kgrid[kmax_idx]
    mmgrid_trim,kkgrid_trim = TrimMesh2d(mgrid,kgrid,mmax_idx,kmax_idx)
    
    c_n_approx_trim = c_n_approx_fix.copy()
    c_n_approx_trim = c_n_approx_trim[:kmax_idx:,:mmax_idx]  # the dimension is transposed for meshgrid.
    distr_fix_trim = distr_fix.copy()

    cn_StE_trim = c_n_StE_fix.copy()
    cn_StE_trim = cn_StE_trim[:kmax_idx,:mmax_idx]  
    distr_fix_trim = distr_fix_trim[:kmax_idx,:mmax_idx]
    
    ## find the maximum z 
    zmax = np.nanmax(c_n_approx_trim)
    
    ## plots 
    ax = fig.add_subplot(2,2,hgrid_id+1, projection='3d')
    scatter = ax.scatter(mmgrid_trim,kkgrid_trim,cn_StE_trim,
               marker='v',
               color='red')
    surface = ax.plot_surface(mmgrid_trim,kkgrid_trim,c_n_approx_trim,
                    cmap='Blues')
    fake2Dline = lines.Line2D([0],[0], 
                              linestyle="none", 
                              c='b',
                              marker='o') # fake line for making the legend for surface
    
    ax.contourf(mmgrid_trim,kkgrid_trim,distr_fix_trim, 
                zdir='z',
                offset=np.min(distr_fix_trim),
                cmap=cm.YlOrRd,
                vmin=distr_min, 
                vmax=distr_max)
    fake2Dline2 = lines.Line2D([0],[0], 
                              linestyle="none", 
                              c='orange',
                              marker='o') # fakeline for making the legend for surface
    
    ax.set_xlabel('m',fontsize=fontsize_lg)
    ax.set_ylabel('k',fontsize=fontsize_lg)
    ax.set_zlabel(r'$c_n(m,k)$',fontsize=fontsize_lg)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    ax.set_zlim([0,zmax])
    ax.set_title(r'$h({})$'.format(hgrid_fix))
    ax.view_init(20, 70)
    ax.legend([scatter,fake2Dline,fake2Dline2], 
              ['Full-grid c','Approximated c','Joint distribution'],
              loc=0)


# %% {"code_folding": []}
# For adjusters: 3D surface plots of consumption function at full grids and approximated by DCT 

    
fig = plt.figure(figsize=(14,14))
fig.suptitle('Consumption of adjusters at grid points of m and k \n where ' +str(int(mass_pct*100))+ '% of agents are distributed  \n (for each h)',
             fontsize=(fontsize_lg))
for hgrid_id in range(EX3SS['mpar']['nh']):
    
    ## get the grids and distr for fixed h
    hgrid_fix=hgrid_id
    c_a_StE_fix = ca_StE[:,:,hgrid_fix]
    c_a_approx_fix = c_a_approx[:,:,hgrid_fix]
    distr_fix = joint_distr[:,:,hgrid_fix]
    
    ## additions to the above cell
    ## for each h grid, take the 90% mass of m and k as the maximum of the m and k axis 
    mk_marginal = joint_distr[:,:,hgrid_fix]
    mmax_idx, kmax_idx = WhereToTrim2d(mk_marginal,mass_pct)
    mmax, kmax = mgrid[mmax_idx],kgrid[kmax_idx]
    mmgrid_trim,kkgrid_trim = TrimMesh2d(mgrid,kgrid,mmax_idx,kmax_idx)
    c_a_approx_trim =c_a_approx_fix.copy()
    c_a_approx_trim  = c_a_approx_trim[:kmax_idx,:mmax_idx]
    distr_fix_trim = distr_fix.copy()
    ca_StE_trim =c_a_StE_fix.copy()
    ca_StE_trim = ca_StE_trim[:kmax_idx,:mmax_idx]    
    distr_fix_trim = distr_fix_trim[:kmax_idx,:mmax_idx]

    
    # get the maximum z
    zmax = np.nanmax(c_a_approx_trim)
    
    ## plots 
    ax = fig.add_subplot(2,2,hgrid_id+1, projection='3d')
    ax.scatter(mmgrid_trim,kkgrid_trim,ca_StE_trim,marker='v',color='red',
                    label='full-grid c:adjuster')
    ax.plot_surface(mmgrid_trim,kkgrid_trim,c_a_approx_trim,cmap='Blues',
               label='approximated c: adjuster')
    fake2Dline = lines.Line2D([0],[0], 
                              linestyle="none", 
                              c='b',
                              marker='o') # fake line for making the legend for surface
    ax.contourf(mmgrid_trim,kkgrid_trim,distr_fix_trim, 
                zdir='z',
                offset=np.min(distr_fix_trim),
                cmap=cm.YlOrRd,
                vmin=distr_min,
                vmax=distr_max)
    fake2Dline2 = lines.Line2D([0],[0], 
                              linestyle="none", 
                              c='orange',
                              marker='o') # fakeline for making the legend for surface
    ax.set_xlabel('m',fontsize=fontsize_lg)
    ax.set_ylabel('k',fontsize=fontsize_lg)
    ax.set_zlabel(r'$c_a(m,k)$',fontsize=fontsize_lg)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    #ax.set_xlim([0,mmax])
    #ax.set_ylim([0,kmax])
    ax.set_zlim([0,zmax])
    ax.set_title(r'$h({})$'.format(hgrid_fix))
    ax.view_init(20, 70)
    ax.legend([scatter,fake2Dline,fake2Dline2], 
              ['Full-grid c','Approx c','Joint distribution'],
              loc=0)

# %% {"code_folding": []}
## 3D scatter plots of the difference of full-grid c and approximated c for non-adjusters

fig = plt.figure(figsize=(14,14))
fig.suptitle('Approximation errors of non-adjusters at grid points of m and k \n where ' +str(int(mass_pct*100))+ '% of agents are distributed \n (for each h)',
             fontsize=(fontsize_lg))
for hgrid_id in range(EX3SS['mpar']['nh']):
    
    ## get the grids and distr for fixed h
    hgrid_fix = hgrid_id    
    cn_diff = c_n_approx-cn_StE
    cn_diff_fix = cn_diff[:,:,hgrid_fix]
    distr_fix = joint_distr[:,:,hgrid_fix]


    ## additions to the above cell
    ## for each h grid, take the 90% mass of m and k as the maximum of the m and k axis 
    mk_marginal = joint_distr[:,:,hgrid_fix]
    mmax_idx, kmax_idx = WhereToTrim2d(mk_marginal,mass_pct)
    mmax, kmax = mgrid[mmax_idx],kgrid[kmax_idx]
    mmgrid_trim,kkgrid_trim = TrimMesh2d(mgrid,kgrid,mmax_idx,kmax_idx)
    c_n_diff_trim = cn_diff_fix.copy()
    c_n_diff_trim = c_n_diff_trim[:kmax_idx,:mmax_idx]  # first k and then m because c is is nk x nm 
    distr_fix_trim = distr_fix.copy()
    distr_fix_trim = distr_fix_trim[:kmax_idx,:mmax_idx]


    ## plots 
    ax = fig.add_subplot(2,2,hgrid_id+1, projection='3d')
    
    ax.plot_surface(mmgrid_trim,kkgrid_trim,c_n_diff_trim, 
                    rstride=1, 
                    cstride=1,
                    cmap=cm.coolwarm, 
                    edgecolor='none')
    fake2Dline_pos = lines.Line2D([0],[0], 
                              linestyle="none", 
                              c='r',
                              marker='o') # fakeline for making the legend for surface
    fake2Dline_neg = lines.Line2D([0],[0], 
                              linestyle="none", 
                              c='b',
                              marker='o') # fakeline for making the legend for surface
    ax.contourf(mmgrid_trim,kkgrid_trim,distr_fix_trim,
                zdir='z',
                offset=np.min(c_n_diff_trim),
                cmap=cm.YlOrRd,
                vmin=distr_min,
                vmax=distr_max)
    fake2Dline2 = lines.Line2D([0],[0], 
                              linestyle="none", 
                              c='orange',
                              marker='o') # fakeline for making the legend for contour
    ax.set_xlabel('m',fontsize=fontsize_lg)
    ax.set_ylabel('k',fontsize=fontsize_lg)
    ax.set_zlabel(r'$c_a(m,k)$',fontsize=fontsize_lg)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    #ax.set_xlim([0,mmax])
    #ax.set_ylim([0,kmax])
    ax.set_title(r'$h({})$'.format(hgrid_fix))
    ax.view_init(20, 40)
    ax.legend([fake2Dline_pos,fake2Dline_neg,fake2Dline2], 
              ['Positive approx errors','Negative approx errors','Joint distribution'],
              loc=0)

# %% {"code_folding": []}
# Difference of full-grid c and DCT compressed c for each level of accuracy


fig = plt.figure(figsize=(14,14))
fig.suptitle('Approximation errors in different levels of accuracy \n where ' +str(int(mass_pct*100))+ '% of agents are distributed \n (non-adjusters)',
             fontsize=(fontsize_lg))

for idx in range(len(acc_lst)):
    EX3SS_cp =cp.deepcopy(EX3SS)
    EX3SS_cp['par']['accuracy'] = acc_lst[idx]
    EX3SR_cp=FluctuationsTwoAsset(**EX3SS_cp)   # Takes StE result as input and get ready to invoke state reduction operation
    SR_cp=EX3SR_cp.StateReduc()
    mut_rdc_idx_flt_cp = SR_cp['indexMUdct']
    mut_rdc_idx_cp = np.unravel_index(mut_rdc_idx_flt_cp,dim_StE,order='F')
    nb_bf_cp = len(mut_rdc_idx_cp[0])
    print(str(nb_bf_cp) +" basis functions used.")
    c_n_approx_cp = DCTApprox(cn_StE,mut_rdc_idx_cp)
    cn_diff_cp = c_n_approx_cp-cn_StE
    
    hgrid_fix=1  # fix level of h as an example 
    c_n_diff_cp_fix = cn_diff_cp[:,:,hgrid_fix]
    distr_fix = joint_distr[:,:,hgrid_fix]
    
    ## for each h grid, take the 90% mass of m and k as the maximum of the m and k axis 
    mk_marginal = joint_distr[:,:,hgrid_fix]
    mmax_idx, kmax_idx = WhereToTrim2d(mk_marginal,mass_pct)
    mmax, kmax = mgrid[mmax_idx],kgrid[kmax_idx]
    mmgrid_trim,kkgrid_trim = TrimMesh2d(mgrid,kgrid,mmax_idx,kmax_idx)
    c_n_diff_cp_trim = c_n_diff_cp_fix.copy()
    c_n_diff_cp_trim = c_n_diff_cp_trim[:kmax_idx:,:mmax_idx]
    distr_fix_trim = distr_fix.copy()
    distr_fix_trim = distr_fix_trim[:kmax_idx,:mmax_idx]
    
    ## plots 
    ax = fig.add_subplot(2,2,idx+1, projection='3d')
    ax.plot_surface(mmgrid_trim,kkgrid_trim,c_n_diff_cp_trim, 
                    rstride=1, 
                    cstride=1,
                    cmap=cm.coolwarm, 
                    edgecolor='none')
    fake2Dline_pos = lines.Line2D([0],[0], 
                              linestyle="none", 
                              c='r',
                              marker='o') # fakeline for making the legend for surface
    fake2Dline_neg = lines.Line2D([0],[0], 
                              linestyle="none", 
                              c='b',
                              marker='o') # fakeline for making the legend for surface
    dst_contour = ax.contourf(mmgrid_trim,kkgrid_trim,distr_fix_trim, 
                              zdir='z',
                              offset=np.min(-2),
                              cmap=cm.YlOrRd,
                              vmin=distr_min, 
                              vmax=distr_max)
    fake2Dline2 = lines.Line2D([0],[0], 
                              linestyle="none", 
                              c='orange',
                              marker='o') # fakeline for making the legend for contour
    ax.set_xlabel('m',fontsize=13)
    ax.set_ylabel('k',fontsize=13)
    ax.set_zlabel('Difference of c functions',fontsize=13)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    #ax.set_xlim([0,mmax])
    #ax.set_ylim([0,kmax])
    ax.set_zlim([-2,2])  # these are magic numbers. need to fix
    ax.set_title(r'accuracy=${}$'.format(acc_lst[idx]))
    ax.view_init(10, 60)
    ax.legend([fake2Dline_pos,fake2Dline_neg,fake2Dline2], 
              ['+ approx errors','- approx errors','Joint distribution'],
              loc=0)

# %% {"code_folding": []}
# Difference of full-grid c and DCT compressed c for difference levels of accuracy

fig = plt.figure(figsize=(14,14))
fig.suptitle('Differences of approximation errors between adjusters/non-adjusters \n where ' +str(int(mass_pct*100))+ '% of agents are distributed \n in different accuracy levels',
             fontsize=(fontsize_lg))

for idx in range(len(acc_lst)):
    EX3SS_cp =cp.deepcopy(EX3SS)
    EX3SS_cp['par']['accuracy'] = acc_lst[idx]
    EX3SR_cp=FluctuationsTwoAsset(**EX3SS_cp)   # Takes StE result as input and get ready to invoke state reduction operation
    SR_cp=EX3SR_cp.StateReduc()
    mut_rdc_idx_flt_cp = SR_cp['indexMUdct']
    mut_rdc_idx_cp = np.unravel_index(mut_rdc_idx_flt_cp,dim_StE,order='F')
    nb_bf_cp = len(mut_rdc_idx_cp[0])
    print(str(nb_bf_cp) +" basis functions used.")
    c_n_approx_cp = DCTApprox(cn_StE,mut_rdc_idx_cp)
    c_a_approx_cp = DCTApprox(ca_StE,mut_rdc_idx_cp)
    cn_diff_cp = c_n_approx_cp-cn_StE
    ca_diff_cp = c_a_approx_cp-ca_StE
    c_diff_cp_apx_error = ca_diff_cp - cn_diff_cp
    
    hgrid_fix=1  # fix level of h as an example 
    c_diff_cp_apx_error_fix = c_diff_cp_apx_error[:,:,hgrid_fix]
    distr_fix = joint_distr[:,:,hgrid_fix]


    ## additions to the above cell
    ## for each h grid, take the 90% mass of m and k as the maximum of the m and k axis 
    mk_marginal = joint_distr[:,:,hgrid_fix]
    mmax_idx, kmax_idx = WhereToTrim2d(mk_marginal,mass_pct)
    mmax, kmax = mgrid[mmax_idx],kgrid[kmax_idx]
    mmgrid_trim,kkgrid_trim = TrimMesh2d(mgrid,kgrid,mmax_idx,kmax_idx)
    c_diff_cp_apx_error_trim = c_diff_cp_apx_error_fix.copy()
    c_diff_cp_apx_error_trim = c_diff_cp_apx_error_trim[:kmax_idx,:mmax_idx]
    distr_fix_trim = distr_fix.copy()
    distr_fix_trim = distr_fix_trim[:kmax_idx,:mmax_idx]
    
    ## get the scale 
    zmin = np.nanmin(c_diff_cp_apx_error)
    zmax = np.nanmax(c_diff_cp_apx_error)
    
    ## plots 
    ax = fig.add_subplot(2,2,idx+1, projection='3d')
    ax.plot_surface(mmgrid_trim,kkgrid_trim,c_diff_cp_apx_error_trim, 
                    rstride=1, 
                    cstride=1,
                    cmap=cm.coolwarm, 
                    edgecolor='none',
                    label='Difference of full-grid and approximated consumption functions')
    fake2Dline_pos = lines.Line2D([0],[0], 
                              linestyle="none", 
                              c='r',
                              marker='o') # fakeline for making the legend for surface
    fake2Dline_neg = lines.Line2D([0],[0], 
                              linestyle="none", 
                              c='b',
                              marker='o') # fakeline for making the legend for surface
    ax.contourf(mmgrid_trim,kkgrid_trim,distr_fix_trim,
                zdir='z',
                offset=np.min(-0.2),
                cmap=cm.YlOrRd,
                vmin=distr_min, 
                vmax=distr_max)
    fake2Dline2 = lines.Line2D([0],[0], 
                              linestyle="none", 
                              c='orange',
                              marker='o') # fakeline for making the legend for contour
    ax.set_xlabel('m',fontsize=fontsize_lg)
    ax.set_ylabel('k',fontsize=fontsize_lg)
    ax.set_zlabel('Difference of approximation errors',fontsize=fontsize_lg)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    #ax.set_xlim([0,mmax])
    #ax.set_ylim([0,kmax])
    ax.set_zlim([-0.2,0.2]) # these are magic numbers. need to fix
    ax.set_title(r'accuracy=${}$'.format(acc_lst[idx]))
    ax.view_init(10, 60)
    ax.legend([fake2Dline_pos,fake2Dline_neg,fake2Dline2],
              ['+ diff','- diff','Joint distribution'],
              loc=0)


# %% [markdown]
# ##### Observation
#
# - For a given grid value of productivity, the remaining grid points after DCT to represent the whole consumption function are concentrated in low values of $k$ and $m$. This is because the slopes of the surfaces of marginal utility are changing the most in these regions.  For larger values of $k$ and $m$ the functions become smooth and only slightly concave, so they can be represented by many fewer points
# - For different grid values of productivity (2 sub plots), the numbers of grid points in the DCT operation differ. From the lowest to highest values of productivity, there are 78, 33, 25 and 18 grid points, respectively. They add up to the total number of gridpoints of 154 after DCT operation, as we noted above for marginal utility function. 

# %% [markdown]
# #### Distribution of states 
#
# - We first plot the distribution of $k$ fixing $m$ and $h$. Next, we plot the joint distribution of $m$ and $k$ only fixing $h$ in 3-dimenstional space.  
# - The joint-distribution can be represented by marginal distributions of $m$, $k$ and $h$ and a copula that describes the correlation between the three states. The former is straightfoward. We plot the copula only. The copula is essentially a multivariate cummulative distribution function where each marginal is uniform. (Translation from the uniform to the appropriate nonuniform distribution is handled at a separate stage).
#

# %% {"code_folding": []}
### Marginalize along h grids

joint_distr =  EX3SS['joint_distr']
joint_distr_km = EX3SS['joint_distr'].sum(axis=2)

### Plot distributions in 2 dimensional graph 

fig = plt.figure(figsize=(10,10))
plt.suptitle('Marginal distribution of k at different m \n(for each h)')

for hgrid_id in range(EX3SS['mpar']['nh']):
    ax = plt.subplot(2,2,hgrid_id+1)
    ax.set_title(r'$h({})$'.format(hgrid_id))
    ax.set_xlabel('k',size=fontsize_lg)
    for id in range(EX3SS['mpar']['nm']):   
        ax.plot(kgrid,joint_distr[id,:,hgrid_id])

# %% {"code_folding": [0]}
## Plot joint distribution of k and m in 3d graph
#for only 90 percent of the distributions 

fig = plt.figure(figsize=(14,14))
fig.suptitle('Joint distribution of m and k \n where ' +str(int(mass_pct*100))+ '% agents are distributed \n(for each h)',
             fontsize=(fontsize_lg))

for hgrid_id in range(EX3SS['mpar']['nh']):
    
    ## get the distr for fixed h
    hgrid_fix = hgrid_id  
    joint_km = joint_distr[:,:,hgrid_fix]
    
    ## additions to the above cell
    ## for each h grid, take the 90% mass of m and k as the maximum of the m and k axis 
    mk_marginal = joint_distr[:,:,hgrid_fix]
    mmax_idx, kmax_idx = WhereToTrim2d(mk_marginal,mass_pct)
    mmax, kmax = mgrid[mmax_idx],kgrid[kmax_idx]
    mmgrid_trim,kkgrid_trim = TrimMesh2d(mgrid,kgrid,mmax_idx,kmax_idx)
    joint_km_trim = joint_km.copy()
    joint_km_trim  = joint_km_trim[:kmax_idx,:mmax_idx]
    
    # get the maximum z
    zmax = np.nanmax(joint_distr)
    
    ## plots 
    ax = fig.add_subplot(2,2,hgrid_id+1, projection='3d')
    ax.plot_surface(mmgrid_trim,kkgrid_trim,joint_km_trim, 
                    rstride=1, 
                    cstride=1,
                    cmap=cm.YlOrRd, 
                    edgecolor='none',
                    vmin=distr_min, 
                    vmax=distr_max)
    fake2Dline = lines.Line2D([0],[0], 
                              linestyle="none", 
                              c='orange',
                              marker='o') # fakeline for making the legend for contour
    ax.set_xlabel('m',fontsize=fontsize_lg)
    ax.set_ylabel('k',fontsize=fontsize_lg)
    ax.set_zlabel('Probability',fontsize=fontsize_lg)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    ax.set_title(r'$h({})$'.format(hgrid_id))
    ax.set_zlim([0,zmax])
    #ax.set_xlim([0,mmax])
    #ax.set_ylim([0,kmax])
    ax.view_init(20, 60)
    ax.legend([fake2Dline], 
              ['joint distribution'],
              loc=0)

# %% [markdown]
# Notice the CDFs in StE copula have 4 modes, corresponding to the number of $h$ gridpoints. Each of the four parts of the cdf is a joint-distribution of $m$ and $k$.  It can be presented in 3-dimensional graph as below.  

# %% {"code_folding": [0]}
## Plot the copula 
# same plot as above for only 90 percent of the distributions 


cdf=EX3SS['Copula']['value'].reshape(4,30,30)   # important: 4,30,30 not 30,30,4? 

fig = plt.figure(figsize=(14,14))
fig.suptitle('Copula of m and k \n where ' +str(int(mass_pct*100))+ '% agents are distributed \n(for each h)',
             fontsize=(fontsize_lg))
for hgrid_id in range(EX3SS['mpar']['nh']):
    
    hgrid_fix = hgrid_id    
    cdf_fix  = cdf[hgrid_fix,:,:]
    
    ## additions to the above cell
    ## for each h grid, take the 90% mass of m and k as the maximum of the m and k axis 
    mk_marginal = joint_distr[:,:,hgrid_fix]
    mmax_idx, kmax_idx = WhereToTrim2d(mk_marginal,mass_pct)
    mmax, kmax = mgrid[mmax_idx],kgrid[kmax_idx]
    mmgrid_trim,kkgrid_trim = TrimMesh2d(mgrid,kgrid,mmax_idx,kmax_idx)
    cdf_fix_trim = cdf_fix.copy()
    cdf_fix_trim  = cdf_fix_trim[:kmax_idx,:mmax_idx]
    
    ## plots 
    ax = fig.add_subplot(2,2,hgrid_id+1, projection='3d')
    ax.plot_surface(mmgrid_trim,kkgrid_trim,cdf_fix_trim, 
                    rstride=1, 
                    cstride=1,
                    cmap =cm.Greens, 
                    edgecolor='None')
    fake2Dline = lines.Line2D([0],[0], 
                              linestyle="none", 
                              c='green',
                              marker='o')
    ax.set_xlabel('m',fontsize=fontsize_lg)
    ax.set_ylabel('k',fontsize=fontsize_lg)
    ax.set_title(r'$h({})$'.format(hgrid_id))
    
    ## for each h grid, take the 95% mass of m and k as the maximum of the m and k axis 
    
    marginal_mk = joint_distr[:,:,hgrid_id]
    marginal_m = marginal_mk.sum(axis=0)
    marginal_k = marginal_mk.sum(axis=1)
    mmax = mgrid[(np.abs(marginal_m.cumsum()-mass_pct*marginal_m.cumsum().max())).argmin()]
    kmax = kgrid[(np.abs(marginal_k.cumsum()-mass_pct*marginal_k.cumsum().max())).argmin()]
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    #ax.set_xlim(0,mmax)
    #ax.set_ylim(0,kmax)
    ax.view_init(30, 60)
    ax.legend([fake2Dline], 
              ['Marginal cdf of the copula'],
              loc=0)

# %% [markdown]
# ## More to do:
#
# 1. Figure out median value of h and normalize c, m, and k by it

# %% [markdown]
# Given the assumption that the copula remains the same after aggregate risk is introduced, we can use the same copula and the marginal distributions to recover the full joint-distribution of the states.  

# %% [markdown]
# ### Summary: what do we achieve after the transformation?
#
# - Using the DCT, the dimension of the policy and value functions are reduced from 3600 to 154 and 94, respectively.
# - By marginalizing the joint distribution with the fixed copula assumption, the marginal distribution is of dimension 64 compared to its joint distribution of a dimension of 3600.
#
#
#
