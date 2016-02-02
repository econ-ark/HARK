'''
This module implements a discrete choice problem in a continuous space, as in
Rust, Iskhakov, Schjerning, and Jorgenson (2015).  It also includes the extension
to "transformed" value functions.
'''
# Import the HARK library.  The assumption is that this code is in a folder
# contained in the HARK folder. 
import sys 
sys.path.insert(0,'../')

import numpy as np
from HARKinterpolation import LinearInterp, Cubic1DInterpDecay
from HARKutilities import CRRAutility_inv, CRRAutility_invP, CRRAutility, CRRAutilityP, CRRAutilityP_inv

identityFunc = lambda Z : Z
onesFunc = lambda Z : np.ones(Z.shape)

class DiscreteChoiceSolution():
    '''
    This class represents the solution to a discrete choice problem.
    '''
    def __init__(self,vFunc,vPfunc):
        self.vFunc=vFunc
        self.vPfunc=vPfunc
        
    def distance(self,other_soln):
        return self.cFunc[0].distance(other_soln.cFunc[0])

def discreteChoiceContinuousStateSolver(solution_tp1, state_grid, sigma_epsilon, transformations):
    '''
    The discrete choice solver for a continuous state space.  Currently only
    works with one continuous state dimension.
    
    Parameters:
    -------------
    solution_tp1:
        An object with the attributes vFunc and vPfunc, each of which is a list
        of functions representing the value functions and marginal value functions
        for each discrete choice available to the agent.
    state_grid: numpy.array
        Gridpoints at which the value function should be calculated.
    sigma_epsilon: float
        Standard deviation of T1EV preference shocks for this choice.
    transformations:
        An object with function pointers for transforming the value functions
        for each choice as well as the output value function.  Attributes:
        
        Qfunc: Function by which value function for each choice is transformed
            when making the discrete choice.
        QfuncP: The first derivative of Qfunc.
        Tfunc: Function by which the output value function is transformed before
            interpolation. Interpolation will be more accurate when the function
            is fairly linear, but value functions are *extremely* curved. Tfunc
            "decurves" the value function before interpolation; the inverse
            instantaneuous utility function is often a good choice.
        TfuncP: The first derivative of Tfunc.
        TfuncInv: The inverse of Tfunc (often the instantaneous utility function).
        Zfunc: Function by which the output marginal value function is transformed
            before interpolation.  The inverse marginal utility function is often
            a good choice.
        ZfuncInv: The inverse of Zfunc.  Often the marginal utility function.
        
    Returns:
    ------------
    solution_t:
        An object with the attributes vFunc and vP func, representing the value
        and marginal value functions before the discrete choice is made.  Also
        has all other attributes of solution_tp1 as a "pass through".
    '''
    # Unpack the transformations for convenient access
    if transformations is None:
        Qfunc = identityFunc
        QfuncP = onesFunc
        Tfunc = identityFunc
        TfuncP = onesFunc
        TfuncInv = identityFunc
        Zfunc = identityFunc
        ZfuncInv = identityFunc
    else:
        Qfunc = transformations.Qfunc
        QfuncP = transformations.QfuncP
        Tfunc = transformations.Tfunc
        TfuncP = transformations.TfuncP
        TfuncInv = transformations.TfuncInv
        Zfunc = transformations.Zfunc
        ZfuncInv = transformations.ZfuncInv
    
    # Unpack the discrete choice value functions
    vFunc_tp1 = solution_tp1.vFunc
    vPfunc_tp1 = solution_tp1.vPfunc
    choice_count = len(vFunc_tp1)
    grid_size = state_grid.size
    
    # Get (marginal) value for each choice at each gridpoint
    v_array = np.zeros((choice_count,grid_size),dtype=float) + np.nan
    vP_array = np.zeros((choice_count,grid_size),dtype=float) + np.nan
    for n in range(choice_count):
        v_array[n,:] = vFunc_tp1[n](state_grid)
        vP_array[n,:] = vPfunc_tp1[n](state_grid)
    these = np.isnan(v_array)
    v_array[these] = 0
    vP_array[these] = 0
        
    # Transform the (marginal) value array
    Qv_array = Qfunc(v_array)
    best_Qv = np.tile(np.amax(Qv_array,axis=0),(choice_count,1))
    Qv_array = Qv_array - best_Qv
    QvP_array = vP_array*QfuncP(v_array)
    Qv_array[these] = -np.inf
    QvP_array[these] = 0
        
    # Calculate the probability of choosing each option at each gridpoint
    exp_Qv_array = np.exp(Qv_array/sigma_epsilon)
    choice_prob_array = exp_Qv_array/np.tile(np.sum(exp_Qv_array,axis=0),(choice_count,1))
        
    # Calculate the marginal probability of choosing each option at each gridpoint
    temp_array = np.tile(np.sum(choice_prob_array*QvP_array,axis=0),(choice_count,1))
    choice_probP_array = choice_prob_array*(QvP_array - temp_array)/sigma_epsilon
    
    # Calculate the (marginal) expected value at each gridpoint
    v_grid = np.sum(choice_prob_array*v_array,axis=0)
    vP_grid = np.sum(choice_probP_array*v_array + choice_prob_array*vP_array,axis=0)
    
    # Construct interpolations of the value grid and marginal value grid
    Tv_grid = Tfunc(v_grid)
    TvP_grid = vP_grid*TfuncP(v_grid)
    TvFunc_t = Cubic1DInterpDecay(state_grid,Tv_grid,TvP_grid,solution_tp1.v_lim_intercept,solution_tp1.v_lim_slope)
    vFunc_t = lambda m : TfuncInv(TvFunc_t(m))
    ZvP_grid = Zfunc(vP_grid)
    ZvPfunc_t = LinearInterp(state_grid,ZvP_grid)
    vPfunc_t = lambda m : ZfuncInv(ZvPfunc_t(m))
    
    # Construct and report the solution (including the pass through attributes)
    solution_t = DiscreteChoiceSolution(vFunc=vFunc_t,vPfunc=vPfunc_t)
    other_attributes = [key for key in solution_tp1.__dict__]
    other_attributes.remove('vFunc')
    other_attributes.remove('vPfunc')
    other_attributes.remove('v_lim_intercept')
    other_attributes.remove('v_lim_slope')
    for name in other_attributes:
        do_string = 'solution_t.' + name + ' = solution_tp1.' + name
        exec(do_string) in locals()
    return solution_t
    
    
class ValueTransformations():
    def __init__(self,Qfunc,QfuncP,Tfunc,TfuncP,TfuncInv,Zfunc,ZfuncInv):
        self.Qfunc = Qfunc
        self.QfuncP = QfuncP
        self.Tfunc = Tfunc
        self.TfuncP = TfuncP
        self.TfuncInv = TfuncInv
        self.Zfunc = Zfunc
        self.ZfuncInv = ZfuncInv

def makeCRRAtransformations(rho,v_shift=0.0,do_Q=False,do_T=True,do_Z=True):
    '''
    Makes constant relative risk aversion value transformation functions for the
    discrete-continuous solver input "transformations".
    
    Parameters:
    -------------
    rho : float
        The coefficient of relative risk aversion.
    v_shift : float
        A level shifter for each value function: v(x) = f(x) + v_shift
    do_Q : boolean
        An indicator for whether the individual choice value functions should
        be transformed by the inverse utility function.
    do_T: boolean
        An indicator for whether the output value function should be transformed
        by the inverse utility function before interpolation.
    do_Z: boolean
        An indicator for whether the output marginal value function should be
        transformed by the inverse marginal utility function before interpolation.
        
    Returns:
    ------------
    transformations : ValueTransformations
        An object with seven attributes: Qfunc, QfuncP, Tfunc, TfuncP, TfuncInv,
        Zfunc, ZfuncInv
    '''
    if do_Q:
        Qfunc = lambda x : CRRAutility_inv(x-v_shift, gam=rho)
        QfuncP = lambda x : CRRAutility_invP(x,gam=rho)
    else:
        Qfunc = identityFunc
        QfuncP = onesFunc
    
    if do_T:
        Tfunc = lambda x : CRRAutility_inv(x-v_shift, gam=rho)
        TfuncP = lambda x : CRRAutility_invP(x,gam=rho)
        TfuncInv = lambda x : CRRAutility(x,gam=rho) + v_shift
    else:
        Tfunc = identityFunc
        TfuncP = onesFunc
        TfuncInv = identityFunc
        
    if do_Z:
        Zfunc = lambda x : CRRAutilityP_inv(x,gam=rho)
        ZfuncInv = lambda x : CRRAutilityP(x,gam=rho)
    else:
        Zfunc = identityFunc
        ZfuncInv = identityFunc
        
    transformations = ValueTransformations(Qfunc,QfuncP,Tfunc,TfuncP,TfuncInv,Zfunc,ZfuncInv)
    return transformations
    
        