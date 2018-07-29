
# -*- coding: utf-8 -*-
'''
State Reduction, SGU_solver, Plot
'''
import sys 
sys.path.insert(0,'../')

import numpy as np
from numpy.linalg import matrix_rank
import scipy as sc
from scipy.stats import norm 
from scipy.interpolate import interp1d, interp2d, griddata, RegularGridInterpolator, interpn
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from math import ceil
import math as mt
from scipy import sparse as sp
from scipy import linalg
from math import log, cos, pi, sqrt
import time
from SharedFunc3 import Transition, ExTransitions, GenWeight, MakeGridkm, Tauchen, Fastroot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.io
import scipy.fftpack as sf

class FluctuationsTwoAsset:
    

    
    def __init__(self, par, mpar, grid, Output, targets, Vm, Vk, joint_distr, Copula, c_n_guess, c_a_guess, psi_guess, m_n_star, m_a_star, cap_a_star, mutil_c_n, mutil_c_a,mutil_c, P_H):
         
        self.par = par
        self.mpar = mpar
        self.grid = grid
        self.Output = Output
        self.targets = targets
        self.Vm = Vm
        self.Vk = Vk
        self.joint_distr = joint_distr
        self.Copula = Copula
        self.mutil_c = mutil_c
        self.P_H = P_H
        
        
    def StateReduc(self):
        invutil = lambda x : ((1-self.par['xi'])*x)**(1./(1-self.par['xi']))
        invmutil = lambda x : (1./x)**(1./self.par['xi'])
                       
        
        Xss=np.asmatrix(np.concatenate((np.sum(np.sum(self.joint_distr.copy(),axis=1),axis =1),  # marginal distribution liquid asset
                       np.transpose(np.sum(np.sum(self.joint_distr.copy(),axis=0),axis=1)),  # marginal distribution illiquid asset
                       np.sum(np.sum(self.joint_distr.copy(),axis=1),axis=0), # marginal distribution productivity
                       [np.log(self.par['RB'])],[ 0.]))).T
        
        
        Yss=np.asmatrix(np.concatenate((invmutil(self.mutil_c.copy().flatten(order = 'F')),invmutil(self.Vk.copy().flatten(order = 'F')),
                      [np.log(self.par['Q'])],[ np.log(self.par['PI'])],[np.log(self.Output)],
                      [np.log(self.par['G'])],[np.log(self.par['W'])],[np.log(self.par['R'])],[np.log(self.par['PROFITS'])],
                      [np.log(self.par['N'])],[np.log(self.targets['T'])],[np.log(self.grid['K'])],
                      [np.log(self.targets['B'])]))).T
        
        # Mapping for Histogram

        Gamma_state = np.zeros((self.mpar['nm']+self.mpar['nk']+self.mpar['nh'], self.mpar['nm']+self.mpar['nk']+self.mpar['nh'] - 4))
        for j in range(self.mpar['nm']-1):
            Gamma_state[0:self.mpar['nm'],j] = -np.squeeze(Xss[0:self.mpar['nm']])
            Gamma_state[j,j]=1. - Xss[j]
            Gamma_state[j,j]=Gamma_state[j,j] - np.sum(Gamma_state[0:self.mpar['nm'],j])
        bb = self.mpar['nm']

        for j in range(self.mpar['nk']-1):
            Gamma_state[bb+np.arange(0,self.mpar['nk'],1), bb+j-1] = -np.squeeze(Xss[bb+np.arange(0,self.mpar['nk'],1)])
            Gamma_state[bb+j,bb-1+j] = 1. - Xss[bb+j]
            Gamma_state[bb+j,bb-1+j] = Gamma_state[bb+j,bb-1+j] - np.sum(Gamma_state[bb+np.arange(0,self.mpar['nk']),bb-1+j])
        bb = self.mpar['nm'] + self.mpar['nk']

        for j in range(self.mpar['nh']-2):
            Gamma_state[bb+np.arange(0,self.mpar['nh']-1,1), bb+j-2] = -np.squeeze(Xss[bb+np.arange(0,self.mpar['nh']-1,1)])
            Gamma_state[bb+j,bb-2+j] = 1. - Xss[bb+j]
            Gamma_state[bb+j,bb-2+j] = Gamma_state[bb+j,bb-2+j] - np.sum(Gamma_state[bb+np.arange(0,self.mpar['nh']-1,1),bb-2+j])


        self.mpar['os'] = len(Xss) - (self.mpar['nm']+self.mpar['nk']+self.mpar['nh'])
        self.mpar['oc'] = len(Yss) - 2*(self.mpar['nm']*self.mpar['nk']*self.mpar['nh'])
        
        aggrshock = self.par['aggrshock']
        accuracy = self.par['accuracy']
       
        indexMUdct = self.do_dct(invmutil(self.mutil_c.copy().flatten(order='F')),self.mpar,accuracy)
               
        indexVKdct = self.do_dct(invmutil(self.Vk.copy()),self.mpar,accuracy)
                
        aux = np.shape(Gamma_state)
        self.mpar['numstates'] = np.int64(aux[1] + self.mpar['os'])
        self.mpar['numcontrols'] = np.int64(len(indexMUdct) + len(indexVKdct) + self.mpar['oc'])
        
        State = np.zeros((self.mpar['numstates'],1))
        State_m = State
        Contr = np.zeros((self.mpar['numcontrols'],1))
        Contr_m = Contr
        
        return {'Xss': Xss, 'Yss':Yss, 'Gamma_state': Gamma_state, 
                'par':self.par, 'mpar':self.mpar, 'aggrshock':aggrshock,
                'Copula':self.Copula,'grid':self.grid,'targets':self.targets,'P_H':self.P_H, 
                'joint_distr': self.joint_distr, 'Output': self.Output, 'indexMUdct':indexMUdct, 'indexVKdct':indexVKdct,
                'State':State, 'State_m':State_m, 'Contr':Contr, 'Contr_m':Contr_m}

    def do_dct(self, obj, mpar, level):

        obj = np.reshape(obj.copy(),(mpar['nm'],mpar['nk'],mpar['nh']),order='F')
        X1 = sf.dct(obj,norm='ortho',axis=0)
        X2 = sf.dct(X1.copy(),norm='ortho',axis=1)
        X3 = sf.dct(X2.copy(),norm='ortho',axis=2)

        
        XX = X3.flatten(order='F')
        ind = np.argsort(abs(XX.copy()))[::-1]
        i = 1   
        while linalg.norm(XX[ind[:i]].copy())/linalg.norm(XX) < level:
              i += 1    
        
        needed = i
        
        index_reduced = np.sort(ind[:i])
        
        
        return index_reduced
   

        
def Fsys(State, Stateminus, Control_sparse, Controlminus_sparse, StateSS, ControlSS, 
         Gamma_state, indexMUdct, indexVKdct, par, mpar, grid, targets, Copula, P, aggrshock):
    
    '''
    System of equations written in Schmitt-GrohÃ©-Uribe generic form with states and controls
    
    Parameters
    ----------
   
    State : ndarray
        Vector of state variables t+1 (only marginal distributions for histogram)
    Stateminus: ndarray
        Vector of state variables t (only marginal distributions for histogram)
    Control_sparse: ndarray
        Vector of state variables t+1 (only coefficients of sparse polynomial)
    Controlminus_sparse: ndarray
        Vector of state variables t (only coefficients of sparse polynomial)
    StateSS and ControlSS: matrix or ndarray
        Value of the state and control variables in steady state. For the Value functions these are at full grids.
    Gamma_state: coo_matrix
        Mapping such that perturbationof marginals are still distributions (sum to 1).
    Gamma_control: ndarray
        Values of the polynomial base at all nodes to map sparse coefficient changes to full grid
    InvGamma: coo_matrix
        Projection of Value functions etc. to Coeffeicent space for sparse polynomials.
    par, moar: dict
        Model and numerical parameters (structure)
    Grid: dict
        Liquid, illiquid and productivity grid
    Targets: dict
        Stores targets for government policy
   Copula : dict
        points for interpolation of joint distribution
    P: ndarray
        steady state transition matrix
    aggrshock: str 
        sets wether the Aggregate shock is TFP or uncertainty
    
    '''
    
    ## Initialization
    mutil = lambda x : 1./np.power(x,par['xi'])
#    invmutil = lambda x : (1./x)**(1./par['xi'])
    invmutil = lambda x : np.power(1./x,1./par['xi'])
    
    # Generate meshes for b,k,h

    
    # number of states, controls
    nx = mpar['numstates'] # number of states
    ny = mpar['numcontrols'] # number of controls
    NxNx= nx - mpar['os'] # number of states w/o aggregates
    Ny = len(indexMUdct) + len(indexVKdct)
    NN = mpar['nm']*mpar['nh']*mpar['nk'] # number of points in the full grid
    
    # Initialize LHS and RHS
    LHS = np.zeros((nx+Ny+mpar['oc'],1))
    RHS = np.zeros((nx+Ny+mpar['oc'],1))
    
    ## Indexes for LHS/RHS
    # Indexes for controls
    mutil_cind = np.array(range(len(indexMUdct)))
    Vkind = len(indexMUdct) + np.array(range(len(indexVKdct)))
    
    Qind = Ny
    PIind = Ny+1
    Yind = Ny+2
    Gind = Ny+3
    Wind = Ny+4
    Rind = Ny+5
    Profitind = Ny+6
    Nind = Ny+7
    Tind = Ny+8
    Kind = Ny+9
    Bind = Ny+10
    
    # Indexes for states
    #distr_ind = np.arange(mpar['nm']*mpar['nh']-mpar['nh']-1)
    marginal_mind = range(mpar['nm']-1)
    marginal_kind = range(mpar['nm']-1,mpar['nm']+mpar['nk']-2)
    marginal_hind = range(mpar['nm']+mpar['nk']-2,mpar['nm']+mpar['nk']+mpar['nh']-4)
    
    RBind = NxNx
    Sind = NxNx+1
    
    ## Control variables
    
    Control = Control_sparse.copy()
    Controlminus = Controlminus_sparse.copy()
           
    Control[-mpar['oc']:] = ControlSS[-mpar['oc']:].copy() + Control_sparse[-mpar['oc']:,:].copy()
    Controlminus[-mpar['oc']:] = ControlSS[-mpar['oc']:].copy() + Controlminus_sparse[-mpar['oc']:,:].copy()
    
    ## State variables
    # read out marginal histogram in t+1, t
    Distribution = StateSS[:-2].copy() + Gamma_state.copy().dot(State[:NxNx].copy())
    Distributionminus = StateSS[:-2].copy() + Gamma_state.copy().dot(Stateminus[:NxNx].copy())

    # Aggregate Endogenous States
    RB = StateSS[-2] + State[-2]
    RBminus = StateSS[-2] + Stateminus[-2]
    
    # Aggregate Exogenous States
    S = StateSS[-1] + State[-1]
    Sminus = StateSS[-1] + Stateminus[-1]
    
    ## Split the control vector into items with names
    # Controls

    XX = np.zeros((NN,1))
    XX[indexMUdct] = Control[mutil_cind]
    
    aux = np.reshape(XX,(mpar['nm'],mpar['nk'],mpar['nh']),order='F')
    aux = sf.idct(aux.copy(),norm='ortho',axis=0)
    aux = sf.idct(aux.copy(),norm='ortho',axis=1)
    aux = sf.idct(aux.copy(),norm='ortho',axis=2)
    
    mutil_c_dev = aux.copy()
    
    mutil_c = mutil(mutil_c_dev.copy().flatten(order='F') + np.squeeze(np.asarray(ControlSS[np.array(range(NN))])))
    
    XX = np.zeros((NN,1))
    XX[indexVKdct] = Control[Vkind]
    
    aux = np.reshape(XX,(mpar['nm'],mpar['nk'],mpar['nh']),order='F')
    aux = sf.idct(aux.copy(),norm='ortho',axis=0)
    aux = sf.idct(aux.copy(),norm='ortho',axis=1)
    aux = sf.idct(aux.copy(),norm='ortho',axis=2)
    

    Vk_dev = aux.copy()
    Vk = mutil(Vk_dev.copy().flatten(order='F')+np.squeeze(np.asarray(ControlSS[np.array(range(NN))+NN])))
    
    
    # Aggregate Controls (t+1)
    PI = np.exp(Control[PIind])
    Y = np.exp(Control[Yind])
    K = np.exp(Control[Kind])
    B = np.exp(Control[Bind])
    
    # Aggregate Controls (t)
    PIminus = np.exp(Controlminus[PIind])
    Qminus = np.exp(Controlminus[Qind])
    Yminus = np.exp(Controlminus[Yind])
    Gminus = np.exp(Controlminus[Gind])
    Wminus = np.exp(Controlminus[Wind])
    Rminus = np.exp(Controlminus[Rind])
    Profitminus = np.exp(Controlminus[Profitind])
    Nminus = np.exp(Controlminus[Nind])
    Tminus = np.exp(Controlminus[Tind])
    Kminus = np.exp(Controlminus[Kind])
    Bminus = np.exp(Controlminus[Bind])
    
    
    ## Write LHS values
    # Controls
    LHS[nx+Vkind] = Controlminus[Vkind]
    LHS[nx+mutil_cind] = Controlminus[mutil_cind]
    LHS[nx+Qind] = Qminus
    LHS[nx+Yind] = Yminus
    LHS[nx+Gind] = Gminus
    LHS[nx+Wind] = Wminus
    LHS[nx+Rind] = Rminus
    LHS[nx+Profitind] = Profitminus
    LHS[nx+Nind] = Nminus
    LHS[nx+Tind] = Tminus
    LHS[nx+Kind] = Kminus
    LHS[nx+Bind] = Bminus
    
    
    # States
    # Marginal Distributions (Marginal histograms)
    #LHS[distr_ind] = Distribution[:mpar['nm']*mpar['nh']-1-mpar['nh']].copy()
    LHS[marginal_mind] = Distribution[:mpar['nm']-1]
    LHS[marginal_kind] = Distribution[mpar['nm']:mpar['nm']+mpar['nk']-1]
    LHS[marginal_hind] = Distribution[mpar['nm']+mpar['nk']:mpar['nm']+mpar['nk']+mpar['nh']-2]
    
    LHS[RBind] = RB
    LHS[Sind] = S
    
    # take into account that RB is in logs
    RB = np.exp(RB.copy())
    RBminus = np.exp(RBminus) 
    
    ## Set of differences for exogenous process
    RHS[Sind] = par['rhoS']*Sminus
    
    if aggrshock == 'MP':
        EPS_TAYLOR = Sminus
        TFP = 1.0
    elif aggrshock == 'TFP':
        TFP = np.exp(Sminus)
        EPS_TAYLOR = 0
    elif aggrshock == 'Uncertainty':
        TFP = 1.0
        EPS_TAYLOR = 0
   
        #Tauchen style for probability distribution next period
        P = ExTransitions(np.exp(Sminus), grid, mpar, par)['P_H']
        
    
    marginal_mminus = np.transpose(Distributionminus[:mpar['nm']].copy())
    marginal_kminus = np.transpose(Distributionminus[mpar['nm']:mpar['nm']+mpar['nk']].copy())
    marginal_hminus = np.transpose(Distributionminus[mpar['nm']+mpar['nk']:mpar['nm']+mpar['nk']+mpar['nh']].copy())
    
    Hminus = np.sum(np.multiply(grid['h'][:-1],marginal_hminus[:,:-1]))
    Lminus = np.sum(np.multiply(grid['m'],marginal_mminus))
    
    RHS[nx+Bind] = Lminus
    RHS[nx+Kind] = np.sum(grid['k']*np.asarray(marginal_kminus))
    
    # Calculate joint distributions
    cumdist = np.zeros((mpar['nm']+1,mpar['nk']+1,mpar['nh']+1))
    cm,ck,ch = np.meshgrid(np.asarray(np.cumsum(marginal_mminus)), np.asarray(np.cumsum(marginal_kminus)), np.asarray(np.cumsum(marginal_hminus)), indexing = 'ij')
    
    # griddata does not support extrapolation for 3D
    #cumdist[1:,1:,1:] = np.reshape(Copula((cm.flatten(order='F').copy(),ck.flatten(order='F').copy(),ch.flatten(order='F').copy())),(mpar['nm'],mpar['nk'],mpar['nh']), order='F')
    Copula_aux = griddata(Copula['grid'],Copula['value'],(cm.flatten(order='F').copy(),ck.flatten(order='F').copy(),ch.flatten(order='F').copy()))
    Copula_bounds = griddata(Copula['grid'],Copula['value'],(cm.flatten(order='F').copy(),ck.flatten(order='F').copy(),ch.flatten(order='F').copy()),method='nearest')
    Copula_aux[np.isnan(Copula_aux.copy())] = Copula_bounds[np.isnan(Copula_aux.copy())].copy()
    
    cumdist[1:,1:,1:] = np.reshape(Copula_aux,(mpar['nm'],mpar['nk'],mpar['nh']), order='F')
    JDminus = np.diff(np.diff(np.diff(cumdist,axis=0),axis=1),axis=2)
    
    meshes={}
    meshes['m'], meshes['k'], meshes['h'] = np.meshgrid(grid['m'],grid['k'],grid['h'], indexing = 'ij')
    
    ## Aggregate Output
    mc = par['mu'] - (par['beta']* np.log(PI)*Y/Yminus - np.log(PIminus))/par['kappa']
    
    RHS[nx+Nind] = np.power(par['tau']*TFP*par['alpha']*np.power(Kminus,(1.-par['alpha']))*mc,1./(1.-par['alpha']+par['gamma']))
    RHS[nx+Yind] = (TFP*np.power(Nminus,par['alpha'])*np.power(Kminus,1.-par['alpha']))
    ## Prices that are not a part of control vector
    # Wage Rate
    RHS[nx+Wind] = TFP * par['alpha'] * mc *np.power((Kminus/Nminus),1.-par['alpha'])
    # Return on Capital
    RHS[nx+Rind] = TFP * (1.-par['alpha']) * mc *np.power((Nminus/Kminus),par['alpha']) - par['delta']
    # Profits for Enterpreneurs
    RHS[nx+Profitind] = (1.-mc)*Yminus - Yminus*(1./(1.-par['mu']))/par['kappa']/2.*np.log(PIminus)**2 + 1./2.*par['phi']*((K-Kminus)**2)/Kminus
    
       
    ## Wages net of leisure services
    WW = (par['gamma']/(1.+par['gamma'])*(Nminus/Hminus)*Wminus).item()*np.ones((mpar['nm'],mpar['nk'],mpar['nh']))
    WW[:,:,-1] = Profitminus.item()*par['profitshare']*np.ones((mpar['nm'],mpar['nk']))
    
    ## Incomes (grids)
    inc ={}
    inc['labor'] = par['tau']*WW.copy()*meshes['h'].copy()
    inc['rent'] = meshes['k']*Rminus.item()
    inc['capital'] = meshes['k']*Qminus.item()
    inc['money'] = meshes['m'].copy()*(RBminus.item()/PIminus.item()+(meshes['m']<0)*par['borrwedge']/PIminus.item())
    
    
    ## Update policies
    EVk = np.reshape(np.asarray(np.reshape(Vk.copy(),(mpar['nm']*mpar['nk'], mpar['nh']),order = 'F').dot(P.copy().T)),(mpar['nm'],mpar['nk'],mpar['nh']),order = 'F')
    RBaux = (RB.item()+(meshes['m']<0).copy()*par['borrwedge'])/PI.item()
    EVm = np.reshape(np.asarray(np.reshape(np.multiply(RBaux.flatten(order='F').T.copy(),mutil_c.flatten(order='F').copy()),(mpar['nm']*mpar['nk'],mpar['nh']),order='F').dot(np.transpose(P.copy()))),(mpar['nm'],mpar['nk'],mpar['nh']),order='F')
    
    
    result_EGM_policyupdate = EGM_policyupdate(EVm,EVk,Qminus.item(),PIminus.item(),RBminus.item(),inc,meshes,grid,par,mpar)
    c_a_star = result_EGM_policyupdate['c_a_star']
    m_a_star = result_EGM_policyupdate['m_a_star']
    k_a_star = result_EGM_policyupdate['k_a_star']
    c_n_star = result_EGM_policyupdate['c_n_star']
    m_n_star = result_EGM_policyupdate['m_n_star']
    
    meshaux = meshes.copy()
    meshaux['h'][:,:,-1] = 1000.
    
    ## Update Marginal Value of Bonds
    mutil_c_n = mutil(c_n_star.copy())
    mutil_c_a = mutil(c_a_star.copy())
    mutil_c_aux = par['nu']*mutil_c_a + (1-par['nu'])*mutil_c_n
    aux = invmutil(mutil_c_aux.copy().flatten(order='F'))-np.squeeze(np.asarray(ControlSS[np.array(range(NN))]))
    aux = np.reshape(aux,(mpar['nm'],mpar['nk'],mpar['nh']),order='F')
    aux = sf.dct(aux.copy(),norm='ortho',axis=0)
    aux = sf.dct(aux.copy(),norm='ortho',axis=1)
    aux = sf.dct(aux.copy(),norm='ortho',axis=2)

    
    DC = np.asmatrix(aux.copy().flatten(order='F')).T
    
    RHS[nx+mutil_cind] = DC[indexMUdct]
    
    
    ## Update Marginal Value of capital
    EVk = np.reshape(Vk,(mpar['nm']*mpar['nk'],mpar['nh']),order='F').dot(P.copy().T)
            
    Vpoints = np.concatenate(( [meshaux['m'].flatten(order='F')],[meshaux['k'].flatten(order='F')],[meshaux['h'].flatten(order='F')]),axis=0).T
    # griddata does not support extrapolation for 3D   
    Vk_next = griddata(Vpoints,np.asarray(EVk).flatten(order='F').copy(),(m_n_star.copy().flatten(order='F'),meshaux['k'].copy().flatten(order='F'),meshaux['h'].copy().flatten(order='F')),method='linear')
    Vk_next_bounds = griddata(Vpoints,np.asarray(EVk).flatten(order='F').copy(),(m_n_star.copy().flatten(order='F'),meshaux['k'].copy().flatten(order='F'),meshaux['h'].copy().flatten(order='F')),method='nearest')
    Vk_next[np.isnan(Vk_next.copy())] = Vk_next_bounds[np.isnan(Vk_next.copy())].copy()
       
    Vk_aux = par['nu']*(Rminus.item()+Qminus.item())*mutil_c_a + (1-par['nu'])*Rminus.item()*mutil_c_n +par['beta']*(1-par['nu'])*np.reshape(Vk_next,(mpar['nm'],mpar['nk'],mpar['nh']),order='F')
    
    aux = invmutil(Vk_aux.copy().flatten(order='F')) - np.squeeze(np.asarray(ControlSS[np.array(range(NN))+NN]))
    aux = np.reshape(aux.copy(),(mpar['nm'],mpar['nk'],mpar['nh']),order='F')
    aux = sf.dct(aux.copy(),norm='ortho',axis=0)
    aux = sf.dct(aux.copy(),norm='ortho',axis=1)
    aux = sf.dct(aux.copy(),norm='ortho',axis=2)    
    
    
    DC = np.asmatrix(aux.copy().flatten(order='F')).T
        
    RHS[nx+Vkind] = DC[indexVKdct]
    
    ## Differences for distriutions
    # find next smallest on-grid value for money choices
    weight11 = np.empty((mpar['nm']*mpar['nk'],mpar['nh'],mpar['nh']))
    weight12 = np.empty((mpar['nm']*mpar['nk'],mpar['nh'],mpar['nh']))
    weight21 = np.empty((mpar['nm']*mpar['nk'],mpar['nh'],mpar['nh']))
    weight22 = np.empty((mpar['nm']*mpar['nk'],mpar['nh'],mpar['nh']))
    
    weightn1 = np.empty((mpar['nm']*mpar['nk'],mpar['nh'],mpar['nh']))
    weightn2 = np.empty((mpar['nm']*mpar['nk'],mpar['nh'],mpar['nh']))
    
    ra_genweight = GenWeight(m_a_star,grid['m'])
    Dist_m_a = ra_genweight['weight'].copy()
    idm_a = ra_genweight['index'].copy()
    
    rn_genweight = GenWeight(m_n_star,grid['m'])
    Dist_m_n = rn_genweight['weight'].copy()
    idm_n = rn_genweight['index'].copy()
    
    rk_genweight = GenWeight(k_a_star,grid['k'])
    Dist_k = rk_genweight['weight'].copy()
    idk_a = rk_genweight['index'].copy()
    
    idk_n = np.reshape(np.tile(np.outer(np.ones((mpar['nm'])),np.array(range(mpar['nk']))),(1,1,mpar['nh'])),(mpar['nm'],mpar['nk'],mpar['nh']),order = 'F')
        
    # Transition matrix for adjustment case
    idm_a = np.tile(np.asmatrix(idm_a.copy().flatten('F')).T,(1,mpar['nh']))
    idk_a = np.tile(np.asmatrix(idk_a.copy().flatten('F')).T,(1,mpar['nh']))
    idh = np.kron(np.array(range(mpar['nh'])),np.ones((1,mpar['nm']*mpar['nk']*mpar['nh'])))
    
    idm_a = idm_a.copy().astype(int)
    idk_a = idk_a.copy().astype(int)
    idh = idh.copy().astype(int)
    
    index11 = np.ravel_multi_index([idm_a.flatten(order='F'),idk_a.flatten(order='F'),idh.flatten(order='F')],
                                       (mpar['nm'],mpar['nk'],mpar['nh']),order='F')
    index12 = np.ravel_multi_index([idm_a.flatten(order='F'),idk_a.flatten(order='F')+1,idh.flatten(order='F')],
                                       (mpar['nm'],mpar['nk'],mpar['nh']),order='F')
    index21 = np.ravel_multi_index([idm_a.flatten(order='F')+1,idk_a.flatten(order='F'),idh.flatten(order='F')],
                                       (mpar['nm'],mpar['nk'],mpar['nh']),order='F')
    index22 = np.ravel_multi_index([idm_a.flatten(order='F')+1,idk_a.flatten(order='F')+1,idh.flatten(order='F')],
                                       (mpar['nm'],mpar['nk'],mpar['nh']),order='F')
    # for no-adjustment case
    idm_n = np.tile(np.asmatrix(idm_n.copy().flatten('F')).T,(1,mpar['nh']))
    idk_n = np.tile(np.asmatrix(idk_n.copy().flatten('F')).T,(1,mpar['nh']))
        
    idm_n = idm_n.copy().astype(int)
    idk_n = idk_n.copy().astype(int)
    
    indexn1 = np.ravel_multi_index([idm_n.flatten(order='F'),idk_n.flatten(order='F'),idh.flatten(order='F')],
                                       (mpar['nm'],mpar['nk'],mpar['nh']),order='F')
    indexn2 = np.ravel_multi_index([idm_n.flatten(order='F')+1,idk_n.flatten(order='F'),idh.flatten(order='F')],
                                       (mpar['nm'],mpar['nk'],mpar['nh']),order='F')
    
    for hh in range(mpar['nh']):
        
        # corresponding weights
        weight11_aux = (1-Dist_m_a[:,:,hh].copy())*(1-Dist_k[:,:,hh].copy())
        weight12_aux = (1-Dist_m_a[:,:,hh].copy())*(Dist_k[:,:,hh].copy())  
        weight21_aux = Dist_m_a[:,:,hh].copy()*(1-Dist_k[:,:,hh].copy())
        weight22_aux = Dist_m_a[:,:,hh].copy()*(Dist_k[:,:,hh].copy())
        
        weightn1_aux = (1-Dist_m_n[:,:,hh].copy())
        weightn2_aux = (Dist_m_n[:,:,hh].copy())
        
        # dimensions (m*k,h',h)
        weight11[:,:,hh] = np.outer(weight11_aux.flatten(order='F').copy(),P[hh,:].copy())
        weight12[:,:,hh] = np.outer(weight12_aux.flatten(order='F').copy(),P[hh,:].copy())
        weight21[:,:,hh] = np.outer(weight21_aux.flatten(order='F').copy(),P[hh,:].copy())
        weight22[:,:,hh] = np.outer(weight22_aux.flatten(order='F').copy(),P[hh,:].copy())
        
        weightn1[:,:,hh] = np.outer(weightn1_aux.flatten(order='F').copy(),P[hh,:].copy())
        weightn2[:,:,hh] = np.outer(weightn2_aux.flatten(order='F').copy(),P[hh,:].copy())
        
    weight11= np.ndarray.transpose(weight11.copy(),(0,2,1))       
    weight12= np.ndarray.transpose(weight12.copy(),(0,2,1))       
    weight21= np.ndarray.transpose(weight21.copy(),(0,2,1))       
    weight22= np.ndarray.transpose(weight22.copy(),(0,2,1))       
    
    rowindex = np.tile(range(mpar['nm']*mpar['nk']*mpar['nh']),(1,4*mpar['nh']))
    
    H_a = sp.coo_matrix((np.hstack((weight11.flatten(order='F'),weight21.flatten(order='F'),weight12.flatten(order='F'),weight22.flatten(order='F'))), 
                   (np.squeeze(rowindex), np.hstack((np.squeeze(np.asarray(index11)),np.squeeze(np.asarray(index21)),np.squeeze(np.asarray(index12)),np.squeeze(np.asarray(index22)))) )), 
                    shape=(mpar['nm']*mpar['nk']*mpar['nh'],mpar['nm']*mpar['nk']*mpar['nh']) )

    weightn1= np.ndarray.transpose(weightn1.copy(),(0,2,1))       
    weightn2= np.ndarray.transpose(weightn2.copy(),(0,2,1))       
    
    rowindex = np.tile(range(mpar['nm']*mpar['nk']*mpar['nh']),(1,2*mpar['nh']))
    
    H_n = sp.coo_matrix((np.hstack((weightn1.flatten(order='F'),weightn2.flatten(order='F'))), 
                   (np.squeeze(rowindex), np.hstack((np.squeeze(np.asarray(indexn1)),np.squeeze(np.asarray(indexn2)))) )), 
                    shape=(mpar['nm']*mpar['nk']*mpar['nh'],mpar['nm']*mpar['nk']*mpar['nh']) )
    
    # Joint transition matrix and transitions
    H = par['nu']*H_a.copy() +(1-par['nu'])*H_n.copy()    
        
    JD_new = JDminus.flatten(order='F').copy().dot(H.todense())
    JD_new = np.reshape(np.asarray(JD_new.copy()),(mpar['nm'],mpar['nk'],mpar['nh']),order='F')
    
    # Next period marginal histograms
    # liquid assets
    aux_m = np.sum(np.sum(JD_new.copy(),axis=1),axis=1)
    RHS[marginal_mind] = np.asmatrix(aux_m[:-1].copy()).T
    
    # illiquid asset
    aux_k = np.sum(np.sum(JD_new.copy(),axis=0),axis=1)
    RHS[marginal_kind] = np.asmatrix(aux_k[:-1].copy()).T
    
    # human capital
    aux_h = np.sum(np.sum(JD_new.copy(),axis=0),axis=0)
    RHS[marginal_hind] = np.asmatrix(aux_h[:-2].copy()).T
    
    ## Third Set: Government Budget constraint
    # Return on bonds (Taylor Rule)
    RHS[RBind] = np.log(par['RB'])+par['rho_R']*np.log(RBminus/par['RB']) + np.log(PIminus/par['PI'])*((1.-par['rho_R'])*par['theta_pi'])+EPS_TAYLOR
    
    # Inflation jumps to equilibrate real bond supply and demand
    
    if par['tau'] < 1:
       
       taxrevenue = (1-par['tau'])*Wminus*Nminus + (1-par['tau'])*Profitminus
       RHS[nx+PIind] = par['rho_B']*np.log(Bminus/targets['B'])+par['rho_B']*np.log(RBminus/par['RB']) - (par['rho_B']+par['gamma_pi'])*np.log(PIminus/par['PI']) - par['gamma_T'] *np.log(Tminus/targets['T'])
                       
       LHS[nx+PIind] = np.log(B/targets['B'])
       
       # Government expenditure
       RHS[nx+Gind] = B - Bminus*RBminus/PIminus +Tminus
       RHS[nx+Tind] = taxrevenue
       
       # Resulting price of capital
       RHS[nx+Qind] = (par['phi']*(K/Kminus-1)+1) - par['ABS']
       
    else:
       RHS[nx+PIind] = targets['B']
       LHS[nx+PIind] = B
       
       RHS[nx+Gind] = targets['G'] 
       RHS[nx+Tind] = 0.
    
       RHS[nx+Qind] = (par['phi']*(K/Kminus-1)+1) - par['ABS']
       
    ## Difference
    Difference = (LHS-RHS)
    
          
    
    return {'Difference':Difference, 'LHS':LHS, 'RHS':RHS, 'JD_new': JD_new, 'c_a_star':c_a_star, 'm_a_star':m_a_star,
            'k_a_star':k_a_star,'c_n_star':c_n_star,'m_n_star':m_n_star,'P':P}




def EGM_policyupdate(EVm,EVk, Qminus, PIminus, RBminus, inc, meshes,grid,par,mpar):
    
    ## EGM step 1
    EMU = par['beta']*np.reshape(EVm.copy(),(mpar['nm'],mpar['nk'],mpar['nh']), order = 'F')
    c_new = 1./np.power(EMU,(1./par['xi']))
    # Calculate assets consistent with choices being (m')
    # Calculate initial money position from the budget constraint,
    # that leads to the optimal consumption choice
    m_star_n = (c_new.copy() + meshes['m'].copy()-inc['labor'].copy()-inc['rent'].copy())
    m_star_n = m_star_n.copy()/(RBminus/PIminus+(m_star_n.copy()<0)*par['borrwedge']/PIminus)
    
    # Identify binding constraints
    binding_constraints = meshes['m'].copy() < np.tile(m_star_n[0,:,:].copy(),(mpar['nm'],1,1))
    
    # Consumption when drawing assets m' to zero: Eat all resources
    Resource = inc['labor'].copy() + inc['rent'].copy() + inc['money'].copy()
    
    m_star_n = np.reshape(m_star_n.copy(),(mpar['nm'],mpar['nk']*mpar['nh']),order='F')
    c_n_aux = np.reshape(c_new.copy(),(mpar['nm'],mpar['nk']*mpar['nh']),order='F')
    
    # Interpolate grid['m'] and c_n_aux defined on m_n_aux over grid['m']
    # Check monotonicity of m_n_aux
    if np.sum(np.abs(np.diff(np.sign(np.diff(m_star_n.copy(),axis=0)),axis=0)),axis=1).max() != 0.:
       print ' Warning: non monotone future liquid asset choice encountered '
       
    c_update = np.zeros((mpar['nm'],mpar['nk']*mpar['nh']))
    m_update = np.zeros((mpar['nm'],mpar['nk']*mpar['nh']))
    
    for hh in range(mpar['nk']*mpar['nh']):
         
        Savings = interp1d(np.squeeze(np.asarray(m_star_n[:,hh].copy())), grid['m'].copy(), fill_value='extrapolate')
        m_update[:,hh] = Savings(grid['m'].copy())
        Consumption = interp1d(np.squeeze(np.asarray(m_star_n[:,hh].copy())), np.squeeze(np.asarray(c_n_aux[:,hh].copy())), fill_value='extrapolate')
        c_update[:,hh] = Consumption(grid['m'].copy())
    
    
    c_n_star = np.reshape(c_update,(mpar['nm'],mpar['nk'],mpar['nh']),order = 'F')
    m_n_star = np.reshape(m_update,(mpar['nm'],mpar['nk'],mpar['nh']),order = 'F')
    
    c_n_star[binding_constraints] = np.squeeze(np.asarray(Resource[binding_constraints].copy() - grid['m'][0]))
    m_n_star[binding_constraints] = grid['m'].copy().min()
    
    m_n_star[m_n_star>grid['m'][-1]] = grid['m'][-1]
    
    ## EGM step 2: find Optimal Portfolio Combinations
    term1 = par['beta']*np.reshape(EVk,(mpar['nm'],mpar['nk'],mpar['nh']),order = 'F')
    
    E_return_diff = term1/Qminus - EMU
    
    # Check quasi-monotonicity of E_return_diff
    if np.sum(np.abs(np.diff(np.sign(E_return_diff),axis=0)),axis = 0).max() > 2.:
       print ' Warning: multiple roots of portfolio choic encountered'
       
    # Find an m_a for given ' taht solves the difference equation
    m_a_aux = Fastroot(grid['m'],E_return_diff)
    m_a_aux = np.maximum(m_a_aux.copy(),grid['m'][0])
    m_a_aux = np.minimum(m_a_aux.copy(),grid['m'][-1])
    m_a_aux = np.reshape(m_a_aux.copy(),(mpar['nk'],mpar['nh']),order = 'F')
    
    ## EGM step 3
    # Constraints for money and capital are not binding
    EMU = np.reshape(EMU.copy(),(mpar['nm'],mpar['nk']*mpar['nh']),order = 'F')

    # Interpolation of psi-function at m*_n(m,k)
    idx = np.digitize(m_a_aux, grid['m'])-1 # find indexes on grid next smallest to optimal policy
    idx[m_a_aux<=grid['m'][0]]   = 0  # if below minimum
    idx[m_a_aux>=grid['m'][-1]] = mpar['nm']-2 #if above maximum
    step = np.diff(grid['m'].copy()) # Stepsize on grid
    s = (m_a_aux.copy() - grid['m'][idx])/step[idx]  # Distance of optimal policy to next grid point

    aux_index = np.array(range(0,(mpar['nk']*mpar['nh'])))*mpar['nm']  # aux for linear indexes
    aux3      = EMU.flatten(order = 'F').copy()[idx.flatten(order='F').copy()+aux_index.flatten(order = 'F').copy()]  # calculate linear indexes

    # Interpolate EMU(m',k',s'*h',M',K') over m*_n(k'), m-dim is dropped
    EMU_star        = aux3 + s.flatten(order = 'F')*(EMU.flatten(order='F').copy()[idx.flatten(order = 'F').copy() + aux_index.flatten(order = 'F').copy()+1]-aux3) # linear interpolation

    c_a_aux         = 1/(EMU_star.copy()**(1/par['xi']))
    cap_expenditure = np.squeeze(inc['capital'][0,:,:])
    auxL            = np.squeeze(inc['labor'][0,:,:])

    # Resources that lead to capital choice k' = c + m*(k') + k' - w*h*N = value of todays cap and money holdings
    Resource = c_a_aux.copy() + m_a_aux.flatten(order = 'F').copy() + cap_expenditure.flatten(order = 'F').copy() - auxL.flatten(order = 'F').copy()

    c_a_aux  = np.reshape(c_a_aux.copy(), (mpar['nk'], mpar['nh']),order = 'F')
    Resource = np.reshape(Resource.copy(), (mpar['nk'], mpar['nh']),order = 'F')

    # Money constraint is not binding, but capital constraint is binding
    m_star_zero = np.squeeze(m_a_aux[0,:].copy()) # Money holdings that correspond to k'=0:  m*(k=0)

    # Use consumption at k'=0 from constrained problem, when m' is on grid
    aux_c     = np.reshape(c_new[:,0,:],(mpar['nm'], mpar['nh']),order = 'F')
    aux_inc   = np.reshape(inc['labor'][0,0,:],(1, mpar['nh']),order = 'F')
    cons_list = []
    res_list  = []
    mon_list  = []
    cap_list  = []


    for j in range(mpar['nh']):
      # When choosing zero capital holdings, HHs might still want to choose money holdings smaller than m*(k'=0)
       if m_star_zero[j]>grid['m'][0]:
        # Calculate consumption policies, when HHs chooses money holdings lower than m*(k'=0) and capital holdings k'=0 and save them in cons_list
        log_index    = grid['m'].T.copy() < m_star_zero[j]
        # aux_c is the consumption policy under no cap. adj.
        c_k_cons     = aux_c[log_index, j].copy()
        cons_list.append( c_k_cons.copy() ) # Consumption at k'=0, m'<m_a*(0)
        # Required Resources: Money choice + Consumption - labor income Resources that lead to k'=0 and m'<m*(k'=0)
        res_list.append( grid['m'].T[log_index] + c_k_cons.copy() - aux_inc[0,j] )
        mon_list.append( grid['m'].T[log_index])
        cap_list.append( np.zeros((np.sum(log_index))))
    

    # Merge lists
    c_a_aux  = np.reshape(c_a_aux.copy(),(mpar['nk'], mpar['nh']),order = 'F')
    m_a_aux  = np.reshape(m_a_aux.copy(),(mpar['nk'], mpar['nh']),order = 'F')
    Resource = np.reshape(Resource.copy(),(mpar['nk'], mpar['nh']),order = 'F')
    
    cons_list_1=[]
    res_list_1=[]
    mon_list_1=[]
    cap_list_1=[]
    
    for j in range(mpar['nh']):
   
      cons_list_1.append( np.vstack((np.asmatrix(cons_list[j]).T, np.asmatrix(c_a_aux[:,j]).T)) )
      res_list_1.append( np.vstack((np.asmatrix(res_list[j]).T, np.asmatrix(Resource[:,j]).T)) )
      mon_list_1.append( np.vstack((np.asmatrix(mon_list[j]).T, np.asmatrix(m_a_aux[:,j]).T)) )
      cap_list_1.append( np.vstack((np.asmatrix(cap_list[j].copy()).T, np.asmatrix(grid['k']).T)) )
    
    ## EGM step 4: Interpolate back to fixed grid
    c_a_star = np.zeros((mpar['nm']*mpar['nk'], mpar['nh']),order = 'F')
    m_a_star = np.zeros((mpar['nm']*mpar['nk'], mpar['nh']),order = 'F')
    k_a_star = np.zeros((mpar['nm']*mpar['nk'], mpar['nh']),order = 'F')
    Resource_grid  = np.reshape(inc['capital']+inc['money']+inc['rent'],(mpar['nm']*mpar['nk'], mpar['nh']),order = 'F')
    labor_inc_grid = np.reshape(inc['labor'],(mpar['nm']*mpar['nk'], mpar['nh']),order = 'F')

    for j in range(mpar['nh']):
      log_index=Resource_grid[:,j] < res_list[j][0]
    
      # when at most one constraint binds:
      # Check monotonicity of resources
      
      if np.sum(np.abs(np.diff(np.sign(np.diff(res_list[j])))),axis = 0).max() != 0. :
         print 'warning(non monotone resource list encountered)'
      cons = interp1d(np.squeeze(np.asarray(res_list_1[j].copy())), np.squeeze(np.asarray(cons_list_1[j].copy())),fill_value='extrapolate')
      c_a_star[:,j] = cons(Resource_grid[:,j].copy())
      mon = interp1d(np.squeeze(np.asarray(res_list_1[j].copy())), np.squeeze(np.asarray(mon_list_1[j].copy())),fill_value='extrapolate')
      m_a_star[:,j] = mon(Resource_grid[:,j].copy())
      cap = interp1d(np.squeeze(np.asarray(res_list_1[j].copy())), np.squeeze(np.asarray(cap_list_1[j].copy())),fill_value='extrapolate')
      k_a_star[:,j] = cap(Resource_grid[:,j].copy())
      # Lowest value of res_list corresponds to m_a'=0 and k_a'=0.
    
      # Any resources on grid smaller then res_list imply that HHs consume all resources plus income.
      # When both constraints are binding:
      c_a_star[log_index,j] = Resource_grid[log_index,j].copy() + labor_inc_grid[log_index,j].copy()-grid['m'][0]
      m_a_star[log_index,j] = grid['m'][0]
      k_a_star[log_index,j] = 0.


    c_a_star = np.reshape(c_a_star.copy(),(mpar['nm'] ,mpar['nk'], mpar['nh']),order = 'F')
    k_a_star = np.reshape(k_a_star.copy(),(mpar['nm'] ,mpar['nk'], mpar['nh']),order = 'F')
    m_a_star = np.reshape(m_a_star.copy(),(mpar['nm'] ,mpar['nk'], mpar['nh']),order = 'F')

    k_a_star[k_a_star.copy()>grid['k'][-1]] = grid['k'][-1]
    m_a_star[m_a_star.copy()>grid['m'][-1]] = grid['m'][-1]    
    
    return {'c_a_star': c_a_star, 'm_a_star': m_a_star, 'k_a_star': k_a_star,'c_n_star': c_n_star, 'm_n_star': m_n_star}


def plot_IRF(mpar,par,gx,hx,joint_distr,Gamma_state,grid,targets,Output):
        
    x0 = np.zeros((mpar['numstates'],1))
    x0[-1] = par['sigmaS']
        
    MX = np.vstack((np.eye(len(x0)), gx))
    IRF_state_sparse=[]
    x=x0.copy()
    mpar['maxlag']=16
        
    for t in range(0,mpar['maxlag']):
        IRF_state_sparse.append(np.dot(MX,x))
        x=np.dot(hx,x)
        
    IRF_state_sparse = np.asmatrix(np.squeeze(np.asarray(IRF_state_sparse))).T
        
    aux = np.sum(np.sum(joint_distr,1),0)
        
    scale={}
    scale['h'] = np.tile(np.vstack((1,aux[-1])),(1,mpar['maxlag']))
        
    IRF_distr = Gamma_state*IRF_state_sparse[:mpar['numstates']-mpar['os'],:mpar['maxlag']]
        
    # preparation
        
    IRF_H = 100*grid['h'][:-1]*IRF_distr[mpar['nm']+mpar['nk']:mpar['nm']+mpar['nk']+mpar['nh']-1,1:]/par['H']
    K = np.asarray(grid['k']*IRF_distr[mpar['nm']:mpar['nm']+mpar['nk'],:] + grid['K']).T
    I = (K[1:] - (1-par['delta'])*K[:-1]).T
    IRF_I = 100*(I/(par['delta']*grid['K'])-1)
    IRF_K = 100*grid['k']*IRF_distr[mpar['nm']:mpar['nm']+mpar['nk'],1:]/grid['K']
    IRF_M = 100*grid['m']*IRF_distr[:mpar['nm'],1:]/(targets['B']+par['ABS']*grid['K'])
    K=K.copy().T
    M = grid['m']*IRF_distr[:mpar['nm'],:] + targets['B'] - par['ABS']*(K-grid['K'])
    IRF_S=100*IRF_state_sparse[mpar['numstates']-1,:-1]
    
    Y = Output*(1+IRF_state_sparse[-1-mpar['oc']+3, :-1])
    G = par['G']*(1+IRF_state_sparse[-1-mpar['oc']+4, :-1])
    IRF_C = 100*((Y-G-I)/(Output-par['G']-par['delta']*grid['K'])-1)
    IRF_Y=100*IRF_state_sparse[-1-mpar['oc']+3, :-1]
    IRF_G=100*IRF_state_sparse[-1-mpar['oc']+4, :-1]
    IRF_W=100*IRF_state_sparse[-1-mpar['oc']+5, :-1]
    IRF_N=100*IRF_state_sparse[-1-mpar['oc']+8, :-1]
    IRF_R=100*IRF_state_sparse[-1-mpar['oc']+6, :-1]
    IRF_PI=100*100*IRF_state_sparse[-1-mpar['oc']+2, :-1]
        
    PI=1 + IRF_state_sparse[-1-mpar['oc']+2, :-1]
    Q = par['Q']*(1+IRF_state_sparse[-1-mpar['oc']+1, :-1])
    R = par['R']*(1+IRF_state_sparse[-1-mpar['oc']+6, :-1])
    RB=par['RB']+(IRF_state_sparse[-2, 1:])
    IRF_RB=100*100*(RB-par['RB'])
    IRF_RBREAL=100*100*(RB/PI-par['RB'])
    IRF_Q = 100*100*(Q-par['Q'])
    IRF_D = 100*100*((1+IRF_R/100)*par['R'] - par['R'])
    Deficit = 100*(M[:,1:] - M[:,:-1]/PI)/Y
    IRF_LP = 100*100*(((Q[:,1:]+R[:,1:])/Q[:,:-1]-RB[:,:-1]/PI[:,1:])-((1+par['R']/par['Q'])-par['RB']))
    
    
    f_Y = plt.figure(1)
    line1,=plt.plot(range(1,mpar['maxlag']),np.squeeze(np.asarray(IRF_Y)),label='IRF_Y')
    plt.plot(range(0,mpar['maxlag']),np.zeros((mpar['maxlag'])),'k--' )
#    patch_Y = mpatches.Patch(color='blue', label='IRF_Y_thetapi')
#    plt.legend(handles=[patch_Y])
    plt.legend(handles=[line1])
    plt.xlabel('Quarter')
    plt.ylabel('Percent') 
    f_Y.show()
#        
    f_C = plt.figure(2)
    line1,=plt.plot(range(1,mpar['maxlag']),np.squeeze(np.asarray(IRF_C)),label='IRF_C')
    plt.plot(range(0,mpar['maxlag']),np.zeros((mpar['maxlag'])),'k--' )
    plt.legend(handles=[line1])
    plt.xlabel('Quarter')
    plt.ylabel('Percent') 
    f_C.show()

    f_I = plt.figure(3)
    line1,=plt.plot(range(1,mpar['maxlag']),np.squeeze(np.asarray(IRF_I)),label='IRF_I')
    plt.plot(range(0,mpar['maxlag']),np.zeros((mpar['maxlag'])),'k--' )
    plt.legend(handles=[line1])
    plt.xlabel('Quarter')
    plt.ylabel('Percent') 
    f_I.show()
        
    f_G = plt.figure(4)
    line1,=plt.plot(range(1,mpar['maxlag']),np.squeeze(np.asarray(IRF_G)), label='IRF_G')
    plt.plot(range(0,mpar['maxlag']),np.zeros((mpar['maxlag'])),'k--' )
    # plt.ylim((-1, 1))
    plt.legend(handles=[line1])
    plt.xlabel('Quarter')
    plt.ylabel('Percent') 
    f_G.show()

    f_Deficit = plt.figure(5)
    line1,=plt.plot(range(1,mpar['maxlag']),np.squeeze(np.asarray(Deficit)), label='IRF_Deficit')
    plt.plot(range(0,mpar['maxlag']),np.zeros((mpar['maxlag'])),'k--' )
    plt.legend(handles=[line1])
    plt.xlabel('Quarter')
    plt.ylabel('Percentage Points') 
    f_Deficit.show()

    f_K = plt.figure(6)
    line1,=plt.plot(range(1,mpar['maxlag']),np.squeeze(np.asarray(IRF_K)), label='IRF_K')
    plt.plot(range(0,mpar['maxlag']),np.zeros((mpar['maxlag'])),'k--' )
    plt.legend(handles=[line1])
    plt.xlabel('Quarter')
    plt.ylabel('Percent') 
    f_K.show()

    f_M = plt.figure(7)
    line1,=plt.plot(range(1,mpar['maxlag']),np.squeeze(np.asarray(IRF_M)), label='IRF_M')
    plt.plot(range(0,mpar['maxlag']),np.zeros((mpar['maxlag'])),'k--' )
    plt.legend(handles=[line1])
    plt.xlabel('Quarter')
    plt.ylabel('Percent') 
    f_M.show()

    f_H = plt.figure(8)
    line1,=plt.plot(range(1,mpar['maxlag']),np.squeeze(np.asarray(IRF_H)), label='IRF_H')
    plt.plot(range(0,mpar['maxlag']),np.zeros((mpar['maxlag'])),'k--' )
    plt.legend(handles=[line1])
    plt.xlabel('Quarter')
    plt.ylabel('Percent') 
    f_H.show()
        
    f_S = plt.figure(10)
    line1,=plt.plot(range(1,mpar['maxlag']),np.squeeze(np.asarray(IRF_S)), label='IRF_S')
    plt.plot(range(0,mpar['maxlag']),np.zeros((mpar['maxlag'])),'k--' )
    plt.legend(handles=[line1])
    plt.xlabel('Quarter')
    plt.ylabel('Percent') 
    f_S.show()        
        
    f_RBPI = plt.figure(11)
    line1,=plt.plot(range(1,mpar['maxlag']),np.squeeze(np.asarray(IRF_RB)), label='nominal', color='red', linestyle='--')
    line2,=plt.plot(range(1,mpar['maxlag']),np.squeeze(np.asarray(IRF_RBREAL)), label='real', color='blue')
    plt.legend(handles=[line1, line2])
    plt.plot(range(0,mpar['maxlag']),np.zeros((mpar['maxlag'])),'k--' )
    plt.xlabel('Quarter')
    plt.ylabel('Basis Points') 
    f_RBPI.show()

    f_RB = plt.figure(12)
    line1,=plt.plot(range(1,mpar['maxlag']),np.squeeze(np.asarray(IRF_RB)), label='IRF_RB')
    plt.plot(range(0,mpar['maxlag']),np.zeros((mpar['maxlag'])),'k--' )
    plt.legend(handles=[line1])
    plt.xlabel('Quarter')
    plt.ylabel('Basis Points') 
    f_RB.show()
        
    f_PI = plt.figure(13)
    line1,=plt.plot(range(1,mpar['maxlag']),np.squeeze(np.asarray(IRF_PI)), label='IRF_PI')
    plt.legend(handles=[line1])
    plt.plot(range(0,mpar['maxlag']),np.zeros((mpar['maxlag'])),'k--' )
    plt.xlabel('Quarter')
    plt.ylabel('Basis Points') 
    f_PI.show()

    f_Q = plt.figure(14)
    line1,=plt.plot(range(1,mpar['maxlag']),np.squeeze(np.asarray(IRF_Q)), label='IRF_Q')
    plt.legend(handles=[line1])
    plt.plot(range(0,mpar['maxlag']),np.zeros((mpar['maxlag'])),'k--' )
    plt.xlabel('Quarter')
    plt.ylabel('Basis Points') 
    f_Q.show()

    f_D = plt.figure(15)
    line1,=plt.plot(range(1,mpar['maxlag']),np.squeeze(np.asarray(IRF_D)), label='IRF_D')
    plt.legend(handles=[line1])
    plt.plot(range(0,mpar['maxlag']),np.zeros((mpar['maxlag'])),'k--' )
    plt.xlabel('Quarter')
    plt.ylabel('Basis Points') 
    f_D.show()

    f_LP = plt.figure(16)
    line1,=plt.plot(range(1,mpar['maxlag']-1),np.squeeze(np.asarray(IRF_LP)), label='IRF_LP')
    plt.legend(handles=[line1])
    plt.plot(range(0,mpar['maxlag']),np.zeros((mpar['maxlag'])),'k--' )
    plt.xlabel('Quarter')
    plt.ylabel('Basis Points') 
    f_LP.show()

    f_N = plt.figure(17)
    line1,=plt.plot(range(1,mpar['maxlag']),np.squeeze(np.asarray(IRF_N)), label='IRF_N')
    plt.legend(handles=[line1])
    plt.plot(range(0,mpar['maxlag']),np.zeros((mpar['maxlag'])),'k--' )
    plt.xlabel('Quarter')
    plt.ylabel('Percent') 
    f_N.show()


def FF_1_3(range_, Xss,Yss,Gamma_state,indexMUdct,indexVKdct,par,mpar,grid,targets,Copula,P_H,aggrshock, 
            Fb,packagesize, bl,ss, cc, out_DF1, out_DF3, out_bl):
        
    DF1=np.asmatrix( np.zeros((len(Fb),len(range_))) )
    DF3=np.asmatrix( np.zeros((len(Fb),len(range_))) )
    
    F = lambda S, S_m, C, C_m : Fsys(S, S_m, C, C_m,
                                         Xss,Yss,Gamma_state,indexMUdct,indexVKdct,
                                         par,mpar,grid,targets,Copula,P_H,aggrshock)

    
    for Xct in range_:
        X=np.zeros((mpar['numstates'],1))
        h=par['scaleval1']
        X[Xct]=h
        Fx=F(ss.copy(),X,cc.copy(),cc.copy())
        DF3[:, Xct - bl*packagesize]=(Fx['Difference'] - Fb) / h
        Fx=F(X,ss.copy(),cc.copy(),cc.copy())
        DF1[:, Xct - bl*packagesize]=(Fx['Difference'] - Fb) / h
    if sum(range_ == mpar['numstates'] - 2) == 1:
        Xct=mpar['numstates'] - 2
        X=np.zeros((mpar['numstates'],1))
        h=par['scaleval2']
        X[Xct]=h
        Fx=F(X,ss.copy(),cc.copy(),cc.copy())
        DF1[:,Xct - bl*packagesize]=(Fx['Difference'] - Fb) / h
    if sum(range_ == mpar['numstates'] - 1) == 1:
        Xct=mpar['numstates'] - 1
        X=np.zeros((mpar['numstates'],1))
        h=par['scaleval2']
        X[Xct]=h
        Fx=F(ss.copy(),X,cc.copy(),cc.copy())
        DF3[:,Xct - bl*packagesize]=(Fx['Difference'] - Fb) / h
        Fx=F(X,ss.copy(),cc.copy(),cc.copy())
        DF1[:,Xct - bl*packagesize]=(Fx['Difference'] - Fb) / h
    
    
    out_DF3.put(DF3)
    out_DF1.put(DF1)
    out_bl.put(bl)
    


def FF_2(range_, Xss,Yss,Gamma_state,indexMUdct,indexVKdct,par,mpar,grid,targets,Copula,P_H,aggrshock, 
            Fb,packagesize, bl,ss, cc, out_DF2, out_bl2):
        
    DF2=np.asmatrix(np.zeros((len(Fb),len(range_))))
    
    F = lambda S, S_m, C, C_m : Fsys(S, S_m, C, C_m,
                                         Xss,Yss,Gamma_state,indexMUdct,indexVKdct,
                                         par,mpar,grid,targets,Copula,P_H,aggrshock)

    for Yct in range_:
            Y=np.zeros((mpar['numcontrols'],1))
            h=par['scaleval2']
            Y[Yct]=h
            Fx=F(ss.copy(),ss.copy(),Y,cc.copy())
            DF2[:,Yct - bl*packagesize]=(Fx['Difference'] - Fb) / h
    
    out_DF2.put(DF2)
    out_bl2.put(bl)
    

def SGU_solver(Xss,Yss,Gamma_state,indexMUdct,indexVKdct,par,mpar,grid,targets,Copula,P_H,aggrshock):
    
    out_DF1 = mp.Queue()
    out_DF3 = mp.Queue()
    out_DF2 = mp.Queue()
    out_bl = mp.Queue()
    out_bl2 = mp.Queue()
    
    State       = np.zeros((mpar['numstates'],1))
    State_m     = State.copy()
    Contr       = np.zeros((mpar['numcontrols'],1))
    Contr_m     = Contr.copy()
        
    F = lambda S, S_m, C, C_m : Fsys(S, S_m, C, C_m,
                                         Xss,Yss,Gamma_state,indexMUdct,indexVKdct,
                                         par,mpar,grid,targets,Copula,P_H,aggrshock)
        
      
    start_time = time.clock() 
    result_F = F(State,State_m,Contr.copy(),Contr_m.copy())
    end_time   = time.clock()
    print 'Elapsed time is ', (end_time-start_time), ' seconds.'
    Fb=result_F['Difference'].copy()
        
    pool=cpu_count()/2

    F1=np.zeros((mpar['numstates'] + mpar['numcontrols'], mpar['numstates']))
    F2=np.zeros((mpar['numstates'] + mpar['numcontrols'], mpar['numcontrols']))
    F3=np.zeros((mpar['numstates'] + mpar['numcontrols'], mpar['numstates']))
    F4=np.asmatrix(np.vstack((np.zeros((mpar['numstates'], mpar['numcontrols'])), np.eye(mpar['numcontrols'],mpar['numcontrols']) )))
        
    print 'Use Schmitt Grohe Uribe Algorithm'
    print ' A *E[xprime uprime] =B*[x u]'
    print ' A = (dF/dxprimek dF/duprime), B =-(dF/dx dF/du)'
        
    #numscale=1
    pnum=pool
    packagesize=int(ceil(mpar['numstates'] / float(3.0*pnum)))
    blocks=int(ceil(mpar['numstates'] / float(packagesize) ))

    par['scaleval1'] = 1e-5
    par['scaleval2'] = 1e-5
        
    start_time = time.clock()
    print 'Computing Jacobian F1=DF/DXprime F3 =DF/DX'
    print 'Total number of parallel blocks: ', str(blocks), '.'
    procs1=[]
    for bl in range(0,blocks):
        range_= range(bl*packagesize, min(packagesize*(bl+1),mpar['numstates']))

        cc=np.zeros((mpar['numcontrols'],1))
        ss=np.zeros((mpar['numstates'],1))
        p1=mp.Process(target=FF_1_3, args=(range_, Xss,Yss,Gamma_state,indexMUdct,indexVKdct,par,mpar,
                                           grid,targets,Copula,P_H,aggrshock,Fb,packagesize, bl,ss,cc,
                                           out_DF1,out_DF3, out_bl))
        procs1.append(p1)
        p1.start()
        
        print 'Block number: ', str(bl)    
        
    FF1 = []    
    FF3 = []
    order_bl = []
       
    for i in range(blocks):
        FF1.append(out_DF1.get())
        FF3.append(out_DF3.get())
        order_bl.append(out_bl.get())
        
    
    print 'bl order'
    print order_bl
    
    for p1 in procs1:
        p1.join()
    
    for i in range(0,int(ceil(mpar['numstates'] / float(packagesize)) )):
        range_= range(i*packagesize, min(packagesize*(i+1),mpar['numstates']))
        F1[:,range_]=FF1[order_bl.index(i)].copy()
        F3[:,range_]=FF3[order_bl.index(i)].copy()    
    
    end_time   = time.clock()        
    print 'Elapsed time is ', (end_time-start_time), ' seconds.'
    
    # jacobian wrt Y'
    packagesize=int(ceil(mpar['numcontrols'] / (3.0*pnum)))
    blocks=int(ceil(mpar['numcontrols'] / float(packagesize)))
    print 'Computing Jacobian F2 - DF/DYprime'
    print 'Total number of parallel blocks: ', str(blocks),'.'
        
    start_time = time.clock()
    
    procs2=[]
    for bl in range(0,blocks):
        range_= range(bl*packagesize,min(packagesize*(bl+1),mpar['numcontrols']))

        cc=np.zeros((mpar['numcontrols'],1))
        ss=np.zeros((mpar['numstates'],1))
        p2=mp.Process(target=FF_2, args=(range_, Xss,Yss,Gamma_state,indexMUdct,indexVKdct,par,mpar,
                                         grid,targets,Copula,P_H,aggrshock,Fb,packagesize, bl,ss,cc,
                                         out_DF2, out_bl2))
        procs2.append(p2)
        p2.start()
        print 'Block number: ', str(bl)
        
    FF=[]
    order_bl2 = []
    
    for i in range(blocks):
        FF.append(out_DF2.get())
        order_bl2.append(out_bl2.get()) 
    
    print 'bl2 order'
    print order_bl2
    
    
    for p2 in procs2:
        p2.join()

    for i in range(0,int(ceil(mpar['numcontrols'] / float(packagesize) ))):
        range_=range(i*packagesize, min(packagesize*(i+1),mpar['numcontrols']))
        
        F2[:,range_]=FF[order_bl2.index(i)]

    end_time = time.clock()
    print 'Elapsed time is ', (end_time-start_time), ' seconds.'
          
    FF=[]
    FF1=[]
    FF3=[]
        
    cc=np.zeros((mpar['numcontrols'],1))
    ss=np.zeros((mpar['numstates'],1))
    
    for Yct in range(0, mpar['oc']):
        Y=np.zeros((mpar['numcontrols'],1))
        h=par['scaleval2']
        Y[-1-Yct]=h
        Fx=F(ss.copy(),ss.copy(),cc.copy(),Y)
        F4[:,-1 - Yct]=(Fx['Difference'] - Fb) / h
        
    F2[mpar['nm']+mpar['nk']-3:mpar['numstates']-2,:] = 0

        
   
    s,t,Q,Z=linalg.qz(np.hstack((F1,F2)), -np.hstack((F3,F4)), output='complex')
    abst = abs(np.diag(t))*(abs(np.diag(t))!=0.)+  (abs(np.diag(t))==0.)*10**(-11)
    #relev=np.divide(abs(np.diag(s)), abs(np.diag(t)))
    relev=np.divide(abs(np.diag(s)), abst)    
    
    ll=sorted(relev)
    slt=relev >= 1
    nk=sum(slt)
    slt=1*slt
    

    s_ord,t_ord,__,__,__,Z_ord=linalg.ordqz(np.hstack((F1,F2)), -np.hstack((F3,F4)), sort='ouc', output='complex')
    
    def sortOverridEigen(x, y):
        out = np.empty_like(x, dtype=bool)
        xzero = (x == 0)
        yzero = (y == 0)
        out[xzero & yzero] = False
        out[~xzero & yzero] = True
        out[~yzero] = (abs(x[~yzero]/y[~yzero]) > ll[-1 - mpar['numstates']])
        return out        
    
    if nk > mpar['numstates']:
       if mpar['overrideEigen']:
          print 'Warning: The Equilibrium is Locally Indeterminate, critical eigenvalue shifted to: ', str(ll[-1 - mpar['numstates']])
          slt=relev > ll[-1 - mpar['numstates']]
          nk=sum(slt)
          s_ord,t_ord,__,__,__,Z_ord=linalg.ordqz(np.hstack((F1,F2)), -np.hstack((F3,F4)), sort=sortOverridEigen, output='complex')
          
       else:
          print 'No Local Equilibrium Exists, last eigenvalue: ', str(ll[-1 - mpar['numstates']])
        
    elif nk < mpar['numstates']:
       if mpar['overrideEigen']:
          print 'Warning: No Local Equilibrium Exists, critical eigenvalue shifted to: ', str(ll[-1 - mpar['numstates']])
          slt=relev > ll[-1 - mpar['numstates']]
          nk=sum(slt)
          s_ord,t_ord,__,__,__,Z_ord=linalg.ordqz(np.hstack((F1,F2)), -np.hstack((F3,F4)), sort=sortOverridEigen, output='complex')
          
       else:
          print 'No Local Equilibrium Exists, last eigenvalue: ', str(ll[-1 - mpar['numstates']])


        
        
    z21=Z_ord[nk:,0:nk]
    z11=Z_ord[0:nk,0:nk]
    s11=s_ord[0:nk,0:nk]
    t11=t_ord[0:nk,0:nk]
    
    if matrix_rank(z11) < nk:
       print 'Warning: invertibility condition violated'
              
    z11i  = np.dot(np.linalg.inv(z11), np.eye(nk)) # compute the solution

    gx = np.real(np.dot(z21,z11i))
    hx = np.real(np.dot(z11,np.dot(np.dot(np.linalg.inv(s11),t11),z11i)))
         
    return{'hx': hx, 'gx': gx, 'F1': F1, 'F2': F2, 'F3': F3, 'F4': F4, 'par': par }


###############################################################################

if __name__ == '__main__':
    
    from time import clock
    import pickle
    
    EX3SS=pickle.load(open("EX3SS_30.p", "rb"))
    
    start_time0 = time.clock()        
    
    
    ## Aggregate shock to perturb(one of three shocks: MP, TFP, Uncertainty)
    EX3SS['par']['aggrshock']           = 'MP'
    EX3SS['par']['rhoS']    = 0.0      # Persistence of variance
    EX3SS['par']['sigmaS']  = 0.001    # STD of variance shocks

#    EX3SS['par']['aggrshock']           = 'TFP'
#    EX3SS['par']['rhoS']    = 0.95
#    EX3SS['par']['sigmaS']  = 0.0075
    
#    EX3SS['par']['aggrshock']           = 'Uncertainty'
#    EX3SS['par']['rhoS']    = 0.84    # Persistence of variance
#    EX3SS['par']['sigmaS']  = 0.54    # STD of variance shocks
    

    
    EX3SS['par']['accuracy'] = 0.99999  # accuracy of approximation with DCT
    EX3SR=FluctuationsTwoAsset(**EX3SS)

    SR=EX3SR.StateReduc()

    print 'SGU_solver'
    SGUresult=SGU_solver(SR['Xss'],SR['Yss'],SR['Gamma_state'],SR['indexMUdct'],SR['indexVKdct'],SR['par'],
                         SR['mpar'],SR['grid'],SR['targets'],SR['Copula'],SR['P_H'],SR['aggrshock'])
    print 'plot_IRF'
    plot_IRF(SR['mpar'],SR['par'],SGUresult['gx'],SGUresult['hx'],SR['joint_distr'],
             SR['Gamma_state'],SR['grid'],SR['targets'],SR['Output'])
    
    end_time0 = time.clock()
    print 'Elapsed time is ',  (end_time0-start_time0), ' seconds.'
    
    