# -*- coding: utf-8 -*-
'''
Shared function for HANK
'''
from __future__ import print_function

import numpy as np
import scipy as sc
from scipy.stats import norm
from scipy.interpolate import interp1d, interp2d
from scipy import sparse as sp


def Transition(N,rho,sigma_e,bounds):
    '''
    Calculate transition probability matrix for a given grid for a Markov chain
    with long-run variance equal to 1 and mean 0
    
     Parameters
    ----------
    N : float
        number of states
    rho : float
    sigma_e : float
    bounds : np.array (1,N+1)

    Returns
    ----------
    P : np.array    
        transition matrix
    '''
    pijfunc = lambda x, bound1, bound2 : norm.pdf(x)*(norm.cdf((bound2-rho*x)/sigma_e)-norm.cdf((bound1-rho*x)/sigma_e))
    
    
    P=np.zeros((N,N))
    for i in range(0, int(np.floor((N-1)/2)+1)):
        for j in range(0, N):
        
            pijvalue, err = sc.integrate.quad(pijfunc, bounds[i], bounds[i+1], args=(bounds[j], bounds[j+1]))
            P[i,j]=pijvalue/(norm.cdf(bounds[i+1])-norm.cdf(bounds[i]))
            
             
    P[int(np.floor((N-1)/2)+1):N,:] = P[int(np.ceil((N-1)/2)-1)::-1, ::-1]
    ps=np.sum(P, axis=1)
    
    P=P.copy()/np.transpose(np.tile(ps,(N,1)))
    
    return P


def ExTransitions(S, grid, mpar, par):
    '''
    Generate transition probabilities and grid
    
    Parameters
    ----------
    S : float
        Aggregate exogenous state
    grid : dict
        grid['m']=grid.m : np.array
        grid['h']=grid.h : np.array
        grid['boundsH']=grid.boundsH : np.array (1,mpar['nh'])
    par : dict
        par['xi]=par.xi : float
        par['rhoS']=par.rhoS : float
        par['rhoH']=par.rhoH : float
    mpar : dict
        mpar['nm']=mpar.nm : int
        mpar['nh']=mpar.nh : int   
        mpar['in']=mpar.in : float
        mpar['out']=mpar.out : float   
    
     Returns
     -------
     P_H : np.array
         Transition probabilities
     grid : dict
         Grid
     par : dict
         Parameters
    '''
    
    aux = np.sqrt(S) * np.sqrt(1-par['rhoH']**2)
    
    P = Transition(mpar['nh']-1, par['rhoH'], aux, grid['boundsH'].copy())
    
    P_H = np.concatenate((P, np.tile(mpar['in'],(int(mpar['nh']-1),1))), axis=1)
    lastrow = np.concatenate((np.zeros((1,mpar['nh']-1)), [[1-mpar['out']]]), axis=1)
    lastrow[0,int(np.ceil(mpar['nh']/2))-1] = mpar['out']
    P_H = np.concatenate((P_H.copy(),lastrow.copy()),axis=0)
    P_H = P_H.copy()/np.transpose(np.tile(np.sum(P_H, axis=1),(mpar['nh'],1)))
    
    return {'P_H': P_H, 'grid': grid, 'par': par}


def GenWeight(x,xgrid):
    '''
    Generate weights and indexes used for linear interpolation
    (no extrapolation allowed)
    
    Parameters
    ----------
    x: np.array
        Points at which function is to be interpolated 
    xgrid: np.array
        grid points at which function is measured
        
    Returns
    -------
    weight : np.array
        weight for each index
    index : np.array
        index for integration    
    '''
    
    index = np.digitize(x, xgrid)-1
    index[x <= xgrid[0]] = 0
    index[x >= xgrid[-1]] = len(xgrid)-2
        
    weight = (x-xgrid[index])/(xgrid[index+1]-xgrid[index]) # weight xm of higher gridpoint
    weight[weight.copy()<=0] = 10**(-16) # no extrapolation
    weight[weight.copy()>=1] = 1-10**(-16)
    
    return {'weight': weight, 'index': index}

def MakeGrid2(mpar, grid, m_min, m_max):
    '''
    Make a double log grid
    
    Parameters
    ----------
    mpar : dict
        mpar['nm']=mpar.nm : int
    grid : dict
        grid['m']=grid.m : np.array
    m_min : float
    m_max : float
            
    
    Returns
    -------
    grid : np.array
        new grid
    '''
    
    grid['m'] = np.exp(np.exp(np.linspace(0., np.log(np.log(m_max - m_min +1)+1), mpar['nm']-1))-1)-1+m_min
    
    # grid['m'][np.abs(grid['m'])==np.min(np.abs(grid['m']))]=0.
    
    grid['m'] = np.sort(np.append(grid['m'],0.))
    
    return grid

def MakeGridkm(mpar, grid, k_min, k_max, m_min, m_max):
    '''
    Make a quadruble log grid
    
    Parameters
    ----------
    mpar : dict
        mpar['nm']=mpar.nm : int
    grid : dict
        grid['m']=grid.m : np.array
    k_min : float
    k_max : float
    m_min : float
    m_max : float        
    
    Returns
    -------
    grid : np.array
        new grid
    '''
    grid['k'] = np.exp(np.linspace(0., np.log(k_max - k_min +1.), mpar['nk']))-1 + k_min # set up quadruple exponential grid
    
    grid['m'] = np.exp(np.exp(np.linspace(0., np.log(np.log(m_max - m_min +1)+1), mpar['nm']-1))-1)-1+m_min
    
    grid['m'] = np.sort(np.append(grid['m'],0.))
    
    return grid

def Tauchen(rho, N, sigma, mue, types):
    '''
    Generates a discrete approximation to an AR 1 process following Tauchen(1987)
    
    Parameters
    ----------
    rho : float
        coefficient for AR1
    N : int
        number of gridpoints
    sigma : float
        long-run variance
    mue :  float   
        mean of AR1 process
    types : string
        grid transition generation alogrithm
        'importance' : importance sampling (Each bin has probability 1/N to realize)
        'equi' : bin-centers are equi-spaced between +-3 std
        'simple' : like equi + Transition Probabilities are calculated without using integrals
        'simple importance' : like simple but with grid from importance
        
    return
    -----------
    grid : np.array
        grid 
    P : np.array
        Markov probability
    bounds : np.array
        bounds
    
    '''
    pijfunc = lambda x, bound1, bound2 : norm.pdf(x)*(norm.cdf((bound2-rho*x)/sigma_e)-norm.cdf((bound1-rho*x)/sigma_e))
    
    if types in {'importance','equi','simple','simple importance'}:
       types = types
    else:
       types = 'importance'
       print('Warning: TAUCHEN:NoOpt','No valid type set. Importance sampling used instead')
       
    if types == 'importance': # Importance sampling
           
       grid_probs = np.linspace(0,1,N+1) 
       bounds = norm.ppf(grid_probs)
       
       # replace (-)Inf bounds by finite numbers
       bounds[0] = bounds[1].copy()-99
       bounds[-1] = bounds[-2].copy()+99
        
       # Calculate grid - centers
       grid = 1*N*( norm.pdf(bounds[:-1]) - norm.pdf(bounds[1:]))
      
       sigma_e = np.sqrt(1-rho**2) # Calculate short run variance
       P=np.zeros((N,N))
    
       for i in range( int(np.floor((N-1)/2+1)) ): # Exploit symmetrie
          for j in range(N):
              pijvalue, err = sc.integrate.quad(pijfunc, bounds[i], bounds[i+1], args=(bounds[j], bounds[j+1]),epsabs=10**(-6))
              P[i,j] = N*pijvalue
              
       
       P[int(np.floor((N-1)/2)+1):N,:] = P[int(np.ceil((N-1)/2))-1::-1,::-1].copy()
       
    elif types == 'equi': # use +-3 std equi-spaced grid
        # Equi-spaced
        step = 6/(N-1)
        grid = np.range(-3.,3+step,step)
        
        bounds = np.concatenate(([-99],grid[:-1].copy()+step/2,[99]),axis=1)
        sigma_e = np.sqrt(1-rho**2) # calculate short run variance
        P=np.zeros((N,N))
        
       
        for i in range( int(np.floor((N-1)/2+1)) ): # Exploit symmetrie
          for j in range(N):
              pijvalue, err = sc.integrate.quad(pijfunc, bounds[i], bounds[i+1], args=(bounds[j], bounds[j+1]))
              P[i,j] = pijvalue/(norm.cdf(bounds[i])-norm.cdf(bounds[i-1]))
       
        P[int(np.floor((N-1)/2)+1):N,:] = P[int(np.ceil((N-1)/2))-1::-1,::-1].copy()
        
    elif types == 'simple': # use simple transition probabilities
        
        step = 12/(N-1)
        grid = np.range(-6.,6+step, step)
        bounds=[]
        sigma_e = np.sqrt(1-rho**2)
        P=np.zeros((N,N))
        
                
        for i in range(N):
            P[i,0] = norm.cdf((grid[0]+step/2-rho*grid[i])/sigma_e)
            P[i,-1] = 1- norm.cdf((grid[-1]+step/2-rho*grid[i])/sigma_e)
            for j in range(1,N-1):
                P[i,j] = norm.cdf((grid[j]+step/2-rho*grid[i])/sigma_e) - norm.cdf((grid[j]-step/2-rho*grid[i])/sigma_e)
                
    elif types == 'simple importance': # use simple transition probabilities
        
        grid_probs = np.linspace(0.,1.,N+1)            
        bounds = norm.ppd(grid_probs.copy())
        
        # calculate grid - centers
        grid = N*(norm.pdf(bounds[:-1])-norm.pdf(bounds[1:]))
        
        #replace -Inf bounds by finite numbers
        bounds[0] = bounds[1] - 99
        bounds[-1] = bounds[-2] + 99
        
        sigma_e = np.sqrt(1-rho**2)
        P=np.zeros((N,N))
        
        for i in range(int(np.floor((N-1)/2))+1):
            P[i,0] = norm.cdf((bounds[1]-rho*grid[i])/sigma_e)
            P[i,-1] = 1- norm.cdf((bounds[-2]-rho*grid[i])/sigma_e)
            for j in range(int(np.floor((N-1)/2))+1):
                P[i,j] = norm.cdf((bounds[j+1]-rho*grid[i])/sigma_e) -norm.cdf((bounds[j]-rho*grid[i])/sigma_e)
            
        P[int(np.floor((N-1)/2))+1:,:] = P[int(np.ceil((N-1)/2))-1::-1,::-1].copy()
    
    
    ps = np.sum(P,axis=1)
    P=P.copy()/np.transpose(np.tile(ps.copy(),(N,1)))
    
    grid = grid.copy()*np.sqrt(sigma) + mue
        
    return {'grid': grid, 'P':P, 'bounds':bounds}

def Fastroot(xgrid, fx):
        
        # fast linear interpolation root finding
        # (=one Newton step at largest negative function value)
        # stripped down version of interp1 that accepts multiple inputs (max 3)
        # that are interpolated over the same grids x & xi
    xgrid = xgrid.flatten(order='F')
    fx = np.reshape( fx, (np.size(xgrid), np.size(fx)//np.size(xgrid)), order='F' )

    dxgrid = np.diff(xgrid)
    dfx = np.diff(fx,axis=0)
    idx = np.zeros((1, np.size(fx)//np.size(xgrid))).T

        # Make use of the fact that the difference equation is monotonically
        # increasing in m
    idx_min = (fx[0,:]>0) # Corner solutions left (if no solution x* to f(x)=0 exists)
    idx_max = (fx[-1,:]<0) # Corner solutions right (if no solution x* to f(x)=0 exists)
    index = np.squeeze(np.asarray((np.where((~idx_min) & (~idx_max))))) # interior solutions (if solution x* to f(x)=0 exists)

    # Find index of two gridpoints where sign of fx changes from positive to negative,
    idx[index] = np.asmatrix(np.argmax(np.diff(np.sign(fx[:,index]),axis=0),axis=0)).T
    
        
    aux_index = np.asmatrix(np.arange(0, np.size(fx)//np.size(xgrid), 1)*np.size(xgrid)).T # aux for linear indexes
    aux_index2 = np.arange(0, np.size(fx)//np.size(xgrid), 1)*(np.size(xgrid)-1)
    fx=fx.flatten(order='F')    
    fxx = fx[idx.astype(int)+aux_index].T
    xl = xgrid[idx.astype(int)].T
    dx = dxgrid[idx.astype(int)].T
    dfx=dfx.flatten(order='F')
    dfxx = dfx[idx.astype(int).T+np.asmatrix(aux_index2)]
    # Because function is piecewise linear in gridpoints, one newton step is enough to find the solution
    dfxx[dfxx.copy()== 0.] = 10**(-20) 
    roots = xl - fxx*dx/dfxx

    roots[np.asmatrix(idx_min)] = xgrid[0] # constrained choice
    roots[np.asmatrix(idx_max)] = xgrid[-1] # no-extrapolation
    
    return roots
    