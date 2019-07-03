# -*- coding: utf-8 -*-
'''
Classes to solve the steady state of One asset IOUs model
'''
from __future__ import print_function


import sys 
sys.path.insert(0,'../')

import numpy as np
import scipy as sc
from scipy.stats import norm 
from scipy.interpolate import interp1d, interp2d
from scipy import sparse as sp
import time
from SharedFunc import Transition, ExTransitions, GenWeight, MakeGrid, Tauchen


class SteadyStateOneAssetIOU:

    '''
    Classes to solve the steady state of One asset IOUs model
    '''

    def __init__(self, par, mpar, grid):
         
        self.par = par
        self.mpar = mpar
        self.grid = grid
       
    def SolveSteadyState(self):
        '''
        solve for steady state
        
        returns
        ----------
         par : dict
             parametres
         mpar : dict
             parametres
         grid: dict
             grid for solution
         Output : float
             steady state output  
         targets : dict
            steady state stats
         Vm : np.array
            marginal value of assets m
         joint_distr : np.array
            joint distribution of m and h
         Copula : function
            interpolated function of joint distribution
         c_policy : np.array
            policy function for consumption
         m_policy : np.array   
            policy function for asset m
         mutil_c : np.array
            marginal utility of c
         P_H : np.array
            transition probability    
            
        
        '''
       
        ## Set grids
        grid = MakeGrid(self.mpar, self.grid)
        resultStVar=self.StochasticsVariance(self.par, self.mpar, grid)
       
        P_H = resultStVar['P_H'].copy()
        grid = resultStVar['grid'].copy()
        par = resultStVar['par'].copy()
       
        # Solve for steady state
        rmax = 1./par['beta']-1.
        rmin = -0.03
        r = (rmax+rmin)/2 # initial r for bisection
       
        par['RB'] = 1+r
        init = 999. 
       
        meshesm, meshesh =  np.meshgrid(grid['m'],grid['h'],indexing='ij')
        meshes = {'m': meshesm, 'h': meshesh}
        count =0
        
        while np.abs(init) > self.mpar['crit']:
            resultFactReturn = self.FactorReturns(meshes, grid, par, self.mpar)
           
            N = resultFactReturn['N']
            W_fc = resultFactReturn['w']
            Profits_fc =resultFactReturn['Profits_fc']
            WW = resultFactReturn['WW'].copy()
            RBRB = resultFactReturn['RBRB'].copy()
            Output = resultFactReturn['Y']
           
            resultPolGuess = self.PolicyGuess(meshes, WW, RBRB,par,self.mpar)
            c_guess = resultPolGuess['c_guess'].copy()
            inc = resultPolGuess['inc'].copy()
            count += 1
           
            print(count)
            # Solve policies and Joint distribution
            print('Solving household problem by EGM')
            start_time = time.clock()
           
            resultPolicySS = self.PoliciesSS(c_guess, grid, inc, RBRB, P_H, self.mpar, par)
            c_guess = resultPolicySS['c_new'].copy()
            m_star = resultPolicySS['m_star'].copy()
            distPOL = resultPolicySS['distPOL']
           
            end_time = time.clock()
            print('Elapsed time is ',  (end_time-start_time), ' seconds.')
           
            print(distPOL)
            print('Calc Joint Distr')
            joint_distr = self.JDiteration(m_star, P_H, self.mpar, grid)
            
            joint_distr = np.reshape(joint_distr.copy(),(self.mpar['nm'],self.mpar['nh']),order='F')
            AggregateSavings = m_star.flatten('F').transpose().dot(joint_distr.flatten('F').transpose())
            ExcessA = AggregateSavings
            
            # Use Bisection to update r
            if ExcessA >0:
                rmax = (r+rmax)/2.
            else: rmin = (r+rmin)/2.
            
            init = rmax -rmin
            print('Starting Iteration for r. Difference remaining:                      ')
            print(init)
            r= (rmax+rmin)/2.
            par['RB']=1.+r
                        
            
        print(par['RB'])        
        ## SS_stats
        tgSB = np.sum((grid['m']<0)*np.sum(joint_distr.copy(),1))
        tgB = grid['m'].copy().dot(np.sum(joint_distr.copy(),1))
        tgBY = tgB/Output
        BCaux_M = np.sum(joint_distr,1) # 
        tgm_bc = BCaux_M[0,:]
        tgm_0 = BCaux_M[grid['m']==0]
        
        labortax = (1.-par['tau'])*W_fc*N +(1.-par['tau'])*Profits_fc
        par['gamma1']=tgB*(1.-par['RB'])+labortax
        
        par['W']=W_fc
        par['PROFITS'] = Profits_fc
        par['N'] = N
        tgGtoY = par['gamma1']/Output
        tgT = labortax
        
        targets = {'ShareBorrower' : tgSB,
                   'B': tgB,
                   'BY': tgBY,
                   'Y': Output,
                   'm_bc': tgm_bc,
                   'm_0': tgm_0,
                   'GtoY': tgGtoY,
                   'T': tgT}   
        
        ## Prepare state and controls        
        grid['B'] = np.sum(grid['m']*np.sum(joint_distr,1))
        
        # Calculate Marginal Values of Assets (m)
        RBRB = (par['RB']+(meshes['m'].copy()<0)*par['borrwedge'])/par['PI']
        
        # Liquid Asset
        mutil_c = 1./(c_guess.copy()**par['xi'])
        Vm = RBRB.copy()*mutil_c.copy()
        Vm = np.reshape(Vm.copy(),(self.mpar['nm'],self.mpar['nh']))
        
        
        ## Produce non-parametric Copula
        cum_dist = np.cumsum(np.cumsum(joint_distr,axis=0),axis=1)
        marginal_m = np.cumsum(np.squeeze(np.sum(joint_distr,axis=1)))
        marginal_h = np.cumsum(np.squeeze(np.sum(joint_distr,axis=0)))
        
        
        Copula = interp2d(marginal_m.copy(), marginal_h.copy(), cum_dist.copy().transpose(),kind='cubic')
        
       
        return {'par':par,
                'mpar':self.mpar,
                'grid':grid,
                'Output':Output,
                'targets':targets,
                'Vm': Vm,
                'joint_distr': joint_distr,
                'Copula': Copula,
                'c_policy': c_guess,
                'm_policy': m_star,
                'mutil_c': mutil_c,
                'P_H' : P_H
                }
       
    def JDiteration(self, m_star, P_H, mpar, grid):
        '''
        Iterates the joint distribution over m,k,h using a transition matrix
        obtained from the house distributing the households optimal choices. 
        It distributes off-grid policies to the nearest on grid values.
        
        parameters
        ------------
        m_star :np.array
            optimal m func
        P_H : np.array
            transition probability    
        mpar : dict
             parameters    
        grid : dict
             grids
             
        returns
        ------------
        joint_distr : np.array
            joint distribution of m and h
        
        '''
        ## find next smallest on-grid value for money and capital choices
        weight11  = np.zeros((mpar['nm'], mpar['nh'],mpar['nh']))
        weight12  = np.zeros((mpar['nm'], mpar['nh'],mpar['nh']))
    
        # Adjustment case
        resultGW = GenWeight(m_star, grid['m'])
        Dist_m = resultGW['weight'].copy()
        idm = resultGW['index'].copy()
        
        idm = np.transpose(np.tile(idm.flatten(order='F'),(mpar['nh'],1)))
        idh = np.kron(range(mpar['nh']),np.ones((1,mpar['nm']*mpar['nh'])))
        idm = idm.copy().astype(int)
        idh = idh.copy().astype(int)
        
        
        index11 = np.ravel_multi_index([idm.flatten(order='F'), idh.flatten(order='F')],(mpar['nm'],mpar['nh']),order='F')
        index12 = np.ravel_multi_index([idm.flatten(order='F')+1, idh.flatten(order='F')],(mpar['nm'],mpar['nh']),order='F')
        
        
        for hh in range(mpar['nh']):
        
            # Corresponding weights
            weight11_aux = (1.-Dist_m[:,hh].copy())
            weight12_aux =  (Dist_m[:,hh].copy())
    
            # Dimensions (mxk,h',h)   
            weight11[:,:,hh]=np.outer(weight11_aux.flatten(order='F'),P_H[hh,:].copy())
            weight12[:,:,hh]=np.outer(weight12_aux.flatten(order='F'),P_H[hh,:].copy())
        
            
        
        weight11 = np.ndarray.transpose(weight11.copy(),(0,2,1))
        weight12 = np.ndarray.transpose(weight12.copy(),(0,2,1))
        
        rowindex = np.tile(range(mpar['nm']*mpar['nh']),(1,2*mpar['nh']))
        
        
        H = sp.coo_matrix((np.concatenate((weight11.flatten(order='F'),weight12.flatten(order='F'))), 
                       (rowindex.flatten(), np.concatenate((index11.flatten(order='F'),index12.flatten(order='F'))))),shape=(mpar['nm']*mpar['nh'], mpar['nm']*mpar['nh']))
        
        ## Joint transition matrix and transitions
        
        distJD = 9999.
        countJD = 1
    
        eigen, joint_distr = sp.linalg.eigs(H.transpose(), k=1, which='LM')
        joint_distr = joint_distr.copy().real
        joint_distr = joint_distr.copy().transpose()/(joint_distr.copy().sum())
        
        while (distJD > 10**(-14) or countJD<50) and countJD<10000:
        
            joint_distr_next = joint_distr.copy().dot(H.copy().todense())
                                       
            joint_distr_next = joint_distr_next.copy()/joint_distr_next.copy().sum(axis=1)
            distJD = np.max((np.abs(joint_distr_next.copy().flatten()-joint_distr.copy().flatten())))
            
            countJD = countJD +1
            joint_distr = joint_distr_next.copy()
                 
            
        return joint_distr
           
        
    def PoliciesSS(self,c_guess, grid, inc, RBRB, P, mpar, par):
        '''
        solves for the household policies for consumption and bonds by EGM
        
        parameters
        -----------
        c_guess : np.array
               guess for c
        grid : dict
             grids
        inc : dict
            guess for incomes      
        RBRB : float    
            interest rate   
        P : np.array
            transition probability    
        par : dict
             parameters
        mpar : dict
             parameters    
             
        returns
        ----------
        c_new : np.array
            optimal c func
        m_star :np.array
            optimal m func
        distPOL : float
            distance of convergence in functions
        '''
    
        ## Apply EGM to slove for optimal policies and marginal utilities
        money_expense = np.transpose(np.tile(grid['m'],(mpar['nh'],1)))
        distC = 99999.
    
        count = 0
    
        while np.max((distC)) > mpar['crit']:
        
            count = count+1
        
            ## update policies
            mutil_c = 1./(c_guess.copy()**par['xi'])
        
            aux = np.reshape(np.ndarray.transpose(mutil_c.copy(),(1,0)),(mpar['nh'],mpar['nm']) )
        
            # form expectations
            EMU_aux = par['beta']*RBRB*np.ndarray.transpose(np.reshape(P.copy().dot(aux.copy()),(mpar['nh'],mpar['nm'])),(1,0))
        
            c_aux = 1./(EMU_aux.copy()**(1/par['xi']))
        
            # Take budget constraint into account
            resultEGM = self.EGM(grid, inc, money_expense, c_aux, mpar, par)
            c_new = resultEGM['c_update'].copy()
            m_star = resultEGM['m_update'].copy()
        
            m_star[m_star>grid['m'][-1]] = grid['m'][-1] # no extrapolation
        
            ## check convergence of policies
            distC = np.max((np.abs(c_guess.copy()-c_new.copy())))
        
            # update c policy guesses
            c_guess = c_new.copy()
        
        distPOL = distC

        return {'c_new':c_new, 'm_star':m_star, 'distPOL':distPOL}      


    def EGM(self, grid, inc, money_expense, c_aux, mpar, par):
        '''
        computes the optimal consumption and corresponding optimal
        bond holdings by taking the budget constraint into account.
        
        parameters
        -----------
        grid : dict
             grids
        inc : dict
            guess for incomes
        par : dict
             parameters
        mpar : dict
             parameters
        money_expense : np.array
             guess for optimal m holding 
        c_aux : np.array
             guess for optimal consumption
        
        
        return
        -----------
        c_update(m,h) : np.array
                  Update for consumption policy 
        m_update(m,h) : np.array
                  Update for bond policy 
        '''    
    
        ## EGM: Calculate assets consistent with choices being (m')
        # Calculate initial money position from the budget constraint,
        # that leads to the optimal consumption choice
        m_star = c_aux + money_expense - inc['labor'] -inc['profits']
        RR = (par['RB']+(m_star.copy()<0.)*par['borrwedge'])/par['PI']
        m_star = m_star.copy()/RR
    
        # Identify binding constraints
        binding_constraints = (money_expense < np.tile(m_star[0,:],(mpar['nm'], 1)))

        # Consumption when drawing assets m' to zero: Eat all Resources
        Resource = inc['labor']  + inc['money'] + inc['profits']
    
        ## Next step : interpolate on grid
        c_update = np.zeros((mpar['nm'],mpar['nh']))
        m_update = np.zeros((mpar['nm'],mpar['nh']))
    
        for hh in range(mpar['nh']):
        
            Savings = interp1d(m_star[:,hh].copy(), grid['m'], fill_value='extrapolate')
            m_update[:,hh] = Savings(grid['m'])
            Consumption = interp1d(m_star[:,hh].copy(), c_aux[:,hh], fill_value='extrapolate')
            c_update[:,hh] = Consumption(grid['m'])
        
        c_update[binding_constraints] = Resource[binding_constraints]-grid['m'][0]
        m_update[binding_constraints] = np.min((grid['m']))    
    
    
        return {'c_update': c_update, 'm_update': m_update}
       
    def PolicyGuess(self, meshes, WW, RBRB, par, mpar):
        '''
        autarky policy guesses 
        
        parameters
        -----------
        meshes : dict
             meshes for m and h
        par : dict
             parameters
        mpar : dict
             parameters
        WW : np.array
            wage for each m and h
        RBRB : float    
            interest rate
            
        returns
        -----------
        c_guess : np.array
            guess for c func
        inc : dict
            guess for incomes
        '''
        inclabor = par['tau']*WW*meshes['h'].copy()
        incmoney = RBRB*meshes['m'].copy()
        incprofits = 0.
    
        inc = {'labor': inclabor.copy(),
               'money': incmoney.copy(),
               'profits': incprofits}
    
#       c_guess = inc['labor'].copy()+np.maximum(inc['money'],0.) + inc['profits']
        c_guess = inc['labor'].copy()+np.maximum(inc['money'].all(),0.) + inc['profits']

        return {'c_guess':c_guess, 'inc': inc}       
         
    def FactorReturns(self, meshes, grid, par, mpar):
        '''
        return factors for steady state
        
        parameters
        -----------
        meshes : dict
             meshes for m and h
        par : dict
             parameters
        mpar : dict
             parameters
        grid : dict
             grids
             
        returns
        ----------
        N : float
           aggregate labor
        w : float
            wage   
        Profits_fc : float
            profit of firm
        WW : np.array
            wage for each m and h
        RBRB : float    
            interest rate
        '''
        ## GHH preferences
    
        mc = par['mu'] # in SS
        N = (par['tau']*par['alpha']*grid['K']**(1-par['alpha'])*mc)**(1/(1-par['alpha']+par['gamma']))
        w = 1./4. * par['alpha'] *mc * (grid['K']/N) **(1-par['alpha'])
    
        Y = 0.25*N**par['alpha']*grid['K']**(1-par['alpha'])
        Profits_fc = (1-mc)*Y
    
        NW = par['gamma']/(1.+par['gamma'])*N/par['H'] *w
    
        WW = NW*np.ones((mpar['nm'],mpar['nh'])) # Wages
        WW[:,-1] = Profits_fc * par['profitshare']
        RBRB = (par['RB']+(meshes['m']<0)*par['borrwedge'])/par['PI']
    
        return {'N':N, 'w':w, 'Profits_fc':Profits_fc,'WW':WW,'RBRB':RBRB,'Y':Y}


    def StochasticsVariance(self, par, mpar, grid):
        '''
        generates transition probabilities for h: P_H
        
        parameters
        -------------
        par : dict
             parameters
        mpar : dict
             parameters
        grid : dict
             grids
             
        return
        -----------
        P_H : np.array
            transition probability
        grid : dict
            grid
        par : dict
            parameters
        '''
    
        # First for human capital
        TauchenResult = Tauchen(par['rhoH'], mpar['nh']-1, 1., 0., mpar['tauchen'])
        hgrid = TauchenResult['grid'].copy()
        P_H = TauchenResult['P'].copy()
        boundsH = TauchenResult['bounds'].copy()
    
        # correct long run variance for human capital
        hgrid = hgrid.copy()*par['sigmaH']/np.sqrt(1-par['rhoH']**2)
        hgrid = np.exp(hgrid.copy()) # levels instead of logs
    
        grid['h'] = np.concatenate((hgrid,[1]), axis=0)
        
        P_H = Transition(mpar['nh']-1, par['rhoH'], np.sqrt(1-par['rhoH']**2), boundsH)
    
        # Transitions to enterpreneur state
        P_H = np.concatenate((P_H.copy(),np.tile(mpar['in'],(mpar['nh']-1,1))), axis=1)
        lastrow = np.concatenate((np.tile(0.,(1,mpar['nh']-1)),[[1-mpar['out']]]), axis=1)
        lastrow[0,int(np.ceil(mpar['nh']/2))-1] = mpar['out'] 
        P_H = np.concatenate((P_H.copy(),lastrow), axis=0)
        P_H = P_H.copy()/np.transpose(np.tile(np.sum(P_H.copy(),1),(mpar['nh'],1)))
    
        Paux = np.linalg.matrix_power(P_H.copy(),1000)
        hh = Paux[0,:mpar['nh']-1].copy().dot(grid['h'][:mpar['nh']-1].copy())
    
        par['H'] = hh # Total employment
        par['profitshare'] = Paux[-1,-1]**(-1) # Profit per household
        grid['boundsH'] = boundsH
     
    
        return {'P_H': P_H, 'grid':grid, 'par':par}


###############################################################################

if __name__ == '__main__':
    
    import defineSSParameters as Params
    from copy import copy
    import pickle
    
    EX1param = copy(Params.parm_one_asset_IOU)
            
    EX1 = SteadyStateOneAssetIOU(**EX1param)
    #print(EX1.par)
    EX1SS = EX1.SolveSteadyState()
    
#   pickle.dump(EX1SS, open("EX1SS.p", "wb"))
    pickle.dump(EX1SS, open("EX1SS_nm50.p", "wb"))