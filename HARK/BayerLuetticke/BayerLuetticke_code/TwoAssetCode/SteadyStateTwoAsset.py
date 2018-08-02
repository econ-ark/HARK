# -*- coding: utf-8 -*-
'''
Classes to solve the steady state of liquid and illiquid assets model
'''
from __future__ import print_function


import sys 
sys.path.insert(0,'../')

import numpy as np
import scipy as sc
from scipy.stats import norm 
from scipy.interpolate import interp1d, interp2d, griddata, RegularGridInterpolator
from scipy import sparse as sp
import time
from SharedFunc3 import Transition, ExTransitions, GenWeight, MakeGridkm, Tauchen, Fastroot


class SteadyStateTwoAsset:

    '''
    Classes to solve the steady state of liquid and illiquid assets model
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
         Vk : np.array
            marginal value of assets m
         joint_distr : np.array
            joint distribution of m and h
         Copula : dict
            points for interpolation of joint distribution
         c_a_star : np.array
            policy function for consumption w/ adjustment
         c_n_star : np.array
            policy function for consumption w/o adjustment
         psi_star : np.array
            continuation value of holding capital
         m_a_star : np.array   
            policy function for asset m w/ adjustment
         m_n_star : np.array   
            policy function for asset m w/o adjustment
         mutil_c_a : np.array
            marginal utility of c w/ adjustment
         mutil_c_n : np.array
            marginal utility of c w/o adjustment   
         mutil_c : np.array
            marginal utility of c w/ & w/o adjustment   
         P_H : np.array
            transition probability    
            
        
        '''
       
        ## Set grid h
        grid = self.grid
        resultStVar=self.StochasticsVariance(self.par, self.mpar, grid)
       
        P_H = resultStVar['P_H'].copy()
        grid = resultStVar['grid'].copy()
        par = resultStVar['par'].copy()
       
        grid = MakeGridkm(self.mpar, grid, grid['k_min'], grid['k_max'], grid['m_min'], grid['m_max'])
        meshes = {}
        meshes['m'], meshes['k'], meshes['h'] =  np.meshgrid(grid['m'],grid['k'],grid['h'],indexing='ij')
        
        ## Solve for steady state capital by bi-section
        result_SS = self.SteadyState(P_H, grid, meshes, self.mpar, par)
        
        c_n_guess = result_SS['c_n_guess'].copy()
        m_n_star = result_SS['m_n_star'].copy()
        c_a_guess = result_SS['c_a_guess'].copy()
        m_a_star = result_SS['m_a_star'].copy()
        cap_a_star = result_SS['cap_a_star'].copy()
        psi_guess = result_SS['psi_guess'].copy()
        joint_distr = result_SS['joint_distr'].copy()
        
        R_fc = result_SS['R_fc']
        W_fc = result_SS['W_fc']
        Profits_fc = result_SS['Profits_fc']
        Output = result_SS['Output']
        grid = result_SS['grid'].copy()
        
        ## SS stats       
        mesh ={}
        mesh['m'],mesh['k'] =np.meshgrid(grid['m'].copy(),grid['k'].copy(), indexing = 'ij')
        
        targets = {}
        targets['ShareBorrower'] = np.sum((grid['m']<0)*np.transpose(np.sum(np.sum(joint_distr.copy(),axis = 1), axis = 1)))
        targets['K'] = np.sum(grid['k'].copy()*np.sum(np.sum(joint_distr.copy(),axis =0),axis=1))
        targets['B'] = np.dot(grid['m'].copy(),np.sum(np.sum(joint_distr.copy(),axis = 1),axis = 1))
        grid['K'] = targets['K']
        grid['B'] = targets['B']

        JDredux = np.sum(joint_distr.copy(),axis =2)
        targets['BoverK']     = targets['B']/targets['K']

        targets['L'] = grid['N']*np.sum(np.dot(grid['h'].copy(),np.sum(np.sum(joint_distr.copy(),axis=0),axis=0)))
        targets['KY'] = targets['K']/Output
        targets['BY'] = targets['B']/Output
        targets['Y'] = Output
        BCaux_M = np.sum(np.sum(joint_distr.copy(),axis =1), axis=1)
        targets['m_bc'] = BCaux_M[0].copy()
        targets['m_0'] = float(BCaux_M[grid['m']==0].copy())
        BCaux_K = np.sum(np.sum(joint_distr.copy(),axis=0),axis=1)
        targets['k_bc'] = BCaux_K[0].copy()
        aux_MK = np.sum(joint_distr.copy(),axis=2)
        
        targets['WtH_b0']=np.sum(aux_MK[(mesh['m']==0)*(mesh['k']>0)].copy())
        targets['WtH_bnonpos']=np.sum(aux_MK[(mesh['m']<=0)*(mesh['k']>0)].copy())

        targets['T'] =(1.0-par['tau'])*W_fc*grid['N'] +(1.0-par['tau'])*Profits_fc
        par['G']=targets['B']*(1.0-par['RB']/par['PI'])+targets['T']
        par['R']=R_fc
        par['W']=W_fc
        par['PROFITS']=Profits_fc
        par['N']=grid['N']
        targets['GtoY']=par['G']/Output

        ## Ginis
        # Net worth Gini
        mplusk=mesh['k'].copy().flatten('F')*par['Q']+mesh['m'].copy().flatten('F')
        
        IX = np.argsort(mplusk.copy())
        mplusk = mplusk[IX.copy()].copy()
        
        moneycapital_pdf   = JDredux.flatten(order='F')[IX].copy()
        moneycapital_cdf   = np.cumsum(moneycapital_pdf.copy())
        targets['NegNetWorth']= np.sum((mplusk.copy()<0)*moneycapital_pdf.copy())

        S                  = np.cumsum(moneycapital_pdf.copy()*mplusk.copy())
        
        S                  = np.concatenate(([0.], S.copy()))
        targets['GiniW']      = 1.0-(np.sum(moneycapital_pdf.copy()*(S[:-1].copy()+S[1:].copy()).transpose())/S[-1])

        # Liquid Gini
        IX = np.argsort(mesh['m'].copy().flatten('F'))
        liquid_sort = mesh['m'].copy().flatten('F')[IX.copy()].copy()
        liquid_pdf         = JDredux.flatten(order='F')[IX.copy()].copy()
        liquid_cdf         = np.cumsum(liquid_pdf.copy())
        targets['Negliquid']  = np.sum((liquid_sort.copy()<0)*liquid_pdf.copy())

        S                  = np.cumsum(liquid_pdf.copy()*liquid_sort.copy())
        S                  = np.concatenate(([0.], S.copy()))
        targets['GiniLI']      = 1.0-(np.sum(liquid_pdf.copy()*(S[:-1].copy()+S[1:].copy()))/S[-1].copy())

        # Illiquid Gini
        IX = np.argsort(mesh['k'].copy().flatten('F'))
        illiquid_sort = mesh['k'].copy().flatten('F')[IX.copy()].copy()
        illiquid_pdf        = JDredux.flatten(order='F')[IX.copy()].copy()
        illiquid_cdf        = np.cumsum(illiquid_pdf.copy());
        targets['Negliquid']   = np.sum((illiquid_sort.copy()<0)*illiquid_pdf.copy())

        S                   = np.cumsum(illiquid_pdf.copy()*illiquid_sort.copy())
        S                   = np.concatenate(([0.], S.copy()))
        targets['GiniIL']      = 1.-(np.sum(illiquid_pdf.copy()*(S[:-1].copy()+S[1:].copy()))/S[-1].copy())

        ##   MPCs
        meshesm, meshesk, meshesh =  np.meshgrid(grid['m'],grid['k'],grid['h'],indexing='ij')

        NW = par['gamma']/(1.+par['gamma'])*(par['N']/par['H'])*par['W']
        WW = NW*np.ones((self.mpar['nm'],self.mpar['nk'],self.mpar['nh']))  # Wages
        WW[:,:,-1]=par['PROFITS']*par['profitshare']
        # MPC
        WW_h=np.squeeze(WW[0,0,:].copy().flatten('F'))
        WW_h_mesh=np.squeeze(WW.copy()*meshes['h'].copy())

        grid_h_aux=grid['h']

        MPC_a_m = np.zeros((self.mpar['nm'],self.mpar['nk'],self.mpar['nh']))
        MPC_n_m = np.zeros((self.mpar['nm'],self.mpar['nk'],self.mpar['nh']))

        for kk in range(0 ,self.mpar['nk']) :
           for hh in range(0, self.mpar['nh']) :
            MPC_a_m[:,kk,hh]=np.gradient(np.squeeze(c_a_guess[:,kk,hh].copy()))/np.gradient(grid['m'].copy()).transpose()
            MPC_n_m[:,kk,hh]=np.gradient(np.squeeze(c_n_guess[:,kk,hh].copy()))/np.gradient(grid['m'].copy()).transpose()
    

        MPC_a_m = MPC_a_m.copy()*(WW_h_mesh.copy()/c_a_guess.copy())
        MPC_n_m = MPC_n_m.copy()*(WW_h_mesh.copy()/c_n_guess.copy())

        MPC_a_h = np.zeros((self.mpar['nm'],self.mpar['nk'],self.mpar['nh']))
        MPC_n_h = np.zeros((self.mpar['nm'],self.mpar['nk'],self.mpar['nh']))

        for mm in range(0, self.mpar['nm']) :
           for kk in range(0, self.mpar['nk']) :
             MPC_a_h[mm,kk,:] = np.gradient(np.squeeze(np.log(c_a_guess[mm,kk,:].copy())))/np.gradient(np.log(WW_h.copy().transpose()*grid_h_aux.copy())).transpose()
             MPC_n_h[mm,kk,:] = np.gradient(np.squeeze(np.log(c_n_guess[mm,kk,:].copy())))/np.gradient(np.log(WW_h.copy().transpose()*grid_h_aux.copy())).transpose()
    

        EMPC_h = np.dot(joint_distr.copy().flatten('F'),(par['nu']*MPC_a_h.copy().flatten('F')+(1.-par['nu'])*MPC_n_h.copy().flatten('F')))
        EMPC_m = np.dot(joint_distr.copy().flatten('F'),(par['nu']*MPC_a_m.copy().flatten('F')+(1.-par['nu'])*MPC_n_m.copy().flatten('F')))

        EMPC_a_h = np.dot(joint_distr.copy().flatten('F'), MPC_a_h.copy().flatten('F'))
        EMPC_a_m = np.dot(joint_distr.copy().flatten('F'), MPC_a_m.copy().flatten('F'))

        EMPC_n_h = np.dot(joint_distr.copy().flatten('F'), MPC_n_h.copy().flatten('F'))
        EMPC_n_m = np.dot(joint_distr.copy().flatten('F'), MPC_n_m.copy().flatten('F'))

        targets['Insurance_coeff']=np.concatenate((np.concatenate(([[1.-EMPC_h]], [[1.-EMPC_m]]), axis =1),
                                                np.concatenate(([[1.-EMPC_a_h]],[[ 1.-EMPC_a_m]]), axis =1),
                                                np.concatenate(([[1.-EMPC_n_h]], [[1.-EMPC_n_m]]), axis =1)) , axis =0)
                         
                         
                         
        
        ## Calculate Value Functions
        # Calculate Marginal Values of Capital (k) and Liquid Assets(m)
        RBRB = par['RB']/par['PI'] + (meshes['m']<0)*(par['borrwedge']/par['PI'])

        # Liquid Asset
        mutil_c_n = 1./(c_n_guess.copy()**par['xi']) # marginal utility at consumption policy no adjustment
        mutil_c_a = 1./(c_a_guess.copy()**par['xi']) # marginal utility at consumption policy adjustment
        mutil_c = par['nu']*mutil_c_a.copy() + (1-par['nu'])*mutil_c_n.copy() # Expected marginal utility at consumption policy (w &w/o adjustment)
        Vm = RBRB.copy()*mutil_c.copy()  # take return on money into account
        Vm = np.reshape(np.reshape(Vm.copy(),(self.mpar['nm']*self.mpar['nk'], self.mpar['nh']),order ='F'),(self.mpar['nm'],self.mpar['nk'], self.mpar['nh']),order ='F')

        # Capital
        Vk = par['nu']*(par['R']+par['Q'])*mutil_c_a.copy() + (1-par['nu'])*par['R']*mutil_c_n.copy() + (1-par['nu'])*psi_guess.copy() # Expected marginal utility at consumption policy (w &w/o adjustment)
        Vk = np.reshape(np.reshape(Vk.copy(),(self.mpar['nm']*self.mpar['nk'], self.mpar['nh']),order = 'F'),(self.mpar['nm'],self.mpar['nk'], self.mpar['nh']), order='F') 

        ## Produce non-parametric Copula
        cum_dist = np.cumsum(np.cumsum(np.cumsum(joint_distr.copy(), axis=0),axis=1),axis=2)
        marginal_m = np.cumsum(np.squeeze(np.sum(np.sum(joint_distr.copy(),axis=1),axis=1)))
        marginal_k = np.cumsum(np.squeeze(np.sum(np.sum(joint_distr.copy(),axis=0),axis=1)))
        marginal_h = np.cumsum(np.squeeze(np.sum(np.sum(joint_distr.copy(),axis=1),axis=0)))
 
            
   
        Cgridm, Cgridk, Cgridh = np.meshgrid(marginal_m.copy(),marginal_k.copy(),marginal_h.copy(),indexing='ij')
        Cpoints = np.concatenate(( [Cgridm.flatten(order='F')],[Cgridk.flatten(order='F')],[Cgridh.flatten(order='F')]),axis=0).T
        
        
        #Copula_aux = griddata((marginal_m.copy(),marginal_k.copy(),marginal_h.copy()),cum_dist.copy().transpose(),(points,points,points),method ='cubic')
        #Copula_aux = griddata(Cpoints,cum_dist.copy().flatten(order='F'),(points0,points1,points2))
        #Copula = RegularGridInterpolator((spm,spk,sph),np.reshape(Copula_aux,(200,200,20),order='F'),bounds_error = False, fill_value = None)
        Copula ={}
        Copula['grid'] = Cpoints.copy()
        Copula['value'] = cum_dist.flatten(order = 'F').copy()

        
       
        return {'par':par,
                'mpar':self.mpar,
                'grid':grid,
                'Output':Output,
                'targets':targets,
                'Vm': Vm,
                'Vk': Vk,
                'joint_distr': joint_distr,
                'Copula': Copula,
                'c_n_guess': c_n_guess,
                'c_a_guess': c_a_guess,
                'psi_guess': psi_guess,
                'm_n_star': m_n_star,
                'm_a_star': m_a_star,
                'cap_a_star':cap_a_star,
                'mutil_c_n': mutil_c_n,
                'mutil_c_a': mutil_c_a,
                'mutil_c': mutil_c,
                'P_H' : P_H
                }
       
    def JDiteration(self, joint_distr, m_n_star, m_a_star, cap_a_star,P_H, par, mpar, grid):
        '''
        Iterates the joint distribution over m,k,h using a transition matrix
        obtained from the house distributing the households optimal choices. 
        It distributes off-grid policies to the nearest on grid values.
        
        parameters
        ------------
        m_a_star :np.array
            optimal m func
        m_n_star :np.array
            optimal m func
        cap_a_star :np.array
            optimal a func    
        P_H : np.array
            transition probability    
        par : dict
             parameters    
        mpar : dict
             parameters    
        grid : dict
             grids
             
        returns
        ------------
        joint_distr : np.array
            joint distribution of m and h
        
        '''
        ## Initialize matirces
                
        weight11  = np.empty((mpar['nm']*mpar['nk'], mpar['nh'],mpar['nh']))
        weight12  = np.empty((mpar['nm']*mpar['nk'], mpar['nh'],mpar['nh']))
        weight21  = np.empty((mpar['nm']*mpar['nk'], mpar['nh'],mpar['nh']))
        weight22  = np.empty((mpar['nm']*mpar['nk'], mpar['nh'],mpar['nh']))
        
    
        # Find next smallest on-grid value for money and capital choices
        resultGWa = GenWeight(m_a_star, grid['m'])
        resultGWn = GenWeight(m_n_star, grid['m'])
        resultGWk = GenWeight(cap_a_star, grid['k'])
        Dist_m_a = resultGWa['weight'].copy()
        idm_a = resultGWa['index'].copy()
        Dist_m_n = resultGWn['weight'].copy()
        idm_n = resultGWn['index'].copy()
        Dist_k = resultGWk['weight'].copy()
        idk_a = resultGWk['index'].copy()
        idk_n = np.tile(np.ones((mpar['nm'],1))*np.arange(mpar['nk']),(1,1,mpar['nh']))
        
        ## Transition matrix adjustment case
        idm_a = np.tile(idm_a.copy().flatten(order='F'),(1, mpar['nh']))
        idk_a = np.tile(idk_a.copy().flatten(order='F'),(1, mpar['nh']))
        idh = np.kron(np.arange(mpar['nh']),np.ones((1,mpar['nm']*mpar['nk']*mpar['nh'])))
        
        idm_a = idm_a.copy().astype(int)
        idk_a = idk_a.copy().astype(int)
        idh = idh.copy().astype(int)
                
        index11 = np.ravel_multi_index([idm_a.flatten(order='F'), idk_a.flatten(order='F'), idh.flatten(order='F')],(mpar['nm'],mpar['nk'],mpar['nh']),order='F')
        index12 = np.ravel_multi_index([idm_a.flatten(order='F'), idk_a.flatten(order='F')+1, idh.flatten(order='F')],(mpar['nm'],mpar['nk'],mpar['nh']),order='F')
        index21 = np.ravel_multi_index([idm_a.flatten(order='F')+1, idk_a.flatten(order='F'), idh.flatten(order='F')],(mpar['nm'],mpar['nk'],mpar['nh']),order='F')
        index22 = np.ravel_multi_index([idm_a.flatten(order='F')+1, idk_a.flatten(order='F')+1, idh.flatten(order='F')],(mpar['nm'],mpar['nk'],mpar['nh']),order='F')
        
        ## Policy Transition Matrix for no-adjustment case
                
        weight13 = np.empty((mpar['nm']*mpar['nk'],mpar['nh'],mpar['nh']))
        weight23 = np.empty((mpar['nm']*mpar['nk'],mpar['nh'],mpar['nh']))
        
        idm_n = np.tile(idm_n.copy().flatten(order='F'),(1,mpar['nh']))
        idk_n = np.tile(idk_n.copy().flatten(order='F'),(1,mpar['nh']))
        
        idm_n = idm_n.copy().astype(int)
        idk_n = idk_n.copy().astype(int)
        
        index13 = np.ravel_multi_index([idm_n.flatten(order='F'), idk_n.flatten(order='F'), idh.flatten(order='F')],(mpar['nm'],mpar['nk'],mpar['nh']),order='F')
        index23 = np.ravel_multi_index([idm_n.flatten(order='F')+1, idk_n.flatten(order='F'), idh.flatten(order='F')],(mpar['nm'],mpar['nk'],mpar['nh']),order='F')
        
        
        for hh in range(mpar['nh']):
        
            # Corresponding weights
            weight21_aux = Dist_m_a[:,:,hh].copy()*(1.-Dist_k[:,:,hh].copy())
            weight11_aux = (1.-Dist_m_a[:,:,hh].copy())*(1-Dist_k[:,:,hh].copy())
            weight22_aux = Dist_m_a[:,:,hh].copy()*Dist_k[:,:,hh].copy()
            weight12_aux = (1.-Dist_m_a[:,:,hh].copy())*Dist_k[:,:,hh].copy()
            
            weight23_aux = Dist_m_n[:,:,hh].copy()
            weight13_aux = (1.-Dist_m_n[:,:,hh].copy())
    
            # Dimensions (mxk,h',h)   
            weight11[:,:,hh]=np.outer(weight11_aux.flatten(order='F'),P_H[hh,:].copy())
            weight12[:,:,hh]=np.outer(weight12_aux.flatten(order='F'),P_H[hh,:].copy())
            weight21[:,:,hh]=np.outer(weight21_aux.flatten(order='F'),P_H[hh,:].copy())
            weight22[:,:,hh]=np.outer(weight22_aux.flatten(order='F'),P_H[hh,:].copy())
            
            weight13[:,:,hh]=np.outer(weight13_aux.flatten(order='F'),P_H[hh,:].copy())
            weight23[:,:,hh]=np.outer(weight23_aux.flatten(order='F'),P_H[hh,:].copy())
            
        
        # Dimensions (m*k,h,h')
        weight11 = np.ndarray.transpose(weight11.copy(),(0,2,1))
        weight12 = np.ndarray.transpose(weight12.copy(),(0,2,1))
        weight21 = np.ndarray.transpose(weight21.copy(),(0,2,1))
        weight22 = np.ndarray.transpose(weight22.copy(),(0,2,1))
                
        rowindex = np.tile(range(mpar['nm']*mpar['nk']*mpar['nh']),(1,4*mpar['nh']))
        
        
        H_a = sp.coo_matrix((np.concatenate((weight11.flatten(order='F'),weight21.flatten(order='F'),weight12.flatten(order='F'),weight22.flatten(order='F'))), 
                       (rowindex.flatten(order='F'), np.concatenate((index11.flatten(order='F'),index21.flatten(order='F'),index12.flatten(order='F'),index22.flatten(order='F'))))),
                       shape=(mpar['nm']*mpar['nk']*mpar['nh'], mpar['nm']*mpar['nk']*mpar['nh'])) # mu'(h',k'), a without interest
       
        
        weight13 = np.ndarray.transpose(weight13.copy(),(0,2,1))
        weight23 = np.ndarray.transpose(weight23.copy(),(0,2,1))
        
        rowindex = np.tile(range(mpar['nm']*mpar['nk']*mpar['nh']),(1,2*mpar['nh']))
        
        H_n = sp.coo_matrix((np.concatenate((weight13.flatten(order='F'),weight23.flatten(order='F'))), 
                       (rowindex.flatten(order='F'), np.concatenate((index13.flatten(order='F'),index23.flatten(order='F'))))),
                       shape=(mpar['nm']*mpar['nk']*mpar['nh'], mpar['nm']*mpar['nk']*mpar['nh'])) # mu'(h',k'), a without interest
        
        ## Joint transition matrix and transitions
        H = par['nu']*H_a.copy() +(1.-par['nu'])*H_n.copy()
        
        ## Joint transition matrix and transitions
        
        distJD = 9999.
        countJD = 1
        joint_distr = (joint_distr.copy().flatten(order='F')).T
        joint_distr_next = joint_distr.copy().dot(H.copy().todense())
        joint_distr_next = joint_distr_next.copy()/joint_distr_next.copy().sum(axis=1)

        distJD = np.max((np.abs(joint_distr_next.copy().flatten(order='F')-joint_distr.copy().flatten(order='F'))))
                          
        if distJD > 10**(-9):
            eigen, joint_distr = sp.linalg.eigs(H.transpose(), k=1, which='LM')
            joint_distr = joint_distr.copy().real  
            joint_distr = joint_distr.copy().transpose()/joint_distr.copy().sum()

        distJD = 9999.
        
        while (distJD > 10**(-14) or countJD<50) and countJD<10000:
        
            joint_distr_next = joint_distr.copy().dot(H.copy().todense())
            joint_distr_next = joint_distr_next.copy()/joint_distr_next.copy().sum(axis=1)
            
            distJD = np.max((np.abs(joint_distr_next.copy().flatten(order='F')-joint_distr.copy().flatten(order='F'))))
            
            countJD += 1
            joint_distr = joint_distr_next.copy()
                 
        joint_distr = np.array(joint_distr.copy())
        
        return {'joint_distr': joint_distr, 'distJD': distJD}
           

    def PoliciesSS(self, c_a_guess, c_n_guess, psi_guess, grid, inc, RR, RBRB, P, mpar, par, meshes):
        
        distC_n = 99999
        distPSI = distC_n
        distC_a = distPSI
        mutil_c_n = 1./(c_n_guess.copy()**par['xi']) # marginal utility at consumption policy no adjustment
        mutil_c_a = 1./(c_a_guess.copy()**par['xi']) # marginal utility at consumption policy adjustment
        mutil_c = par['nu']*mutil_c_a.copy() + (1-par['nu'])*mutil_c_n.copy() # Expected marginal utility at consumption policy (w &w/o adjustment)

        count=0
        while max(distC_n, distC_a, distPSI)>mpar['crit'] and count<100000:
        
        
            
            
            count=count+1
    
            # Step 1: Update policies for only money adjustment
            mutil_c=RBRB*mutil_c.copy() # take return on money into account
            aux=np.reshape(np.ndarray.transpose(mutil_c.copy(),(2, 0, 1)), (mpar['nh'], mpar['nm']*mpar['nk']),order='F')
            # form expectations
            EMU_aux = par['beta']*np.ndarray.transpose(np.reshape(P.copy().dot(aux.copy()),(mpar['nh'], mpar['nm'], mpar['nk']),order='F'),(1, 2, 0))
            
            c_n_aux = 1./(EMU_aux.copy()**(1./par['xi']))
            
            # Take borrowing constraint into account
            results_EGM_Step1_b=self.EGM_Step1_b(grid,inc,c_n_aux,mpar,par,meshes)
            c_n_new=results_EGM_Step1_b['c_update'].copy()
            m_n_star=results_EGM_Step1_b['m_update'].copy()
            
            m_n_star[m_n_star.copy()>grid['m'][-1]] = grid['m'][-1] # not extrapolation
                                    
            # Step 2: Find for every k on grid some off-grid m*(k')
            m_a_star_aux = self.EGM_Step2_SS(mutil_c_n,mutil_c_a, psi_guess, grid,P,RBRB,RR,par,mpar)
            m_a_star_aux = m_a_star_aux['mstar'].copy()
            
            # Step 3: Solve for initial resources / consumption in  adjustment case
            results_EGM_Step3 = self.EGM_Step3(EMU_aux,grid,inc,m_a_star_aux,c_n_aux,mpar,par)
            cons_list = results_EGM_Step3['cons_list']
            res_list  = results_EGM_Step3['res_list']
            mon_list = results_EGM_Step3['mon_list']
            cap_list = results_EGM_Step3['cap_list']
                                    
            
            # Step 4: Interpolate Consumption Policy
            results_EGM_Step4 = self.EGM_Step4( cons_list,res_list, mon_list,cap_list,inc,mpar,grid ) 
            c_a_new = results_EGM_Step4['c_a_new'].copy()
            m_a_star = results_EGM_Step4['m_a_star'].copy()
            cap_a_star = results_EGM_Step4['cap_a_star'].copy()
            
            # a = cap_a_star>grid['k'].T[-1]
            # log_index = indices(a, lambda x: x==1)
            
            cap_a_star[cap_a_star.copy()>grid['k'][-1]] = grid['k'][-1] # not extrapolation
            m_a_star[m_a_star.copy()>grid['m'][-1]] = grid['m'][-1] # not extrapolation
            
                        
            # Step 5: Update ~psi
            mutil_c_n = 1./(c_n_new.copy()**par['xi']) # marginal utility at consumption policy no adjustment
            mutil_c_a = 1./(c_a_new.copy()**par['xi']) # marginal utility at consumption policy adjustment
            mutil_c = par['nu']*mutil_c_a.copy() + (1-par['nu'])*mutil_c_n.copy() # Expected marginal utility at consumption policy (w &w/o adjustment)
            
            
            # VFI analogue in updating psi
            term1=((par['nu']* mutil_c_a.copy() *(par['Q'] + RR)) + ((1.-par['nu'])* mutil_c_n.copy()* RR) + (1.-par['nu'])* psi_guess.copy())
            
            aux = np.reshape(np.ndarray.transpose(term1.copy(),(2, 0, 1)),(mpar['nh'], mpar['nm']*mpar['nk']),order='F')
            E_rhs_psi = par['beta']*np.ndarray.transpose(np.reshape(P.copy().dot(aux.copy()),(mpar['nh'], mpar['nm'], mpar['nk']),order='F'),(1, 2, 0))
            
            E_rhs_psi=np.reshape( E_rhs_psi.copy(), (mpar['nm'], mpar['nk']*mpar['nh']), order='F' )
            m_n_star=np.reshape( m_n_star.copy(), (mpar['nm'], mpar['nk']*mpar['nh']), order='F' )
            
               
            
            # Interpolation of psi-function at m*_n(m,k)
            index = np.digitize(m_n_star.copy(),grid['m'])-1 # find indexes on grid next smallest to optimal policy
            index[m_n_star <= grid['m'][0]] = 0 # if below minimum
            index[m_n_star >= grid['m'][-1]] = len(grid['m'])-2 # if above maximum
            
            step = np.squeeze(np.diff(grid['m'])) # Stepsize on grid
            s = (np.asmatrix(m_n_star.copy()) - np.squeeze(grid['m'].T[index]))/step[index] # Distance of optimal policy to next grid point
                                    
            aux_index = np.ones((mpar['nm'],1))*np.arange(0, mpar['nk']*mpar['nh'])*mpar['nm'] # aux for linear indexes
            E_rhs_psi = E_rhs_psi.flatten(order='F').copy()
            aux3 = E_rhs_psi[(index.flatten(order='F').copy()+aux_index.flatten(order='F').copy()).astype(int)] # calculate linear indexes
            
            psi_new = aux3.copy() + np.squeeze(np.asarray(s.flatten(order='F').copy()))*(E_rhs_psi[(index.flatten(order='F')+aux_index.flatten(order='F')).astype(int)+1].copy()-aux3.copy()) # linear interpolation
            psi_new = np.reshape( psi_new.copy(), (mpar['nm'], mpar['nk'], mpar['nh']), order='F' )
            m_n_star = np.reshape( m_n_star.copy(), (mpar['nm'], mpar['nk'], mpar['nh']), order='F' )
            distPSI = max( (abs(psi_guess.flatten(order='F').copy()-psi_new.flatten(order='F').copy())) )
            
                        
            # Step 6: Check convergence of policies
            distC_n = max( (abs(c_n_guess.flatten(order='F').copy()-c_n_new.flatten(order='F').copy())) )
            distC_a = max( (abs(c_a_guess.flatten(order='F').copy()-c_a_new.flatten(order='F').copy())) )
            
            # Update c policy guesses
            c_n_guess = c_n_new.copy()
            c_a_guess = c_a_new.copy()
            psi_guess = psi_new.copy()
            
           
        #distPOL=(distC_n, distC_a, distPSI)
        distPOL=np.array((distC_n.copy(), distC_a.copy(), distPSI.copy()))
        print(max(distC_n, distC_a, distPSI))
        print(count)
        return {'c_n_guess':c_n_new, 
                'm_n_star':m_n_star,
                'c_a_guess':c_a_new,
                'm_a_star':m_a_star,
                'cap_a_star':cap_a_star,
                'psi_guess':psi_new,
                'distPOL':distPOL}
    
    def EGM_Step1_b(self, grid,inc,c_n_aux,mpar,par,meshes):
        ## EGM_Step1_b computes the optimal consumption and corresponding optimal money
         # holdings in case the capital stock cannot be adjusted by taking the budget constraint into account.
         # c_update(m,k,h):    Update for consumption policy under no-adj.
         # m_update(m,k,h):    Update for money policy under no-adj.
        m_star_n = (c_n_aux.copy() + meshes['m'] - inc['labor'] - inc['rent'] - inc['profits'])
        m_star_n = (m_star_n.copy() < 0) * m_star_n.copy() / ((par['RB']+par['borrwedge'])/par['PI']) + (m_star_n.copy() >= 0) * m_star_n.copy()/(par['RB']/par['PI'])

        # Identify binding constraints
        binding_constraints = meshes['m'] < np.tile(m_star_n[0,:,:].copy(),(mpar['nm'], 1, 1))

        # Consumption when drawing assets m' to zero: Eat all Resources
        Resource = inc['labor'] + inc['rent'] + inc['money'] + inc['profits']

        ## Next step: Interpolate w_guess and c_guess from new k-grids
         # using c(s,h,k',K), k(s,h,k',K)

        m_star_n = np.reshape(m_star_n.copy(),(mpar['nm'], mpar['nk']*mpar['nh']), order='F')
        c_n_aux= np.reshape(c_n_aux.copy(),(mpar['nm'], mpar['nk']*mpar['nh']), order='F')

        # Interpolate grid.m and c_n_aux defined on m_star_n over grid.m
        # [c_update, m_update]=egm1b_aux_mex(grid.m,m_star_n,c_n_aux);
        c_update=np.zeros((mpar['nm'], mpar['nk']*mpar['nh']))
        m_update=np.zeros((mpar['nm'], mpar['nk']*mpar['nh']))
        
        for hh in range(mpar['nk']*mpar['nh']):
            
            Savings=interp1d(m_star_n[:,hh].copy(),grid['m'], fill_value='extrapolate') # generate savings function a(s,a*)=a'
            m_update[:,hh] = Savings(grid['m']) # Obtain m'(m,h) by Interpolation
            Consumption = interp1d(m_star_n[:,hh].copy(),c_n_aux[:,hh],fill_value='extrapolate') # generate consumption function c(s,a*(s,a'))
            c_update[:,hh] = Consumption(grid['m'])  # Obtain c(m,h) by interpolation (notice this is out of grid, used linear interpolation)

        c_update = np.reshape(c_update,(mpar['nm'], mpar['nk'], mpar['nh']), order='F')
        m_update = np.reshape(m_update,(mpar['nm'], mpar['nk'], mpar['nh']), order='F')

        c_update[binding_constraints] = Resource[binding_constraints].copy()-grid['m'].T[0]
        m_update[binding_constraints] = min(grid['m'].T)
        
        return {'c_update':c_update, 'm_update':m_update}
    
    
    def EGM_Step2_SS(self, mutil_c_n,mutil_c_a, psi_guess, grid,P,RBRB,RR,par,mpar):
        
        term1 = ((par['nu'] * mutil_c_a.copy() * (par['Q'] + RR))+((1-par['nu']) * mutil_c_n.copy() * RR)+(1-par['nu']) * psi_guess.copy())
        aux = np.reshape( np.ndarray.transpose(term1.copy(),(2, 0, 1)),(mpar['nh'], mpar['nm']*mpar['nk']), order='F' )
        #term1 = par['beta']*np.ndarray.transpose( np.reshape(np.array(np.matrix(P.copy())*np.matrix(aux.copy())),(mpar['nh'], mpar['nm'], mpar['nk']), order='F'), (1, 2, 0) )
        term1 = par['beta']*np.ndarray.transpose( np.reshape(P.copy().dot(aux.copy()),(mpar['nh'], mpar['nm'], mpar['nk']), order='F'), (1, 2, 0) )
        
        term2 = RBRB*( par['nu'] * mutil_c_a.copy() +(1-par['nu']) * mutil_c_n.copy() )
        aux = np.reshape( np.ndarray.transpose(term2.copy(), (2, 0, 1)), (mpar['nh'], mpar['nm']*mpar['nk']), order='F' )
        #term2 = par['beta']*np.ndarray.transpose( np.reshape(np.array(np.matrix(P.copy())*np.matrix(aux.copy())),(mpar['nh'], mpar['nm'], mpar['nk']), order='F'), (1, 2, 0) )
        term2 = par['beta']*np.ndarray.transpose( np.reshape(P.copy().dot(aux.copy()),(mpar['nh'], mpar['nm'], mpar['nk']), order='F'), (1, 2, 0) )


        # Equation (59) in Appedix B.4.
        E_return_diff=term1.copy()/par['Q']-term2.copy()

        # Find an m*_n for given k' that solves the difference equation (59)
        mstar = Fastroot(grid['m'], E_return_diff)
        mstar = np.maximum(mstar.copy(),grid['m'].T[0]) # Use non-negativity constraint and monotonicity
        mstar = np.minimum(mstar.copy(),grid['m'].T[-1]) # Do not allow for extrapolation
        mstar = np.reshape(mstar.copy(), (mpar['nk'], mpar['nh']), order='F')
        
        return {'mstar':mstar}

#    xgrid=grid['m']
#    fx = E_return_diff 
# np.savetxt('fx.csv', fx, delimiter=',')
        

    
    def EGM_Step3(self, EMU,grid,inc,m_a_star,c_n_aux,mpar,par):
        
        # EGM_Step3 returns the resources (res_list), consumption (cons_list)
        # and money policy (mon_list) for given capital choice (cap_list).
        # For k'=0, there doesn't need to be a unique corresponding m*. We get a
        # list of consumption choices and resources for money choices m'<m* (mon_list) and cap
        # choices k'=0 (cap_list) and merge them with consumption choices and
        # resources that we obtain if capital constraint doesn't bind next period.

        # c_star: optimal consumption policy as function of k',h (both
        # constraints do not bind)
        # Resource: required resource for c_star

        # cons_list: optimal consumption policy if a) only k>=0 binds and b) both
        # constraints do not bind

        # res_list: Required resorces for cons_list
        # c_n_aux: consumption in t as function of t+1 grid (constrained version)

        # Constraints for money and capital are not binding
        EMU=np.reshape(EMU.copy(), (mpar['nm'], mpar['nk']*mpar['nh']),order='F')
        m_a_star=m_a_star.flatten(order='F').copy()
        # Interpolation of psi-function at m*_n(m,k)
        index = np.digitize(np.asarray(m_a_star.copy()),np.squeeze(grid['m']))-1 # find indexes on grid next smallest to optimal policy
        index[m_a_star<=grid['m'].T[0]] = 0 # if below minimum
        index[m_a_star>=grid['m'].T[-1]] = mpar['nm']-2 # if above maximum
        step = np.squeeze(np.diff(grid['m'])) # Stepsize on grid
        s = (np.asmatrix(m_a_star.T) - grid['m'].T[index].T)/step.T[index].T # Distance of optimal policy to next grid point

        aux_index=np.arange(0,(mpar['nk']*mpar['nh']),1)*mpar['nm'] # aux for linear indexes
        EMU=EMU.flatten(order='F').copy()
        aux3=EMU[index.flatten(order='F')+aux_index.flatten(order='F')].copy() # calculate linear indexes

        # Interpolate EMU(m',k',h') over m*_n(k'), m-dim is dropped
        EMU_star = ( aux3.copy() + np.asarray(s.copy())*np.asarray( EMU[index.flatten(order='F').copy() + aux_index.flatten(order='F').copy()+1].copy() - aux3.copy() ) ).T # linear interpolation
        
        c_star = 1./(EMU_star.copy()**(1/par['xi']))
        cap_expenditure = np.squeeze(inc['capital'][0,:,:])
        auxL = np.squeeze(inc['labor'][0,:,:])
        auxP = inc['profits']
        # Resources that lead to capital choice k'
        # = c + m*(k') + k' - w*h*N = value of todays cap and money holdings
        Resource = c_star.flatten(order='F').copy() + m_a_star.flatten(order='F').copy() + cap_expenditure.flatten(order='F').copy() - auxL.flatten(order='F').copy() - auxP

        c_star = np.reshape( c_star.copy(), (mpar['nk'], mpar['nh']), order='F' )
        Resource = np.reshape( Resource.copy(), (mpar['nk'], mpar['nh']), order='F' )

        # Money constraint is not binding, but capital constraint is binding
        m_a_star = np.reshape(m_a_star.copy(), (mpar['nk'], mpar['nh']), order='F')
        m_star_zero = np.squeeze(m_a_star[0,:].copy()) # Money holdings that correspond to k'=0:  m*(k=0)

        # Use consumption at k'=0 from constrained problem, when m' is on grid
        aux_c = np.reshape(c_n_aux[:,0,:].copy(), (mpar['nm'], mpar['nh']), order='F')
        aux_inc = np.reshape( inc['labor'][0,0,:].copy() + inc['profits'], (1, mpar['nh']), order='F' )
        cons_list = []
        res_list = []
        mon_list = []
        cap_list = []

        # j=0
        for j in range(mpar['nh']):
            # When choosing zero capital holdings, HHs might still want to choose money
            # holdings smaller than m*(k'=0)
            if m_star_zero[j] > grid['m'].T[0]:
                # Calculate consumption policies, when HHs chooses money holdings
                # lower than m*(k'=0) and capital holdings k'=0 and save them in cons_list
                a = (grid['m']< m_star_zero[j]).astype(int).T
                log_index = self.indices(a, lambda x: x==1)
                # log_index = np.squeeze((np.cumsum(log_index.copy())-1)[:-1])
                # aux_c is the consumption policy under no cap. adj. (fix kï¿½=0), for mï¿½<m_a*(k'=0)
                c_k_cons=aux_c[log_index, j].T.copy()
                cons_list.append(c_k_cons.copy()) # Consumption at k'=0, m'<m_a*(0)
                # Required Resources: Money choice + Consumption - labor income
                # Resources that lead to k'=0 and m'<m*(k'=0)
               
                res_list.append(np.squeeze(grid['m'].T[log_index]) + c_k_cons.copy() - aux_inc.T[j])
                mon_list.append(np.squeeze(grid['m'].T[log_index]))
                
                #log_index = (grid['m']< m_star_zero[j]).astype(int)
                cap_list.append(np.zeros((np.sum(a.copy()),1)))


        # Merge lists
        c_star = np.reshape(c_star.copy(),(mpar['nk'], mpar['nh']), order='F')
        m_a_star = np.reshape(m_a_star.copy(),(mpar['nk'], mpar['nh']), order='F')
        Resource = np.reshape(Resource.copy(),(mpar['nk'], mpar['nh']), order='F')
        
        cons_list_1=[]
        res_list_1=[]
        mon_list_1=[]
        cap_list_1=[]
        
        for j in range(mpar['nh']):
            if m_star_zero[j] > grid['m'].T[0]:
               cons_list_1.append( np.vstack((np.asmatrix(cons_list[j]).T, np.asmatrix(c_star[:,j]).T)) )
               res_list_1.append( np.vstack((np.asmatrix(res_list[j]).T, np.asmatrix(Resource[:,j]).T)) )
               mon_list_1.append( np.vstack((np.asmatrix(mon_list[j]).T, np.asmatrix(m_a_star[:,j]).T)) )
               cap_list_1.append( np.vstack((np.asmatrix(cap_list[j]), np.asmatrix(grid['k']).T)) )
            else:
                cons_list_1.append(np.asmatrix(c_star[:,j]).T)
                res_list_1.append(np.asmatrix(Resource[:,j]).T)
                mon_list_1.append(np.asmatrix(m_a_star[:,j]).T)
                cap_list_1.append( np.asmatrix(grid['k'].T))
                
        return {'c_star': c_star, 'Resource': Resource, 'cons_list':cons_list_1, 'res_list':res_list_1, 'mon_list':mon_list_1, 'cap_list':cap_list_1}


    def indices(self, a, func):
        return [i for (i, val) in enumerate(a) if func(val)] 

            
    def EGM_Step4(self, cons_list, res_list, mon_list, cap_list, inc, mpar, grid ):
        # EGM_Step4 obtains consumption, money, and capital policy under adjustment.
        # The function uses the {(cons_list{j},res_list{j})} as measurement
        # points. The consumption function in (m,k) can be obtained from
        # interpolation by using the total resources available at (m,k): R(m,k)=qk+m/pi.
        # c_a_new(m,k,h): Update for consumption policy under adjustment
        # m_a_new(m,k,h): Update for money policy under adjustment
        # k_a_new(m,k,h): Update for capital policy under adjustment
        
        c_a_new=np.empty((mpar['nm']*mpar['nk'], mpar['nh']))
        m_a_new=np.empty((mpar['nm']*mpar['nk'], mpar['nh']))
        k_a_new=np.empty((mpar['nm']*mpar['nk'], mpar['nh']))
        Resource_grid=np.reshape(inc['capital']+inc['money']+inc['rent'], (mpar['nm']*mpar['nk'], mpar['nh']),order='F')
        labor_inc_grid=np.reshape(inc['labor'] + inc['profits'], (mpar['nm']*mpar['nk'], mpar['nh']), order='F')

        for j in range(mpar['nh']):
            a = (Resource_grid[:,j].copy() < res_list[j][0]).astype(int).T
            log_index = self.indices(a, lambda x: x==1)
            # when at most one constraint binds:
            #     [c_a_new(:,j), m_a_new(:,j),k_a_new(:,j)] = ...
            #         myinter1m_mex(res_list{j},Resource_grid(:,j),cons_list{j},mon_list{j},cap_list{j});
            
            cons = interp1d(np.squeeze(np.asarray(res_list[j].copy())), np.squeeze(np.asarray(cons_list[j].copy())), fill_value='extrapolate')
            c_a_new[:,j] = cons(Resource_grid[:,j])
            mon = interp1d(np.squeeze(np.asarray(res_list[j].copy())), np.squeeze(np.asarray(mon_list[j].copy())), fill_value='extrapolate')
            m_a_new[:,j] = mon(Resource_grid[:,j])
            cap = interp1d(np.squeeze(np.asarray(res_list[j].copy())), np.squeeze(np.asarray(cap_list[j].copy())), fill_value='extrapolate')
            k_a_new[:,j] = cap(Resource_grid[:,j])
    
            # Lowest value of res_list corresponds to m_a'=0 and k_a'=0.
            # Any resources on grid smaller then res_list imply that HHs consume all
            # resources plus income.
            # When both constraints are binding:
            c_a_new[log_index,j] = Resource_grid[log_index,j].copy() + labor_inc_grid[log_index,j].copy() - grid['m'].T[0]
            m_a_new[log_index,j] = grid['m'].T[0]
            k_a_new[log_index,j] = 0

        c_a_new = np.reshape(c_a_new.copy(),(mpar['nm'], mpar['nk'], mpar['nh']), order='F')
        k_a_new = np.reshape(k_a_new.copy(),(mpar['nm'], mpar['nk'], mpar['nh']), order='F')
        m_a_new = np.reshape(m_a_new.copy(),(mpar['nm'], mpar['nk'], mpar['nh']), order='F')
        
        return {'c_a_new':c_a_new, 'm_a_star':m_a_new, 'cap_a_star':k_a_new}
  
 

       
    def PolicyGuess(self, meshes, WW, RR, RBRB, par, mpar):
        '''
        policyguess returns autarky policy guesses (in the first period only):
        c_a_guess, c_n_guess, psi_guess
        as well as income matrices later on used in EGM: inc
  
        Consumption is compositite leisure and physical consumption (x_it) in the
        paper, therefore labor income is reduced by the fraction of leisure consumed
       
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
        RR : float    
            rental rate    
        RBRB : float    
            interest rate
            
        returns
        -----------
        c_a_guess : np.array
            guess for c func
        c_n_guess : np.array
        psi_guess : np.array
        inc : dict
            guess for incomes
        '''
        inc = { }
        
        inc['labor'] = par['tau']*WW.copy()*meshes['h'].copy()
        inc['rent']  = RR*meshes['k'].copy()
        inc['money'] = RBRB.copy()*meshes['m'].copy()
        inc['capital'] = par['Q']*meshes['k'].copy()
        inc['profits'] = 0. # lumpsum profits
        
        ## Initial policy guesses: Autarky policies as guess
        # consumption guess
        c_a_guess = inc['labor'].copy() + inc['rent'].copy() + inc['capital'].copy() + np.maximum(inc['money'].copy(),0.) + inc['profits']
        c_n_guess = inc['labor'].copy() + inc['rent'].copy() +                         np.maximum(inc['money'].copy(),0.) + inc['profits']
    
        # initially guessed marginal continuation value of holding capital
        psi_guess = np.zeros((mpar['nm'],mpar['nk'],mpar['nh']))            
            
        return {'c_a_guess':c_a_guess, 'c_n_guess':c_n_guess, 'psi_guess':psi_guess, 'inc': inc}       
         
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
        mc = par['mu'] - ((par['beta'] - 1.)*np.log(par['PI']))/par['kappa']        
        N = (par['tau']*par['alpha']*grid['K']**(1.-par['alpha'])*mc)**(1./(1.-par['alpha']+par['gamma']))
        W_fc = par['alpha'] *mc *(grid['K']/N)**(1.-par['alpha'])
        
        # Before tax return on capital
        R_fc = (1.-par['alpha'])*mc *(N/grid['K'])**par['alpha'] - par['delta'] 
            
        Y = N**par['alpha']*grid['K']**(1.-par['alpha'])
        Profits_fc = (1.-mc)*Y - Y*(1./(1.-par['mu'])) /par['kappa'] /2. *np.log(par['PI'])**2.
    
        NW = par['gamma']/(1.+par['gamma'])*N/par['H'] *W_fc
    
        WW = NW*np.ones((mpar['nm'], mpar['nk'] ,mpar['nh'])) # Wages
        WW[:,:,-1] = Profits_fc * par['profitshare']
        RR = R_fc # Rental rates
        RBRB = (par['RB']+(meshes['m']<0)*par['borrwedge'])/par['PI']
    
        return {'N':N, 'R_fc':R_fc, 'W_fc':W_fc, 'Profits_fc':Profits_fc, 'WW':WW, 'RR':RR, 'RBRB':RBRB,'Y':Y}


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
        grid['boundsH'] = boundsH.copy()
     
    
        return {'P_H': P_H, 'grid':grid, 'par':par}
    
    
    def SteadyState(self, P_H, grid, meshes, mpar, par):
        '''
        Prepare items for EGM
        Layout of matrices:
         Dimension 1: money m
         Dimension 2: capital k
         Dimension 3: stochastic sigma s x stochastic human capital h
         Dimension 4: aggregate money M
         Dimension 5: aggregate capital K
        '''
        
        # 1)  Construct relevant return matrices
        joint_distr = np.ones((mpar['nm'],mpar['nk'],mpar['nh']))/(mpar['nh']*mpar['nk']*mpar['nm'])
        fr_result = self.FactorReturns(meshes, grid, par, mpar)
        WW = fr_result['WW'].copy()
        RR = fr_result['RR']
        RBRB = fr_result['RBRB'].copy()
        
        # 2) Guess initial policies
        pg_result = self.PolicyGuess(meshes, WW, RR, RBRB, par, mpar)
        c_a_guess = pg_result['c_a_guess'].copy()
        c_n_guess = pg_result['c_n_guess'].copy()
        psi_guess = pg_result['psi_guess'].copy()
        
        KL = 0.5*grid['K']
        KH = 1.5*grid['K']
        
        r0_excessL = self.ExcessK(KL, c_a_guess, c_n_guess, psi_guess, joint_distr, grid, P_H, mpar, par, meshes)
        r0_excessH = self.ExcessK(KH, c_a_guess, c_n_guess, psi_guess, joint_distr, grid, P_H, mpar, par, meshes)
        excessL = r0_excessL['excess']
        excessH = r0_excessH['excess']
        
        if np.sign(excessL) == np.sign(excessH):
            print('ERROR! Sign not diff')
        
        ## Brent  
        fa = excessL
        fb = excessH
        a = KL
        b = KH
        
        if fa*fb > 0. :
            print('Error! f(a) and f(b) should have different signs')

        c = a
        fc = fa            
        d = b-1.0
        e = d
        
        
        iter = 0
        maxiter = 1000
        
        
        while iter<maxiter :
            iter += 1
            print(iter)
            
            if fb*fc>0:
              c = a
              fc = fa
              d = b-a
              e = d
    
    
            if np.abs(fc) < np.abs(fb) :
              a = b
              b = c
              c = a
              
              fa = fb
              fb = fc
              fc = fa

            eps = np.spacing(1)     
            tol = 2*eps*np.abs(b) + mpar['crit']
            m = (c-b)/2. # tolerance
    
            if (np.abs(m)>tol) and (np.abs(fb)>0) :
        
              if (np.abs(e)<tol) or (np.abs(fa)<=np.abs(fb)) :
                d = m
                e = m
              
              else:
                s = fb/fa
                
                if a==c:
                   p=2*m*s
                   q=1-s
                else :
                   q=fa/fc
                   r=fb/fc
                   p=s*(2*m*q*(q-r)-(b-a)*(r-1))
                   q=(q-1)*(r-1)*(s-1)
            
            
                if p>0 :
                   q=-q
                else :
                   p=-p
            
                s=e
                e=d
                 
                if ( 2*p<3*m*q-np.abs(tol*q) ) and (p<np.abs(s*q/2)) :
                   d=p/q
                else:
                   d=m
                   e=m
            
              a=b
              fa=fb
        
              if np.abs(d)>tol :
                b=b+d
              else :
                if m>0 :
                  b=b+tol
                else:
                  b=b-tol
            
            else:
              break
            
            
            r_excessK = self.ExcessK(b,c_a_guess,c_n_guess,psi_guess,joint_distr, grid,P_H,mpar,par,meshes)
            fb = r_excessK['excess']
            c_n_guess = r_excessK['c_n_guess'].copy()
            c_a_guess = r_excessK['c_a_guess'].copy()
            psi_guess = r_excessK['psi_guess'].copy()
            joint_distr = r_excessK['joint_distr'].copy()
            
            
        Kcand = b
        grid['K'] = b
       
        ## Update
        r1_excessK=self.ExcessK(Kcand,c_a_guess,c_n_guess,psi_guess,joint_distr, grid,P_H,mpar,par,meshes)
        excess = r1_excessK['excess']
        N = r1_excessK['N']
       
        grid['N'] = N
        
        return {'c_n_guess':r1_excessK['c_n_guess'], 'm_n_star':r1_excessK['m_n_star'], 'c_a_guess':r1_excessK['c_a_guess'],
                'm_a_star':r1_excessK['m_a_star'], 'cap_a_star':r1_excessK['cap_a_star'], 'psi_guess':r1_excessK['psi_guess'], 
                'joint_distr':r1_excessK['joint_distr'],
                'R_fc':r1_excessK['R_fc'], 'W_fc':r1_excessK['W_fc'], 'Profits_fc':r1_excessK['Profits_fc'], 
                'Output':r1_excessK['Output'], 'grid':grid }

    def ExcessK(self,K,c_a_guess,c_n_guess,psi_guess,joint_distr, grid,P_H,mpar,par,meshes):
        
        grid['K'] = K
        fr_ek_result = self.FactorReturns(meshes, grid, par, mpar)
        N    = fr_ek_result['N']
        R_fc = fr_ek_result['R_fc']
        W_fc = fr_ek_result['W_fc']
        Profits_fc = fr_ek_result['Profits_fc']
        WW = fr_ek_result['WW'].copy()
        RR = fr_ek_result['RR']
        RBRB = fr_ek_result['RBRB'].copy()
        Output = fr_ek_result['Y']
                
        pg_ek_result = self.PolicyGuess(meshes, WW, RR, RBRB, par, mpar)
        c_a_guess = pg_ek_result['c_a_guess'].copy()
        c_n_guess = pg_ek_result['c_n_guess'].copy()
        psi_guess = pg_ek_result['psi_guess'].copy()
        inc = pg_ek_result['inc'].copy()
        
                
        # solve policies and joint distr
        print('Solving Household problem by EGM')
        start_time = time.clock()
                            
        pSS_ek_result = self.PoliciesSS(c_a_guess,c_n_guess,psi_guess, grid, inc, RR,RBRB,P_H,mpar,par,meshes)
        c_n_guess = pSS_ek_result['c_n_guess'].copy()
        m_n_star = pSS_ek_result['m_n_star'].copy()
        c_a_guess = pSS_ek_result['c_a_guess'].copy()
        m_a_star = pSS_ek_result['m_a_star'].copy()
        cap_a_star = pSS_ek_result['cap_a_star'].copy()
        psi_guess = pSS_ek_result['psi_guess'].copy()
        distPOL = pSS_ek_result['distPOL'].copy()
        
        
        print(distPOL)
        
        end_time = time.clock()
        print('Elapsed time is ',  (end_time-start_time), ' seconds.')
        
        print('Calc Joint Distr')
        start_time = time.clock()
        jd_ek_result = self.JDiteration(joint_distr, m_n_star, m_a_star, cap_a_star, P_H, par, mpar, grid)
        joint_distr = np.reshape(jd_ek_result['joint_distr'].copy(),(mpar['nm'],mpar['nk'],mpar['nh']),order='F')
        print(jd_ek_result['distJD'])
        
        end_time = time.clock()
        print('Elapsed time is ',  (end_time-start_time), ' seconds.')
        
        AggregateCapitalDemand = np.sum(grid['k']*np.sum(np.sum(joint_distr.copy(),axis = 0),axis=1))
        excess = grid['K']-AggregateCapitalDemand
        
        return{'excess':excess,'c_n_guess':c_n_guess,'m_n_star':m_n_star,'c_a_guess':c_a_guess,
               'm_a_star':m_a_star,'cap_a_star':cap_a_star,'psi_guess':psi_guess,
               'joint_distr':joint_distr, 'R_fc':R_fc,'W_fc':W_fc,'Profits_fc':Profits_fc,'Output':Output,'N':N}

###############################################################################

if __name__ == '__main__':
    
    import defineSSParametersTwoAsset as Params
    from copy import copy
    import time
    import pickle
    
    EX3param = copy(Params.parm_TwoAsset)
    
    start_time0 = time.clock()        
    EX3 = SteadyStateTwoAsset(**EX3param)
    
    EX3SS = EX3.SolveSteadyState()
    
    end_time0 = time.clock()
    print('Elapsed time is ',  (end_time0-start_time0), ' seconds.')
    
    pickle.dump(EX3SS, open("EX3SS_20.p", "wb"))
    open("EX3SS_20.p", "wb").close
    
    
    #EX3SS=pickle.load(open("EX3SS_ag.p", "rb"))