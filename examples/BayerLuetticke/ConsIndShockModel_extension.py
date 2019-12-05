'''
Extends the IndShockConsumerType agent to store a distribution of agents and
calculates a transition matrix for this distribution, along with the steady
state distribution
'''
from __future__ import print_function
import sys 
import os
from copy import copy, deepcopy
import numpy as np
import scipy as sc
from scipy import sparse as sp
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.utilities import makeGridExpMult


class IndShockConsumerType_extend(IndShockConsumerType):
    '''
    An extension of the IndShockConsumerType that adds methods to handle
    the distribution of agents over market resources and permanent income.
    These methods could eventually become part of IndShockConsumterType itself
    '''        
    
    def __init__(self,cycles=1,time_flow=True,**kwds):
        '''
        Just calls on IndShockConsumperType
        
        Parameters
        ----------
        cycles : int
            Number of times the sequence of periods should be solved.
        time_flow : boolean
            Whether time is currently "flowing" forward for this instance.
        
        Returns
        -------
        None
        '''       
        # Initialize an IndShockConsumerType
        IndShockConsumerType.__init__(self,cycles=cycles,time_flow=time_flow,**kwds)
        
    def DefineDistributionGrid(self, Dist_mGrid=None, Dist_pGrid=None):
        '''
        Defines the grid on which the distribution is defined
        
        Parameters
        ----------
        Dist_mGrid : np.array()
            Grid for distribution over normalized market resources
        Dist_pGrid : np.array()
            Grid for distribution over permanent income
        
        Returns
        -------
        None
        '''  
        if self.cycles != 0:
            print('Distributional methods presently only work for perpetual youth agents (cycles=0)')
        else:
            if Dist_mGrid == None:
                self.Dist_mGrid = self.aXtraGrid
            else:
                self.Dist_mGrid = Dist_mGrid
            if Dist_pGrid == None:
                num_points = 50
                #Dist_pGrid is taken to cover most of the ergodic distribution
                p_variance = self.PermShkStd[0]**2
                max_p = 20.0*(p_variance/(1-self.LivPrb[0]))**0.5
                one_sided_grid = makeGridExpMult(1.0+1e-3, np.exp(max_p), num_points, 2)
                self.Dist_pGrid = np.append(np.append(1.0/np.fliplr([one_sided_grid])[0],np.ones(1)),one_sided_grid)
            else:
                self.Dist_pGrid = Dist_pGrid
            
    def CalcTransitionMatrix(self):
        '''
        Calculates how the distribution of agents across market resources 
        transitions from one period to the next
        ''' 
        Dist_mGrid = self.Dist_mGrid
        Dist_pGrid = self.Dist_pGrid
        aNext = Dist_mGrid - self.solution[0].cFunc(Dist_mGrid)
        bNext = self.Rfree*aNext
        ShockProbs = self.IncomeDstn[0][0]
        TranShocks = self.IncomeDstn[0][1]
        PermShocks = self.IncomeDstn[0][2]
        LivPrb = self.LivPrb[0]
        #New borns have this distribution (assumes start with no assets and permanent income=1)
        NewBornDist = self.JumpToGrid(TranShocks,np.ones_like(TranShocks),ShockProbs)
        TranMatrix = np.zeros((len(Dist_mGrid)*len(Dist_pGrid),len(Dist_mGrid)*len(Dist_pGrid)))
        for i in range(len(Dist_mGrid)):
            for j in range(len(Dist_pGrid)):
                mNext_ij = bNext[i]/PermShocks + TranShocks
                pNext_ij = Dist_pGrid[j]*PermShocks
                TranMatrix[:,i*len(Dist_pGrid)+j] = LivPrb*self.JumpToGrid(mNext_ij, pNext_ij, ShockProbs) + (1.0-LivPrb)*NewBornDist
        self.TranMatrix = TranMatrix
                
    def JumpToGrid(self,m_vals, perm_vals, probs):
        '''
        Distributes values onto a predefined grid, maintaining the means
        ''' 
        probGrid = np.zeros((len(self.Dist_mGrid),len(self.Dist_pGrid)))
        mIndex = np.digitize(m_vals,self.Dist_mGrid) - 1
        mIndex[m_vals <= self.Dist_mGrid[0]] = -1
        mIndex[m_vals >= self.Dist_mGrid[-1]] = len(self.Dist_mGrid)-1
        
        pIndex = np.digitize(perm_vals,self.Dist_pGrid) - 1
        pIndex[perm_vals <= self.Dist_pGrid[0]] = -1
        pIndex[perm_vals >= self.Dist_pGrid[-1]] = len(self.Dist_pGrid)-1
        
        for i in range(len(m_vals)):
            if mIndex[i]==-1:
                mlowerIndex = 0
                mupperIndex = 0
                mlowerWeight = 1.0
                mupperWeight = 0.0
            elif mIndex[i]==len(self.Dist_mGrid)-1:
                mlowerIndex = -1
                mupperIndex = -1
                mlowerWeight = 1.0
                mupperWeight = 0.0
            else:
                mlowerIndex = mIndex[i]
                mupperIndex = mIndex[i]+1
                mlowerWeight = (self.Dist_mGrid[mupperIndex]-m_vals[i])/(self.Dist_mGrid[mupperIndex]-self.Dist_mGrid[mlowerIndex])
                mupperWeight = 1.0 - mlowerWeight
                
            if pIndex[i]==-1:
                plowerIndex = 0
                pupperIndex = 0
                plowerWeight = 1.0
                pupperWeight = 0.0
            elif pIndex[i]==len(self.Dist_pGrid)-1:
                plowerIndex = -1
                pupperIndex = -1
                plowerWeight = 1.0
                pupperWeight = 0.0
            else:
                plowerIndex = pIndex[i]
                pupperIndex = pIndex[i]+1
                plowerWeight = (self.Dist_pGrid[pupperIndex]-perm_vals[i])/(self.Dist_pGrid[pupperIndex]-self.Dist_pGrid[plowerIndex])
                pupperWeight = 1.0 - plowerWeight
                
            probGrid[mlowerIndex][plowerIndex] = probGrid[mlowerIndex][plowerIndex] + probs[i]*mlowerWeight*plowerWeight
            probGrid[mlowerIndex][pupperIndex] = probGrid[mlowerIndex][pupperIndex] + probs[i]*mlowerWeight*pupperWeight
            probGrid[mupperIndex][plowerIndex] = probGrid[mupperIndex][plowerIndex] + probs[i]*mupperWeight*plowerWeight
            probGrid[mupperIndex][pupperIndex] = probGrid[mupperIndex][pupperIndex] + probs[i]*mupperWeight*pupperWeight
            
        return probGrid.flatten()
    
    def CalcErgodicDist(self):
        '''
        Calculates the egodic distribution across normalized market resources and
        permanent income as the eigenvector associated with the eigenvalue 1.
        The distribution is reshaped as an array with the ij'th element representing
        the probability of being at the i'th point on the mGrid and the j'th
        point on the pGrid.
        ''' 
        eigen, ergodic_distr = sp.linalg.eigs(self.TranMatrix, k=1, which='LM')
        ergodic_distr = ergodic_distr.real/np.sum(ergodic_distr.real)
        self.ergodic_distr = ergodic_distr.reshape((len(self.Dist_mGrid),len(self.Dist_pGrid)))
        
if __name__ == '__main__':
    import HARK.ConsumptionSaving.ConsumerParameters as Params
    from HARK.utilities import plotFuncsDer, plotFuncs
    from time import clock
    mystr = lambda number : "{:.4f}".format(number)
    
    
    # Make and solve an example consumer with idiosyncratic income shocks
    IndShock_extendExample = IndShockConsumerType_extend(**Params.init_idiosyncratic_shocks)
    IndShock_extendExample.cycles = 0 # Make this type have an infinite horizon
    
    start_time = clock()
    IndShock_extendExample.solve()
    end_time = clock()
    print('Solving a consumer with idiosyncratic shocks took ' + mystr(end_time-start_time) + ' seconds.')
    
    IndShock_extendExample.DefineDistributionGrid()   
    start_time = clock()
    IndShock_extendExample.CalcTransitionMatrix()
    end_time = clock()
    print('Calculating the transition matrix took ' + mystr(end_time-start_time) + ' seconds.')
    start_time = clock()
    IndShock_extendExample.CalcErgodicDist()
    end_time = clock()
    print('Calculating the ergodic distribution took ' + mystr(end_time-start_time) + ' seconds.')
    
 
