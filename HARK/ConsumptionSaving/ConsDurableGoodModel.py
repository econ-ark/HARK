'''
Consumption-saving model with durable good
Tianyang He
'''

import numpy as np
from HARK.ConsumptionSaving.ConsGenIncProcessModel import MargValueFunc2D
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType, ConsumerSolution
import HARK.distribution
from HARK.utilities import CRRAutility, CRRAutilityP, CRRAutilityP_inv, CRRAutility_inv
from HARK.interpolation import Curvilinear2DInterp


class BabyDurableConsumerType(IndShockConsumerType):
    time_inv = IndShockConsumerType.time_inv_ + ['DurCost0', 'DurCost1']

    def _init_(self, cycles=1, verbose=1, quiet=False, **kwds):
        IndShockConsumerType.__init__(cycles=cycles, verbose=verbose, quiet=quiet, **kwds)
        self.solveOnePeriod = solveBabyDurable

    def update(self):
        IndShockConsumerType.update(self)
        self.updateDeprFacDstn()
        self.updateDNrmGrid()
        self.updateShockDstn()

    def updateDeprFacDstn(self):
        bot = self.DeprFacMean - 0.5 * self.DeprFacSpread
        top = self.DeprFacMean + 0.5 * self.DeprFacSpread
        N = self.DerpFacCount
        Uni_DeprFacDstn = HARK.distribution.Uniform(bot, top)
        self.DeprFacDstn = Uni_DeprFacDstn.approx(N)
        self.addToTimeInv('DeprFacDstn')

    def updateDNrmGrid(self):
        self.DNrmGrid = np.linspace(self.DNrmMin, self.DNrMax, self.DnrmCount)
        self.addToTimeInv('DNrmGrid')

    def updateShockDstn(self):
        self.ShockDstn = list()
        for j in range(0, self.T_age):
            self.ShockDstn[j] = HARK.distribution.combineIndepDstns(self.IncomeDstn[j], self.DeprFacDstn)
        self.addToTimeVary('ShockDstn')

    def solveBabyDurable(self, solution_next, ShockDstn, LivPrb, DiscFac, CRRA, Rfree, PermGroFac, aXtraGrid, DNrmGrid,
                         alpha, kappa0, kappa):
        
        
        '''
        u: utility
        uPC marginal utility with respect to C
        uPD marginal utility with respect to D
        uinv_C inverse utility with respect to C
        uinv_D inverse utility with respect to D
        uPinv_C inverse marginal utility with respect to C
        uPinv_D inverse marginal utility with respect to D
        
        '''
        u = lambda C, D: CRRAutility(C, CRRA) * CRRAutility((D / C) ** alpha, CRRA)
        uPC = lambda C, D: CRRAutility(D ** alpha, CRRA) * (1 - alpha) * C ** (-alpha - CRRA + alpha * CRRA)
        uPD = lambda C, D: CRRAutility(C ** (1 - alpha), CRRA) * alpha * D ** (alpha - alpha * CRRA - 1)
        uinv_C = lambda u, D: (CRRAutility_inv(u, CRRA) * D ** (-alpha)) ** (1 / (1 - alpha))
        uinv_D = lambda u, C: (CRRAutility_inv(u, CRRA) * C ** (alpha - 1)) ** (1 / alpha)
        uPinv_C = lambda uPC, D: ((1 - CRRA) * uPC / ((1 - alpha) * D ** (alpha - alpha * CRRA))) ** (-1 / (alpha + CRRA + alpha * CRRA))
        uPinv_D = lambda uPD, C: ((1 - CRRA) * uPD / (alpha * C ** ((1 - CRRA)(1 - alpha))) ** (1 / (alpha - alpha * CRRA - 1)))
        gfunc = lambda n: kappa0 * n ** kappa
        
        
        
        '''
        2. unpack IncomeDstn into its component arrays: probabilities, transitory shocks 
        permannent shock and depreciation shocks
        ShockProbs: shock probability
        PermshockVals: permanment shock
        TranshockVals: transitory shock
        DeprshockVals: depreciation shock
        
        '''

        ShockProbs = ShockDstn.pmf
        PermshockVals = ShockDstn.X[0]
        TranshockVals = ShockDstn.X[1]
        DeprshockVals = ShockDstn.X[2]

        '''
        3. Make tiled versions of the probability and shock vectors
        these should have shape (aXtraGrid.size, DNrmGrid.size,ShockProbs.size)
        
        '''
        aXtraGrid_size = aXtraGrid.size
        DNrmGrid_size = DNrmGrid.size
        ShockProbs_size = ShockProbs.size
        PermShk = np.tile(PermshockVals, aXtraGrid_size).transpose()
        TranShk = np.tile(TranshockVals, aXtraGrid_size).transpose()
        DeprShk = np.tile(DeprshockVals, DNrmGrid_size).transpose()
        ShkProbs = np.tile(ShockProbs, ShockProbs_size).transpose()
        
        '''
        4. Make tiled version of DNrmGrid that has the same shape as the tiled shock arrays. 
        Also make a tiled version of aXtraGrid with the same shape
        aNrm： a_t
        dNrm: D_t
        
        '''
        aNrmNow = np.asarray(aXtraGrid)
        TranShkCount = TranShk.size
        aNrm = np.tile(aNrmNow, (TranShkCount, 1))
        dNrmNow = np.asarray(DNrmGrid)
        DeprShkCount = DeprShk.size
        dNrm = np.tile(dNrmNow, (DeprShkCount, 1))
        
        '''
        5.Using the intertemporal transition equations, make arrays named mNrmNextArray 
        and dNrmNextArray representing realizations of m_t+1 and d_t+1 from each end-of-period state 
        when each different shock combination arrives
        
        '''

        mNrmNextArray = Rfree / (PermGroFac * PermShk) * aNrm + TranShk
        dNrmNextArray = (1 - DeprShk) * dNrm / (PermGroFac * PermShk)
        
        '''
        6.Make arrays called dvdmNextArray and dvddNextArray by evaluating next period' marginal value functions
        at the future state realizations arrays. These functions can be found in solution_next.dvdmFunc and 
        solution_next.dvddFunc
        
        '''
        dvdmNextArray = solution_next.dvdmFunc
        dvddNextArray = solution_next.dvddFunc
        
        '''
        7. Calculate end-of-period expected marginal value function, along with the tiled arrays that have been
        constructed. You will probably want to multiply by the shock probabilities and sum along axis=2. These
        arrays should be called EndOfPrddvada and EndOfPrddvdD.
        
        '''
        
        EndOfPrddvda = Rfree * DiscFac * LivPrb * (PermGroFac * PermShk) ** (-CRRA) * np.sum(
                dvdmNextArray(mNrmNextArray, dNrmNextArray) * ShkProbs, axis=2)
        
        EndOfPrddvdD = Rfree * DiscFac * LivPrb * (1 - DeprShk) * (PermGroFac * PermShk) ** (-CRRA) * np.sum(
                dvddNextArray(mNrmNextArray, dNrmNextArray) * ShkProbs, axis=2)
                       
        '''
        8. Algebraically solve FOC-c for c_t, and use EndOfPrddvada to calculate c_t for each end-of-period
        state (a_t, D_t). Call the result cNrmArray
        
        '''                

        cNrmArray = (EndOfPrddvda * dNrm ** (alpha * CRRA - alpha) / (1 - alpha)) ** (1 / ((1 - alpha)(1 - CRRA) - 1))
        
        '''
        9.Algebraically solve FOC-n for n_t, and use EndOfPrddvada, EndOfPrddvdD, and cNrmArray to calculate
        n_t for each end-of-period state (a_t, D_t). Call the result nNrmArray
        
        '''
        
        nNrmArray = ((alpha * cNrmArray ** ((1 - alpha) * (1 - CRRA)) * dNrm ** (alpha - CRRA * alpha - 1) +
                      EndOfPrddvdD) / (EndOfPrddvda * kappa * kappa0)) ** (1 / (kappa - 1))
        
        '''
        10. Use the inverted intra-period transition equations to calculate the associate endogenous
        gridpoints (m_t, d_t), calling the resulting arrays mNrmArray and dNrmArray
        
        '''

        mNrmArray = aNrm + cNrmArray + gfunc(nNrmArray)
        dNrmArray = dNrm - nNrmArray
        
        
        '''
        11. The envelope condition for m_t does not take its standard form here.
        Instead, make an array called dvNvrsdmArray that equals the inverse marginal utility function
        evaluated at EndOfPrddvda
        Envelope condition: v'(m) = u'(c(m))   
        
        '''
        dvNvrsdmArray = (EndOfPrddvda * dNrm ** (CRRA - 1) / (1 - alpha)) ** (1 / (-alpha - CRRA + alpha * CRRA)) + aNrm + gfunc(nNrmArray)
        
        
        '''
        12.The envelope condition for d_t: v_d(m_t, d_t) =u_d(c_t, D_t) + EndOfPrddvdD
        Run the array through the inverse marginal utility function, calling dvdNvrsddArray
        
        '''
        dvNvrsddArray = uPD(dvNvrsdmArray, dNrm) + EndOfPrddvdD
        
        
        '''
        13 Concatenate a column of zeros of length DNrmGrid.size onto the left side of mNrmArray, 
        cNrrmArray, and nNrmArray, and a column of DNrmGrid onto dNrmArray
        
        '''

        mNrmArray = np.insert(mNrmArray, 0, 0, 0.0, axis=1)
        cNrmArray = np.insert(cNrmArray, 0, 0.0, axis=1)
        nNrmArray = np.insert(nNrmArray, 0, 0.0, axis=1)
        dNrmArray = np.insert(dNrmArray, 0, 0.0, axis=1)
        
        '''
        14.Construct instances of Curvilinear2DInterp from HARK.interpolation named cFunc and nFunc, 
        the consumption and investment functions. 
        
        '''
        cFunc = Curvilinear2DInterp(cNrmArray, mNrmArray, dNrmArray)
        nFunc = Curvilinear2DInterp(nNrmArray, mNrmArray, dNrmArray)
        
        
        '''
        15.Concatenate a column of zeros of length DNrmGrid.size onto the left side of dvNvrsdmArray
        and dvNvrsddArray. Then construct instances of Curvilinear2DInterp called dvNvrsdmFunc and dvNvrsddFunc
        
        '''
        dvNvrsdmArray = np.insert(dvNvrsdmArray, 0, 0.0, axis=1)
        dvNvrsddArray = np.insert(dvNvrsddArray, 0, 0.0, axis=1)
        dvNvrsdmFunc = Curvilinear2DInterp(dvNvrsdmArray, mNrmArray, dNrmArray)
        dvNvrsddFunc = Curvilinear2DInterp(dvNvrsddArray, mNrmArray, dNrmArray)
        
        
        '''
        16.Make instances of MargValueFunc2D called dvdmFuncNow and dvddFuncNow, using dvNvrsdmFunc
        and dvNvrsddFunc and CRRA
        
        '''
        dvdmFuncNow = MargValueFunc2D(dvNvrsdmFunc, CRRA)
        dvddFuncNow = MargValueFunc2D(dvNvrsddFunc, CRRA)
        
        '''
        17.Construct a solution object that includes the policy functions and marginal value functions
        called solution_now, and return it
        '''

        solution_now = BabyDurableConsumerType(cFunc=cFunc, nFunc=nFunc, dvdmFunc=dvdmFuncNow, dvddFunc=dvddFuncNow)
        return solution_now
