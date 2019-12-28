'''
Consumption-saving models that also include medical spending.
'''
from __future__ import division, print_function
from __future__ import absolute_import
from builtins import str
from builtins import range
import numpy as np
from scipy.optimize import brentq
from HARK import HARKobject
from HARK.utilities import approxLognormal, addDiscreteOutcomeConstantMean, CRRAutilityP_inv,\
                           CRRAutility, CRRAutility_inv, CRRAutility_invP, CRRAutilityPP,\
                           makeGridExpMult, NullFunc
from HARK.ConsumptionSaving.ConsIndShockModel import ConsumerSolution
from HARK.interpolation import BilinearInterpOnInterp1D, TrilinearInterp, BilinearInterp, CubicInterp,\
                               LinearInterp, LowerEnvelope3D, UpperEnvelope, LinearInterpOnInterp1D,\
                               VariableLowerBoundFunc3D
from HARK.ConsumptionSaving.ConsGenIncProcessModel import ConsGenIncProcessSolver,\
                            PersistentShockConsumerType, ValueFunc2D, MargValueFunc2D,\
                            MargMargValueFunc2D, VariableLowerBoundFunc2D
from copy import deepcopy

utility_inv   = CRRAutility_inv
utilityP_inv  = CRRAutilityP_inv
utility       = CRRAutility
utility_invP  = CRRAutility_invP
utilityPP     = CRRAutilityPP


class MedShockPolicyFunc(HARKobject):
    '''
    Class for representing the policy function in the medical shocks model: opt-
    imal consumption and medical care for given market resources, permanent income,
    and medical need shock.  Always obeys Con + MedPrice*Med = optimal spending.
    '''
    distance_criteria = ['xFunc','cFunc','MedPrice']

    def __init__(self,xFunc,xLvlGrid,MedShkGrid,MedPrice,CRRAcon,CRRAmed,xLvlCubicBool=False,
                 MedShkCubicBool=False):
        '''
        Make a new MedShockPolicyFunc.

        Parameters
        ----------
        xFunc : function
            Optimal total spending as a function of market resources, permanent
            income, and the medical need shock.
        xLvlGrid : np.array
            1D array of total expenditure levels.
        MedShkGrid : np.array
            1D array of medical shocks.
        MedPrice : float
            Relative price of a unit of medical care.
        CRRAcon : float
            Coefficient of relative risk aversion for consumption.
        CRRAmed : float
            Coefficient of relative risk aversion for medical care.
        xLvlCubicBool : boolean
            Indicator for whether cubic spline interpolation (rather than linear)
            should be used in the xLvl dimension.
        MedShkCubicBool : boolean
            Indicator for whether bicubic interpolation should be used; only
            operative when xLvlCubicBool=True.

        Returns
        -------
        None
        '''
        # Store some of the inputs in self
        self.MedPrice = MedPrice
        self.xFunc = xFunc

        # Calculate optimal consumption at each combination of mLvl and MedShk.
        cLvlGrid = np.zeros((xLvlGrid.size,MedShkGrid.size)) # Initialize consumption grid
        for i in range(xLvlGrid.size):
            xLvl = xLvlGrid[i]
            for j in range(MedShkGrid.size):
                MedShk = MedShkGrid[j]
                if xLvl == 0: # Zero consumption when mLvl = 0
                    cLvl = 0.0
                elif MedShk == 0: # All consumption when MedShk = 0
                    cLvl = xLvl
                else:
                    optMedZeroFunc = lambda c : (MedShk/MedPrice)**(-1.0/CRRAcon)*\
                                     ((xLvl-c)/MedPrice)**(CRRAmed/CRRAcon) - c
                    cLvl = brentq(optMedZeroFunc,0.0,xLvl) # Find solution to FOC
                cLvlGrid[i,j] = cLvl

        # Construct the consumption function and medical care function
        if xLvlCubicBool:
            if MedShkCubicBool:
                raise NotImplementedError()('Bicubic interpolation not yet implemented')
            else:
                xLvlGrid_tiled   = np.tile(np.reshape(xLvlGrid,(xLvlGrid.size,1)),
                                           (1,MedShkGrid.size))
                MedShkGrid_tiled = np.tile(np.reshape(MedShkGrid,(1,MedShkGrid.size)),
                                           (xLvlGrid.size,1))
                dfdx = (CRRAmed/(CRRAcon*MedPrice))*(MedShkGrid_tiled/MedPrice)**(-1.0/CRRAcon)*\
                       ((xLvlGrid_tiled - cLvlGrid)/MedPrice)**(CRRAmed/CRRAcon - 1.0)
                dcdx = dfdx/(dfdx + 1.0)
                dcdx[0,:] = dcdx[1,:] # approximation; function goes crazy otherwise
                dcdx[:,0] = 1.0 # no Med when MedShk=0, so all x is c
                cFromxFunc_by_MedShk = []
                for j in range(MedShkGrid.size):
                    cFromxFunc_by_MedShk.append(CubicInterp(xLvlGrid,cLvlGrid[:,j],dcdx[:,j]))
                cFunc = LinearInterpOnInterp1D(cFromxFunc_by_MedShk,MedShkGrid)
        else:
            cFunc = BilinearInterp(cLvlGrid,xLvlGrid,MedShkGrid)
        self.cFunc = cFunc

    def __call__(self,mLvl,pLvl,MedShk):
        '''
        Evaluate optimal consumption and medical care at given levels of market
        resources, permanent income, and medical need shocks.

        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        pLvl : np.array
            Permanent income levels; should be same size as mLvl.
        MedShk : np.array
            Medical need shocks; should be same size as mLvl.

        Returns
        -------
        cLvl : np.array
            Optimal consumption for each point in (xLvl,MedShk).
        Med : np.array
            Optimal medical care for each point in (xLvl,MedShk).
        '''
        xLvl = self.xFunc(mLvl,pLvl,MedShk)
        cLvl = self.cFunc(xLvl,MedShk)
        Med  = (xLvl-cLvl)/self.MedPrice
        return cLvl,Med

    def derivativeX(self,mLvl,pLvl,MedShk):
        '''
        Evaluate the derivative of consumption and medical care with respect to
        market resources at given levels of market resources, permanent income,
        and medical need shocks.

        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        pLvl : np.array
            Permanent income levels; should be same size as mLvl.
        MedShk : np.array
            Medical need shocks; should be same size as mLvl.

        Returns
        -------
        dcdm : np.array
            Derivative of consumption with respect to market resources for each
            point in (xLvl,MedShk).
        dMeddm : np.array
            Derivative of medical care with respect to market resources for each
            point in (xLvl,MedShk).
        '''
        xLvl = self.xFunc(mLvl,pLvl,MedShk)
        dxdm = self.xFunc.derivativeX(mLvl,pLvl,MedShk)
        dcdx = self.cFunc.derivativeX(xLvl,MedShk)
        dcdm = dxdm*dcdx
        dMeddm = (dxdm - dcdm)/self.MedPrice
        return dcdm,dMeddm

    def derivativeY(self,mLvl,pLvl,MedShk):
        '''
        Evaluate the derivative of consumption and medical care with respect to
        permanent income at given levels of market resources, permanent income,
        and medical need shocks.

        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        pLvl : np.array
            Permanent income levels; should be same size as mLvl.
        MedShk : np.array
            Medical need shocks; should be same size as mLvl.

        Returns
        -------
        dcdp : np.array
            Derivative of consumption with respect to permanent income for each
            point in (xLvl,MedShk).
        dMeddp : np.array
            Derivative of medical care with respect to permanent income for each
            point in (xLvl,MedShk).
        '''
        xLvl = self.xFunc(mLvl,pLvl,MedShk)
        dxdp = self.xFunc.derivativeY(mLvl,pLvl,MedShk)
        dcdx = self.cFunc.derivativeX(xLvl,MedShk)
        dcdp = dxdp*dcdx
        dMeddp = (dxdp - dcdp)/self.MedPrice
        return dcdp,dMeddp

    def derivativeZ(self,mLvl,pLvl,MedShk):
        '''
        Evaluate the derivative of consumption and medical care with respect to
        medical need shock at given levels of market resources, permanent income,
        and medical need shocks.

        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        pLvl : np.array
            Permanent income levels; should be same size as mLvl.
        MedShk : np.array
            Medical need shocks; should be same size as mLvl.

        Returns
        -------
        dcdShk : np.array
            Derivative of consumption with respect to medical need for each
            point in (xLvl,MedShk).
        dMeddShk : np.array
            Derivative of medical care with respect to medical need for each
            point in (xLvl,MedShk).
        '''
        xLvl = self.xFunc(mLvl,pLvl,MedShk)
        dxdShk = self.xFunc.derivativeZ(mLvl,pLvl,MedShk)
        dcdx = self.cFunc.derivativeX(xLvl,MedShk)
        dcdShk = dxdShk*dcdx + self.cFunc.derivativeY(xLvl,MedShk)
        dMeddShk = (dxdShk - dcdShk)/self.MedPrice
        return dcdShk,dMeddShk


class cThruXfunc(HARKobject):
    '''
    Class for representing consumption function derived from total expenditure
    and consumption.
    '''
    distance_criteria = ['xFunc','cFunc']

    def __init__(self,xFunc,cFunc):
        '''
        Make a new instance of MedFromXfunc.

        Parameters
        ----------
        xFunc : function
            Optimal total spending as a function of market resources, permanent
            income, and the medical need shock.
        cFunc : function
            Optimal consumption as a function of total spending and the medical
            need shock.

        Returns
        -------
        None
        '''
        self.xFunc = xFunc
        self.cFunc = cFunc

    def __call__(self,mLvl,pLvl,MedShk):
        '''
        Evaluate optimal consumption at given levels of market resources, perma-
        nent income, and medical need shocks.

        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        pLvl : np.array
            Permanent income levels; should be same size as mLvl.
        MedShk : np.array
            Medical need shocks; should be same size as mLvl.

        Returns
        -------
        cLvl : np.array
            Optimal consumption for each point in (xLvl,MedShk).
        '''
        xLvl = self.xFunc(mLvl,pLvl,MedShk)
        cLvl = self.cFunc(xLvl,MedShk)
        return cLvl

    def derivativeX(self,mLvl,pLvl,MedShk):
        '''
        Evaluate the derivative of consumption with respect to market resources
        at given levels of market resources, permanent income, and medical need
        shocks.

        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        pLvl : np.array
            Permanent income levels; should be same size as mLvl.
        MedShk : np.array
            Medical need shocks; should be same size as mLvl.

        Returns
        -------
        dcdm : np.array
            Derivative of consumption with respect to market resources for each
            point in (xLvl,MedShk).
        '''
        xLvl = self.xFunc(mLvl,pLvl,MedShk)
        dxdm = self.xFunc.derivativeX(mLvl,pLvl,MedShk)
        dcdx = self.cFunc.derivativeX(xLvl,MedShk)
        dcdm = dxdm*dcdx
        return dcdm

    def derivativeY(self,mLvl,pLvl,MedShk):
        '''
        Evaluate the derivative of consumption and medical care with respect to
        permanent income at given levels of market resources, permanent income,
        and medical need shocks.

        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        pLvl : np.array
            Permanent income levels; should be same size as mLvl.
        MedShk : np.array
            Medical need shocks; should be same size as mLvl.

        Returns
        -------
        dcdp : np.array
            Derivative of consumption with respect to permanent income for each
            point in (xLvl,MedShk).
        '''
        xLvl = self.xFunc(mLvl,pLvl,MedShk)
        dxdp = self.xFunc.derivativeY(mLvl,pLvl,MedShk)
        dcdx = self.cFunc.derivativeX(xLvl,MedShk)
        dcdp = dxdp*dcdx
        return dcdp

    def derivativeZ(self,mLvl,pLvl,MedShk):
        '''
        Evaluate the derivative of consumption and medical care with respect to
        medical need shock at given levels of market resources, permanent income,
        and medical need shocks.

        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        pLvl : np.array
            Permanent income levels; should be same size as mLvl.
        MedShk : np.array
            Medical need shocks; should be same size as mLvl.

        Returns
        -------
        dcdShk : np.array
            Derivative of consumption with respect to medical need for each
            point in (xLvl,MedShk).
        '''
        xLvl = self.xFunc(mLvl,pLvl,MedShk)
        dxdShk = self.xFunc.derivativeZ(mLvl,pLvl,MedShk)
        dcdx = self.cFunc.derivativeX(xLvl,MedShk)
        dcdShk = dxdShk*dcdx + self.cFunc.derivativeY(xLvl,MedShk)
        return dcdShk

class MedThruXfunc(HARKobject):
    '''
    Class for representing medical care function derived from total expenditure
    and consumption.
    '''
    distance_criteria = ['xFunc','cFunc','MedPrice']

    def __init__(self,xFunc,cFunc,MedPrice):
        '''
        Make a new instance of MedFromXfunc.

        Parameters
        ----------
        xFunc : function
            Optimal total spending as a function of market resources, permanent
            income, and the medical need shock.
        cFunc : function
            Optimal consumption as a function of total spending and the medical
            need shock.
        MedPrice : float
            Relative price of a unit of medical care.

        Returns
        -------
        None
        '''
        self.xFunc = xFunc
        self.cFunc = cFunc
        self.MedPrice = MedPrice

    def __call__(self,mLvl,pLvl,MedShk):
        '''
        Evaluate optimal medical care at given levels of market resources,
        permanent income, and medical need shock.

        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        pLvl : np.array
            Permanent income levels; should be same size as mLvl.
        MedShk : np.array
            Medical need shocks; should be same size as mLvl.

        Returns
        -------
        Med : np.array
            Optimal medical care for each point in (xLvl,MedShk).
        '''
        xLvl = self.xFunc(mLvl,pLvl,MedShk)
        Med  = (xLvl-self.cFunc(xLvl,MedShk))/self.MedPrice
        return Med

    def derivativeX(self,mLvl,pLvl,MedShk):
        '''
        Evaluate the derivative of consumption and medical care with respect to
        market resources at given levels of market resources, permanent income,
        and medical need shocks.

        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        pLvl : np.array
            Permanent income levels; should be same size as mLvl.
        MedShk : np.array
            Medical need shocks; should be same size as mLvl.

        Returns
        -------
        dcdm : np.array
            Derivative of consumption with respect to market resources for each
            point in (xLvl,MedShk).
        dMeddm : np.array
            Derivative of medical care with respect to market resources for each
            point in (xLvl,MedShk).
        '''
        xLvl = self.xFunc(mLvl,pLvl,MedShk)
        dxdm = self.xFunc.derivativeX(mLvl,pLvl,MedShk)
        dcdx = self.cFunc.derivativeX(xLvl,MedShk)
        dcdm = dxdm*dcdx
        dMeddm = (dxdm - dcdm)/self.MedPrice
        return dcdm,dMeddm

    def derivativeY(self,mLvl,pLvl,MedShk):
        '''
        Evaluate the derivative of medical care with respect to permanent income
        at given levels of market resources, permanent income, and medical need
        shocks.

        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        pLvl : np.array
            Permanent income levels; should be same size as mLvl.
        MedShk : np.array
            Medical need shocks; should be same size as mLvl.

        Returns
        -------
        dMeddp : np.array
            Derivative of medical care with respect to permanent income for each
            point in (xLvl,MedShk).
        '''
        xLvl = self.xFunc(mLvl,pLvl,MedShk)
        dxdp = self.xFunc.derivativeY(mLvl,pLvl,MedShk)
        dMeddp = (dxdp - dxdp*self.cFunc.derivativeX(xLvl,MedShk))/self.MedPrice
        return dMeddp

    def derivativeZ(self,mLvl,pLvl,MedShk):
        '''
        Evaluate the derivative of medical care with respect to medical need
        shock at given levels of market resources, permanent income, and medical
        need shocks.

        Parameters
        ----------
        mLvl : np.array
            Market resource levels.
        pLvl : np.array
            Permanent income levels; should be same size as mLvl.
        MedShk : np.array
            Medical need shocks; should be same size as mLvl.

        Returns
        -------
        dMeddShk : np.array
            Derivative of medical care with respect to medical need for each
            point in (xLvl,MedShk).
        '''
        xLvl = self.xFunc(mLvl,pLvl,MedShk)
        dxdShk = self.xFunc.derivativeZ(mLvl,pLvl,MedShk)
        dcdx = self.cFunc.derivativeX(xLvl,MedShk)
        dcdShk = dxdShk*dcdx + self.cFunc.derivativeY(xLvl,MedShk)
        dMeddShk = (dxdShk - dcdShk)/self.MedPrice
        return dMeddShk


###############################################################################

class MedShockConsumerType(PersistentShockConsumerType):
    '''
    A class to represent agents who consume two goods: ordinary composite consumption
    and medical care; both goods yield CRRAutility, and the coefficients on the
    goods might be different.  Agents expect to receive shocks to permanent and
    transitory income as well as multiplicative shocks to utility from medical care.
    '''
    shock_vars_ = PersistentShockConsumerType.shock_vars_ + ['MedShkNow']

    def __init__(self,cycles=1,time_flow=True,**kwds):
        '''
        Instantiate a new ConsumerType with given data, and construct objects
        to be used during solution (income distribution, assets grid, etc).
        See ConsumerParameters.init_med_shock for a dictionary of the keywords
        that should be passed to the constructor.

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
        PersistentShockConsumerType.__init__(self,cycles=cycles,**kwds)
        self.solveOnePeriod = solveConsMedShock # Choose correct solver
        self.addToTimeInv('CRRAmed')
        self.addToTimeVary('MedPrice')
        
    def preSolve(self):
        self.updateSolutionTerminal()

    def update(self):
        '''
        Update the income process, the assets grid, the permanent income grid,
        the medical shock distribution, and the terminal solution.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        self.updateIncomeProcess()
        self.updateAssetsGrid()
        self.updatepLvlNextFunc()
        self.updatepLvlGrid()
        self.updateMedShockProcess()
        self.updateSolutionTerminal()

    def updateMedShockProcess(self):
        '''
        Constructs discrete distributions of medical preference shocks for each
        period in the cycle.  Distributions are saved as attribute MedShkDstn,
        which is added to time_vary.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        MedShkDstn = [] # empty list for medical shock distribution each period
        for t in range(self.T_cycle):
            MedShkAvgNow  = self.MedShkAvg[t] # get shock distribution parameters
            MedShkStdNow  = self.MedShkStd[t]
            MedShkDstnNow = approxLognormal(mu=np.log(MedShkAvgNow)-0.5*MedShkStdNow**2,\
                            sigma=MedShkStdNow,N=self.MedShkCount, tail_N=self.MedShkCountTail,
                            tail_bound=[0,0.9])
            MedShkDstnNow = addDiscreteOutcomeConstantMean(MedShkDstnNow,0.0,0.0,sort=True) # add point at zero with no probability
            MedShkDstn.append(MedShkDstnNow)
        self.MedShkDstn = MedShkDstn
        self.addToTimeVary('MedShkDstn')

    def updateSolutionTerminal(self):
        '''
        Update the terminal period solution for this type.  Similar to other models,
        optimal behavior involves spending all available market resources; however,
        the agent must split his resources between consumption and medical care.

        Parameters
        ----------
        None

        Returns:
        --------
        None
        '''
        # Take last period data, whichever way time is flowing
        if self.time_flow:
            MedPrice = self.MedPrice[-1]
            MedShkVals = self.MedShkDstn[-1][1]
            MedShkPrbs = self.MedShkDstn[-1][0]
        else:
            MedPrice = self.MedPrice[0]
            MedShkVals = self.MedShkDstn[0][1]
            MedShkPrbs = self.MedShkDstn[0][0]

        # Initialize grids of medical need shocks, market resources, and optimal consumption
        MedShkGrid = MedShkVals
        xLvlMin = np.min(self.aXtraGrid)*np.min(self.pLvlGrid)
        xLvlMax = np.max(self.aXtraGrid)*np.max(self.pLvlGrid)
        xLvlGrid = makeGridExpMult(xLvlMin, xLvlMax, 3*self.aXtraGrid.size, 8)
        trivial_grid = np.array([0.0,1.0]) # Trivial grid

        # Make the policy functions for the terminal period
        xFunc_terminal = TrilinearInterp(np.array([[[0.0,0.0],[0.0,0.0]],[[1.0,1.0],[1.0,1.0]]]),\
                         trivial_grid,trivial_grid,trivial_grid)
        policyFunc_terminal = MedShockPolicyFunc(xFunc_terminal,xLvlGrid,MedShkGrid,MedPrice,
                              self.CRRA,self.CRRAmed,xLvlCubicBool=self.CubicBool)
        cFunc_terminal = cThruXfunc(xFunc_terminal,policyFunc_terminal.cFunc)
        MedFunc_terminal = MedThruXfunc(xFunc_terminal,policyFunc_terminal.cFunc,MedPrice)

        # Calculate optimal consumption on a grid of market resources and medical shocks
        mLvlGrid = xLvlGrid
        mLvlGrid_tiled = np.tile(np.reshape(mLvlGrid,(mLvlGrid.size,1)),(1,MedShkGrid.size))
        pLvlGrid_tiled = np.ones_like(mLvlGrid_tiled) # permanent income irrelevant in terminal period
        MedShkGrid_tiled = np.tile(np.reshape(MedShkVals,(1,MedShkGrid.size)),(mLvlGrid.size,1))
        cLvlGrid,MedGrid = policyFunc_terminal(mLvlGrid_tiled,pLvlGrid_tiled,MedShkGrid_tiled)

        # Integrate marginal value across shocks to get expected marginal value
        vPgrid = cLvlGrid**(-self.CRRA)
        vPgrid[np.isinf(vPgrid)] = 0.0 # correct for issue at bottom edges
        PrbGrid = np.tile(np.reshape(MedShkPrbs,(1,MedShkGrid.size)),(mLvlGrid.size,1))
        vP_expected = np.sum(vPgrid*PrbGrid,axis=1)

        # Construct the marginal (marginal) value function for the terminal period
        vPnvrs = vP_expected**(-1.0/self.CRRA)
        vPnvrs[0] = 0.0
        vPnvrsFunc = BilinearInterp(np.tile(np.reshape(vPnvrs,(vPnvrs.size,1)),
                     (1,trivial_grid.size)),mLvlGrid,trivial_grid)
        vPfunc_terminal = MargValueFunc2D(vPnvrsFunc,self.CRRA)
        vPPfunc_terminal = MargMargValueFunc2D(vPnvrsFunc,self.CRRA)

        # Integrate value across shocks to get expected value
        vGrid = utility(cLvlGrid,gam=self.CRRA) + MedShkGrid_tiled*utility(MedGrid,gam=self.CRRAmed)
        vGrid[:,0] = utility(cLvlGrid[:,0],gam=self.CRRA) # correct for issue when MedShk=0
        vGrid[np.isinf(vGrid)] = 0.0 # correct for issue at bottom edges
        v_expected = np.sum(vGrid*PrbGrid,axis=1)

        # Construct the value function for the terminal period
        vNvrs = utility_inv(v_expected,gam=self.CRRA)
        vNvrs[0] = 0.0
        vNvrsP = vP_expected*utility_invP(v_expected,gam=self.CRRA) # NEED TO FIGURE OUT MPC MAX IN THIS MODEL
        vNvrsP[0] = 0.0
        tempFunc = CubicInterp(mLvlGrid,vNvrs,vNvrsP)
        vNvrsFunc = LinearInterpOnInterp1D([tempFunc,tempFunc],trivial_grid)
        vFunc_terminal = ValueFunc2D(vNvrsFunc,self.CRRA)

        # Make the terminal period solution
        self.solution_terminal.cFunc = cFunc_terminal
        self.solution_terminal.MedFunc = MedFunc_terminal
        self.solution_terminal.policyFunc = policyFunc_terminal
        self.solution_terminal.vPfunc = vPfunc_terminal
        self.solution_terminal.vFunc = vFunc_terminal
        self.solution_terminal.vPPfunc = vPPfunc_terminal
        self.solution_terminal.hNrm = 0.0 # Don't track normalized human wealth
        self.solution_terminal.hLvl = lambda p : np.zeros_like(p) # But do track absolute human wealth by permanent income
        self.solution_terminal.mLvlMin = lambda p : np.zeros_like(p) # And minimum allowable market resources by perm inc


    def updatepLvlGrid(self):
        '''
        Update the grid of permanent income levels.  Currently only works for
        infinite horizon models (cycles=0) and lifecycle models (cycles=1).  Not
        clear what to do about cycles>1.  Identical to version in persistent
        shocks model, but pLvl=0 is manually added to the grid (because there is
        no closed form lower-bounding cFunc for pLvl=0).

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        # Run basic version of this method
        PersistentShockConsumerType.updatepLvlGrid(self)
        for j in range(len(self.pLvlGrid)): # Then add 0 to the bottom of each pLvlGrid
            this_grid = self.pLvlGrid[j]
            self.pLvlGrid[j] = np.insert(this_grid,0,0.0001)

    def getShocks(self):
        '''
        Gets permanent and transitory income shocks for this period as well as medical need shocks
        and the price of medical care.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        PersistentShockConsumerType.getShocks(self) # Get permanent and transitory income shocks
        MedShkNow = np.zeros(self.AgentCount) # Initialize medical shock array
        MedPriceNow = np.zeros(self.AgentCount) # Initialize relative price array
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            N = np.sum(these)
            if N > 0:
                MedShkAvg = self.MedShkAvg[t]
                MedShkStd = self.MedShkStd[t]
                MedPrice  = self.MedPrice[t]
                MedShkNow[these] = self.RNG.permutation(approxLognormal(N,mu=np.log(MedShkAvg)-0.5*MedShkStd**2,sigma=MedShkStd)[1])
                MedPriceNow[these] = MedPrice
        self.MedShkNow = MedShkNow
        self.MedPriceNow = MedPriceNow

    def getControls(self):
        '''
        Calculates consumption and medical care for each consumer of this type using the consumption
        and medical care functions.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        cLvlNow = np.zeros(self.AgentCount) + np.nan
        MedNow  = np.zeros(self.AgentCount) + np.nan
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            cLvlNow[these], MedNow[these] = self.solution[t].policyFunc(self.mLvlNow[these],self.pLvlNow[these],self.MedShkNow[these])
        self.cLvlNow = cLvlNow
        self.MedNow  = MedNow
        return None

    def getPostStates(self):
        '''
        Calculates end-of-period assets for each consumer of this type.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.aLvlNow = self.mLvlNow - self.cLvlNow - self.MedPriceNow*self.MedNow
        return None

###############################################################################

class ConsMedShockSolver(ConsGenIncProcessSolver):
    '''
    Class for solving the one period problem for the "medical shocks" model, in
    which consumers receive shocks to permanent and transitory income as well as
    shocks to "medical need"-- multiplicative utility shocks for a second good.
    '''
    def __init__(self,solution_next,IncomeDstn,MedShkDstn,LivPrb,DiscFac,CRRA,CRRAmed,Rfree,MedPrice,
                 pLvlNextFunc,BoroCnstArt,aXtraGrid,pLvlGrid,vFuncBool,CubicBool):
        '''
        Constructor for a new solver for a one period problem with idiosyncratic
        shocks to permanent and transitory income and shocks to medical need.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncomeDstn : [np.array]
            A list containing three arrays of floats, representing a discrete
            approximation to the income process between the period being solved
            and the one immediately following (in solution_next). Order: event
            probabilities, permanent shocks, transitory shocks.
        MedShkDstn : [np.array]
            Discrete distribution of the multiplicative utility shifter for med-
            ical care. Order: probabilities, preference shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.
        DiscFac : float
            Intertemporal discount factor for future utility.
        CRRA : float
            Coefficient of relative risk aversion for composite consumption.
        CRRAmed : float
            Coefficient of relative risk aversion for medical care.
        Rfree : float
            Risk free interest factor on end-of-period assets.
        MedPrice : float
            Price of unit of medical care relative to unit of consumption.
        pLvlNextFunc : float
            Expected permanent income next period as a function of current pLvl.
        BoroCnstArt: float or None
            Borrowing constraint for the minimum allowable assets to end the
            period with.
        aXtraGrid: np.array
            Array of "extra" end-of-period (normalized) asset values-- assets
            above the absolute minimum acceptable level.
        pLvlGrid: np.array
            Array of permanent income levels at which to solve the problem.
        vFuncBool: boolean
            An indicator for whether the value function should be computed and
            included in the reported solution.
        CubicBool: boolean
            An indicator for whether the solver should use cubic or linear inter-
            polation.

        Returns
        -------
        None
        '''
        ConsGenIncProcessSolver.__init__(self,solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,
                 pLvlNextFunc,BoroCnstArt,aXtraGrid,pLvlGrid,vFuncBool,CubicBool)
        self.MedShkDstn = MedShkDstn
        self.MedPrice   = MedPrice
        self.CRRAmed    = CRRAmed

    def setAndUpdateValues(self,solution_next,IncomeDstn,LivPrb,DiscFac):
        '''
        Unpacks some of the inputs (and calculates simple objects based on them),
        storing the results in self for use by other methods.  These include:
        income shocks and probabilities, medical shocks and probabilities, next
        period's marginal value function (etc), the probability of getting the
        worst income shock next period, the patience factor, human wealth, and
        the bounding MPCs.

        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncomeDstn : [np.array]
            A list containing three arrays of floats, representing a discrete
            approximation to the income process between the period being solved
            and the one immediately following (in solution_next). Order: event
            probabilities, permanent shocks, transitory shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.
        DiscFac : float
            Intertemporal discount factor for future utility.

        Returns
        -------
        None
        '''
        # Run basic version of this method
        ConsGenIncProcessSolver.setAndUpdateValues(self,self.solution_next,self.IncomeDstn,
                                                     self.LivPrb,self.DiscFac)

        # Also unpack the medical shock distribution
        self.MedShkPrbs = self.MedShkDstn[0]
        self.MedShkVals = self.MedShkDstn[1]

    def defUtilityFuncs(self):
        '''
        Defines CRRA utility function for this period (and its derivatives,
        and their inverses), saving them as attributes of self for other methods
        to use.  Extends version from ConsIndShock models by also defining inverse
        marginal utility function over medical care.

        Parameters
        ----------
        none

        Returns
        -------
        none
        '''
        ConsGenIncProcessSolver.defUtilityFuncs(self) # Do basic version
        self.uMedPinv = lambda Med : utilityP_inv(Med,gam=self.CRRAmed)
        self.uMed     = lambda Med : utility(Med,gam=self.CRRAmed)
        self.uMedPP   = lambda Med : utilityPP(Med,gam=self.CRRAmed)

    def defBoroCnst(self,BoroCnstArt):
        '''
        Defines the constrained portion of the consumption function as cFuncNowCnst,
        an attribute of self.  Uses the artificial and natural borrowing constraints.

        Parameters
        ----------
        BoroCnstArt : float or None
            Borrowing constraint for the minimum allowable (normalized) assets
            to end the period with.  If it is less than the natural borrowing
            constraint at a particular permanent income level, then it is irrelevant;
            BoroCnstArt=None indicates no artificial borrowing constraint.

        Returns
        -------
        None
        '''
        # Find minimum allowable end-of-period assets at each permanent income level
        PermIncMinNext = self.PermShkMinNext*self.pLvlNextFunc(self.pLvlGrid)
        IncLvlMinNext  = PermIncMinNext*self.TranShkMinNext
        aLvlMin = (self.solution_next.mLvlMin(PermIncMinNext) - IncLvlMinNext)/self.Rfree

        # Make a function for the natural borrowing constraint by permanent income
        BoroCnstNat = LinearInterp(np.insert(self.pLvlGrid,0,0.0),np.insert(aLvlMin,0,0.0))
        self.BoroCnstNat = BoroCnstNat

        # Define the minimum allowable level of market resources by permanent income
        if self.BoroCnstArt is not None:
            BoroCnstArt = LinearInterp([0.0,1.0],[0.0,self.BoroCnstArt])
            self.mLvlMinNow = UpperEnvelope(BoroCnstNat,BoroCnstArt)
        else:
            self.mLvlMinNow = BoroCnstNat

        # Make the constrained total spending function: spend all market resources
        trivial_grid = np.array([0.0,1.0]) # Trivial grid
        spendAllFunc = TrilinearInterp(np.array([[[0.0,0.0],[0.0,0.0]],[[1.0,1.0],[1.0,1.0]]]),\
                       trivial_grid,trivial_grid,trivial_grid)
        self.xFuncNowCnst = VariableLowerBoundFunc3D(spendAllFunc,self.mLvlMinNow)

        self.mNrmMinNow = 0.0 # Needs to exist so as not to break when solution is created
        self.MPCmaxEff  = 0.0 # Actually might vary by p, but no use formulating as a function

    def getPointsForInterpolation(self,EndOfPrdvP,aLvlNow):
        '''
        Finds endogenous interpolation points (x,m) for the expenditure function.

        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal values.
        aLvlNow : np.array
            Array of end-of-period asset values that yield the marginal values
            in EndOfPrdvP.

        Returns
        -------
        x_for_interpolation : np.array
            Total expenditure points for interpolation.
        m_for_interpolation : np.array
            Corresponding market resource points for interpolation.
        p_for_interpolation : np.array
            Corresponding permanent income points for interpolation.
        '''
        # Get size of each state dimension
        mCount      = aLvlNow.shape[1]
        pCount      = aLvlNow.shape[0]
        MedCount = self.MedShkVals.size

        # Calculate endogenous gridpoints and controls
        cLvlNow = np.tile(np.reshape(self.uPinv(EndOfPrdvP),(1,pCount,mCount)),(MedCount,1,1))
        MedBaseNow = np.tile(np.reshape(self.uMedPinv(self.MedPrice*EndOfPrdvP),(1,pCount,mCount)),
                             (MedCount,1,1))
        MedShkVals_tiled = np.tile(np.reshape(self.MedShkVals**(1.0/self.CRRAmed),(MedCount,1,1)),
                                   (1,pCount,mCount))
        MedLvlNow = MedShkVals_tiled*MedBaseNow
        aLvlNow_tiled = np.tile(np.reshape(aLvlNow,(1,pCount,mCount)),(MedCount,1,1))
        xLvlNow = cLvlNow + self.MedPrice*MedLvlNow
        mLvlNow = xLvlNow + aLvlNow_tiled

        # Limiting consumption is zero as m approaches the natural borrowing constraint
        x_for_interpolation = np.concatenate((np.zeros((MedCount,pCount,1)),xLvlNow),axis=-1)
        temp = np.tile(self.BoroCnstNat(np.reshape(self.pLvlGrid,(1,self.pLvlGrid.size,1))),
                       (MedCount,1,1))
        m_for_interpolation = np.concatenate((temp,mLvlNow),axis=-1)

        # Make a 3D array of permanent income for interpolation
        p_for_interpolation = np.tile(np.reshape(self.pLvlGrid,(1,pCount,1)),(MedCount,1,mCount+1))

        # Store for use by cubic interpolator
        self.cLvlNow = cLvlNow
        self.MedLvlNow = MedLvlNow
        self.MedShkVals_tiled = np.tile(np.reshape(self.MedShkVals,(MedCount,1,1)),(1,pCount,mCount))

        return x_for_interpolation, m_for_interpolation, p_for_interpolation

    def usePointsForInterpolation(self,xLvl,mLvl,pLvl,MedShk,interpolator):
        '''
        Constructs a basic solution for this period, including the consumption
        function and marginal value function.

        Parameters
        ----------
        xLvl : np.array
            Total expenditure points for interpolation.
        mLvl : np.array
            Corresponding market resource points for interpolation.
        pLvl : np.array
            Corresponding permanent income level points for interpolation.
        MedShk : np.array
            Corresponding medical need shocks for interpolation.
        interpolator : function
            A function that constructs and returns a consumption function.

        Returns
        -------
        solution_now : ConsumerSolution
            The solution to this period's consumption-saving problem, with a
            consumption function, marginal value function, and minimum m.
        '''
        # Construct the unconstrained total expenditure function
        xFuncNowUnc = interpolator(mLvl,pLvl,MedShk,xLvl)
        xFuncNowCnst = self.xFuncNowCnst
        xFuncNow = LowerEnvelope3D(xFuncNowUnc,xFuncNowCnst)

        # Transform the expenditure function into policy functions for consumption and medical care
        aug_factor = 2
        xLvlGrid = makeGridExpMult(np.min(xLvl),np.max(xLvl),aug_factor*self.aXtraGrid.size,8)
        policyFuncNow = MedShockPolicyFunc(xFuncNow,xLvlGrid,self.MedShkVals,self.MedPrice,
                        self.CRRA,self.CRRAmed,xLvlCubicBool=self.CubicBool)
        cFuncNow = cThruXfunc(xFuncNow,policyFuncNow.cFunc)
        MedFuncNow = MedThruXfunc(xFuncNow,policyFuncNow.cFunc,self.MedPrice)

        # Make the marginal value function (and the value function if vFuncBool=True)
        vFuncNow, vPfuncNow = self.makevAndvPfuncs(policyFuncNow)

        # Pack up the solution and return it
        solution_now = ConsumerSolution(cFunc=cFuncNow, vFunc=vFuncNow, vPfunc=vPfuncNow,\
                                        mNrmMin=self.mNrmMinNow)
        solution_now.MedFunc = MedFuncNow
        solution_now.policyFunc = policyFuncNow
        return solution_now

    def makevAndvPfuncs(self,policyFunc):
        '''
        Constructs the marginal value function for this period.

        Parameters
        ----------
        policyFunc : function
            Consumption and medical care function for this period, defined over
            market resources, permanent income level, and the medical need shock.

        Returns
        -------
        vFunc : function
            Value function for this period, defined over market resources and
            permanent income.
        vPfunc : function
            Marginal value (of market resources) function for this period, defined
            over market resources and permanent income.
        '''
        # Get state dimension sizes
        mCount   = self.aXtraGrid.size
        pCount   = self.pLvlGrid.size
        MedCount = self.MedShkVals.size

        # Make temporary grids to evaluate the consumption function
        temp_grid  = np.tile(np.reshape(self.aXtraGrid,(mCount,1,1)),(1,pCount,MedCount))
        aMinGrid   = np.tile(np.reshape(self.mLvlMinNow(self.pLvlGrid),(1,pCount,1)),
                             (mCount,1,MedCount))
        pGrid      = np.tile(np.reshape(self.pLvlGrid,(1,pCount,1)),(mCount,1,MedCount))
        mGrid      = temp_grid*pGrid + aMinGrid
        if self.pLvlGrid[0] == 0:
            mGrid[:,0,:] = np.tile(np.reshape(self.aXtraGrid,(mCount,1)),(1,MedCount))
        MedShkGrid = np.tile(np.reshape(self.MedShkVals,(1,1,MedCount)),(mCount,pCount,1))
        probsGrid  = np.tile(np.reshape(self.MedShkPrbs,(1,1,MedCount)),(mCount,pCount,1))

        # Get optimal consumption (and medical care) for each state
        cGrid,MedGrid = policyFunc(mGrid,pGrid,MedShkGrid)

        # Calculate expected value by "integrating" across medical shocks
        if self.vFuncBool:
            MedGrid = np.maximum(MedGrid,1e-100) # interpolation error sometimes makes Med < 0 (barely)
            aGrid = np.maximum(mGrid - cGrid - self.MedPrice*MedGrid, aMinGrid) # interpolation error sometimes makes tiny violations
            vGrid = self.u(cGrid) + MedShkGrid*self.uMed(MedGrid) + self.EndOfPrdvFunc(aGrid,pGrid)
            vNow  = np.sum(vGrid*probsGrid,axis=2)

        # Calculate expected marginal value by "integrating" across medical shocks
        vPgrid = self.uP(cGrid)
        vPnow  = np.sum(vPgrid*probsGrid,axis=2)

        # Add vPnvrs=0 at m=mLvlMin to close it off at the bottom (and vNvrs=0)
        mGrid_small = np.concatenate((np.reshape(self.mLvlMinNow(self.pLvlGrid),(1,pCount)),mGrid[:,:,0]))
        vPnvrsNow   = np.concatenate((np.zeros((1,pCount)),self.uPinv(vPnow)))
        if self.vFuncBool:
            vNvrsNow  = np.concatenate((np.zeros((1,pCount)),self.uinv(vNow)),axis=0)
            vNvrsPnow = vPnow*self.uinvP(vNow)
            vNvrsPnow = np.concatenate((np.zeros((1,pCount)),vNvrsPnow),axis=0)

        # Construct the pseudo-inverse value and marginal value functions over mLvl,pLvl
        vPnvrsFunc_by_pLvl = []
        vNvrsFunc_by_pLvl = []
        for j in range(pCount): # Make a pseudo inverse marginal value function for each pLvl
            pLvl = self.pLvlGrid[j]
            m_temp = mGrid_small[:,j] - self.mLvlMinNow(pLvl)
            vPnvrs_temp = vPnvrsNow[:,j]
            vPnvrsFunc_by_pLvl.append(LinearInterp(m_temp,vPnvrs_temp))
            if self.vFuncBool:
                vNvrs_temp  = vNvrsNow[:,j]
                vNvrsP_temp = vNvrsPnow[:,j]
                vNvrsFunc_by_pLvl.append(CubicInterp(m_temp,vNvrs_temp,vNvrsP_temp))
        vPnvrsFuncBase = LinearInterpOnInterp1D(vPnvrsFunc_by_pLvl,self.pLvlGrid)
        vPnvrsFunc = VariableLowerBoundFunc2D(vPnvrsFuncBase,self.mLvlMinNow) # adjust for the lower bound of mLvl
        if self.vFuncBool:
            vNvrsFuncBase  = LinearInterpOnInterp1D(vNvrsFunc_by_pLvl,self.pLvlGrid)
            vNvrsFunc = VariableLowerBoundFunc2D(vNvrsFuncBase,self.mLvlMinNow) # adjust for the lower bound of mLvl

        # "Re-curve" the (marginal) value function
        vPfunc = MargValueFunc2D(vPnvrsFunc,self.CRRA)
        if self.vFuncBool:
            vFunc = ValueFunc2D(vNvrsFunc,self.CRRA)
        else:
            vFunc = NullFunc()

        return vFunc, vPfunc

    def makeLinearxFunc(self,mLvl,pLvl,MedShk,xLvl):
        '''
        Constructs the (unconstrained) expenditure function for this period using
        bilinear interpolation (over permanent income and the medical shock) among
        an array of linear interpolations over market resources.

        Parameters
        ----------
        mLvl : np.array
            Corresponding market resource points for interpolation.
        pLvl : np.array
            Corresponding permanent income level points for interpolation.
        MedShk : np.array
            Corresponding medical need shocks for interpolation.
        xLvl : np.array
            Expenditure points for interpolation, corresponding to those in mLvl,
            pLvl, and MedShk.

        Returns
        -------
        xFuncUnc : BilinearInterpOnInterp1D
            Unconstrained total expenditure function for this period.
        '''
        # Get state dimensions
        pCount = mLvl.shape[1]
        MedCount = mLvl.shape[0]

        # Loop over each permanent income level and medical shock and make a linear xFunc
        xFunc_by_pLvl_and_MedShk = [] # Initialize the empty list of lists of 1D xFuncs
        for i in range(pCount):
            temp_list = []
            pLvl_i = pLvl[0,i,0]
            mLvlMin_i = self.BoroCnstNat(pLvl_i)
            for j in range(MedCount):
                m_temp = mLvl[j,i,:] - mLvlMin_i
                x_temp = xLvl[j,i,:]
                temp_list.append(LinearInterp(m_temp,x_temp))
            xFunc_by_pLvl_and_MedShk.append(deepcopy(temp_list))

        # Combine the nested list of linear xFuncs into a single function
        pLvl_temp = pLvl[0,:,0]
        MedShk_temp = MedShk[:,0,0]
        xFuncUncBase = BilinearInterpOnInterp1D(xFunc_by_pLvl_and_MedShk,pLvl_temp,MedShk_temp)
        xFuncUnc = VariableLowerBoundFunc3D(xFuncUncBase,self.BoroCnstNat)
        return xFuncUnc

    def makeCubicxFunc(self,mLvl,pLvl,MedShk,xLvl):
        '''
        Constructs the (unconstrained) expenditure function for this period using
        bilinear interpolation (over permanent income and the medical shock) among
        an array of cubic interpolations over market resources.

        Parameters
        ----------
        mLvl : np.array
            Corresponding market resource points for interpolation.
        pLvl : np.array
            Corresponding permanent income level points for interpolation.
        MedShk : np.array
            Corresponding medical need shocks for interpolation.
        xLvl : np.array
            Expenditure points for interpolation, corresponding to those in mLvl,
            pLvl, and MedShk.

        Returns
        -------
        xFuncUnc : BilinearInterpOnInterp1D
            Unconstrained total expenditure function for this period.
        '''
        # Get state dimensions
        pCount = mLvl.shape[1]
        MedCount = mLvl.shape[0]

        # Calculate the MPC and MPM at each gridpoint
        EndOfPrdvPP = self.DiscFacEff*self.Rfree*self.Rfree*np.sum(self.vPPfuncNext(self.mLvlNext,\
                      self.pLvlNext)*self.ShkPrbs_temp,axis=0)
        EndOfPrdvPP = np.tile(np.reshape(EndOfPrdvPP,(1,pCount,EndOfPrdvPP.shape[1])),(MedCount,1,1))
        dcda        = EndOfPrdvPP/self.uPP(np.array(self.cLvlNow))
        dMedda      = EndOfPrdvPP/(self.MedShkVals_tiled*self.uMedPP(self.MedLvlNow))
        dMedda[0,:,:] = 0.0 # dMedda goes crazy when MedShk=0
        MPC         = dcda/(1.0 + dcda + self.MedPrice*dMedda)
        MPM         = dMedda/(1.0 + dcda + self.MedPrice*dMedda)

        # Convert to marginal propensity to spend
        MPX = MPC + self.MedPrice*MPM
        MPX = np.concatenate((np.reshape(MPX[:,:,0],(MedCount,pCount,1)),MPX),axis=2) # NEED TO CALCULATE MPM AT NATURAL BORROWING CONSTRAINT
        MPX[0,:,0] = self.MPCmaxNow

        # Loop over each permanent income level and medical shock and make a cubic xFunc
        xFunc_by_pLvl_and_MedShk = [] # Initialize the empty list of lists of 1D xFuncs
        for i in range(pCount):
            temp_list = []
            pLvl_i = pLvl[0,i,0]
            mLvlMin_i = self.BoroCnstNat(pLvl_i)
            for j in range(MedCount):
                m_temp = mLvl[j,i,:] - mLvlMin_i
                x_temp = xLvl[j,i,:]
                MPX_temp = MPX[j,i,:]
                temp_list.append(CubicInterp(m_temp,x_temp,MPX_temp))
            xFunc_by_pLvl_and_MedShk.append(deepcopy(temp_list))

        # Combine the nested list of cubic xFuncs into a single function
        pLvl_temp = pLvl[0,:,0]
        MedShk_temp = MedShk[:,0,0]
        xFuncUncBase = BilinearInterpOnInterp1D(xFunc_by_pLvl_and_MedShk,pLvl_temp,MedShk_temp)
        xFuncUnc = VariableLowerBoundFunc3D(xFuncUncBase,self.BoroCnstNat)
        return xFuncUnc

    def makeBasicSolution(self,EndOfPrdvP,aLvl,interpolator):
        '''
        Given end of period assets and end of period marginal value, construct
        the basic solution for this period.

        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal values.
        aLvl : np.array
            Array of end-of-period asset values that yield the marginal values
            in EndOfPrdvP.
        interpolator : function
            A function that constructs and returns a consumption function.

        Returns
        -------
        solution_now : ConsumerSolution
            The solution to this period's consumption-saving problem, with a
            consumption function, marginal value function, and minimum m.
        '''
        xLvl,mLvl,pLvl = self.getPointsForInterpolation(EndOfPrdvP,aLvl)
        MedShk_temp    = np.tile(np.reshape(self.MedShkVals,(self.MedShkVals.size,1,1)),\
                         (1,mLvl.shape[1],mLvl.shape[2]))
        solution_now   = self.usePointsForInterpolation(xLvl,mLvl,pLvl,MedShk_temp,interpolator)
        return solution_now

    def addvPPfunc(self,solution):
        '''
        Adds the marginal marginal value function to an existing solution, so
        that the next solver can evaluate vPP and thus use cubic interpolation.

        Parameters
        ----------
        solution : ConsumerSolution
            The solution to this single period problem, which must include the
            consumption function.

        Returns
        -------
        solution : ConsumerSolution
            The same solution passed as input, but with the marginal marginal
            value function for this period added as the attribute vPPfunc.
        '''
        vPPfuncNow        = MargMargValueFunc2D(solution.vPfunc.cFunc,self.CRRA)
        solution.vPPfunc  = vPPfuncNow
        return solution

    def solve(self):
        '''
        Solves a one period consumption saving problem with risky income and
        shocks to medical need.

        Parameters
        ----------
        None

        Returns
        -------
        solution : ConsumerSolution
            The solution to the one period problem, including a consumption
            function, medical spending function ( both defined over market re-
            sources, permanent income, and medical shock), a marginal value func-
            tion (defined over market resources and permanent income), and human
            wealth as a function of permanent income.
        '''
        aLvl,trash  = self.prepareToCalcEndOfPrdvP()
        EndOfPrdvP = self.calcEndOfPrdvP()
        if self.vFuncBool:
            self.makeEndOfPrdvFunc(EndOfPrdvP)
        if self.CubicBool:
            interpolator = self.makeCubicxFunc
        else:
            interpolator = self.makeLinearxFunc
        solution   = self.makeBasicSolution(EndOfPrdvP,aLvl,interpolator)
        solution   = self.addMPCandHumanWealth(solution)
        if self.CubicBool:
            solution = self.addvPPfunc(solution)
        return solution


def solveConsMedShock(solution_next,IncomeDstn,MedShkDstn,LivPrb,DiscFac,CRRA,CRRAmed,Rfree,MedPrice,
                 pLvlNextFunc,BoroCnstArt,aXtraGrid,pLvlGrid,vFuncBool,CubicBool):
    '''
    Solve the one period problem for a consumer with shocks to permanent and
    transitory income as well as medical need shocks (as multiplicative shifters
    for utility from a second medical care good).

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncomeDstn : [np.array]
        A list containing three arrays of floats, representing a discrete
        approximation to the income process between the period being solved
        and the one immediately following (in solution_next). Order: event
        probabilities, permanent shocks, transitory shocks.
    MedShkDstn : [np.array]
        Discrete distribution of the multiplicative utility shifter for med-
        ical care. Order: probabilities, preference shocks.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.
    DiscFac : float
        Intertemporal discount factor for future utility.
    CRRA : float
        Coefficient of relative risk aversion for composite consumption.
    CRRAmed : float
        Coefficient of relative risk aversion for medical care.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    MedPrice : float
        Price of unit of medical care relative to unit of consumption.
    pLvlNextFunc : float
        Expected permanent income next period as a function of current pLvl.
    PrstIncCorr : float
        Correlation of permanent income from period to period.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.
    aXtraGrid: np.array
        Array of "extra" end-of-period (normalized) asset values-- assets
        above the absolute minimum acceptable level.
    pLvlGrid: np.array
        Array of permanent income levels at which to solve the problem.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear inter-
        polation.

    Returns
    -------
    solution : ConsumerSolution
        Solution to this period's problem, including a consumption function,
        medical spending function, and marginal value function.  The former two
        are defined over (mLvl,pLvl,MedShk), while the latter is defined only
        on (mLvl,pLvl), with MedShk integrated out.
    '''
    solver = ConsMedShockSolver(solution_next,IncomeDstn,MedShkDstn,LivPrb,DiscFac,CRRA,CRRAmed,Rfree,
                MedPrice,pLvlNextFunc,BoroCnstArt,aXtraGrid,pLvlGrid,vFuncBool,CubicBool)
    solver.prepareToSolve()       # Do some preparatory work
    solution_now = solver.solve() # Solve the period
    return solution_now


###############################################################################

def main():
    import HARK.ConsumptionSaving.ConsumerParameters as Params
    from HARK.utilities import CRRAutility_inv
    from time import perf_counter
    import matplotlib.pyplot as plt
    mystr = lambda number : "{:.4f}".format(number)

    do_simulation = True

    # Make and solve an example medical shocks consumer type
    MedicalExample = MedShockConsumerType(**Params.init_medical_shocks)
    t_start = perf_counter()
    MedicalExample.solve()
    t_end = perf_counter()
    print('Solving a medical shocks consumer took ' + mystr(t_end-t_start) + ' seconds.')

    # Plot the consumption function
    M = np.linspace(0,30,300)
    pLvl = 1.0
    P = pLvl*np.ones_like(M)
    for j in range(MedicalExample.MedShkDstn[0][0].size):
        MedShk = MedicalExample.MedShkDstn[0][1][j]*np.ones_like(M)
        M_temp = M + MedicalExample.solution[0].mLvlMin(pLvl)
        C = MedicalExample.solution[0].cFunc(M_temp,P,MedShk)
        plt.plot(M_temp,C)
    print('Consumption function by medical need shock (constant permanent income)')
    plt.show()

    # Plot the medical care function
    for j in range(MedicalExample.MedShkDstn[0][0].size):
        MedShk = MedicalExample.MedShkDstn[0][1][j]*np.ones_like(M)
        Med = MedicalExample.solution[0].MedFunc(M_temp,P,MedShk)
        plt.plot(M_temp,Med)
    print('Medical care function by medical need shock (constant permanent income)')
    plt.ylim([0,20])
    plt.show()

    # Plot the savings function
    for j in range(MedicalExample.MedShkDstn[0][0].size):
        MedShk = MedicalExample.MedShkDstn[0][1][j]*np.ones_like(M)
        Sav = M_temp - MedicalExample.solution[0].cFunc(M_temp,P,MedShk) - MedicalExample.MedPrice[0]*\
              MedicalExample.solution[0].MedFunc(M_temp,P,MedShk)
        plt.plot(M_temp,Sav)
    print('End of period savings by medical need shock (constant permanent income)')
    plt.show()

    # Plot the marginal value function
    M = np.linspace(0.0,30,300)
    for p in range(MedicalExample.pLvlGrid[0].size):
        pLvl = MedicalExample.pLvlGrid[0][p]
        M_temp = pLvl*M + MedicalExample.solution[0].mLvlMin(pLvl)
        P = pLvl*np.ones_like(M)
        vP = MedicalExample.solution[0].vPfunc(M_temp,P)**(-1.0/MedicalExample.CRRA)
        plt.plot(M_temp,vP)
    print('Marginal value function (pseudo inverse)')
    plt.show()

    if MedicalExample.vFuncBool:
        # Plot the value function
        M = np.linspace(0.0,1,300)
        for p in range(MedicalExample.pLvlGrid[0].size):
            pLvl = MedicalExample.pLvlGrid[0][p]
            M_temp = pLvl*M + MedicalExample.solution[0].mLvlMin(pLvl)
            P = pLvl*np.ones_like(M)
            v = CRRAutility_inv(MedicalExample.solution[0].vFunc(M_temp,P),gam=MedicalExample.CRRA)
            plt.plot(M_temp,v)
        print('Value function (pseudo inverse)')
        plt.show()

    if do_simulation:
        t_start = perf_counter()
        MedicalExample.T_sim = 100
        MedicalExample.track_vars = ['mLvlNow','cLvlNow','MedNow']
        MedicalExample.makeShockHistory()
        MedicalExample.initializeSim()
        MedicalExample.simulate()
        t_end = perf_counter()
        print('Simulating ' + str(MedicalExample.AgentCount) + ' agents for ' + str(MedicalExample.T_sim) + ' periods took ' + mystr(t_end-t_start) + ' seconds.')

if __name__ == '__main__':
    main()

