# NOTE Need to decide on Rshare vs RiskyShare
import math
import scipy.optimize as sciopt
from HARK import Solution, NullFunc
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType,solveConsIndShock, ConsIndShockSolver, MargValueFunc, ConsumerSolution
from HARK.utilities import approxLognormal, combineIndepDstns
from HARK.interpolation import LinearInterp, LowerEnvelope

from copy import deepcopy
import numpy as np

class PortfolioConsumerSolution(Solution):
    distance_criteria = ['vPfunc']

    def __init__(self, cFunc=None, vFunc=None,
                       vPfunc=None, RiskyShareFunc=None, vPPfunc=None,
                       mNrmMin=None, hNrm=None, MPCmin=None, MPCmax=None):
        # Change any missing function inputs to NullFunc
        if cFunc is None:
            cFunc = NullFunc()
        if vFunc is None:
            vFunc = NullFunc()
        if RiskyShareFunc is None:
            RiskyShareFunc = NullFunc()
        if vPfunc is None:
            vPfunc = NullFunc()
        if vPPfunc is None:
            vPPfunc = NullFunc()
        self.cFunc        = cFunc
        self.vFunc        = vFunc
        self.vPfunc       = vPfunc
        self.RiskyShareFunc = RiskyShareFunc
        # self.vPPfunc      = vPPfunc
        # self.mNrmMin      = mNrmMin
        # self.hNrm         = hNrm
        # self.MPCmin       = MPCmin
        # self.MPCmax       = MPCmax

class PortfolioChoiceConsumerType(IndShockConsumerType):

    time_inv_ = IndShockConsumerType.time_inv_ + ['RiskyDstn', 'RiskyShareCount']
    cFunc_terminal_      = LinearInterp([0.0, 1.0],[0.0,1.0]) # c=m in terminal period
    vFunc_terminal_      = LinearInterp([0.0, 1.0],[0.0,0.0]) # This is overwritten
    RiskyShareFunc_terminal_ = LinearInterp([0.0, 1.0],[0.0,0.0]) # c=m in terminal period
    solution_terminal_   = PortfolioConsumerSolution(cFunc = cFunc_terminal_, RiskyShareFunc = RiskyShareFunc_terminal_,
                                            vFunc = vFunc_terminal_, mNrmMin=0.0, hNrm=None,
                                            MPCmin=None, MPCmax=None)

    def __init__(self,cycles=1,time_flow=True,verbose=False,quiet=False,**kwds):

        IndShockConsumerType.__init__(self,cycles=cycles, time_flow=time_flow,
                                      verbose=verbose, quiet=quiet, **kwds)

        if self.BoroCnstArt is not 0.0:
            print("Setting BoroCnstArt to 0.0 as this is required by PortfolioChoiceConsumerType.")
            self.BoroCnstArt = 0.0

        # Chose specialized solver for Portfolio choice model
        self.solveOnePeriod = solveConsPortfolioChoice

        # Update this *once* as it's time invariant
        self.updateRiskyDstn()

    def updateRiskyDstn(self):
        # Note, we expect the thing in self.approxRiskyDstn to be a funtion of
        # the number of nodes only!
        self.RiskyDstn = self.approxRiskyDstn(self.RiskyCount)


class ConsIndShockPortfolioSolver(ConsIndShockSolver):
    def __init__(self, solution_next, IncomeDstn, LivPrb, DiscFac, CRRA, Rfree,
                      PermGroFac, BoroCnstArt, aXtraGrid, vFuncBool, CubicBool, RiskyDstn, RiskyShareCount):


        ConsIndShockSolver.__init__(self, solution_next, IncomeDstn, LivPrb, DiscFac, CRRA, Rfree,
                          PermGroFac, BoroCnstArt, aXtraGrid, vFuncBool, CubicBool)

        # Store the Risky asset shock distribution
        self.RiskyDstn = RiskyDstn
        # Store the number of grid points used approximate the FOC in the port-
        # folio sub-problem.
        self.RiskyShareCount = RiskyShareCount
        self.vPfuncNext = solution_next.vPfunc
        self.updateShockDstn()
        self.makeRshareGrid()

    def updateShockDstn(self):
        self.ShockDstn = combineIndepDstns(self.IncomeDstn, self.RiskyDstn)

    def makeRshareGrid(self):
        self.RshareGrid = np.linspace(0, 1, self.RiskyShareCount)
        return self.RshareGrid

    def prepareToCalcRiskyShare(self):
        # Hard restriction on aNrm. We'd need to define more elaborate model
        # specifics if a could become negative (or a positive return shock
        # would make you worse off!)
        aNrmPort = self.aXtraGrid[self.aXtraGrid >= 0]

        self.aNrmPort = aNrmPort
        RshareGrid = self.makeRshareGrid()
        self.RshareNow = np.array([])
        vHatP = np.zeros((len(aNrmPort), len(RshareGrid)))

        # Evaluate the non-constant part of the first order conditions wrt the
        # portfolio share. This requires the implied resources tomorrow given
        # todays shocks to be evaluated.

        i_a = 0
        for a in aNrmPort:
            # for all possible a's today
            i_s = 0
            for s in RshareGrid:
                Rtilde = self.RiskyShkValsNext - self.Rfree
                Reff = self.Rfree + Rtilde*s
                mNext = a*Reff/(self.PermGroFac*self.PermShkValsNext) + self.TranShkValsNext
                vHatP_a_s = Rtilde*self.PermShkValsNext**(-self.CRRA)*self.vPfuncNext(mNext)
                vHatP[i_a, i_s] = np.dot(vHatP_a_s, self.ShkPrbsNext)
                i_s += 1
            i_a += 1

        return vHatP

    def calcRiskyShare(self):
        aGrid = np.array([0.0,])
        Rshare = np.array([1.0,])

        i_a = 0
        for a in self.aNrmPort:
            aGrid = np.append(aGrid, a)
            if self.vHatP[i_a, -1] >= 0.0:
                Rshare = np.append(Rshare, 1.0)
            elif self.vHatP[i_a, 0] < 0.0:
                Rshare = np.append(Rshare, 0.0)
            else:
                residual = LinearInterp(self.RshareGrid, self.vHatP[i_a, :])
                zero  = sciopt.fsolve(residual, 0.5)
                Rshare = np.append(Rshare, zero)
            i_a += 1
        RiskyShareFunc = LinearInterp(aGrid, Rshare)
        return RiskyShareFunc

    def prepareToCalcEndOfPrdvP(self):
        '''
        Prepare to calculate end-of-period marginal value by creating an array
        of market resources that the agent could have next period, considering
        the grid of end-of-period assets and the distribution of shocks he might
        experience next period. This method adds extra steps because it first
        solves the portfolio problem given the end-of-period assets to be able
        to get next period resources.

        Parameters
        ----------
        none

        Returns
        -------
        aNrmNow : np.array
            A 1D array of end-of-period assets; also stored as attribute of self.
        '''

        # We define aNrmNow all the way from BoroCnstNat up to max(self.aXtraGrid)
        # even if BoroCnstNat < BoroCnstArt, so we can construct the consumption
        # function as the lower envelope of the (by the artificial borrowing con-
        # straint) uconstrained consumption function, and the artificially con-
        # strained consumption function.
        aNrmNow     = np.asarray(self.aXtraGrid)
        ShkCount    = self.TranShkValsNext.size
        aNrm_temp   = np.tile(aNrmNow,(ShkCount,1))

        # Tile arrays of the income shocks and put them into useful shapes
        aNrmCount         = aNrmNow.shape[0]
        PermShkVals_temp  = (np.tile(self.PermShkValsNext,(aNrmCount,1))).transpose()
        TranShkVals_temp  = (np.tile(self.TranShkValsNext,(aNrmCount,1))).transpose()
        RiskyShkVals_temp  = (np.tile(self.RiskyShkValsNext,(aNrmCount,1))).transpose()
        ShkPrbs_temp      = (np.tile(self.ShkPrbsNext,(aNrmCount,1))).transpose()


        # sAtA
        sAt_aNrm = self.RiskyShareFunc(aNrmNow)
        # Get cash on hand next period

        Rtilde = RiskyShkVals_temp - self.Rfree
        self.Reff = (self.Rfree + Rtilde*sAt_aNrm)
        mNrmPreTran = self.Reff/(self.PermGroFac*PermShkVals_temp)*aNrm_temp
        mNrmNext = mNrmPreTran + TranShkVals_temp

        # Store and report the results
        self.PermShkVals_temp  = PermShkVals_temp
        self.ShkPrbs_temp      = ShkPrbs_temp
        self.mNrmNext          = mNrmNext
        self.aNrmNow           = aNrmNow
        return aNrmNow

    def calcEndOfPrdvP(self):
        '''
        Calculate end-of-period marginal value of assets at each point in aNrmNow.
        Does so by taking a weighted sum of next period marginal values across
        income shocks (in a preconstructed grid self.mNrmNext).

        Parameters
        ----------
        none

        Returns
        -------
        EndOfPrdvP : np.array
            A 1D array of end-of-period marginal value of assets
        '''

        EndOfPrdvP  = np.sum(self.DiscFacEff*self.Reff*self.PermGroFac**(-self.CRRA)*
                      self.PermShkVals_temp**(-self.CRRA)*
                      self.vPfuncNext(self.mNrmNext)*self.ShkPrbs_temp,axis=0)
        return EndOfPrdvP



    def setAndUpdateValues(self,solution_next,IncomeDstn,LivPrb,DiscFac):
        '''
        Unpacks some of the inputs (and calculates simple objects based on them),
        storing the results in self for use by other methods.  These include:
        income shocks and probabilities, next period's marginal value function
        (etc), the probability of getting the worst income shock next period,
        the patience factor, human wealth, and the bounding MPCs.

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
        self.DiscFacEff       = DiscFac*LivPrb # "effective" discount factor
        self.ShkPrbsNext  = self.ShockDstn[0] # but ConsumtionSolver doesn't store the risky shocks
        self.PermShkValsNext  = self.ShockDstn[1] # but ConsumtionSolver doesn't store the risky shocks
        self.TranShkValsNext  = self.ShockDstn[2] # but ConsumtionSolver doesn't store the risky shocks
        self.RiskyShkValsNext  = self.ShockDstn[3] # but ConsumtionSolver doesn't store the risky shocks
        self.PermShkMinNext   = np.min(self.PermShkValsNext)
        self.TranShkMinNext   = np.min(self.TranShkValsNext)
        self.vPfuncNext       = solution_next.vPfunc
        self.WorstIncPrb      = np.sum(self.ShkPrbsNext[
                                (self.PermShkValsNext*self.TranShkValsNext)==
                                (self.PermShkMinNext*self.TranShkMinNext)])

        if self.CubicBool:
            self.vPPfuncNext  = solution_next.vPPfunc

        if self.vFuncBool:
            self.vFuncNext    = solution_next.vFunc

        # Update the bounding MPCs and PDV of human wealth:
        # self.PatFac       = ((self.Rfree*self.DiscFacEff)**(1.0/self.CRRA))/self.Rfree
        # self.MPCminNow    = 1.0/(1.0 + self.PatFac/solution_next.MPCmin)
        # self.ExIncNext    = np.dot(self.ShkPrbsNext,self.TranShkValsNext*self.PermShkValsNext)
        # self.hNrmNow      = self.PermGroFac/self.Rfree*(self.ExIncNext + solution_next.hNrm)
        # self.MPCmaxNow    = 1.0/(1.0 + (self.WorstIncPrb**(1.0/self.CRRA))*
        #                                 self.PatFac/solution_next.MPCmax)




    def defBoroCnst(self,BoroCnstArt):
        '''
        Defines the constrained portion of the consumption function as cFuncNowCnst,
        an attribute of self.  Uses the artificial and natural borrowing constraints.

        Parameters
        ----------
        BoroCnstArt : float or None
            Borrowing constraint for the minimum allowable assets to end the
            period with.  If it is less than the natural borrowing constraint,
            then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
            rowing constraint.

        Returns
        -------
        none
        '''
        # Calculate the minimum allowable value of money resources in this period
        self.BoroCnstNat = 0.0 #(self.solution_next.mNrmMin - self.TranShkMinNext)*\
                           #(self.PermGroFac*self.PermShkMinNext)/self.Rfree

        if BoroCnstArt is None:
            self.mNrmMinNow = self.BoroCnstNat
        else:
            self.mNrmMinNow = np.max([self.BoroCnstNat,BoroCnstArt])

        # Can we just put in *some value* for MPCmaxNow here?
        # if self.BoroCnstNat < self.mNrmMinNow:
        #     self.MPCmaxEff = 1.0 # If actually constrained, MPC near limit is 1
        # else:
        #     self.MPCmaxEff = self.MPCmaxNow

        # Define the borrowing constraint (limiting consumption function)
        self.cFuncNowCnst = LinearInterp(np.array([0.0, 1.0]),
                                         np.array([0.0, 1.0]))


    def solve(self):
        '''
        Solves a one period consumption saving problem with risky income and
        a portfolio choice over a riskless and a risky asset.

        Parameters
        ----------
        None

        Returns
        -------
        solution : ConsumerSolution
            The solution to the one period problem.
        '''

        # First, solve the first sub-problem: the portfolio choice
        self.vHatP = self.prepareToCalcRiskyShare()
        self.RiskyShareFunc = self.calcRiskyShare()

        # Then solve the consumption choice given optimal portfolio choice
        aNrm       = self.prepareToCalcEndOfPrdvP()
        EndOfPrdvP = self.calcEndOfPrdvP()

        # Todo!
        self.cFuncLimitIntercept = None
        self.cFuncLimitSlope = None

        cs_solution   = self.makeBasicSolution(EndOfPrdvP,aNrm,self.makeLinearcFunc)
        cs_solution.vPfunc(0.3)
        # solution   = self.addMPCandHumanWealth(solution)
        solution = PortfolioConsumerSolution(cFunc=cs_solution.cFunc,
                                             vPfunc=cs_solution.vPfunc,
                                             RiskyShareFunc=self.RiskyShareFunc)
        return solution

def solveConsPortfolioChoice(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,PermGroFac,
                                BoroCnstArt,aXtraGrid,vFuncBool,CubicBool, RiskyDstn, RiskyShareCount):

    solver = ConsIndShockPortfolioSolver(solution_next, IncomeDstn, LivPrb, DiscFac, CRRA, Rfree,
    PermGroFac, BoroCnstArt, aXtraGrid, vFuncBool, CubicBool, RiskyDstn, RiskyShareCount)

    solver.prepareToSolve()       # Do some preparatory work

    portsolution = solver.solve()

    return portsolution



def RiskyDstnFactory(RiskyAvg=1.0, RiskyStd=0.0):
    RiskyAvgSqrd = RiskyAvg**2
    RiskyVar = RiskyStd**2

    mu = math.log(RiskyAvg/(math.sqrt(1+RiskyVar/RiskyAvgSqrd)))
    sigma = math.sqrt(math.log(1+RiskyVar/RiskyAvgSqrd))

    return lambda RiskyCount: approxLognormal(RiskyCount, mu=mu, sigma=sigma)





# def ConsIndShockPortfolioSolver(ConsIndShockSolver):
#
# #    def setAndUpdateValues(self):
#         # great but wait
#         #   calc sUnderbar -> mertonsammuelson
#         #   calc MPC kappaUnderbar
#         #   calc human wealth
#
