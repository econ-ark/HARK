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

    time_inv_ = IndShockConsumerType.time_inv_ + ['RiskyDstn', 'RiskyAvg', 'RshareCount']

    def __init__(self,cycles=1,time_flow=True,verbose=False,quiet=False,**kwds):

        IndShockConsumerType.__init__(self,cycles=cycles, time_flow=time_flow,
                                      verbose=verbose, quiet=quiet, **kwds)

        self.solveOnePeriod = solveConsPortfolioChoice
        # Update this *once* as it's time invariant
        self.updateRiskyDstn()

    def updateRiskyDstn(self):
        self.RiskyDstn = self.approxRiskyDstn(self.RiskyCount)
        self.RiskyAvg = np.mean(self.RiskyDstn[1])


class ConsIndShockPortfolioSolver(ConsIndShockSolver):
    def __init__(self, solution_next, IncomeDstn, LivPrb, DiscFac, CRRA, Rfree,
                      PermGroFac, BoroCnstArt, aXtraGrid, vFuncBool, CubicBool, RiskyDstn, RiskyAvg, RshareCount):


        ConsIndShockSolver.__init__(self, solution_next, IncomeDstn, LivPrb, DiscFac, CRRA, Rfree,
                          PermGroFac, BoroCnstArt, aXtraGrid, vFuncBool, CubicBool)
        self.RiskyDstn = RiskyDstn
        self.RiskyAvg = RiskyAvg
        self.RshareCount = RshareCount
        self.vPfuncNext = solution_next.vPfunc
        self.updateShockDstn()
        self.makeRshareGrid()

    def updateShockDstn(self):
        self.ShockDstn = combineIndepDstns(self.IncomeDstn, self.RiskyDstn)

    def makeRshareGrid(self):
        self.RshareGrid = np.linspace(0, 1, self.RshareCount)
        return self.RshareGrid

    def prepareToCalcRshare(self):
        # Hard restriction on a
        aXtraGridPos = self.aXtraGrid[self.aXtraGrid >= 0]
        self.aXtraGridPos = aXtraGridPos
        RshareGrid = self.makeRshareGrid()
        self.RshareNow = np.array([])
        vHatP = np.zeros((len(aXtraGridPos), len(RshareGrid)))

        Probs = self.ShockDstn[0] # use the prepareotsolve vars inm self _tmp
        PermVal = self.ShockDstn[1]
        TransVal = self.ShockDstn[2]
        RiskyVal = self.ShockDstn[3]


        # Evaluate the non-constant part of the first order conditions wrt the
        # portfolio share. This requires the implied resources tomorrow given
        # todays shocks to be evaluated.

        i_a = 0
        for a in aXtraGridPos:
            i_s = 0
            for s in RshareGrid:
                Rtilde = RiskyVal - self.Rfree
                Rbold = self.Rfree + Rtilde*s
                mNext = a*Rbold/(self.PermGroFac*PermVal) + TransVal
                vHatP_a_s = Rtilde*PermVal**(-self.CRRA)*self.vPfuncNext(mNext)
                vHatP[i_a, i_s] = np.dot(vHatP_a_s, Probs)
                i_s += 1
            i_a += 1

        return vHatP

    def calcRshare(self):
        aGrid = np.array([0.0,])
        Rshare = np.array([1.0,])

        i_a = 0
        for a in self.aXtraGridPos:
            aGrid = np.append(aGrid, a)
            if self.vHatP[i_a, -1] >= 0.0:
                Rshare = np.append(Rshare, 1.0)
            elif self.vHatP[i_a, 0] < 0.0:
                Rshare = np.append(Rshare, 0.0)
                print("The first is neg")
            else:
                residual = LinearInterp(self.RshareGrid, self.vHatP[i_a, :])
                zero  = sciopt.fsolve(residual, 0.5)
                Rshare = np.append(Rshare, zero)
            i_a += 1
        RshareFunc = LinearInterp(aGrid, Rshare)
        return RshareFunc

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
        sAt_aNrm = self.RshareFunc(aNrmNow)
        # Get cash on hand next period

        Rtilde = RiskyShkVals_temp - self.Rfree
        self.Rbold = (self.Rfree + Rtilde*sAt_aNrm)
        mNrmPreTran = self.Rbold/(self.PermGroFac*PermShkVals_temp)*aNrm_temp
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

        EndOfPrdvP  = np.sum(self.DiscFacEff*self.Rbold*self.PermGroFac**(-self.CRRA)*
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

        # Note: need to be sure to handle BoroCnstArt==None appropriately.
        # In Py2, this would evaluate to 5.0:  np.max([None, 5.0]).
        # However in Py3, this raises a TypeError. Thus here we need to directly
        # address the situation in which BoroCnstArt == None:
        # if BoroCnstArt is None:
        #     self.mNrmMinNow = self.BoroCnstNat
        # else:
        #     self.mNrmMinNow = np.max([self.BoroCnstNat,BoroCnstArt])
        # if self.BoroCnstNat < self.mNrmMinNow:
        #     self.MPCmaxEff = 1.0 # If actually constrained, MPC near limit is 1
        # else:
        #     self.MPCmaxEff = self.MPCmaxNow

        # Define the borrowing constraint (limiting consumption function)
        self.cFuncNowCnst = LinearInterp(np.array([0.0, 1.0]),
                                         np.array([0.0, 1.0]))


    def usePointsForInterpolation(self,cNrm,mNrm,interpolator):
        '''
        Constructs a basic solution for this period, including the consumption
        function and marginal value function.

        Parameters
        ----------
        cNrm : np.array
            (Normalized) consumption points for interpolation.
        mNrm : np.array
            (Normalized) corresponding market resource points for interpolation.
        interpolator : function
            A function that constructs and returns a consumption function.

        Returns
        -------
        solution_now : ConsumerSolution
            The solution to this period's consumption-saving problem, with a
            consumption function, marginal value function, and minimum m.
        '''
        # Construct the unconstrained consumption function
        print(mNrm[0].shape)
        print(cNrm[0].shape)
        cFuncNowUnc = interpolator(mNrm,cNrm)
        # Combine the constrained and unconstrained functions into the true consumption function
        cFuncNow = LowerEnvelope(cFuncNowUnc,self.cFuncNowCnst)

        # Make the marginal value function and the marginal marginal value function
        vPfuncNow = MargValueFunc(cFuncNow,self.CRRA)

        # Pack up the solution and return it
        solution_now = ConsumerSolution(cFunc=cFuncNow, vPfunc=vPfuncNow)
        return solution_now


    def makeLinearcFunc(self,mNrm,cNrm):
        '''
        Makes a linear interpolation to represent the (unconstrained) consumption function.

        Parameters
        ----------
        mNrm : np.array
            Corresponding market resource points for interpolation.
        cNrm : np.array
            Consumption points for interpolation.

        Returns
        -------
        cFuncUnc : LinearInterp
            The unconstrained consumption function for this period.
        '''
        cFuncUnc = LinearInterp(mNrm,cNrm)
        return cFuncUnc

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
        print("solveing")
        self.vHatP = self.prepareToCalcRshare()
        self.RshareFunc = self.calcRshare()
        aNrm       = self.prepareToCalcEndOfPrdvP()
        EndOfPrdvP = self.calcEndOfPrdvP()
        cs_solution   = self.makeBasicSolution(EndOfPrdvP,aNrm,self.makeLinearcFunc)
        cs_solution.vPfunc(0.3)
        # solution   = self.addMPCandHumanWealth(solution)
        solution = PortfolioConsumerSolution(cFunc=cs_solution.cFunc,
                                             vPfunc=cs_solution.vPfunc,
                                             RiskyShareFunc=self.RshareFunc)
        return solution

def solveConsPortfolioChoice(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,PermGroFac,
                                BoroCnstArt,aXtraGrid,vFuncBool,CubicBool, RiskyDstn, RiskyAvg, RshareCount):

#    solution = solveConsIndShock(solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,PermGroFac,
#                                    BoroCnstArt,aXtraGrid,vFuncBool,CubicBool)

    solver = ConsIndShockPortfolioSolver(solution_next, IncomeDstn, LivPrb, DiscFac, CRRA, Rfree,
    PermGroFac, BoroCnstArt, aXtraGrid, vFuncBool, CubicBool, RiskyDstn, RiskyAvg, RshareCount)

    solver.prepareToSolve()       # Do some preparatory work

    portsolution = solver.solve()

    return portsolution



def RiskyDstnFactory(RiskyAvg=1.0, RiskyStd=0.0):
    RiskyAvgSqrd = RiskyAvg**2
    RiskyVar = RiskyStd**2

    mu = math.log(RiskyAvg/(math.sqrt(1+RiskyVar/RiskyAvgSqrd)))
    sigma = math.sqrt(math.log(1+RiskyVar/RiskyAvgSqrd))

    return lambda RiskyCount: approxLognormal(RiskyCount, mu=mu, sigma=sigma)





#
#     def solveTerminal(self):
#         do same as IndShockConsumerTyppe.solveTerminal and add
#         RshareFunc = 0 to ssolution_terminal
#
#
#
#
# def solveConsPortfolioChoice(solution_next, IncomeDstn,
#                              LivPrb,DiscFac, CRRA,
#                              Rfree, PermGroFac,
#                              BoroCnstArt, aXtraGrid,
#                              vFuncBool, CubicBool,
#                              ShockDstn):
#     return solution_next
#
# def ConsIndShockPortfolioSolver(ConsIndShockSolver):
#
# #    def setAndUpdateValues(self):
#         # great but wait
#         #   calc sUnderbar -> mertonsammuelson
#         #   calc MPC kappaUnderbar
#         #   calc human wealth
#
#     def calcdvdsEndOfPrd(self):
#         tensorGRid = aXtraGrid x RshareGrid
#         dvdsEndOfPrd = dvds(tensorGrid)
##
#     def solveShares(self):
#         for i in range(len(self.aNrmNowPos)):
#             result = solve_for_share(i)
#             self.RshareNow[i] = result.zero
#
#         return self
#
#     def calcEndofPrdvP(self):
#         return self
#
#     def calcCrmNowandmNrmNow(self): # this is get points???
#         return
#
#
#     def solve(self):
#         '''
#         Solves a one period consumption saving problem with risky income.
#
#         Parameters
#         ----------
#         None
#
#         Returns
#         -------
#         solution : ConsumerSolution
#             The solution to the one period problem.
#         '''
#         aNrm       = self.prepareToCalcEndOfPrdvP()
#         EndOfPrdvP = self.calcEndOfPrdvP()
#         solution   = self.makeBasicSolution(EndOfPrdvP,aNrm,self.makeLinearcFunc)
#         solution   = self.addMPCandHumanWealth(solution)
#         return solution
#
#     def makeBasicSolution(self,EndOfPrdvP,aNrm,interpolator):
#         '''
#         Given end of period assets and end of period marginal value, construct
#         the basic solution for this period.
#
#         Parameters
#         ----------
#         EndOfPrdvP : np.array
#             Array of end-of-period marginal values.
#         aNrm : np.array
#             Array of end-of-period asset values that yield the marginal values
#             in EndOfPrdvP.
#
#         interpolator : function
#             A function that constructs and returns a consumption function.
#
#         Returns
#         -------
#         solution_now : ConsumerSolution
#             The solution to this period's consumption-saving problem, with a
#             consumption function, marginal value function, and minimum m.
#         '''
#
#         cNrm,mNrm    = self.getPointsForInterpolation(EndOfPrdvP,aNrm)
#         solution_now = self.usePointsForInterpolation(cNrm,mNrm,interpolator)
#         return solution_now
#
#
#    def usePointsForInterpolation(self,cNrm,mNrm,interpolator):
#         '''
#         Constructs a basic solution for this period, including the consumption
#         function and marginal value function.
#         Parameters
#         ----------
#         cNrm : np.array
#             (Normalized) consumption points for interpolation.
#         mNrm : np.array
#             (Normalized) corresponding market resource points for interpolation.
#         interpolator : function
#             A function that constructs and returns a consumption function.
#         Returns
#         -------
#         solution_now : ConsumerSolution
#             The solution to this period's consumption-saving problem, with a
#             consumption function, marginal value function, and minimum m.
#         '''
#         # Construct the unconstrained consumption function
#         cFuncNowUnc = interpolator(mNrm,cNrm)
#
#         # Combine the constrained and unconstrained functions into the true consumption function
#         cFuncNow = LowerEnvelope(cFuncNowUnc,self.cFuncNowCnst)
#
#         # Make the marginal value function and the marginal marginal value function
#         vPfuncNow = MargValueFunc(cFuncNow,self.CRRA)
#
#         # Pack up the solution and return it
#         solution_now = PortfolioConsumerSolution(cFunc=cFuncNow,
#                                                  vPfunc=vPfuncNow,
#                                                  mNrmMin=self.mNrmMinNow,
#                                                  RshareFunc=self.RshareFunc,
#                                                  vhatPfunc=self.vhatPfunc)
#         return solution_now
#



    #
    #
    #
    #
    #
    # def updateRiskyPremium(self):
    #     self.RiskyPremium = self.IncomeDstn[3] - self.Rfree
    #
    # def Rbold(self, StockShare):
    #     self.updateRiskyPremium() # this is not optimal, but let's do it everywhere for now
    #     return self.Rfree + self.RiskyPremium*StockShare
    #
    # def PortfolioObjective(self, StockShare):
    #     # The portfolio objective is the expected value of interering tomorrow
    #     # with the market resources implied by the investment strategy and the
    #     # particular realizations of the stock price evolution.
    #     mNrm = self.mNrmNextAta(StockShare)
    #     VLvlNext = self.DiscFacEff*self.IncomeDstn[1]**(1.0-self.CRRA)*self.PermGroFac**(1.0-self.CRRA)*self.vFuncNext(mNrm)
    #
    #     return -np.sum(VLvlNext*self.IncomeDstn[0],axis=0)
    #
    # def vOptFuncNextFromPortfolioSubproblem(self):
    #     riskyShare = np.array([])
    #     vOpt = np.array([])
    #     vOptPa = np.array([])
    #     for a in self.aNrmNow:
    #         # set aPortfolio for use in PortfolioObjective
    #         self.aPortfolio = a
    #
    #         # Set the ratio between 0 and 1
    #         optRes = minimize_scalar(self.PortfolioObjective, bounds=(0, 1), method='bounded')
    #         riskyShare = np.append(riskyShare, optRes.x)
    #         vOpt = np.append(vOpt, -self.PortfolioObjective(optRes.x))
    #         mNrmOpt = self.mNrmNextAta(optRes.x)
    #
    #         # This is v^a(a, share(a))
    #         vPNext = self.uP(self.solution_next.cFunc(mNrmOpt))
    #         Rbold = self.Rbold(optRes.x)
    #         vOptPa_single = self.DiscFacEff*Rbold*self.PermGroFac**(-self.CRRA)*sum(
    #         self.IncomeDstn[1]**(-self.CRRA)*
    #         vPNext*self.IncomeDstn[0])
    #         vOptPa = np.append(vOptPa, np.sum(vOptPa_single))
    #         # grab best policy and value and append it
    #     self.riskyShareFunc = LinearInterp(self.aNrmNow, riskyShare)
    #
    #     vOptNvrs  = self.uinv(vOpt) # value transformed through inverse utility
    #     vOptNvrs  = np.insert(vOptNvrs,0,0.0)
    #     aNrm_temp = np.insert(self.aNrmNow,0,self.BoroCnstNat)
    #     vOptNvrsFuncNext = LinearInterp(self.aNrmNow, vOptNvrs)
    #     self.vOptFuncNext  = ValueFunc(vOptNvrsFuncNext,self.CRRA)
    #
    #     self.vOptPaFuncNext = LinearInterp(self.aNrmNow, vOptPa)
    #
    #     return self.vOptFuncNext, self.riskyShareFunc
    #
    # def mNrmNextAta(self, StockShare):
    #     # Get cash on hand next period
    #     mNrmNext = self.Rbold(StockShare)/(self.PermGroFac*self.IncomeDstn[1])*self.aPortfolio + self.IncomeDstn[2]
    #     return mNrmNext
    #
    # def makeEndOfPrdvFunc(self,EndOfPrdvP):
    #     '''
    #     Construct the end-of-period value function for this period, storing it
    #     as an attribute of self for use by other methods.
    #     Parameters
    #     ----------
    #     EndOfPrdvP : np.array
    #         Array of end-of-period marginal value of assets corresponding to the
    #         asset values in self.aNrmNow.
    #     Returns
    #     -------
    #     none
    #     '''
    #     VLvlNext            = self.vOptFuncNext(self.aNrmNow)
    #     EndOfPrdv           = VLvlNext
    #
    #     EndOfPrdvNvrs       = self.uinv(EndOfPrdv) # value transformed through inverse utility
    #     EndOfPrdvNvrs       = np.insert(EndOfPrdvNvrs,0,0.0)
    #     aNrm_temp           = np.insert(self.aNrmNow,0,self.BoroCnstNat)
    #     EndOfPrdvNvrsFunc   = LinearInterp(aNrm_temp,EndOfPrdvNvrs)
    #     self.EndOfPrdvFunc  = ValueFunc(EndOfPrdvNvrsFunc,self.CRRA)
    #
    # # This could be an extension to the basic solver, but it seems that there's
    # # going to be a complete overhaul. Keeping it here for now.
    # def calcEndOfPrdvP(self):
    #     '''
    #     Calculate end-of-period marginal value of assets at each point in aNrmNow.
    #     Does so by taking a weighted sum of next period marginal values across
    #     income shocks (in a preconstructed grid self.mNrmNext).
    #     Parameters
    #     ----------
    #     none
    #     Returns
    #     -------
    #     EndOfPrdvP : np.array
    #         A 1D array of end-of-period marginal value of assets
    #     '''
    #
    #     # Solve portfolio problem in a stage of its own.
    #     self.vOptFunc, self.riskyShareFunc = self.vOptFuncNextFromPortfolioSubproblem()
    #
    #     # use them to construct a vPNext
    #     EndOfPrdvP  = self.vOptPaFuncNext(self.aNrmNow)
    #
    #     return EndOfPrdvP
