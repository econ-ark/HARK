import HARK.ConsIndShockModel: IndShockConsumerType

class PortfolioConsumerSolution(ConsumerSolution):
    def __init__(self):
        self.share = 0

class PortfolioChoiceConsumerType(IndShockConsumerType):
    def __init(self):
        self.solveOnePeriod =  solveConsPortfolioChoice
        return self

    def updateRxsDstn(self):
        # Should be lognormal
        self.Rfree
        self.RriskAvg
        self.RriskStd
        self.RxsCount
        self.RxsDstn = False
        return self.RxsDstn

    def updateShockDstn(self):
        self.ShockDstn = combineIncAndRxs(self.IncomeDstn, self.RxsDstn)
        #add ShockDstn to time_vary if not already
        return

    def makeRshareGrid(self):
        given RshareCount return RshareGrid and add to time_inv

    def update(self):
        updateRxsDstn
        updateShockdstn
        makeRshareGrid

    def solveTerminal(self):
        do same as IndShockConsumerTyppe.solveTerminal and add
        RshareFunc = 0 to ssolution_terminal

    def makeEulerErrorFunc(self)
    as in KinkedRConsumertpye

    def calcBoundingValues(self)
    as above



def solveConsPortfolioChoice(solution_next):
    return solution_next

def ConsIndShockPortfolioSolver(ConsIndShockSolver):
    def __init__(self):
        return self

#    def setAndUpdateValues(self):
        # great but wait
        #   calc sUnderbar -> mertonsammuelson
        #   calc MPC kappaUnderbar
        #   calc human wealth

    def calcdvdsEndOfPrd(self):
        tensorGRid = aXtraGrid x RshareGrid
        dvdsEndOfPrd = dvds(tensorGrid)

    def prepRshareNow(self):
        self.RshareNow = np.zeros(len(aXtraGrid))

    def solveShares(self):
        return self

    def calcEndofPrdvP(self):
        return self

    def calcCrmNowandmNrmNow(self): # this is get points???
        return


    def solve(self):
        '''
        Solves a one period consumption saving problem with risky income.

        Parameters
        ----------
        None

        Returns
        -------
        solution : ConsumerSolution
            The solution to the one period problem.
        '''
        aNrm       = self.prepareToCalcEndOfPrdvP()
        EndOfPrdvP = self.calcEndOfPrdvP()
        solution   = self.makeBasicSolution(EndOfPrdvP,aNrm,self.makeLinearcFunc)
        solution   = self.addMPCandHumanWealth(solution)
        return solution

    def makeBasicSolution(self,EndOfPrdvP,aNrm,interpolator):
        '''
        Given end of period assets and end of period marginal value, construct
        the basic solution for this period.

        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal values.
        aNrm : np.array
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
        solveShares()
        cNrm,mNrm    = self.getPointsForInterpolation(EndOfPrdvP,aNrm)
        solution_now = self.usePointsForInterpolation(cNrm,mNrm,interpolator)
        return solution_now


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
        cFuncNowUnc = interpolator(mNrm,cNrm)

        # Combine the constrained and unconstrained functions into the true consumption function
        cFuncNow = LowerEnvelope(cFuncNowUnc,self.cFuncNowCnst)

        # Make the marginal value function and the marginal marginal value function
        vPfuncNow = MargValueFunc(cFuncNow,self.CRRA)

        # Pack up the solution and return it
        solution_now = PortfolioConsumerSolution(cFunc=cFuncNow,
                                                 vPfunc=vPfuncNow,
                                                 mNrmMin=self.mNrmMinNow,
                                                 RshareFunc=self.RshareFunc,
                                                 vhatPfunc=self.vhatPfunc)
return solution_now













    def updateRiskyPremium(self):
        self.RiskyPremium = self.IncomeDstn[3] - self.Rfree

    def Rbold(self, StockShare):
        self.updateRiskyPremium() # this is not optimal, but let's do it everywhere for now
        return self.Rfree + self.RiskyPremium*StockShare

    def PortfolioObjective(self, StockShare):
        # The portfolio objective is the expected value of interering tomorrow
        # with the market resources implied by the investment strategy and the
        # particular realizations of the stock price evolution.
        mNrm = self.mNrmNextAta(StockShare)
        VLvlNext = self.DiscFacEff*self.IncomeDstn[1]**(1.0-self.CRRA)*self.PermGroFac**(1.0-self.CRRA)*self.vFuncNext(mNrm)

        return -np.sum(VLvlNext*self.IncomeDstn[0],axis=0)

    def vOptFuncNextFromPortfolioSubproblem(self):
        riskyShare = np.array([])
        vOpt = np.array([])
        vOptPa = np.array([])
        for a in self.aNrmNow:
            # set aPortfolio for use in PortfolioObjective
            self.aPortfolio = a

            # Set the ratio between 0 and 1
            optRes = minimize_scalar(self.PortfolioObjective, bounds=(0, 1), method='bounded')
            riskyShare = np.append(riskyShare, optRes.x)
            vOpt = np.append(vOpt, -self.PortfolioObjective(optRes.x))
            mNrmOpt = self.mNrmNextAta(optRes.x)

            # This is v^a(a, share(a))
            vPNext = self.uP(self.solution_next.cFunc(mNrmOpt))
            Rbold = self.Rbold(optRes.x)
            vOptPa_single = self.DiscFacEff*Rbold*self.PermGroFac**(-self.CRRA)*sum(
            self.IncomeDstn[1]**(-self.CRRA)*
            vPNext*self.IncomeDstn[0])
            vOptPa = np.append(vOptPa, np.sum(vOptPa_single))
            # grab best policy and value and append it
        self.riskyShareFunc = LinearInterp(self.aNrmNow, riskyShare)

        vOptNvrs  = self.uinv(vOpt) # value transformed through inverse utility
        vOptNvrs  = np.insert(vOptNvrs,0,0.0)
        aNrm_temp = np.insert(self.aNrmNow,0,self.BoroCnstNat)
        vOptNvrsFuncNext = LinearInterp(self.aNrmNow, vOptNvrs)
        self.vOptFuncNext  = ValueFunc(vOptNvrsFuncNext,self.CRRA)

        self.vOptPaFuncNext = LinearInterp(self.aNrmNow, vOptPa)

        return self.vOptFuncNext, self.riskyShareFunc

    def mNrmNextAta(self, StockShare):
        # Get cash on hand next period
        mNrmNext = self.Rbold(StockShare)/(self.PermGroFac*self.IncomeDstn[1])*self.aPortfolio + self.IncomeDstn[2]
        return mNrmNext

    def makeEndOfPrdvFunc(self,EndOfPrdvP):
        '''
        Construct the end-of-period value function for this period, storing it
        as an attribute of self for use by other methods.
        Parameters
        ----------
        EndOfPrdvP : np.array
            Array of end-of-period marginal value of assets corresponding to the
            asset values in self.aNrmNow.
        Returns
        -------
        none
        '''
        VLvlNext            = self.vOptFuncNext(self.aNrmNow)
        EndOfPrdv           = VLvlNext

        EndOfPrdvNvrs       = self.uinv(EndOfPrdv) # value transformed through inverse utility
        EndOfPrdvNvrs       = np.insert(EndOfPrdvNvrs,0,0.0)
        aNrm_temp           = np.insert(self.aNrmNow,0,self.BoroCnstNat)
        EndOfPrdvNvrsFunc   = LinearInterp(aNrm_temp,EndOfPrdvNvrs)
        self.EndOfPrdvFunc  = ValueFunc(EndOfPrdvNvrsFunc,self.CRRA)

    # This could be an extension to the basic solver, but it seems that there's
    # going to be a complete overhaul. Keeping it here for now.
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

        # Solve portfolio problem in a stage of its own.
        self.vOptFunc, self.riskyShareFunc = self.vOptFuncNextFromPortfolioSubproblem()

        # use them to construct a vPNext
        EndOfPrdvP  = self.vOptPaFuncNext(self.aNrmNow)

        return EndOfPrdvP
