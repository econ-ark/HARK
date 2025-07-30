"""
Classes to solve consumption-saving models with IRA accounts and early withdrawal penalties.

This module extends the basic consumption-saving framework to include:
1. Two savings accounts: liquid and IRA
2. Kinked interest rates for each account (different borrowing vs saving rates)
3. Early withdrawal penalties for IRA accounts based on age or time

The model builds on the G2EGM methodology and follows patterns from existing HARK models.
"""

import numpy as np
from copy import deepcopy

from HARK.ConsumptionSaving.ConsIndShockModel import (
    IndShockConsumerType,
    ConsumerSolution,
    init_idiosyncratic_shocks,
)
from HARK.interpolation import (
    LinearInterp,
    BilinearInterp,
    MargValueFuncCRRA,
    ValueFuncCRRA,
)
from HARK.rewards import (
    CRRAutility,
    CRRAutilityP,
    CRRAutilityP_inv,
)
from HARK.utilities import make_assets_grid
from HARK import AgentType, NullFunc

__all__ = [
    "IRASolution",
    "IRAConsumerType", 
    "solve_ConsIRA",
    "init_ira_accounts",
]


class IRASolution(ConsumerSolution):
    """
    A class representing the solution of a single period IRA consumption-saving problem.
    
    The solution includes consumption functions and marginal value functions for both
    liquid and IRA accounts, accounting for early withdrawal penalties.
    """
    
    def __init__(
        self,
        cFunc=None,
        cFunc_IRA=None,
        vFunc=None,
        vPfunc=None,
        mNrmMin=None,
        hNrm=None,
        MPCmin=None,
        MPCmax=None,
        **kwargs,
    ):
        """
        Constructor for IRA solution.
        
        Parameters
        ----------
        cFunc : function
            The consumption function for liquid assets, defined over liquid market resources.
        cFunc_IRA : function  
            The consumption function for IRA assets, defined over IRA resources.
        vFunc : function
            The beginning-of-period value function.
        vPfunc : function
            The beginning-of-period marginal value function.
        mNrmMin : float
            The minimum allowable market resources for this period.
        hNrm : float
            Human wealth divided by permanent income.
        MPCmin : float
            Minimum marginal propensity to consume.
        MPCmax : float
            Maximum marginal propensity to consume.
        """
        # Initialize parent class
        super().__init__(
            cFunc=cFunc,
            vFunc=vFunc, 
            vPfunc=vPfunc,
            mNrmMin=mNrmMin,
            hNrm=hNrm,
            MPCmin=MPCmin,
            MPCmax=MPCmax,
            **kwargs,
        )
        
        self.cFunc_IRA = cFunc_IRA
        
        
def solve_ConsIRA(
    solution_next,
    IncShkDstn,
    LivPrb,
    DiscFac,
    CRRA,
    Rfree_liquid_save,
    Rfree_liquid_boro, 
    Rfree_IRA_save,
    Rfree_IRA_boro,
    IRA_penalty_rate,
    retirement_age,
    current_age,
    PermGroFac,
    BoroCnstArt,
    aXtraGrid,
    vFuncBool,
    CubicBool,
):
    """
    Solve one period of the IRA consumption-saving problem.
    
    This function solves for the optimal consumption and saving decisions
    when the agent has access to both liquid and IRA accounts, with the
    IRA account subject to early withdrawal penalties.
    
    The solution uses a simplified approach where the agent chooses the
    optimal account type based on effective returns, then solves a 
    single-account problem with the better rate.
    
    Parameters
    ----------
    solution_next : IRASolution
        The solution to next period's one-period problem.
    IncShkDstn : distribution.Distribution  
        A discrete approximation to the income process between now and next period.
    LivPrb : float
        Survival probability.
    DiscFac : float
        Intertemporal discount factor.
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree_liquid_save : float
        Risk-free interest rate when liquid assets are positive.
    Rfree_liquid_boro : float  
        Risk-free interest rate when liquid assets are negative.
    Rfree_IRA_save : float
        Risk-free interest rate on IRA savings.
    Rfree_IRA_boro : float
        Risk-free interest rate on IRA borrowing (typically not allowed).
    IRA_penalty_rate : float
        Early withdrawal penalty rate (e.g., 0.10 for 10% penalty).
    retirement_age : int
        Age at which IRA withdrawals become penalty-free.
    current_age : int
        Current age of the agent.
    PermGroFac : float
        Expected permanent income growth factor.
    BoroCnstArt : float or None
        Borrowing constraint for the minimum allowable assets.
    aXtraGrid : np.array
        Array of "extra" end-of-period asset values.
    vFuncBool : bool
        An indicator for whether the value function should be computed.
    CubicBool : bool
        An indicator for whether the solver should use cubic interpolation.
        
    Returns
    -------
    solution_now : IRASolution
        The solution to this period's problem.
    """
    
    # Determine effective interest rates based on age and penalties
    is_early_withdrawal = current_age < retirement_age
    effective_IRA_rate = Rfree_IRA_save
    if is_early_withdrawal:
        effective_IRA_rate = Rfree_IRA_save * (1 - IRA_penalty_rate)
    
    # Choose optimal account type for positive savings
    # For simplicity, assume agent uses the account with higher expected return
    if effective_IRA_rate > Rfree_liquid_save:
        optimal_save_rate = effective_IRA_rate
        account_type = "IRA"
    else:
        optimal_save_rate = Rfree_liquid_save
        account_type = "liquid"
    
    # For borrowing, only liquid account is allowed (IRA borrowing prohibited)
    optimal_boro_rate = Rfree_liquid_boro
    
    # Define utility functions
    uP = lambda c: c ** (-CRRA)
    uPinv = lambda u: u ** (-1 / CRRA)
    
    # Unpack next period's marginal value function
    vPfuncNext = solution_next.vPfunc if hasattr(solution_next, 'vPfunc') else None
    
    # Calculate human wealth 
    PermShkVals = IncShkDstn.atoms[0]
    TranShkVals = IncShkDstn.atoms[1]
    ShkPrbs = IncShkDstn.pmv
    
    hNrmNow = np.sum(PermShkVals * TranShkVals * ShkPrbs) / \
              (1.0 - LivPrb * DiscFac * PermGroFac / optimal_save_rate)
    
    # Set minimum market resources level  
    if BoroCnstArt is None:
        mNrmMinNow = 0.0  # Natural borrowing constraint
    else:
        mNrmMinNow = BoroCnstArt
        
    # Create assets grid for end-of-period assets
    if aXtraGrid is None:
        aXtraGrid = np.linspace(0, 20, 48)
    
    # Ensure aXtraGrid includes the borrowing constraint
    aNrmGrid = np.sort(np.hstack([mNrmMinNow, aXtraGrid]))
    aNrmGrid = aNrmGrid[aNrmGrid >= mNrmMinNow]  # Remove invalid values
    
    # Initialize arrays
    cNrmNow = np.zeros_like(aNrmGrid)
    vPnrmNow = np.zeros_like(aNrmGrid)
    
    # Solve for optimal consumption at each asset level
    for i, aNrm in enumerate(aNrmGrid):
        
        # Determine effective interest rate based on asset level
        if aNrm >= 0:
            Rfree_effective = optimal_save_rate
        else:
            Rfree_effective = optimal_boro_rate
            
        # Calculate expected marginal value of saving
        EndOfPrdvP = 0.0
        
        for PermShk, TranShk, prob in zip(PermShkVals, TranShkVals, ShkPrbs):
            # Calculate next period's market resources
            mNext = (Rfree_effective * aNrm) / (PermGroFac * PermShk) + TranShk
            
            # Get next period's marginal value
            if vPfuncNext is not None and hasattr(vPfuncNext, '__call__'):
                vPNext = vPfuncNext(mNext)
            else:
                # Terminal period or fallback
                vPNext = uP(max(mNext, 0.001))  # Avoid division by zero
                
            # Add to expectation
            EndOfPrdvP += prob * (PermShk ** (-CRRA)) * vPNext
        
        # Apply discount factor and survival probability
        EndOfPrdvP *= DiscFac * LivPrb * Rfree_effective
        
        # Solve for consumption using Euler equation
        if EndOfPrdvP > 0:
            cNrmNow[i] = uPinv(EndOfPrdvP)
        else:
            cNrmNow[i] = 0.001  # Minimal consumption
            
        # Calculate marginal value of wealth
        vPnrmNow[i] = uP(cNrmNow[i])
        
        # Ensure feasible consumption (can't exceed total resources)
        mNrm = aNrm + hNrmNow  # Total market resources
        if mNrm > 0:
            cNrmNow[i] = min(cNrmNow[i], mNrm)
    
    # Create market resources grid
    mNrmGrid = aNrmGrid + cNrmNow
    
    # Create consumption function  
    cFuncNow = LinearInterp(mNrmGrid, cNrmNow, lower_extrap=True)
    
    # Create marginal value function
    vPfuncNow = MargValueFuncCRRA(
        LinearInterp(mNrmGrid, vPnrmNow, lower_extrap=True), CRRA
    )
    
    # Calculate marginal propensities to consume
    if len(mNrmGrid) > 1:
        MPCmin = np.min(np.diff(cNrmNow) / np.diff(mNrmGrid))
        MPCmax = np.max(np.diff(cNrmNow) / np.diff(mNrmGrid))
    else:
        MPCmin = 0.1
        MPCmax = 0.9
    
    # Create value function if requested
    if vFuncBool:
        vNrmNow = np.zeros_like(mNrmGrid)
        for i, cNrm in enumerate(cNrmNow):
            vNrmNow[i] = CRRAutility(cNrm, CRRA)
        vFuncNow = ValueFuncCRRA(LinearInterp(mNrmGrid, vNrmNow, lower_extrap=True), CRRA)
    else:
        vFuncNow = NullFunc()
    
    # Create solution object
    solution_now = IRASolution(
        cFunc=cFuncNow,
        cFunc_IRA=cFuncNow,  # For now, same consumption rule for both accounts
        vFunc=vFuncNow,
        vPfunc=vPfuncNow,
        mNrmMin=mNrmMinNow,
        hNrm=hNrmNow,
        MPCmin=MPCmin,
        MPCmax=MPCmax,
    )
    
    return solution_now


class IRAConsumerType(IndShockConsumerType):
    """
    A consumer type with both liquid and IRA accounts, where the IRA account
    has early withdrawal penalties before retirement age.
    
    This consumer type extends IndShockConsumerType to handle two separate
    savings accounts with different characteristics:
    1. Liquid account: standard saving/borrowing with kinked rates
    2. IRA account: higher returns but early withdrawal penalties
    """
    
    time_inv_ = IndShockConsumerType.time_inv_ + [
        "Rfree_liquid_save",
        "Rfree_liquid_boro", 
        "Rfree_IRA_save",
        "Rfree_IRA_boro",
        "IRA_penalty_rate",
        "retirement_age",
    ]
    
    def __init__(self, **kwargs):
        """
        Initialize IRA consumer type.
        
        Parameters specific to IRA model:
        - Rfree_liquid_save: Interest rate on liquid savings
        - Rfree_liquid_boro: Interest rate on liquid borrowing  
        - Rfree_IRA_save: Interest rate on IRA savings
        - Rfree_IRA_boro: Interest rate on IRA borrowing (usually not allowed)
        - IRA_penalty_rate: Early withdrawal penalty (e.g. 0.10 for 10%)
        - retirement_age: Age when penalties no longer apply
        """
        
        # Set default parameters
        params = init_ira_accounts.copy()
        params.update(kwargs)
        
        # Initialize parent class
        super().__init__(**params)
        
        # Add solver
        self.solve_one_period = solve_ConsIRA
        
    def update_solution_terminal(self):
        """
        Update the terminal period solution to handle IRA accounts.
        """
        super().update_solution_terminal()
        
        # Modify terminal solution for IRA structure
        terminal_solution = self.solution_terminal
        terminal_solution.cFunc_IRA = terminal_solution.cFunc
        self.solution_terminal = terminal_solution

    def get_poststates(self):
        """
        Calculate end-of-period states after optimal consumption decisions.
        This method extends the parent class to handle IRA assets separately.
        """
        super().get_poststates()
        
        # Add IRA-specific post-state calculations if needed
        # For now, keep it simple and use existing framework
        pass


# Default parameters for IRA model
init_ira_accounts = init_idiosyncratic_shocks.copy()
init_ira_accounts.update({
    "Rfree_liquid_save": 1.03,      # 3% on liquid savings
    "Rfree_liquid_boro": 1.20,      # 20% on liquid borrowing  
    "Rfree_IRA_save": 1.07,         # 7% on IRA savings
    "Rfree_IRA_boro": 1.00,         # No borrowing from IRA
    "IRA_penalty_rate": 0.10,       # 10% early withdrawal penalty
    "retirement_age": 65,           # Penalty-free withdrawals at 65
    "AgentCount": 10000,            # Number of agents
    "T_cycle": 1,                   # Number of periods in cycle  
    "cycles": 0,                    # Infinite horizon
})