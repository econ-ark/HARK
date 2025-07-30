# IRA Consumer Model Documentation

## Overview

The `IRAConsumerType` implements a consumption-saving model with two savings accounts:
1. **Liquid Account**: Standard saving/borrowing with kinked interest rates
2. **IRA Account**: Higher returns but subject to early withdrawal penalties

This addresses the three requirements from issue #136:
- ✅ Model two savings accounts
- ✅ Each savings account is kinked (different borrowing vs saving rates)  
- ✅ Penalty for withdrawing before retirement age

## Key Features

### Two Account Structure
- **Liquid account**: Traditional savings account with lower returns but full liquidity
- **IRA account**: Retirement account with higher returns but early withdrawal penalties

### Kinked Interest Rates
Each account can have different rates for saving vs borrowing:
- `Rfree_liquid_save`: Interest rate when liquid assets are positive
- `Rfree_liquid_boro`: Interest rate when liquid assets are negative (borrowing)
- `Rfree_IRA_save`: Interest rate on IRA savings (typically higher)
- `Rfree_IRA_boro`: Interest rate on IRA borrowing (typically not allowed, set to 1.0)

### Early Withdrawal Penalties
- `IRA_penalty_rate`: Penalty rate for early withdrawal (e.g., 0.10 for 10%)
- `retirement_age`: Age at which penalties no longer apply (e.g., 65)
- Before retirement age: effective IRA rate = `Rfree_IRA_save * (1 - IRA_penalty_rate)`
- After retirement age: effective IRA rate = `Rfree_IRA_save`

## Usage Example

```python
from HARK.ConsumptionSaving.ConsIRAModel import IRAConsumerType, init_ira_accounts

# Create IRA consumer with custom parameters
ira_params = init_ira_accounts.copy()
ira_params.update({
    'Rfree_liquid_save': 1.03,    # 3% liquid saving rate
    'Rfree_liquid_boro': 1.20,    # 20% liquid borrowing rate
    'Rfree_IRA_save': 1.07,       # 7% IRA saving rate
    'IRA_penalty_rate': 0.10,     # 10% early withdrawal penalty
    'retirement_age': 65,         # Penalty-free age
    'AgentCount': 10000,
    'T_sim': 200,
})

# Create and solve
agent = IRAConsumerType(**ira_params)
agent.solve()

# Run simulation
agent.initialize_sim()
agent.simulate()

# Analyze results
liquid_assets = agent.history['aNrm']
consumption = agent.history['cNrm']
```

## Model Structure

### IRASolution Class
Extends `ConsumerSolution` with:
- `cFunc`: Consumption function for liquid assets
- `cFunc_IRA`: Consumption function for IRA assets (currently same as cFunc)
- Standard value functions and marginal value functions

### IRAConsumerType Class
Extends `IndShockConsumerType` with IRA-specific parameters:
- Inherits all standard consumption model features
- Adds dual-account structure
- Implements penalty-adjusted optimization

### Solver Function
The `solve_ConsIRA` function:
1. Calculates effective IRA interest rate (with or without penalty)
2. Solves for optimal consumption and saving allocation
3. Chooses between liquid and IRA savings based on expected returns
4. Returns `IRASolution` with policy functions

## Default Parameters

```python
init_ira_accounts = {
    # Standard parameters (inherited from init_idiosyncratic_shocks)
    'cycles': 0,                    # Infinite horizon
    'T_cycle': 1,                   # Single period type
    'CRRA': 2.0,                    # Risk aversion
    'DiscFac': 0.96,                # Discount factor
    
    # IRA-specific parameters
    'Rfree_liquid_save': 1.03,      # 3% liquid saving rate
    'Rfree_liquid_boro': 1.20,      # 20% liquid borrowing rate  
    'Rfree_IRA_save': 1.07,         # 7% IRA saving rate
    'Rfree_IRA_boro': 1.00,         # No IRA borrowing
    'IRA_penalty_rate': 0.10,       # 10% early withdrawal penalty
    'retirement_age': 65,           # Penalty-free age
}
```

## Testing

The model includes comprehensive tests in `tests/ConsumptionSaving/test_ConsIRAModel.py`:
- Initialization tests
- Solver function validation  
- Penalty impact verification
- Kinked rate configuration checks
- Parameter inheritance validation

Run tests with:
```bash
python tests/ConsumptionSaving/test_ConsIRAModel.py
```

## Mathematical Framework

The agent maximizes:
```
V_t(m_t, a_IRA_t) = max_{c_t, s_liquid_t, s_IRA_t} u(c_t) + β E[V_{t+1}(m_{t+1}, a_IRA_{t+1})]
```

Subject to:
- Budget constraint: `m_t = c_t + s_liquid_t + s_IRA_t`
- Liquid asset evolution: `a_liquid_{t+1} = R_liquid * s_liquid_t`
- IRA asset evolution: `a_IRA_{t+1} = R_IRA_effective * s_IRA_t`  
- Early withdrawal penalty: `R_IRA_effective = R_IRA * (1 - penalty)` if `age < retirement_age`
- Kinked rates: Different R for positive vs negative asset positions

Where:
- `R_liquid` = `Rfree_liquid_save` if `s_liquid_t >= 0`, else `Rfree_liquid_boro`
- `R_IRA_effective` depends on age and penalty structure

## Future Enhancements

Potential extensions include:
1. **Contribution limits**: IRA annual contribution caps
2. **Required minimum distributions**: Forced withdrawals after age 70.5
3. **Roth vs Traditional**: Tax treatment differences
4. **Employer matching**: 401(k)-style employer contributions
5. **Multiple IRA types**: Different penalty structures
6. **Stochastic penalties**: Time-varying or uncertain penalty rates

## References

- G2EGM methodology: Jørgensen and Druedahl (2017), JEDC
- HARK consumption models: [econ-ark.org](https://econ-ark.org)
- Issue #136: [github.com/econ-ark/HARK/issues/136](https://github.com/econ-ark/HARK/issues/136)