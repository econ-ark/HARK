# ConsumptionSavingX: Timing-Corrected Model Architecture

This module provides timing-corrected versions of HARK's consumption-saving models. The key difference from the original `ConsumptionSaving` module is in parameter indexing conventions.

## Timing Issue in Original HARK

In the original HARK design, parameters are indexed by when the **solver** needs them rather than the actual period they conceptually belong to. This creates confusing offsets:

- Parameters like `Rfree` or shock distributions that apply in period t+1 are fed into the period t solver
- Lifecycle implementations require shifting parameter lists by one index to align with true timing
- Inconsistent indexing between different parameter types (e.g., `Rfree[t_cycle]` vs `PermGroFac[t_cycle-1]`)
- Newborn agents need special handling because terminal period values don't exist

### Example of Original Timing Issue

```python
# In get_shocks() - uses t_cycle - 1
self.shocks["PermShk"] = PermGroFac[self.t_cycle - 1]  

# In get_Rfree() - uses t_cycle directly  
return Rfree_array[self.t_cycle]

# In sim_death() - conditional logic
DiePrb = DiePrb_by_t_cycle[
    self.t_cycle - 1 if self.cycles == 1 else self.t_cycle
]

# Newborn hack - arbitrary use of period 0
IncShkDstnNow = self.IncShkDstn[0]  # For newborns!
```

## Timing-Corrected Design

In `ConsumptionSavingX`, period t parameters correspond to period t:

- `Rfree[t]` is the interest rate that applies in period t
- `LivPrb[t]` is the survival probability for period t  
- `PermGroFac[t]` is the growth factor applied in period t
- **Consistent indexing logic**: `t_cycle - 1 if self.cycles == 1 else t_cycle` for all parameters
- **Eliminates newborn hack**: Proper parameter indexing means newborns get correct distributions

### Example of Timing-Corrected Code

```python
# Consistent indexing in all methods
t_index = t_cycle - 1 if self.cycles == 1 else t_cycle

# All parameters use the same logic
Rfree_array[t_index]
PermGroFac[t_index] 
IncShkDstn[t_index]

# No special newborn handling needed - they get proper parameters automatically
```

## Usage

To use the timing-corrected models, simply change your import:

```python
# Original
from HARK.ConsumptionSaving.ConsIndShockModel import init_lifecycle

# Timing-corrected  
from HARK.ConsumptionSavingX.ConsIndShockModel import init_lifecycle_X
```

## Key Changes

1. **Parameter Creation**: `init_lifecycle_X` creates parameter lists with corrected timing
2. **Parameter Access**: All methods use consistent indexing logic
3. **Newborn Handling**: Eliminates arbitrary period-0 fallback for newborns
4. **Documentation**: Clear comments explain timing conventions

## Compatibility

- **Infinite-horizon models**: Should produce identical results (timing doesn't matter for cyclical patterns)
- **Finite-horizon models**: May show small differences due to corrected parameter timing
- **Interface**: Same API, just different timing semantics

This timing-corrected architecture provides a foundation for cleaner model specification and better modularity between solvers, simulators, and model definitions.