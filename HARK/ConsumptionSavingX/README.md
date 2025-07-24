# ConsumptionSavingX: Timing-Corrected Model Architecture

This module provides timing-corrected versions of HARK's consumption-saving models. The key difference from the original `ConsumptionSaving` module is in parameter indexing conventions.

## Timing Issue in Original HARK

In the original HARK design, parameters are indexed by when the **solver** needs them rather than the actual period they conceptually belong to. This creates confusing offsets:

- Parameters like `Rfree` or shock distributions that apply in period t+1 are fed into the period t solver
- Lifecycle implementations require shifting parameter lists by one index to align with true timing
- Inconsistent indexing between different parameter types (e.g., `Rfree[t_cycle]` vs `PermGroFac[t_cycle-1]`)

## Timing-Corrected Design

In `ConsumptionSavingX`, period t parameters correspond to period t:

- `Rfree[t]` is the interest rate that applies in period t
- `LivPrb[t]` is the survival probability for period t  
- `PermGroFac[t]` is the growth factor applied in period t
- Consistent indexing logic: `t_cycle - 1 if self.cycles == 1 else t_cycle`

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
2. **Parameter Access**: `get_Rfree()` and `get_shocks()` use consistent indexing logic
3. **Documentation**: Clear comments explain timing conventions

## Compatibility

- **Infinite-horizon models**: Should produce identical results (timing doesn't matter for cyclical patterns)
- **Finite-horizon models**: May show small differences due to corrected parameter timing
- **Interface**: Same API, just different timing semantics

This timing-corrected architecture provides a foundation for cleaner model specification and better modularity between solvers, simulators, and model definitions.