# History of Heterogeneous Agent Models

## The "Bottom Up" Approach

The development of heterogeneous agent models follows a "bottom up" methodology: researchers first calibrate models to match micro data on individual behavior, then aggregate these models to understand macroeconomic implications. This approach contrasts with representative agent models that assume away heterogeneity from the start.

## Foundational Contributions

### Early Life-Cycle Theory

**Modigliani and Brumberg (1954)** - Life-Cycle Hypothesis
- Established the foundation that consumption depends on lifetime resources, not just current income
- Introduced the idea that individuals smooth consumption over their lifetime
- Provided the first rigorous micro foundation for aggregate consumption behavior

**Friedman (1957)** - Permanent Income Hypothesis
- Distinguished between permanent and transitory income components
- Showed that consumption responds primarily to permanent income changes
- Laid groundwork for understanding how uncertainty affects saving behavior

### Zeldes (1989) - Precautionary Saving Under Uncertainty

Stephen P. Zeldes's 1989 paper "Optimal Consumption with Stochastic Income: Deviations from Certainty Equivalence" (Quarterly Journal of Economics, 104(2), 275-298) represents a watershed moment in consumption theory and heterogeneous agent modeling.

**Key Contributions:**
- First rigorous numerical solution of consumption models with precautionary saving motives
- Demonstrated that uncertainty about future income leads to additional saving beyond certainty-equivalent models
- Showed that liquidity constraints amplify precautionary motives
- Introduced computational methods that became standard in the field

**Technical Innovations:**
- Used dynamic programming with numerical integration
- Solved backward from the last period of life
- Incorporated realistic income processes with permanent and transitory shocks
- Demonstrated the importance of the third derivative of utility (prudence)

### Deaton (1991) - Liquidity Constraints and Buffer-Stock Saving

Angus Deaton's 1991 paper "Saving and Liquidity Constraints" (Econometrica, 59(5), 1221-1248) introduced the concept of buffer-stock saving behavior.

**Key Contributions:**
- Showed that transitory income shocks combined with liquidity constraints generate buffer-stock saving
- Demonstrated that consumers hold a target level of wealth as a buffer against income uncertainty
- Explained why consumption tracks income closely despite forward-looking behavior
- Introduced the term "buffer-stock" saving to the literature

**Model Features:**
- Infinite horizon framework
- Transitory income shocks only (no permanent shocks)
- Hard borrowing constraint at zero
- CRRA utility function

### Carroll (1992, 1997) - Buffer-Stock Theory Without Liquidity Constraints

Christopher D. Carroll extended and refined the buffer-stock theory in two landmark papers:

**"The Buffer-Stock Theory of Saving" (1992, Brookings Papers on Economic Activity)**
- Showed buffer-stock behavior emerges even WITHOUT liquidity constraints
- Key insight: impatience (high time preference) combined with prudence generates target saving
- Explained cyclical consumption/saving dynamics over the business cycle
- Introduced the concept of the "buffer-stock range" of wealth

**"Buffer-Stock Saving and the Life Cycle/Permanent Income Hypothesis" (1997, QJE)**
- Extended buffer-stock theory to life cycle models
- Incorporated both permanent and transitory income shocks
- Showed buffer-stock behavior dominates early in life, transitioning to life-cycle saving later
- Developed conditions for when buffer-stock behavior emerges (growth, impatience, prudence)

### Hubbard, Skinner, and Zeldes (1994, 1995) - Social Insurance and Precautionary Saving

The papers by R. Glenn Hubbard, Jonathan Skinner, and Stephen P. Zeldes extended the framework to policy-relevant questions:

1. **"The Importance of Precautionary Motives in Explaining Individual and Aggregate Saving"** (Carnegie-Rochester Conference Series on Public Policy, 1994)
2. **"Precautionary Saving and Social Insurance"** (Journal of Political Economy, 1995)

**Key Contributions:**
- Incorporated health expenditure uncertainty as a major source of precautionary saving
- Analyzed the crowd-out effects of social insurance programs on private saving
- Demonstrated that means-tested programs can have large effects on wealth accumulation
- Quantified the insurance value of Social Security and Medicare

**Model Features:**
- Multi-period lifecycle framework
- Uncertain medical expenses that increase with age
- Means-tested social insurance (Medicaid)
- Bequest motives
- Realistic calibration to U.S. data

## Connection to Modern HARK Implementation

HARK (Heterogeneous Agents Resources and toolKit) builds directly on these foundational insights:

### Buffer Stock Models
- `ConsIndShockModel` implements the basic framework from Zeldes (1989)
- Precautionary saving emerges naturally from prudent preferences
- Numerical methods follow the dynamic programming approach pioneered by Zeldes

### Healthcare and Insurance Models
- Extensions to include medical expense shocks follow Hubbard, Skinner, and Zeldes
- Policy analysis tools can evaluate social insurance programs
- Means-testing and asset limits can be incorporated

### Key Parameters Influenced by These Papers:
```python
# Zeldes (1989) influence - prudence and precautionary saving
CRRA = 2.0  # Coefficient of relative risk aversion (implies prudence = CRRA + 1)

# Hubbard, Skinner, Zeldes influence - medical expense risk
MedShkPrb = 0.10  # Probability of medical expense shock
MedShkSize = 5.0  # Size of medical expense relative to income
```

## Modern Extensions in HARK

Building on this foundation, HARK now includes:
- Aggregate uncertainty (Krusell-Smith type models)
- Portfolio choice with risky assets
- Durable goods and housing
- Heterogeneous preferences and beliefs
- Machine learning integration for policy functions

## References

1. Modigliani, F., & Brumberg, R. (1954). "Utility Analysis and the Consumption Function: An Interpretation of Cross-Section Data." In K. K. Kurihara (Ed.), *Post-Keynesian Economics*. New Brunswick, NJ: Rutgers University Press.

2. Friedman, M. (1957). *A Theory of the Consumption Function*. Princeton, NJ: Princeton University Press.

3. Zeldes, S. P. (1989). "Optimal Consumption with Stochastic Income: Deviations from Certainty Equivalence." *Quarterly Journal of Economics*, 104(2), 275-298.

4. Deaton, A. (1991). "Saving and Liquidity Constraints." *Econometrica*, 59(5), 1221-1248.

5. Carroll, C. D. (1992). "The Buffer-Stock Theory of Saving: Some Macroeconomic Evidence." *Brookings Papers on Economic Activity*, 1992(2), 61-156.

6. Hubbard, R. G., Skinner, J., & Zeldes, S. P. (1994). "The Importance of Precautionary Motives in Explaining Individual and Aggregate Saving." *Carnegie-Rochester Conference Series on Public Policy*, 40, 59-125.

7. Hubbard, R. G., Skinner, J., & Zeldes, S. P. (1995). "Precautionary Saving and Social Insurance." *Journal of Political Economy*, 103(2), 360-399.

8. Carroll, C. D. (1997). "Buffer-Stock Saving and the Life Cycle/Permanent Income Hypothesis." *Quarterly Journal of Economics*, 112(1), 1-55.

9. Gourinchas, P. O., & Parker, J. A. (2002). "Consumption Over the Life Cycle." *Econometrica*, 70(1), 47-89. [Empirical validation of buffer-stock and precautionary saving models] 