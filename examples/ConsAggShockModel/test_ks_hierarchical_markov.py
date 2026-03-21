"""
Comparison test: original KrusellSmithType vs. AggIndMarkovConsumerType-based
KrusellSmithTypeHM.  Both models use identical parameters and the same
aggregate Markov shock history.  We verify that the converged aggregate
saving rules match within tolerance.
"""

import time

from HARK.ConsumptionSaving.ConsAggShockModel import (
    KrusellSmithType,
    KrusellSmithEconomy,
    KrusellSmithTypeHM,
    KrusellSmithEconomyHM,
)

# ── 1. Solve the ORIGINAL Krusell-Smith model ──────────────────────────────

print("=" * 70)
print("ORIGINAL KrusellSmithType / KrusellSmithEconomy")
print("=" * 70)

KSagents_orig = KrusellSmithType(seed=0)
KSeconomy_orig = KrusellSmithEconomy(agents=[KSagents_orig], verbose=True)
KSeconomy_orig.make_Mrkv_history()
KSeconomy_orig.give_agent_params()

t0 = time.time()
KSeconomy_orig.solve()
t1 = time.time()
print(f"Original model solved in {t1 - t0:.2f} seconds.\n")

orig_intercepts = list(KSeconomy_orig.intercept_prev)
orig_slopes = list(KSeconomy_orig.slope_prev)
print(f"  intercept = {orig_intercepts}")
print(f"  slope     = {orig_slopes}")

# ── 2. Solve the NEW (Hierarchical Markov) Krusell-Smith model ─────────────

print()
print("=" * 70)
print("NEW KrusellSmithTypeHM / KrusellSmithEconomyHM")
print("=" * 70)

KSagents_new = KrusellSmithTypeHM(seed=0)
KSeconomy_new = KrusellSmithEconomyHM(agents=[KSagents_new], verbose=True)
KSeconomy_new.make_Mrkv_history()
KSeconomy_new.give_agent_params()

t0 = time.time()
KSeconomy_new.solve()
t1 = time.time()
print(f"New model solved in {t1 - t0:.2f} seconds.\n")

new_intercepts = list(KSeconomy_new.intercept_prev)
new_slopes = list(KSeconomy_new.slope_prev)
print(f"  intercept = {new_intercepts}")
print(f"  slope     = {new_slopes}")

# ── 3. Compare results ─────────────────────────────────────────────────────

print()
print("=" * 70)
print("COMPARISON")
print("=" * 70)

tol = 1e-6
all_pass = True

for i, label in enumerate(["Bad", "Good"]):
    d_int = abs(orig_intercepts[i] - new_intercepts[i])
    d_slp = abs(orig_slopes[i] - new_slopes[i])
    pass_int = d_int < tol
    pass_slp = d_slp < tol
    status_int = "PASS" if pass_int else "FAIL"
    status_slp = "PASS" if pass_slp else "FAIL"
    print(
        f"  {label} state intercept: orig={orig_intercepts[i]:.10f}  new={new_intercepts[i]:.10f}  diff={d_int:.2e}  [{status_int}]"
    )
    print(
        f"  {label} state slope:     orig={orig_slopes[i]:.10f}  new={new_slopes[i]:.10f}  diff={d_slp:.2e}  [{status_slp}]"
    )
    all_pass = all_pass and pass_int and pass_slp

print()
if all_pass:
    print("*** ALL CHECKS PASSED — models produce identical results. ***")
else:
    print("*** SOME CHECKS FAILED — see above for details. ***")
    # Also compare at a looser tolerance
    loose_tol = 1e-3
    loose_pass = True
    for i in range(2):
        if abs(orig_intercepts[i] - new_intercepts[i]) > loose_tol:
            loose_pass = False
        if abs(orig_slopes[i] - new_slopes[i]) > loose_tol:
            loose_pass = False
    if loose_pass:
        print(f"  (Results match within loose tolerance of {loose_tol})")
    else:
        print(f"  (Results do NOT match even at loose tolerance of {loose_tol})")
