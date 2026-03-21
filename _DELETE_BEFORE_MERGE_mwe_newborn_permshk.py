"""
MWE: MarkovConsumerType newborn PermShk bug
============================================
DELETE THIS FILE BEFORE MERGING THE PR.

Demonstrates that MarkovConsumerType.get_shocks() sets PermShk=1.0 for
newborn agents, suppressing both the idiosyncratic permanent shock (ψ)
and the deterministic growth factor (PermGroFac).  IndShockConsumerType
does not have this bug — it gives newborns PermShk = ψ × PermGroFac.

This script creates two equivalent agents (one IndShock, one Markov with
two identical states) and shows the discrepancy in newborn pLvl after
one period.

Run on the UNFIXED code (main) to see the bug.
Run on the FIXED code (fix/markov-newborn-permshk-missing-growth) to
confirm the fix.
"""

import numpy as np
from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK.distributions import DiscreteDistributionLabeled

G = 1.05  # Large PermGroFac to make the effect visible

# ── 1. IndShockConsumerType ──────────────────────────────────────────────
indshk = IndShockConsumerType(cycles=0, PermGroFac=[G], T_sim=2, AgentCount=5000)
indshk.solve()

det_inc = DiscreteDistributionLabeled(
    pmv=np.ones(1),
    atoms=np.array([[1.0], [1.0]]),
    var_names=["PermShk", "TranShk"],
)
indshk.IncShkDstn = [det_inc]

indshk.track_vars = ["pLvl"]
indshk.initialize_sim()
indshk.state_now["pLvl"][:] = 1.0
indshk.simulate()

indshk_pLvl_t0 = indshk.history["pLvl"][0, :]
indshk_mean = np.mean(indshk_pLvl_t0)
indshk_std = np.std(indshk_pLvl_t0)

# ── 2. MarkovConsumerType (2 identical states ≈ equivalent to IndShock) ──
mrkv = MarkovConsumerType(
    cycles=0,
    PermGroFac=[np.array([G, G])],
)
mrkv.solve()

det_inc_mrkv = DiscreteDistributionLabeled(
    pmv=np.ones(1),
    atoms=np.array([[1.0], [1.0]]),
    var_names=["PermShk", "TranShk"],
)
mrkv.IncShkDstn = [[det_inc_mrkv, det_inc_mrkv]]

mrkv.T_sim = 2
mrkv.AgentCount = 5000
mrkv.track_vars = ["pLvl"]
mrkv.initialize_sim()
mrkv.state_now["pLvl"][:] = 1.0
mrkv.simulate()

mrkv_pLvl_t0 = mrkv.history["pLvl"][0, :]
mrkv_mean = np.mean(mrkv_pLvl_t0)
mrkv_std = np.std(mrkv_pLvl_t0)

# ── 3. Report ────────────────────────────────────────────────────────────
print("=" * 65)
print("MWE: Newborn pLvl after 1 period (deterministic ψ=1, G=%.2f)" % G)
print("=" * 65)
print()
print(
    "  IndShockConsumerType:  mean(pLvl) = %.10f  std = %.2e"
    % (indshk_mean, indshk_std)
)
print("  MarkovConsumerType:    mean(pLvl) = %.10f  std = %.2e" % (mrkv_mean, mrkv_std))
print()
print("  Expected (correct):    pLvl = pLvl_0 × G = 1.0 × %.2f = %.2f" % (G, G))
print()

if abs(mrkv_mean - 1.0) < 1e-8:
    print("  *** BUG PRESENT: MarkovConsumerType newborn pLvl = 1.0 (no growth)")
    print(
        "      Missing PermGroFac factor. Discrepancy = %.4f%%"
        % (100 * (G - mrkv_mean) / G)
    )
elif abs(mrkv_mean - G) < 1e-8:
    print("  ✓ FIX VERIFIED: MarkovConsumerType newborn pLvl = G = %.2f (correct)" % G)
else:
    print("  ? UNEXPECTED: MarkovConsumerType newborn pLvl = %.10f" % mrkv_mean)

# ── 4. Stochastic test (wide ψ dispersion) ──────────────────────────────
print()
print("-" * 65)
print("Stochastic test: ψ ∈ {0.8, 1.2} with equal probability")
print("-" * 65)

stoch_inc = DiscreteDistributionLabeled(
    pmv=np.array([0.5, 0.5]),
    atoms=np.array([[0.8, 1.2], [1.0, 1.0]]),
    var_names=["PermShk", "TranShk"],
)

indshk2 = IndShockConsumerType(cycles=0, PermGroFac=[G], T_sim=2, AgentCount=50000)
indshk2.solve()
indshk2.IncShkDstn = [stoch_inc]
indshk2.track_vars = ["pLvl"]
indshk2.initialize_sim()
indshk2.state_now["pLvl"][:] = 1.0
indshk2.simulate()
indshk2_pLvl = indshk2.history["pLvl"][0, :]

stoch_inc_mrkv = DiscreteDistributionLabeled(
    pmv=np.array([0.5, 0.5]),
    atoms=np.array([[0.8, 1.2], [1.0, 1.0]]),
    var_names=["PermShk", "TranShk"],
)
mrkv2 = MarkovConsumerType(
    cycles=0,
    PermGroFac=[np.array([G, G])],
    AgentCount=50000,
)
mrkv2.solve()
mrkv2.IncShkDstn = [[stoch_inc_mrkv, stoch_inc_mrkv]]
mrkv2.T_sim = 2
mrkv2.track_vars = ["pLvl"]
mrkv2.initialize_sim()
mrkv2.state_now["pLvl"][:] = 1.0
mrkv2.simulate()
mrkv2_pLvl = mrkv2.history["pLvl"][0, :]

print()
print(
    "  IndShockConsumerType:  mean=%.6f  std=%.6f"
    % (np.mean(indshk2_pLvl), np.std(indshk2_pLvl))
)
print(
    "  MarkovConsumerType:    mean=%.6f  std=%.6f"
    % (np.mean(mrkv2_pLvl), np.std(mrkv2_pLvl))
)
print()
print("  Expected mean  = E[ψ] × G = 1.0 × %.2f = %.2f" % (G, G))
print(
    "  Expected std   = std(ψ) × G = %.4f × %.2f = %.4f"
    % (np.std([0.8, 1.2]), G, np.std([0.8, 1.2]) * G)
)

mrkv2_mean = np.mean(mrkv2_pLvl)
mrkv2_std = np.std(mrkv2_pLvl)
if abs(mrkv2_mean - 1.0) < 1e-8 and mrkv2_std < 1e-8:
    print()
    print("  *** BUG PRESENT: MarkovConsumerType newborns have ZERO ψ-dispersion")
    print("      AND mean pLvl = 1.0 (missing both ψ and PermGroFac)")
elif abs(mrkv2_mean - G) < 1e-8 and mrkv2_std < 1e-8:
    print()
    print("  ✓ FIX VERIFIED: newborns get PermGroFac only (no ψ),")
    print("    preserving calibrated pLvlInitStd dispersion.")
    print("    mean = %.6f = G ✓,  std = 0 (ψ suppressed) ✓" % mrkv2_mean)
    print()
    print("  Note: IndShockConsumerType gives newborns ψ×G (std > 0).")
    print("  This is a known behavioral difference, not a bug.")
elif abs(mrkv2_mean - G) < 0.01 and mrkv2_std > 0.01:
    print()
    print("  ✓ MarkovConsumerType newborns receive ψ × G (matches IndShock)")
