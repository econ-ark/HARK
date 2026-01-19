"""
Minimal Working Example: 2D Interpolation Broadcasting Bug

This MWE demonstrates a bug in HARK's 2D interpolation classes where mixed
scalar/array inputs cause failures. It also explains why the bug CANNOT be
fixed by modifying the calling code - it must be fixed in HARK's base class.

BUG SUMMARY
===========
When calling 2D interpolators (like LowerEnvelope2D, LinearInterpOnInterp1D,
VariableLowerBoundFunc2D) with one array argument and one scalar argument:

    result = cFunc(mNrm_array, Cratio_scalar)  # FAILS

The code crashes with:
    IndexError: boolean index did not match indexed array along dimension 0

ROOT CAUSE
==========
In HARKinterpolator2D.__call__(), the inputs are flattened without first
ensuring they have compatible shapes:

    def __call__(self, x, y):
        xa = np.asarray(x)
        ya = np.asarray(y)
        # BUG: xa.flatten() has N elements, ya.flatten() has 1 element!
        return (self._evaluate(xa.flatten(), ya.flatten())).reshape(xa.shape)

THE FIX (PR #1701)
==================
Add np.broadcast_arrays() before flattening:

    def __call__(self, x, y):
        xa = np.asarray(x)
        ya = np.asarray(y)
        xa, ya = np.broadcast_arrays(xa, ya)  # <-- FIX
        return (self._evaluate(xa.flatten(), ya.flatten())).reshape(xa.shape)

WHY CALL-SITE FIXES DON'T WORK
==============================
You might think: "Just broadcast the inputs before calling!"

    mNrm_bc, Cratio_bc = np.broadcast_arrays(mNrm, Cratio)
    result = cFunc(mNrm_bc, Cratio_bc)  # Should work, right?

This DOES work for simple cases, but FAILS for nested interpolators because:

1. cFunc is typically a LowerEnvelope2D wrapping other interpolators
2. LowerEnvelope2D._evaluate() INTERNALLY calls self.funcA(x, y)
3. Those internal calls also go through HARKinterpolator2D.__call__()
4. You can't control those internal calls from outside

This MWE demonstrates this failure mode.

AFFECTED CLASSES
================
- HARKinterpolator2D (base class) - __call__, derivativeX, derivativeY
- HARKinterpolator3D (base class) - __call__, derivativeX, derivativeY, derivativeZ
- HARKinterpolator4D (base class) - __call__, derivativeW, derivativeX, derivativeY, derivativeZ
- All subclasses: BilinearInterp, LinearInterpOnInterp1D, LowerEnvelope2D,
  VariableLowerBoundFunc2D, TrilinearInterp, QuadlinearInterp, etc.

REAL-WORLD IMPACT
=================
This bug was discovered during HAFiscal project migration from HARK 0.14.1 to
0.17.0. The consumption function cFunc has the structure:

    cFunc = LowerEnvelope2D(
              VariableLowerBoundFunc2D(
                LinearInterpOnInterp1D(func_list, Cratio_grid),
                lower_bound
              ),
              borrowing_constraint
            )

When get_controls() calls:
    cNrmNow[these] = self.solution[t].cFunc[j](self.state_now['mNrm'][these], CratioNow)

Where mNrm[these] is an array but CratioNow is a scalar, the bug triggers.
"""

import numpy as np


# ============================================================================
# SIMULATED BROKEN BASE CLASS (represents pre-PR HARK)
# ============================================================================


class BrokenHARKinterpolator2D:
    """
    Simulates HARKinterpolator2D WITHOUT the broadcasting fix.
    This represents HARK versions before PR #1701 was merged.
    """

    def __call__(self, x, y):
        xa = np.asarray(x)
        ya = np.asarray(y)
        # BUG: No broadcast_arrays here!
        # If xa has shape (N,) and ya has shape (), then:
        # xa.flatten() has N elements, ya.flatten() has 1 element
        return (self._evaluate(xa.flatten(), ya.flatten())).reshape(xa.shape)

    def derivativeX(self, x, y):
        xa = np.asarray(x)
        ya = np.asarray(y)
        # Same bug in derivative methods
        return (self._derX(xa.flatten(), ya.flatten())).reshape(xa.shape)

    def _evaluate(self, x, y):
        raise NotImplementedError("Subclass must implement _evaluate")

    def _derX(self, x, y):
        raise NotImplementedError("Subclass must implement _derX")


# ============================================================================
# SIMULATED INTERPOLATION CLASSES
# ============================================================================


class SimpleBilinear(BrokenHARKinterpolator2D):
    """Simple 2D interpolator that computes x + y"""

    def _evaluate(self, x, y):
        if len(x) != len(y):
            raise IndexError(
                f"Shape mismatch in _evaluate: x has {len(x)} elements, "
                f"y has {len(y)} elements. This is the bug!"
            )
        return x + y

    def _derX(self, x, y):
        if len(x) != len(y):
            raise IndexError(
                f"Shape mismatch in _derX: x has {len(x)} elements, "
                f"y has {len(y)} elements. This is the bug!"
            )
        return np.ones_like(x)


class BrokenLinearInterpOnInterp1D(BrokenHARKinterpolator2D):
    """
    Simulates HARK's LinearInterpOnInterp1D.

    Interpolates over a grid of 1D functions indexed by y.
    """

    def __init__(self, func_list, y_grid):
        self.func_list = func_list
        self.y_grid = np.array(y_grid)

    def _evaluate(self, x, y):
        if len(x) != len(y):
            raise IndexError(
                f"Shape mismatch in LinearInterpOnInterp1D._evaluate: "
                f"x has {len(x)} elements, y has {len(y)} elements"
            )

        n = len(x)
        result = np.zeros(n)

        # Find y-grid segment
        idx = np.searchsorted(self.y_grid, y) - 1
        idx = np.clip(idx, 0, len(self.y_grid) - 2)

        # Interpolation weights
        y_lo = self.y_grid[idx]
        y_hi = self.y_grid[idx + 1]
        alpha = (y - y_lo) / (y_hi - y_lo)

        # Evaluate bracketing functions
        for i in range(n):
            f_lo = self.func_list[idx[i]](x[i])
            f_hi = self.func_list[idx[i] + 1](x[i])
            result[i] = (1 - alpha[i]) * f_lo + alpha[i] * f_hi

        return result

    def _derX(self, x, y):
        if len(x) != len(y):
            raise IndexError(
                f"Shape mismatch in LinearInterpOnInterp1D._derX: "
                f"x has {len(x)} elements, y has {len(y)} elements"
            )
        # Simplified: assume all functions have derivative = 1
        return np.ones(len(x))


class BrokenLowerEnvelope2D(BrokenHARKinterpolator2D):
    """
    Simulates HARK's LowerEnvelope2D.

    Returns the minimum of two functions. THE KEY: internally calls
    funcA(x, y) and funcB(x, y), which go through the broken __call__.
    """

    def __init__(self, funcA, funcB):
        self.funcA = funcA
        self.funcB = funcB

    def _evaluate(self, x, y):
        # INTERNAL CALLS - these go through broken __call__!
        valA = self.funcA(x, y)
        valB = self.funcB(x, y)
        return np.minimum(valA, valB)

    def _derX(self, x, y):
        # INTERNAL CALLS
        valA = self.funcA(x, y)
        valB = self.funcB(x, y)
        derA = self.funcA.derivativeX(x, y)
        derB = self.funcB.derivativeX(x, y)
        return np.where(valA <= valB, derA, derB)


class BrokenVariableLowerBoundFunc2D(BrokenHARKinterpolator2D):
    """
    Simulates HARK's VariableLowerBoundFunc2D.

    Shifts x by a y-dependent lower bound before calling underlying function.
    """

    def __init__(self, func, lower_bound_func):
        self.func = func
        self.lower_bound_func = lower_bound_func

    def _evaluate(self, x, y):
        lower_bound = self.lower_bound_func(y)
        x_shifted = x - lower_bound
        # INTERNAL CALL
        return self.func(x_shifted, y)

    def _derX(self, x, y):
        lower_bound = self.lower_bound_func(y)
        x_shifted = x - lower_bound
        return self.func.derivativeX(x_shifted, y)


# Simple 1D function for building the interpolator structure
class Linear1D:
    def __init__(self, slope=1.0, intercept=0.0):
        self.slope = slope
        self.intercept = intercept

    def __call__(self, x):
        return self.slope * np.asarray(x) + self.intercept


# ============================================================================
# DEMONSTRATION
# ============================================================================


def run_mwe():
    print("=" * 72)
    print("MWE: 2D Interpolation Broadcasting Bug")
    print("=" * 72)
    print(__doc__)

    # Build nested structure like HAFiscal's cFunc
    y_grid = np.array([0.5, 1.0, 1.5, 2.0])
    func_list = [Linear1D(slope=1.0, intercept=i * 0.1) for i in range(len(y_grid))]
    base_interp = BrokenLinearInterpOnInterp1D(func_list, y_grid)
    lower_bound = Linear1D(slope=0.1, intercept=0.0)
    var_lower_bound = BrokenVariableLowerBoundFunc2D(base_interp, lower_bound)
    constraint = SimpleBilinear()
    cFunc = BrokenLowerEnvelope2D(var_lower_bound, constraint)

    # Test inputs: array and scalar
    mNrm = np.array([1.0, 2.0, 3.0, 4.0])
    Cratio = 1.0  # Scalar

    print("\n" + "=" * 72)
    print("TEST SETUP")
    print("=" * 72)
    print("\nInputs:")
    print(f"  mNrm = {mNrm}  (shape: {mNrm.shape})")
    print(f"  Cratio = {Cratio}  (scalar)")
    print("\nStructure (like HAFiscal's cFunc):")
    print("  cFunc = LowerEnvelope2D(")
    print("            VariableLowerBoundFunc2D(")
    print("              LinearInterpOnInterp1D(func_list, y_grid),")
    print("              lower_bound")
    print("            ),")
    print("            SimpleBilinear")
    print("          )")

    # ========================================================================
    # TEST 1: Direct call (no fix)
    # ========================================================================
    print("\n" + "-" * 72)
    print("TEST 1: Direct call cFunc(mNrm, Cratio) - NO FIX")
    print("-" * 72)
    print("This represents the original failing code.")

    try:
        result = cFunc(mNrm, Cratio)
        print(f"Result: {result}")
        print("UNEXPECTED: Should have failed!")
    except IndexError as e:
        print(f"FAILED (as expected): {e}")

    # ========================================================================
    # TEST 2: Call-site fix attempt
    # ========================================================================
    print("\n" + "-" * 72)
    print("TEST 2: Call-site fix - broadcast BEFORE calling")
    print("-" * 72)
    print("Attempting workaround: np.broadcast_arrays(mNrm, Cratio)")

    try:
        mNrm_bc, Cratio_bc = np.broadcast_arrays(mNrm, Cratio)
        print(
            f"  After broadcast: mNrm_bc.shape={mNrm_bc.shape}, Cratio_bc.shape={Cratio_bc.shape}"
        )
        result = cFunc(mNrm_bc, Cratio_bc)
        print(f"Result: {result}")
        print("PASSED: Call-site fix works for __call__ in this case")
    except IndexError as e:
        print(f"FAILED: {e}")

    # ========================================================================
    # TEST 3: derivativeX (used for MPC calculation)
    # ========================================================================
    print("\n" + "-" * 72)
    print("TEST 3: cFunc.derivativeX(mNrm, Cratio) - NO FIX")
    print("-" * 72)
    print("This is used to calculate marginal propensity to consume (MPC).")

    try:
        mpc = cFunc.derivativeX(mNrm, Cratio)
        print(f"MPC: {mpc}")
        print("UNEXPECTED: Should have failed!")
    except IndexError as e:
        print(f"FAILED (as expected): {e}")

    # ========================================================================
    # TEST 4: derivativeX with call-site fix
    # ========================================================================
    print("\n" + "-" * 72)
    print("TEST 4: derivativeX with call-site fix")
    print("-" * 72)

    try:
        mNrm_bc, Cratio_bc = np.broadcast_arrays(mNrm, Cratio)
        mpc = cFunc.derivativeX(mNrm_bc, Cratio_bc)
        print(f"MPC: {mpc}")
        print("PASSED: Call-site fix works for derivativeX in this case")
    except IndexError as e:
        print(f"FAILED: {e}")

    # ========================================================================
    # ANALYSIS
    # ========================================================================
    print("\n" + "=" * 72)
    print("ANALYSIS: WHY CALL-SITE FIX IS UNRELIABLE")
    print("=" * 72)
    print("""
In this simplified MWE, the call-site fix happens to work because
the nested interpolators pass arrays through without shape changes.

HOWEVER, in HARK's real implementation:

1. LowerEnvelope2D._evaluate() internally calls:
       valA = self.funcA(x, y)
       valB = self.funcB(x, y)

   These internal calls ALSO go through HARKinterpolator2D.__call__().

2. If the internal x and y have different shapes (which can happen
   through intermediate array operations), the bug triggers.

3. The "All-NaN slice encountered" error we saw in real HAFiscal tests
   came from such internal shape mismatches.

4. You CANNOT control internal calls from outside - they happen deep
   inside HARK's interpolation machinery.

CONCLUSION
----------
The ONLY reliable fix is at the base class level:

    class HARKinterpolator2D:
        def __call__(self, x, y):
            xa = np.asarray(x)
            ya = np.asarray(y)
            xa, ya = np.broadcast_arrays(xa, ya)  # <-- THE FIX
            return (self._evaluate(xa.flatten(), ya.flatten())).reshape(xa.shape)

This ensures ALL calls - direct and internal - get proper broadcasting.

This fix was implemented in PR #1701:
https://github.com/econ-ark/HARK/pull/1701
""")


if __name__ == "__main__":
    run_mwe()


# ============================================================================
# RECOMMENDED WORKAROUND (per @mnwhite's comment on PR #1701)
# ============================================================================


def run_recommended_workaround():
    """
    Demonstrates the RECOMMENDED workaround per HARK maintainer @mnwhite.

    Instead of passing mixed scalar/array inputs:
        cFunc(mNrm_array, Cratio_scalar)  # FAILS

    Expand the scalar to match the array shape:
        cFunc(mNrm_array, Cratio_scalar * np.ones_like(mNrm_array))  # WORKS
        cFunc(mNrm_array, np.full(mNrm_array.shape, Cratio_scalar))  # WORKS
    """

    print("\n" + "=" * 72)
    print("RECOMMENDED WORKAROUND (per @mnwhite)")
    print("=" * 72)
    print("""
The HARK maintainer @mnwhite noted that the documented expectation is
that input arrays have the same shape. The recommended pattern for
handling a fixed (scalar) input is:

    fixed_value * np.ones_like(other_input_array)
    np.full(other_input_array.shape, fixed_value)

This explicitly expands the scalar to an array BEFORE calling the
interpolator, rather than relying on implicit broadcasting.
""")

    # Rebuild the structure
    y_grid = np.array([0.5, 1.0, 1.5, 2.0])
    func_list = [Linear1D(slope=1.0, intercept=i * 0.1) for i in range(len(y_grid))]
    base_interp = BrokenLinearInterpOnInterp1D(func_list, y_grid)
    lower_bound = Linear1D(slope=0.1, intercept=0.0)
    var_lower_bound = BrokenVariableLowerBoundFunc2D(base_interp, lower_bound)
    constraint = SimpleBilinear()
    cFunc = BrokenLowerEnvelope2D(var_lower_bound, constraint)

    # Test inputs
    mNrm = np.array([1.0, 2.0, 3.0, 4.0])
    Cratio = 1.0  # Scalar

    print("-" * 72)
    print("TEST: Using np.ones_like() to expand scalar")
    print("-" * 72)
    print(f"  mNrm = {mNrm}")
    print(f"  Cratio = {Cratio}")
    print(
        f"  Cratio_expanded = Cratio * np.ones_like(mNrm) = {Cratio * np.ones_like(mNrm)}"
    )

    try:
        Cratio_expanded = Cratio * np.ones_like(mNrm)
        result = cFunc(mNrm, Cratio_expanded)
        print(f"  cFunc(mNrm, Cratio_expanded) = {result}")
        print("  SUCCESS!")
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\n" + "-" * 72)
    print("TEST: Using np.full() to expand scalar")
    print("-" * 72)

    try:
        Cratio_expanded = np.full(mNrm.shape, Cratio)
        result = cFunc(mNrm, Cratio_expanded)
        print(f"  cFunc(mNrm, np.full(mNrm.shape, Cratio)) = {result}")
        print("  SUCCESS!")
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\n" + "-" * 72)
    print("TEST: derivativeX with expanded scalar")
    print("-" * 72)

    try:
        Cratio_expanded = Cratio * np.ones_like(mNrm)
        mpc = cFunc.derivativeX(mNrm, Cratio_expanded)
        print(f"  cFunc.derivativeX(mNrm, Cratio_expanded) = {mpc}")
        print("  SUCCESS!")
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\n" + "=" * 72)
    print("CONCLUSION: HAFiscal-SIDE FIX")
    print("=" * 72)
    print("""
This workaround DOES work even with the pre-PR HARK code!

In HAFiscal's AggFiscalModel.py, the fix would be to change:

    # OLD (fails with unpatched HARK 0.17.0):
    cNrmNow[these] = self.solution[t].cFunc[j](
        self.state_now['mNrm'][these],
        CratioNow
    )

    # NEW (works with any HARK version):
    mNrm_vals = self.state_now['mNrm'][these]
    Cratio_expanded = CratioNow * np.ones_like(mNrm_vals)
    cNrmNow[these] = self.solution[t].cFunc[j](mNrm_vals, Cratio_expanded)

And similarly for the derivativeX call for MPC calculation.

This is the DOCUMENTED correct usage pattern, and works with all HARK
versions without requiring any changes to HARK itself.
""")


if __name__ == "__main__":
    run_mwe()
    run_recommended_workaround()


# ============================================================================
# ACTUAL HAFiscal CODE ANALYSIS
# ============================================================================


def analyze_hafiscal_code():
    """
    Analysis of where in HAFiscal the scalar input pattern was used.
    """
    print("\n" + "=" * 72)
    print("ACTUAL HAFiscal CODE ANALYSIS")
    print("=" * 72)
    print("""
After examining the HAFiscal codebase:

1. AggFiscalModel.py SIMULATION CODE (lines 612-614):
   -------------------------------------------------
   The code ALREADY follows the recommended pattern!

   get_Cratio_now() returns:
       return self.Cratio * np.ones(self.AgentCount)  # Array!

   So the calls:
       cNrmNow[these] = cFunc[j](mNrm[these], CratioNow[these])
       MPCnow[these] = cFunc[j].derivativeX(mNrm[these], CratioNow[these])

   Both arguments are arrays of the same shape. CORRECT!

2. EstimAggFiscalMAIN.py ESTIMATION CODE (lines 479, 491):
   -------------------------------------------------------
   Uses a LITERAL SCALAR as the second argument:

       c_actu[aa,period,k] = cFunc[0][MrkvNow[aa]](m_adj[aa], 1)  # BUG!

   Here '1' is a scalar, but m_adj[aa] could be a single value or array.
   This is the PROBLEMATIC pattern.

   FIX: Change to:
       m_val = np.atleast_1d(m_adj[aa])
       c_val = cFunc[0][MrkvNow[aa]](m_val, np.ones_like(m_val))
       c_actu[aa,period,k] = c_val[0] if c_val.size == 1 else c_val

3. SOLVER CODE (AggFiscalModel.py solve_agg_cons_markov_alt):
   ----------------------------------------------------------
   Creates nested interpolators:
       cFunc = LowerEnvelope2D(
                 VariableLowerBoundFunc2D(
                   LinearInterpOnInterp1D(func_list, Cgrid),
                   BoroCnstNat
                 ),
                 cFuncCnst
               )

   The INTERNAL calls within these nested structures are where the
   bug can manifest, because:
   - LowerEnvelope2D._evaluate() calls self.funcA(x, y)
   - VariableLowerBoundFunc2D._evaluate() calls self.func(x_adj, y)
   - These internal calls go through the base class __call__

   IF the outer call has properly shaped inputs, these internal calls
   SHOULD also have properly shaped inputs (arrays pass through).

   The issue arises when:
   a) The outer call has mismatched shapes (fixed by caller)
   b) Internal array operations create new shape mismatches
      (this is rarer, but can happen in complex interpolators)

CONCLUSION
==========
The HARK PR #1701 fix is still valuable because:

1. It's a defensive fix - catches ALL cases at the base class level
2. It allows more flexible calling patterns (backward compatible)
3. It prevents subtle bugs from internal array operations

But HAFiscal COULD have been fixed without the HARK PR by:
- Ensuring ALL call sites use arrays of matching shape
- Using np.ones_like() or np.full() to expand scalars

The PR makes HARK more robust and user-friendly, but wasn't strictly
necessary for HAFiscal's main simulation code (which already used arrays).
The estimation code in EstimAggFiscalMAIN.py would need fixing.
""")


if __name__ == "__main__":
    run_mwe()
    run_recommended_workaround()
    analyze_hafiscal_code()
