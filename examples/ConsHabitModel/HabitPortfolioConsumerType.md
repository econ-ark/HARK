# HabitPortfolioConsumerType: Mathematical Description

## Overview

The `HabitPortfolioConsumerType` is a lifecycle consumption-saving model that combines **habit formation** (where utility depends on the ratio of consumption to a habit stock) with **portfolio choice** (where the agent allocates savings between a risk-free and a risky asset). It is implemented in the HARK toolkit as `HARK.ConsumptionSaving.ConsHabitPortfolioModel`.

The model extends two parent models:
- **HabitConsumerType** (`ConsHabitModel.py`): 2D state space (m, h), EGM with FOC-inverter, risk-free savings only.
- **PortfolioConsumerType** (`ConsPortfolioModel.py`): 1D state space (m), portfolio choice via share search over risky return distribution.

---

## 1. Preferences

The agent has CRRA preferences over effective consumption, where the habit stock acts as a scaling divisor:

$$u(c_t, h_t) = \frac{(c_t / h_t^\alpha)^{1-\rho}}{1-\rho}$$

**Parameters:**
- $\rho > 1$: Coefficient of relative risk aversion (CRRA). Code: `CRRA`.
- $\alpha \in (0, 1]$: Habit weight — exponent on habit stock. Code: `HabitWgt`.

When $\alpha = 0$, this reduces to standard CRRA utility $u(c) = c^{1-\rho}/(1-\rho)$.

The marginal utilities are:

$$u_c(c, h) = h^{-\alpha(1-\rho)} \cdot c^{-\rho}$$

$$u_h(c, h) = -\alpha \cdot c^{1-\rho} \cdot h^{-\alpha(1-\rho) - 1}$$

---

## 2. State Variables and Timing

All variables are normalized by the current level of permanent income $p_t$.

### Beginning-of-period states (pre-income)
- $k_t$: Normalized capital (carried from previous period via twist). Code: `kNrm`.
- $h_t^{\text{pre}}$: Normalized pre-growth habit stock. Code: `hPre`.
- $p_{t-1}$: Previous permanent income level. Code: `pLvlPrev`.

### Within-period transitions
1. **Income shocks drawn:** $(\psi_t, \theta_t) \sim F_t$ where $\psi$ is permanent, $\theta$ is transitory. Code: `IncShkDstn`.
2. **Growth:** $G_t = \Gamma_t \cdot \psi_t$ where $\Gamma_t$ is expected growth. Code: `PermGroFac`.
3. **Permanent income:** $p_t = p_{t-1} \cdot G_t$.
4. **Normalize to new permanent income:**
   - $b_t = k_t / G_t$ (bank balances; the portfolio return was already applied to get $k_t$)
   - $h_t = h_t^{\text{pre}} / G_t$ (habit stock in new normalization)
5. **Market resources:** $m_t = b_t + \theta_t$.

### Decision-time states
- $m_t$: Normalized market resources (cash-on-hand).
- $h_t$: Normalized habit stock.

### Controls
- $c_t$: Normalized consumption. Code: `cNrm`, policy function `cFunc(m, h)`.
- $s_t \in [0, 1]$: Share of savings allocated to the risky asset. Code: `Share`, policy function `ShareFunc(m, h)`.

### End-of-period transitions
1. **Savings:** $w_t = m_t - c_t$ (pre-return savings).
2. **Risky return drawn:** $\mathfrak{R}_{t+1} \sim G_t$ (risky asset return). Code: `RiskyDstn`.
3. **Portfolio return:** $R_{t+1}^{\text{port}} = R_f + s_t(\mathfrak{R}_{t+1} - R_f)$ where $R_f$ is the risk-free rate. Code: `Rfree`.
4. **Post-return assets:** $a_t = R_{t+1}^{\text{port}} \cdot w_t$. Code: `aNrm`.
5. **End-of-period habit stock:** $H_t = \lambda c_t + (1-\lambda) h_t$ where $\lambda \in (0, 1)$ is the habit updating rate. Code: `HabitRte`.
6. **Survival:** Agent survives with probability $\varsigma_t$. Code: `LivPrb`.

### Twist (inter-period mapping)
- $a_t \to k_{t+1}$ (post-return assets become next period's capital)
- $H_t \to h_{t+1}^{\text{pre}}$ (end-of-period habit becomes next period's pre-habit)
- $p_t \to p_t^{\text{prev}}$ for the next period

---

## 3. Bellman Equation

$$V_t(m_t, h_t) = \max_{c_t, s_t} \left\{ u(c_t, h_t) + \beta \varsigma_t \, \mathbb{E}_t \left[ G_{t+1}^{(1-\alpha)(1-\rho)} \, V_{t+1}(m_{t+1}, h_{t+1}) \right] \right\}$$

subject to:
$$m_t - c_t \geq 0, \quad s_t \in [0, 1]$$

The growth adjustment $G_{t+1}^{(1-\alpha)(1-\rho)}$ arises from the permanent income normalization when the utility function involves the habit stock with exponent $\alpha$.

---

## 4. First-Order Conditions

### Consumption FOC

From the envelope theorem and the habit transition $H = \lambda c + (1-\lambda) h$:

$$u_c(c_t, h_t) = \frac{\partial W}{\partial w_t} - \lambda \frac{\partial W}{\partial H_t}$$

where $W(w_t, H_t; s_t)$ is the end-of-period continuation value. Rearranging:

$$u_c = W_w - \lambda W_H$$

### Portfolio FOC

$$\frac{\partial W}{\partial s_t} = \beta \varsigma_t \, \mathbb{E}\left[(\mathfrak{R}_{t+1} - R_f) \cdot w_t \cdot \frac{\partial v_{t+1}}{\partial k_{t+1}}(R_t^{\text{port}} \cdot w_t, \, H_t)\right] = 0$$

This is the standard Euler equation for portfolio choice: the expected excess return weighted by marginal value equals zero.

### Marginal values at end of period

$$W_w(w, H; s) = \beta \varsigma \, \mathbb{E}_{\mathfrak{R}}\left[R^{\text{port}} \cdot \frac{\partial v_{t+1}}{\partial k}(R^{\text{port}} \cdot w, \, H)\right]$$

$$W_H(w, H; s) = \beta \varsigma \, \mathbb{E}_{\mathfrak{R}}\left[\frac{\partial v_{t+1}}{\partial h^{\text{pre}}}(R^{\text{port}} \cdot w, \, H)\right]$$

Here $\frac{\partial v}{\partial k}$ and $\frac{\partial v}{\partial h^{\text{pre}}}$ are the **beginning-of-period** marginal value functions stored in the solution. They already integrate over income shocks.

### Envelope conditions

By the envelope theorem:

$$\frac{\partial V}{\partial m} = W_w = u_c + \lambda W_H$$

$$\frac{\partial V}{\partial h} = u_h + (1-\lambda) W_H$$

---

## 5. Solution Algorithm

The solver `solve_one_period_HabitPortfolio` uses backward induction. Given the solution for period $t+1$ (functions $\frac{\partial v_{t+1}}{\partial k}$ and $\frac{\partial v_{t+1}}{\partial h^{\text{pre}}}$), it computes period $t$'s solution in three stages.

### Stage 1: Optimal Risky Share

**Grid:** $(w, H, s)$ — pre-return savings $\times$ habit stock $\times$ share.

For each $(w, H, s)$ triple, compute end-of-period marginal values by integrating over the risky return distribution $G$:

$$\text{dvdw}(w, H, s) = \beta\varsigma \sum_j \pi_j \cdot R_j^{\text{port}} \cdot \frac{\partial v_{t+1}}{\partial k}(R_j^{\text{port}} \cdot w, \, H)$$

$$\text{dvds}(w, H, s) = \beta\varsigma \sum_j \pi_j \cdot (\mathfrak{R}_j - R_f) \cdot w \cdot \frac{\partial v_{t+1}}{\partial k}(R_j^{\text{port}} \cdot w, \, H)$$

$$\text{dvdH}(w, H, s) = \beta\varsigma \sum_j \pi_j \cdot \frac{\partial v_{t+1}}{\partial h^{\text{pre}}}(R_j^{\text{port}} \cdot w, \, H)$$

For each $(w, H)$, find $s^*$ where $\text{dvds} = 0$ by searching for sign changes across the $s$ grid and linearly interpolating. Handle corner solutions ($s^* = 0$ or $s^* = 1$).

Record $\text{dvdw}^*(w, H)$, $\text{dvdH}^*(w, H)$, and $s^*(w, H)$ at the optimal share.

**HARK tool:** `expected()` from `HARK.distributions` integrates over the discrete risky return distribution.

### Stage 2: Optimal Consumption via Habit EGM

With the optimized end-of-period values, apply the **endogenous grid method** with the habit FOC-inverter.

1. Compute transformed marginal value:
$$\chi(w, H) = \left(\text{dvdw}^* - \lambda \cdot \text{dvdH}^*\right)^{-1/\rho}$$

2. Invert the FOC using the `HabitFormationInverter`:
$$(\hat{c}, \hat{h}) = Q(H, \chi)$$
This recovers consumption $\hat{c}$ and decision-time habit $\hat{h}$ from the end-of-period habit $H$ and transformed marginal value $\chi$.

3. Compute endogenous market resources:
$$\hat{m} = w + \hat{c}$$

4. Build policy functions on the endogenous $(\hat{m}, \hat{h})$ grid:
   - `cFunc(m, h)` as a `Curvilinear2DInterp` of $\hat{c}$ on $(\hat{m}, \hat{h})$.
   - `ShareFunc(m, h)` as a `Curvilinear2DInterp` of $s^*$ on the same grid.

5. Add borrowing constraint: `cFunc = LowerEnvelope2D(cFuncUnc, cFuncCnst)`.

**HARK tools:**
- `HabitFormationInverter` (from `ConsHabitModel.py`): Pre-computed lookup table that inverts the nonlinear FOC mapping $(H, \chi) \to (c, h)$.
- `Curvilinear2DInterp`: Interpolation on a warped (endogenous) 2D grid.
- `LowerEnvelope2D`: Takes the lower envelope of two 2D functions (imposing the borrowing constraint).

### Stage 3: Marginal Value Functions

Compute beginning-of-period marginal values by integrating over income shocks:

$$\frac{\partial v_t}{\partial k}(k, h^{\text{pre}}) = \mathbb{E}_F\left[G^{(1-\rho)(1-\alpha) - 1} \cdot \frac{\partial V_t}{\partial m}\right]$$

$$\frac{\partial v_t}{\partial h^{\text{pre}}}(k, h^{\text{pre}}) = \mathbb{E}_F\left[G^{(1-\rho)(1-\alpha) - 1} \cdot \frac{\partial V_t}{\partial h}\right]$$

where for each income shock realization:
- $m = k/G + \theta$, $h = h^{\text{pre}}/G$
- $c = \text{cFunc}(m, h)$, $w = m - c$, $H = \lambda c + (1-\lambda)h$
- $\frac{\partial V}{\partial m} = u_c(c, h) + \lambda \cdot \text{dvdH\_cont}(w, H)$
- $\frac{\partial V}{\partial h} = u_h(c, h) + (1-\lambda) \cdot \text{dvdH\_cont}(w, H)$

Here `dvdH_cont(w, H)` is a `BilinearInterp` built from the Stage 1 results, giving the discounted expected continuation value of the habit stock at optimal share.

Store as inverse-transformed `BilinearInterp` wrapped in `MargValueFuncCRRA` / `ValueFuncCRRA`.

**HARK tools:**
- `expected()`: Integrates over the discrete income shock distribution.
- `BilinearInterp`: 2D interpolation on a regular grid.
- `MargValueFuncCRRA`: Wrapper that stores the inverse of the marginal value function for numerical stability, applying $f(x)^{-\rho}$ when evaluated.
- `ValueFuncCRRA`: Similar wrapper applying $f(x)^{1-\rho}/(1-\rho)$.

---

## 6. Terminal Period

The pseudo-terminal solution has:
- $\frac{\partial v}{\partial k} = 0$ and $\frac{\partial v}{\partial h^{\text{pre}}} = 0$ (stored as `ConstantFunction(0.0)`)
- $k_{\min} = 0$

When the solver detects the terminal solution (checking `isinstance(dvdkFunc_next, ConstantFunction)`), it sets:
- `cFunc = IdentityFunction` (consume everything: $c = m$)
- `ShareFunc = ConstantFunction(ShareLimit)` (Merton-Samuelson limiting share)

Then Stage 3 computes proper marginal values from this consume-everything policy using `dvdH_cont = ConstantFunction(0.0)`.

---

## 7. Simulation (YAML Dynamics)

The simulation uses HARK's YAML-based simulator defined in `ConsHabitPortfolio.yaml`. The key difference from the habit-only model is:
- `bNrm = kNrm / G` (no $R_f$ factor, because the portfolio return is already embedded in `kNrm` via the twist)
- `Share = ShareFunc@(mNrm, hNrm)` is a 2D policy function
- `Rport = Rfree + (Risky - Rfree) * Share` computes the realized portfolio return
- `aNrm = Rport * wNrm` applies the portfolio return to savings

The twist maps `aNrm → kNrm`, `HNrm → hPre`, `pLvl → pLvlPrev` for the next period.

---

## 8. Default Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `CRRA` | $\rho$ | 2.0 | Coefficient of relative risk aversion |
| `HabitWgt` | $\alpha$ | 0.5 | Habit weight exponent |
| `HabitRte` | $\lambda$ | 0.2 | Habit updating rate |
| `Rfree` | $R_f$ | 1.03 | Risk-free gross return |
| `RiskyAvg` | $\mathbb{E}[\mathfrak{R}]$ | 1.08 | Mean risky gross return |
| `RiskyStd` | $\sigma_\mathfrak{R}$ | 0.184 | Std dev of log risky return |
| `DiscFac` | $\beta$ | 0.96 | Discount factor |
| `LivPrb` | $\varsigma$ | 0.98 | Survival probability |
| `PermGroFac` | $\Gamma$ | 1.01 | Expected permanent income growth |
| `BoroCnstArt` | $\underline{a}$ | 0.0 | Borrowing constraint |

---

## 9. Class Hierarchy

```
AgentType (HARK.core)
  └── HabitPortfolioConsumerType (ConsHabitPortfolioModel.py)
        - default_:
            solver: solve_one_period_HabitPortfolio
            model: ConsHabitPortfolio.yaml
            track_vars: [aNrm, cNrm, mNrm, hNrm, Share, pLvl]
        - time_inv_: [DiscFac, CRRA, BoroCnstArt, aXtraGrid, HabitGrid,
                       ShareGrid, FOCinverter, HabitWgt, HabitRte, RiskyDstn]
        - time_vary_: [IncShkDstn, Rfree, PermGroFac, LivPrb, ShareLimit]
        - distributions: [IncShkDstn, PermShkDstn, TranShkDstn, RiskyDstn,
                          kNrmInitDstn, pLvlInitDstn, HabitInitDstn]
```

The class inherits from `AgentType` (not from `PortfolioConsumerType`) because the habit model uses dict-based solutions and the YAML simulator rather than the legacy `ConsumerSolution` / Monte Carlo path.

---

## 10. Solution Object

Each period's solution is a Python dict:

```python
{
    "cFunc":     Curvilinear2DInterp,   # c(m, h)
    "ShareFunc": Curvilinear2DInterp,   # s(m, h)
    "dvdkFunc":  MargValueFuncCRRA,     # ∂v/∂k(k, hPre)
    "dvdhFunc":  ValueFuncCRRA,         # ∂v/∂hPre(k, hPre)
    "kNrmMin":   float,                 # minimum capital
}
```

Note: `cFunc` and `ShareFunc` are defined on decision-time states $(m, h)$, while `dvdkFunc` and `dvdhFunc` are defined on beginning-of-period states $(k, h^{\text{pre}})$.

---

## 11. Key Economic Predictions

1. **State-dependent portfolio allocation:** The optimal risky share depends on both $m$ and $h$. Agents with high habit stock (relative to wealth) hold more conservative portfolios.

2. **"Gambling for resurrection":** After a negative wealth shock, agents whose habit stock remains high relative to their reduced wealth increase their risky share to try to rebuild consumption capacity.

3. **Consumption smoothing through habits:** With habits, consumption adjusts more slowly to wealth shocks than in the standard model, because deviating from the habit level is costly.
