---
title: Ranked Annotated Bibliography on Population Simulation After Solving Bellman Problems
author: OpenAI ChatGPT
date: 2026-03-15
bibliography:
  - bibliography.bib
---

# Ranked annotated bibliography

## Scope

This bibliography surveys the main resources on **population simulation after solving a Bellman problem**, with emphasis on sources that contain either:

1. **explicit comparisons of simulation/aggregation methods**;
2. **clear mathematics for distribution evolution**; or
3. **runnable computational examples**.

The focus is on the heterogeneous-agent literature, because that is where the comparison between
Monte Carlo simulation, transition-matrix / non-stochastic propagation, parameterized distributions,
explicit aggregation, perturbation methods, and HJB--Kolmogorov-forward approaches is most developed.

A theme that emerges quickly is that there is **no single canonical textbook** whose main mission is:
solve one Bellman problem and then compare *all* major population simulators in a unified way.
The best resources are therefore a mix of survey chapters, benchmark-comparison papers, mathematical papers,
and code repositories.

## Ranking criteria

The ranking below weights four things:

1. **Directness**: how explicitly the source compares post-solution simulators.
2. **Mathematical clarity**: how clearly it formulates laws of motion for the distribution.
3. **Computational concreteness**: whether it contains worked numerical examples or code.
4. **Usefulness as an entry point**: whether it helps organize the surrounding literature.

## Method map

| Key | Source | Rank | Monte Carlo | Transition matrix / non-stochastic distribution propagation | Parameterized distribution / moments | Explicit aggregation | Perturbation / linearization | HJB--KF / PDE | Public code / worked example | Finite horizon relevance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| {cite}`algan2014_handbook` | Algan et al. (2014) | 1 | Yes | Yes | Yes | Yes | Yes | Partial | Survey, benchmark references | Mostly infinite horizon |
| {cite}`denhaan2010_comparison` | den Haan (2010a) | 2 | Yes | Indirectly | Yes | Yes | Yes | No | Benchmark comparison | Infinite horizon |
| {cite}`denhaan2010_simulation_slides` | den Haan slides | 3 | Yes | Yes | Yes | Partial | Partial | No | Pedagogical formulas | Mostly infinite horizon |
| {cite}`algan2008_parameterized_density` | Algan, Allais, den Haan (2008) | 4 | Minor role | Yes | Yes | No | No | No | Detailed algorithm paper | Infinite horizon |
| {cite}`young2010_nonstochastic` | Young (2010) | 5 | Compared | Yes | KS moments | No | No | No | Clean benchmark note | Infinite horizon |
| {cite}`denhaan2010_explicit_aggregation` | den Haan and Rendahl (2010) | 6 | Avoided | Avoided | No | Yes | No | No | Clean benchmark note | Infinite horizon |
| {cite}`reiter2009_projection` | Reiter (2009) | 7 | Avoided | State vector for distribution | No | No | Yes | No | Full method paper | Infinite horizon / local dynamics |
| {cite}`winberry2018_method` | Winberry (2018) | 8 | No | No | Yes | No | Partial | No | Dynare implementation | Infinite horizon |
| {cite}`achdou2022_continuoustime` | Achdou et al. (2022) | 9 | No | Yes | No | No | No | Yes | Mathematical + code ecosystem | Infinite horizon and transitions |
| {cite}`hark2026_transition_matrix` | HARK transition-matrix notebook | 10 | Yes | Yes | No | No | SSJ adjacent | No | Yes | Both, via notebook examples |
| {cite}`quantecon_aiyagari` | QuantEcon Aiyagari lecture | 11 | No | Yes | No | No | No | No | Yes | Stationary infinite horizon |
| {cite}`hark2026_lifecycle` | HARK life-cycle notebook | 12 | Yes | Some | No | No | No | No | Yes | **Directly finite horizon** |

## Tier I. Essential starting points

### 1. Algan, Allais, den Haan, and Rendahl (2014), *Solving and Simulating Models with Heterogeneous Agents and Aggregate Uncertainty* {cite}`algan2014_handbook`

**Why it matters.** This is the closest thing to the survey you asked for. It explicitly says it reviews
different algorithms to **solve and simulate** heterogeneous-agent models with aggregate uncertainty, and it also discusses
accuracy tests. It is the best single map of the terrain.

**What it covers.** The chapter organizes the field around several competing strategies:
Krusell--Smith style methods, parameterized cross-sectional densities, explicit aggregation, perturbation / Reiter-style methods,
and accuracy diagnostics. It is also unusually good at separating questions about the **individual policy problem**
from questions about the **distribution simulator**.

**Mathematical/computational value.** High on both dimensions. The chapter is not a line-by-line coding manual,
but it is the best reference for understanding which algorithms are genuinely alternatives to each other.

**Limitation.** Most of the benchmark material is infinite-horizon and centered on the canonical incomplete-markets-with-aggregate-risk environment.

### 2. den Haan (2010a), *Comparison of Solutions to the Incomplete Markets Model with Aggregate Uncertainty* {cite}`denhaan2010_comparison`

**Why it matters.** This is the benchmark comparison paper in the JEDC computational-suite project.
It compares alternative algorithms on the same model and reports differences in accuracy and speed.

**What it covers.** Although it is broader than “simulation methods only,” it is the best source for seeing how the competing
approaches behave when held up against the same target problem. It is especially useful because the benchmark spawned several short
method notes in the same issue.

**Mathematical/computational value.** Very high computational value; moderate mathematical exposition.
Best read together with the suite introduction and the individual method papers in the same issue.

**Limitation.** It is still a benchmark paper, not a general textbook treatment.

### 3. den Haan, *Simulating Models with Heterogeneous Agents* (slides) {cite}`denhaan2010_simulation_slides`

**Why it matters.** These slides are probably the clearest side-by-side pedagogical presentation of the main simulation families:
Monte Carlo simulation, non-random cross-section methods, grid methods, and parameterized CDF / density approaches.

**What it covers.** The slides explicitly compare:
random simulation with large populations,
grid methods that propagate cross-sectional mass deterministically,
and parameterized representations of the distribution.
For someone trying to understand “what are the competing simulators after the Bellman problem is solved?”, these slides are unusually direct.

**Mathematical/computational value.** Strong computational intuition with enough formulas to be implementable.
In some ways this is the best quick-start resource.

**Limitation.** It is lecture material, not a polished archival paper.

## Tier II. Core comparison and algorithm papers

### 4. Algan, Allais, and den Haan (2008), *Solving heterogeneous-agent models with parameterized cross-sectional distributions* {cite}`algan2008_parameterized_density`

**Why it matters.** This is one of the foundational alternatives to brute-force Monte Carlo simulation.
It develops a method in which projection methods do most of the work and simulation plays only a minor role.

**What it covers.** The paper uses a parameterized representation of the cross-sectional distribution and also develops a simulation procedure
that avoids cross-sectional sampling variation. It is especially relevant if you are interested in “functional approximation to distributions.”

**Mathematical/computational value.** High. This is one of the best method papers if you want both an algorithm and discussion of accuracy tests.

**Limitation.** It is more method-development than broad comparison.

### 5. Young (2010), *Solving the incomplete markets model with aggregate uncertainty using the Krusell--Smith algorithm and non-stochastic simulations* {cite}`young2010_nonstochastic`

**Why it matters.** This is one of the cleanest controlled comparisons inside the KS tradition.
It takes the familiar KS structure and swaps in a non-stochastic simulation routine.

**What it covers.** The paper is short but strategically important: it isolates the effect of replacing stochastic panel simulation
with non-stochastic propagation while keeping the overall algorithmic framework familiar.

**Mathematical/computational value.** High computational value for modest reading cost.
Excellent as a bridge paper between KS and deterministic distribution propagation.

**Limitation.** Narrow scope by design.

### 6. den Haan and Rendahl (2010), *Solving the incomplete markets model with aggregate uncertainty using explicit aggregation* {cite}`denhaan2010_explicit_aggregation`

**Why it matters.** This paper is useful because it removes simulation from the critical step altogether:
aggregate laws of motion are obtained directly from the individual policy rule.

**What it covers.** It proposes explicit aggregation as an alternative to parameterizing the distribution or simulating a large population.
That makes it especially relevant if your interest is not merely how to simulate the population, but whether some simulation step can be bypassed.

**Mathematical/computational value.** Conceptually elegant and computationally important.
Good for understanding how far one can get without tracking the full distribution.

**Limitation.** Best for aggregate dynamics, not always for every distributional statistic one might want.

### 7. Reiter (2009), *Solving heterogeneous-agent models by projection and perturbation* {cite}`reiter2009_projection`

**Why it matters.** Reiter is the classic paper for representing the cross-sectional distribution inside a perturbation / linearization framework.
It is one of the central alternatives to KS-style simulation methods.

**What it covers.** The idea is to solve for the stationary heterogeneous-agent economy first and then perturb around it for aggregate shocks.
This turns the distribution into a large but finite-dimensional object in the state vector.

**Mathematical/computational value.** Very high mathematical content; also computationally influential.
Essential if you want to understand the lineage that later leads to Bayer--Luetticke and sequence-space methods.

**Limitation.** It is not a “simulator comparison” paper in the narrow sense; it is a different overall strategy.

### 8. Winberry (2018), *A method for solving and estimating heterogeneous agent macro models* {cite}`winberry2018_method`

**Why it matters.** Winberry gives a practically important parametric-family approach to the distribution
and ties it to estimation and Dynare implementation.

**What it covers.** The infinite-dimensional distribution is approximated by a flexible finite-dimensional parametric family.
For researchers who want a tractable workflow rather than only a conceptual map, this is one of the most useful modern method papers.

**Mathematical/computational value.** High computational value; mathematically clean enough to be transparent.
Strong for users who care about estimation as well as simulation.

**Limitation.** It is more a practical parametric-distribution framework than a broad comparison resource.

### 9. Harmenberg (2021), *Aggregating heterogeneous-agent models with permanent income shocks* {cite}`harmenberg2021_permanent_income`

**Why it matters.** This is a sharp example of how changing the measure under which one simulates can simplify distribution tracking.

**What it covers.** The paper introduces a permanent-income-neutral measure under which one need not explicitly track the permanent-income distribution in the usual way.
It is a specialized but conceptually valuable addition to the aggregation/simulation toolbox.

**Mathematical/computational value.** High conceptual payoff for readers thinking hard about what exactly must be simulated.

**Limitation.** Specialized to settings with permanent-income shocks; not the place to start.


### 9a. den Haan (2010b), *Assessing the Accuracy of the Aggregate Law of Motion in Models with Heterogeneous Agents* {cite}`denhaan2010_accuracy`

**Why it matters.** Many papers report a high $R^2$ for the estimated aggregate law of motion and then move on.
This paper is the antidote to that habit.

**What it covers.** It argues that standard diagnostics such as the $R^2$ and regression standard error can be seriously misleading as accuracy tests,
and it develops more informative ways to assess the quality of an approximate aggregate law of motion.

**Mathematical/computational value.** High practical value for anyone comparing simulators or aggregation schemes:
a simulator is only as good as the diagnostics used to evaluate it.

**Limitation.** This is an accuracy-assessment paper, not a broad simulation-method tutorial.

### 9b. Companion benchmark notes in the JEDC computational suite {cite}`denhaan2010_suite` {cite}`maliar2010_krusellsmith` {cite}`algan2010_parameterized_note`

**Why they matter.** The 2010 JEDC special issue is more useful when read as a package than as isolated articles.
The suite introduction defines the common benchmark problem, while the short companion notes show how different methods behave on exactly the same target economy.

**What they cover.** The key benchmark notes are:
Maliar, Maliar, and Valli on a KS implementation;
Algan, Allais, and den Haan on parameterized cross-sectional distributions;
and the suite introduction by den Haan, Judd, and Juillard.
Together with {cite}`young2010_nonstochastic` and {cite}`denhaan2010_explicit_aggregation`,
they form the most concrete public comparison set in the literature.

**Mathematical/computational value.** High computational value because they reduce “method comparison” to a common benchmark.
They are especially useful when you want to see what changes when the model is fixed and the computational representation of the distribution changes.

**Limitation.** The notes are short and assume familiarity with the benchmark environment.

## Tier III. Mathematical foundations and simulation-accuracy theory

### 10. Achdou, Han, Lasry, Lions, and Moll (2022), *Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach* {cite}`achdou2022_continuoustime`

**Why it matters.** This is the best mathematical resource for distribution propagation in the modern literature.

**What it covers.** The paper recasts heterogeneous-agent models in continuous time as a coupled backward--forward system:
an HJB equation for individual optimization and a Kolmogorov Forward / Fokker--Planck equation for the distribution.
It also treats both stationary equilibria and transition dynamics.

**Mathematical/computational value.** Extremely high. If your goal is to understand the simulator as an operator acting on distributions,
this is probably the clearest modern reference.

**Limitation.** Continuous time is not the same as the classic discrete-time KS environment, so direct one-for-one comparisons require some translation.

### 11. Achdou et al. (2020), *Online Appendix: Numerical Methods for “Income and Wealth Distribution in Macroeconomics: A Continuous-Time Approach”* {cite}`achdou2020_numerical_appendix`

**Why it matters.** The appendix is often more useful than the main paper if you intend to implement anything.

**What it covers.** It writes out the HJB and KF equations explicitly, shows the finite-difference discretization,
explains upwinding and stability, and makes clear how the KF equation is solved once the HJB side is discretized.
It is one of the best public sources for concrete numerical details.

**Mathematical/computational value.** Exceptionally high for implementation.
A modeler can move from this appendix to working code with relatively little guesswork.

**Limitation.** Continuous-time focus; not a survey.

### 12. Santos and Peralta-Alva (2005), *Accuracy of Simulations for Stochastic Dynamic Models* {cite}`santos2005_accuracy`

**Why it matters.** This is the theoretical paper to read when you want guarantees connecting numerical approximation and simulation output.

**What it covers.** The paper studies convergence of simulated moments generated by approximate solutions and provides error bounds under contraction-type conditions.
It is a theory paper about the reliability of simulation-based quantitative conclusions.

**Mathematical/computational value.** High mathematical value; lower direct coding value.
It is important because many applied papers simulate first and worry about justification later.

**Limitation.** General simulation theory, not tailored only to heterogeneous-agent distribution simulators.

### 13. Peralta-Alva and Santos (2010), *Problems in the Numerical Simulation of Models with Heterogeneous Agents and Economic Distortions* {cite}`peraltaalva2010_problems`

**Why it matters.** This paper is valuable precisely because it is skeptical.
It emphasizes that numerical simulation in heterogeneous-agent environments can be more fragile than standard reporting practices suggest.

**What it covers.** It discusses difficulties that arise when distortions and heterogeneity interact, and why naive diagnostics can mislead.

**Mathematical/computational value.** Good companion to the more optimistic method papers.
Useful when assessing whether a simulator is merely fast or genuinely trustworthy.

**Limitation.** More cautionary and conceptual than tutorial.

## Tier IV. Code, notebooks, and computational examples

### 14. HARK, *Transition Matrix Example* {cite}`hark2026_transition_matrix`

**Why it matters.** Public code that directly juxtaposes Monte Carlo and transition-matrix methods is rare; this notebook does exactly that.

**What it covers.** The notebook compares Monte Carlo simulation against transition-matrix propagation for aggregate consumption and assets,
and it also links naturally to sequence-space calculations.

**Mathematical/computational value.** Excellent computational value.
This is one of the few places where a reader can inspect and modify code rather than just reading prose about the alternatives.

**Limitation.** Documentation-level example rather than a formal paper.

### 15. QuantEcon, *The Aiyagari Model* {cite}`quantecon_aiyagari`

**Why it matters.** This is one of the cleanest public expositions of the “policy-induced Markov chain” view.

**What it covers.** After solving the household problem, the lecture constructs the transition operator implied by the policy rule and computes the stationary distribution.
That makes it an excellent entry point for the transition-matrix / deterministic-distribution perspective.

**Mathematical/computational value.** High pedagogical value and runnable code.
A very good place to see the operator viewpoint in discrete time.

**Limitation.** It is not a comparison document; it mainly teaches one approach very well.

### 16. Benjamin Moll, *Codes* {cite}`moll2026_codes`

**Why it matters.** This is the main public code hub for the continuous-time HJB--KF approach.

**What it covers.** The page collects codes for stationary equilibria, transition dynamics, diffusion versions,
and related numerical experiments in continuous-time heterogeneous-agent models.

**Mathematical/computational value.** Very high computational value.
Particularly useful if you want examples beyond the canonical one-asset stationary case.

**Limitation.** More repository hub than synthesized survey.

### 17. HARK, *A Life Cycle Model: The Distribution of Assets By Age* {cite}`hark2026_lifecycle`

**Why it matters.** This is one of the most useful public examples for the **finite-horizon** side of your question.

**What it covers.** The notebook simulates a life-cycle consumption-saving model and studies cross-sectional asset distributions by age.
It is not a formal comparison of simulator families, but it is important because the comparison literature is much thinner for finite-horizon models than for stationary infinite-horizon models.

**Mathematical/computational value.** High practical value.
A good place to see how finite-horizon distribution simulation is actually handled in public code.

**Limitation.** Does not itself benchmark multiple simulators against one another.

### 18. Luetticke (2023), *Heterogeneous Agent Macroeconomics: Methods and Applications* {cite}`luetticke2023_methods`

**Why it matters.** These slides give a modern methods-oriented map from KS and Reiter to Bayer--Luetticke, MIT-shock methods, and sequence-space approaches.

**What it covers.** The slides are especially useful for seeing how the older simulator-comparison literature connects to modern solution methods and code repositories.
They also point to publicly available Matlab, Julia, and Python implementations.

**Mathematical/computational value.** Strong as a roadmap and literature guide.
Especially useful after reading Reiter and before diving into modern HANK code.

**Limitation.** Broad methods course, not narrowly focused on post-solution simulators.

## Tier V. Background books and broader context

### 19. Judd (1998), *Numerical Methods in Economics* {cite}`judd1998_numerical`

**Why it matters.** Judd is still the standard general reference for numerical methods in economics.

**What it covers.** It does not focus on population simulators in Bellman/heterogeneous-agent models,
but it provides the general toolbox for approximation, integration, interpolation, and numerical diagnostics.

**Mathematical/computational value.** Very high as background.
Still worth having open on the desk while reading the more specialized papers.

**Limitation.** Too broad to answer your question directly.

### 20. Adda and Cooper (2003), *Dynamic Economics* {cite}`adda2003_dynamic`

**Why it matters.** This is a strong background text on dynamic programming, numerical methods, and simulation-based quantitative work.

**What it covers.** It bridges theory and empirical quantitative methods and is useful for readers who want a unified language for dynamic programming and numerical implementation.

**Mathematical/computational value.** Good general background; not the main source for cross-sectional distribution simulators.

**Limitation.** Not specialized to heterogeneous-agent distribution propagation.

### 21. Heer and Maussner (2009), *Dynamic General Equilibrium Modelling* {cite}`heer2009_dge`

**Why it matters.** Among textbooks, this is one of the more relevant ones because it explicitly treats heterogeneous-agent economies with endogenous distributions.

**What it covers.** It is useful as textbook scaffolding around the specialized papers, especially for readers who want a longer-form book treatment.

**Mathematical/computational value.** Good background, especially for discrete-time macro computation.

**Limitation.** Still not a canonical “compare all simulators” source.

## Adjacent but increasingly important modern resources

### 22. Auclert, Bardoczy, Rognlie, and Straub (2021), *Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models* {cite}`auclert2021_ssj`

**Why it matters.** This is not mainly a paper about cross-sectional simulator comparison, but it is now central to the computational practice of heterogeneous-agent macro.

**What it covers.** It shows how to compute Jacobians and transition dynamics efficiently, using the solved micro problem and structured linear responses of the distribution.

**Mathematical/computational value.** Very high. It is especially relevant if your eventual interest is not only simulation of a population in isolation, but general-equilibrium transitions and estimation.

**Limitation.** Best thought of as the modern continuation of the literature, not as the canonical answer to the narrower simulator-comparison question.

### 23. Bayer and Luetticke (2020), *Solving discrete time heterogeneous agent models with aggregate risk and many idiosyncratic states by perturbation* {cite}`bayer2020_perturbation`

**Why it matters.** This is one of the major modern descendants of the Reiter tradition.

**What it covers.** It extends perturbation-based methods to richer discrete-time heterogeneous-agent environments with many idiosyncratic states.

**Mathematical/computational value.** High. Important if your reading path moves from benchmark comparison to current research-grade methods.

**Limitation.** Again, this is more an advanced solution strategy than a narrow study of simulator properties.

## Suggested reading paths

## Reading path A: direct answer to the original question

Read these in order:

1. {cite}`algan2014_handbook`
2. {cite}`denhaan2010_simulation_slides`
3. {cite}`denhaan2010_comparison`
4. {cite}`young2010_nonstochastic`
5. {cite}`denhaan2010_explicit_aggregation`
6. {cite}`algan2008_parameterized_density`

This path gets you closest to a comparative survey of the available simulators after the Bellman problem is solved.

## Reading path B: strongest mathematical route

Read these in order:

1. {cite}`achdou2022_continuoustime`
2. {cite}`achdou2020_numerical_appendix`
3. {cite}`santos2005_accuracy`
4. {cite}`peraltaalva2010_problems`

This path is best if your real interest is to understand the simulator as a mathematically defined operator on distributions.

## Reading path C: quickest route to code you can run

Read or run these in order:

1. {cite}`quantecon_aiyagari`
2. {cite}`hark2026_transition_matrix`
3. {cite}`hark2026_lifecycle`
4. {cite}`moll2026_codes`
5. {cite}`luetticke2023_methods`

This path is best if you want concrete examples rather than literature first.

## Reading path D: finite-horizon emphasis

There is much less explicit comparison literature for finite-horizon models than for stationary infinite-horizon models.
For that reason, the most useful sources are:

1. {cite}`hark2026_lifecycle`
2. {cite}`hark2026_transition_matrix`
3. {cite}`adda2003_dynamic`
4. then back to the general comparative sources {cite}`algan2014_handbook` and {cite}`denhaan2010_simulation_slides`

## Bottom line

The best **single** source is {cite}`algan2014_handbook`.
The best **benchmark-comparison** source is {cite}`denhaan2010_comparison`.
The best **pedagogical method-comparison** source is {cite}`denhaan2010_simulation_slides`.
The best **mathematical** source is {cite}`achdou2022_continuoustime` together with {cite}`achdou2020_numerical_appendix`.
The best **public code examples** are {cite}`hark2026_transition_matrix`, {cite}`quantecon_aiyagari`, and {cite}`moll2026_codes`.
For **finite-horizon** work, public examples exist, but explicit simulator-comparison studies are much scarcer; {cite}`hark2026_lifecycle` is a good place to start.

## References

```{bibliography}
:style: plain
```
