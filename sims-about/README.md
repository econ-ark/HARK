# Monte Carlo vs Transition Matrix Methods — Notebook Progression

This directory contains a series of pedagogical Jupyter notebooks that
progressively build intuition for transition matrix (TM) simulation methods
as an alternative to Monte Carlo (MC) in heterogeneous-agent models solved
by HARK.

## Notebook sequence

Read in this order.  Each notebook builds on the previous.

| # | Notebook | Model | Key idea |
|---|----------|-------|----------|
| 1 | `01-markov-tm-prototype.ipynb` | 2-state `MarkovConsumerType`, PermGroFac=1 | Hand-built 1D TM, MC vs TM comparison, ergodic distribution |
| 2 | `02-serial-unemployment-tm.ipynb` | 4-state serial unemployment, PermGroFac=1 | Scale to more Markov states, same 1D grid technique |
| 3 | `03-serial-growth-tm-2d.ipynb` | 5-state serial growth, PermGroFac!=1 | 2D (m,p) grid required; reveals p-truncation problem |
| 4 | `04-serial-growth-tm-harmenberg.ipynb` | 5-state serial growth, Harmenberg measure | Neutral measure collapses back to 1D; fixes level aggregates |
| 5 | `05-tm-consolidation.ipynb` | Single-state, neutral measure | Validates hand-built TM against HARK's `NewKeynesianConsumerType` |
| 6 | `06-agg-shock-markov-tm.ipynb` | 2-state Krusell-Smith economy | TM with endogenous aggregate state, 2D cFunc |
| 7 | `07-validate-markov-tm-methods.ipynb` | 2-state symmetric Markov | Validates `MarkovConsumerType` production TM methods |
| 8 | `08-tm-in-ks.ipynb` | 2-state Krusell-Smith economy | TM forward propagation via `make_history_tm()` |
| 9 | `09-markov-ssj.ipynb` | 2-state symmetric Markov | Sequence-space Jacobians via `calc_jacobian()` |

## Supporting documents

| File | Contents |
|------|----------|
| `mathematical-framework.ipynb` | Mathematical framework: perch notation, MC vs TM theory, Markov/Harmenberg/KS/SSJ extensions |
| `LESSONS-LEARNED.md` | Bugs, gotchas, process lessons, and proposed HARK source improvements |
| `bibliography.md` | Annotated bibliography of simulation methods literature |
| `bibliography.bib` | BibTeX references |
| `pdfs/` | Downloaded PDFs of referenced papers |

## Archive

The `_archive/` subdirectory contains scaffolding documents (project plans,
AI context prompts, reference analyses) that were used during development but
are no longer needed for day-to-day work.  They are preserved for historical
reference.
