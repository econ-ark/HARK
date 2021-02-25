# Tools for calibrating income profiles

This folder contains various tools that are used for replicating calibrations/estimates of income processes from the literature.

## `IncomeTools.py`

This is the only file for now. Its main components are:

- `sabelhaus_song_var_profile`: a function for producing life-cycle profiles of the volatilities of transitory and permanent shocks to income using the results from [Sabelhaus & Song (2010)](https://www.sciencedirect.com/science/article/abs/pii/S0304393210000358). The estimates used to reproduce the results from the paper were generously provided by [John Sabelhaus](https://www.sites.google.com/view/johnsabelhaus).

- `Cagetti_income`: a representation of the income specification used by [Cagetti (2003)](https://www.jstor.org/stable/1392584?seq=1). The estimates used to reproduce these results were generously shared by [Marco Cagetti](https://www.federalreserve.gov/econres/marco-cagetti.htm).

- `CGM_income`: a representation of the income specification used by [Cocco, Gomes, and Maenhout (2005)](https://academic.oup.com/rfs/article-abstract/18/2/491/1599892). The estimates used to reproduce these results come directly from the published version of the paper.

- `ParseIncomeSpec`: a function that takes in calibrations in various formats that are common in the literature and produces the parameters that represent them in the format that HARK expects.
