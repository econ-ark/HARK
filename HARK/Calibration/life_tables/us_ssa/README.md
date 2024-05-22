# United States Social Security Administration Life-Tables

This folder contains tools for producing sequences of life-cycle survival probabilities for microeconomic
models from official U.S. mortality estimates.

## Sources

The life tables contained in this folder correspond to the downloadable
''Period Life Tables'' from the 2020 Annual Trustees Report of the SSA.

They were downloaded from [this link](https://www.ssa.gov/oact/HistEst/PerLifeTables/2020/PerLifeTables2020.html)
on January 7, 2021.

## Format

There are four `.csv` files:

- `PerLifeTables_F_Hist_TR2020.csv` contains historical (1900-2017) information for females.
- `PerLifeTables_F_Alt2_TR2020.csv` contains projected (2018-2095) information for females.
- `PerLifeTables_M_Hist_TR2020.csv` contains historical (1900-2017) information for males.
- `PerLifeTables_M_Alt2_TR2020.csv` contains projected (2018-2095) information for males.

All the tables have the same format. There are three columns of interest for our purposes:

- `Year`.
- `x`: age.
- `q(x)`: probability of death.

As an example, the probability that a male who was 27 years old in 1990 would die within a year (from 1990 to 1991) is found in file `PerLifeTables_M_Hist_TR2020.csv`, in column `q(x)` and the row in which `Year == 1990` and `x == 27`.

Visit the [SSA's site](https://www.ssa.gov/oact/HistEst/PerLifeTables/2020/PerLifeTables2020.html) for a complete description of the tables.

## Usage

`SSATools.py` contains functions that translate the information in the `.csv`
life-tables into sequences of survival probabilities in the format that HARK's
life-cycle models require.

The main function is `parse_ssa_life_table`, which produces survival
probabilities for a given sex, age-range, and year of birth. See the function's
documentation for details. `examples/Calibration/US_SSA_life_tables.py` contains
examples of its use.
