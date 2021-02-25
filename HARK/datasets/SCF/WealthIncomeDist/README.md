# Summary statistics of wealth and permanent income in the United States

The summary statistics in `WealthIncomeStats.csv` are computed using the
Survey of Consumer Finances. The file can be replicated directly from the
unprocessed SCF summary files using the repository [SCF-IncWealthDist](https://github.com/Mv77/SCF-IncWealthDist), created by [Mateo Velasquez-Giraldo](https://mv77.github.io/).

**NOTE:** wealth and permanent income levels in `WealthIncomeStats.csv` are expressed in thousands of dollars before taking logarithms.

# `WealthIncomeStats.csv`

Both the file and this description come from the repository
[SCF-IncWealthDist](https://github.com/Mv77/SCF-IncWealthDist).

This is a table with summary statistics from the SCF. The file has the following
columns, described by groups:
- Demographic and sample-defining variables: these variables describe the sample on which the summary statistics were computed on. Their values should be read as filters applied
  to the SCF's population before computing the summary statistics.
  - `Educ`: education level. It can take the levels `NoHS` for individuals without a high-school diploma, `HS` for individuals with a high-school diploma but no college degree, and
    `College` for individuals with a college degree. `All` marks rows in which individuals from all educational attainment levels were used.
  - `Age_grp`: age group. I split the sample in 5-year brackets according to their age and this variable indicates which bracket the statistics correspond to. Note that the left
     extreme of brackets is not included, so `(20,25]` corresponds to ages `{21,22,23,24,25}`. `All` marks rows in which all age groups were used.
  - `YEAR`: survey wave of the SCF. It indicates which waves of the SCF were used in calculating the row's statistics. `All` marks rows in which all waves were pooled. **NOTE:**
     when combining multiple waves,  I do not re-weight observations: I continue to use the weight variable as if all observations came from the same wave.

 - Summary statistics:
  - `lnPermIncome.mean` and `lnPermIncome.sd`: survey-weighted mean and standard deviation of the natural logarithm of "permanent income", as measured by the variable `norminc`
    in the SCF's summary files. Note that this measure contains, among others, capital gains and pension-fund withdrawals and might therefore substantially differ from the
    popular concept of the "permanent component" of labor income, especially at older ages
    (see [the SCF's website for exact definitions](https://www.federalreserve.gov/econres/scfindex.htm)).

# SCFDistTools.py

Contains functions to read the table and its columns, and convert them to
parameters to be used by HARK's `AgentType` classes in their `sim_birth()`
methods to produce realistically calibrated distributions of initial permanent
income and wealth levels.
