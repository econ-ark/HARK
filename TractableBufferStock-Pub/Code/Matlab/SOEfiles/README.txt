This folder contains Matlab script and functions used in the small open economy
portion of the problem.  A list of files in this directory is below.

- README.txt                  : The text file you are currently reading.
- AddNewGen.m                 : Adds a new period to the current population census,
				using the given stake, population growth factor,
				and wage growth factor.
- CensusMakeNoStakes.m        : Initializes the population census for an economy
				with no stakes, where everyone starts with with
				zero assets.
- CensusMakeStakes.m          : Initializes the population census for an economy
				with stakes, where everyone starts at the steady
				state level of assets.
- CensusPrint.m               : Displays the values in CensusMeans.
- NIPAAggPrint.m              : Displays the aggregate census variables.
- NIPAIndPrint.m              : Displays the individual census variables.
- setupSOE.m                  : Initializes the necessary variables for the small
				open economy part of the problem.
- TabulateLastCensus.m        : Calculates the means of each census variable ac-
				ross subpopulation weighting by the population
				size of each subpopulation.
