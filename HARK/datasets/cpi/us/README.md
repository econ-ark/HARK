# Inflation adjustments using the CPI research series.

This folder contains tools for transforming nominal U.S. dollar quantities to different base years using the consumer price index.

## Data

The dataset is stored in file `./r-cpi-u-rs-allitems.xlsx`, which comes directly from the [U.S. Bureau of Labor Statistics](https://www.bls.gov/cpi/research-series/r-cpi-u-rs-home.htm).
As of January 21, 2021 the direct link to the file was [https://www.bls.gov/cpi/research-series/r-cpi-u-rs-allitems.xlsx](https://www.bls.gov/cpi/research-series/r-cpi-u-rs-allitems.xlsx).

The file contains the monthly research series (retroactive, using current methods) of the consumer price index (all items) starting in 1977.

### Format

The Excel file is formatted so that each row represents a year, with CPI measurements for each month stored in columns in ascending order. There is an
additional column labeled `AVG` which corresponds to the average of CPI measurements across months.

## Tools

`CPITools.py` contains functions to:
- Download the latest series file from the BLS to the working directory.
- Read and format the CPI series as a Pandas DataFrame.
- Produce CPI deflators using the series for any given pair of years (between 1977 and 2019) and taking any month as base.
