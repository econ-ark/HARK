from pandas_datareader.fred import FredReader
import pandas as pd

start='1966-01'
end='2004-12'

# Define series to load from Fred
series = {"PCEPILFE": "pcecore",          # PCE deflator excl food and energy (index 2012) [M]
          'GDPC1': 'gdp',                 # real GDP (index 2012)
          "FEDFUNDS": "ffr"}              # Fed Funds rate

# Load series from Fred
df_fred = FredReader(series.keys(), start='1959-01').read().rename(series, axis='columns')
df_fred.index = df_fred.index.to_period('M')

# make everything quarterly
df_fred = df_fred.groupby(pd.PeriodIndex(df_fred.index, freq='Q')).mean()

# start new dataframe into which we'll selectively load
df = pd.DataFrame(index=df_fred.index)

# Load series
df['pi'] = 100*((df_fred['pcecore']/df_fred['pcecore'].shift(1))**4-1)  # inflation is PCE
df['Y'] = df_fred['gdp']
df['i'] = df_fred['ffr']  # fed funds rate

# only keep start to end and define time variable
df = df[start:end]
df.index.name = 't'

df.to_csv('us_data.csv')
