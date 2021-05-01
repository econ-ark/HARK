from HARK.datasets.life_tables.us_ssa.SSATools import (
    parse_ssa_life_table,
    get_ssa_life_tables
)

import numpy as np
import matplotlib.pyplot as plt

# %% Inspect lifetables

tables = get_ssa_life_tables()
print(tables.head)

# %% Survival probabilities from the SSA

# We will find 1-year survival probabilities from ages 21 to 100
min_age = 21
max_age = 100
ages = np.arange(min_age, max_age + 1)

# In the years 1900 and 1950
years = [1900, 1950]

# %%

# First, the "longitudinal method", which gives us the probabilities
# experienced by agents born in "year" throughout their lived
plt.figure()
for cohort in years:
    for s in ['male', 'female']:
    
        fem = s == 'female'
        LivPrb = parse_ssa_life_table(female = fem, cohort = cohort,
                                      min_age = min_age, max_age = max_age)
    
        plt.plot(ages, LivPrb, label = s + ' born in ' + str(cohort))
    
plt.legend()
plt.title('Longitudinal survival probabilities')

# %%

# Second, the "cross-sectional method", which gives us the probabilities of
# survivals of individuals of differnet ages that are alive in the given year.
plt.figure()
for year in years:
    for s in ['male', 'female']:
    
        fem = s == 'female'
        LivPrb = parse_ssa_life_table(female = fem, year = year, cross_sec= True,
                                      min_age = min_age, max_age = max_age)
    
        plt.plot(ages, LivPrb, label = s + 's in ' + str(year))
    
plt.legend()
plt.title('Cross-sectional survival probabilities')
