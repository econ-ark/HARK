'''
This module sets up the SCF data for use in the SolvingMicroDSOPs estimation.
'''
# Import the HARK library.  The assumption is that this code is in a folder
# contained in the HARK folder. 
import sys 
sys.path.insert(0,'../')

# The following libraries are part of the standard python distribution
from __future__ import division                         # Use new division function
import numpy as np                                      # Numerical Python
import csv
from SetupConsumerParameters import initial_age, empirical_cohort_age_groups                           

# Libraries below are part of HARK's module system and must be in this directory
from HARKutilities import warnings

# Set the path to the empirical data:
scf_data_path = './'

# Open the file handle and create a reader object and a csv header
infile = open(scf_data_path + 'SCFdata.csv', 'rb')  
csv_reader = csv.reader(infile)
data_csv_header = csv_reader.next()

# Pull the column index from the data_csv_header
data_column_index = data_csv_header.index('wealth_income_ratio') # scf_w_col
age_group_column_index = data_csv_header.index('age_group')      # scf_ages_col
data_weight_column_index = data_csv_header.index('weight')       # scf_weights_col

# Initialize empty lists for the data
w_to_y_data = []              # Ratio of wealth to permanent income
empirical_weights = []        # Weighting for this observation
empirical_groups = []         # Which age group this observation belongs to (1-7)

# Read in the data from the datafile by looping over each record (row) in the file.
for record in csv_reader:
    w_to_y_data.append(np.float64(record[data_column_index]))
    empirical_groups.append(np.float64(record[age_group_column_index]))
    empirical_weights.append(np.float64(record[data_weight_column_index]))

# Generate a single array of SCF data, useful for resampling for bootstrap
scf_data_array = np.array([w_to_y_data,empirical_groups,empirical_weights]).T

# Convert SCF data to numpy's array format for easier math
w_to_y_data = np.array(w_to_y_data)
empirical_groups = np.array(empirical_groups)
empirical_weights = np.array(empirical_weights)

# Close the data file
infile.close()


# Generate a mapping between the real ages in the groups and the indices of simulated data
simulation_map_cohorts_to_age_indices = []
for ages in empirical_cohort_age_groups:
    simulation_map_cohorts_to_age_indices.append(np.array(ages) - initial_age)
    