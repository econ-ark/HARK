'''
Sets up the SCF data for use in the SolvingMicroDSOPs estimation.
'''
from __future__ import division      # Use new division function
from __future__ import print_function
from __future__ import absolute_import
from builtins import zip
from builtins import str
from builtins import range

import os, sys

# Find pathname to this file:
my_file_path = os.path.dirname(os.path.abspath(__file__))

# Pathnames to the other files:
calibration_dir = os.path.join(my_file_path, "../Calibration/") # Relative directory for primitive parameter files
tables_dir = os.path.join(my_file_path, "../Tables/") # Relative directory for primitive parameter files
figures_dir = os.path.join(my_file_path, "../Figures/") # Relative directory for primitive parameter files
code_dir = os.path.join(my_file_path, "../Code/") # Relative directory for primitive parameter files


# Import modules from local repository. If local repository is part of HARK, 
# this will import from HARK. Otherwise manual pathname specification is in 
# order.
try: 
    # Import from core HARK code first:
    from HARK.SolvingMicroDSOPs.Calibration.EstimationParameters import initial_age, empirical_cohort_age_groups

except:
    # Need to rely on the manual insertion of pathnames to all files in do_all.py
    # NOTE sys.path.insert(0, os.path.abspath(tables_dir)), etc. may need to be 
    # copied from do_all.py to here

    # Import files first:
    from EstimationParameters import initial_age, empirical_cohort_age_groups


# The following libraries are part of the standard python distribution
import numpy as np                   # Numerical Python
import csv                           # Comma-separated variable reader

# Set the path to the empirical data:
scf_data_path = data_location = os.path.dirname(os.path.abspath(__file__))  # os.path.abspath('./')   #'./'

# Open the file handle and create a reader object and a csv header
infile = open(scf_data_path + '/SCFdata.csv', 'r')
csv_reader = csv.reader(infile)
data_csv_header = next(csv_reader)

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


if __name__ == '__main__':
    print("Sorry, SetupSCFdata doesn't actually do anything on its own.")
    print("This module is imported by StructEstimation, providing data for")
    print("the example estimation.  Please see that module if you want more")
    print("interesting output.")

