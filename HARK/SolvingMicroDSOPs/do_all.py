'''
Run all of the plots and tables in SolvingMicroDSOPs.

To execute, do the following on the Python command line:

    from HARK.[YOUR-MODULE-NAME-HERE].do_all import run_replication
    run_replication()

You will be presented with an interactive prompt that asks what level of 
replication you would like to have. 

More Details
------------

This example script allows the user to create all of the Figures and Tables 
modules for SolvingMicroDSOPs.StructuralEstimation. 

This is example is kept as simple and minimal as possible to illustrate the 
format of a "replication archive."

The file structure is as follows:

./SolvingMicroDSOPs/
    Calibration/        # Directory that contain the necessary code and data to parameterize the model 
    Code/               # The main estimation code, in this case StructuralEstimation.py
    Figures/            # Any Figures created by the main code
    Tables/             # Any tables created by the main code

Because computational modeling can be very memory- and time-intensive, this file 
also allows the user to choose whether to run files based on there resouce 
requirements. Files are categorized as one of the following three:

- low_resource:     low RAM needed and runs quickly, say less than 1-5 minutes
- medium_resource:  moderate RAM needed and runs moderately quickly, say 5-10+ mintues
- high_resource:    high RAM needed (and potentially parallel computing required), and high time to run, perhaps even hours, days, or longer. 

The designation is purposefully vague and left up the to researcher to specify 
more clearly below. Using time taken on an example machine is entirely reasonable 
here. 

Finally, this code may serve as example code for efforts that fall outside 
the HARK package structure for one reason or another. Therefore this script will 
attempt to import the necessary MicroDSOP sub-modules as though they are part of 
the HARK package; if that fails, this script reverts to manaully updating the 
Python PATH with the locations of the MicroDSOP directory structure so it can 
still run. 
'''

from __future__ import division, print_function
from builtins import str, range

import os, sys

# Find pathname to this file:
my_file_path = os.path.dirname(os.path.abspath(__file__))

# Pathnames to the other files:
calibration_dir = os.path.join(my_file_path, "Calibration") # Relative directory for primitive parameter files
tables_dir = os.path.join(my_file_path, "Tables") # Relative directory for primitive parameter files
figures_dir = os.path.join(my_file_path, "Figures") # Relative directory for primitive parameter files
code_dir = os.path.join(my_file_path, "Code") # Relative directory for primitive parameter files

# Import modules from local repository. If local repository is part of HARK, 
# this will import from HARK. Otherwise manual pathname specification is in 
# order.
try: 
    # Import from core HARK code first:
    from HARK.SolvingMicroDSOPs.Code import StructEstimation as struct
except:
    print("**************** Manually specifying pathnames for modules *******************")
    # It appears that the current module is not part of HARK, therefore we will
    # manually add the pathnames to the various files directly to the beginning
    # of the Python path. This will be needed for all files that will run in 
    # lower directories.
    sys.path.insert(0, calibration_dir)
    sys.path.insert(0, tables_dir)
    sys.path.insert(0, figures_dir)
    sys.path.insert(0, code_dir)
    sys.path.insert(0, my_file_path)

    # Manual import needed, should draw from first instance at start of Python 
    # PATH added above: 
    import StructEstimation as struct


# Define settings for "main()" function in StructuralEstiamtion.py based on 
# resource requirements: 

low_resource = {'estimate_model':True, 'make_contour_plot':False, 'compute_standard_errors':False}
# Author note: 
# This takes approximately 90 seconds on a laptop with the following specs:
# Linux, Ubuntu 14.04.1 LTS, 8G of RAM, Intel(R) Core(TM) i7-4700MQ CPU @ 2.40GHz

medium_resource = {'estimate_model':True, 'make_contour_plot':True, 'compute_standard_errors':False}
# Author note: 
# This takes approximately 7 minutes on a laptop with the following specs:
# Linux, Ubuntu 14.04.1 LTS, 8G of RAM, Intel(R) Core(TM) i7-4700MQ CPU @ 2.40GHz

high_resource = {'estimate_model':True, 'make_contour_plot':False, 'compute_standard_errors':True}
# Author note: 
# This takes approximately 30 minutes on a laptop with the following specs:
# Linux, Ubuntu 14.04.1 LTS, 8G of RAM, Intel(R) Core(TM) i7-4700MQ CPU @ 2.40GHz

all_replications = {'estimate_model':True, 'make_contour_plot':True, 'compute_standard_errors':True}
# Author note: 
# This takes approximately 40 minutes on a laptop with the following specs:
# Linux, Ubuntu 14.04.1 LTS, 8G of RAM, Intel(R) Core(TM) i7-4700MQ CPU @ 2.40GHz


# Ask the user which replication to run, and run it: 
def run_replication():
    which_replication = input("""Which replication would you like to run? (See documentation in do_all.py for details.) Please enter the option number to run that option; default is in brackets:
        
        [1] low-resource:    ~90 sec; output ./Tables/estimate_results.csv
        
         2  medium-resource: ~7 min;  output ./Figures/SMMcontour.pdf
                                             ./Figures/SMMcontour.png
         3  high-resource:   ~30 min; output ./Tables/bootstrap_results.csv

         4  all:             ~40 min; output: all above.
         
         q  quit: exit without executing.\n\n""")


    if which_replication == 'q':
        return

    elif which_replication == '1' or which_replication == '':
        print("Running low-resource replication...")
        struct.main(**low_resource)
        
    elif which_replication == '2':
        print("Running medium-resource replication...")
        struct.main(**medium_resource)

    elif which_replication == '3':
        print("Running high-resource replication...")
        struct.main(**high_resource)

    elif which_replication == '4':
        print("Running all replications...")
        struct.main(**all_replications)

    else:
        return

if __name__ == '__main__':
    run_replication()
