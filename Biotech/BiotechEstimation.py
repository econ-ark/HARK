'''
This is an early run of the Biotech structural estimation.
'''

import numpy as np
import BiotechModel as Model
from time import time
import csv

# Choose whether Phase III is an exogenous process
simple_model = False

# Load the data into memory
infile = open('BiotechEstimationData.txt','r') 
my_reader = csv.reader(infile,delimiter='\t')
all_data = list(my_reader)
infile.close()
observations = len(all_data)
firm_data = np.zeros(observations,dtype=int)
time_data = np.zeros(observations,dtype=int)
value_data = np.zeros(observations,dtype=float)
event_data = np.zeros(observations,dtype=int)
share_data = np.zeros(observations,dtype=float)
cash_data = np.zeros(observations,dtype=float)
for j in range(observations):
    firm_data[j] = int(all_data[j][0])
    time_data[j] = int(all_data[j][1])
    if all_data[j][2] != '':
        value_data[j] = float(all_data[j][2])
    else:
        value_data[j] = np.nan
    event_data[j] = int(all_data[j][3])
    if all_data[j][2] != '':
        share_data[j] = float(all_data[j][4])
    else:
        share_data[j] = np.nan
    if all_data[j][2] != '':
        cash_data[j] = float(all_data[j][5])
    else:
        cash_data[j] = np.nan
        
# Shape the data into objects useful for the estimation
t_min = np.min(time_data)
t_max = np.max(time_data)
time_size = t_max - t_min + 1
firm_size = np.max(firm_data) + 1
time_data = time_data - t_min
# Boolean array of which firms start in each period
start_array = np.zeros((firm_size,time_size),dtype=bool)
starts = event_data == 1
start_array[firm_data[starts],time_data[starts]] = True
# Boolean array of which firms raise capital in each period
VC_array = np.zeros((firm_size,time_size),dtype=bool)
VCs = event_data == 2
VC_array[firm_data[VCs],time_data[VCs]] = True
# Boolean array of which firms go public in each period
IPO_array = np.zeros((firm_size,time_size),dtype=bool)
IPOs = event_data == 3
IPO_array[firm_data[IPOs],time_data[IPOs]] = True
# Boolean array of which firms privately sell in each period
sale_array = np.zeros((firm_size,time_size),dtype=bool)
sales = event_data == 4
sale_array[firm_data[sales],time_data[sales]] = True
# Boolean array of which firms go bankrupt in each period
bankrupt_array = np.zeros((firm_size,time_size),dtype=bool)
bankruptcies = event_data == 5
bankrupt_array[firm_data[bankruptcies],time_data[bankruptcies]] = True
# Float array of valuations of all types (pre-funding)
value_array = np.zeros((firm_size,time_size),dtype=float) + np.nan
value_array[firm_data[VCs],time_data[VCs]] = value_data[VCs]
value_array[firm_data[IPOs],time_data[IPOs]] = value_data[IPOs]
value_array[firm_data[sales],time_data[sales]] = value_data[sales]
value_array[firm_data[bankruptcies],time_data[bankruptcies]] = 0.0
# Float array of sharesold at VC, with nan otherwise
share_array = np.zeros((firm_size,time_size),dtype=float) + np.nan
share_array[firm_data[VCs],time_data[VCs]] = share_data[VCs]
# Float array of cash injection at VC, with nan otherwise
cash_array = np.zeros((firm_size,time_size),dtype=float) + np.nan
cash_array[firm_data[VCs],time_data[VCs]] = cash_data[VCs]
# Boolean array of firms active in each period
active_array = np.ones((firm_size,time_size),dtype=bool)
start_idx = np.where(starts)[0]
for j in range(np.sum(starts)):
    firm = firm_data[start_idx[j]]
    t_start = time_data[start_idx[j]]
    active_array[firm,0:t_start] = False
term_idx = np.where(np.bitwise_or(np.bitwise_or(IPOs,sales),bankruptcies))[0]
for j in range(term_idx.size):
    firm = firm_data[term_idx[j]]
    t_end = time_data[term_idx[j]]
    active_array[firm,(t_end+1):time_size] = False

def listParams(idx=None,values=None,alt=False):
    '''
    Lists the requested model parameters, or all of them if no input is given.
    '''
    if alt:
        names = Model.BiotechType.param_list_alt
    else:
        names = Model.BiotechType.param_list
    if idx is None:
        index = range(len(names))
    else:
        index = np.sort(idx).tolist()
    i = 0
    for j in index:
        this_line = str(j) + ': ' + names[j]
        if values is not None:
            this_line += ' = ' + str(values[i])
        print(this_line)
        i += 1

# Set the calibrated parameters and initial values for estimated parameters
if simple_model:
    param_count = Model.BiotechType.param_count_alt
    calibrated_parameter_idx = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,14,15,17,19,20,46,53,55,56])
    calibrated_parameter_values = np.array([0.000001,150.0,14,-1.0,4.0,15,0.5,20.0,16,-2.5,2.5,17,7,0,0,1,0,0,0.33,0.8,2,0.5])
    estimated_parameter_idx = np.setdiff1d(np.arange(param_count),calibrated_parameter_idx)
    estimated_parameter_guess = np.array([0,1,1,0.1,5,0.3,0.1,0.5,0.2,1.5,0.3,-1,1.5,0.1,-0.01,0.02,0,1.5,0.1,-0.01,0.02,0.3,0,1.5,0.1,-0.01,0.02,0.2,-2,-0.15,0.4,-0.02,-0.01,0.2,0.99,0.01,0.03])
else:
    param_count = Model.BiotechType.param_count_alt
    calibrated_parameter_idx = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,18,20,21,51,53,54])    
    calibrated_parameter_values = np.array([0.000001,150.0,14,-1.0,4.0,15,0.5,20.0,16,-2.5,2.5,17,10,7,0,0,1,0,0,0.8,2,0.5])
    estimated_parameter_idx = np.setdiff1d(np.arange(param_count),calibrated_parameter_idx)
    estimated_parameter_guess = np.array([0,1,1,0.1,5,0.3,0.1,0.5,0.2,1.5,0.3,-1,1.5,0.1,-0.01,0.02,0,1.5,0.1,-0.01,0.02,0.3,0,1.5,0.1,-0.01,0.02,0.2,0.1,1.1,0.1,0.3,0.95,0.01,0.03])


# Make a sample type
my_params = np.zeros(param_count)
my_params[calibrated_parameter_idx] = calibrated_parameter_values
my_params[estimated_parameter_idx] = estimated_parameter_guess
my_type = Model.BiotechType(simple=simple_model)
my_type.updateParams(my_params)
my_type.update()

# Solve the model!
t_start = time()
my_type.solve()
t_end = time()
print('Solving took ' + str(t_end-t_start) + ' seconds.')
