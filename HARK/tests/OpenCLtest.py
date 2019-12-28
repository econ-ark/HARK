'''
Simple test for opencl4py, edited from the example distributed in that package.
'''

import os
import opencl4py as cl
import numpy as np
os.environ["PYOPENCL_CTX"] = "0:0" # This is where you set which devices are in the context
# EVERY machine will have a device 0:0, which by default is the CPU
# Other devices will have various numbers
# Substitute her the device you want to compare to the CPU
from time import perf_counter

if __name__ == "__main__":
    
    N = 20000000 # Size of vectors to work with
    use_DP = True # Whether to use double precision floating point
    
    # Print all of the platforms and devices to screen
    platforms = cl.Platforms()
    print('List of all platforms and devices:')
    print(platforms.dump_devices())
    print('')
    
    # Create a context and a queue
    ctx = platforms.create_some_context()
    queue = ctx.create_queue(ctx.devices[0]) # This is where you choose a device
    
    # Tell the user about the test that will be run
    device_can_use_DP = ('cl_khr_fp64' in queue.device.extensions) or ('cl_amd_fp64' in queue.device.extensions)
    print('Will test OpenCL on ' + queue.device.name + '.')
    if use_DP and device_can_use_DP:
        print('The test will use double precision, which this device can handle.')
    elif (not use_DP) and device_can_use_DP:
        print('The test will use single precision, but the device is capable of double precision.')
    elif use_DP and (not device_can_use_DP):
        print ('The test would use double precision, but the device is not capable; expect an error.')
    elif (not use_DP) and (not device_can_use_DP):
        print('The test will use single precision.  This device is not capable of double precision.')
    print('Test vectors have ' + str(N) + ' elements each.\n')
    
    
    # Define a simple kernel using double precision floating point
    double_code = """
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        
        __kernel void test(__global const double *a, __global const double *b,
                           __global double *c, const double k) {
          size_t i = get_global_id(0);
          c[i] = (a[i] + b[i]) * k;
        }
        """
        
    # Define a simple kernel using single precision floating point
    single_code = """
        __kernel void test(__global const float *a, __global const float *b,
                           __global float *c, const float k) {
          size_t i = get_global_id(0);
          c[i] = (a[i] + b[i]) * k;
        }
        """
    
    # Define the kernel using single or double precision as selected above
    if use_DP:
        my_type = np.float64
        prg = ctx.create_program(double_code)
    else:
        my_type = np.float32
        prg = ctx.create_program(single_code)
    krn = prg.get_kernel("test")
    
    # Define numpy arrays with arbitrary numbers
    a = np.arange(N, dtype=my_type) # Input array a
    b = np.arange(N, dtype=my_type) # Input array b
    c = np.empty(N, dtype=my_type)  # Input array c (which will hold result of calculation)
    k = np.array([0.8], dtype=my_type) # Constant scalar
    
    # Define OpenCL memory buffers, passing appropriate flags and inputs
    a_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, a) # Read only, copy values from host, use array a
    b_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR, b) # Read only, copy values from host, use array b
    c_buf = ctx.create_buffer(cl.CL_MEM_WRITE_ONLY | cl.CL_MEM_ALLOC_HOST_PTR, size=c.nbytes) # Write only, allocate memory, use byte size of array c
    
    # Run the kernel and time it
    t_start = perf_counter()
    krn.set_args(a_buf, b_buf, c_buf, k[0:1]) # Set kernel arguments as the three buffers and a float
    queue.execute_kernel(krn, [N], None) # Execute the simple kernel, specifying the global workspace dimensionality and local workspace dimensionality (None uses some default)  
    queue.read_buffer(c_buf, c) # Read the memory buffer for c into the numpy array for c 
    t_end = perf_counter()
    print('OpenCL took ' + str(t_end-t_start) + ' seconds.')
    
    # Now do the equivalent work as the kernel, but in Python (and time it)
    t_start = perf_counter()
    truth = (a + b) * k[0]
    t_end = perf_counter()
    print('Python took ' + str(t_end-t_start) + ' seconds.')
    
    # Make sure that OpenCL and Python actually agree on their results
    max_diff = np.max(np.fabs(c - truth))
    print("Maximum difference between OpenCL and Python calculations is " +  str(max_diff))
