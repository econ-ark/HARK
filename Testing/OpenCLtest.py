'''
Simple test for opencl4py, edited from the example distributed in that package.
'''

import os
import opencl4py as cl
import numpy
os.environ["PYOPENCL_CTX"] = "0:2" # This is where you set which devices are in the context
from time import clock

if __name__ == "__main__":
    platforms = cl.Platforms()
    print(platforms.dump_devices())
    ctx = platforms.create_some_context()
    queue = ctx.create_queue(ctx.devices[0]) # This is where you choose a device
    double_code = """
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        #pragma OPENCL EXTENSION cl_amd_fp64 : enable
        
        __kernel void test(__global const double *a, __global const double *b,
                           __global double *c, const double k) {
          size_t i = get_global_id(0);
          c[i] = (a[i] + b[i]) * k;
        }
        """
    single_code = """
        __kernel void test(__global const float *a, __global const float *b,
                           __global float *c, const float k) {
          size_t i = get_global_id(0);
          c[i] = (a[i] + b[i]) * k;
        }
        """
        
    N = 2000000 # Size of vector
    my_type = numpy.float32
    prg = ctx.create_program(single_code)
    krn = prg.get_kernel("test")
    a = numpy.arange(N, dtype=my_type)
    b = numpy.arange(N, dtype=my_type)
    c = numpy.empty(N, dtype=my_type)
    d = numpy.empty(N, dtype=my_type)
    k = numpy.array([0.8], dtype=my_type) # Constant scalar
    a_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,a)
    b_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,b)
    c_buf = ctx.create_buffer(cl.CL_MEM_WRITE_ONLY | cl.CL_MEM_ALLOC_HOST_PTR,size=c.nbytes)
    d_buf = ctx.create_buffer(cl.CL_MEM_WRITE_ONLY | cl.CL_MEM_ALLOC_HOST_PTR,size=c.nbytes)
    
    t_start = clock()
    krn.set_args(a_buf, b_buf, c_buf, k[0:1])
    queue.execute_kernel(krn, [a.size], None)
    
    queue.read_buffer(c_buf, c)
    
    #krn.set_args(a_buf, c_buf, d_buf, k[0:1])
    #queue.execute_kernel(krn, [a.size], None)
    #queue.read_buffer(d_buf, d)
    
    t_end = clock()
    print('OpenCL took ' + str(t_end-t_start) + ' seconds.')
    
    t_start = clock()
    truth = (a + b) * k[0]
    #truth = ((a + b) * k[0] + a) * k[0]
    t_end = clock()
    print('Python took ' + str(t_end-t_start) + ' seconds.')
    
    max_diff = numpy.fabs(c - truth).max()
    print("max_diff = " +  str(max_diff))
