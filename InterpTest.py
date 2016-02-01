'''
Tests of the new interpolators.
'''
import numpy as np
from HARKinterpolation import *
from copy import deepcopy
from time import clock
import matplotlib.pyplot as plt

RNG = np.random.RandomState(123)

if False:
    f = lambda x,y : 3.0*x**2.0 + x*y + 4.0*y**2.0
    dfdx = lambda x,y : 6.0*x + y
    dfdy = lambda x,y : x + 8.0*y
    
    y_list = np.linspace(0,5,100,dtype=float)
    xInterpolators = []
    xInterpolators_alt = []
    for y in y_list:
        this_x_list = np.sort((RNG.rand(100)*5.0))
        this_interpolation = LinearInterp(this_x_list,f(this_x_list,y*np.ones(this_x_list.size)))
        that_interpolation = Cubic1DInterpDecay(this_x_list,f(this_x_list,y*np.ones(this_x_list.size)),dfdx(this_x_list,y*np.ones(this_x_list.size)))
        xInterpolators.append(this_interpolation)
        xInterpolators_alt.append(that_interpolation)
    g = LinearInterpOnInterp1D(xInterpolators,y_list)
    h = LinearInterpOnInterp1D(xInterpolators_alt,y_list)
    
    rand_x = RNG.rand(100)*5.0
    rand_y = RNG.rand(100)*5.0
    z = (f(rand_x,rand_y) - g(rand_x,rand_y))/f(rand_x,rand_y)
    q = (dfdx(rand_x,rand_y) - g.derivativeX(rand_x,rand_y))/dfdx(rand_x,rand_y)
    r = (dfdy(rand_x,rand_y) - g.derivativeY(rand_x,rand_y))/dfdy(rand_x,rand_y)
    #print(z)
    #print(q)
    #print(r)
    
    z = (f(rand_x,rand_y) - g(rand_x,rand_y))/f(rand_x,rand_y)
    q = (dfdx(rand_x,rand_y) - g.derivativeX(rand_x,rand_y))/dfdx(rand_x,rand_y)
    r = (dfdy(rand_x,rand_y) - g.derivativeY(rand_x,rand_y))/dfdy(rand_x,rand_y)
    print(z)
    #print(q)
    #print(r)


if False:
    f = lambda x,y,z : 3.0*x**2.0 + x*y + 4.0*y**2.0 - 5*z**2.0 + 1.5*x*z
    dfdx = lambda x,y,z : 6.0*x + y + 1.5*z
    dfdy = lambda x,y,z : x + 8.0*y
    dfdz = lambda x,y,z : -10.0*z + 1.5*x
    
    y_list = np.linspace(0,5,51,dtype=float)
    z_list = np.linspace(0,5,51,dtype=float)
    xInterpolators = []
    for y in y_list:
        temp = []
        for z in z_list:
            this_x_list = np.sort((RNG.rand(100)*5.0))
            this_interpolation = LinearInterp(this_x_list,f(this_x_list,y*np.ones(this_x_list.size),z*np.ones(this_x_list.size)))
            temp.append(this_interpolation)
        xInterpolators.append(deepcopy(temp))
    g = BilinearInterpOnInterp1D(xInterpolators,y_list,z_list)
    
    rand_x = RNG.rand(1000)*5.0
    rand_y = RNG.rand(1000)*5.0
    rand_z = RNG.rand(1000)*5.0
    z = (f(rand_x,rand_y,rand_z) - g(rand_x,rand_y,rand_z))/f(rand_x,rand_y,rand_z)
    q = (dfdx(rand_x,rand_y,rand_z) - g.derivativeX(rand_x,rand_y,rand_z))/dfdx(rand_x,rand_y,rand_z)
    r = (dfdy(rand_x,rand_y,rand_z) - g.derivativeY(rand_x,rand_y,rand_z))/dfdy(rand_x,rand_y,rand_z)
    p = (dfdz(rand_x,rand_y,rand_z) - g.derivativeZ(rand_x,rand_y,rand_z))/dfdz(rand_x,rand_y,rand_z)
    z.sort()



if False:
    f = lambda w,x,y,z : 4.0*w*z - 2.5*w*x + w*y + 6.0*x*y - 10.0*x*z + 3.0*y*z - 7.0*z + 4.0*x + 2.0*y - 5.0*w
    dfdw = lambda w,x,y,z : 4.0*z - 2.5*x + y - 5.0
    dfdx = lambda w,x,y,z : -2.5*w + 6.0*y - 10.0*z + 4.0
    dfdy = lambda w,x,y,z : w + 6.0*x + 3.0*z + 2.0
    dfdz = lambda w,x,y,z : 4.0*w - 10.0*x + 3.0*y - 7
    
    x_list = np.linspace(0,5,16,dtype=float)
    y_list = np.linspace(0,5,16,dtype=float)
    z_list = np.linspace(0,5,16,dtype=float)
    wInterpolators = []
    for x in x_list:
        temp = []
        for y in y_list:
            temptemp = []
            for z in z_list:
                this_w_list = np.sort((RNG.rand(16)*5.0))
                this_interpolation = LinearInterp(this_w_list,f(this_w_list,x*np.ones(this_w_list.size),y*np.ones(this_w_list.size),z*np.ones(this_w_list.size)))
                temptemp.append(this_interpolation)
            temp.append(deepcopy(temptemp))
        wInterpolators.append(deepcopy(temp))
    g = TrilinearInterpOnInterp1D(wInterpolators,x_list,y_list,z_list)
    
    N = 20000
    rand_w = RNG.rand(N)*5.0
    rand_x = RNG.rand(N)*5.0
    rand_y = RNG.rand(N)*5.0
    rand_z = RNG.rand(N)*5.0
    t_start = clock()
    z = (f(rand_w,rand_x,rand_y,rand_z) - g(rand_w,rand_x,rand_y,rand_z))/f(rand_w,rand_x,rand_y,rand_z)
    q = (dfdw(rand_w,rand_x,rand_y,rand_z) - g.derivativeW(rand_w,rand_x,rand_y,rand_z))/dfdw(rand_w,rand_x,rand_y,rand_z)
    r = (dfdx(rand_w,rand_x,rand_y,rand_z) - g.derivativeX(rand_w,rand_x,rand_y,rand_z))/dfdx(rand_w,rand_x,rand_y,rand_z)
    p = (dfdy(rand_w,rand_x,rand_y,rand_z) - g.derivativeY(rand_w,rand_x,rand_y,rand_z))/dfdy(rand_w,rand_x,rand_y,rand_z)
    s = (dfdz(rand_w,rand_x,rand_y,rand_z) - g.derivativeZ(rand_w,rand_x,rand_y,rand_z))/dfdz(rand_w,rand_x,rand_y,rand_z)
    t_end = clock()
    
    z.sort()
    print(z)
    print(t_end-t_start)

if False:
    f = lambda x,y : 3.0*x**2.0 + x*y + 4.0*y**2.0
    dfdx = lambda x,y : 6.0*x + y
    dfdy = lambda x,y : x + 8.0*y
    
    x_list = np.linspace(0,5,101,dtype=float)
    y_list = np.linspace(0,5,101,dtype=float)
    x_temp,y_temp = np.meshgrid(x_list,y_list,indexing='ij')
    g = BilinearInterp(f(x_temp,y_temp),x_list,y_list)
    
    rand_x = RNG.rand(100)*5.0
    rand_y = RNG.rand(100)*5.0
    z = (f(rand_x,rand_y) - g(rand_x,rand_y))/f(rand_x,rand_y)
    q = (f(x_temp,y_temp) - g(x_temp,y_temp))/f(x_temp,y_temp)
    #print(z)
    #print(q)
    
    
if False:
    f = lambda x,y,z : 3.0*x**2.0 + x*y + 4.0*y**2.0 - 5*z**2.0 + 1.5*x*z
    dfdx = lambda x,y,z : 6.0*x + y + 1.5*z
    dfdy = lambda x,y,z : x + 8.0*y
    dfdz = lambda x,y,z : -10.0*z + 1.5*x
    
    x_list = np.linspace(0,5,11,dtype=float)
    y_list = np.linspace(0,5,11,dtype=float)
    z_list = np.linspace(0,5,101,dtype=float)
    x_temp,y_temp,z_temp = np.meshgrid(x_list,y_list,z_list,indexing='ij')
    g = TrilinearInterp(f(x_temp,y_temp,z_temp),x_list,y_list,z_list)
    
    rand_x = RNG.rand(1000)*5.0
    rand_y = RNG.rand(1000)*5.0
    rand_z = RNG.rand(1000)*5.0
    z = (f(rand_x,rand_y,rand_z) - g(rand_x,rand_y,rand_z))/f(rand_x,rand_y,rand_z)
    q = (dfdx(rand_x,rand_y,rand_z) - g.derivativeX(rand_x,rand_y,rand_z))/dfdx(rand_x,rand_y,rand_z)
    r = (dfdy(rand_x,rand_y,rand_z) - g.derivativeY(rand_x,rand_y,rand_z))/dfdy(rand_x,rand_y,rand_z)
    p = (dfdz(rand_x,rand_y,rand_z) - g.derivativeZ(rand_x,rand_y,rand_z))/dfdz(rand_x,rand_y,rand_z)
    p.sort()    
    plt.plot(p)
    
    
if False:
    f = lambda w,x,y,z : 4.0*w*z - 2.5*w*x + w*y + 6.0*x*y - 10.0*x*z + 3.0*y*z - 7.0*z + 4.0*x + 2.0*y - 5.0*w
    dfdw = lambda w,x,y,z : 4.0*z - 2.5*x + y - 5.0
    dfdx = lambda w,x,y,z : -2.5*w + 6.0*y - 10.0*z + 4.0
    dfdy = lambda w,x,y,z : w + 6.0*x + 3.0*z + 2.0
    dfdz = lambda w,x,y,z : 4.0*w - 10.0*x + 3.0*y - 7
    
    w_list = np.linspace(0,5,16,dtype=float)
    x_list = np.linspace(0,5,16,dtype=float)
    y_list = np.linspace(0,5,16,dtype=float)
    z_list = np.linspace(0,5,16,dtype=float)
    w_temp,x_temp,y_temp,z_temp = np.meshgrid(w_list,x_list,y_list,z_list,indexing='ij')
    mySearch = lambda trash,x : np.floor(x/5*32).astype(int)
    g = QuadlinearInterp(f(w_temp,x_temp,y_temp,z_temp),w_list,x_list,y_list,z_list)
    
    N = 1000
    rand_w = RNG.rand(N)*5.0
    rand_x = RNG.rand(N)*5.0
    rand_y = RNG.rand(N)*5.0
    rand_z = RNG.rand(N)*5.0
    t_start = clock()
    z = (f(rand_w,rand_x,rand_y,rand_z) - g(rand_w,rand_x,rand_y,rand_z))/f(rand_w,rand_x,rand_y,rand_z)
    t_end = clock()
    #print(z)
    print(t_end-t_start)


if False:
    f = lambda x,y : 3.0*x**2.0 + x*y + 4.0*y**2.0
    dfdx = lambda x,y : 6.0*x + y
    dfdy = lambda x,y : x + 8.0*y
    
    warp_factor = 0.01
    x_list = np.linspace(0,5,71,dtype=float)
    y_list = np.linspace(0,5,51,dtype=float)
    x_temp,y_temp = np.meshgrid(x_list,y_list,indexing='ij')
    x_adj = x_temp + warp_factor*(RNG.rand(x_list.size,y_list.size) - 0.5)
    y_adj = y_temp + warp_factor*(RNG.rand(x_list.size,y_list.size) - 0.5)
    g = Curvilinear2DInterp(f(x_adj,y_adj),x_adj,y_adj)
    
    rand_x = RNG.rand(1000)*5.0
    rand_y = RNG.rand(1000)*5.0
    t_start = clock()
    z = (f(rand_x,rand_y) - g(rand_x,rand_y))/f(rand_x,rand_y)
    t_end = clock()
    z.sort()
    print(z)
    print(t_end-t_start)
    
    
if False:
    f = lambda x,y,z : 3.0*x**2.0 + x*y + 4.0*y**2.0 - 5*z**2.0 + 1.5*x*z
    dfdx = lambda x,y,z : 6.0*x + y + 1.5*z
    dfdy = lambda x,y,z : x + 8.0*y
    dfdz = lambda x,y,z : -10.0*z + 1.5*x
    
    warp_factor = 0.01
    x_list = np.linspace(0,5,11,dtype=float)
    y_list = np.linspace(0,5,11,dtype=float)
    z_list = np.linspace(0,5,101,dtype=float)
    x_temp,y_temp = np.meshgrid(x_list,y_list,indexing='ij')
    xyInterpolators = []
    for j in range(z_list.size):
        x_adj = x_temp + warp_factor*(RNG.rand(x_list.size,y_list.size) - 0.5)
        y_adj = y_temp + warp_factor*(RNG.rand(x_list.size,y_list.size) - 0.5)
        z_temp = z_list[j]*np.ones(x_adj.shape)
        thisInterp = Curvilinear2DInterp(f(x_adj,y_adj,z_temp),x_adj,y_adj)
        xyInterpolators.append(thisInterp)
    g = LinearInterpOnInterp2D(xyInterpolators,z_list)
    
    N = 1000
    rand_x = RNG.rand(N)*5.0
    rand_y = RNG.rand(N)*5.0
    rand_z = RNG.rand(N)*5.0
    z = (f(rand_x,rand_y,rand_z) - g(rand_x,rand_y,rand_z))/f(rand_x,rand_y,rand_z)
    p = (dfdz(rand_x,rand_y,rand_z) - g.derivativeZ(rand_x,rand_y,rand_z))/dfdz(rand_x,rand_y,rand_z)
    p.sort()
    plt.plot(p)
    
    
if True:
    f = lambda w,x,y,z : 4.0*w*z - 2.5*w*x + w*y + 6.0*x*y - 10.0*x*z + 3.0*y*z - 7.0*z + 4.0*x + 2.0*y - 5.0*w
    dfdw = lambda w,x,y,z : 4.0*z - 2.5*x + y - 5.0
    dfdx = lambda w,x,y,z : -2.5*w + 6.0*y - 10.0*z + 4.0
    dfdy = lambda w,x,y,z : w + 6.0*x + 3.0*z + 2.0
    dfdz = lambda w,x,y,z : 4.0*w - 10.0*x + 3.0*y - 7
    
    warp_factor = 0.1
    w_list = np.linspace(0,5,16,dtype=float)
    x_list = np.linspace(0,5,16,dtype=float)
    y_list = np.linspace(0,5,16,dtype=float)
    z_list = np.linspace(0,5,16,dtype=float)
    w_temp,x_temp = np.meshgrid(w_list,x_list,indexing='ij')
    wxInterpolators = []
    for i in range(y_list.size):
        temp = []
        for j in range(z_list.size):
            w_adj = w_temp + warp_factor*(RNG.rand(w_list.size,x_list.size) - 0.5)
            x_adj = x_temp + warp_factor*(RNG.rand(w_list.size,x_list.size) - 0.5)
            y_temp = y_list[i]*np.ones(w_adj.shape)
            z_temp = z_list[j]*np.ones(w_adj.shape)
            thisInterp = Curvilinear2DInterp(f(w_adj,x_adj,y_temp,z_temp),w_adj,x_adj)
            temp.append(thisInterp)
        wxInterpolators.append(temp)
    g = BilinearInterpOnInterp2D(wxInterpolators,y_list,z_list)
    
    N = 1000000
    rand_w = RNG.rand(N)*5.0
    rand_x = RNG.rand(N)*5.0
    rand_y = RNG.rand(N)*5.0
    rand_z = RNG.rand(N)*5.0
    
    t_start = clock()
    z = (f(rand_w,rand_x,rand_y,rand_z) - g(rand_w,rand_x,rand_y,rand_z))/f(rand_w,rand_x,rand_y,rand_z)
    t_end = clock()
    z.sort()
    print(z)
    print(t_end-t_start)
    
    