'''
This module contains custom interpolation methods for representing approximations
to functions.  Also includes wrapper classes to enforce standard methods across
classes.  Each interpolation class must have a distance() method that compares
itself to another instance (returning an arbitrarily large distance if the input
is not of the same class); this is used in HARKcore's solve() method to check
for solution convergence.
'''

import warnings
import numpy as np
from scipy.interpolate import UnivariateSpline
from copy import deepcopy

def _isscalar(x):
    """Check whether x is if a scalar type, or 0-dim"""
    return np.isscalar(x) or hasattr(x, 'shape') and x.shape == ()


class HARKinterpolator1D():
    '''
    A wrapper class for 1D interpolation methods in HARK.
    '''
    def __call__(self,x):
        z = np.asarray(x)
        return (self._evaluate(z.flatten())).reshape(z.shape)
        
    def derivative(self,x):
        z = np.asarray(x)
        return (self._der(z.flatten())).reshape(z.shape)
        
    def eval_with_derivative(self,x):
        z = np.asarray(x)
        y, dydx = self._evalAndDer(z.flatten())
        return y.reshape(z.shape), dydx.reshape(z.shape)
        
    def _evaluate(self,x):
        raise NotImplementedError()
        
    def _der(self,x):
        raise NotImplementedError()
        
    def _evalAndDer(self,x):
        raise NotImplementedError()
        
    def distance(self,other):
        raise NotImplementedError()
 
       
        
class HARKinterpolator2D():
    '''
    A wrapper class for 2D interpolation methods in HARK.
    '''
    def __call__(self,x,y):
        xa = np.asarray(x)
        ya = np.asarray(y)
        return (self._evaluate(xa.flatten(),ya.flatten())).reshape(xa.shape)
        
    def derivativeX(self,x,y):
        xa = np.asarray(x)
        ya = np.asarray(y)
        return (self._derX(xa.flatten(),ya.flatten())).reshape(xa.shape)
        
    def derivativeY(self,x,y):
        xa = np.asarray(x)
        ya = np.asarray(y)
        return (self._derY(xa.flatten(),ya.flatten())).reshape(xa.shape)
        
    def _evaluate(self,x,y):
        raise NotImplementedError()
        
    def _derX(self,x,y):
        raise NotImplementedError()
        
    def _derY(self,x,y):
        raise NotImplementedError()
        
    def distance(self,other):
        raise NotImplementedError()


        
class HARKinterpolator3D():
    '''
    A wrapper class for 3D interpolation methods in HARK.
    '''
    def __call__(self,x,y,z):
        xa = np.asarray(x)
        ya = np.asarray(y)
        za = np.asarray(z)
        return (self._evaluate(xa.flatten(),ya.flatten(),za.flatten())).reshape(xa.shape)
        
    def derivativeX(self,x,y,z):
        xa = np.asarray(x)
        ya = np.asarray(y)
        za = np.asarray(z)
        return (self._derX(xa.flatten(),ya.flatten(),za.flatten())).reshape(xa.shape)
        
    def derivativeY(self,x,y,z):
        xa = np.asarray(x)
        ya = np.asarray(y)
        za = np.asarray(z)
        return (self._derY(xa.flatten(),ya.flatten(),za.flatten())).reshape(xa.shape)
        
    def derivativeZ(self,x,y,z):
        xa = np.asarray(x)
        ya = np.asarray(y)
        za = np.asarray(z)
        return (self._derZ(xa.flatten(),ya.flatten(),za.flatten())).reshape(xa.shape)
        
    def _evaluate(self,x,y,z):
        raise NotImplementedError()
        
    def _derX(self,x,y,z):
        raise NotImplementedError()
        
    def _derY(self,x,y,z):
        raise NotImplementedError()
        
    def _derZ(self,x,y,z):
        raise NotImplementedError()
        
    def distance(self,other):
        raise NotImplementedError()
        
        
class HARKinterpolator4D():
    '''
    A wrapper class for 4D interpolation methods in HARK.
    '''
    def __call__(self,w,x,y,z):
        wa = np.asarray(w)
        xa = np.asarray(x)
        ya = np.asarray(y)
        za = np.asarray(z)
        return (self._evaluate(wa.flatten(),xa.flatten(),ya.flatten(),za.flatten())).reshape(wa.shape)

    def derivativeW(self,w,x,y,z):
        wa = np.asarray(w)
        xa = np.asarray(x)
        ya = np.asarray(y)
        za = np.asarray(z)
        return (self._derW(wa.flatten(),xa.flatten(),ya.flatten(),za.flatten())).reshape(wa.shape)

    def derivativeX(self,w,x,y,z):
        wa = np.asarray(w)
        xa = np.asarray(x)
        ya = np.asarray(y)
        za = np.asarray(z)
        return (self._derX(wa.flatten(),xa.flatten(),ya.flatten(),za.flatten())).reshape(wa.shape)
        
    def derivativeY(self,w,x,y,z):
        wa = np.asarray(w)
        xa = np.asarray(x)
        ya = np.asarray(y)
        za = np.asarray(z)
        return (self._derY(wa.flatten(),xa.flatten(),ya.flatten(),za.flatten())).reshape(wa.shape)

    def derivativeZ(self,w,x,y,z):
        wa = np.asarray(w)
        xa = np.asarray(x)
        ya = np.asarray(y)
        za = np.asarray(z)
        return (self._derZ(wa.flatten(),xa.flatten(),ya.flatten(),za.flatten())).reshape(wa.shape)
        
    def _evaluate(self,w,x,y,z):
        raise NotImplementedError()
        
    def _derW(self,w,x,y,z):
        raise NotImplementedError()
        
    def _derX(self,w,x,y,z):
        raise NotImplementedError()
        
    def _derY(self,w,x,y,z):
        raise NotImplementedError()
        
    def _derZ(self,w,x,y,z):
        raise NotImplementedError()
        
    def distance(self,other):
        raise NotImplementedError()


class CubicInterp(HARKinterpolator1D):
    """
    An interpolating function using piecewise cubic splines and "decay extrapolation"
    above the highest gridpoint.  Matches level and slope of 1D function at gridpoints,
    smoothly interpolating in between.  Extrapolation above highest gridpoint approaches
    the limiting linear function y = b_limit + m_limit*x.
    """ 

    def __init__(self,x_list,y_list,dydx_list,b_limit=None,m_limit=None,lower_extrap=False):
        '''
        The interpolation constructor.
        
        Parameters:
        -----------
        x_list : [float]
            List of x values composing the grid.
        y_list : [float]
            List of y values, representing f(x) at the points in x_list.
        dydx_list : [float]
            List of dydx
        b_limit : float
            Intercept of limiting linear function.
        m_limit : float
            Slope of limiting linear function.
            
        NOTE: When no input is given for the limiting linear function, linear
        extrapolation is used above the highest gridpoint.        
        '''
        self.x_list = np.asarray(x_list)
        self.y_list = np.asarray(y_list)
        self.dydx_list = np.asarray(dydx_list)
        self.n = len(x_list)
        
        # Define lower extrapolation as linear function (or just NaN)
        if lower_extrap:
            self.coeffs = [[y_list[0],dydx_list[0],0,0]]
        else:
            self.coeffs = [[np.nan,np.nan,np.nan,np.nan]]

        # Calculate interpolation coefficients on segments mapped to [0,1]
        for i in xrange(self.n-1):
           x0 = x_list[i]
           y0 = y_list[i]
           x1 = x_list[i+1]
           y1 = y_list[i+1]
           Span = x1 - x0
           dydx0 = dydx_list[i]*Span
           dydx1 = dydx_list[i+1]*Span
           
           temp = [y0, dydx0, 3*(y1 - y0) - 2*dydx0 - dydx1, 2*(y0 - y1) + dydx0 + dydx1];
           self.coeffs.append(temp)

        # Calculate extrapolation coefficients as a decay toward limiting function y = mx+b
        if m_limit is None and b_limit is None:
            m_limit = dydx_list[-1]
            b_limit = y_list[-1] - m_limit*x_list[-1]
        gap = m_limit*x1 + b_limit - y1
        slope = m_limit - dydx_list[self.n-1]
        if (gap != 0) and (slope <= 0):
            temp = [b_limit, m_limit, gap, slope/gap]
        elif slope > 0:
            temp = [b_limit, m_limit, 0, 0] # fixing a problem when slope is positive
        else:
            temp = [b_limit, m_limit, gap, 0]
        self.coeffs.append(temp)
        self.coeffs = np.array(self.coeffs)


    def _evaluate(self,x):
        '''
        Returns the level of the function at each value in x.
        '''
        if _isscalar(x):
            pos = np.searchsorted(self.x_list,x)
            if pos == 0:
                y = self.coeffs[0,0] + self.coeffs[0,1]*(x - self.x_list[0])
            elif (pos < self.n):
                alpha = (x - self.x_list[pos-1])/(self.x_list[pos] - self.x_list[pos-1])
                y = self.coeffs[pos,0] + alpha*(self.coeffs[pos,1] + alpha*(self.coeffs[pos,2] + alpha*self.coeffs[pos,3]))
            else:
                alpha = x - self.x_list[self.n-1]
                y = self.coeffs[pos,0] + x*self.coeffs[pos,1] - self.coeffs[pos,2]*np.exp(alpha*self.coeffs[pos,3])
        else:
            m = len(x)
            pos = np.searchsorted(self.x_list,x)
            y = np.zeros(m)
            if y.size > 0:
                out_bot   = pos == 0
                out_top   = pos == self.n
                in_bnds   = np.logical_not(np.logical_or(out_bot, out_top))
                
                # Do the "in bounds" evaluation points
                i = pos[in_bnds]
                coeffs_in = self.coeffs[i,:]
                alpha = (x[in_bnds] - self.x_list[i-1])/(self.x_list[i] - self.x_list[i-1])
                y[in_bnds] = coeffs_in[:,0] + alpha*(coeffs_in[:,1] + alpha*(coeffs_in[:,2] + alpha*coeffs_in[:,3]))
                
                # Do the "out of bounds" evaluation points
                y[out_bot] = self.coeffs[0,0] + self.coeffs[0,1]*(x[out_bot] - self.x_list[0])
                alpha = x[out_top] - self.x_list[self.n-1]
                y[out_top] = self.coeffs[self.n,0] + x[out_top]*self.coeffs[self.n,1] - self.coeffs[self.n,2]*np.exp(alpha*self.coeffs[self.n,3])                      
        return y


    def _der(self,x):
        '''
        Returns the first derivative of the function at each value in x.
        '''
        if _isscalar(x):
            pos = np.searchsorted(self.x_list,x)
            if pos == 0:
                dydx = self.coeffs[0,1]
            elif (pos < self.n):
                alpha = (x - self.x_list[pos-1])/(self.x_list[pos] - self.x_list[pos-1])
                dydx = (self.coeffs[pos,1] + alpha*(2*self.coeffs[pos,2] + alpha*3*self.coeffs[pos,3]))/(self.x_list[pos] - self.x_list[pos-1])
            else:
                alpha = x - self.x_list[self.n-1]
                dydx = self.coeffs[pos,1] - self.coeffs[pos,2]*self.coeffs[pos,3]*np.exp(alpha*self.coeffs[pos,3])
        else:
            m = len(x)
            pos = np.searchsorted(self.x_list,x)
            dydx = np.zeros(m)
            if dydx.size > 0:
                out_bot   = pos == 0
                out_top   = pos == self.n
                in_bnds   = np.logical_not(np.logical_or(out_bot, out_top))
                
                # Do the "in bounds" evaluation points
                i = pos[in_bnds]
                coeffs_in = self.coeffs[i,:]
                alpha = (x[in_bnds] - self.x_list[i-1])/(self.x_list[i] - self.x_list[i-1])
                dydx[in_bnds] = (coeffs_in[:,1] + alpha*(2*coeffs_in[:,2] + alpha*3*coeffs_in[:,3]))/(self.x_list[i] - self.x_list[i-1])
                
                # Do the "out of bounds" evaluation points
                dydx[out_bot] = self.coeffs[0,1]
                alpha = x[out_top] - self.x_list[self.n-1]
                dydx[out_top] = self.coeffs[self.n,1] - self.coeffs[self.n,2]*self.coeffs[self.n,3]*np.exp(alpha*self.coeffs[self.n,3])
        return dydx


    def _evalAndDer(self,x):
        '''
        Returns the level and first derivative of the function at each value in x.
        '''
        if _isscalar(x):
            pos = np.searchsorted(self.x_list,x)
            if pos == 0:
                y = self.coeffs[0,0] + self.coeffs[0,1]*(x - self.x_list[0])
                dydx = self.coeffs[0,1]
            elif (pos < self.n):
                alpha = (x - self.x_list[pos-1])/(self.x_list[pos] - self.x_list[pos-1])
                y = self.coeffs[pos,0] + alpha*(self.coeffs[pos,1] + alpha*(self.coeffs[pos,2] + alpha*self.coeffs[pos,3]))
                dydx = (self.coeffs[pos,1] + alpha*(2*self.coeffs[pos,2] + alpha*3*self.coeffs[pos,3]))/(self.x_list[pos] - self.x_list[pos-1])
            else:
                alpha = x - self.x_list[self.n-1]
                y = self.coeffs[pos,0] + x*self.coeffs[pos,1] - self.coeffs[pos,2]*np.exp(alpha*self.coeffs[pos,3])
                dydx = self.coeffs[pos,1] - self.coeffs[pos,2]*self.coeffs[pos,3]*np.exp(alpha*self.coeffs[pos,3])
        else:
            m = len(x)
            pos = np.searchsorted(self.x_list,x)
            y = np.zeros(m)
            dydx = np.zeros(m)
            if y.size > 0:
                out_bot   = pos == 0
                out_top   = pos == self.n
                in_bnds   = np.logical_not(np.logical_or(out_bot, out_top))
                
                # Do the "in bounds" evaluation points
                i = pos[in_bnds]
                coeffs_in = self.coeffs[i,:]
                alpha = (x[in_bnds] - self.x_list[i-1])/(self.x_list[i] - self.x_list[i-1])
                y[in_bnds] = coeffs_in[:,0] + alpha*(coeffs_in[:,1] + alpha*(coeffs_in[:,2] + alpha*coeffs_in[:,3]))
                dydx[in_bnds] = (coeffs_in[:,1] + alpha*(2*coeffs_in[:,2] + alpha*3*coeffs_in[:,3]))/(self.x_list[i] - self.x_list[i-1])
                
                # Do the "out of bounds" evaluation points
                y[out_bot] = self.coeffs[0,0] + self.coeffs[0,1]*(x[out_bot] - self.x_list[0])
                dydx[out_bot] = self.coeffs[0,1]
                alpha = x[out_top] - self.x_list[self.n-1]
                y[out_top] = self.coeffs[self.n,0] + x[out_top]*self.coeffs[self.n,1] - self.coeffs[self.n,2]*np.exp(alpha*self.coeffs[self.n,3])
                dydx[out_top] = self.coeffs[self.n,1] - self.coeffs[self.n,2]*self.coeffs[self.n,3]*np.exp(alpha*self.coeffs[self.n,3])
        return y, dydx

    def distance(self,other_function):
        '''
        The distance between two instances of the class is the largest difference
        in x values or y values.  If the instances have different numbers of gridpoints,
        then the difference in the number of gridpoints is returned instead.
        '''
        other_class = other_function.__class__.__name__
        if other_class is not 'CubicInterp':
            return 1000000
        xA = self.x_list
        xB = other_function.x_list
        yA = self.y_list
        yB = other_function.y_list
        dydxA = self.dydx_list
        dydxB = other_function.dydx_list
        if (self.n == other_function.n):            
            x_dist = np.max(np.abs(np.subtract(xA,xB)))
            y_dist = np.max(np.abs(np.subtract(yA,yB)))
            dydx_dist = np.max(np.abs(np.subtract(dydxA,dydxB)))
            
            dist = max([x_dist,y_dist,dydx_dist])
        else:
            dist = np.abs(self.n - other_function.n)
        return dist


class LinearInterp(HARKinterpolator1D):
    '''
    A slight extension of scipy.interpolate's UnivariateSpline for linear interpolation.
    Adds a distance method to allow convergence checks.
    '''
    def __init__(self,x,y,intercept_limit=None,slope_limit=None,lower_extrap=False):
        self.function = UnivariateSpline(x,y,k=1,s=0)
        self.lower_extrap = lower_extrap
        # Make a decay extrapolation
        if intercept_limit is not None and slope_limit is not None:
            slope_at_top = self.function(x[-1],1)
            level_diff = intercept_limit + slope_limit*x[-1] - y[-1]
            slope_diff = slope_limit - slope_at_top
            self.decay_extrap_A = level_diff
            self.decay_extrap_B = -slope_diff/level_diff
            self.intercept_limit = intercept_limit
            self.slope_limit = slope_limit
            self.decay_extrap = True
        else:
            self.decay_extrap = False
        
    def _evaluate(self,x):
        out = self.function(x)
        if not self.lower_extrap:
            below_lower_bound = x < self.function._data[0][0]
            out[below_lower_bound] = np.nan
        if self.decay_extrap:
            above_upper_bound = x > self.function._data[0][-1]
            x_temp = x[above_upper_bound] - self.function._data[0][-1]
            out[above_upper_bound] = self.intercept_limit + self.slope_limit*x[above_upper_bound] - self.decay_extrap_A*np.exp(-self.decay_extrap_B*x_temp)
        return out
        
    def _der(self,x):
        out = self.function(x,1)
        if not self.lower_extrap:
            below_lower_bound = x < self.function._data[0][0]
            out[below_lower_bound] = np.nan
        if self.decay_extrap:
            above_upper_bound = x > self.function._data[0][-1]
            x_temp = x[above_upper_bound] - self.function._data[0][-1]
            out[above_upper_bound] = self.slope_limit + self.decay_extrap_B*self.decay_extrap_A*np.exp(-self.decay_extrap_B*x_temp)
        return out
        
    def _evalAndDer(self,x):
        out1 = self.function(x)
        out2 = self.function(x,1)
        if not self.lower_extrap:
            below_lower_bound = x < self.function._data[0][0]
            out1[below_lower_bound] = np.nan
            out2[below_lower_bound] = np.nan
        if self.decay_extrap:
            above_upper_bound = x > self.function._data[0][-1]
            x_temp = x[above_upper_bound] - self.function._data[0][-1]
            out1[above_upper_bound] = self.intercept_limit + self.slope_limit*x[above_upper_bound] - self.decay_extrap_A*np.exp(-self.decay_extrap_B*x_temp)
            out2[above_upper_bound] = self.slope_limit + self.decay_extrap_B*self.decay_extrap_A*np.exp(-self.decay_extrap_B*x_temp)
        return out1, out2

    def distance(self,other_func):
        '''
        The distance between two instances of the class is the largest difference
        in x values or y values.  If the instances have different numbers of gridpoints,
        then the difference in the number of gridpoints is returned instead.
        '''
        other_class = other_func.__class__.__name__
        if other_class is not 'LinearInterp':
            return 1000000
        xA = self.function._data[0]
        xB = other_func.function._data[0]
        yA = self.function._data[1]
        yB = other_func.function._data[1]
        if (xA.size == xB.size):
            x_dist = np.max(np.abs(xA - xB))
            y_dist = np.max(np.abs(yA - yB))
            dist = max(x_dist,y_dist)
        else:
            dist = np.abs(xA.size - xB.size)
        return dist
        
        
class BilinearInterp(HARKinterpolator2D):
    '''
    Bilinear full (or tensor) grid interpolation of a function f(x,y).
    '''
    def __init__(self,f_values,x_list,y_list,xSearchFunc=None,ySearchFunc=None):
        '''
        Parameters:
        -------------
        f_values : numpy.array
            An array of size (x_n,y_n) such that f_values[i,j] = f(x_list[i],y_list[j])
        x_list : numpy.array
            An array of x values, with length designated x_n.
        y_list : numpy.array
            An array of y values, with length designated y_n.
        xSearchFunc : function
            An optional function that returns the reference location for x values:
            indices = xSearchFunc(x_list,x).  Default is np.searchsorted
        ySearchFunc : function
            An optional function that returns the reference location for y values:
            indices = ySearchFunc(y_list,y).  Default is np.searchsorted
        '''
        self.f_values = f_values
        self.x_list = x_list
        self.y_list = y_list
        self.x_n = x_list.size
        self.y_n = y_list.size
        if xSearchFunc is None:
            xSearchFunc = np.searchsorted
        if ySearchFunc is None:
            ySearchFunc = np.searchsorted
        self.xSearchFunc = xSearchFunc
        self.ySearchFunc = ySearchFunc
        
    def _evaluate(self,x,y):
        if _isscalar(x):
            x_pos = max(min(self.xSearchFunc(self.x_list,x),self.x_n-1),1)
            y_pos = max(min(self.ySearchFunc(self.y_list,y),self.y_n-1),1)
        else:
            x_pos = self.xSearchFunc(self.x_list,x)
            x_pos[x_pos < 1] = 1
            x_pos[x_pos > self.x_n-1] = self.x_n-1
            y_pos = self.ySearchFunc(self.y_list,y)
            y_pos[y_pos < 1] = 1
            y_pos[y_pos > self.y_n-1] = self.y_n-1
        alpha = (x - self.x_list[x_pos-1])/(self.x_list[x_pos] - self.x_list[x_pos-1])
        beta = (y - self.y_list[y_pos-1])/(self.y_list[y_pos] - self.y_list[y_pos-1])
        f = (
             (1-alpha)*(1-beta)*self.f_values[x_pos-1,y_pos-1]
          +  (1-alpha)*beta*self.f_values[x_pos-1,y_pos]
          +  alpha*(1-beta)*self.f_values[x_pos,y_pos-1]
          +  alpha*beta*self.f_values[x_pos,y_pos])
        return f
        
    def _derX(self,x,y):
        if _isscalar(x):
            x_pos = max(min(self.xSearchFunc(self.x_list,x),self.x_n-1),1)
            y_pos = max(min(self.ySearchFunc(self.y_list,y),self.y_n-1),1)
        else:
            x_pos = self.xSearchFunc(self.x_list,x)
            x_pos[x_pos < 1] = 1
            x_pos[x_pos > self.x_n-1] = self.x_n-1
            y_pos = self.ySearchFunc(self.y_list,y)
            y_pos[y_pos < 1] = 1
            y_pos[y_pos > self.y_n-1] = self.y_n-1
        beta = (y - self.y_list[y_pos-1])/(self.y_list[y_pos] - self.y_list[y_pos-1])
        dfdx = (
              ((1-beta)*self.f_values[x_pos,y_pos-1]
            +  beta*self.f_values[x_pos,y_pos]) -
              ((1-beta)*self.f_values[x_pos-1,y_pos-1]
            +  beta*self.f_values[x_pos-1,y_pos]))/(self.x_list[x_pos] - self.x_list[x_pos-1])
        return dfdx
        
    def _derY(self,x,y):
        if _isscalar(x):
            x_pos = max(min(self.xSearchFunc(self.x_list,x),self.x_n-1),1)
            y_pos = max(min(self.ySearchFunc(self.y_list,y),self.y_n-1),1)
        else:
            x_pos = self.xSearchFunc(self.x_list,x)
            x_pos[x_pos < 1] = 1
            x_pos[x_pos > self.x_n-1] = self.x_n-1
            y_pos = self.ySearchFunc(self.y_list,y)
            y_pos[y_pos < 1] = 1
            y_pos[y_pos > self.y_n-1] = self.y_n-1
        alpha = (x - self.x_list[x_pos-1])/(self.x_list[x_pos] - self.x_list[x_pos-1])
        dfdy = (
              ((1-alpha)*self.f_values[x_pos-1,y_pos]
            +  alpha*self.f_values[x_pos,y_pos]) -
              ((1-alpha)*self.f_values[x_pos-1,y_pos]
            +  alpha*self.f_values[x_pos-1,y_pos-1]))/(self.y_list[y_pos] - self.y_list[y_pos-1])
        return dfdy
        
    def distance(self,other):
        other_class = other.__class__.__name__
        if other_class is not 'BilinearInterp':
            return 1000000
        if (self.x_n != other.x_n) or (self.y_n != other.y_n):
            return max(abs(self.x_n - other.x_n), abs(self.y_n - other.y_n))
        f_dist = np.max(np.abs(self.f_values - other.f_values))
        x_dist = np.max(np.abs(self.x_list - other.x_list))
        y_dist = np.max(np.abs(self.y_list - other.y_list))
        return max(f_dist,x_dist,y_dist)



class TrilinearInterp(HARKinterpolator3D):
    '''
    Trilinear full (or tensor) grid interpolation of a function f(x,y,z).
    '''
    def __init__(self,f_values,x_list,y_list,z_list,xSearchFunc=None,ySearchFunc=None,zSearchFunc=None):
        '''
        Parameters:
        -------------
        f_values : numpy.array
            An array of size (x_n,y_n,z_n) such that f_values[i,j,k] = f(x_list[i],y_list[j],z_list[k])
        x_list : numpy.array
            An array of x values, with length designated x_n.
        y_list : numpy.array
            An array of y values, with length designated y_n.
        z_list : numpy.array
            An array of z values, with length designated z_n.
        xSearchFunc : function
            An optional function that returns the reference location for x values:
            indices = xSearchFunc(x_list,x).  Default is np.searchsorted
        ySearchFunc : function
            An optional function that returns the reference location for y values:
            indices = ySearchFunc(y_list,y).  Default is np.searchsorted
        zSearchFunc : function
            An optional function that returns the reference location for z values:
            indices = zSearchFunc(z_list,z).  Default is np.searchsorted
        '''
        self.f_values = f_values
        self.x_list = x_list
        self.y_list = y_list
        self.z_list = z_list
        self.x_n = x_list.size
        self.y_n = y_list.size
        self.z_n = z_list.size
        if xSearchFunc is None:
            xSearchFunc = np.searchsorted
        if ySearchFunc is None:
            ySearchFunc = np.searchsorted
        if zSearchFunc is None:
            zSearchFunc = np.searchsorted
        self.xSearchFunc = xSearchFunc
        self.ySearchFunc = ySearchFunc
        self.zSearchFunc = zSearchFunc
        
    def _evaluate(self,x,y,z):
        if _isscalar(x):
            x_pos = max(min(self.xSearchFunc(self.x_list,x),self.x_n-1),1)
            y_pos = max(min(self.ySearchFunc(self.y_list,y),self.y_n-1),1)
            z_pos = max(min(self.zSearchFunc(self.z_list,z),self.z_n-1),1)
        else:
            x_pos = self.xSearchFunc(self.x_list,x)
            x_pos[x_pos < 1] = 1
            x_pos[x_pos > self.x_n-1] = self.x_n-1
            y_pos = self.ySearchFunc(self.y_list,y)
            y_pos[y_pos < 1] = 1
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            z_pos = self.zSearchFunc(self.z_list,z)
            z_pos[z_pos < 1] = 1
            z_pos[z_pos > self.z_n-1] = self.z_n-1
        alpha = (x - self.x_list[x_pos-1])/(self.x_list[x_pos] - self.x_list[x_pos-1])
        beta = (y - self.y_list[y_pos-1])/(self.y_list[y_pos] - self.y_list[y_pos-1])
        gamma = (z - self.z_list[z_pos-1])/(self.z_list[z_pos] - self.z_list[z_pos-1])
        f = (
             (1-alpha)*(1-beta)*(1-gamma)*self.f_values[x_pos-1,y_pos-1,z_pos-1]
          +  (1-alpha)*(1-beta)*gamma*self.f_values[x_pos-1,y_pos-1,z_pos]
          +  (1-alpha)*beta*(1-gamma)*self.f_values[x_pos-1,y_pos,z_pos-1]
          +  (1-alpha)*beta*gamma*self.f_values[x_pos-1,y_pos,z_pos]
          +  alpha*(1-beta)*(1-gamma)*self.f_values[x_pos,y_pos-1,z_pos-1]
          +  alpha*(1-beta)*gamma*self.f_values[x_pos,y_pos-1,z_pos]
          +  alpha*beta*(1-gamma)*self.f_values[x_pos,y_pos,z_pos-1]
          +  alpha*beta*gamma*self.f_values[x_pos,y_pos,z_pos])
        return f
        
    def _derX(self,x,y,z):
        if _isscalar(x):
            x_pos = max(min(self.xSearchFunc(self.x_list,x),self.x_n-1),1)
            y_pos = max(min(self.ySearchFunc(self.y_list,y),self.y_n-1),1)
            z_pos = max(min(self.zSearchFunc(self.z_list,z),self.z_n-1),1)
        else:
            x_pos = self.xSearchFunc(self.x_list,x)
            x_pos[x_pos < 1] = 1
            x_pos[x_pos > self.x_n-1] = self.x_n-1
            y_pos = self.ySearchFunc(self.y_list,y)
            y_pos[y_pos < 1] = 1
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            z_pos = self.zSearchFunc(self.z_list,z)
            z_pos[z_pos < 1] = 1
            z_pos[z_pos > self.z_n-1] = self.z_n-1
        beta = (y - self.y_list[y_pos-1])/(self.y_list[y_pos] - self.y_list[y_pos-1])
        gamma = (z - self.z_list[z_pos-1])/(self.z_list[z_pos] - self.z_list[z_pos-1])
        dfdx = (
           (  (1-beta)*(1-gamma)*self.f_values[x_pos,y_pos-1,z_pos-1]
           +  (1-beta)*gamma*self.f_values[x_pos,y_pos-1,z_pos]
           +  beta*(1-gamma)*self.f_values[x_pos,y_pos,z_pos-1]
           +  beta*gamma*self.f_values[x_pos,y_pos,z_pos]) -
           (  (1-beta)*(1-gamma)*self.f_values[x_pos-1,y_pos-1,z_pos-1]
           +  (1-beta)*gamma*self.f_values[x_pos-1,y_pos-1,z_pos]
           +  beta*(1-gamma)*self.f_values[x_pos-1,y_pos,z_pos-1]
           +  beta*gamma*self.f_values[x_pos-1,y_pos,z_pos]))/(self.x_list[x_pos] - self.x_list[x_pos-1])
        return dfdx
        
    def _derY(self,x,y,z):
        if _isscalar(x):
            x_pos = max(min(self.xSearchFunc(self.x_list,x),self.x_n-1),1)
            y_pos = max(min(self.ySearchFunc(self.y_list,y),self.y_n-1),1)
            z_pos = max(min(self.zSearchFunc(self.z_list,z),self.z_n-1),1)
        else:
            x_pos = self.xSearchFunc(self.x_list,x)
            x_pos[x_pos < 1] = 1
            x_pos[x_pos > self.x_n-1] = self.x_n-1
            y_pos = self.ySearchFunc(self.y_list,y)
            y_pos[y_pos < 1] = 1
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            z_pos = self.zSearchFunc(self.z_list,z)
            z_pos[z_pos < 1] = 1
            z_pos[z_pos > self.z_n-1] = self.z_n-1
        alpha = (x - self.x_list[x_pos-1])/(self.x_list[x_pos] - self.x_list[x_pos-1])
        gamma = (z - self.z_list[z_pos-1])/(self.z_list[z_pos] - self.z_list[z_pos-1])
        dfdy = (
           (  (1-alpha)*(1-gamma)*self.f_values[x_pos-1,y_pos,z_pos-1]
           +  (1-alpha)*gamma*self.f_values[x_pos-1,y_pos,z_pos]
           +  alpha*(1-gamma)*self.f_values[x_pos,y_pos,z_pos-1]
           +  alpha*gamma*self.f_values[x_pos,y_pos,z_pos]) -
           (  (1-alpha)*(1-gamma)*self.f_values[x_pos-1,y_pos-1,z_pos-1]
           +  (1-alpha)*gamma*self.f_values[x_pos-1,y_pos-1,z_pos]
           +  alpha*(1-gamma)*self.f_values[x_pos,y_pos-1,z_pos-1]
           +  alpha*gamma*self.f_values[x_pos,y_pos-1,z_pos]))/(self.y_list[y_pos] - self.y_list[y_pos-1])
        return dfdy
        
    def _derZ(self,x,y,z):
        if _isscalar(x):
            x_pos = max(min(self.xSearchFunc(self.x_list,x),self.x_n-1),1)
            y_pos = max(min(self.ySearchFunc(self.y_list,y),self.y_n-1),1)
            z_pos = max(min(self.zSearchFunc(self.z_list,z),self.z_n-1),1)
        else:
            x_pos = self.xSearchFunc(self.x_list,x)
            x_pos[x_pos < 1] = 1
            x_pos[x_pos > self.x_n-1] = self.x_n-1
            y_pos = self.ySearchFunc(self.y_list,y)
            y_pos[y_pos < 1] = 1
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            z_pos = self.zSearchFunc(self.z_list,z)
            z_pos[z_pos < 1] = 1
            z_pos[z_pos > self.z_n-1] = self.z_n-1
        alpha = (x - self.x_list[x_pos-1])/(self.x_list[x_pos] - self.x_list[x_pos-1])
        beta = (y - self.y_list[y_pos-1])/(self.y_list[y_pos] - self.y_list[y_pos-1])
        dfdz = (
           (  (1-alpha)*(1-beta)*self.f_values[x_pos-1,y_pos-1,z_pos]
           +  (1-alpha)*beta*self.f_values[x_pos-1,y_pos,z_pos]
           +  alpha*(1-beta)*self.f_values[x_pos,y_pos-1,z_pos]
           +  alpha*beta*self.f_values[x_pos,y_pos,z_pos]) -
           (  (1-alpha)*(1-beta)*self.f_values[x_pos-1,y_pos-1,z_pos-1]
           +  (1-alpha)*beta*self.f_values[x_pos-1,y_pos,z_pos-1]
           +  alpha*(1-beta)*self.f_values[x_pos,y_pos-1,z_pos-1]
           +  alpha*beta*self.f_values[x_pos,y_pos,z_pos-1]))/(self.z_list[z_pos] - self.z_list[z_pos-1])
        return dfdz
        
    def distance(self,other):
        other_class = other.__class__.__name__
        if other_class is not 'TrilinearInterp':
            return 1000000
        if (self.x_n != other.x_n) or (self.y_n != other.y_n) or (self.z_n != other.z_n):
            return max(abs(self.x_n - other.x_n), abs(self.y_n - other.y_n), abs(self.z_n - other.z_n))
        f_dist = np.max(np.abs(self.f_values - other.f_values))
        x_dist = np.max(np.abs(self.x_list - other.x_list))
        y_dist = np.max(np.abs(self.y_list - other.y_list))
        z_dist = np.max(np.abs(self.z_list - other.z_list))
        return max(f_dist,x_dist,y_dist,z_dist)
        


class QuadlinearInterp(HARKinterpolator4D):
    '''
    Quadlinear full (or tensor) grid interpolation of a function f(w,x,y,z).
    '''
    def __init__(self,f_values,w_list,x_list,y_list,z_list,wSearchFunc=None,xSearchFunc=None,ySearchFunc=None,zSearchFunc=None):
        '''
        Parameters:
        -------------
        f_values : numpy.array
            An array of size (w_n,x_n,y_n,z_n) such that f_values[i,j,k,l] = f(w_list[i],x_list[j],y_list[k],z_list[l])
        w_list : numpy.array
            An array of x values, with length designated w_n.
        x_list : numpy.array
            An array of x values, with length designated x_n.
        y_list : numpy.array
            An array of y values, with length designated y_n.
        z_list : numpy.array
            An array of z values, with length designated z_n.
        wSearchFunc : function
            An optional function that returns the reference location for w values:
            indices = wSearchFunc(w_list,w).  Default is np.searchsorted
        xSearchFunc : function
            An optional function that returns the reference location for x values:
            indices = xSearchFunc(x_list,x).  Default is np.searchsorted
        ySearchFunc : function
            An optional function that returns the reference location for y values:
            indices = ySearchFunc(y_list,y).  Default is np.searchsorted
        zSearchFunc : function
            An optional function that returns the reference location for z values:
            indices = zSearchFunc(z_list,z).  Default is np.searchsorted
        '''
        self.f_values = f_values
        self.w_list = w_list
        self.x_list = x_list
        self.y_list = y_list
        self.z_list = z_list
        self.w_n = w_list.size
        self.x_n = x_list.size
        self.y_n = y_list.size
        self.z_n = z_list.size
        if wSearchFunc is None:
            wSearchFunc = np.searchsorted
        if xSearchFunc is None:
            xSearchFunc = np.searchsorted
        if ySearchFunc is None:
            ySearchFunc = np.searchsorted
        if zSearchFunc is None:
            zSearchFunc = np.searchsorted
        self.wSearchFunc = wSearchFunc
        self.xSearchFunc = xSearchFunc
        self.ySearchFunc = ySearchFunc
        self.zSearchFunc = zSearchFunc
        
    def _evaluate(self,w,x,y,z):
        if _isscalar(w):
            w_pos = max(min(self.wSearchFunc(self.w_list,w),self.w_n-1),1)
            x_pos = max(min(self.xSearchFunc(self.x_list,x),self.x_n-1),1)
            y_pos = max(min(self.ySearchFunc(self.y_list,y),self.y_n-1),1)
            z_pos = max(min(self.zSearchFunc(self.z_list,z),self.z_n-1),1)
        else:
            w_pos = self.wSearchFunc(self.w_list,w)
            w_pos[w_pos < 1] = 1
            w_pos[w_pos > self.w_n-1] = self.w_n-1
            x_pos = self.xSearchFunc(self.x_list,x)
            x_pos[x_pos < 1] = 1
            x_pos[x_pos > self.x_n-1] = self.x_n-1
            y_pos = self.ySearchFunc(self.y_list,y)
            y_pos[y_pos < 1] = 1
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            z_pos = self.zSearchFunc(self.z_list,z)
            z_pos[z_pos < 1] = 1
            z_pos[z_pos > self.z_n-1] = self.z_n-1
        i = w_pos # for convenience
        j = x_pos
        k = y_pos
        l = z_pos
        alpha = (w - self.w_list[i-1])/(self.w_list[i] - self.w_list[i-1])
        beta = (x - self.x_list[j-1])/(self.x_list[j] - self.x_list[j-1])
        gamma = (y - self.y_list[k-1])/(self.y_list[k] - self.y_list[k-1])
        delta = (z - self.z_list[l-1])/(self.z_list[l] - self.z_list[l-1])
        f = (
             (1-alpha)*((1-beta)*((1-gamma)*(1-delta)*self.f_values[i-1,j-1,k-1,l-1]
           + (1-gamma)*delta*self.f_values[i-1,j-1,k-1,l]
           + gamma*(1-delta)*self.f_values[i-1,j-1,k,l-1]
           + gamma*delta*self.f_values[i-1,j-1,k,l])
           + beta*((1-gamma)*(1-delta)*self.f_values[i-1,j,k-1,l-1]
           + (1-gamma)*delta*self.f_values[i-1,j,k-1,l]
           + gamma*(1-delta)*self.f_values[i-1,j,k,l-1]
           + gamma*delta*self.f_values[i-1,j,k,l]))
           + alpha*((1-beta)*((1-gamma)*(1-delta)*self.f_values[i,j-1,k-1,l-1]
           + (1-gamma)*delta*self.f_values[i,j-1,k-1,l]
           + gamma*(1-delta)*self.f_values[i,j-1,k,l-1]
           + gamma*delta*self.f_values[i,j-1,k,l])
           + beta*((1-gamma)*(1-delta)*self.f_values[i,j,k-1,l-1]
           + (1-gamma)*delta*self.f_values[i,j,k-1,l]
           + gamma*(1-delta)*self.f_values[i,j,k,l-1]
           + gamma*delta*self.f_values[i,j,k,l])))       
        return f
        
    def _derW(self,w,x,y,z):
        if _isscalar(w):
            w_pos = max(min(self.wSearchFunc(self.w_list,w),self.w_n-1),1)
            x_pos = max(min(self.xSearchFunc(self.x_list,x),self.x_n-1),1)
            y_pos = max(min(self.ySearchFunc(self.y_list,y),self.y_n-1),1)
            z_pos = max(min(self.zSearchFunc(self.z_list,z),self.z_n-1),1)
        else:
            w_pos = self.wSearchFunc(self.w_list,w)
            w_pos[w_pos < 1] = 1
            w_pos[w_pos > self.w_n-1] = self.w_n-1
            x_pos = self.xSearchFunc(self.x_list,x)
            x_pos[x_pos < 1] = 1
            x_pos[x_pos > self.x_n-1] = self.x_n-1
            y_pos = self.ySearchFunc(self.y_list,y)
            y_pos[y_pos < 1] = 1
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            z_pos = self.zSearchFunc(self.z_list,z)
            z_pos[z_pos < 1] = 1
            z_pos[z_pos > self.z_n-1] = self.z_n-1
        i = w_pos # for convenience
        j = x_pos
        k = y_pos
        l = z_pos
        beta = (x - self.x_list[j-1])/(self.x_list[j] - self.x_list[j-1])
        gamma = (y - self.y_list[k-1])/(self.y_list[k] - self.y_list[k-1])
        delta = (z - self.z_list[l-1])/(self.z_list[l] - self.z_list[l-1])
        dfdw = (
          (  (1-beta)*(1-gamma)*(1-delta)*self.f_values[i,j-1,k-1,l-1]
           + (1-beta)*(1-gamma)*delta*self.f_values[i,j-1,k-1,l]
           + (1-beta)*gamma*(1-delta)*self.f_values[i,j-1,k,l-1]
           + (1-beta)*gamma*delta*self.f_values[i,j-1,k,l]
           + beta*(1-gamma)*(1-delta)*self.f_values[i,j,k-1,l-1]
           + beta*(1-gamma)*delta*self.f_values[i,j,k-1,l]
           + beta*gamma*(1-delta)*self.f_values[i,j,k,l-1]
           + beta*gamma*delta*self.f_values[i,j,k,l] ) - 
          (  (1-beta)*(1-gamma)*(1-delta)*self.f_values[i-1,j-1,k-1,l-1]
           + (1-beta)*(1-gamma)*delta*self.f_values[i-1,j-1,k-1,l]
           + (1-beta)*gamma*(1-delta)*self.f_values[i-1,j-1,k,l-1]
           + (1-beta)*gamma*delta*self.f_values[i-1,j-1,k,l]
           + beta*(1-gamma)*(1-delta)*self.f_values[i-1,j,k-1,l-1]
           + beta*(1-gamma)*delta*self.f_values[i-1,j,k-1,l]
           + beta*gamma*(1-delta)*self.f_values[i-1,j,k,l-1]
           + beta*gamma*delta*self.f_values[i-1,j,k,l] )
              )/(self.w_list[i] - self.w_list[i-1])
        return dfdw
        
    def _derX(self,w,x,y,z):
        if _isscalar(w):
            w_pos = max(min(self.wSearchFunc(self.w_list,w),self.w_n-1),1)
            x_pos = max(min(self.xSearchFunc(self.x_list,x),self.x_n-1),1)
            y_pos = max(min(self.ySearchFunc(self.y_list,y),self.y_n-1),1)
            z_pos = max(min(self.zSearchFunc(self.z_list,z),self.z_n-1),1)
        else:
            w_pos = self.wSearchFunc(self.w_list,w)
            w_pos[w_pos < 1] = 1
            w_pos[w_pos > self.w_n-1] = self.w_n-1
            x_pos = self.xSearchFunc(self.x_list,x)
            x_pos[x_pos < 1] = 1
            x_pos[x_pos > self.x_n-1] = self.x_n-1
            y_pos = self.ySearchFunc(self.y_list,y)
            y_pos[y_pos < 1] = 1
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            z_pos = self.zSearchFunc(self.z_list,z)
            z_pos[z_pos < 1] = 1
            z_pos[z_pos > self.z_n-1] = self.z_n-1
        i = w_pos # for convenience
        j = x_pos
        k = y_pos
        l = z_pos
        alpha = (w - self.w_list[i-1])/(self.w_list[i] - self.w_list[i-1])
        gamma = (y - self.y_list[k-1])/(self.y_list[k] - self.y_list[k-1])
        delta = (z - self.z_list[l-1])/(self.z_list[l] - self.z_list[l-1])
        dfdx = (
          (  (1-alpha)*(1-gamma)*(1-delta)*self.f_values[i-1,j,k-1,l-1]
           + (1-alpha)*(1-gamma)*delta*self.f_values[i-1,j,k-1,l]
           + (1-alpha)*gamma*(1-delta)*self.f_values[i-1,j,k,l-1]
           + (1-alpha)*gamma*delta*self.f_values[i-1,j,k,l]
           + alpha*(1-gamma)*(1-delta)*self.f_values[i,j,k-1,l-1]
           + alpha*(1-gamma)*delta*self.f_values[i,j,k-1,l]
           + alpha*gamma*(1-delta)*self.f_values[i,j,k,l-1]
           + alpha*gamma*delta*self.f_values[i,j,k,l] ) - 
          (  (1-alpha)*(1-gamma)*(1-delta)*self.f_values[i-1,j-1,k-1,l-1]
           + (1-alpha)*(1-gamma)*delta*self.f_values[i-1,j-1,k-1,l]
           + (1-alpha)*gamma*(1-delta)*self.f_values[i-1,j-1,k,l-1]
           + (1-alpha)*gamma*delta*self.f_values[i-1,j-1,k,l]
           + alpha*(1-gamma)*(1-delta)*self.f_values[i,j-1,k-1,l-1]
           + alpha*(1-gamma)*delta*self.f_values[i,j-1,k-1,l]
           + alpha*gamma*(1-delta)*self.f_values[i,j-1,k,l-1]
           + alpha*gamma*delta*self.f_values[i,j-1,k,l] )
              )/(self.x_list[j] - self.x_list[j-1])
        return dfdx
        
    def _derY(self,w,x,y,z):
        if _isscalar(w):
            w_pos = max(min(self.wSearchFunc(self.w_list,w),self.w_n-1),1)
            x_pos = max(min(self.xSearchFunc(self.x_list,x),self.x_n-1),1)
            y_pos = max(min(self.ySearchFunc(self.y_list,y),self.y_n-1),1)
            z_pos = max(min(self.zSearchFunc(self.z_list,z),self.z_n-1),1)
        else:
            w_pos = self.wSearchFunc(self.w_list,w)
            w_pos[w_pos < 1] = 1
            w_pos[w_pos > self.w_n-1] = self.w_n-1
            x_pos = self.xSearchFunc(self.x_list,x)
            x_pos[x_pos < 1] = 1
            x_pos[x_pos > self.x_n-1] = self.x_n-1
            y_pos = self.ySearchFunc(self.y_list,y)
            y_pos[y_pos < 1] = 1
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            z_pos = self.zSearchFunc(self.z_list,z)
            z_pos[z_pos < 1] = 1
            z_pos[z_pos > self.z_n-1] = self.z_n-1
        i = w_pos # for convenience
        j = x_pos
        k = y_pos
        l = z_pos
        alpha = (w - self.w_list[i-1])/(self.w_list[i] - self.w_list[i-1])
        beta = (x - self.x_list[j-1])/(self.x_list[j] - self.x_list[j-1])
        delta = (z - self.z_list[l-1])/(self.z_list[l] - self.z_list[l-1])
        dfdy = (
          (  (1-alpha)*(1-beta)*(1-delta)*self.f_values[i-1,j-1,k,l-1]
           + (1-alpha)*(1-beta)*delta*self.f_values[i-1,j-1,k,l]
           + (1-alpha)*beta*(1-delta)*self.f_values[i-1,j,k,l-1]
           + (1-alpha)*beta*delta*self.f_values[i-1,j,k,l]
           + alpha*(1-beta)*(1-delta)*self.f_values[i,j-1,k,l-1]
           + alpha*(1-beta)*delta*self.f_values[i,j-1,k,l]
           + alpha*beta*(1-delta)*self.f_values[i,j,k,l-1]
           + alpha*beta*delta*self.f_values[i,j,k,l] ) - 
          (  (1-alpha)*(1-beta)*(1-delta)*self.f_values[i-1,j-1,k-1,l-1]
           + (1-alpha)*(1-beta)*delta*self.f_values[i-1,j-1,k-1,l]
           + (1-alpha)*beta*(1-delta)*self.f_values[i-1,j,k-1,l-1]
           + (1-alpha)*beta*delta*self.f_values[i-1,j,k-1,l]
           + alpha*(1-beta)*(1-delta)*self.f_values[i,j-1,k-1,l-1]
           + alpha*(1-beta)*delta*self.f_values[i,j-1,k-1,l]
           + alpha*beta*(1-delta)*self.f_values[i,j,k-1,l-1]
           + alpha*beta*delta*self.f_values[i,j,k-1,l] )
              )/(self.y_list[k] - self.y_list[k-1])
        return dfdy
        
    def _derZ(self,w,x,y,z):
        if _isscalar(w):
            w_pos = max(min(self.wSearchFunc(self.w_list,w),self.w_n-1),1)
            x_pos = max(min(self.xSearchFunc(self.x_list,x),self.x_n-1),1)
            y_pos = max(min(self.ySearchFunc(self.y_list,y),self.y_n-1),1)
            z_pos = max(min(self.zSearchFunc(self.z_list,z),self.z_n-1),1)
        else:
            w_pos = self.wSearchFunc(self.w_list,w)
            w_pos[w_pos < 1] = 1
            w_pos[w_pos > self.w_n-1] = self.w_n-1
            x_pos = self.xSearchFunc(self.x_list,x)
            x_pos[x_pos < 1] = 1
            x_pos[x_pos > self.x_n-1] = self.x_n-1
            y_pos = self.ySearchFunc(self.y_list,y)
            y_pos[y_pos < 1] = 1
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            z_pos = self.zSearchFunc(self.z_list,z)
            z_pos[z_pos < 1] = 1
            z_pos[z_pos > self.z_n-1] = self.z_n-1
        i = w_pos # for convenience
        j = x_pos
        k = y_pos
        l = z_pos
        alpha = (w - self.w_list[i-1])/(self.w_list[i] - self.w_list[i-1])
        beta = (x - self.x_list[j-1])/(self.x_list[j] - self.x_list[j-1])
        gamma = (y - self.y_list[k-1])/(self.y_list[k] - self.y_list[k-1])
        dfdz = (
          (  (1-alpha)*(1-beta)*(1-gamma)*self.f_values[i-1,j-1,k-1,l]
           + (1-alpha)*(1-beta)*gamma*self.f_values[i-1,j-1,k,l]
           + (1-alpha)*beta*(1-gamma)*self.f_values[i-1,j,k-1,l]
           + (1-alpha)*beta*gamma*self.f_values[i-1,j,k,l]
           + alpha*(1-beta)*(1-gamma)*self.f_values[i,j-1,k-1,l]
           + alpha*(1-beta)*gamma*self.f_values[i,j-1,k,l]
           + alpha*beta*(1-gamma)*self.f_values[i,j,k-1,l]
           + alpha*beta*gamma*self.f_values[i,j,k,l] ) - 
          (  (1-alpha)*(1-beta)*(1-gamma)*self.f_values[i-1,j-1,k-1,l-1]
           + (1-alpha)*(1-beta)*gamma*self.f_values[i-1,j-1,k,l-1]
           + (1-alpha)*beta*(1-gamma)*self.f_values[i-1,j,k-1,l-1]
           + (1-alpha)*beta*gamma*self.f_values[i-1,j,k,l-1]
           + alpha*(1-beta)*(1-gamma)*self.f_values[i,j-1,k-1,l-1]
           + alpha*(1-beta)*gamma*self.f_values[i,j-1,k,l-1]
           + alpha*beta*(1-gamma)*self.f_values[i,j,k-1,l-1]
           + alpha*beta*gamma*self.f_values[i,j,k,l-1] )
              )/(self.z_list[l] - self.z_list[l-1])
        return dfdz
        
    def distance(self,other):
        other_class = other.__class__.__name__
        if other_class is not 'QuadlinearInterp':
            return 1000000
        if (self.w_n != other.w_n) or (self.x_n != other.x_n) or (self.y_n != other.y_n) or (self.z_n != other.z_n):
            return max(abs(self.w_n - other.w_n), abs(self.x_n - other.x_n), abs(self.y_n - other.y_n), abs(self.z_n - other.z_n))
        f_dist = np.max(np.abs(self.f_values - other.f_values))
        w_dist = np.max(np.abs(self.w_list - other.w_list))
        x_dist = np.max(np.abs(self.x_list - other.x_list))
        y_dist = np.max(np.abs(self.y_list - other.y_list))
        z_dist = np.max(np.abs(self.z_list - other.z_list))
        return max(f_dist,w_dist,x_dist,y_dist,z_dist)
        


class LowerEnvelope(HARKinterpolator1D):
    """
    The lower envelope of a finite set of 1D functions, each of which can be of
    any class that has the methods __call__, derivative, and eval_with_derivative.
    Generally: it combines HARKinterpolator1Ds. 
    """ 

    def __init__(self,*functions):
        '''
        Constructor method for the interpolation.
        '''
        self.functions = []
        for function in functions:
            self.functions.append(function)
        self.funcCount = len(self.functions)

    def _evaluate(self,x):
        '''
        Returns the level of the function at each value in x as the minimum among
        all of the functions.
        '''
        if _isscalar(x):
            y = np.nanmin([f(x) for f in self.functions])
        else:
            m = len(x)
            fx = np.zeros((m,self.funcCount))
            for j in range(self.funcCount):
                fx[:,j] = self.functions[j](x)
            y = np.nanmin(fx,axis=1)       
        return y

    def _der(self,x):
        '''
        Returns the first derivative of the function at each value in x.
        '''
        y,dydx = self.eval_with_derivative(x)
        return dydx  # Sadly, this is the fastest / most convenient way...

    def _evalAndDer(self,x):
        '''
        Returns the level and first derivative of the function at each value in x.
        '''
        m = len(x)
        fx = np.zeros((m,self.funcCount))
        for j in range(self.funcCount):
            fx[:,j] = self.functions[j](x)
        fx[np.isnan(fx)] = np.inf
        i = np.argmin(fx,axis=1)
        y = fx[np.arange(m),i]
        dydx = np.zeros_like(y)
        for j in range(self.funcCount):
            c = i == j
            dydx[c] = self.functions[j].derivative(x[c])
        return y,dydx
    
    def distance(self,function_other):
        '''
        Returns the distance between this instance and another instance of the class.
        The distance is the distance between the corresponding unconstrained functions.
        '''
        if self.funcCount == function_other.funcCount:
            dist_list = np.zeros(self.funcCount)
            for j in range(self.funcCount):
                dist_list[j] = self.functions[j].distance(function_other.functions[j])
            return np.max(dist_list)
        else:
            return 100*np.abs(self.funcCount - function_other.funcCount)


'''
A 2D interpolator that linearly interpolates among a list of 1D interpolators.
'''
class LinearInterpOnInterp1D(HARKinterpolator2D):
    
    def __init__(self,xInterpolators,y_values):
        '''
        Constructor for the class, generating an approximation to a function of
        the form f(x,y) using interpolations over f(x,y_0) for a fixed grid of
        y_0 values.
        
        Parameters:
        -------------
        xInterpolators : [HARKinterpolator1D]
            A list of 1D interpolations over the x variable.  The nth element of
            xInterpolators represents f(x,y_values[n]).
        y_values: numpy.array
            An array of y values equal in length to xInterpolators.
        '''
        self.xInterpolators = xInterpolators
        self.y_list = y_values
        self.y_n = y_values.size
        
    def _evaluate(self,x,y):
        if _isscalar(x):
            y_pos = max(min(np.searchsorted(self.y_list,y),self.y_n-1),1)
            alpha = (y - self.y_list[y_pos-1])/(self.y_list[y_pos] - self.y_list[y_pos-1])
            f = (1-alpha)*self.xInterpolators[y_pos-1](x) + alpha*self.xInterpolators[y_pos](x)
        else:
            m = len(x)
            y_pos = np.searchsorted(self.y_list,y)
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            y_pos[y_pos < 1] = 1
            f = np.zeros(m) + np.nan
            if y.size > 0:
                for i in xrange(1,self.y_n):
                    c = y_pos == i
                    if np.any(c):
                        alpha = (y[c] - self.y_list[i-1])/(self.y_list[i] - self.y_list[i-1])
                        f[c] = (1-alpha)*self.xInterpolators[i-1](x[c]) + alpha*self.xInterpolators[i](x[c]) 
        return f
        
    def _derX(self,x,y):
        if _isscalar(x):
            y_pos = max(min(np.searchsorted(self.y_list,y),self.y_n-1),1)
            alpha = (y - self.y_list[y_pos-1])/(self.y_list[y_pos] - self.y_list[y_pos-1])
            dfdx = (1-alpha)*self.xInterpolators[y_pos-1]._der(x) + alpha*self.xInterpolators[y_pos]._der(x)
        else:
            m = len(x)
            y_pos = np.searchsorted(self.y_list,y)
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            y_pos[y_pos < 1] = 1
            dfdx = np.zeros(m) + np.nan
            if y.size > 0:
                for i in xrange(1,self.y_n):
                    c = y_pos == i
                    if np.any(c):
                        alpha = (y[c] - self.y_list[i-1])/(self.y_list[i] - self.y_list[i-1])
                        dfdx[c] = (1-alpha)*self.xInterpolators[i-1]._der(x[c]) + alpha*self.xInterpolators[i]._der(x[c])
        return dfdx
        
    def _derY(self,x,y):
        if _isscalar(x):
            y_pos = max(min(np.searchsorted(self.y_list,y),self.y_n-1),1)
            dfdy = (self.xInterpolators[y_pos](x) - self.xInterpolators[y_pos-1](x))/(self.y_list[y_pos] - self.y_list[y_pos-1])
        else:
            m = len(x)
            y_pos = np.searchsorted(self.y_list,y)
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            y_pos[y_pos < 1] = 1
            dfdy = np.zeros(m) + np.nan
            if y.size > 0:
                for i in xrange(1,self.y_n):
                    c = y_pos == i
                    if np.any(c):
                        dfdy[c] = (self.xInterpolators[i](x[c]) - self.xInterpolators[i-1](x[c]))/(self.y_list[i] - self.y_list[i-1])
        return dfdy
        
    def distance(self,other_func):
        '''
        Distance between one instance and another is the greater of the maximum
        of the distance between corresponding xInterpolators and the maximum of
        the distance between corresponding y_list values.
        '''
        other_class = other_func.__class__.__name__
        if other_class is not 'LinearInterpOnInterp1D':
            return 1000000
        if self.y_n != other_func.y_n:
            return np.abs(self.y_n - other_func.y_n)
        x_dist_vec = np.zeros(self.y_n) + np.nan
        for j in range(self.y_n):
            x_dist_vec[j] = self.xInterpolators[j].distance(other_func.xInterpolators[j])
        y_dist_vec = np.abs(self.y_list - other_func.y_list)
        x_dist = np.max(x_dist_vec)
        y_dist = np.max(y_dist_vec)
        return max(x_dist,y_dist)
        
        
'''
A 3D interpolator that bilinearly interpolates among a list of lists of 1D interpolators.
'''
class BilinearInterpOnInterp1D(HARKinterpolator3D):
    
    def __init__(self,xInterpolators,y_values,z_values):
        '''
        Constructor for the class, generating an approximation to a function of
        the form f(x,y,z) using interpolations over f(x,y_0,z_0) for a fixed grid
        of y_0 and z_0 values.
        
        Parameters:
        -------------
        xInterpolators : [[HARKinterpolator1D]]
            A list of lists of 1D interpolations over the x variable.  The i,j-th
            element of xInterpolators represents f(x,y_values[i],z_values[j]).
        y_values: numpy.array
            An array of y values equal in length to xInterpolators.
        z_values: numpy.array
            An array of z values equal in length to xInterpolators[0].
        '''
        self.xInterpolators = xInterpolators
        self.y_list = y_values
        self.y_n = y_values.size
        self.z_list = z_values
        self.z_n = z_values.size
        
    def _evaluate(self,x,y,z):
        if _isscalar(x):
            y_pos = max(min(np.searchsorted(self.y_list,y),self.y_n-1),1)
            z_pos = max(min(np.searchsorted(self.z_list,z),self.z_n-1),1)
            alpha = (y - self.y_list[y_pos-1])/(self.y_list[y_pos] - self.y_list[y_pos-1])
            beta = (z - self.z_list[z_pos-1])/(self.z_list[z_pos] - self.z_list[z_pos-1])
            f = ((1-alpha)*(1-beta)*self.xInterpolators[y_pos-1][z_pos-1](x)
              + (1-alpha)*beta*self.xInterpolators[y_pos-1][z_pos](x)
              + alpha*(1-beta)*self.xInterpolators[y_pos][z_pos-1](x)
              + alpha*beta*self.xInterpolators[y_pos][z_pos](x))              
        else:
            m = len(x)
            y_pos = np.searchsorted(self.y_list,y)
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list,z)
            z_pos[z_pos > self.z_n-1] = self.z_n-1
            z_pos[z_pos < 1] = 1
            f = np.zeros(m) + np.nan
            for i in xrange(1,self.y_n):
                for j in xrange(1,self.z_n):
                    c = np.logical_and(i == y_pos, j == z_pos)
                    if np.any(c):
                        alpha = (y[c] - self.y_list[i-1])/(self.y_list[i] - self.y_list[i-1])
                        beta = (z[c] - self.z_list[j-1])/(self.z_list[j] - self.z_list[j-1])
                        f[c] = (
                          (1-alpha)*(1-beta)*self.xInterpolators[i-1][j-1](x[c])
                          + (1-alpha)*beta*self.xInterpolators[i-1][j](x[c])
                          + alpha*(1-beta)*self.xInterpolators[i][j-1](x[c])
                          + alpha*beta*self.xInterpolators[i][j](x[c]))
        return f
        
    def _derX(self,x,y,z):
        if _isscalar(x):
            y_pos = max(min(np.searchsorted(self.y_list,y),self.y_n-1),1)
            z_pos = max(min(np.searchsorted(self.z_list,z),self.z_n-1),1)
            alpha = (y - self.y_list[y_pos-1])/(self.y_list[y_pos] - self.y_list[y_pos-1])
            beta = (z - self.z_list[z_pos-1])/(self.z_list[z_pos] - self.z_list[z_pos-1])
            dfdx = ((1-alpha)*(1-beta)*self.xInterpolators[y_pos-1][z_pos-1]._der(x)
              + (1-alpha)*beta*self.xInterpolators[y_pos-1][z_pos]._der(x)
              + alpha*(1-beta)*self.xInterpolators[y_pos][z_pos-1]._der(x)
              + alpha*beta*self.xInterpolators[y_pos][z_pos]._der(x))              
        else:
            m = len(x)
            y_pos = np.searchsorted(self.y_list,y)
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list,z)
            z_pos[z_pos > self.z_n-1] = self.z_n-1
            z_pos[z_pos < 1] = 1
            dfdx = np.zeros(m) + np.nan
            for i in xrange(1,self.y_n):
                for j in xrange(1,self.z_n):
                    c = np.logical_and(i == y_pos, j == z_pos)
                    if np.any(c):
                        alpha = (y[c] - self.y_list[i-1])/(self.y_list[i] - self.y_list[i-1])
                        beta = (z[c] - self.z_list[j-1])/(self.z_list[j] - self.z_list[j-1])
                        dfdx[c] = (
                          (1-alpha)*(1-beta)*self.xInterpolators[i-1][j-1]._der(x[c])
                          + (1-alpha)*beta*self.xInterpolators[i-1][j]._der(x[c])
                          + alpha*(1-beta)*self.xInterpolators[i][j-1]._der(x[c])
                          + alpha*beta*self.xInterpolators[i][j]._der(x[c]))
        return dfdx
        
    def _derY(self,x,y,z):
        if _isscalar(x):
            y_pos = max(min(np.searchsorted(self.y_list,y),self.y_n-1),1)
            z_pos = max(min(np.searchsorted(self.z_list,z),self.z_n-1),1)
            beta = (z - self.z_list[z_pos-1])/(self.z_list[z_pos] - self.z_list[z_pos-1])
            dfdy = (((1-beta)*self.xInterpolators[y_pos][z_pos-1](x) + beta*self.xInterpolators[y_pos][z_pos](x))
                 -  ((1-beta)*self.xInterpolators[y_pos-1][z_pos-1](x) + beta*self.xInterpolators[y_pos-1][z_pos](x)))/(self.y_list[y_pos] - self.y_list[y_pos-1])
        else:
            m = len(x)
            y_pos = np.searchsorted(self.y_list,y)
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list,z)
            z_pos[z_pos > self.z_n-1] = self.z_n-1
            z_pos[z_pos < 1] = 1
            dfdy = np.zeros(m) + np.nan
            for i in xrange(1,self.y_n):
                for j in xrange(1,self.z_n):
                    c = np.logical_and(i == y_pos, j == z_pos)
                    if np.any(c):
                        beta = (z[c] - self.z_list[j-1])/(self.z_list[j] - self.z_list[j-1])
                        dfdy[c] = (((1-beta)*self.xInterpolators[i][j-1](x[c]) + beta*self.xInterpolators[i][j](x[c]))
                                -  ((1-beta)*self.xInterpolators[i-1][j-1](x[c]) + beta*self.xInterpolators[i-1][j](x[c])))/(self.y_list[i] - self.y_list[i-1])
        return dfdy
        
    def _derZ(self,x,y,z):
        if _isscalar(x):
            y_pos = max(min(np.searchsorted(self.y_list,y),self.y_n-1),1)
            z_pos = max(min(np.searchsorted(self.z_list,z),self.z_n-1),1)
            alpha = (y - self.y_list[y_pos-1])/(self.y_list[y_pos] - self.y_list[y_pos-1])
            dfdz = (((1-alpha)*self.xInterpolators[y_pos-1][z_pos](x) + alpha*self.xInterpolators[y_pos][z_pos](x))
                 -  ((1-alpha)*self.xInterpolators[y_pos-1][z_pos-1](x) + alpha*self.xInterpolators[y_pos][z_pos-1](x)))/(self.z_list[z_pos] - self.z_list[z_pos-1])
        else:
            m = len(x)
            y_pos = np.searchsorted(self.y_list,y)
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list,z)
            z_pos[z_pos > self.z_n-1] = self.z_n-1
            z_pos[z_pos < 1] = 1
            dfdz = np.zeros(m) + np.nan
            for i in xrange(1,self.y_n):
                for j in xrange(1,self.z_n):
                    c = np.logical_and(i == y_pos, j == z_pos)
                    if np.any(c):
                        alpha = (y[c] - self.y_list[i-1])/(self.y_list[i] - self.y_list[i-1])
                        dfdz[c] = (((1-alpha)*self.xInterpolators[i-1][j](x[c]) + alpha*self.xInterpolators[i][j](x[c]))
                                -  ((1-alpha)*self.xInterpolators[i-1][j-1](x[c]) + alpha*self.xInterpolators[i][j-1](x[c])))/(self.z_list[j] - self.z_list[j-1])
        return dfdz
        
    def distance(self,other_func):
        '''
        Distance between one instance and another is the greater of the maximum
        of the distance between corresponding xInterpolators and the maximum of
        the distance between corresponding y_list and z_list values.
        '''
        other_class = other_func.__class__.__name__
        if other_class is not 'BilinearInterpOnInterp1D':
            return 1000000
        if (self.y_n != other_func.y_n) or (self.z_n != other_func.z_n):
            return max(np.abs(self.y_n - other_func.y_n),np.abs(self.z_n - other_func.z_n))
        x_dist_matrix = np.zeros((self.y_n,self.z_n)) + np.nan
        for i in range(self.y_n):
            for j in range(self.z_n):
                x_dist_matrix[i,j] = self.xInterpolators[i][j].distance(other_func.xInterpolators[i][j])
        y_dist_vec = np.abs(self.y_list - other_func.y_list)
        z_dist_vec = np.abs(self.z_list - other_func.z_list)
        x_dist = np.max(x_dist_matrix)
        y_dist = np.max(y_dist_vec)
        z_dist = np.max(z_dist_vec)
        return max(x_dist,y_dist,z_dist)
        
             
'''
A 4D interpolator that trilinearly interpolates among a list of lists of 1D interpolators.
'''
class TrilinearInterpOnInterp1D(HARKinterpolator4D):
    
    def __init__(self,wInterpolators,x_values,y_values,z_values):
        '''
        Constructor for the class, generating an approximation to a function of
        the form f(w,x,y,z) using interpolations over f(w,x_0,y_0,z_0) for a fixed
        grid of y_0 and z_0 values.
        
        Parameters:
        -------------
        wInterpolators : [[[HARKinterpolator1D]]]
            A list of lists of lists of 1D interpolations over the x variable.
            The i,j,k-th element of wInterpolators represents f(w,x_values[i],y_values[j],z_values[k]).
        x_values: numpy.array
            An array of x values equal in length to wInterpolators.
        y_values: numpy.array
            An array of y values equal in length to wInterpolators[0].
        z_values: numpy.array
            An array of z values equal in length to wInterpolators[0][0]
        '''
        self.wInterpolators = wInterpolators
        self.x_list = x_values
        self.x_n = x_values.size
        self.y_list = y_values
        self.y_n = y_values.size
        self.z_list = z_values
        self.z_n = z_values.size

    def _evaluate(self,w,x,y,z):
        if _isscalar(w):
            x_pos = max(min(np.searchsorted(self.x_list,x),self.x_n-1),1)
            y_pos = max(min(np.searchsorted(self.y_list,y),self.y_n-1),1)
            z_pos = max(min(np.searchsorted(self.z_list,z),self.z_n-1),1)
            alpha = (x - self.x_list[x_pos-1])/(self.x_list[x_pos] - self.x_list[x_pos-1])
            beta = (y - self.y_list[y_pos-1])/(self.y_list[y_pos] - self.y_list[y_pos-1])
            gamma = (z - self.z_list[z_pos-1])/(self.z_list[z_pos] - self.z_list[z_pos-1])
            f = (
                (1-alpha)*(1-beta)*(1-gamma)*self.wInterpolators[x_pos-1][y_pos-1][z_pos-1](w)
              + (1-alpha)*(1-beta)*gamma*self.wInterpolators[x_pos-1][y_pos-1][z_pos](w)
              + (1-alpha)*beta*(1-gamma)*self.wInterpolators[x_pos-1][y_pos][z_pos-1](w)
              + (1-alpha)*beta*gamma*self.wInterpolators[x_pos-1][y_pos][z_pos](w)
              + alpha*(1-beta)*(1-gamma)*self.wInterpolators[x_pos][y_pos-1][z_pos-1](w)
              + alpha*(1-beta)*gamma*self.wInterpolators[x_pos][y_pos-1][z_pos](w)
              + alpha*beta*(1-gamma)*self.wInterpolators[x_pos][y_pos][z_pos-1](w)
              + alpha*beta*gamma*self.wInterpolators[x_pos][y_pos][z_pos](w))
        else:
            m = len(x)
            x_pos = np.searchsorted(self.x_list,x)
            x_pos[x_pos > self.x_n-1] = self.x_n-1
            y_pos = np.searchsorted(self.y_list,y)
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list,z)
            z_pos[z_pos > self.z_n-1] = self.z_n-1
            z_pos[z_pos < 1] = 1
            f = np.zeros(m) + np.nan
            for i in xrange(1,self.x_n):
                for j in xrange(1,self.y_n):
                    for k in xrange(1,self.z_n):
                        c = np.logical_and(np.logical_and(i == x_pos, j == y_pos),k == z_pos)
                        if np.any(c):
                            alpha = (x[c] - self.x_list[i-1])/(self.x_list[i] - self.x_list[i-1])
                            beta = (y[c] - self.y_list[j-1])/(self.y_list[j] - self.y_list[j-1])
                            gamma = (z[c] - self.z_list[k-1])/(self.z_list[k] - self.z_list[k-1])
                            f[c] = (
                                (1-alpha)*(1-beta)*(1-gamma)*self.wInterpolators[i-1][j-1][k-1](w[c])
                              + (1-alpha)*(1-beta)*gamma*self.wInterpolators[i-1][j-1][k](w[c])
                              + (1-alpha)*beta*(1-gamma)*self.wInterpolators[i-1][j][k-1](w[c])
                              + (1-alpha)*beta*gamma*self.wInterpolators[i-1][j][k](w[c])
                              + alpha*(1-beta)*(1-gamma)*self.wInterpolators[i][j-1][k-1](w[c])
                              + alpha*(1-beta)*gamma*self.wInterpolators[i][j-1][k](w[c])
                              + alpha*beta*(1-gamma)*self.wInterpolators[i][j][k-1](w[c])
                              + alpha*beta*gamma*self.wInterpolators[i][j][k](w[c]))
        return f
        
    def _derW(self,w,x,y,z):
        if _isscalar(w):
            x_pos = max(min(np.searchsorted(self.x_list,x),self.x_n-1),1)
            y_pos = max(min(np.searchsorted(self.y_list,y),self.y_n-1),1)
            z_pos = max(min(np.searchsorted(self.z_list,z),self.z_n-1),1)
            alpha = (x - self.x_list[x_pos-1])/(self.x_list[x_pos] - self.x_list[x_pos-1])
            beta = (y - self.y_list[y_pos-1])/(self.y_list[y_pos] - self.y_list[y_pos-1])
            gamma = (z - self.z_list[z_pos-1])/(self.z_list[z_pos] - self.z_list[z_pos-1])
            dfdw = (
                (1-alpha)*(1-beta)*(1-gamma)*self.wInterpolators[x_pos-1][y_pos-1][z_pos-1]._der(w)
              + (1-alpha)*(1-beta)*gamma*self.wInterpolators[x_pos-1][y_pos-1][z_pos]._der(w)
              + (1-alpha)*beta*(1-gamma)*self.wInterpolators[x_pos-1][y_pos][z_pos-1]._der(w)
              + (1-alpha)*beta*gamma*self.wInterpolators[x_pos-1][y_pos][z_pos]._der(w)
              + alpha*(1-beta)*(1-gamma)*self.wInterpolators[x_pos][y_pos-1][z_pos-1]._der(w)
              + alpha*(1-beta)*gamma*self.wInterpolators[x_pos][y_pos-1][z_pos]._der(w)
              + alpha*beta*(1-gamma)*self.wInterpolators[x_pos][y_pos][z_pos-1]._der(w)
              + alpha*beta*gamma*self.wInterpolators[x_pos][y_pos][z_pos]._der(w))
        else:
            m = len(x)
            x_pos = np.searchsorted(self.x_list,x)
            x_pos[x_pos > self.x_n-1] = self.x_n-1
            y_pos = np.searchsorted(self.y_list,y)
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list,z)
            z_pos[z_pos > self.z_n-1] = self.z_n-1
            z_pos[z_pos < 1] = 1
            dfdw = np.zeros(m) + np.nan
            for i in xrange(1,self.x_n):
                for j in xrange(1,self.y_n):
                    for k in xrange(1,self.z_n):
                        c = np.logical_and(np.logical_and(i == x_pos, j == y_pos),k == z_pos)
                        if np.any(c):
                            alpha = (x[c] - self.x_list[i-1])/(self.x_list[i] - self.x_list[i-1])
                            beta = (y[c] - self.y_list[j-1])/(self.y_list[j] - self.y_list[j-1])
                            gamma = (z[c] - self.z_list[k-1])/(self.z_list[k] - self.z_list[k-1])
                            dfdw[c] = (
                                (1-alpha)*(1-beta)*(1-gamma)*self.wInterpolators[i-1][j-1][k-1]._der(w[c])
                              + (1-alpha)*(1-beta)*gamma*self.wInterpolators[i-1][j-1][k]._der(w[c])
                              + (1-alpha)*beta*(1-gamma)*self.wInterpolators[i-1][j][k-1]._der(w[c])
                              + (1-alpha)*beta*gamma*self.wInterpolators[i-1][j][k]._der(w[c])
                              + alpha*(1-beta)*(1-gamma)*self.wInterpolators[i][j-1][k-1]._der(w[c])
                              + alpha*(1-beta)*gamma*self.wInterpolators[i][j-1][k]._der(w[c])
                              + alpha*beta*(1-gamma)*self.wInterpolators[i][j][k-1]._der(w[c])
                              + alpha*beta*gamma*self.wInterpolators[i][j][k]._der(w[c]))
        return dfdw
        
    def _derX(self,w,x,y,z):
        if _isscalar(w):
            x_pos = max(min(np.searchsorted(self.x_list,x),self.x_n-1),1)
            y_pos = max(min(np.searchsorted(self.y_list,y),self.y_n-1),1)
            z_pos = max(min(np.searchsorted(self.z_list,z),self.z_n-1),1)
            beta = (y - self.y_list[y_pos-1])/(self.y_list[y_pos] - self.y_list[y_pos-1])
            gamma = (z - self.z_list[z_pos-1])/(self.z_list[z_pos] - self.z_list[z_pos-1])
            dfdx = (
             ((1-beta)*(1-gamma)*self.wInterpolators[x_pos][y_pos-1][z_pos-1](w)
              + (1-beta)*gamma*self.wInterpolators[x_pos][y_pos-1][z_pos](w)
              + beta*(1-gamma)*self.wInterpolators[x_pos][y_pos][z_pos-1](w)
              + beta*gamma*self.wInterpolators[x_pos][y_pos][z_pos](w)) - 
              ((1-beta)*(1-gamma)*self.wInterpolators[x_pos-1][y_pos-1][z_pos-1](w)
              + (1-beta)*gamma*self.wInterpolators[x_pos-1][y_pos-1][z_pos](w)
              + beta*(1-gamma)*self.wInterpolators[x_pos-1][y_pos][z_pos-1](w)
              + beta*gamma*self.wInterpolators[x_pos-1][y_pos][z_pos](w)))/(self.x_list[x_pos] - self.x_list[x_pos-1])
        else:
            m = len(x)
            x_pos = np.searchsorted(self.x_list,x)
            x_pos[x_pos > self.x_n-1] = self.x_n-1
            y_pos = np.searchsorted(self.y_list,y)
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list,z)
            z_pos[z_pos > self.z_n-1] = self.z_n-1
            z_pos[z_pos < 1] = 1
            dfdx = np.zeros(m) + np.nan
            for i in xrange(1,self.x_n):
                for j in xrange(1,self.y_n):
                    for k in xrange(1,self.z_n):
                        c = np.logical_and(np.logical_and(i == x_pos, j == y_pos),k == z_pos)
                        if np.any(c):
                            beta = (y[c] - self.y_list[j-1])/(self.y_list[j] - self.y_list[j-1])
                            gamma = (z[c] - self.z_list[k-1])/(self.z_list[k] - self.z_list[k-1])
                            dfdx[c] = (
                              ((1-beta)*(1-gamma)*self.wInterpolators[i][j-1][k-1](w[c])
                            + (1-beta)*gamma*self.wInterpolators[i][j-1][k](w[c])
                            + beta*(1-gamma)*self.wInterpolators[i][j][k-1](w[c])
                            + beta*gamma*self.wInterpolators[i][j][k](w[c])) - 
                             ((1-beta)*(1-gamma)*self.wInterpolators[i-1][j-1][k-1](w[c])
                            + (1-beta)*gamma*self.wInterpolators[i-1][j-1][k](w[c])
                            + beta*(1-gamma)*self.wInterpolators[i-1][j][k-1](w[c])
                            + beta*gamma*self.wInterpolators[i-1][j][k](w[c])))/(self.x_list[i] - self.x_list[i-1])
        return dfdx
        
    def _derY(self,w,x,y,z):
        if _isscalar(w):
            x_pos = max(min(np.searchsorted(self.x_list,x),self.x_n-1),1)
            y_pos = max(min(np.searchsorted(self.y_list,y),self.y_n-1),1)
            z_pos = max(min(np.searchsorted(self.z_list,z),self.z_n-1),1)
            alpha = (x - self.x_list[x_pos-1])/(self.y_list[x_pos] - self.x_list[x_pos-1])
            gamma = (z - self.z_list[z_pos-1])/(self.z_list[z_pos] - self.z_list[z_pos-1])
            dfdy = (
             ((1-alpha)*(1-gamma)*self.wInterpolators[x_pos-1][y_pos][z_pos-1](w)
              + (1-alpha)*gamma*self.wInterpolators[x_pos-1][y_pos][z_pos](w)
              + alpha*(1-gamma)*self.wInterpolators[x_pos][y_pos][z_pos-1](w)
              + alpha*gamma*self.wInterpolators[x_pos][y_pos][z_pos](w)) - 
              ((1-alpha)*(1-gamma)*self.wInterpolators[x_pos-1][y_pos-1][z_pos-1](w)
              + (1-alpha)*gamma*self.wInterpolators[x_pos-1][y_pos-1][z_pos](w)
              + alpha*(1-gamma)*self.wInterpolators[x_pos][y_pos-1][z_pos-1](w)
              + alpha*gamma*self.wInterpolators[x_pos][y_pos-1][z_pos](w)))/(self.y_list[y_pos] - self.y_list[y_pos-1])
        else:
            m = len(x)
            x_pos = np.searchsorted(self.x_list,x)
            x_pos[x_pos > self.x_n-1] = self.x_n-1
            y_pos = np.searchsorted(self.y_list,y)
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list,z)
            z_pos[z_pos > self.z_n-1] = self.z_n-1
            z_pos[z_pos < 1] = 1
            dfdy = np.zeros(m) + np.nan
            for i in xrange(1,self.x_n):
                for j in xrange(1,self.y_n):
                    for k in xrange(1,self.z_n):
                        c = np.logical_and(np.logical_and(i == x_pos, j == y_pos),k == z_pos)
                        if np.any(c):
                            alpha = (x[c] - self.x_list[i-1])/(self.x_list[i] - self.x_list[i-1])
                            gamma = (z[c] - self.z_list[k-1])/(self.z_list[k] - self.z_list[k-1])
                            dfdy[c] = (
                                  ((1-alpha)*(1-gamma)*self.wInterpolators[i-1][j][k-1](w[c])
                                 + (1-alpha)*gamma*self.wInterpolators[i-1][j][k](w[c])
                                 + alpha*(1-gamma)*self.wInterpolators[i][j][k-1](w[c])
                                 + alpha*gamma*self.wInterpolators[i][j][k](w[c])) - 
                                  ((1-alpha)*(1-gamma)*self.wInterpolators[i-1][j-1][k-1](w[c])
                                 + (1-alpha)*gamma*self.wInterpolators[i-1][j-1][k](w[c])
                                 + alpha*(1-gamma)*self.wInterpolators[i][j-1][k-1](w[c])
                                 + alpha*gamma*self.wInterpolators[i][j-1][k](w[c])))/(self.y_list[j] - self.y_list[j-1])
        return dfdy
        
    def _derZ(self,w,x,y,z):
        if _isscalar(w):
            x_pos = max(min(np.searchsorted(self.x_list,x),self.x_n-1),1)
            y_pos = max(min(np.searchsorted(self.y_list,y),self.y_n-1),1)
            z_pos = max(min(np.searchsorted(self.z_list,z),self.z_n-1),1)
            alpha = (x - self.x_list[x_pos-1])/(self.y_list[x_pos] - self.x_list[x_pos-1])
            beta = (y - self.y_list[y_pos-1])/(self.y_list[y_pos] - self.y_list[y_pos-1])
            dfdz = (
             ((1-alpha)*(1-beta)*self.wInterpolators[x_pos-1][y_pos-1][z_pos](w)
              + (1-alpha)*beta*self.wInterpolators[x_pos-1][y_pos][z_pos](w)
              + alpha*(1-beta)*self.wInterpolators[x_pos][y_pos-1][z_pos](w)
              + alpha*beta*self.wInterpolators[x_pos][y_pos][z_pos](w)) - 
              ((1-alpha)*(1-beta)*self.wInterpolators[x_pos-1][y_pos-1][z_pos-1](w)
              + (1-alpha)*beta*self.wInterpolators[x_pos-1][y_pos][z_pos-1](w)
              + alpha*(1-beta)*self.wInterpolators[x_pos][y_pos-1][z_pos-1](w)
              + alpha*beta*self.wInterpolators[x_pos][y_pos][z_pos-1](w)))/(self.z_list[z_pos] - self.z_list[z_pos-1])
        else:
            m = len(x)
            x_pos = np.searchsorted(self.x_list,x)
            x_pos[x_pos > self.x_n-1] = self.x_n-1
            y_pos = np.searchsorted(self.y_list,y)
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list,z)
            z_pos[z_pos > self.z_n-1] = self.z_n-1
            z_pos[z_pos < 1] = 1
            dfdz = np.zeros(m) + np.nan
            for i in xrange(1,self.x_n):
                for j in xrange(1,self.y_n):
                    for k in xrange(1,self.z_n):
                        c = np.logical_and(np.logical_and(i == x_pos, j == y_pos),k == z_pos)
                        if np.any(c):
                            alpha = (x[c] - self.x_list[i-1])/(self.x_list[i] - self.x_list[i-1])
                            beta = (y[c] - self.y_list[j-1])/(self.y_list[j] - self.y_list[j-1])
                            dfdz[c] = (
                                  ((1-alpha)*(1-beta)*self.wInterpolators[i-1][j-1][k](w[c])
                                 + (1-alpha)*beta*self.wInterpolators[i-1][j][k](w[c])
                                 + alpha*(1-beta)*self.wInterpolators[i][j-1][k](w[c])
                                 + alpha*beta*self.wInterpolators[i][j][k](w[c])) - 
                                  ((1-alpha)*(1-beta)*self.wInterpolators[i-1][j-1][k-1](w[c])
                                 + (1-alpha)*beta*self.wInterpolators[i-1][j][k-1](w[c])
                                 + alpha*(1-beta)*self.wInterpolators[i][j-1][k-1](w[c])
                                 + alpha*beta*self.wInterpolators[i][j][k-1](w[c])))/(self.z_list[k] - self.z_list[k-1])
        return dfdz
        
    def distance(self,other_func):
        '''
        Distance between one instance and another is the greater of the maximum
        of the distance between corresponding wInterpolators and the maximum of
        the distance between corresponding x_list, y_list, and z_list values.
        '''
        other_class = other_func.__class__.__name__
        if other_class is not 'TrilinearInterpOnInterp1D':
            return 1000000
        if (self.x_n != other_func.x_n) or (self.y_n != other_func.y_n) or (self.z_n != other_func.z_n):
            return max(np.abs(self.x_n - other_func.x_n),np.abs(self.y_n - other_func.y_n),np.abs(self.z_n - other_func.z_n))
        w_dist_matrix = np.zeros((self.x_n,self.y_n,self.z_n)) + np.nan
        for i in range(self.x_n):
            for j in range(self.y_n):
                for k in range(self.z_n):
                    w_dist_matrix[i,j,k] = self.wInterpolators[i][j][k].distance(other_func.wInterpolators[i][j][k])
        x_dist_vec = np.abs(self.x_list - other_func.x_list)
        y_dist_vec = np.abs(self.y_list - other_func.y_list)
        z_dist_vec = np.abs(self.z_list - other_func.z_list)
        w_dist = np.max(w_dist_matrix)
        x_dist = np.max(x_dist_vec)
        y_dist = np.max(y_dist_vec)
        z_dist = np.max(z_dist_vec)
        return max(w_dist,x_dist,y_dist,z_dist)
        
        
        
    '''
    A 3D interpolation method that linearly interpolates between "layers" of
    arbitrary 2D interpolations.  Useful for models with 2 endogenous state
    variables and one exogenous state variable when solving with the endogenous
    grid method.  NOTE: should not be used if an exogenous 3D grid is used, will
    be significantly slower than TrilinearInterp.
    '''    
class LinearInterpOnInterp2D(HARKinterpolator3D):
    
    def __init__(self,xyInterpolators,z_values):
        '''
        Constructor for the class, generating an approximation to a function of
        the form f(x,y,z) using interpolations over f(x,y,z_0) for a fixed grid
        of z_0 values.
        
        Parameters:
        -------------
        xyInterpolators : [HARKinterpolator2D]
            A list of 2D interpolations over the x and y variables.  The nth
            element of xyInterpolators represents f(x,y,z_values[n]).
        z_values: numpy.array
            An array of z values equal in length to xyInterpolators.
        '''
        self.xyInterpolators = xyInterpolators
        self.z_list = z_values
        self.z_n = z_values.size
        
    def _evaluate(self,x,y,z):
        if _isscalar(x):
            z_pos = max(min(np.searchsorted(self.z_list,z),self.z_n-1),1)
            alpha = (z - self.z_list[z_pos-1])/(self.z_list[z_pos] - self.z_list[z_pos-1])
            f = (1-alpha)*self.xyInterpolators[z_pos-1](x,y) + alpha*self.xyInterpolators[z_pos](x,y)
        else:
            m = len(x)
            z_pos = np.searchsorted(self.z_list,z)
            z_pos[z_pos > self.z_n-1] = self.z_n-1
            z_pos[z_pos < 1] = 1
            f = np.zeros(m) + np.nan
            if x.size > 0:
                for i in xrange(1,self.z_n):
                    c = z_pos == i
                    if np.any(c):
                        alpha = (z[c] - self.z_list[i-1])/(self.z_list[i] - self.z_list[i-1])
                        f[c] = (1-alpha)*self.xyInterpolators[i-1](x[c],y[c]) + alpha*self.xyInterpolators[i](x[c],y[c]) 
        return f
        
    def _derX(self,x,y,z):
        if _isscalar(x):
            z_pos = max(min(np.searchsorted(self.z_list,z),self.z_n-1),1)
            alpha = (z - self.z_list[z_pos-1])/(self.z_list[z_pos] - self.z_list[z_pos-1])
            dfdx = (1-alpha)*self.xyInterpolators[z_pos-1].derivativeX(x,y) + alpha*self.xyInterpolators[z_pos].derivativeX(x,y)
        else:
            m = len(x)
            z_pos = np.searchsorted(self.z_list,z)
            z_pos[z_pos > self.z_n-1] = self.z_n-1
            z_pos[z_pos < 1] = 1
            dfdx = np.zeros(m) + np.nan
            if x.size > 0:
                for i in xrange(1,self.z_n):
                    c = z_pos == i
                    if np.any(c):
                        alpha = (z[c] - self.z_list[i-1])/(self.z_list[i] - self.z_list[i-1])
                        dfdx[c] = (1-alpha)*self.xyInterpolators[i-1].derivativeX(x[c],y[c]) + alpha*self.xyInterpolators[i].derivativeX(x[c],y[c]) 
        return dfdx
        
    def _derY(self,x,y,z):
        if _isscalar(x):
            z_pos = max(min(np.searchsorted(self.z_list,z),self.z_n-1),1)
            alpha = (z - self.z_list[z_pos-1])/(self.z_list[z_pos] - self.z_list[z_pos-1])
            dfdy = (1-alpha)*self.xyInterpolators[z_pos-1].derivativeY(x,y) + alpha*self.xyInterpolators[z_pos].derivativeY(x,y)
        else:
            m = len(x)
            z_pos = np.searchsorted(self.z_list,z)
            z_pos[z_pos > self.z_n-1] = self.z_n-1
            z_pos[z_pos < 1] = 1
            dfdy = np.zeros(m) + np.nan
            if x.size > 0:
                for i in xrange(1,self.z_n):
                    c = z_pos == i
                    if np.any(c):
                        alpha = (z[c] - self.z_list[i-1])/(self.z_list[i] - self.z_list[i-1])
                        dfdy[c] = (1-alpha)*self.xyInterpolators[i-1].derivativeY(x[c],y[c]) + alpha*self.xyInterpolators[i].derivativeY(x[c],y[c]) 
        return dfdy
        
    def _derZ(self,x,y,z):
        if _isscalar(x):
            z_pos = max(min(np.searchsorted(self.z_list,z),self.z_n-1),1)
            dfdz = (self.xyInterpolators[z_pos].derivativeX(x,y) - self.xyInterpolators[z_pos-1].derivativeX(x,y))/(self.z_list[z_pos] - self.z_list[z_pos-1])
        else:
            m = len(x)
            z_pos = np.searchsorted(self.z_list,z)
            z_pos[z_pos > self.z_n-1] = self.z_n-1
            z_pos[z_pos < 1] = 1
            dfdz = np.zeros(m) + np.nan
            if x.size > 0:
                for i in xrange(1,self.z_n):
                    c = z_pos == i
                    if np.any(c):
                        dfdz[c] = (self.xyInterpolators[i](x[c],y[c]) - self.xyInterpolators[i-1](x[c],y[c]))/(self.z_list[i] - self.z_list[i-1])
        return dfdz
        
    def distance(self,other):
        '''
        The distance between instances is the highest among the maximum distance
        between corresponding xyInterpolators, and maximum absolute difference
        between corresponding values of z_list.
        '''
        other_class = other.__class__.__name__
        if other_class is not 'LinearInterpOnInterp2D':
            return 1000000
        if self.z_n != other.z_n:
            return np.abs(self.z_n - other.z_n)
        xy_dist_vec = np.zeros(self.z_n) + np.nan
        for j in range(self.z_n):
            xy_dist_vec[j] = self.xyInterpolators[j].distance(other.xyInterpolators[j])
        z_dist_vec = np.abs(self.z_list - other.z_list)
        xy_dist = np.max(xy_dist_vec)
        z_dist = np.max(z_dist_vec)
        return max(xy_dist,z_dist)
        


'''
    A 4D interpolation method that bilinearly interpolates among "layers" of
    arbitrary 2D interpolations.  Useful for models with two endogenous state
    variables and two exogenous state variables when solving with the endogenous
    grid method.  NOTE: should not be used if an exogenous 4D grid is used, will
    be significantly slower than QuadlinearInterp.
    '''    
class BilinearInterpOnInterp2D(HARKinterpolator4D):
    
    def __init__(self,wxInterpolators,y_values,z_values):
        '''
        Constructor for the class, generating an approximation to a function of
        the form f(w,x,y,z) using interpolations over f(w,x,y_0,z_0) for a fixed
        grid of y_0 and z_0 values.
        
        Parameters:
        -------------
        wxInterpolators : [[HARKinterpolator2D]]
            A list of lists of 2D interpolations over the w and x variables.
            The i,j-th element of wxInterpolators represents
            f(w,x,y_values[i],z_values[j]).
        y_values: numpy.array
            An array of y values equal in length to wxInterpolators.
        z_values: numpy.array
            An array of z values equal in length to wxInterpolators[0].
        '''
        self.wxInterpolators = wxInterpolators
        self.y_list = y_values
        self.y_n = y_values.size
        self.z_list = z_values
        self.z_n = z_values.size
        
    def _evaluate(self,w,x,y,z):
        if _isscalar(x):
            y_pos = max(min(np.searchsorted(self.y_list,y),self.y_n-1),1)
            z_pos = max(min(np.searchsorted(self.z_list,z),self.z_n-1),1)
            alpha = (y - self.y_list[y_pos-1])/(self.y_list[y_pos] - self.y_list[y_pos-1])
            beta = (z - self.z_list[z_pos-1])/(self.z_list[z_pos] - self.z_list[z_pos-1])
            f = ((1-alpha)*(1-beta)*self.wxInterpolators[y_pos-1][z_pos-1](w,x)
              + (1-alpha)*beta*self.wxInterpolators[y_pos-1][z_pos](w,x)
              + alpha*(1-beta)*self.wxInterpolators[y_pos][z_pos-1](w,x)
              + alpha*beta*self.wxInterpolators[y_pos][z_pos](w,x))              
        else:
            m = len(x)
            y_pos = np.searchsorted(self.y_list,y)
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list,z)
            z_pos[z_pos > self.z_n-1] = self.z_n-1
            z_pos[z_pos < 1] = 1
            f = np.zeros(m) + np.nan
            for i in xrange(1,self.y_n):
                for j in xrange(1,self.z_n):
                    c = np.logical_and(i == y_pos, j == z_pos)
                    if np.any(c):
                        alpha = (y[c] - self.y_list[i-1])/(self.y_list[i] - self.y_list[i-1])
                        beta = (z[c] - self.z_list[j-1])/(self.z_list[j] - self.z_list[j-1])
                        f[c] = (
                          (1-alpha)*(1-beta)*self.wxInterpolators[i-1][j-1](w[c],x[c])
                          + (1-alpha)*beta*self.wxInterpolators[i-1][j](w[c],x[c])
                          + alpha*(1-beta)*self.wxInterpolators[i][j-1](w[c],x[c])
                          + alpha*beta*self.wxInterpolators[i][j](w[c],x[c]))
        return f
        
    def _derW(self,w,x,y,z):
        # This may look strange, as we call the derivativeX() method to get the
        # derivative with respect to w, but that's just a quirk of 4D interpolations
        # beginning with w rather than x.  The derivative wrt the first dimension
        # of an element of wxInterpolators is the w-derivative of the main function.
        if _isscalar(x):
            y_pos = max(min(np.searchsorted(self.y_list,y),self.y_n-1),1)
            z_pos = max(min(np.searchsorted(self.z_list,z),self.z_n-1),1)
            alpha = (y - self.y_list[y_pos-1])/(self.y_list[y_pos] - self.y_list[y_pos-1])
            beta = (z - self.z_list[z_pos-1])/(self.z_list[z_pos] - self.z_list[z_pos-1])
            dfdw = ((1-alpha)*(1-beta)*self.wxInterpolators[y_pos-1][z_pos-1].derivativeX(w,x)
              + (1-alpha)*beta*self.wxInterpolators[y_pos-1][z_pos].derivativeX(w,x)
              + alpha*(1-beta)*self.wxInterpolators[y_pos][z_pos-1].derivativeX(w,x)
              + alpha*beta*self.wxInterpolators[y_pos][z_pos].derivativeX(w,x))              
        else:
            m = len(x)
            y_pos = np.searchsorted(self.y_list,y)
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list,z)
            z_pos[z_pos > self.z_n-1] = self.z_n-1
            z_pos[z_pos < 1] = 1
            dfdw = np.zeros(m) + np.nan
            for i in xrange(1,self.y_n):
                for j in xrange(1,self.z_n):
                    c = np.logical_and(i == y_pos, j == z_pos)
                    if np.any(c):
                        alpha = (y[c] - self.y_list[i-1])/(self.y_list[i] - self.y_list[i-1])
                        beta = (z[c] - self.z_list[j-1])/(self.z_list[j] - self.z_list[j-1])
                        dfdw[c] = (
                          (1-alpha)*(1-beta)*self.wxInterpolators[i-1][j-1].derivativeX(w[c],x[c])
                          + (1-alpha)*beta*self.wxInterpolators[i-1][j].derivativeX(w[c],x[c])
                          + alpha*(1-beta)*self.wxInterpolators[i][j-1].derivativeX(w[c],x[c])
                          + alpha*beta*self.wxInterpolators[i][j].derivativeX(w[c],x[c]))
        return dfdw
        
    def _derX(self,w,x,y,z):
        # This may look strange, as we call the derivativeY() method to get the
        # derivative with respect to x, but that's just a quirk of 4D interpolations
        # beginning with w rather than x.  The derivative wrt the second dimension
        # of an element of wxInterpolators is the x-derivative of the main function.
        if _isscalar(x):
            y_pos = max(min(np.searchsorted(self.y_list,y),self.y_n-1),1)
            z_pos = max(min(np.searchsorted(self.z_list,z),self.z_n-1),1)
            alpha = (y - self.y_list[y_pos-1])/(self.y_list[y_pos] - self.y_list[y_pos-1])
            beta = (z - self.z_list[z_pos-1])/(self.z_list[z_pos] - self.z_list[z_pos-1])
            dfdx = ((1-alpha)*(1-beta)*self.wxInterpolators[y_pos-1][z_pos-1].derivativeY(w,x)
              + (1-alpha)*beta*self.wxInterpolators[y_pos-1][z_pos].derivativeY(w,x)
              + alpha*(1-beta)*self.wxInterpolators[y_pos][z_pos-1].derivativeY(w,x)
              + alpha*beta*self.wxInterpolators[y_pos][z_pos].derivativeY(w,x))              
        else:
            m = len(x)
            y_pos = np.searchsorted(self.y_list,y)
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list,z)
            z_pos[z_pos > self.z_n-1] = self.z_n-1
            z_pos[z_pos < 1] = 1
            dfdx = np.zeros(m) + np.nan
            for i in xrange(1,self.y_n):
                for j in xrange(1,self.z_n):
                    c = np.logical_and(i == y_pos, j == z_pos)
                    if np.any(c):
                        alpha = (y[c] - self.y_list[i-1])/(self.y_list[i] - self.y_list[i-1])
                        beta = (z[c] - self.z_list[j-1])/(self.z_list[j] - self.z_list[j-1])
                        dfdx[c] = (
                          (1-alpha)*(1-beta)*self.wxInterpolators[i-1][j-1].derivativeY(w[c],x[c])
                          + (1-alpha)*beta*self.wxInterpolators[i-1][j].derivativeY(w[c],x[c])
                          + alpha*(1-beta)*self.wxInterpolators[i][j-1].derivativeY(w[c],x[c])
                          + alpha*beta*self.wxInterpolators[i][j].derivativeY(w[c],x[c]))
        return dfdx
        
    def _derY(self,w,x,y,z):
        if _isscalar(x):
            y_pos = max(min(np.searchsorted(self.y_list,y),self.y_n-1),1)
            z_pos = max(min(np.searchsorted(self.z_list,z),self.z_n-1),1)
            beta = (z - self.z_list[z_pos-1])/(self.z_list[z_pos] - self.z_list[z_pos-1])
            dfdy = (((1-beta)*self.wxInterpolators[y_pos][z_pos-1](w,x) + beta*self.wxInterpolators[y_pos][z_pos](w,x))
                 -  ((1-beta)*self.wxInterpolators[y_pos-1][z_pos-1](w,x) + beta*self.wxInterpolators[y_pos-1][z_pos](w,x)))/(self.y_list[y_pos] - self.y_list[y_pos-1])
        else:
            m = len(x)
            y_pos = np.searchsorted(self.y_list,y)
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list,z)
            z_pos[z_pos > self.z_n-1] = self.z_n-1
            z_pos[z_pos < 1] = 1
            dfdy = np.zeros(m) + np.nan
            for i in xrange(1,self.y_n):
                for j in xrange(1,self.z_n):
                    c = np.logical_and(i == y_pos, j == z_pos)
                    if np.any(c):
                        beta = (z[c] - self.z_list[j-1])/(self.z_list[j] - self.z_list[j-1])
                        dfdy[c] = (((1-beta)*self.wxInterpolators[i][j-1](w[c],x[c]) + beta*self.wxInterpolators[i][j](w[c],x[c]))
                                -  ((1-beta)*self.wxInterpolators[i-1][j-1](w[c],x[c]) + beta*self.wxInterpolators[i-1][j](w[c],x[c])))/(self.y_list[i] - self.y_list[i-1])
        return dfdy
        
    def _derZ(self,w,x,y,z):
        if _isscalar(x):
            y_pos = max(min(np.searchsorted(self.y_list,y),self.y_n-1),1)
            z_pos = max(min(np.searchsorted(self.z_list,z),self.z_n-1),1)
            alpha = (y - self.y_list[y_pos-1])/(self.y_list[y_pos] - self.y_list[y_pos-1])
            dfdz = (((1-alpha)*self.wxInterpolators[y_pos-1][z_pos](w,x) + alpha*self.wxInterpolators[y_pos][z_pos](w,x))
                 -  ((1-alpha)*self.wxInterpolators[y_pos-1][z_pos-1](w,x) + alpha*self.wxInterpolators[y_pos][z_pos-1](w,x)))/(self.z_list[z_pos] - self.z_list[z_pos-1])
        else:
            m = len(x)
            y_pos = np.searchsorted(self.y_list,y)
            y_pos[y_pos > self.y_n-1] = self.y_n-1
            y_pos[y_pos < 1] = 1
            z_pos = np.searchsorted(self.z_list,z)
            z_pos[z_pos > self.z_n-1] = self.z_n-1
            z_pos[z_pos < 1] = 1
            dfdz = np.zeros(m) + np.nan
            for i in xrange(1,self.y_n):
                for j in xrange(1,self.z_n):
                    c = np.logical_and(i == y_pos, j == z_pos)
                    if np.any(c):
                        alpha = (y[c] - self.y_list[i-1])/(self.y_list[i] - self.y_list[i-1])
                        dfdz[c] = (((1-alpha)*self.wxInterpolators[i-1][j](w[c],x[c]) + alpha*self.wxInterpolators[i][j](w[c],x[c]))
                                -  ((1-alpha)*self.wxInterpolators[i-1][j-1](w[c],x[c]) + alpha*self.wxInterpolators[i][j-1](w[c],x[c])))/(self.z_list[j] - self.z_list[j-1])
        return dfdz
        
    def distance(self,other):
        '''
        The distance between instances is the highest among the maximum distance
        between corresponding wxInterpolators, maximum absolute difference
        between corresponding values of y_list, and maximum absolute difference
        between corresponding values of z_list.
        '''
        other_class = other.__class__.__name__
        if other_class is not 'BilinearInterpOnInterp2D':
            return 1000000
        if (self.y_n != other.y_n) or (self.z_n != other.z_n):
            return max(np.abs(self.y_n - other.y_n),np.abs(self.z_n - other.z_n))
        wx_dist_matrix = np.zeros((self.y_n,self.z_n)) + np.nan
        for i in range(self.y_n):
            for j in range(self.z_n):
                wx_dist_matrix[i,j] = self.wxInterpolators[i,j].distance(other.wxInterpolators[i,j])
        y_dist_vec = np.abs(self.y_list - other.y_list)
        z_dist_vec = np.abs(self.z_list - other.z_list)
        wx_dist = np.max(wx_dist_matrix)
        y_dist = np.max(y_dist_vec)
        z_dist = np.max(z_dist_vec)
        return max(wx_dist,y_dist,z_dist)
        

        
        
class Curvilinear2DInterp(HARKinterpolator2D):
    '''
    A 2D interpolation method for curvilinear or "warped grid" interpolation, as
    in White (2015).  Used for models with two endogenous states that are solved
    with the endogenous grid method.
    '''
    def __init__(self,f_values,x_values,y_values):
        '''
        Constructor for 2D curvilinear interpolation for a function f(x,y)
        
        Parameters:
        -------------
        f_values: numpy.array
            A 2D array of function values such that f_values[i,j] =
            f(x_values[i,j],y_values[i,j]).
        x_values: numpy.array
            A 2D array of x values of the same size as f_values.
        y_values: numpy.array
            A 2D array of y values of the same size as f_values.
        '''
        self.f_values = f_values
        self.x_values = x_values
        self.y_values = y_values
        my_shape = f_values.shape
        self.x_n = my_shape[0]
        self.y_n = my_shape[1]
        self.updatePolarity()
        
    def updatePolarity(self):
        '''
        Fills in the polarity attribute of the interpolation, determining whether
        the "plus" (True) or "minus" (False) solution of the system of equations
        should be used for each sector.  Needs to be called in __init__.
        '''
        
        # Grab a point known to be inside each sector: the midway point between
        # the lower left and upper right vertex of each sector
        x_temp = 0.5*(self.x_values[0:(self.x_n-1),0:(self.y_n-1)] + self.x_values[1:self.x_n,1:self.y_n])
        y_temp = 0.5*(self.y_values[0:(self.x_n-1),0:(self.y_n-1)] + self.y_values[1:self.x_n,1:self.y_n])
        size = (self.x_n-1)*(self.y_n-1)
        x_temp = np.reshape(x_temp,size)
        y_temp = np.reshape(y_temp,size)
        y_pos = np.tile(np.arange(0,self.y_n-1),self.x_n-1)
        x_pos = np.reshape(np.tile(np.arange(0,self.x_n-1),(self.y_n-1,1)).transpose(),size)
        
        # Set the polarity of all sectors to "plus", then test each sector
        self.polarity = np.ones((self.x_n-1,self.y_n-1),dtype=bool)
        alpha, beta = self.findCoords(x_temp,y_temp,x_pos,y_pos)
        polarity = np.logical_and(
            np.logical_and(alpha > 0, alpha < 1),
            np.logical_and(beta > 0, beta < 1))
        
        # Update polarity: if (alpha,beta) not in the unit square, then that
        # sector must use the "minus" solution instead
        self.polarity = np.reshape(polarity,(self.x_n-1,self.y_n-1))
        
    def findSector(self,x,y):
        # Initialize the sector guess
        m = x.size
        x_pos_guess = (np.ones(m)*self.x_n/2).astype(int)
        y_pos_guess = (np.ones(m)*self.y_n/2).astype(int)
        
        # Define a function that checks whether a set of points violates a linear
        # boundary defined by (x_bound_1,y_bound_1) and (x_bound_2,y_bound_2),
        # where the latter is *COUNTER CLOCKWISE* from the former.  Returns
        # 1 if the point is outside the boundary and 0 otherwise.
        violationCheck = lambda x_check,y_check,x_bound_1,y_bound_1,x_bound_2,y_bound_2 : (
            (y_bound_2 - y_bound_1)*x_check - (x_bound_2 - x_bound_1)*y_check > x_bound_1*y_bound_2 - y_bound_1*x_bound_2 ) + 0
        
        # Identify the correct sector for each point to be evaluated
        these = np.ones(m,dtype=bool)
        max_loops = self.x_n + self.y_n
        loops = 0
        while np.any(these) and loops < max_loops:
            # Get coordinates for the four vertices: (xA,yA),...,(xD,yD)
            x_temp = x[these]
            y_temp = y[these]
            xA = self.x_values[x_pos_guess[these],y_pos_guess[these]]
            xB = self.x_values[x_pos_guess[these]+1,y_pos_guess[these]]
            xC = self.x_values[x_pos_guess[these],y_pos_guess[these]+1]
            xD = self.x_values[x_pos_guess[these]+1,y_pos_guess[these]+1]
            yA = self.y_values[x_pos_guess[these],y_pos_guess[these]]
            yB = self.y_values[x_pos_guess[these]+1,y_pos_guess[these]]
            yC = self.y_values[x_pos_guess[these],y_pos_guess[these]+1]
            yD = self.y_values[x_pos_guess[these]+1,y_pos_guess[these]+1]
            
            # Check the "bounding box" for the sector: is this guess plausible?
            move_down = (y_temp < np.minimum(yA,yB)) + 0
            move_right = (x_temp > np.maximum(xB,xD)) + 0
            move_up = (y_temp > np.maximum(yC,yD)) + 0
            move_left = (x_temp < np.minimum(xA,xC)) + 0
            
            # Check which boundaries are violated (and thus where to look next)
            c = (move_down + move_right + move_up + move_left) == 0
            move_down[c] = violationCheck(x_temp[c],y_temp[c],xA[c],yA[c],xB[c],yB[c])
            move_right[c] = violationCheck(x_temp[c],y_temp[c],xB[c],yB[c],xD[c],yD[c])
            move_up[c] = violationCheck(x_temp[c],y_temp[c],xD[c],yD[c],xC[c],yC[c])
            move_left[c] = violationCheck(x_temp[c],y_temp[c],xC[c],yC[c],xA[c],yA[c])
            
            # Update the sector guess based on the violations
            x_pos_next = x_pos_guess[these] - move_left + move_right
            x_pos_next[x_pos_next < 0] = 0
            x_pos_next[x_pos_next > (self.x_n-2)] = self.x_n-2
            y_pos_next = y_pos_guess[these] - move_down + move_up
            y_pos_next[y_pos_next < 0] = 0
            y_pos_next[y_pos_next > (self.y_n-2)] = self.y_n-2
            
            # Check which sectors have not changed, and mark them as complete
            no_move = np.array(np.logical_and(x_pos_guess[these] == x_pos_next, y_pos_guess[these] == y_pos_next))
            x_pos_guess[these] = x_pos_next
            y_pos_guess[these] = y_pos_next
            temp = these.nonzero()
            these[temp[0][no_move]] = False
            
            # Move to the next iteration of the search
            loops += 1
            
        #print(np.sum(these))
        return x_pos_guess, y_pos_guess
        
    def findCoords(self,x,y,x_pos,y_pos):
        # Calculate relative coordinates in the sector for each point
        xA = self.x_values[x_pos,y_pos]
        xB = self.x_values[x_pos+1,y_pos]
        xC = self.x_values[x_pos,y_pos+1]
        xD = self.x_values[x_pos+1,y_pos+1]
        yA = self.y_values[x_pos,y_pos]
        yB = self.y_values[x_pos+1,y_pos]
        yC = self.y_values[x_pos,y_pos+1]
        yD = self.y_values[x_pos+1,y_pos+1]
        polarity = 2.0*self.polarity[x_pos,y_pos] - 1.0
        a = xA
        b = (xB-xA)
        c = (xC-xA)
        d = (xA-xB-xC+xD)
        e = yA
        f = (yB-yA)
        g = (yC-yA)
        h = (yA-yB-yC+yD)
        denom = (d*g-h*c)
        mu = (h*b-d*f)/denom
        tau = (h*(a-x) - d*(e-y))/denom
        zeta = a - x + c*tau
        eta = b + c*mu + d*tau
        theta = d*mu
        alpha = (-eta + polarity*np.sqrt(eta**2.0 - 4.0*zeta*theta))/(2.0*theta)
        beta = mu*alpha + tau
        
        return alpha, beta
        
    def _evaluate(self,x,y):
        x_pos, y_pos = self.findSector(x,y)
        alpha, beta = self.findCoords(x,y,x_pos,y_pos)
        
        # Calculate the function at each point using bilinear interpolation
        f = (
             (1-alpha)*(1-beta)*self.f_values[x_pos,y_pos]
          +  (1-alpha)*beta*self.f_values[x_pos,y_pos+1]
          +  alpha*(1-beta)*self.f_values[x_pos+1,y_pos]
          +  alpha*beta*self.f_values[x_pos+1,y_pos+1])
        return f
        
    # Need to add _derX and _derY methods; math is in desk drawer at UD
        
        
if __name__ == '__main__':       
    '''
    Tests of some of the interpolation methods.  Should be expanded and cleaned up.
    '''
    from time import clock
    import matplotlib.pyplot as plt
    
    RNG = np.random.RandomState(123)
    
    if True:
        x = np.linspace(1,20,39)
        y = np.log(x)
        dydx = 1.0/x
        f = CubicInterp(x,y,dydx)
        x_test = np.linspace(0,30,200)
        y_test = f(x_test)
        plt.plot(x_test,y_test)
        plt.show()
    
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
            that_interpolation = CubicInterp(this_x_list,f(this_x_list,y*np.ones(this_x_list.size)),dfdx(this_x_list,y*np.ones(this_x_list.size)))
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
        
        N = 1000000
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
        
        
    if False:
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
        
        
