'''
This module contains custom interpolation methods for representing approximations
to functions.  Classes defined here should descend from classes in scipy.interpolate.
'''

import warnings
import numpy as np
from scipy.interpolate import UnivariateSpline

def _isscalar(x):
    """Check whether x is if a scalar type, or 0-dim"""
    return np.isscalar(x) or hasattr(x, 'shape') and x.shape == ()
    

class HARKinterpolator():
    '''
    A wrapper class for interpolation methods in HARK.
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


class Cubic1DInterpDecay(HARKinterpolator):
    """
    An interpolating function using piecewise cubic splines and "decay extrapolation"
    above the highest gridpoint.  Matches level and slope of 1D function at gridpoints,
    smoothly interpolating in between.  Extrapolation above highest gridpoint approaches
    the limiting linear function y = b_limit + m_limit*x.
    """ 

    def __init__(self,x_list,y_list,dydx_list,b_limit,m_limit):
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
        
        '''
        self.x_list = x_list
        self.n = len(x_list)
        
        # Define lower extrapolation as linear function 
        self.coeffs = [[y_list[0],dydx_list[0]]]

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
        gap = m_limit*x1 + b_limit - y1
        slope = m_limit - dydx_list[self.n-1]
        if (gap != 0) and (slope <= 0):
            temp = [b_limit, m_limit, gap, slope/gap]
        elif slope > 0:
            temp = [b_limit, m_limit, 0, 0] # fixing a problem when slope is positive
        else:
            temp = [b_limit, m_limit, gap, 0]
        self.coeffs.append(temp)


    def _evaluate(self,x):
        '''
        Returns the level of the function at each value in x.
        '''
        if _isscalar(x):
            pos = np.searchsorted(self.x_list,x)
            if pos == 0:
                y = self.coeffs[0][0] + self.coeffs[0][1]*(x - self.x_list[0])
            elif (pos < self.n):
                alpha = (x - self.x_list[pos-1])/(self.x_list[pos] - self.x_list[pos-1])
                y = self.coeffs[pos][0] + alpha*(self.coeffs[pos][1] + alpha*(self.coeffs[pos][2] + alpha*self.coeffs[pos][3]))
            else:
                alpha = x - self.x_list[self.n-1]
                y = self.coeffs[pos][0] + x*self.coeffs[pos][1] - self.coeffs[pos][2]*np.exp(alpha*self.coeffs[pos][3])
        else:
            m = len(x)
            pos = np.searchsorted(self.x_list,x)
            y = np.zeros(m)
            if y.size > 0:
                for i in xrange(self.n+1):
                    c = pos == i
                    if i == 0:
                        y[c] = self.coeffs[0][0] + self.coeffs[0][1]*(x[c] - self.x_list[0])
                    elif i < self.n:
                        alpha = (x[c] - self.x_list[i-1])/(self.x_list[i] - self.x_list[i-1])
                        y[c] = self.coeffs[i][0] + alpha*(self.coeffs[i][1] + alpha*(self.coeffs[i][2] + alpha*self.coeffs[i][3]))
                    else:
                        alpha = x[c] - self.x_list[self.n-1]
                        y[c] = self.coeffs[i][0] + x[c]*self.coeffs[i][1] - self.coeffs[i][2]*np.exp(alpha*self.coeffs[i][3])        
        return y


    def _der(self,x):
        '''
        Returns the first derivative of the function at each value in x.
        '''
        if _isscalar(x):
            pos = np.searchsorted(self.x_list,x)
            if pos == 0:
                dydx = self.coeffs[0][1]
            elif (pos < self.n):
                alpha = (x - self.x_list[pos-1])/(self.x_list[pos] - self.x_list[pos-1])
                dydx = (self.coeffs[pos][1] + alpha*(2*self.coeffs[pos][2] + alpha*3*self.coeffs[pos][3]))/(self.x_list[pos] - self.x_list[pos-1])
            else:
                alpha = x - self.x_list[self.n-1]
                dydx = self.coeffs[pos][1] - self.coeffs[pos][2]*self.coeffs[pos][3]*np.exp(alpha*self.coeffs[pos][3])
        else:
            m = len(x)
            pos = np.searchsorted(self.x_list,x)
            dydx = np.zeros(m)
            if dydx.size > 0:
                for i in xrange(self.n+1):
                    c = pos == i
                    if i == 0:
                        dydx[c] = self.coeffs[0][1]
                    elif i < self.n:
                        alpha = (x[c] - self.x_list[i-1])/(self.x_list[i] - self.x_list[i-1])
                        dydx[c] = (self.coeffs[i][1] + alpha*(2*self.coeffs[i][2] + alpha*3*self.coeffs[i][3]))/(self.x_list[i] - self.x_list[i-1])
                    else:
                        alpha = x[c] - self.x_list[self.n-1]
                        dydx[c] = self.coeffs[i][1] - self.coeffs[i][2]*self.coeffs[i][3]*np.exp(alpha*self.coeffs[i][3])        
        return dydx



    def _evalAndDer(self,x):
        '''
        Returns the level and first derivative of the function at each value in x.
        '''
        if _isscalar(x):
            pos = np.searchsorted(self.x_list,x)
            if pos == 0:
                y = self.coeffs[0][0] + self.coeffs[0][1]*(x - self.x_list[0])
                dydx = self.coeffs[0][1]
            elif (pos < self.n):
                alpha = (x - self.x_list[pos-1])/(self.x_list[pos] - self.x_list[pos-1])
                y = self.coeffs[pos][0] + alpha*(self.coeffs[pos][1] + alpha*(self.coeffs[pos][2] + alpha*self.coeffs[pos][3]))
                dydx = (self.coeffs[pos][1] + alpha*(2*self.coeffs[pos][2] + alpha*3*self.coeffs[pos][3]))/(self.x_list[pos] - self.x_list[pos-1])
            else:
                alpha = x - self.x_list[self.n-1]
                y = self.coeffs[pos][0] + x*self.coeffs[pos][1] - self.coeffs[pos][2]*np.exp(alpha*self.coeffs[pos][3])
                dydx = self.coeffs[pos][1] - self.coeffs[pos][2]*self.coeffs[pos][3]*np.exp(alpha*self.coeffs[pos][3])
        else:
            m = len(x)
            pos = np.searchsorted(self.x_list,x)
            y = np.zeros(m)
            dydx = np.zeros(m)
            if y.size > 0:
                for i in xrange(self.n+1):
                    c = pos == i
                    if i == 0:
                        y[c] = self.coeffs[0][0] + self.coeffs[0][1]*(x[c] - self.x_list[0])
                        dydx[c] = self.coeffs[0][1]
                    elif i < self.n:
                        alpha = (x[c] - self.x_list[i-1])/(self.x_list[i] - self.x_list[i-1])
                        y[c] = self.coeffs[i][0] + alpha*(self.coeffs[i][1] + alpha*(self.coeffs[i][2] + alpha*self.coeffs[i][3]))
                        dydx[c] = (self.coeffs[i][1] + alpha*(2*self.coeffs[i][2] + alpha*3*self.coeffs[i][3]))/(self.x_list[i] - self.x_list[i-1])
                    else:
                        alpha = x[c] - self.x_list[self.n-1]
                        y[c] = self.coeffs[i][0] + x[c]*self.coeffs[i][1] - self.coeffs[i][2]*np.exp(alpha*self.coeffs[i][3])
                        dydx[c] = self.coeffs[i][1] - self.coeffs[i][2]*self.coeffs[i][3]*np.exp(alpha*self.coeffs[i][3])      
        return y,dydx

    def distance(self,other_function):
        '''
        The distance between two instances of the class is the largest difference
        in x values or y values.  If the instances have different numbers of gridpoints,
        then the difference in the number of gridpoints is returned instead.
        '''
        other_class = other_function.__class__.__name__
        if other_class is not 'Cubic1DInterpDecay':
            return 1000000
        xA = self.x_list
        xB = other_function.x_list
        if (len(xA) == len(xB)):
            yA = [self.coeffs[j][0] for j in range(len(xA)-1)]
            yA.append(self.coeffs[-1][0] + self.coeffs[-1][1]*xA[-1] - self.coeffs[-1][2])
            yA = np.array(yA)
            yB = [other_function.coeffs[j][0] for j in range(len(xB)-1)]
            yB.append(other_function.coeffs[-1][0] + other_function.coeffs[-1][1]*xB[-1] - other_function.coeffs[-1][2])
            yB = np.array(yB)
            
            x_dist = np.max(np.abs(np.subtract(xA,xB)))
            y_dist = np.max(np.abs(np.subtract(yA,yB)))
            dist = max(x_dist,y_dist)
        else:
            dist = np.abs(len(xA) - len(xB))
        return dist


class LinearInterp(HARKinterpolator):
    '''
    A slight extension of scipy.interpolate's UnivariateSpline for linear interpolation.
    Adds a distance method to allow convergence checks.
    '''
    def __init__(self,x,y):
        self.function = UnivariateSpline(x,y,k=1,s=0)
        
    def _evaluate(self,x):
        return self.function(x)
        
    def _der(self,x):
        return self.function(x,1)
        
    def _evalAndDer(self,x):
        return self.function(x), self.function(x,1)

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


class ConstrainedComposite(HARKinterpolator):
    """
    An arbitrary 1D function that has a linear constraint with slope of 1.  The
    unconstrained function can be of any class that has the methods __call__,
    derivative, and eval_with_derivative.
    """ 

    def __init__(self,the_function,constraint):
        '''
        Constructor method for the interpolation.
        '''
        self.function = the_function
        self.constraint = constraint

    def _evaluate(self,x):
        '''
        Returns the level of the function at each value in x.
        '''
        unconstrained = self.function(x)
        constrained = self.constraint(x)
        if _isscalar(x):
            y = np.min([unconstrained,constrained])
        else:
            m = len(x)
            y = np.zeros(m)
            c = unconstrained < constrained
            y[c] = unconstrained[c]
            y[~c] = constrained[~c]
        return y

    def _der(self,x):
        '''
        Returns the first derivative of the function at each value in x.
        '''
        temp = self.function.eval_with_derivative(x)
        unconstrained = temp[0]
        slope = temp[1]
        constrained = self.constraint(x)
        if _isscalar(x):
            if (constrained < unconstrained):
                dydx = 1
            else:
                dydx = slope
        else:
            m = len(x)
            dydx = np.zeros(m)
            c = unconstrained < constrained
            dydx[c] = slope[c]
            dydx[~c] = 1
        return dydx


    def _evalAndDer(self,x):
        '''
        Returns the level and first derivative of the function at each value in x.
        '''
        unconstrained,slope = self.function.eval_with_derivative(x)
        constrained = self.constraint(x)
        if _isscalar(x):
            y = np.min([unconstrained,constrained])
            if (constrained < unconstrained):
                slope = 1
        else:
            m = len(x)
            y = np.zeros(m)
            dydx = np.zeros(m)
            c = unconstrained < constrained
            y[c] = unconstrained[c]
            y[~c] = constrained[~c]
            dydx[c] = slope[c]
            dydx[~c] = 1
        return y,dydx

    
    def distance(self,function_other):
        '''
        Returns the distance between this instance and another instance of the class.
        The distance is the distance between the corresponding unconstrained functions.
        '''
        return self.function.distance(function_other.function)

    