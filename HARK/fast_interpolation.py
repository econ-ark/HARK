import numpy as np
from interpolation import interp # econ forge
from scipy.interpolate import CubicSpline # scipy
from scipy.interpolate import CubicHermiteSpline # scipy

def fast_linear_1d_interp(x, obj):
    result = interp(obj.x_list, obj.y_list, x)
    above_upper_bound = x > obj.x_list[-1]
    i = len(obj.x_list) - 1
    alpha = (x[above_upper_bound] - obj.x_list[i - 1]) / (obj.x_list[i] - obj.x_list[i - 1])
    result[above_upper_bound] = (1.0 - alpha) * obj.y_list[i - 1] + alpha * obj.y_list[i]
    return [result]

def get_cubic_spline(obj):
    return CubicSpline(np.array(obj.x_list), np.array(obj.y_list))

def get_cubic_hermite_spline(obj):
    return CubicHermiteSpline(np.array(obj.x_list), np.array(obj.y_list), np.array(obj.dydx_list))
