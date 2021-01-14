import numpy as np
from math import sin
from numba import njit

def rel_err(X1, X2):
    """Sum of relative error between two numpy arrays."""
    return np.sum(np.abs((X1.ravel() - X2.ravel())/X1.ravel()))

def chkrebtii_kalman(x_t, t, theta=None, x_out=None):
    """Chkrebtii function in kalman format."""
    x_out[0] = sin(2*t) - x_t[0]
    return

def chkrebtii_odeint(x_t, t):
    """Chkrebtii function in odeint format."""
    return [x_t[1], sin(2*t) - x_t[0]]

@njit
def chkrebtii_kalman_nb(x_t, t, theta=None, x_out=None):
    """Chkrebtii function in kalman format."""
    x_out[0] = sin(2*t) - x_t[0]
    return
