import numpy as np
from math import sin
import matplotlib.pyplot as plt

from rodeo.ibm import ibm_init
from rodeo.cython.KalmanODE import KalmanODE
from rodeo.utils import indep_init, zero_pad
from readme_graph import readme_graph

# ODE function
def ode_fun(x, t, theta=None, x_out=None):
    if x_out is None:
        x_out = np.empty(1)
    x_out[0] = sin(2*t) - x[0]
    return

def chkrebtii_example():
    r"Produces the graph in Figure 1 of the paper."
    # LHS vector of ODE
    # 2.  Define the IVP

    W = np.array([[0.0, 0.0, 1.0]])  # LHS vector of ODE
    x0 = np.array([-1., 0., 1.])  # initial value for the IVP

    # Time interval on which a solution is sought.
    tmin = 0
    tmax = 10

    # 3.  Define the prior process
    #
    # (Perhaps best to describe this in text, not code comments)
    #
    # We're going to use a solution prior that has one more derivative than as specified in the IVP.  
    # To do this, we'll pad the original IVP with zeros, for which we have the convenience function 
    # zero_pad().

    n_deriv = [2]  # number of derivatives in IVP
    n_deriv_prior = [4]  # number of derivatives in IBM prior

    # zero padding
    W_pad = zero_pad(W, n_deriv, n_deriv_prior)
    x0_pad = zero_pad(x0, n_deriv, n_deriv_prior)

    # IBM process scale factor
    sigma = [.5]

    # 4.  Instantiate the ODE solver object.

    n_points = 80  # number of steps in which to discretize the time interval.
    dt = (tmax-tmin)/n_points  # step size

    # generate the Kalman parameters corresponding to the prior
    prior = ibm_init(dt, n_deriv_prior, sigma)

    # instantiate the ODE solver
    ode = KalmanODE(W=W_pad,
                    tmin=tmin,
                    tmax=tmax,
                    n_eval=n_points,
                    fun=ode_fun,
                    **prior)


    # 5.  Evaluate the ODE solution

    # deterministic output: posterior mean
    mut, Sigmat = ode.solve_mv(x0=x0_pad)

    # probabilistic output: draw from posterior
    xt = ode.solve_sim(x0=x0_pad)
    
    # Produces the graph in Figure 1
    draws = 100
    readme_graph(ode_fun, n_deriv, n_deriv_prior, tmin, tmax, W, x0, draws)
    return

if __name__ == '__main__':
    chkrebtii_example()
    