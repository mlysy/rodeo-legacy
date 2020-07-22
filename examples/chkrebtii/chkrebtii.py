import numpy as np
from math import sin
import matplotlib.pyplot as plt

from probDE.car import car_init
from probDE.cython.KalmanODE import KalmanODE
from probDE.utils import indep_init
from readme_graph import readme_graph

def ode_fun(x_t, t, theta=None):
    return np.array([sin(2*t) - x_t[0]])

def chkrebtii_example():
    # LHS vector of ODE
    w_vec = np.array([0.0, 0.0, 1.0])

    # These parameters define the order of the ODE and the CAR(p) process
    n_meas = 1
    n_state = 4

    # it is assumed that the solution is sought on the interval [tmin, tmax].
    n_eval = 200
    tmin = 0
    tmax = 10

    # The rest of the parameters can be tuned according to ODE
    # For this problem, we will use
    tau = 50
    sigma = .001

    # Initial value, x0, for the IVP
    x0 = np.array([-1., 0., 1.])

    # Get parameters needed to run the solver
    dt = (tmax-tmin)/n_eval

    # Get initial parameters for the Kalman solver using the CAR(p) process
    kinit, x0_state = indep_init([car_init(n_state, tau, sigma, dt, w_vec, x0)], n_state)

    # Initialize the Kalman class
    kalmanode = KalmanODE(n_state, n_meas, tmin, tmax, n_eval, ode_fun, **kinit)
    # Run the solver to get an approximation
    kalman_sim = kalmanode.solve(x0_state, mv=False, sim=True)

    # Produces the graph in Figure 1
    draws = 100
    readme_graph(ode_fun, n_state, n_meas, tmin, tmax, w_vec, x0, draws)
    return

if __name__ == '__main__':
    chkrebtii_example()
    