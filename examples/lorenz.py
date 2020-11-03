import numpy as np
import matplotlib.pyplot as plt

from probDE.car import car_init
from probDE.cython.KalmanODE import KalmanODE
from probDE.utils import indep_init, zero_pad
from lorenz_graph import lorenz_graph

# RHS of ODE
def lorenz(X, t, theta,  out=None):
    if out is None:
        out = np.empty(3)
    rho, sigma, beta = theta
    p = len(X)//3
    x, y, z = X[p*0], X[p*1], X[p*2]
    out[0] = -sigma*x + sigma*y
    out[1] = rho*x - y -x*z
    out[2] = -beta*z + x*y
    return out

def lorenz_example():
    r"Produces the graph in Figure 2 of the paper."
    
    # prior process definition
    # number of derivatives per variable in prior
    n_deriv = [2, 2, 2]
    # Initial value, x0, for the IVP
    x0 = [-12, -5, 38]
    v0 = [70, 125, -124/3]
    X0 = np.column_stack([x0, v0])

    # prior process definition
    # number of derivatives per variable in prior
    n_deriv_prior = [3, 3, 3]
    p = sum(n_deriv_prior)
    n_obs = 3 # Number of observations from interrogation
    # LHS Matrix of ODE
    W_mat = np.zeros((len(n_deriv), sum(n_deriv)))
    for i in range(len(n_deriv)): W_mat[i, sum(n_deriv[:i])+1] = 1
    # pad the inputs
    W = zero_pad(W_mat, n_deriv, n_deriv_prior)

    # it is assumed that the solution is sought on the interval [tmin, tmax].
    n_eval = 5000
    tmin = 0
    tmax = 20
    theta = np.array([28, 10, 8/3])

    # The rest of the parameters can be tuned according to ODE
    # For this problem, we will use
    tau = np.array([1, 1, 1])
    sigma = np.array([.5, .5, .5])

    # Get parameters needed to run the solver
    dt = (tmax-tmin)/n_eval
    ode_init, v_init = car_init(n_deriv_prior, tau, sigma, dt, X0)
    kinit = indep_init(ode_init, n_deriv_prior)

    # Initialize the Kalman class
    kalmanode = KalmanODE(p, n_obs, tmin, tmax, n_eval, lorenz, **kinit)
    # Run the solver to get an approximation
    kalman_sim = kalmanode.solve_sim(v_init, W, theta)

    # Produce the graph in Figure 2
    draws = 1000
    lorenz_graph(lorenz, n_deriv, n_deriv_prior, n_obs, tmin, tmax, n_eval, W_mat, tau, sigma, X0, theta, draws)

    return

if __name__ == '__main__':
    lorenz_example()
