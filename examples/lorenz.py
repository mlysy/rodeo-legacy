import numpy as np
import matplotlib.pyplot as plt

from probDE.car import car_init
from probDE.cython.KalmanODE import KalmanODE
from probDE.utils import indep_init
from lorenz_graph import lorenz_graph

# RHS of ODE
def lorenz(X_out, X, t, theta=(28, 10, 8/3)):
    rho, sigma, beta = theta
    p = len(X)//3
    x, y, z = X[p*0], X[p*1], X[p*2]
    X_out[0] = -sigma*x + sigma*y
    X_out[1] = rho*x - y -x*z
    X_out[2] = -beta*z + x*y
    return 

def lorenz_example():
    r"Produces the graph in Figure 2 of the paper."
    # LHS Matrix of ODE
    w_mat = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

    # These parameters define the order of the ODE and the CAR(p) process
    n_obs = 3
    n_deriv = 9 # number of continuous derivatives of CAR(p) solution prior
    n_deriv_var = [3, 3, 3]

    # it is assumed that the solution is sought on the interval [tmin, tmax].
    n_eval = 5000
    tmin = 0
    tmax = 20
    theta = (28, 10, 8/3)

    # The rest of the parameters can be tuned according to ODE
    # For this problem, we will use
    tau = np.array([1.3, 1.3, 1.3])
    sigma = np.array([.5, .5, .5])

    # Initial value, x0, for the IVP
    x0 = [-12, -5, 38]
    v0 = [70, 125, -124/3]
    x0 = np.column_stack([x0, v0])

    # Get parameters needed to run the solver
    dt = (tmax-tmin)/n_eval
    # Get initial parameters for the Kalman solver using 3 CAR(p) process
    kinit, W, x0_state = indep_init(car_init(n_deriv_var, tau, sigma, dt, x0), w_mat, n_deriv)

    # Initialize the Kalman class
    kalmanode = KalmanODE(n_deriv, n_obs, tmin, tmax, n_eval, lorenz, **kinit)
    # Run the solver to get an approximation
    kalman_sim = kalmanode.solve(x0_state, W, theta, mv=False, sim=True)

    # Produce the graph in Figure 2
    tau = np.array([1, 1, 1])
    sigma = np.array([.5, .5, .5])
    draws = 1000
    lorenz_graph(lorenz, n_deriv, n_obs, tmin, tmax, n_eval, w_mat, n_deriv_var, tau, sigma, x0, theta, draws)

    return

if __name__ == '__main__':
    lorenz_example()
