import numpy as np
import matplotlib.pyplot as plt

from probDE.car import car_init
from probDE.cython.KalmanODE import KalmanODE
from probDE.utils import indep_init
from lorenz_graph import lorenz_graph

# RHS of ODE
def lorenz(X_t, t, theta=(28, 10, 8/3)):
    r"Loren63 ODE function"
    rho, sigma, beta = theta
    p = len(X_t)//3
    x, y, z = X_t[p*0], X_t[p*1], X_t[p*2]
    return np.array([-sigma*x + sigma*y, rho*x - y -x*z, -beta*z + x*y])

def lorenz_example():
    r"Produces the graph in Figure 2 of the paper."
    # LHS Matrix of ODE
    w_mat = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

    # These parameters define the order of the ODE and the CAR(p) process
    n_meas = 3
    n_state = 9 # number of continuous derivatives of CAR(p) solution prior
    n_state1 = n_state2 = n_state3 = 3
    n_var_states = np.array([n_state1, n_state2, n_state3])

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
    kinit, x0_state = indep_init([car_init(n_state1, tau[0], sigma[0], dt, w_mat[0], x0[0]),
                                  car_init(n_state2, tau[1], sigma[1], dt, w_mat[1], x0[1]),
                                  car_init(n_state3, tau[2], sigma[2], dt, w_mat[2], x0[2])], n_state)

    # Initialize the Kalman class
    kalmanode = KalmanODE(n_state, n_meas, tmin, tmax, n_eval, lorenz, **kinit)
    # Run the solver to get an approximation
    kalman_sim = kalmanode.solve(x0_state, theta, mv=False, sim=True)

    # Produce the graph in Figure 2
    tau = np.array([1, 1, 1])
    sigma = np.array([.5, .5, .5])
    draws = 1000
    lorenz_graph(lorenz, n_state, n_meas, tmin, tmax, n_eval, w_mat, n_var_states, tau, sigma, x0, theta, draws)

    return

if __name__ == '__main__':
    lorenz_example()
