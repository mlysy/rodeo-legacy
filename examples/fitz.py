import numpy as np
from inference import inference
from probDE.car import car_init
from probDE.cython.KalmanODE import KalmanODE
from probDE.utils import indep_init

def fitz(X_out, X_t, t, theta):
    "FitzHugh-Nagumo ODE function."
    a, b, c = theta
    n_deriv1 = len(X_t)//2
    V, R = X_t[0], X_t[n_deriv1] 
    X_out[0] = c*(V - V*V*V/3 + R)
    X_out[1] = -1/c*(V - a + b*R)
    return

def fitz_odeint(X_t, t, theta):
    "FitzHugh-Nagumo ODE function."
    a, b, c = theta
    n_deriv1 = len(X_t)//2
    V, R = X_t[0], X_t[n_deriv1] 
    return np.array([c*(V - V*V*V/3 + R), -1/c*(V - a + b*R)])

def fitz_example():
    "Perform parameter inference using the FitzHugh-Nagumo function."
    # These parameters define the order of the ODE and the CAR(p) process
    n_deriv = 6 # Total state
    n_obs = 2 # Total measures
    n_deriv_var = [3, 3]
    state_ind = [0, 3] # Index of 0th derivative of each state

    # it is assumed that the solution is sought on the interval [tmin, tmax].
    tmin = 0 
    tmax = 40
    h = 0.1 # step size
    n_eval = int((tmax-tmin)/h)

    # The rest of the parameters can be tuned according to ODE
    # For this problem, we will use
    n_var = 2
    tau = [100]*n_var
    sigma = [.1]*n_var

    # Initial value, x0, for the IVP
    x0 = np.array([-1., 1.])
    v0 = np.array([1, 1/3])
    X0 = np.column_stack([x0, v0])
    w_mat = np.array([[0.0, 1.0], [0.0, 1.0]])

    # logprior parameters
    theta_true = np.array([0.2, 0.2, 3]) # True theta
    n_theta = len(theta_true)
    phi_sd = np.ones(n_theta) 

    # Observation noise
    gamma = 0.2

    # Number of samples to draw from posterior
    n_samples = 100000

    # Initialize inference class and simulate observed data
    inf = inference(state_ind, tmin, tmax, fitz)
    Y_t = inf.simulate(fitz_odeint, x0, theta_true, gamma)

    # Parameter inference using Euler's approximation
    hlst = np.array([0.1, 0.05, 0.02, 0.01, 0.005])
    theta_euler = np.zeros((len(hlst), n_samples, n_theta))
    for i in range(len(hlst)):
        phi_hat, phi_var = inf.phi_fit(Y_t, x0, hlst[i], theta_true, phi_sd, gamma, False)
        theta_euler[i] = inf.theta_sample(phi_hat, phi_var, n_samples)

    # Parameter inference using Kalman solver
    theta_kalman = np.zeros((len(hlst), n_samples, n_theta))
    for i in range(len(hlst)):
        kinit, W, x0_state = indep_init(car_init(n_deriv_var, tau, sigma, hlst[i], X0), w_mat, n_deriv)
        n_eval = int((tmax-tmin)/hlst[i])
        kode = KalmanODE(n_deriv, n_obs, tmin, tmax, n_eval, fitz, **kinit)
        inf.kode = kode
        inf.W = W
        phi_hat, phi_var = inf.phi_fit(Y_t, x0_state, hlst[i], theta_true, phi_sd, gamma, True)
        theta_kalman[i] = inf.theta_sample(phi_hat, phi_var, n_samples)
    
    # Produces the graph in Figure 3
    inf.theta_plot(theta_euler, theta_kalman, theta_true, hlst)
    return

if __name__ == '__main__':
    fitz_example()
    