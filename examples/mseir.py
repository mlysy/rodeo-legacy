import numpy as np
from inference import inference
from probDE.car import car_init
from probDE.cython.KalmanODE import KalmanODE
from probDE.utils import indep_init

def mseir(X_t, t, theta):
    "MSEIR ODE function"
    p = len(X_t)//5
    M, S, E, I, R = X_t[::p]
    N = M+S+E+I+R
    Lambda, delta, beta, mu, epsilon, gamma = theta
    dM = Lambda - delta*M - mu*M
    dS = delta*M - beta*S*I/N - mu*S
    dE = beta*S*I/N - (epsilon + mu)*E
    dI = epsilon*E - (gamma + mu)*I
    dR = gamma*I - mu*R
    return np.array([dM, dS, dE, dI, dR])

def mseir_example():
    "Perform parameter inference using the MSEIR function."
    # LHS Matrix of ODE
    w_mat = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

    # These parameters define the order of the ODE and the CAR(p) process
    n_meas = 5 # Total measures
    n_state = 15 # Total state
    n_state1 = n_state2 = n_state3 = n_state4 = n_state5 = 3
    state_ind = [0, 3, 6, 9, 12] # Index of 0th derivative of each state

    # it is assumed that the solution is sought on the interval [tmin, tmax].
    tmin = 0
    tmax = 40
    h = 0.1 # step size
    n_eval = int((tmax-tmin)/h)

    # The rest of the parameters can be tuned according to ODE
    # For this problem, we will use
    tau = np.array([100, 100, 100, 100, 100])
    sigma = np.array([.1, .1, .1, .1, .1])

    # Initial value, x0, for the IVP
    theta_true = (1.1, 0.7, 0.4, 0.005, 0.02, 0.03) # True theta
    x0 = np.array([1000, 100, 50, 3, 3])
    v0 = mseir(x0, 0, theta_true)
    X0 = np.column_stack([x0, v0])

    # logprior parameters
    n_theta = len(theta_true)
    phi_sd = np.ones(n_theta)

    # Observation noise
    gamma = 0.2

    # Number of samples to draw from posterior
    n_samples = 100000

    # Initialize inference class and simulate observed data
    inf = inference(state_ind, tmin, tmax, mseir)
    Y_t = inf.simulate(x0, theta_true, gamma)

    # Parameter inference using Euler's approximation
    hlst = np.array([0.1, 0.05, 0.02, 0.01, 0.005])
    theta_euler = np.zeros((len(hlst), n_samples, n_theta))
    for i in range(len(hlst)):
        phi_hat, phi_var = inf.phi_fit(Y_t, x0, hlst[i], theta_true, phi_sd, gamma, False)
        theta_euler[i] = inf.theta_sample(phi_hat, phi_var, n_samples)
    
    # Parameter inference using Kalman solver
    theta_kalman = np.zeros((len(hlst), n_samples, n_theta))
    for i in range(len(hlst)):
        kinit, x0_state = indep_init([car_init(n_state1, tau[0], sigma[0], hlst[i], w_mat[0], X0[0]),
                                      car_init(n_state2, tau[1], sigma[1], hlst[i], w_mat[1], X0[1]),
                                      car_init(n_state3, tau[2], sigma[2], hlst[i], w_mat[2], X0[2]),
                                      car_init(n_state4, tau[3], sigma[3], hlst[i], w_mat[3], X0[3]),
                                      car_init(n_state5, tau[4], sigma[4], hlst[i], w_mat[4], X0[4])], n_state)
        n_eval = int((tmax-tmin)/hlst[i])
        kode = KalmanODE(n_state, n_meas, tmin, tmax, n_eval, mseir, **kinit)
        inf.kode = kode
        phi_hat, phi_var = inf.phi_fit(Y_t, x0_state, hlst[i], theta_true, phi_sd, gamma, True)
        theta_kalman[i] = inf.theta_sample(phi_hat, phi_var, n_samples)
    
    # Produces the graph in Figure 4
    inf.theta_plot(theta_euler, theta_kalman, theta_true, hlst)
    return

if __name__ == '__main__':
    mseir_example()