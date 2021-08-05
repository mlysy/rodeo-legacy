import numpy as np
from inference.normal import normal as inference
from rodeo.ibm import ibm_init
from rodeo.cython.KalmanODE import KalmanODE
from rodeo.utils import indep_init, zero_pad

def mseir(X_t, t, theta, out=None):
    "MSEIR ODE function"
    if out is None:
        out = np.empty(5)
    p = len(X_t)//5
    M, S, E, I, R = X_t[::p]
    N = M+S+E+I+R
    Lambda, delta, beta, mu, epsilon, gamma = theta
    out[0] = Lambda - delta*M - mu*M
    out[1] = delta*M - beta*S*I/N - mu*S
    out[2] = beta*S*I/N - (epsilon + mu)*E
    out[3] = epsilon*E - (gamma + mu)*I
    out[4] = gamma*I - mu*R
    return out

def mseir_example():
    "Perform parameter inference using the MSEIR function."
    # These parameters define the order of the ODE and the CAR(p) process
    n_deriv = [1]*5 # Total state
    n_deriv_prior= [3]*5
    state_ind = [0, 3, 6, 9, 12] # Index of 0th derivative of each state

    # it is assumed that the solution is sought on the interval [tmin, tmax].
    tmin = 0
    tmax = 40

    # The rest of the parameters can be tuned according to ODE
    # For this problem, we will use
    n_var = 5
    sigma = [.1]*n_var

    # Initial value, x0, for the IVP
    theta_true = (1.1, 0.7, 0.4, 0.005, 0.02, 0.03) # True theta
    x0 = np.array([1000, 100, 50, 3, 3])
    v0 = mseir(x0, 0, theta_true)
    X0 = np.ravel([x0, v0], 'F')

    # W matrix: dimension is n_eq x sum(n_deriv)
    W_mat = np.zeros((len(n_deriv), sum(n_deriv)+len(n_deriv)))
    for i in range(len(n_deriv)): W_mat[i, sum(n_deriv[:i])+i+1] = 1
    W = zero_pad(W_mat, n_deriv, n_deriv_prior)

    # logprior parameters
    n_theta = len(theta_true)
    phi_sd = np.ones(n_theta)

    # Observation noise
    gamma = 0.2

    # Number of samples to draw from posterior
    n_samples = 100000

    # Initialize inference class and simulate observed data
    inf = inference(state_ind, tmin, tmax, mseir)
    tseq = np.linspace(tmin, tmax, tmax-tmin+1)
    Y_t = inf.simulate(mseir, x0, theta_true, gamma, tseq)

    # Parameter inference using Euler's approximation
    hlst = np.array([0.1, 0.05, 0.02, 0.01, 0.005])
    theta_euler = np.zeros((len(hlst), n_samples, n_theta))
    for i in range(len(hlst)):
        phi_hat, phi_var = inf.phi_fit(Y_t, x0, hlst[i], theta_true, phi_sd, inf.euler_nlpost,
                                       inf.euler_solve, inf.loglike, gamma)
        theta_euler[i] = inf.theta_sample(phi_hat, phi_var, n_samples)
    
    # Parameter inference using Kalman solver
    theta_kalman = np.zeros((len(hlst), n_samples, n_theta))
    for i in range(len(hlst)):
        ode_init= ibm_init(hlst[i], n_deriv_prior, sigma)
        x0_state = zero_pad(X0, n_deriv, n_deriv_prior)
        kinit = indep_init(ode_init, n_deriv_prior)
        n_eval = int((tmax-tmin)/hlst[i])
        kode = KalmanODE(W, tmin, tmax, n_eval, mseir, **kinit)
        inf.kode = kode
        inf.W = W
        phi_hat, phi_var = inf.phi_fit(Y_t, x0_state, hlst[i], theta_true, phi_sd, inf.kalman_nlpost,
                                       inf.kalman_solve, inf.loglike, gamma)
        theta_kalman[i] = inf.theta_sample(phi_hat, phi_var, n_samples)
    
    # Produces the graph in Figure 4
    inf.theta_plot(theta_euler, theta_kalman, theta_true, hlst)
    return

if __name__ == '__main__':
    mseir_example()
