import numpy as np
import matplotlib.pyplot as plt
from inference.normal import normal as inference
from rodeo.ibm import ibm_init
from rodeo.cython.KalmanODE import KalmanODE
from rodeo.utils import indep_init, zero_pad

def fitz(X_t, t, theta, out=None):
    "FN ODE function with optional overwriting of output."
    if out is None:
        out = np.empty(2)
    a, b, c = theta
    p = len(X_t)//2
    V, R = X_t[0], X_t[p]
    out[0] = c*(V - V*V*V/3 + R)
    out[1] = -1/c*(V - a + b*R)
    return out

def fitzpad(X_t, t, theta):
    a, b, c = theta
    p = len(X_t)//2
    V, R = X_t[0], X_t[p]
    return np.array([V, c*(V - V*V*V/3 + R), 0,
                     R, -1/c*(V - a + b*R), 0])

def fitz_example(load_calcs=False):
    "Perform parameter inference using the FitzHugh-Nagumo function."
    # These parameters define the order of the ODE and the CAR(p) process
    n_deriv = [1, 1] # Total state
    n_obs = 2 # Total measures
    n_deriv_prior = [3, 3]
    p = sum(n_deriv_prior)
    state_ind = [0, 3] # Index of 0th derivative of each state

    # it is assumed that the solution is sought on the interval [tmin, tmax].
    tmin = 0 
    tmax = 40

    # The rest of the parameters can be tuned according to ODE
    # For this problem, we will use
    n_var = 2
    sigma = [.1]*n_var

    # Initial value, x0, for the IVP
    x0 = np.array([-1., 1.])
    v0 = np.array([1, 1/3])
    X0 = np.ravel([x0, v0], 'F')

    # pad the inputs
    w_mat = np.array([[0., 1., 0., 0.], [0., 0., 0., 1.]])
    W = zero_pad(w_mat, n_deriv, n_deriv_prior)

    # logprior parameters
    theta_true = np.array([0.2, 0.2, 3]) # True theta
    n_theta = len(theta_true)
    phi_mean = np.zeros(n_theta)
    phi_sd = np.log(10)*np.ones(n_theta) 

    # Observation noise
    gamma = 0.2

    # Number of samples to draw from posterior
    n_samples = 100000

    # Initialize inference class and simulate observed data
    inf = inference(state_ind, tmin, tmax, fitz)
    inf.funpad = fitzpad
    tseq0 = np.linspace(tmin, tmax, tmax-tmin+1)
    Y_t, X_t = inf.simulate(fitz, x0, theta_true, gamma, tseq0)
    tseq = np.linspace(tmin, tmax, 41)
    plt.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].plot(tseq, X_t[:,0], label = 'X_t')
    axs[0].scatter(tseq, Y_t[:,0], label = 'Y_t', color='orange')
    axs[0].set_title("$V^{(0)}_t$")
    axs[1].plot(tseq, X_t[:,1], label = 'X_t')
    axs[1].scatter(tseq, Y_t[:,1], label = 'Y_t', color='orange')
    axs[1].set_title("$R^{(0)}_t$")
    axs[1].legend(loc='upper left', bbox_to_anchor=[1, 1])
    fig.savefig('figures/fitzsim.pdf')
    
    hlst = np.array([0.1, 0.05, 0.02, 0.01, 0.005])
    if load_calcs:
        theta_euler = np.load('saves/fitz_theta_euler.npy')
        theta_kalman = np.load('saves/fitz_theta_kalman.npy')
    else:
        # Parameter inference using Euler's approximation
        theta_euler = np.zeros((len(hlst), n_samples, n_theta))
        phi_init = np.append(np.log(theta_true), x0)
        for i in range(len(hlst)):
            print(hlst[i])
            phi_hat, phi_var = inf.phi_fit(Y_t, np.array([None, None]), hlst[i], phi_mean, phi_sd, inf.euler_nlpost,
                                           inf.euler_solve, inf.loglike, gamma, phi_init=phi_init)
            theta_euler[i] = inf.theta_sample(phi_hat[:3], phi_var[:3, :3], n_samples)
        np.save('saves/fitz_theta_euler.npy', theta_euler)
        
        # Parameter inference using Kalman solver
        theta_kalman = np.zeros((len(hlst), n_samples, n_theta))
        for i in range(len(hlst)):
            print(hlst[i])
            ode_init = ibm_init(hlst[i], n_deriv_prior, sigma)
            kinit = indep_init(ode_init, n_deriv_prior)
            n_eval = int((tmax-tmin)/hlst[i])
            kode = KalmanODE(W, tmin, tmax, n_eval, fitz, **kinit)
            inf.kode = kode
            inf.W = W
            phi_hat, phi_var = inf.phi_fit(Y_t, np.array([None, None]), hlst[i], phi_mean, phi_sd, inf.kalman_nlpost,
                                           inf.kalman_solve, inf.loglike, gamma, phi_init = phi_init)
            theta_kalman[i] = inf.theta_sample(phi_hat[:3], phi_var[:3, :3], n_samples)
        np.save('saves/fitz_theta_kalman.npy', theta_kalman)
        
    # Produces the graph in Figure 3
    plt.rcParams.update({'font.size': 20})
    var_names = ['a', 'b', 'c']
    figure = inf.theta_plot(theta_euler, theta_kalman, theta_true, hlst, var_names, clip=[None, (0, 0.5), None])
    figure.savefig('figures/fitzfigure.pdf')
    return

if __name__ == '__main__':
    fitz_example()
    