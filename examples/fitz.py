import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import jit

from inference.normal import normal as inference
from rodeo.ibm import ibm_init
from rodeo.jax.KalmanODE import *
from rodeo.utils import indep_init, zero_pad

@jit
def fitz(X_t, t, theta):
    "Fitz ODE written for jax"
    a, b, c = theta
    p = len(X_t)//2
    V, R = X_t[0], X_t[p]
    return jnp.array([c*(V - V*V*V/3 + R),
                      -1/c*(V - a + b*R)])

def fitzpad(X_t, t, theta):
    a, b, c = theta
    p = len(X_t)//2
    V, R = X_t[0], X_t[p]
    return jnp.array([V, c*(V - V*V*V/3 + R), 0,
                      R, -1/c*(V - a + b*R), 0])

def fitz_example(load_calcs=False):
    "Perform parameter inference using the FitzHugh-Nagumo function."
    # These parameters define the order of the ODE and the CAR(p) process
    n_deriv = [1, 1] # Total state
    n_deriv_prior = [3, 3]
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
    W = jnp.array(zero_pad(w_mat, n_deriv, n_deriv_prior))

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
    #fig.savefig('figures/fitzsim.pdf')
    
    dtlst = np.array([0.1])
    obs_t = 1
    if load_calcs:
        theta_euler = np.load('saves/fitz_theta_euler3.npy')
        theta_kalman = np.load('saves/fitz_theta_kalman3.npy')
    else:
        # Parameter inference using Euler's approximation
        theta_euler = np.zeros((len(dtlst), n_samples, n_theta+n_var))
        phi_init = np.append(np.log(theta_true), x0)
        for i in range(len(dtlst)):
            phi_hat, phi_var = inf.phi_fit(Y_t, np.array([None, None]), dtlst[i], obs_t, phi_mean, phi_sd, inf.euler_nlpost,
                                           inf.euler_solve, inf.loglike, gamma, phi_init=phi_init)
            theta_euler[i] = inf.phi_sample(phi_hat, phi_var, n_samples)
            theta_euler[i, :, :n_theta] = np.exp(theta_euler[i, :, :n_theta])
        #np.save('saves/fitz_theta_euler3.npy', theta_euler)
        
        # Parameter inference using Kalman solver
        theta_kalman = np.zeros((len(dtlst), n_samples, n_theta+n_var))
        for i in range(len(dtlst)):
            ode_init = ibm_init(dtlst[i], n_deriv_prior, sigma)
            kinit = indep_init(ode_init, n_deriv_prior)
            kinit = dict((k, jnp.array(v)) for k, v in kinit.items())
            n_eval = int((tmax-tmin)/dtlst[i])
            inf.n_eval = n_eval
            inf.kinit = kinit
            inf.W = W
            phi_hat, phi_var = inf.phi_fit(Y_t, np.array([None, None]), dtlst[i], obs_t, phi_mean, phi_sd, inf.kalman_nlpost,
                                           inf.kalman_solve, inf.loglike, gamma, phi_init = phi_init)
            theta_kalman[i] = inf.phi_sample(phi_hat, phi_var, n_samples)
            theta_kalman[i, :, :n_theta] = np.exp(theta_kalman[i, :, :n_theta])
        #np.save('saves/fitz_theta_kalman3.npy', theta_kalman)
        
    # Produces the graph in Figure 3
    plt.rcParams.update({'font.size': 20})
    var_names = ['a', 'b', 'c', r"$V_t^{(0)}$", r"$R_t^{(0)}$"]
    param_true = np.append(theta_true, np.array([-1, 1]))
    figure = inf.theta_plot(theta_euler, theta_kalman, param_true, dtlst, var_names, clip=[None, (0, 0.5), None, None, None], rows=2)
    #figure.savefig('figures/fitzfigure.pdf')
    return

if __name__ == '__main__':
    fitz_example(False)
    