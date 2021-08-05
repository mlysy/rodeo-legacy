from jax import grad, jacfwd, jacrev
from kalmanODE_jax import *
import jax.numpy as jnp
import numpy as np
import scipy as sp
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import warnings
warnings.filterwarnings('ignore')

class inference:
    r"""
    Perform parameter inference for the model via mode/quadrature using Euler's 
    approximation and the KalmanODE solver. 

    Args:
        state_ind (list): Index of the 0th derivative, :math:`x^{(0)}` of each state.
        tmin (int): First time point of the time interval to be evaluated; :math:`a`.
        tmax (int): Last time point of the time interval to be evaluated; :math:`b`.
        fun (function): Higher order ODE function :math:`W x_t = F(x_t, t)` taking 
            arguments :math:`x` and :math:`t`.
        n_eval (int): Number of evaluation points; :math:`N`.
        data_tseq (ndarray(n)): Time points of the observed data.
        ode_tseq (ndarray(N)): Time points of the ODE solution.
        x0 (ndarray(n_state)): Initial value of the state variable :math:`x_t` at 
            time :math:`t = 0`.
        gamma (float): Noise parameter to simulate the observations.
        phi (ndarray(n_theta)): Log of observed :math:`\theta`.
        Y_t (ndarray(n_steps, n_state)): Simulated observations.
        step_size (float): Distance between discretisation points.
        phi_true (ndarray(n_theta)): True value of :math:`\phi = \log{\theta}` in the ODE function.
        phi_sd (ndarray(n_theta)): Standard deviation of :math:`\phi`.
        kalman_solve (fun): Kalman solve method defined by the parameter inference problem.
        euler_solve (fun): Euler solve method defined by the parameter inference problem.
        solve (fun): Either kalman_solve or euler_solve.
        theta (ndarray(n_theta)): Observed :math:`\theta`.
        phi_hat (ndarray(n_theta)): Optimized observed :math:`\phi`.
        phi_var (ndarray(n_theta, n_theta)): Variance matrix of phi_hat.
        n_samples (int): Number of samples of :math:`\theta` to simulate.
        kalman_phi (ndarray(n_samples, n_theta)): Simulated n_samples of 
            :math:`\phi` using KalmanODE solver.
    """
    def __init__(self, state_ind, tmin, tmax, n_eval, fun, W):
        self.state_ind = state_ind
        self.tmin = tmin
        self.tmax = tmax
        self.fun = fun
        self.W = W
        self.n_eval = n_eval
        self.f = None

    def thinning(self, data_tseq, ode_tseq, X):
        r"Thin a highly discretized ODE solution to match the observed data."
        data_i = 0
        ode_i = 0
        diff = 1000
        ind = np.zeros(len(data_tseq), dtype=int)
        while data_i < len(data_tseq) and ode_i < len(ode_tseq):
            if data_tseq[data_i] > ode_tseq[ode_i]:
                diff = min(diff, abs(data_tseq[data_i] - ode_tseq[ode_i]))
                ode_i+=1
            else:
                if diff > abs(data_tseq[data_i] - ode_tseq[ode_i]):
                    ind[data_i] = ode_i
                else:
                    ind[data_i] = ode_i-1
                data_i+=1
                diff = 1000
        return X[ind,:]
    
    def kalman_nlpost(self, phi, Y_t, deltat, h, phi_mean, phi_sd, prior, x0=None, sigma=None, log=False):
        r"Compute the negative loglikihood of :math:`Y_t` using the KalmanODE."
        phi_ind = len(phi_mean)
        theta = jnp.exp(phi[0:len(phi_mean)])
        if sigma is None:
            sigma = jnp.exp(phi[phi_ind])
            phi_ind += 1
        if None in x0:
            len_x0 = len(x0)
            x00 = phi[phi_ind:phi_ind+len_x0]
            x0 = self.f(x00, 0, theta)
        phi = phi[0:len(phi_mean)]
        X_t = solve_sim(self.fun, x0, self.tmin, self.tmax, self.n_eval, self.W, **prior, theta=theta)
        X_t = X_t[::int(deltat/h)][1:][:, self.state_ind]
        if log:
            X_t = jnp.exp(X_t)
        lp = self.loglike(Y_t, X_t, sigma)
        lp += self.loglike(phi, phi_mean, phi_sd)
        return -lp

    def phi_fit(self, Y_t, x0, deltat, h, phi_true, phi_sd, prior, sigma=None, log=False):
        r"""Compute the optimized :math:`\log{\theta}` and its variance given 
            :math:`Y_t`."""
        n_theta = len(phi_true)    
        if sigma is None:
            n_theta += 1
        if None in x0:
            n_theta += len(x0)
        phi_init = jnp.zeros(n_theta)
        kalman_grad = grad(self.kalman_nlpost)
        kalman_hes = jacfwd(jacrev(self.kalman_nlpost))
        opt_res = sp.optimize.minimize(self.kalman_nlpost, phi_init,
                                       args=(Y_t, deltat, h, phi_true, phi_sd, prior, x0, sigma, log),
                                       method='Newton-CG',
                                       jac=kalman_grad)
        phi_fisher = kalman_hes(opt_res.x, Y_t, deltat, h, phi_true, phi_sd, prior, x0, sigma, log)
        phi_cho, low = sp.linalg.cho_factor(phi_fisher)
        phi_var = sp.linalg.cho_solve((phi_cho, low), jnp.eye(n_theta))
        return opt_res.x, phi_var

    def theta_sample(self, phi_hat, phi_var, n_samples):
        r"""Simulate :math:`\log{\theta}` given the :math:`\log{\hat{\theta}}` 
            and its variance."""
        phi = np.random.multivariate_normal(phi_hat, phi_var, n_samples)
        return phi
    
    def theta_plot(self, kalman_phi, phi_true, var_names):
        r"Plot the posterior distribution against the true theta."
        kalman_theta = jnp.exp(kalman_phi)
        theta_true = jnp.exp(phi_true)
        n_theta = len(theta_true)
        fig, axs = plt.subplots(1, n_theta, figsize=(20, 10))
        for i in range(n_theta):
            sns.kdeplot(kalman_theta[:,i], ax=axs[i])
            axs[i].axvline(x=theta_true[i], linewidth=1, color='r', linestyle='dashed')
            axs[i].set_title(var_names[i])
        plt.close()
        return fig

