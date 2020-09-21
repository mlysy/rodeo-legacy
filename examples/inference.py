from scipy.integrate import odeint
import numpy as np
import scipy as sp
import scipy.stats
import numdifftools as nd
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
        kode (object): KalmanODE solver used to perform parameter inference.
        x0 (ndarray(n_state)): Initial value of the state variable :math:`x_t` at 
            time :math:`t = 0`.
        theta_true (ndarray(n_theta)): True value of :math:`\theta` in the ODE function.
        gamma (float): Noise parameter to simulate the observations.
        phi (ndarray(n_theta)): Log of observed :math:`\theta`.
        Y_t (ndarray(n_steps, n_state)): Simulated observations.
        step_size (float): Distance between discretisation points.
        phi_mean (ndarray(n_theta)): Mean of :math:`\phi`.
        phi_sd (ndarray(n_theta)): Standard deviation of :math:`\phi`.
        theta (ndarray(n_theta)): Observed :math:`\theta`.
        kalman (bool): Flag to indicate if the KalmanODE solver or Euler's method 
            is used.
        phi_hat (ndarray(n_theta)): Optimized observed :math:`\phi`.
        phi_var (ndarray(n_theta, n_theta)): Variance matrix of phi_hat.
        n_samples (int): Number of samples of :math:`\theta` to simulate.
        theta_euler (ndarray(n_samples, n_theta)): Simulated n_samples of 
            :math:`\theta` using Euler's approximation.
        theta_kalman (ndarray(n_samples, n_theta)): Simulated n_samples of 
            :math:`\theta` using KalmanODE solver.
    """
    def __init__(self, state_ind, tmin, tmax, fun, W=None, kode=None):
        self.state_ind = state_ind
        self.tmin = tmin
        self.tmax = tmax
        self.fun = fun
        self.W = W
        self.kode = kode
    
    def loglike(self, x, mean, var):
        r"Calculate the loglikelihood of the lognormal distribution."
        return np.sum(sp.stats.norm.logpdf(x=x, loc=mean, scale=var))
    
    def simulate(self, fun, x0, theta_true, gamma):
        r"Simulate observed data for inference"
        tseq = np.linspace(self.tmin, self.tmax, self.tmax-self.tmin+1)
        X_t = odeint(fun, x0, tseq, args=(theta_true,))[1:,]
        e_t = np.random.default_rng().normal(loc=0.0, scale=1, size=X_t.shape)
        Y_t = X_t + gamma*e_t
        return Y_t

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

    def kalman_nlpost(self, phi, Y_t, x0, step_size, phi_mean, phi_sd, gamma):
        r"Compute the negative loglikihood of :math:`Y_t` using the Euler method."
        theta = np.exp(phi)
        data_tseq = np.linspace(1, self.tmax, self.tmax-self.tmin)
        ode_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/step_size)+1)
        X_t = self.kode.solve(x0, self.W, theta)
        X_t = self.thinning(data_tseq, ode_tseq, X_t)[:, self.state_ind]
        lp = self.loglike(Y_t, X_t, gamma)
        lp += self.loglike(phi, phi_mean, phi_sd)
        return -lp
        
    def euler(self, x0, step_size, theta):
        r"Evaluate Euler approximation given :math:`\theta`"
        n_eval = int((self.tmax-self.tmin)/step_size)
        data_tseq = np.linspace(1, self.tmax, self.tmax-self.tmin)
        ode_tseq = np.linspace(self.tmin, self.tmax, n_eval+1)
        X_t = np.zeros((n_eval+1, len(x0)))
        X_t[0] = x0
        for i in range(n_eval):
            self.fun(X_t[i], step_size*i, theta, X_t[i+1])
            X_t[i+1] = X_t[i] + X_t[i+1]*step_size
        X_t = self.thinning(data_tseq, ode_tseq, X_t)
        return X_t
    
    def euler_nlpost(self, phi, Y_t, x0, step_size, phi_mean, phi_sd, gamma):
        r"Compute the negative loglikihood of :math:`Y_t` using the Euler method."
        theta = np.exp(phi)
        X_t = self.euler(x0, step_size, theta)
        lp = self.loglike(Y_t, X_t, gamma)
        lp += self.loglike(phi, phi_mean, phi_sd)
        return -lp
    
    def phi_fit(self, Y_t, x0, step_size, theta_true, phi_sd, gamma, kalman):
        r"""Compute the optimized :math:`\log{\theta}` and its variance given 
            :math:`Y_t`."""
        phi_mean = np.log(theta_true)
        n_theta = len(theta_true)
        if kalman:
            obj_fun = self.kalman_nlpost
        else:
            obj_fun = self.euler_nlpost
        opt_res = sp.optimize.minimize(obj_fun, phi_mean+.1,
                                    args=(Y_t, x0, step_size, phi_mean, phi_sd, gamma),
                                    method='Nelder-Mead')
        phi_hat = opt_res.x
        hes = nd.Hessian(obj_fun)
        phi_fisher = hes(phi_hat, Y_t, x0, step_size, phi_mean, phi_sd, gamma)
        phi_cho, low = sp.linalg.cho_factor(phi_fisher)
        phi_var = sp.linalg.cho_solve((phi_cho, low), np.eye(n_theta))
        return phi_hat, phi_var
    
    def theta_sample(self, phi_hat, phi_var, n_samples):
        r"""Simulate :math:`\theta` given the :math:`\log{\hat{\theta}}` 
            and its variance."""
        phi = np.random.multivariate_normal(phi_hat, phi_var, n_samples)
        theta = np.exp(phi)
        return theta
    
    def theta_plot(self, theta_euler, theta_kalman, theta_true, step_sizes):
        r"""Plot the distribution of :math:`\theta` using the Kalman solver 
            and the Euler approximation."""
        n_size, _, n_theta = theta_euler.shape
        nrow = 2
        fig, axs = plt.subplots(nrow, n_theta, sharex='col', figsize=(20, 5))
        patches = [None]*(n_size+1)
        for col in range(n_theta):
            axs[0, col].set_title('$\\theta_{}$'.format(col))
            for row in range(nrow):
                axs[row, col].axvline(x=theta_true[col], linewidth=1, color='r', linestyle='dashed')
                axs[row, col].locator_params(axis='x', nbins=3)
                axs[row, col].set_yticks([])
            for i in range(n_size):
                if col==0:
                    patches[i] = mpatches.Patch(color='C{}'.format(i), label='h={}'.format(step_sizes[i]))
                sns.kdeplot(theta_euler[i, :, col], ax=axs[0, col])
                sns.kdeplot(theta_kalman[i, :, col], ax=axs[1, col])

        axs[0, 0].set_ylabel('Euler')
        axs[1, 0].set_ylabel('KalmanODE')
        patches[-1] = mlines.Line2D([], [], color='r', linestyle='dashed', linewidth=1, label='True $\\theta$')
        axs[0, -1].legend(handles=patches)
        fig.tight_layout()
        plt.show()
        return