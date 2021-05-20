from scipy.integrate import odeint
from math import ceil
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
    
    def loglikep(self, x, mean):
        r"Calculate the loglikelihood of the poisson distribution."
        return np.sum(sp.stats.poisson.logpmf(x, mean))
    
    def covid_obs(self, X_t, theta):
        r"""Get the observations for the SEIRAH Covid example.
        None of the compartments are directly observed, however 
        the daily infections and hospitalizations are observed. 
        They can be computed as
        
        .. math::

            I^{(in)}(t) = rE(t)/D_e
            H^{(in)}(t) = I(t)/D_q

        """
        I_in = theta[1]*X_t[:,1]/theta[3]
        H_in = X_t[:,2]/theta[5]
        X_in = np.array([I_in, H_in]).T
        return X_in

    def simulate(self, fun, x0, theta_true, gamma):
        r"Simulate observed data for inference"
        tseq = np.linspace(self.tmin, self.tmax, self.tmax-self.tmin+1)
        X_t = odeint(fun, x0, tseq, args=(theta_true,))[1:,]
        e_t = np.random.default_rng().normal(loc=0.0, scale=1, size=X_t.shape)
        Y_t = X_t + gamma*e_t
        return Y_t, X_t

    def simulatep(self, fun, x0, theta_true):
        r"Simulate observed data for SEIRAH Covid model."
        tseq = np.linspace(self.tmin, self.tmax, self.tmax-self.tmin+1)
        X_t = odeint(fun, x0, tseq, args=(theta_true,))[1:,]
        X_in = self.covid_obs(X_t, theta_true)
        Y_in = np.random.default_rng().poisson(X_in)
        return Y_in, X_in

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
        r"Compute the negative loglikihood of :math:`Y_t` using the KalmanODE."
        theta = np.exp(phi)
        data_tseq = np.linspace(1, self.tmax, self.tmax-self.tmin)
        ode_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/step_size)+1)
        X_t = self.kode.solve_mv(x0, self.W, theta)[0]
        X_t = self.thinning(data_tseq, ode_tseq, X_t)[:, self.state_ind]
        lp = self.loglike(Y_t, X_t, gamma)
        lp += self.loglike(phi, phi_mean, phi_sd)
        return -lp
    
    def kalman_nlpostp(self, phi, Y_t, x0, step_size, phi_mean, phi_sd):
        r"Compute the negative loglikihood of :math:`Y_t` using the KalmanODE for the SEIRAH example."
        theta = np.exp(phi)
        data_tseq = np.linspace(1, self.tmax, self.tmax-self.tmin)
        ode_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/step_size)+1)
        X_t = self.kode.solve_mv(x0, self.W, theta)[0]
        X_t = self.thinning(data_tseq, ode_tseq, X_t)[:, self.state_ind]
        X_in = self.covid_obs(X_t, theta)
        lp = self.loglikep(Y_t, X_in)
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
    
    def euler_nlpostp(self, phi, Y_t, x0, step_size, phi_mean, phi_sd):
        r"Compute the negative loglikihood of :math:`Y_t` using the Euler method for the SEIRAH example."
        theta = np.exp(phi)
        X_t = self.euler(x0, step_size, theta)
        X_in = self.covid_obs(X_t, theta)
        lp = self.loglikep(Y_t, X_in)
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
        phi_init = phi_mean + .1
        opt_res = sp.optimize.minimize(obj_fun, phi_init,
                                    args=(Y_t, x0, step_size, phi_mean, phi_sd, gamma),
                                    method='Nelder-Mead')
        phi_hat = opt_res.x
        hes = nd.Hessian(obj_fun)
        phi_fisher = hes(phi_hat, Y_t, x0, step_size, phi_mean, phi_sd, gamma)
        phi_cho, low = sp.linalg.cho_factor(phi_fisher)
        phi_var = sp.linalg.cho_solve((phi_cho, low), np.eye(n_theta))
        return phi_hat, phi_var
    
    def phi_fitp(self, Y_t, x0, step_size, theta_true, phi_sd, kalman):
        r"""Compute the optimized :math:`\log{\theta}` and its variance given 
            :math:`Y_t` for the SEIRAH model."""
        phi_mean = np.log(theta_true)
        n_theta = len(theta_true)
        if kalman:
            obj_fun = self.kalman_nlpostp
        else:
            obj_fun = self.euler_nlpostp
        phi_init = phi_mean + .1
        opt_res = sp.optimize.minimize(obj_fun, phi_init,
                                    args=(Y_t, x0, step_size, phi_mean, phi_sd),
                                    method='Nelder-Mead')
        phi_hat = opt_res.x
        hes = nd.Hessian(obj_fun)
        phi_fisher = hes(phi_hat, Y_t, x0, step_size, phi_mean, phi_sd)
        phi_cho, low = sp.linalg.cho_factor(phi_fisher)
        phi_var = sp.linalg.cho_solve((phi_cho, low), np.eye(n_theta))
        return phi_hat, phi_var

    def theta_sample(self, phi_hat, phi_var, n_samples):
        r"""Simulate :math:`\theta` given the :math:`\log{\hat{\theta}}` 
            and its variance."""
        phi = np.random.multivariate_normal(phi_hat, phi_var, n_samples)
        theta = np.exp(phi)
        return theta
    
    def theta_plot(self, theta_euler, theta_kalman, theta_true, step_sizes, clip=None, rows=1):
        r"""Plot the distribution of :math:`\theta` using the Kalman solver 
            and the Euler approximation."""
        n_hlst, _, n_theta = theta_euler.shape
        ncol = ceil(n_theta/rows)
        nrow = 2
        fig = plt.figure(figsize=(20, 10*rows))
        patches = [None]*(n_hlst+1)
        if clip is None:
            clip = [None]*ncol*rows 

        for t in range(1,n_theta+1):
            row = (t-1)//ncol
            axs1 = fig.add_subplot(rows*nrow, ncol, t+row*ncol)
            axs2 = fig.add_subplot(rows*nrow, ncol, t+(row+1)*ncol)
            axs2.get_shared_x_axes().join(axs2, axs1)
            if t%ncol==1:
                axs1.set_ylabel('Euler')
                axs2.set_ylabel('rodeo')
            
            for axs in [axs1, axs2]:
                axs.axvline(x=theta_true[t-1], linewidth=1, color='r', linestyle='dashed')
                axs.locator_params(axis='x', tight=True, nbins=3)
                axs.set_yticks([])

            for h in range(n_hlst):
                if t==1:
                    patches[h] = mpatches.Patch(color='C{}'.format(h), label='h={}'.format(step_sizes[h]))
                sns.kdeplot(theta_euler[h, :, t-1], ax=axs1, clip=clip[t-1])
                sns.kdeplot(theta_kalman[h, :, t-1], ax=axs2, clip=clip[t-1])
            
            if t==n_theta:
                patches[-1] = mlines.Line2D([], [], color='r', linestyle='dashed', linewidth=1, label='True $\\theta$')
                axs2.legend(handles=patches, framealpha=0.5)
        
        fig.tight_layout()
        plt.show()
        return fig
