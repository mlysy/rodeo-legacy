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
        data_tseq (ndarray(n)): Time points of the observed data.
        ode_tseq (ndarray(N)): Time points of the ODE solution.
        x0 (ndarray(n_state)): Initial value of the state variable :math:`x_t` at 
            time :math:`t = 0`.
        theta_true (ndarray(n_theta)): True value of :math:`\theta` in the ODE function.
        gamma (float): Noise parameter to simulate the observations.
        phi (ndarray(n_theta)): Log of observed :math:`\theta`.
        Y_t (ndarray(n_steps, n_state)): Simulated observations.
        step_size (float): Distance between discretisation points.
        phi_mean (ndarray(n_theta)): Mean of :math:`\phi`.
        phi_sd (ndarray(n_theta)): Standard deviation of :math:`\phi`.
        kalman_solve (fun): Kalman solve method defined by the parameter inference problem.
        euler_solve (fun): Euler solve method defined by the parameter inference problem.
        solve (fun): Either kalman_solve or euler_solve.
        theta (ndarray(n_theta)): Observed :math:`\theta`.
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
        self.funpad = None
    
    def loglike(self, x, mean, var):
        r"Calculate the loglikelihood of the lognormal distribution."
        return np.sum(sp.stats.norm.logpdf(x=x, loc=mean, scale=var))

    def thinning(self, ode_tseq, data_tseq, X):
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
    
    
    def kalman_nlpost(self, phi, Y_t, x0, step_size, obs_size, phi_mean, phi_sd, kalman_solve, loglik, *args):
        r"Compute the negative loglikihood of :math:`Y_t` using the KalmanODE."
        phi_ind = len(phi_mean)
        j = 0
        xx0 = np.copy(x0)
        for i in range(len(x0)):
            if x0[i] is None:
                xx0[i] = phi[phi_ind + j]
                j+=1
        phi = phi[:phi_ind]
        theta = np.exp(phi)
        xx0 = self.funpad(xx0, 0, theta)
        X_t = kalman_solve(step_size, obs_size, xx0, theta)
        lp = loglik(Y_t, X_t, *args)
        lp += self.loglike(phi, phi_mean, phi_sd)
        return -lp

    def euler(self, x0, ode_tseq, data_tseq, step_size, theta):
        r"Evaluate Euler approximation given :math:`\theta`"
        n_eval = len(ode_tseq) - 1
        X_t = np.zeros((n_eval+1, len(x0)))
        X_t[0] = x0
        for i in range(n_eval):
            self.fun(X_t[i], step_size*i, theta, X_t[i+1])
            X_t[i+1] = X_t[i] + X_t[i+1]*step_size
        X_t = self.thinning(ode_tseq, data_tseq, X_t)
        return X_t
    
    def euler_nlpost(self, phi, Y_t, x0, step_size, obs_size, phi_mean, phi_sd, euler_solve, loglik, *args):
        r"Compute the negative loglikihood of :math:`Y_t` using the Euler method."
        phi_ind = len(phi_mean)
        j=0
        xx0 = np.copy(x0)
        for i in range(len(x0)):
            if x0[i] is None:
                xx0[i] = phi[phi_ind+j]
                j+=1
        phi = phi[:phi_ind]
        theta = np.exp(phi)
        X_t = euler_solve(xx0, step_size, obs_size, theta)
        lp = loglik(Y_t, X_t, *args)
        lp += self.loglike(phi, phi_mean, phi_sd)
        return -lp
    
    def phi_fit(self, Y_t, x0, step_size, obs_size, phi_mean, phi_sd, obj_fun, solve, loglik, *args, phi_init=None, bounds=None):
        r"""Compute the optimized :math:`\log{\theta}` and its variance given 
            :math:`Y_t`."""
        if phi_init is None:
            n_theta = len(phi_mean)
            n_x0 = len([i for i in x0 if i is None])
            phi_init = np.zeros(n_theta + n_x0)
        
        n_phi = len(phi_init)
        opt_res = sp.optimize.minimize(obj_fun, phi_init,
                                       args=(Y_t, x0, step_size, obs_size, phi_mean, phi_sd, solve, loglik, *args),
                                       method='Nelder-Mead',
                                       bounds=bounds)
        phi_hat = opt_res.x
        hes = nd.Hessian(obj_fun)
        phi_fisher = hes(phi_hat, Y_t, x0, step_size, obs_size,phi_mean, phi_sd, solve, loglik, *args)
        phi_cho, low = sp.linalg.cho_factor(phi_fisher)
        phi_var = sp.linalg.cho_solve((phi_cho, low), np.eye(n_phi))
        return phi_hat, phi_var

    def phi_sample(self, phi_hat, phi_var, n_samples):
        r"""Simulate :math:`\theta` given the :math:`\log{\hat{\theta}}` 
            and its variance."""
        phi = np.random.multivariate_normal(phi_hat, phi_var, n_samples)
        return phi
    
    def theta_plot(self, theta_euler, theta_kalman, theta_true, step_sizes, var_names, clip=None, rows=1):
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
            axs1.set_title(var_names[t-1])
            if t%ncol==1:
                axs1.set_ylabel('Euler')
                axs2.set_ylabel('rodeo')
            
            for axs in [axs1, axs2]:
                axs.axvline(x=theta_true[t-1], linewidth=1, color='r', linestyle='dashed')
                axs.locator_params(axis='x', tight=True, nbins=3)
                axs.set_yticks([])

            for h in range(n_hlst):
                if t==1:
                    patches[h] = mpatches.Patch(color='C{}'.format(h), label='$\\Delta$ t ={}'.format(step_sizes[h]))
                sns.kdeplot(theta_euler[h, :, t-1], ax=axs1, clip=clip[t-1])
                sns.kdeplot(theta_kalman[h, :, t-1], ax=axs2, clip=clip[t-1])
            
            if t==n_theta:
                patches[-1] = mlines.Line2D([], [], color='r', linestyle='dashed', linewidth=1, label='True $\\theta$')
                fig.legend(handles=patches, framealpha=0.5, loc="best")
        
        fig.tight_layout()
        plt.show()
        return fig
