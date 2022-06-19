from math import ceil
from functools import partial
import numpy as np
import scipy as sp
import scipy.stats
import jax
from jax import random, jacfwd, jacrev, grad, lax
import jax.numpy as jnp
import jax.scipy as jsp

from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController
from rodeo.jax.ode_solve import *
import rodeo.jax.ode_bridge_solve as bsol
from euler_solve import euler
#from ode_forward import solve_forward
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import warnings
warnings.filterwarnings('ignore')

mv_jit = jax.jit(solve_sim, static_argnums=(1, 6))
mv_jitb = jax.jit(bsol.solve_sim, static_argnums=(1, 6))
#fw_jit = jax.jit(solve_forward, static_argnums=(1, 6))

class inference:
    r"""
    Perform parameter inference for the model via mode/quadrature using Euler's 
    approximation and the KalmanODE solver. 

    Args:
        key (PRNGKey): PRNG key.
        tmin (int): First time point of the time interval to be evaluated; :math:`a`.
        tmax (int): Last time point of the time interval to be evaluated; :math:`b`.
        fun (function): Higher order ODE function :math:`W x_t = F(x_t, t)` taking 
            arguments :math:`x` and :math:`t`.
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
    def __init__(self, key, tmin, tmax, fun, n_eval=None, W=None, kinit=None):
        self.key = key
        self.tmin = tmin
        self.tmax = tmax
        self.n_eval = n_eval
        self.fun = fun
        self.W = W
        self.kinit = kinit
        self.funpad = None

    def logprior(self, x, mean, sd):
        r"Calculate the loglikelihood of the lognormal distribution."
        return jnp.sum(jsp.stats.norm.logpdf(x=x, loc=mean, scale=sd))

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
    
    def x0_initialize(self, phi, x0, phi_len):
        j = 0
        xx0 = []
        for i in range(len(x0)):
            if x0[i] is None:
                xx0.append(phi[phi_len+j])
                j+=1
            else:
                xx0.append(x0[i])
        return jnp.array(xx0)

    def kalman_nlpost(self, phi, Y_t, x0, step_size, obs_size, phi_mean, phi_sd, *args):
        r"Compute the negative loglikihood of :math:`Y_t` using the KalmanODE."
        phi_ind = len(phi_mean)
        xx0 = self.x0_initialize(phi, x0, phi_ind)
        phi = phi[:phi_ind]
        theta = jnp.exp(phi)
        xx0 = self.funpad(xx0, 0, theta)
        self.key, subkey = jax.random.split(self.key)
        X_t = mv_jit(subkey, self.fun, xx0, theta, self.tmin, self.tmax, self.n_eval, self.W, **self.kinit)
        X_t = X_t[:, :, 0]
        lp = self.loglike(Y_t, X_t, step_size, obs_size, theta, *args)
        lp += self.logprior(phi, phi_mean, phi_sd)
        return -lp


    def bridge_nlpost(self, phi, Y_t, x0, step_size, obs_size, phi_mean, phi_sd, *args):
        phi_ind = len(phi_mean)
        xx0 = self.x0_initialize(phi, x0, phi_ind)
        phi = phi[:phi_ind]
        theta = jnp.exp(phi)
        xx0 = self.funpad(xx0, 0, theta)
        self.key, subkey = jax.random.split(self.key)
        X_t = mv_jitb(subkey, self.fun, xx0, theta, self.tmin, self.tmax, self.n_eval, self.W, **self.kinit)
        X_t = X_t[:, :, 0]
        lp = self.loglike(Y_t, X_t, step_size, obs_size, theta, *args)
        lp += self.logprior(phi, phi_mean, phi_sd)
        return -lp

    # def euler(self, fun, x0, ode_tseq, data_tseq, step_size, theta):
    #     r"Evaluate Euler approximation given :math:`\theta`"
    #     n_eval = len(ode_tseq) - 1

    #     # setup lax.scan:
    #     # scan function
    #     def scan_fun(x_old, t):
    #         x_new = x_old + fun(x_old, self.tmin + step_size*t, theta)*step_size
    #         return x_new, x_new
    #     (_, X_t) = lax.scan(scan_fun, x0, jnp.arange(n_eval))

    #     X_t = jnp.concatenate([x0[None], X_t])
    #     return X_t
    
    def euler_nlpost(self, phi, Y_t, x0, step_size, obs_size, phi_mean, phi_sd, *args):
        r"Compute the negative loglikihood of :math:`Y_t` using the Euler method."
        phi_ind = len(phi_mean)
        xx0 = self.x0_initialize(phi, x0, phi_ind)
        phi = phi[:phi_ind]
        theta = jnp.exp(phi)
        X_t = euler(self.ode_fun, xx0, theta, self.tmin, self.tmax, self.n_eval)
        lp = self.loglike(Y_t, X_t, step_size, obs_size, theta, *args)
        lp += self.logprior(phi, phi_mean, phi_sd)
        return -lp
    
    def diffrax_nlpost(self, phi, Y_t, x0, step_size, obs_size, phi_mean, phi_sd, *args):
        phi_ind = len(phi_mean)
        xx0 = self.x0_initialize(phi, x0, phi_ind)
        phi = phi[:phi_ind]
        theta = jnp.exp(phi)
        term = ODETerm(self.rax_fun)
        solver = Dopri5()
        tseq = jnp.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/step_size)+1)
        saveat = SaveAt(ts=tseq)
        stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
        X_t = diffeqsolve(term, solver, args = theta, t0=self.tmin, t1=self.tmax, dt0=step_size, y0=jnp.array(xx0), saveat=saveat,
                          stepsize_controller=stepsize_controller).ys
        lp = self.loglike(Y_t, X_t, step_size, obs_size, theta, *args)
        lp += self.logprior(phi, phi_mean, phi_sd)
        return -lp

    def phi_fit(self, Y_t, x0, step_size, obs_size, phi_mean, phi_sd, obj_fun, *args, phi_init=None, method="Newton-CG"):
        r"""Compute the optimized :math:`\log{\theta}` and its variance given 
            :math:`Y_t`."""
        if phi_init is None:
            n_theta = len(phi_mean)
            n_x0 = len([i for i in x0 if i is None])
            phi_init = np.zeros(n_theta + n_x0)
        
        n_phi = len(phi_init)
        gradf = grad(obj_fun)
        hes = jacfwd(jacrev(obj_fun))
        opt_res = sp.optimize.minimize(obj_fun, phi_init,
                                       args=(Y_t, x0, step_size, obs_size, phi_mean, phi_sd, *args),
                                       method=method,
                                       jac=gradf)
        phi_hat = opt_res.x
        phi_fisher = hes(phi_hat, Y_t, x0, step_size, obs_size, phi_mean, phi_sd, *args)
        phi_cho, low = jsp.linalg.cho_factor(phi_fisher)
        phi_var = jsp.linalg.cho_solve((phi_cho, low), jnp.eye(n_phi))
        return phi_hat, phi_var
        
    def phi_sample(self, phi_hat, phi_var, n_samples):
        r"""Simulate :math:`\theta` given the :math:`\log{\hat{\theta}}` 
            and its variance."""
        phi = np.random.default_rng(12345).multivariate_normal(phi_hat, phi_var, n_samples)
        return phi
    
    def theta_plot(self, theta_euler, theta_kalman, theta_diffrax, theta_true, step_sizes, var_names, clip=None, rows=1):
        r"""Plot the distribution of :math:`\theta` using the Kalman solver 
            and the Euler approximation."""
        n_hlst, _, n_theta = theta_euler.shape
        ncol = ceil(n_theta/rows) +1
        nrow = 2
        fig = plt.figure(figsize=(20, 10*rows))
        patches = [None]*(n_hlst+2)
        if clip is None:
            clip = [None]*ncol*rows 
        carry = 0
        for t in range(1,n_theta+1):
            row = (t-1)//(ncol-1)
            if t%(ncol)==0:
                carry +=1
            
            axs1 = fig.add_subplot(rows*nrow, ncol, t+row*(ncol)+carry)
            axs2 = fig.add_subplot(rows*nrow, ncol, t+(row+1)*(ncol)+carry)
            axs2.get_shared_x_axes().join(axs2, axs1)
            axs1.set_title(var_names[t-1])
            if (t+carry)%ncol==1:
                axs1.set_ylabel('Euler')
                axs2.set_ylabel('rodeo')
            
            for axs in [axs1, axs2]:
                axs.axvline(x=theta_true[t-1], linewidth=1, color='r', linestyle='dashed')
                axs.set_yticks([])

            for h in range(n_hlst):
                if t==1:
                    patches[h] = mpatches.Patch(color='C{}'.format(h), label='$\\Delta$ t ={}'.format(step_sizes[h]))
                sns.kdeplot(theta_euler[h, :, t-1], ax=axs1, clip=clip[t-1])
                sns.kdeplot(theta_kalman[h, :, t-1], ax=axs2, clip=clip[t-1])
            

            sns.kdeplot(theta_diffrax[:, t-1], ax=axs1, color='black', clip=clip[t-1])
            sns.kdeplot(theta_diffrax[:, t-1], ax=axs2, color='black', clip=clip[t-1])
            if t==n_theta:
                patches[-2] = mpatches.Patch(color='black', label="True Posterior")
                patches[-1] = mlines.Line2D([], [], color='r', linestyle='dashed', linewidth=1, label='True $\\theta$')
                
        fig.legend(handles=patches, framealpha=0.5, loc=7)
        
        fig.tight_layout()
        plt.show()
        return fig

    def theta_plotsingle(self, theta, theta_diffrax, theta_true, step_sizes, var_names, clip=None, rows=1):
        r"""Plot the distribution of :math:`\theta` using the Kalman solver 
            and the Euler approximation."""
        n_hlst, _, n_theta = theta.shape
        ncol = ceil(n_theta/rows) +1
        fig = plt.figure(figsize=(20, 5*rows))
        patches = [None]*(n_hlst+2)
        if clip is None:
            clip = [None]*ncol*rows 
        carry = 0
        for t in range(1,n_theta+1):
            row = (t-1)//(ncol-1)
            if t%(ncol)==0:
                carry +=1
            
            axs = fig.add_subplot(rows, ncol, t+carry)
            axs.set_title(var_names[t-1])
            axs.axvline(x=theta_true[t-1], linewidth=1, color='r', linestyle='dashed')
            axs.set_yticks([])

            for h in range(n_hlst):
                if t==1:
                    patches[h] = mpatches.Patch(color='C{}'.format(h), label='$\\Delta$ t ={}'.format(step_sizes[h]))
                sns.kdeplot(theta[h, :, t-1], ax=axs, clip=clip[t-1])
            
            sns.kdeplot(theta_diffrax[:, t-1], ax=axs, color='black', clip=clip[t-1])
            
            if t==n_theta:
                patches[-2] = mpatches.Patch(color='black', label="True Posterior")
                patches[-1] = mlines.Line2D([], [], color='r', linestyle='dashed', linewidth=1, label='True $\\Theta$')
                
        fig.legend(handles=patches, framealpha=0.5, loc=7)
        
        fig.tight_layout()
        plt.show()
        return fig
