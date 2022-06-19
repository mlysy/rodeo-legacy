import numpy as np
import scipy as sp
import jax
from jax import grad
import jax.numpy as jnp
from jax.config import config
import matplotlib.pyplot as plt

from diffrax import diffeqsolve, Dopri5, ODETerm, SaveAt, PIDController
from rodeo.jax.ibm_init import ibm_init
from rodeo.jax.ode_solve import *

config.update("jax_enable_x64", True)


def logprior(x, mean, sd):
    r"Calculate the loglikelihood of the lognormal distribution."
    return jnp.sum(jsp.stats.norm.logpdf(x=x, loc=mean, scale=sd))

def thinning(ode_tseq, data_tseq, X):
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

def x0_initialize(phi, x0, phi_len):
    j = 0
    xx0 = []
    for i in range(len(x0)):
        if x0[i] is None:
            xx0.append(phi[phi_len+j])
            j+=1
        else:
            xx0.append(x0[i])
    return jnp.array(xx0)

def kalman_nlpost(phi, Y_t, x0, step_size, obs_size, phi_mean, phi_sd, *args):
    r"Compute the negative loglikihood of :math:`Y_t` using the KalmanODE."
    phi_ind = len(phi_mean)
    xx0 = x0_initialize(phi, x0, phi_ind)
    phi = phi[:phi_ind]
    theta = jnp.exp(phi)
    xx0 = ode_pad(xx0, 0, theta)
    key = jax.random.PRNGKey(0)
    X_t = mv_jit(key, ode_fun, xx0, theta, tmin, tmax, n_eval, W, **kinit)[0]
    X_t = X_t[:, :, 0]
    lp = loglike(Y_t, X_t, step_size, obs_size, theta, *args)
    lp += logprior(phi, phi_mean, phi_sd)
    return -lp


def diffrax_nlpost(phi, Y_t, x0, step_size, obs_size, phi_mean, phi_sd, *args):
    phi_ind = len(phi_mean)
    xx0 = x0_initialize(phi, x0, phi_ind)
    phi = phi[:phi_ind]
    theta = jnp.exp(phi)
    X_t = diffeqsolve(term, solver, args = theta, t0=tmin, t1=tmax, dt0=step_size, y0=jnp.array(xx0), saveat=saveat,
                      stepsize_controller=stepsize_controller).ys
    lp = loglike(Y_t, X_t, step_size, obs_size, theta, *args)
    lp += logprior(phi, phi_mean, phi_sd)
    return -lp

def phi_fit(Y_t, x0, step_size, obs_size, phi_mean, phi_sd, obj_fun, *args, method="Newton-CG"):
    r"""Compute the optimized :math:`\log{\theta}` and its variance given 
        :math:`Y_t`."""
    gradf = grad(obj_fun)
    opt_res = sp.optimize.minimize(obj_fun, phi_init,
                                   args=(Y_t, x0, step_size, obs_size, phi_mean, phi_sd, *args),
                                   method=method,
                                   jac=gradf)
    phi_hat = opt_res.x
    return phi_hat


