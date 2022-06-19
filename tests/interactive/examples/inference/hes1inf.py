from scipy.integrate import odeint
from .inference import inference
import numpy as np
import jax
from jax.config import config
config.update("jax_enable_x64", True)

from rodeo.jax.ode_solve import *

class hes1inf(inference):
    r"Inference for the Hes1 model"
    
    def ode_fun(self, X_t, t, theta):
        "Hes1 ODE written for odeint"
        P, M, H = jnp.exp(X_t)
        a, b, c, d, e, f, g = theta
        x1 = -a*H + b*M/P - c
        x2 = -d + e/(1+P*P)/M
        x3 = -a*P + f/(1+P*P)/H - g
        return jnp.array([x1, x2, x3])

    def rax_fun(self, t, X_t, theta):
        "Hes1 ODE written for odeint"
        P, M, H = jnp.exp(X_t)
        a, b, c, d, e, f, g = theta
        x1 = -a*H + b*M/P - c
        x2 = -d + e/(1+P*P)/M
        x3 = -a*P + f/(1+P*P)/H - g
        return jnp.array([x1, x2, x3])

    def loglike(self, Y_t, X_t, step_size, obs_size, theta, gamma):
        data_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/obs_size)+1)
        ode_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/step_size)+1)
        X_t = self.thinning(ode_tseq, data_tseq, X_t)
        X_t = self.hes1_obs(X_t)
        return self.logprior(Y_t, X_t, gamma)

    def hes1_obs(self, sol):
        r"Given the solution process, get the corresponding observations"
        return jnp.append(sol[::2, 0], sol[1::2, 1])

    def simulate(self, x0, theta, gamma, tseq):
        r"Get the observations assuming a normal distribution."
        sol = odeint(self.ode_fun, x0, tseq, args=(theta,))
        X_t = self.hes1_obs(sol)
        e_t = np.random.default_rng(0).normal(loc=0.0, scale=1, size=X_t.shape)
        Y_t = X_t + gamma*e_t
        return Y_t, X_t
    
    # def kalman_solve(self, step_size, obs_size, x0, theta):
    #     r"Using Kalman solver to compute solutions"
    #     self.key, subkey = jax.random.split(self.key)
    #     data_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/obs_size)+1)
    #     ode_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/step_size)+1)
    #     sol = mv_jit(subkey, self.fun, x0, theta, self.tmin, self.tmax, self.n_eval, self.W, **self.kinit)[0]
    #     sol = sol[:, :, 0]
    #     sol = self.thinning(ode_tseq, data_tseq, sol)
    #     X_t = self.hes1_obs(sol)
    #     return X_t
    
    # def euler_solve(self, x0, step_size, obs_size, theta):
    #     r"Using Euler method to compute solutions"
    #     data_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/obs_size)+1)
    #     ode_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/step_size)+1)
    #     sol = euler(hes1, x0, ode_tseq, data_tseq, step_size, theta)
    #     X_t = self.hes1_obs(sol)
    #     return X_t
    
    # def diffrax_solve(self, x0, step_size, obs_size, theta):
    #     term = ODETerm(hes1_rax)
    #     solver = Dopri5()
    #     data_tseq = jnp.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/obs_size)+1)
    #     saveat = SaveAt(ts=data_tseq)
    #     stepsize_controller = PIDController(rtol=1e-5, atol=1e-5)
    #     sol = diffeqsolve(term, solver, args = theta, t0=self.tmin, t1=self.tmax, dt0=step_size, y0=jnp.array(x0), saveat=saveat,
    #                     stepsize_controller=stepsize_controller)
    #     X_t = self.hes1_obs(sol.ys)
    #     return X_t
