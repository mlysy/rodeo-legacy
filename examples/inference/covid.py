from scipy.integrate import odeint
from .inference import inference
import numpy as np
import scipy as sp
import scipy.stats

class covid(inference):
    r"Inference using the France Covid data from Prague et al."
    
    def loglike_pois(self, x, mean):
        r"Calculate the loglikelihood of the poisson distribution."
        return np.sum(sp.stats.poisson.logpmf(x, mean))
    
    def covid_obs(self, X_t, theta):
        r"Compute the observations as detailed in the paper"
        I_in = theta[1]*X_t[:,1]/theta[3]
        H_in = X_t[:,2]/theta[5]
        X_in = np.array([I_in, H_in]).T
        return X_in
    
    def simulate(self, fun, x0, theta):
        r"""Get the observations for the SEIRAH Covid example.
        None of the compartments are directly observed, however 
        the daily infections and hospitalizations are observed. 
        They can be computed as
        
        .. math::

            I^{(in)}(t) = rE(t)/D_e
            H^{(in)}(t) = I(t)/D_q

        """
        tseq = np.linspace(self.tmin, self.tmax, self.tmax-self.tmin+1)
        X_t = odeint(fun, x0, tseq, args=(theta,))[1:,]
        X_in = self.covid_obs(X_t, theta)
        Y_in = np.random.default_rng().poisson(X_in)
        return Y_in, X_in
    
    def kalman_solve(self, step_size, obs_size, x0, theta):
        r"Using Kalman solver to compute solutions"
        data_tseq = np.linspace(self.tmin+1, self.tmax, int((self.tmax-self.tmin)/obs_size))
        ode_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/step_size)+1)
        X_t = self.kode.solve_sim(x0, self.W, theta)
        X_t = self.thinning(ode_tseq, data_tseq, X_t)[:, self.state_ind]
        X_in = self.covid_obs(X_t, theta)
        return X_in
    
    def euler_solve(self, x0, step_size, obs_size, theta):
        r"Using Euler method to compute solutions"
        data_tseq = np.linspace(self.tmin+1, self.tmax, int((self.tmax-self.tmin)/obs_size))
        ode_tseq = np.linspace(self.tmin, self.tmax, int((self.tmax-self.tmin)/step_size)+1)
        X_t = self.euler(x0, ode_tseq, data_tseq, step_size, theta)
        X_in = self.covid_obs(X_t, theta)
        return X_in
