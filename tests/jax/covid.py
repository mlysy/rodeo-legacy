from scipy.integrate import odeint
from inference_jax import inference
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import jax.scipy.stats

class covid(inference):
    r"Inference using the France Covid data from Prague et al."
    
    def loglike(self, x, mean, theta):
        r"Calculate the loglikelihood of the poisson distribution."
        mean = self.covid_obs(mean, theta)
        return jnp.sum(jsp.stats.poisson.logpmf(x, mean))

    def logprior(self, x, mean, sd):
        r"Calculate the loglikelihood of the normal distribution."
        return jnp.sum(jsp.stats.norm.logpdf(x=x, loc=mean, scale=sd))

    def covid_obs(self, X_t, theta):
        r"Compute the observations as detailed in the paper"
        I_in = theta[1]*X_t[:,1]/theta[3]
        H_in = X_t[:,2]/theta[5]
        X_in = jnp.array([I_in, H_in]).T
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
        tseq = jnp.linspace(self.tmin, self.tmax, self.tmax-self.tmin+1)
        X_t = odeint(fun, x0, tseq, args=(theta,))[1:,]
        X_in = self.covid_obs(X_t, theta)
        Y_in = np.random.default_rng().poisson(X_in)
        return Y_in, X_in
    
    def kalman_solve(self, data_tseq, ode_tseq, x0, theta):
        r"Using Kalman solver to compute solutions"
        X_t = self.kode.solve_mv(x0, self.W, theta)[0]
        X_t = self.thinning(data_tseq, ode_tseq, X_t)[:, self.state_ind]
        X_in = self.covid_obs(X_t, theta)
        return X_in
