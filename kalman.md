---
pagetitle: "Kalman Filter"
---

# Implementation of the Kalman Filter

## Manual Implementation

1. A function to do one forward step (can improve the notation, but this notation is consistent with the derivation in the document):

    ```python
	def kalman_fwd(m_prev, V_prev, A_curr, b_curr, C_curr, D_curr, e_curr, F_new):
		"""Given the mean and variance of `p(Z_n | X_0:n)`, 
		return the mean and variance of `p(Z_{n+1} | X_{0:n+1})`.
		"""
		return m_curr, V_curr
	```
	
    I believe this function can be checked with the `filter_update` function in the [**pykalman**](https://pykalman.github.io/) library.

2. A function to do one backward step, i.e., given $Z_{n+1}$ (and other things), obtain one draw from $p(Z_n \mid Z_{n+1}, X_{0:N})$.  Note that the mean of the collection of all these draws, $E[Z_{0:N} \mid X_{0:N}]$, can be obtained from the `smooth` function in **pykalman**.

## In the Context of the ODE Solver

Actually, it seems we only need the `filter_update` and `smooth` function to write a Bayesian ODE solver which output only the posterior mean of the solution, not a random draw from it.

**TODO:** The implementation of this should return the exact same result as for the `exp` kernel.  So let's write the ODE solver for the `exp` kernel using the Kalman filter and smoother as implemented in **pykalman**.  Sort of looks like this:

```python
def bayes_ode_kalman(...):
    # filtering step is sequential.  Note: yt = (xt, vt).
    for n in range(N):
	    [m[i,:], V[i,i,:]] = filter_update(current_state, v_star[i], ...)
		v_star[i+1] = model_interrogation(fun, ...)
	# smoothing step is global
	yt_mean = smooth(v_star) # will have to initialize a new KalmanFilter object
	return yt_mean
```

Should be able to test this with "pre-generated" `v_star` the same way we tested the implementation for general solution prior.

### Details

- *Setup.* Let $y_n = (x_n, v_n)$ denote the solution and its derivative at time $t = n/T$, i.e., $v_n = f(x_n, n/T)$.  Assume that $x_0 = 0$.  

    Also, the model interrogation generates a value of $v_n$, so the Kalman filter will consider $v_n$ to be observed and $x_n$ to be missing.  
	
	Since the solution prior on $y_n$ is a (homogeneous) Markov chain, it can be written in the form
	$$
	y_n = A y_{n-1} + C \varepsilon_n, \qquad \varepsilon_n \sim_{iid} \mathcal N(0, I_{2\times 2}).
	$$
	Because of the way we've set things up with $x_0 = 0$, the mean term drops out.  Moreover, I am almost certain that it is also the case that we can write
	$$
	x_n = a x_{n-1} + c \eta_n, \qquad \eta_n \sim_{iid} \mathcal N(0, 1).
	$$
	

- *Model Interrogations.*  These are not part of the Kalman filter.  Rather, they generate the data which goes into the Kalman filter.  So at the beginning of step $n$, we have $(m_{n-1}, V_{n-1})$, the mean and variance of $p(x_{n-1} \mid v_{0:n-1})$, where $v_{0:n-1}$ are the previous model interrogations (let's drop the *).  The model interrogation at step $n$ proceeds in two steps:

    1.  Generate $x_n \sim p(x_n \mid v_{0:n-1})$ via:
	    $$
		x_{n-1} \sim \mathcal N(m_{n-1}, V_{n-1}), \qquad x_n = a x_{n-1} + c \eta_n.
		$$
	2.  Convert to exact derivative via $v_n = f(x_n, n/T)$.

<!-- Here's some Python pseudocode: -->

<!--     ```python -->
<!-- 	# please improve notation -->
<!-- 	def interrogate_v(m_prev, V_prev, a, c, ode_fun, t_curr): -->
<!-- 	    # generate x_curr -->
<!-- 		x_prev = random_normal(m_prev, V_prev) # draw from p(x_prev | all interrogations up to t_prev) -->
<!-- 		x_curr = a * x_prev + c * random_normal(0, 1) # draw p(x_curr | x_prev) -->
<!-- 		# convert to "interrogation" -->
<!-- 		v_curr = ode_fun(x_curr, t_curr)  -->
<!-- 		return v_curr -->
<!-- 	``` -->

- *Kalman Filter.*  Now we need to write down the Kalman state-space equations corresponding to the filter.  The latent variable equation is
    $$
	x_n = a x_{n-1} + c \eta_n.
	$$
	
