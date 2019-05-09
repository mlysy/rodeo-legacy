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
