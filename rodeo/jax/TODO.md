# TODO for JAX

- [ ] Don't jit anything in the library.  The reason is that you should always jit at the last step, i.e., after combining non-jitted functions, after taking gradients, etc.

	Also, when jitting don't declare arguments `tmin` and `tmax` as `static_argnums`.

- [x] Don't have argument `z_state`.  There's no reason to anymore, since we should be doing that via `key`.

	**Update:** Don't do this for the Kalman functions, since it will make testing more inconvenient.  Or, should `kalmantv.smooth_sim()` output the mean and variance of the conditional distribution instead?
	
- [ ] Never use the same `key` twice (except for testing/debugging).  Otherwise, always create a new key via `key, subkey = jax.random.split(key)`.

	**Update:** I found this in `solve_sim()` but fixed now.  It might still be in `solve()` though...

- [ ] Make `interrogate` methods public, i.e., instead of passing them through `method` argument.  So the `interrogate` signature could be

	```python
	interrogate_xyx(key, wgt_meas, mu_state_pred, var_state_pred)
	```
	
	**Update:** Done for `interrogate_rodeo()`, but not the others.

- [x] Why do we need `` :math:`\\theta` `` instead of `` :math:`\theta` `` in the 2nd line of the docstrings but not elsewhere?

	**Solved:** Has to do with prepending `r` to docstring.  In this case should always use single slash.

- [ ] The ODE solver functions should use argument names which are more evocative of the ODE problem and less of the Kalman filter which underlies it.  Specifically, suppose the ODE-IVP is written as

	```
	W X(t) = f(X(t), t, theta)
	X(t_min) = x0
	t \in [t_min, t_max]
	```
	
	Then how about:

	- `wgt_meas` -> `W`
	- `{mu/wgt/var}_state` -> `{mu/wgt/var}_prior`
	- `{mu/var}_state_pred` -> `{mu/var}_prior_pred`
	
	Also, need to explain what are `n_meas` and `n_state` in the context of the ODE problem.  Perhaps we can call these `n_eqs` and `n_vars`, i.e., the number of equations and (total) number of variables in the ODE-IVP?
	
	**Update:** May not be necessary to change variable names, now that the module-level description of the solver makes a direct connection to `kalmantv`...
	
- [ ] Time is now the leading dimension, which greatly simplifies indexing and the use of `lax.scan()`.  Please update documentation as required.

	**Update:** All done except `solve()`.

- [x] Seems like much of the documentation in `solve()` should be at the module level instead, like what you did with `kalmantv.py`.

- [ ] In the `kalmantv.py` documentation, don't use ODE terms like "solution prior".  Call them "state variables" and "measurement variables" (not "measure variables", as you sometimes do).

	Similarly in `ode_solve.py`, try to relate things to the "solution prior" instead of the "state variables".  In this context, you can call `wgt_meas` the weight matrix :math:`W` in the ODE-IVP.

- [ ] Please go over the documentation very carefully.  There's much less to maintain now, so shouldn't be too painful :)

	- Do copy-paste identical arguments whenever it makes sense, please use hyperlinks to other parts of the documentation whenever possible, give DOIs for the papers we cite, etc.
	
	- In the return of the solve methods, the time variable `t` should be between `t_min` and `t_max`, not `0` and `1`.

	- Use the same letters `Q`, `\lambda`, and `R` in the solver documentation to correspond with those in the Kalman docs.  Or, perhaps use a different letter than `\lambda` to have all roman letters (vs mix of roman and greek)?
	
	**Update:** Turns out that `\lambda` is not the same thing as `mu_state`.  So, let's use the following notation in `kalmantv`:
	
	```
	x_n = c_n + Q_n x_{n-1} + R_n^{1/2} \epsilon_n

    y_n = d_n + W_n x_n + V_n^{1/2} \eta_n.
	```
	
	Then in `ode_solve` let's use:
	
	```
	x_n = c_n + Q_n x_{n-1} + R_n^{1/2} \epsilon_n

    y_n = W x_n + V_n^{1/2} \eta_n.
	```
	
- [ ] I think your `ibm_state()` function is incorrect, in that it should return a `(q+1) x (q+1)` matrix instead of `q x q`.  Please fix.

	Also, please finish converting `rodeo/jax/ibm_init.py` from NumPy to JAX.  I've already done `ibm_state()` for you (up to issue above).

- [ ] Unit tests for JAX:

	- `lax` constructs vs for-loops.
	
	- jitted vs unjitted  and jit + grad vs grad.
	
- [ ] Unit tests for rodeo (in addition to the JAX tests above):

	- rodeo vs odeint.

- [ ] Unit tests for kalmantv (in addition to the JAX tests above):

	- kalman vs gss.
