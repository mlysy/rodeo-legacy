# Bayesian Solver for Univariate ODEs

*Mo Han Wu, Martin Lysy*

----

## TODO

- [ ] Add documentation for covariance functions.  
    - Use [Sphinx](http://www.sphinx-doc.org/en/master/) for this.  For a quick example see [here](https://pythonhosted.org/an_example_pypi_project/sphinx.html).
	- As suggested by the [Python Developers Guide](https://devguide.python.org/documenting/), whenever possible in the doc use *sentence case*, i.e., start with a capital letter and end with a period, and only capitalize other words if there is a specific rule requiring it. (This rule is really easy to remember, so you can apply it uniformly.  There's nothing more annoying than having to fix inconsistent documentation formatting).
- [ ] Think about how to best name things.
- [ ] Get this package to work :)  For more info on creating Python packages see [here](https://docs.python.org/3/tutorial/modules.html).  Ultimately though we should be able run e.g.:

    ```python
	import numpy as np
	import BayesODE as pos # probabilitic ODE solver
	
	def f(x,t):
		return 3*t - x/t
	a = 0
	b = 4.75
	x0 = 10
	N = 100
	tseq = np.linspace(a, b, N)
	gamma = (b-a)/10
	alpha = N/10
	
	# package functions
	Sigma_vv, Sigma_xx, Sigma_xv = pos.cov_prior(tseq, gamma, alpha, 'exp2')
	xt = pos.ode_bayes(f, tseq, x0, Sigma_vv, Sigma_xx, Sigma_xv)
	```
	
- [ ] Run unit tests on the covariance functions to make sure they are correct.  In other words, you should numerically integrate `cov(v_t, v_s)` with randomly selected `t` and `s` once and twice, and make sure it is identical to the analytic expressions `cov(x_t, v_s)` and `cov(x_t, v_t)`.  Please see package [`unittest`](https://docs.python.org/3/library/unittest.html) for how to set these up formally.
- [ ] Add some ODE examples, e.g., using Jupyter Notebook.
