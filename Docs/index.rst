.. probDE documentation master file

*********************************************************
Probabilistic solution of ordinary differential equations
*********************************************************

.. toctree::
   :maxdepth: 2

Description
===========

**probDE** is a Python library that uses `probabilistic numerics <http://probabilistic-numerics.org/>`_ to solve ordinary differential equations (ODEs). 
That is, most ODE solvers (such as `Euler's method <https://en.wikipedia.org/wiki/Euler_method>`_) produce a deterministic approximation to the ODE on a grid of size :math:`\delta`.  
As :math:`\delta` goes to zero, the approximation converges to the true ODE solution.  
Probabilistic solvers such as **probDE** also output a solution an a grid of size :math:`\delta`; however, the solution is random.  
Still, as :math:`\delta` goes to zero we get the correct answer.

**probDE** provides a probabilistic solver for univariate ordinary differential equations (ODEs) of the form

.. math::
    \begin{equation*}
    \boldsymbol{w'}\boldsymbol{x}_t = f(\boldsymbol{x}_t, t), \qquad \boldsymbol{x}_L = \boldsymbol{a},
    \end{equation*}

where :math:`\boldsymbol{x}_t = \big(x_t^{(0)}, x_t^{(1)}, ..., x_t^{(q)}\big)` consists of the first :math:`q` derivatives of the process :math:`x_t = x_t^{(0)}`, 
and a solution is sought on the interval :math:`t \in [L, U]`.  

**probDE** implements the probabilistic solver of `Chkrebtii et al (2016) <https://projecteuclid.org/euclid.ba/1473276259>`_. 
This begins by putting a `Gaussian process <https://en.wikipedia.org/wiki/Gaussian_process>`_ prior on the ODE solution, and updating it sequentially as the solver steps through the grid.

Walkthrough
===========

To illustrate, let's consider the following ODE example of order :math:`q = 2`:

.. math::
    \begin{equation*}
    x_t^{(2)} = \sin(2t) − x_t^{(0)}, \qquad \boldsymbol{x_0} = (-1, 0, 1),
    \end{equation*}

where the solution :math:`x_t` is sought on the interval :math:`t \in [0, 10]`.  In this case, the ODE has an analytic solution,

.. math::
    \begin{equation*}
    x_t = \frac{1}{3} \big(2\sin(t) - 3\cos(t) - \sin(2t)\big).
    \end{equation*}

To approximate the solution with the probabilistic solver, the Gaussian process prior we will use is a so-called 
`Continuous Autoregressive Process <https://CRAN.R-project.org/package=cts/vignettes/kf.pdf>`_ of order :math:`p`, 

.. math::
    \begin{equation*}
    \boldsymbol{X}_t \sim \mathrm{CAR}_p(\boldsymbol{\mu}, \boldsymbol{\rho}, \sigma).
    \end{equation*}

Here :math:`\boldsymbol{X}_t = \big(x_t^{(0)}, ..., x_t^{(p-1)}\big)` consists of :math:`x_t` and its first :math:`p-1` derivatives. 
The :math:`\mathrm{CAR}(p)` model specifies that each of these is continuous, but :math:`x_t^{(p)}` is not. 
Therefore, we need to pick :math:`p > q`. It's usually a good idea to have :math:`p` a bit larger than :math:`q`, 
especially when we think that the true solution :math:`x_t` is smooth. However, increasing :math:`p` also increases the computational burden, 
and doesn't necessarily have to be large for the solver to work.  For this example, we will use :math:`p=4`. The tuning parameters of the :math:`\mathrm{CAR}(p)` prior are:

- The mean vector :math:`\boldsymbol{\mu}`.  By default we will set this to 0.
- The scale parameter :math:`\sigma`.
- The "roots" of the process :math:`\boldsymbol{\rho} = (\rho_0, \ldots, \rho_{p-1})`. These can be any distinct set of negative numbers. 
  We suggest parametrizing them as :math:`\rho_0 = -1/\tau` and :math:`\rho_k = -(1 + \tfrac{k}{10(p-1)})` for :math:`k > 0`, 
  in which case :math:`\tau` becomes a decorrelation-time parameter.

Finally, we need a way to initialize the remaining derivatives :math:`\boldsymbol{y}_t = \big(x_t^{(q+1)}, ..., x_t^{(p-1)}\big)` at time :math:`t = L`. 
Since the :math:`\mathrm{CAR}(p)` process has a multivariate normal stationary distribtuion,
we suggest initializing :math:`\boldsymbol{y}_L \sim p(\boldsymbol{y}_L \mid \boldsymbol{x}_L = \boldsymbol{a})`,
i.e., as a random draw from this stationary distribution conditional on the value of :math:`\boldsymbol{x}_L = \boldsymbol{a}`.
The Python code to implement all this is as follows.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint

    from probDE.car import car_init
    from probDE.cython.KalmanODE import KalmanODE
    from probDE.utils import indep_init

.. code-block:: python

    # RHS of ODE
    from math import sin, cos
    def ode_fun(x_t, t, theta=None):
        return sin(2*t) - x_t[0]

    # LHS vector of ODE
    w_vec = np.array([0.0, 0.0, 1.0])

    # These parameters define the order of the ODE and the CAR(p) process
    n_meas = 1
    n_state = 4

    # it is assumed that the solution is sought on the interval [tmin, tmax].
    n_eval = 200
    tmin = 0
    tmax = 10

    # The rest of the parameters can be tuned according to ODE
    # For this problem, we will use
    tau = 50
    sigma = .001

    # Initial value, x0, for the IVP
    x0 = np.array([-1., 0., 1.])

    # Get parameters needed to run the solver
    dt = (tmax-tmin)/n_eval
    # All necessary parameters are in kinit, namely, T, c, R, W
    kinit, x0_state = indep_init([car_init(n_state, tau, sigma, dt, w_vec, x0)], n_state)

    # Initialize the Kalman class
    kalmanode = KalmanODE(n_state, n_meas, tmin, tmax, n_eval, ode_fun, **kinit)
    # Run the solver to get an approximation
    kalman_sim = kalmanode.solve(x0_state, mv=False, sim=True)

We drew 100 samples from the solver to compare them to the exact solution and the Euler approximation to the problem. 

For :math:`x^{(0)}_t`:

.. image:: figures/chkrebtii_x0.png

For :math:`x^{(1)}_t`:

.. image:: figures/chkrebtii_x1.png

Installation
============

You can get the very latest code by getting it from GitHub and then performing
the installation.

.. code-block:: bash

    git clone https://github.com/mlysy/probDE.git
    cd probDE
    pip install .

Functions Documentation
=======================
.. toctree::
   :maxdepth: 1

   ./car
   ./KalmanODE
   ./utils
   