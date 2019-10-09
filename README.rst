Bayesian Solver for Univariate ODEs
===================================

probDE is a Python library that implements a couple of ODE solvers including
Bayesian methods and Kalman methods. The main focus of this library is using
Kalman Filter and Kalman Smoother to solve higher ordered ODE problems. 

This is accomplished by starting with an initial value. Then at each time 
point, the mean and variance is updated using Kalman Filter so that a new 
observation can be interrogated. After interrogating all the required
observations, they are smoothed over with Kalman Smoother.  

Installation
============

You can get the very latest code by getting it from GitHub and then performing
the installation.

.. code-block:: bash

    $ git clone https://github.com/mlysy/probDE.git
    $ cd filterpy
    $ python setup.py install

Usage
=====

As a simple example, consider the second order initial value ODE problem,

.. math::
    \begin{align*}
    x^{(2)}(t) &= sin(2t) − x, \quad t ∈ [0, 10], \\
    x^{(1)}(0) &= 0, \quad x(0) = −1. 
    \end{align*}

Its exact solution, :math:`x^{\star}(t) = \frac{−3 cos(t) + 2 sin(t) − sin(2t)}{3}`

The exact solution can be coded in python as follow:

.. code-block:: Python

    from math import sin, cos
    def chk_F(y_t, t):
        return sin(2*t) - y_t[0] #X^{2} = sing(2t) - X

    def chk_exact_x(t):
        return (-3*cos(t) + 2*sin(t) - sin(2*t))/3

    def chk_exact_x1(t):
        return (-2*cos(2*t) + 3*sin(t) + 2*cos(t))/3

Now given the initial value, one way to apply the algorithm to find the ode solution
is to use CAR(p) process. Let q be the number of times the derivative is defined for
in the problem. In our example above, it would be :math:`q=2`. Now we need to find
the initial observation from the CAR(p) process as follows:

.. code-block:: Python
    
    import numpy as np
    from BayesODE.Tests.root_gen import root_gen
    from BayesODE.Kalman.kalman_initial_draw import kalman_initial_draw
    N = 100
    q = 2
    p = q+2

    delta_t = np.array([1/N])
    r0 = 0.5
    sigma = 0.001
    roots = root_gen(r0, p) #Generate roots to draw x^{(3)}_0
    a = np.array([0,0,1])
    x0 = np.array([-1,0,0])
    x0 = np.array([-1,0,chk_F(x0, 0)]) #Initial state
    x_0 = kalman_initial_draw(roots, sigma, x0, p)

Next, we need to compute the transition matrix and variance matrix defining the
solution prior. If we assume the prior is :math:`CAR_p(0, \rho, \sigma)` where 
:math:`\rho =roots` then we can compute them as follows:

.. code-block:: Python

   wgtState, varState = higher_mvCond(delta_t, roots, sigma) 
   muState = np.zeros(p)

Finally, to run the solver:

.. code-block:: Python

   Yn, Yn_chk_mean, Yn_chk_var = kalman_ode_higher(chk_F, x_0, N-1, wgtState, muState, varState, a)