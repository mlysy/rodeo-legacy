# Bayesian Solver for Univariate ODEs

*Mohan Wu, Martin Lysy*

---

## Description

**probDE** is a Python library that uses [probabilistic numerics](http://probabilistic-numerics.org/) to solve ordinary differential equations (ODEs).  That is, most ODE solvers (such as [Euler's method](https://en.wikipedia.org/wiki/Euler_method)) produce a deterministic approximation to the ODE on a grid of size `delta`.  As `delta` goes to zero, the approximation converges to the true ODE solution.  Probabilistic solvers such as **probDE** also output a solution an a grid of size `delta`; however, the solution is random.  Still, as `delta` goes to zero we get the correct answer.

Currently, **probDE** implements the probabilistic solver of [Chkrebtii et al (2016)](https://projecteuclid.org/euclid.ba/1473276259).  This begins by putting a [Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process) prior on the ODE solution, and updating it sequentially as the solver steps through the grid.

## Installation

Download the repo from GitHub and then install with the `setup.py` script:

```bash
git clone https://github.com/mlysy/probDE.git
cd probDE
python setup.py install
```

## Usage

**probDE** currently supports two types of Gaussian process priors: [Continuous Autoregressive](https://CRAN.R-project.org/package=cts/vignettes/kf.pdf) (CAR(p)) processes, and non-Markov priors described below. For a more detailed example, checkout BayesODE/Examples/tutorial.ipynb.

### CAR(p) Processes
------

The advantage of CAR(p) processes is that they are Markov, and so the probabilistic solver can be efficiently implemented in linear time using the [Kalman](https://en.wikipedia.org/wiki/Kalman_filter) filtering and smoothing recursions.  For more information, please see full documentation in `Docs/Kalman`, which contains math formatting.

As present, **probDE** can be used to solve any ODE initial value problem of the form 

```
a' X_t = F(X_t, t)
X_0 = x0
```

where `X_t = (x_t^{(0)}, x_t^{(1)}, ..., x_t^{(q)})` consists of the first q derivatives of the process, and a solution is sought on the interval `t \in [0, 1]`.

As a simple example, consider the second order ODE initial value problem

```
x_t^{(2)} = sin(2t) - x_t^{(0)}
x_0^{(1)} = 0
x_0^{(0)} = -1
```

Its exact solution is 

```
x_t = (-3cos(t) + 2sin(t) - sin(2t))/3
```

The CAR(p) solution prior is a continuous process of the form `Y_t = (y_t^{(0)}, ..., y_t^{(p-1)})`, where we should have `p > q` so that it is smooth enough to solver the ODE above.

Here's the code to calculate the probabilistic solution with **probDE**:

```python
import numpy as np
from math import sin, cos
from BayesODE.Tests.root_gen import root_gen
from BayesODE.Kalman.kalman_initial_draw import kalman_initial_draw

# ODE definition
# LHS vector
a_vec = np.array([0, 0, 1.0])
# RHS function
def ode_F(X_t, t):
    return sin(2*t) - X_t[0]
    
# algorithm tuning parameters
q = 2 # ODE order
p = q+2 # number of continuous derivatives of CAR(p) solution prior

# it is assumed that the solution is sought on the interval [0,1].
# this next parameter specifies the size of the discretization grid
N = 100 # grid size delta = 1/N

# now the tuning parameters of the CAR(p) prior
sigma = 0.001 # scale paramater
r0 = 0.5 # decorrelation parameter
```

The exact solution can be coded in python as follow:
```python
from math import sin, cos
def chk_F(y_t, t):
    return sin(2*t) - y_t[0] #X^{2} = sing(2t) - X

def chk_exact_x(t):
    return (-3*cos(t) + 2*sin(t) - sin(2*t))/3

def chk_exact_x1(t):
    return (-2*cos(2*t) + 3*sin(t) + 2*cos(t))/3
```
Now given the initial value, one way to apply the algorithm to find the ode solution
is to use CAR(p) process. Let q be the number of times the derivative is defined for
in the problem. In our example above, it would be q=2. Now we need to find
the initial observation from the CAR(p) process as follows:
```python
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
```
Next, we need to compute the transition matrix and variance matrix defining the
solution prior. If we assume the prior is CAR(p) then we can compute them as follows:
```python
wgtState, varState = higher_mvCond(delta_t, roots, sigma) 
muState = np.zeros(p)
```
Finally, to run the solver:
```python
Yn, Yn_chk_mean, Yn_chk_var = kalman_ode_higher(chk_F, x_0, N-1, wgtState, muState, varState, a)
```
We drew 100 samples from the solver to compare them to the exact solution and the Euler approximation to the problem. 

For x^(0):
![chkrebtii_x0](/Docs/Kalman/chkrebtii_x0.png)

For x^(1):
![chkrebtii_x1](/Docs/Kalman/chkrebtii_x1.png)

Bayesian
--------

For the Bayesian method, we use this simple example:
```python
def f(x,t):
    return  3*(t+1/4) - x/(t+1/4)
```
The initial values of the problem and the grid size can be defined as follows:
```python
a = 0
b = 4.75
x0_f1 = 10
N = 100
tseq1 = np.linspace(a, b, N)
```
The two parameters *alpha* and *gamma* can be tuned. The Tuning jupyter notebook
shows an example of how to tune these parameters based on the kernel. For this example, we can use:
```python
gamma = 0.1
alpha = 1000
```
Now, all we need to do is choose the kernel we want and get the initial *Sigma* matrix. For 
example, we can use the square exponential kernel:
```python
import BayesODE.Bayesian as bo
Sigma_vv = bo.cov_vv_se(tseq1, tseq1, gamma, alpha)
Sigma_xx = bo.cov_xx_se(tseq1, tseq1, gamma, alpha)
Sigma_xv = bo.cov_xv_se(tseq1, tseq1, gamma, alpha)
```
Finally, we run the solver to get an approximate solution, the mean and the variance of the approximation:
```python
xt1,mu_x,var_x = bo.bayes_ode(f, tseq1, x0_f1, Sigma_vv, Sigma_xx, Sigma_xv)
```
We can look at the results of all three kernels compared to the exact solution.
![simple_ode](/Docs/Bayesian/simple_ode.png)
