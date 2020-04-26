# **probDE:** A Probabilistic Numerical Solver for Ordinary Differential Equations

*Mohan Wu, Martin Lysy*

---

## Description

**probDE** is a Python library that uses [probabilistic numerics](http://probabilistic-numerics.org/) to solve ordinary differential equations (ODEs).  That is, most ODE solvers (such as [Euler's method](https://en.wikipedia.org/wiki/Euler_method)) produce a deterministic approximation to the ODE on a grid of size `delta`.  As `delta` goes to zero, the approximation converges to the true ODE solution.  Probabilistic solvers such as **probDE** also output a solution an a grid of size `delta`; however, the solution is random.  Still, as `delta` goes to zero, the probabilistic numerical approximation converges to the true solution.

Currently, **probDE** implements the probabilistic solver of [Chkrebtii et al (2016)](https://projecteuclid.org/euclid.ba/1473276259).  This begins by putting a [Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process) prior on the ODE solution, and updating it sequentially as the solver steps through the grid.

## Installation

Download the repo from GitHub and then install with the `setup.py` script:

```bash
git clone https://github.com/mlysy/probDE.git
cd probDE
python setup.py install
```

## Usage

Please see the detailed example in the tutorial [here](https://nbviewer.jupyter.org/github/mlysy/probDE/blob/master/probDE/Examples/tutorial.ipynb).  Running the tutorial compares the deterministic Euler solver to the probabilistic solver for the ODE initial value problem

```
x_t^{(2)} = sin(2t) - x_t^{(0)}
x_0^{(1)} = 0
x_0^{(0)} = -1
```

The results for N = 50, 100, and 200 grid points for both solvers is shown below.

For `x_t^{(0)}`:
![chkrebtii_x0](/Docs/Figures/chkrebtii_x0.png)

For `x_t^{(1)}`:
![chkrebtii_x1](/Docs/Figures/chkrebtii_x1.png)

## C++
C++ version is being worked on. For a quick look at the graphs and timings, you can checkout some examples [here](https://nbviewer.jupyter.org/github/mlysy/probDE/blob/cythonize/cython/KalmanODE.ipynb).