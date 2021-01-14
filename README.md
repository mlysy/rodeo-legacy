# **rodeo:** Probabilistic ODE Solver

*Mohan Wu, Martin Lysy*

---

## Description

**rodeo** is a fast and flexible Python front-end library with a Fortran/C++ back-end that uses [probabilistic numerics](http://probabilistic-numerics.org/) to solve ordinary differential equations (ODEs).  That is, most ODE solvers (such as [Euler's method](https://en.wikipedia.org/wiki/Euler_method)) produce a deterministic approximation to the ODE on a grid of size `delta`.  As `delta` goes to zero, the approximation converges to the true ODE solution.  Probabilistic solvers such as **rodeo** also output a solution an a grid of size `delta`; however, the solution is random.  Still, as `delta` goes to zero, the probabilistic numerical approximation converges to the true solution.

**rodeo** implements the probabilistic solver of [Chkrebtii et al (2016)](https://projecteuclid.org/euclid.ba/1473276259). This begins by putting a [Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process) prior on the ODE solution, and updating it sequentially as the solver steps through the grid. The user-facing interface is written in Python to allow for a wide appeal. The back-end is a time-varying Kalman filter implemented in C++ and Fortran using state-of-the-art linear algrebra routines found [here](https://github.com/mlysy/kalmantv). **rodeo** is 10-40x faster than a pure Python implementation, achieving comparable speeds to the widely-used deterministic solver **odein** in the **Scipy** library. Various low-level backends are provided in the following modules:

- `rodeo.cython`: This module performs the underlying linear algebra using the BLAS/LAPACK routines provided by NumPy through a Cython interface. 
To maximize speed, no input checks are provided.  All inputs must be `float64` NumPy arrays in *Fortran* order. 

- `rodeo.eigen`: This module uses the C++ Eigen library for linear algebra.  The interface is also through Cython.  
  Here again we have the same input requirements and lack of checks.  Eigen is known to be faster than most BLAS/LAPACK implementations, 
  but it needs to be compiled properly to achieve maximum performance.  In particular this involves linking against an installed version of Eigen (not provided)
  and setting the right compiler flags for SIMD and OpenMP support.  Some defaults are provided in `setup.py`, but tweaks may be required depending on the user's system. 

- `rodeo.numba`: This module once again uses BLAS/LAPACK but the interface is through Numba.  Here input checks are performed and the inputs can be 
  in either C or Fortran order, and single or double precision (`float32` and `float64`).  However, C ordered arrays are first converted to Fortran order, 
  so the latter is preferable for performance considerations.

## Installation

Download the repo from GitHub and then install with the `setup.py` script:

```bash
git clone https://github.com/mlysy/rodeo.git
cd rodeo
pip install .
```

## Unit Testing

The unit tests are done against the deterministic ode solver **odeint** to ensure that the solutions are approximately equal. They can be ran through the following commands:

```bash
cd tests
python -m unittest discover -v
```

## Examples

We provide four separate ODE problems as examples to demonstrate the capabilities of **rodeo**. These examples are best viewed in the `examples/tutorial.ipynb` jupyter notebook, hence extra installations are required.

```bash
pip install .[examples]
```

## Documentation

The HTML documentation can be compiled from the **kalmantv** root folder:
```bash
pip install .[docs]
cd docs
make html
```
This will create the documentation in `docs/build`.

## Usage

Please see the detailed example in the tutorial [here](https://nbviewer.jupyter.org/github/mlysy/rodeo/blob/cythonize/examples/tutorial.ipynb).  Running the tutorial compares the deterministic Euler solver to the probabilistic solver for the ODE initial value problem

```
x_t^{(2)} = sin(2t) - x_t^{(0)}
x_0^{(1)} = 0
x_0^{(0)} = -1
```

The results for N = 50, 100, and 200 grid points for both solvers is shown below.

![chkrebtii](/docs/figures/chkrebtiifigure.png)

**rodeo** is also capable of performing parameter inference. An example of this is on the **FitzHugh-Nagumo** model in the tutorial. A comparison of the deterministic Euler solver to the probabilistic solver is shown below.

![fitzhugh](/docs/figures/fitzfigure.png)
