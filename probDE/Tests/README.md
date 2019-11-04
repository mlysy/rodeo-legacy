# Tests Folder Function Structure
This README provides the structure of the contents of this folder and a brief description of its usage.

    Tests
    ├── Analytical vs Numerical Test.ipynb      # Graphs the computed results for each of the three kernels
    |                                             in Bayesian solver against their exact solutions. 
    |
    ├── test_exp.py
    |   ├── cov_vv_ex2                          # Analytical covariance function for the derivative
    |   |                                         v_t using exponential kernel. 
    │   ├── cov_xv_ex2                          # Analytical cross-covariance function for the solution x_t
    |   |                                         and its derivative v_t using exponential kernel. 
    │   ├── cov_xx_ex2                          # Analytical covariance function for the solution process
    |   |                                         x_t using exponential kernel.
    │   └── test_cov_ex                         # Computes the analytical and numerical covariance functions 
    |                                             for the solution process x_t and its derivative v_t 
    |                                             using exponential kernel. 
    ├── test_rect.py
    |   ├── cov_vv_re2                          # Analytical covariance function for the derivative
    |   |                                         v_t using rectangular kernel. 
    │   ├── cov_xv_re2                          # Analytical cross-covariance function for the solution x_t
    |   |                                         and its derivative v_t using rectangular kernel. 
    │   ├── cov_xx_re2                          # Analytical covariance function for the solution process
    |   |                                         x_t using rectangular kernel.
    │   └── test_cov_re                         # Computes the analytical and numerical covariance functions 
    |                                             for the solution process x_t and its derivative v_t 
    |                                             using rectangular kernel. 
    └── test_square_exp.py
        ├── cov_vv_se2                          # Analytical covariance function for the derivative
        |                                         v_t using square exponential kernel. 
        ├── cov_xv_se2                          # Analytical cross-covariance function for the solution x_t
        |                                         and its derivative v_t using square exponential kernel. 
        ├── cov_xx_se2                          # Analytical covariance function for the solution process
        |                                         x_t using square exponential kernel.
        └── test_cov_se                         # Computes the analytical and numerical covariance functions 
                                                  for the solution process x_t and its derivative v_t 
                                                  using square exponential kernel. 
