# Bayesian Folder Function Structure
This README provides the structure of the contents of this folder and a brief description of its usage.

    Bayesian
    ├── bayes_ode.py
    |   └── bayes_ode           # Bayesian solver of ODE problem dx_t/dt = f(x_t, t).
    ├── cov_exp.py
    |   ├── cov_vv_ex           # Computes the covariance function for the derivative v_t 
    |   |                           using exponential kernel. 
    │   ├── cov_xv_ex           # Computes the cross-covariance function for the solution 
    |   |                         process x_t and its derivative v_t using exponential kernel. 
    │   └── cov_xx_ex           # Computes the covariance function for the solution process 
    |                              x_t using exponential kernel. 
    ├── cov_rect.py
    |   ├── cov_vv_re           # Computes the covariance function for the derivative v_t 
    |   |                           using rectangular kernel. 
    │   ├── cov_xv_re           # Computes the cross-covariance function for the solution 
    |   |                         process x_t and its derivative v_t using rectangular kernel. 
    │   └── cov_xx_re           # Computes the covariance function for the solution process 
    |                              x_t using rectangular kernel. 
    └── cov_square_exp.py
        ├── cov_vv_se           # Computes the covariance function for the derivative v_t 
        |                           using square exponential kernel. 
        ├── cov_xv_se           # Computes the cross-covariance function for the solution 
        |                         process x_t and its derivative v_t using square exponential kernel. 
        └── cov_xx_se           # Computes the covariance function for the solution process 
                                  x_t using square exponential kernel. 
                                  