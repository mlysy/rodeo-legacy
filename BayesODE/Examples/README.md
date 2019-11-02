# Examples Folder Function Structure
This README provides the structure of the contents of this folder and a brief description of its usage.

    Examples
    ├── euler_approx.py
    |   └── euler_approx            # Uses the Euler approximation for univariate ordinary 
    |                                 differential equations (ODEs).
    |
    ├── readme_graph.py
    |   ├── ode_exact_x             # Exact solution for the example ODE in tutorials for x_t^{(0)}.
    |   ├── ode_exact_x1            # Exact solution for the example ODE in tutorials for x_t^{(1)}.
    |   ├── ode_euler               # Example ode written for Euler approximation.
    |   ├── readme_kalman_draw      # Draw samples from kalman solver for the example in tutorial.
    |   ├── readme_solver           # Calculates kalman_ode, euler_ode, and exact_ode on the given 
    |   |                             grid for the tutorial ode.
    |   └── readme_graph            # Produces the graph in tutorial.
    |
    └── tutorial.ipynb              # Tutorial on the usage of the kalman solver.
