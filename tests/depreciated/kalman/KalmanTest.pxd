cdef extern from "KalmanTest.h" namespace "KalmanTest":
    cdef cppclass KalmanTest:
        KalmanTest(int, int, int) except +
        void filter_smooth(double* mu_state_smooths,
                           double* var_state_smooths,
                           const double* x0_state,
                           const double* wgt_state,
                           const double* mu_state, 
                           const double* var_state,
                           const double* wgt_meas,
                           const double* x_meass,
                           const double* z_state_sim)
                           