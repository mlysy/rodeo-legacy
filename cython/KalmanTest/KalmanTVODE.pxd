cdef extern from "KalmanTVODE.h" namespace "KalmanTVODE":
    cdef cppclass KalmanTVODE:
        KalmanTVODE(int, int, int, double*) except +
        void predict(const int cur_step,
                     const double* mu_state,
                     const double* wgt_state,
                     const double* var_state)
        void update(const int cur_step,
                    const double* x_meas,
                    const double* wgt_meas)
        void filter(const int cur_step,
                    const double* mu_state,
                    const double* wgt_state,
                    const double* var_state,
    		            const double* x_meas,
                    const double* wgt_meas)
        void smooth_mv(double* mu_state_smooths,
                       double* var_state_smooths,
                       const int cur_step,
                       const double* wgt_state) 
        void smooth_sim(double* x_state_smooths,
                        const int cur_step,
                        const double* wgt_state,
                        const double* z_states)
        void smooth(double* x_state_smooths,
                    double* mu_state_smooths,
                    double* var_state_smooths,
                    const int cur_step,
                    const double* wgt_state,
                    const double* z_states)
        void smooth_update(double* x_state_smooths,
                           double* mu_state_smooths,
                           double* var_state_smooths,
                           const double* wgt_state,
                           const double* z_states,
                           const bint smooths_mv,
                           const bint smooths_sim)
        void chkrebtii_int(double* x_state,
                           const int cur_step,
                           const double* wgt_meas,
                           const double* z_states)
