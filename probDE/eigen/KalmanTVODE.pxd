cdef extern from "KalmanTVODE.h" namespace "KalmanTVODE":
    cdef cppclass KalmanTVODE:
        KalmanTVODE(int, int, int, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*) except +
        void predict(const int cur_step)
        void update(const int cur_step)
        void filter(const int cur_step)
        void smooth_mv(const int cur_step) 
        void smooth_sim(const int cur_step)
        void smooth(const int cur_step)
        void smooth_update(const bint sim_sol)
        void forecast(const int cur_step)
        void forecast_sch(const int cur_step)
