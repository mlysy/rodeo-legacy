cdef extern from "KalmanTVODE.h" namespace "kalmantvode":
    cdef cppclass KalmanTVODE:
        KalmanTVODE(int, int, int, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*, double*) except +
        void predict(const int cur_step)
        void update(const int cur_step)
        void filter(const int cur_step)
        void smooth_mv(const int cur_step) 
        void smooth_sim(const int cur_step)
        void smooth(const int cur_step)
        void smooth_update(const bint smooths_mv,
                           const bint smooths_sim)
        void interrogate_chkrebtii(const int cur_step)
        void interrogate_schober(const int cur_step)
        void interrogate_rodeo(const int cur_step)
