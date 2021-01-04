/// @file KalmanODE.h

#ifndef KalmanODE_h
#define KalmanODE_h 1

// This is looking in the `include` folder of the kalmantv library
#include <kalmantv/KalmanTV.h>
#include <Eigen/Dense>

namespace kalmanode {

  using namespace Eigen;

  /// Implementation of the KalmanODE 
  class KalmanODE {
  private:
    // set by constructor
    int n_state_;
    int n_meas_;
    double t_min_;
    double t_max_;
    int n_eval_;
    VectorXd mu_state_;
    MatrixXd wgt_state_;
    MatrixXd var_state_;
    MatrixXd z_state;
    // internal variables
    int n_steps_;
    MatrixXd mu_state_pred_;
    MatrixXd var_state_pred_;
    MatrixXd mu_state_filt_;
    MatrixXd var_state_filt_;
    VectorXd x_meas_;
    VectorXd mu_meas_;
    MatrixXd var_meas_;
    kalmantv::KalmanTV ktv_;
    VectorXd time_;
    VectorXd tx_state_;
    MatrixXd twgt_meas_;
    LLT<MatrixXd> tllt_state_;
  public:
    /// Typedefs
    typedef Ref<VectorXd> RefVectorXd;
    typedef const Ref<const VectorXd> cRefVectorXd;
    typedef Ref<MatrixXd> RefMatrixXd;
    typedef const Ref<const MatrixXd> cRefMatrixXd;
    typedef Map<VectorXd> MapVectorXd;
    typedef Map<const VectorXd> cMapVectorXd;
    typedef Map<MatrixXd> MapMatrixXd;
    typedef Map<const MatrixXd> cMapMatrixXd;
    /// Constructor.
    KalmanODE(int n_state, int n_meas, double t_min, double t_max, int n_eval);
    /// Solve ODE system.
    template class<model>
      void solve_sim(RefMatrix x_sol, model& ode, cRefVectorXd& x0,
		     cRefVector& W, cRefVectorXd& theta, cRefVector& z_state);
    // template class<model>
    //   void solve_mv(RefMatrix mu_sol, RefMatrix var_sol,
    // 		    model& ode, cRefVectorXd& x0,
    // 		    cRefVector& W, cRefVectorXd& theta);
  };

  inline KalmanODE::KalmanODE(int n_state, int n_meas,
			      double t_min, double t_max, int n_eval) {
    n_state_ = n_state;
    n_meas_ = n_meas;
    ///
  }
  
} // end namespace kalmanode

#endif
