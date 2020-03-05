/// @file KalmanTest.h

#ifndef KalmanTest_h
#define KalmanTest_h 1

#include <Eigen/Dense>
#include <iostream>
#include "KalmanTV.h"
#include <random>

namespace KalmanTest {
  using namespace Eigen;
  class KalmanTest : public KalmanTV::KalmanTV {
  private:
    int n_meas_;
    int n_state_;
    int n_steps_;
    VectorXd mu_meas;
    MatrixXd var_meas;
    MatrixXd mu_state_filts;
    MatrixXd var_state_filts;
    MatrixXd mu_state_preds;
    MatrixXd var_state_preds;
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
    /// Constructor
    KalmanTest(int n_meas, int n_state, int n_steps);
    void filter_smooth(RefMatrixXd mu_state_smooths,
                       RefMatrixXd var_state_smooths,
                       cRefVectorXd& x0_state,
                       cRefMatrixXd& wgt_state,
                       cRefVectorXd& mu_state,
                       cRefMatrixXd& var_state,
                       cRefMatrixXd& wgt_meas,
                       cRefMatrixXd& x_meass,
                       cRefMatrixXd& z_state_sim);

    void filter_smooth(double* mu_state_smooths,
                       double* var_state_smooths,
                       const double* x0_state,
                       const double* wgt_state,
                       const double* mu_state, 
                       const double* var_state,
                       const double* wgt_meas,
                       const double* x_meass,
                       const double* z_state_sim);
    void test();
  };
  inline KalmanTest::KalmanTest(int n_meas, int n_state, int n_steps) : KalmanTV(n_meas, n_state) {
    // problem dimensions
    n_meas_ = n_meas;
    n_state_ = n_state;
    n_steps_ = n_steps;
    // memory allocation
    mu_meas = VectorXd::Zero(n_meas_);
    var_meas = MatrixXd::Zero(n_meas_, n_meas_);
    mu_state_filts = MatrixXd::Zero(n_state_, n_steps_);
    var_state_filts = MatrixXd::Zero(n_state_, n_state_*n_steps_);
    mu_state_preds = MatrixXd::Zero(n_state_, n_steps_);
    var_state_preds = MatrixXd::Zero(n_state_, n_state_*n_steps_);
  }
  inline void KalmanTest::filter_smooth(RefMatrixXd mu_state_smooths,
                                        RefMatrixXd var_state_smooths,
                                        cRefVectorXd& x0_state,
                                        cRefMatrixXd& wgt_state,
                                        cRefVectorXd& mu_state,
                                        cRefMatrixXd& var_state,
                                        cRefMatrixXd& wgt_meas,
                                        cRefMatrixXd& x_meass,
                                        cRefMatrixXd& z_state_sim) {
    mu_state_filts.col(0) = x0_state;
    mu_state_preds.col(0) = mu_state_filts.col(0);
    var_state_preds.block(0, 0, n_state_, n_state_) = var_state_filts.block(0, 0, n_state_, n_state_);
    // forward pass
    for(int t=0; t<n_steps_-1; t++) {
      filter(mu_state_preds.col(t+1),
             var_state_preds.block(0, n_state_*(t+1),
                                   n_state_, n_state_),
             mu_state_filts.col(t+1),
             var_state_filts.block(0, n_state_*(t+1),
                                   n_state_, n_state_),
             mu_state_filts.col(t),
             var_state_filts.block(0, n_state_*t,
                                   n_state_, n_state_),
             mu_state,
             wgt_state,
             var_state,
             x_meass.col(t+1),
             mu_meas,
             wgt_meas,
             var_meas);
    }
    // backward pass
    mu_state_smooths.col(n_steps_-1) = mu_state_filts.col(n_steps_-1);
    var_state_smooths.block(0, n_state_*(n_steps_-1), n_state_, n_state_) = var_state_filts.block(0, n_state_*(n_steps_-1),
                                                                                                      n_state_, n_state_);
    for(int t=n_steps_-2; t>-1; t--) {
      smooth_mv(mu_state_smooths.col(t),
                var_state_smooths.block(0, n_state_*t,
                                         n_state_, n_state_),
                mu_state_smooths.col(t+1),
                var_state_smooths.block(0, n_state_*(t+1),
                                        n_state_, n_state_),
                mu_state_filts.col(t),
                var_state_filts.block(0, n_state_*t,
                                      n_state_, n_state_),
                mu_state_preds.col(t+1),
                var_state_preds.block(0, n_state_*(t+1),
                                      n_state_, n_state_),
                wgt_state);
    }
    return;
  }
  inline void KalmanTest::filter_smooth(double* mu_state_smooths,
                                        double* var_state_smooths,
                                        const double* x0_state,
                                        const double* wgt_state,
                                        const double* mu_state, 
                                        const double* var_state,
                                        const double* wgt_meas,
                                        const double* x_meass,
                                        const double* z_state_sim) {
    MapMatrixXd mu_state_smooths_(mu_state_smooths, n_state_, n_steps_);
    MapMatrixXd var_state_smooths_(var_state_smooths, n_state_, n_state_ * n_steps_);
    cMapVectorXd x0_state_(x0_state, n_state_);
    cMapMatrixXd wgt_state_(wgt_state, n_state_, n_state_);
    cMapVectorXd mu_state_(mu_state, n_state_);
    cMapMatrixXd var_state_(var_state, n_state_, n_state_);
    cMapMatrixXd wgt_meas_(wgt_meas, n_meas_, n_state_);
    cMapMatrixXd x_meass_(x_meass, n_meas_, n_steps_);
    cMapMatrixXd z_state_sim_(z_state_sim, n_state_, n_steps_);
    filter_smooth(mu_state_smooths_, var_state_smooths_,
                  x0_state_, wgt_state_, mu_state_,
                  var_state_, wgt_meas_, x_meass_,
                  z_state_sim_);
    return;
  }
} // namespace KalmanTest

#endif
