#include <Eigen/Dense>
using namespace Eigen;
#include "KalmanTV.h"
#include <random>

// just does smooth
void kalman_ode_chkrebtii(double* mu_state_smooths,
                          double* var_state_smooths,
                          const int n_dim_meas, const int n_dim_state,
                          const int n_eval,
                          const double* x0_state,
                          const double tmin, const double tmax,
                          const double* wgt_state,
                          const double* mu_state, const double* var_state,
                          const double* wgt_meas, const double* z_state_sim) {
  // convert to Eigen types, e.g.,
  // note that Eigen only knows about vectors and matrices, so
  // 3d arrays should be converted to matrices by concatenating 3rd dimensions
  // into second.
  // typedefs
  typedef Map<VectorXd> MapVectorXd;
  typedef Map<const VectorXd> cMapVectorXd;
  typedef Map<MatrixXd> MapMatrixXd;
  typedef Map<const MatrixXd> cMapMatrixXd;

  // mappings
  MapMatrixXd mu_state_smooths_(mu_state_smooths, n_dim_state, n_eval+1);
  MapMatrixXd var_state_smooths_(var_state_smooths, n_dim_state, n_dim_state*(n_eval+1));
  cMapVectorXd x0_state_(x0_state, n_dim_state);
  cMapVectorXd wgt_state_(wgt_state, n_dim_state);
  cMapVectorXd mu_state_(mu_state, n_dim_state);
  cMapMatrixXd var_state_(var_state, n_dim_state, n_dim_state);
  cMapMatrixXd wgt_meas_(wgt_meas, n_dim_meas, n_dim_state);
  cMapMatrixXd z_state_sim_(z_state_sim, n_dim_state, n_eval+1);

  // memory allocation
  VectorXd mu_meas = VectorXd::Zero(n_dim_meas);
  MatrixXd var_meass = MatrixXd::Zero(n_dim_meas, n_dim_meas*(n_eval+1));
  MatrixXd x_meass = MatrixXd::Zero(n_dim_meas, n_eval+1);
  MatrixXd mu_state_filts = MatrixXd::Zero(n_dim_state, n_eval+1);
  MatrixXd var_state_filts = MatrixXd::Zero(n_dim_state, n_dim_state*(n_eval+1));
  MatrixXd mu_state_preds = MatrixXd::Zero(n_dim_state, n_eval+1);
  MatrixXd var_state_preds = MatrixXd::Zero(n_dim_state, n_dim_state*(n_eval+1));
  
  // temp variables
  MatrixXd tvar_state_ = MatrixXd::Zero(n_dim_state, n_dim_state);
  VectorXd x_state_tt = VectorXd::Zero(n_dim_state);
  MatrixXd var_state_meas = MatrixXd::Zero(n_dim_meas, n_dim_state);

  // initialization
  // use (i), (i,j), for elementwise access, and .block() for matrix subblocks.
  // more on all this here: https://eigen.tuxfamily.org/dox/group__TutorialBlockOperations.html
  mu_state_filts.col(0) = x0_state_;
  x_meass.col(0).noalias() = x0_state_*wgt_meas_.adjoint();
  mu_state_preds.col(0) = mu_state_filts.col(0);
  var_state_preds.block(0, 0, n_dim_state, n_dim_state) = var_state_filts.block(0, 0, n_dim_state, n_dim_state);

  // forward pass
  KalmanTV::KalmanTV KFS(n_dim_meas, n_dim_state);
  for(int t=0; t<n_eval; t++) {
    KFS.predict(mu_state_preds.col(t+1),
	            	var_state_preds.block(0, n_dim_state*(t+1),
                                      n_dim_state, n_dim_state),
		            mu_state_filts.col(t),
                var_state_filts.block(0, n_dim_state*(t+1),
				                              n_dim_state, n_dim_state),
                mu_state_,
                wgt_state_,
                var_state_);
    tvar_state_.noalias() = wgt_meas_*var_state_preds.block(0, n_dim_state*(t+1),
                                                           n_dim_state, n_dim_state);
    var_meass.block(0, n_dim_meas*(t+1), n_dim_meas, n_dim_meas).noalias() = tvar_state_ * wgt_meas_.adjoint();
    KFS.state_sim(x_state_tt,
                  mu_state_preds.col(t+1),
                  var_state_preds.block(0, n_dim_state*(t+1),
                                        n_dim_state, n_dim_state),
                  z_state_sim_.col(t));
    x_meass(0, t+1) = std::sin(2*(tmin + (tmax-tmin)*(t+1)/n_eval)) - x_state_tt(0);
    KFS.update(mu_state_filts.col(t+1),
               var_state_filts.block(0, n_dim_state*(t+1),
                                    n_dim_state, n_dim_state),
               mu_state_preds.col(t+1),
               var_state_preds.block(0, n_dim_state*(t+1),
                                     n_dim_state, n_dim_state),
               x_meass.col(t+1),
               mu_meas,
               wgt_meas_,
               var_meass.block(0, n_dim_meas*(t+1),
                              n_dim_meas, n_dim_meas));
  }
  // backward pass
  mu_state_smooths_.col(n_eval) = mu_state_filts.col(n_eval);
  var_state_smooths_.block(0, n_dim_state*n_eval, n_dim_state, n_dim_state) = var_state_filts.block(0, n_dim_state*n_eval,
                                                                                                    n_dim_state, n_dim_state);
  for(int t=n_eval-1; t>-1; t--) {
    KFS.smooth_mv(mu_state_smooths_.col(t),
                  var_state_smooths_.block(0, n_dim_state*t,
                                           n_dim_state, n_dim_state),
                  mu_state_smooths_.col(t+1),
                  var_state_smooths_.block(0, n_dim_state*(t+1),
                                           n_dim_state, n_dim_state),
                  mu_state_filts.col(t),
                  var_state_filts.block(0, n_dim_state*t,
                                        n_dim_state, n_dim_state),
                  mu_state_preds.col(t+1),
                  var_state_preds.block(0, n_dim_state*(t+1),
                                        n_dim_state, n_dim_state),
                  wgt_state_);
  }
  return;
}

int main(){
  int N, q, dim_state, dim_meas;
  double tmin, tmax;
  
  N = 49;
  q = 2;
  dim_state = q+2;
  dim_meas = 1;
  tmin = 0;
  tmax = 10;
  double x0_state[dim_state] = {-1, 0, 1, -0.17677222817373264};
  double wgt_meas[dim_state] = {0, 0, 1, 0};
  double mu_state[dim_state] = {0, 0, 0, 0};
  double wgt_state[dim_state*dim_state] = {9.99999858e-01,  1.99928416e-01,  1.97963822e-02, 1.13705826e-03,
                            -2.75723995e-06,  9.98613478e-01,  1.96041243e-01, 1.61555217e-02,
                            -3.91753450e-05, -1.97006822e-02,  9.43383853e-01, 1.44311263e-01,
                            -3.49938778e-04, -1.75993409e-01, -5.13046373e-01, 4.81299188e-01};
  double var_state[dim_state*dim_state] = {9.61550128e-09, 1.61612052e-07, 1.78475368e-06, 8.39746571e-06,
                            1.61611289e-07, 2.80769413e-06, 3.26251049e-05, 1.70003458e-04,
                            1.78475089e-06, 3.26251088e-05, 4.12852485e-04, 2.60321758e-03,
                            8.39746640e-06, 1.70003456e-04, 2.60321757e-03, 2.71508959e-02};
  
  std::random_device rd;
  std::mt19937 gen(rd());

  double z_state_sim[dim_state*dim_state];
  for(int i=0; i<dim_state*dim_state; ++i) {
    std::normal_distribution<double> d(0, 1);
    z_state_sim[i] = d(gen);
    
  }

  double mu_state_smooths[dim_state][50] = {};
  double var_state_smooths[dim_state][dim_state*50] = {};
  kalman_ode_chkrebtii(& mu_state_smooths, & var_state_smooths, & dim_meas, & dim_state, & N, & x0_state, & tmin, & tmax, & wgt_state, & mu_state, & var_state, & wgt_meas, & z_state_sim);
  std::cout <<"mu_state_smooths="<< mu_state_smooths[0];
  return 0;
}