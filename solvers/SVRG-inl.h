
#include "Loss.h"
#include "Prox.h"
#include "Util.h"

namespace solvers {

template <typename Derived>
void SVRG::iterate(const Eigen::MatrixBase<Derived>& x, // x is a row vector
                   const Double y,
                   const size_t idx) {
  const Double pred = x * w_;
  const Double predSnap = x * wSnap_;

  Loss::computeGradient<Vector, Derived>(grad_, loss_, x, pred, y);
  Loss::computeGradient<Vector, Derived>(gradSnap_, loss_, x, predSnap, y);
  grad_ += lambda_ * w_;
  gradSnap_ += lambda_ * wSnap_;

  w_ = w_ - lr_ * (grad_ - gradSnap_ + gbarSnap_);

  Prox::applyProx(w_, prox_, lr_ * proxWeight_);

  ++t_;
}

template <typename Derived>
void SVRG::computeSnapshot(const Eigen::MatrixBase<Derived>& X,
                           const Double* const y) {
  Double pred, grad;
  wSnap_ = w_;
  gbarSnap_ = Vector::Zero(nfeatures());

  for (size_t i = 0; i < X.rows(); ++i) {
    pred = X.row(i) * w_;
    grad = Loss::computeGradient(loss_, pred, y[i]);
    gbarSnap_ += grad * X.row(i).transpose();
  }
  gbarSnap_ *= static_cast<Double>(1.0 / X.rows());
  gbarSnap_ += lambda_ * w_;
}
}
