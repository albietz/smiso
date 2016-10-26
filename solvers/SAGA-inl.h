
#include "Loss.h"

namespace solvers {

template <typename Derived>
void SAGA::iterate(const Eigen::MatrixBase<Derived>& x, // x is a row vector
                   const Double y,
                   const size_t idx) {
  const Double stepSize = getStepSize();
  const Double pred = x * w_;

  Loss::computeGradient<Vector, Derived>(grad_, loss_, x, pred, y);
  grad_ += lambda_ * w_;

  w_ = w_ - stepSize * (grad_ - z_.row(idx).transpose() + zbar_);

  zbar_ = zbar_ + 1.0 / n_ * (grad_ - z_.row(idx).transpose());

  z_.row(idx) = grad_.transpose();

  ++t_;
}
}
