
#include "Loss.h"
#include "Prox.h"

namespace solvers {

template <typename Derived>
void SGD::iterate(const Eigen::MatrixBase<Derived>& x, // x is a row vector
                  const Double y,
                  const size_t idx) {
  const Double stepSize = getStepSize();

  Loss::computeGradient<Vector, Derived>(grad_, loss_, x, x * w_, y);

  // SGD update
  w_ = w_ - stepSize * (grad_ + lambda_ * w_);

  Prox::applyProx(w_, prox_, stepSize * proxWeight_);


  ++t_;
}

template <typename Derived>
void SparseSGD::iterate(
    const Eigen::SparseMatrixBase<Derived>& x, // x is a row vector
    const Double y,
    const size_t idx) {
  const Double stepSize = getStepSize();

  Loss::computeGradient<SpVector, Derived>(
      grad_, loss_, x, s_ * static_cast<Double>(x * ws_), y);

  s_ *= (1 - stepSize * lambda_);
  ws_ -= (stepSize / s_) * grad_;

  ++t_;

  // for numerical stability
  if (s_ < 1e-9) {
    LOG(INFO) << "resetting ws and s";
    ws_ = s_ * ws_;
    s_ = 1.0;
  }
}
}
