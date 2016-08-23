
#pragma once

#include <string>

#include "common.h"
#include "Loss.h"
#include "Solver.h"

namespace solvers {

class SGD : public Solver {
 public:
  SGD(const size_t nfeatures,
      const Double lr,
      const Double lambda,
      const std::string& loss);

  void startDecay();

  void decay(const double multiplier = 0.5);

  size_t t() const {
    return t_;
  }

  template <typename Derived>
  void iterate(const Eigen::MatrixBase<Derived>& x, // x is a row vector
               const Double y,
               const size_t idx) {
    const Double stepSize = decay_ ? 2.0 / (lambda_ * (t_ - t0_ + gamma_)) : lr_;

    grad_ = Loss::computeGradient(loss_, x, x * w_, y);

    // SGD update
    w_ = w_ - stepSize * (grad_ + lambda_ * w_);

    ++t_;
  }

 private:
  Double lr_;

  const Double lambda_;

  bool decay_;

  size_t t_; // iteration

  size_t t0_;

  size_t gamma_;

  Vector grad_;
};
}
