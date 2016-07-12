
#pragma once

#include <string>

#include "common.h"
#include "loss.h"
#include "Solver.h"

namespace solvers {

class SGD : public Solver {
 public:
  SGD(const size_t nfeatures,
      const Double lr,
      const Double lambda,
      const std::string& loss);

  void startDecay();

  size_t t() const {
    return t_;
  }

  template <typename Derived>
  void iterate(const Eigen::MatrixBase<Derived>& x,
               const Double y,
               const size_t idx) {
    const Double stepSize = decay_ ? lr_ / (t_ - t0_ + 1) : lr_;

    const auto grad = computeGradient(loss_, x, x * w_, y);

    // SGD update
    w_ = w_ - stepSize * (grad + lambda_ * w_);

    ++t_;
  }

 private:
  const Double lr_;

  const Double lambda_;

  const std::string loss_;

  bool decay_;

  size_t t_; // iteration

  size_t t0_;
};
}
