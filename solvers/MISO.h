
#pragma once

#include <string>

#include "common.h"
#include "loss.h"
#include "Solver.h"

namespace solvers {

class MISO : public Solver {
 public:
  MISO(const size_t nfeatures,
       const size_t nexamples,
       const Double lambda,
       const std::string& loss);

  void startDecay();

  size_t nexamples() const {
    return n_;
  }

  size_t t() const {
    return t_;
  }

  template <typename Derived>
  void iterate(const Eigen::MatrixBase<Derived>& x, // x is a row vector
               const Double y,
               const size_t idx) {
    const Double stepSize = decay_ ?
      std::min<Double>(1, 2 * static_cast<Double>(n_) / (t_ - t0_ + n_)) : 1;

    const auto grad = computeGradient(loss_, x, x * w_, y);

    const auto ziOld = z_.row(idx).transpose();

    const Vector zi = (1 - stepSize) * ziOld - stepSize / lambda_ * grad;

    w_ = w_ + 1.0 / n_ * (zi - ziOld);

    z_.row(idx) = zi.transpose();

    ++t_;
  }

 private:
  const size_t n_; // number of examples/clusters in the dataset

  Matrix z_;

  const Double lambda_;

  bool decay_;

  size_t t_; // iteration

  size_t t0_;
};
}
