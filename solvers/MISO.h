
#pragma once

#include <string>

#include "common.h"
#include "Loss.h"
#include "Solver.h"

namespace solvers {

class MISO : public Solver {
 public:
  MISO(const size_t nfeatures,
       const size_t nexamples,
       const Double lambda,
       const std::string& loss,
       const bool computeLB);

  void startDecay();

  void decay(const Double multiplier = 0.5);

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
      std::min<Double>(alpha_, 2 * static_cast<Double>(n_) / (t_ - t0_ + n_)) : alpha_;

    const Double pred = x * w_;

    const auto grad = Loss::computeGradient(loss_, x, pred, y);

    const auto ziOld = z_.row(idx).transpose();

    const Vector zi = (1 - stepSize) * ziOld - stepSize / lambda_ * grad;

    if (computeLB_) {
      c_[idx] = (1 - stepSize) * c_[idx] +
        stepSize * (Loss::computeLoss(loss_, pred, y) - grad.dot(w_));
    }

    w_ = w_ + 1.0 / n_ * (zi - ziOld);

    z_.row(idx) = zi.transpose();

    ++t_;
  }

  Double lowerBound() const;

 private:
  const size_t n_; // number of examples/clusters in the dataset

  Matrix z_;

  Double alpha_; // step size

  const Double lambda_;

  bool decay_;

  size_t t_; // iteration

  size_t t0_;

  bool computeLB_; // whether to compute lower bounds

  Vector c_; // for computing lower bounds
};
}
