
#pragma once

#include <glog/logging.h>
#include <string>

#include "common.h"
#include "Loss.h"
#include "Solver.h"

namespace solvers {

class MISOBase : public Solver {
 public:
  MISOBase(const size_t nfeatures,
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

  virtual Double lowerBound() const = 0;

 protected:
  Double getStepSize() const {
    return decay_ ?
      std::min<Double>(alpha_, 2 * static_cast<Double>(n_) / (t_ - t0_ + gamma_)) : alpha_;
  }

  const size_t n_; // number of examples/clusters in the dataset

  Double alpha_; // step size

  const Double lambda_;

  bool decay_;

  size_t t_; // iteration

  size_t t0_;

  size_t gamma_; // offset for decaying stepsize C / (gamma + t - t0)

  bool computeLB_; // whether to compute lower bounds

  Vector c_; // for computing lower bounds
};

class MISO : public MISOBase {
 public:
  MISO(const size_t nfeatures,
       const size_t nexamples,
       const Double lambda,
       const std::string& loss,
       const bool computeLB);

  template <typename Derived>
  void iterate(const Eigen::MatrixBase<Derived>& x, // x is a row vector
               const Double y,
               const size_t idx) {
    const Double stepSize = getStepSize();

    const Double pred = x * w_;

    Loss::computeGradient<Vector, Derived>(grad_, loss_, x, pred, y);

    const auto ziOld = z_.row(idx).transpose();

    zi_ = (1 - stepSize) * ziOld - stepSize / lambda_ * grad_;

    if (computeLB_) {
      c_[idx] = (1 - stepSize) * c_[idx] +
        stepSize * (Loss::computeLoss(loss_, pred, y) - grad_.dot(w_));
    }

    w_ = w_ + 1.0 / n_ * (zi_ - ziOld);

    z_.row(idx) = zi_.transpose();

    ++t_;
  }

  Double lowerBound() const;

 private:
  Matrix z_;

  Vector grad_;

  Vector zi_;
};

class SparseMISO : public MISOBase {
 public:
  SparseMISO(const size_t nfeatures,
             const size_t nexamples,
             const Double lambda,
             const std::string& loss,
             const bool computeLB);

  void initZ(const size_t nnz,
             const int32_t* const Xindptr,
             const int32_t* const Xindices,
             const Double* const Xvalues);

  template <typename Derived>
  void iterate(const Eigen::SparseMatrixBase<Derived>& x, // x is a row vector
               const Double y,
               const size_t idx) {
    const Double stepSize = getStepSize();

    const Double pred = x * w_;

    Loss::computeGradient<SpVector, Derived>(grad_, loss_, x, pred, y);

    const auto ziOld = z_.row(idx).transpose();

    zi_ = (1 - stepSize) * ziOld - stepSize / lambda_ * grad_;

    if (computeLB_) {
      c_[idx] = (1 - stepSize) * c_[idx] +
        stepSize * (Loss::computeLoss(loss_, pred, y) - grad_.dot(w_));
    }

    w_ += 1.0 / n_ * (zi_ - ziOld);

    z_.row(idx) = zi_.transpose();

    ++t_;
  }

  Double lowerBound() const;

 private:
  SpMatrix z_;

  SpVector grad_;

  SpVector zi_;
};
}
