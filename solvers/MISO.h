
#pragma once

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

// Naive Eigen-based sparse implementation that doesn't require the same
// sparsity pattern every time the same idx appears (and hence doesn't need
// special initialization for the Z matrix).
class SparseMISONaive : public MISOBase {
 public:
  SparseMISONaive(const size_t nfeatures,
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

// optimized sparse implementation which requires that examples have the same
// sparsity pattern for any fixed idx as that given during initialization.
// Initialization with a full data matrix using initZ is required.
// Indices in each CSR row vector need to be sorted.
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

    // [hacky] retrieve original sparse matrix from the view x
    // assumes that x was passed as a block of a sparse matrix
    const auto& Xblock = x.derived();
    const auto& X = Xblock.nestedExpression();
    const size_t sz = z_.outerIndexPtr()[idx + 1] - z_.outerIndexPtr()[idx];
    const size_t xRow = Xblock.startRow();
    if (static_cast<size_t>(X.outerIndexPtr()[xRow + 1] -
                            X.outerIndexPtr()[xRow]) != sz) {
      LOG_EVERY_N(ERROR, 1000)
        << "size mismatch in sparse value arrays! Did you call initZ?";
      return;
    }
    RowVectorMap xMap(X.valuePtr() + X.outerIndexPtr()[xRow], sz);
    Eigen::Map<Vector> zMap(z_.valuePtr() + z_.outerIndexPtr()[idx], sz);

    grad_.resize(sz);
    Loss::computeGradient<Vector, RowVectorMap>(grad_, loss_, xMap, pred, y);

    ziOld_ = z_.row(idx).transpose();
    zMap = (1 - stepSize) * zMap - stepSize / lambda_ * grad_;

    w_ += 1.0 / n_ * (z_.row(idx).transpose() - ziOld_);

    ++t_;
  }

  Double lowerBound() const;

 private:
  SpMatrix z_;

  Vector grad_;

  SpVector ziOld_;
};
}
