
#include "Loss.h"
#include "Prox.h"
#include "Util.h"

namespace solvers {

template <typename Derived>
void MISO::iterate(const Eigen::MatrixBase<Derived>& x, // x is a row vector
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

  Prox::applyProx(w_, prox_, proxWeight_ * stepSize / (n_ * lambda_));

  ++t_;
}

template <typename Derived>
void SparseMISONaive::initFromX(const Eigen::SparseMatrixBase<Derived>& X) {
  // this didn't seem to help much, even hurts
  // later iterations compared to just reserve(nnz)
  // z_ = 0.0 * X;

  z_.reserve(X.nonZeros());
}

template <typename Derived>
void SparseMISONaive::iterate(
    const Eigen::SparseMatrixBase<Derived>& x, // x is a row vector
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

template <typename Derived>
void SparseMISO::initFromX(const Eigen::SparseMatrixBase<Derived>& X) {
  z_ = 0.0 * X;
}

template <typename Derived>
void SparseMISO::iterate(
    const Eigen::SparseMatrixBase<Derived>& x, // x is a row vector
    const Double y,
    const size_t idx) {
  const Double stepSize = getStepSize();

  const Double pred = x * w_;

  // [hacky] retrieve original sparse matrix from the view x
  // assumes that x was passed as a block of a sparse matrix
  const auto& Xblock = x.derived();
  const auto& X = Xblock.nestedExpression();
  RowVectorMap xMap = Util::denseRowMap<RowVector>(X, Xblock.startRow());
  auto zMap = Util::denseRowMap<Vector>(z_, idx);
  if (xMap.size() != zMap.size()) {
    LOG_EVERY_N(ERROR, 1000)
        << "size mismatch in sparse value arrays! ("
        << xMap.size() << " vs " << zMap.size()
        << ") Did you call init?";
    return;
  }

  grad_.resize(xMap.size());
  Loss::computeGradient<Vector, RowVectorMap>(grad_, loss_, xMap, pred, y);

  ziOld_ = z_.row(idx).transpose();
  zMap = (1 - stepSize) * zMap - stepSize / lambda_ * grad_;

  w_ += 1.0 / n_ * (z_.row(idx).transpose() - ziOld_);

  ++t_;
}
}
