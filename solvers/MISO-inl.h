
#include "Loss.h"

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

  ++t_;
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
}
