
#pragma once

#include "common.h"

namespace solvers {

class Util {
 public:
  template <typename VectorT, typename Derived>
  static Eigen::Map<VectorT> denseRowMap(Eigen::SparseMatrixBase<Derived>& X,
                                         const size_t row);

  template <typename VectorT, typename Derived>
  static Eigen::Map<const VectorT>
      denseRowMap(const Eigen::SparseMatrixBase<Derived>& Xin,
                  const size_t row);
};

template <typename VectorT, typename Derived>
Eigen::Map<VectorT> Util::denseRowMap(Eigen::SparseMatrixBase<Derived>& Xin,
                                      const size_t row) {
  auto& X = Xin.derived();
  const size_t sz = X.outerIndexPtr()[row + 1] - X.outerIndexPtr()[row];
  return Eigen::Map<VectorT>(X.valuePtr() + X.outerIndexPtr()[row], sz);
}

template <typename VectorT, typename Derived>
Eigen::Map<const VectorT>
    Util::denseRowMap(const Eigen::SparseMatrixBase<Derived>& Xin,
                      const size_t row) {
  const auto& X = Xin.derived();
  const size_t sz = X.outerIndexPtr()[row + 1] - X.outerIndexPtr()[row];
  return Eigen::Map<const VectorT>(X.valuePtr() + X.outerIndexPtr()[row], sz);
}
}
