
#pragma once

#include <Eigen/Core>
#include <iostream>

namespace solvers {

using Double = float;

using Matrix =
  Eigen::Matrix<Double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixMap = Eigen::Map<const Matrix>;

using Vector = Eigen::Matrix<Double, Eigen::Dynamic, 1>;
using VectorMap = Eigen::Map<const Vector>;

using IdxVector = Eigen::Matrix<int64_t, Eigen::Dynamic, 1>;
using IdxVectorMap = Eigen::Map<const IdxVector>;
}