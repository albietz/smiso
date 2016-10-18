
#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <iostream>

namespace solvers {

#if USE_FLOAT
using Double = float;
#else
using Double = double;
#endif

using Matrix =
  Eigen::Matrix<Double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixMap = Eigen::Map<const Matrix>;

using Vector = Eigen::Matrix<Double, Eigen::Dynamic, 1>;
using VectorMap = Eigen::Map<const Vector>;

using IdxVector = Eigen::Matrix<int64_t, Eigen::Dynamic, 1>;
using IdxVectorMap = Eigen::Map<const IdxVector>;

using SpMatrix = Eigen::SparseMatrix<Double, Eigen::RowMajor>;
using SpMatrixMap = Eigen::Map<const SpMatrix>;
using SpVector = Eigen::SparseVector<Double, Eigen::RowMajor>;

/** subtract row mean from each row in data matrix */
inline void center(Double* const XData,
                   const size_t rows,
                   const size_t cols) {
#pragma omp parallel for
  for (size_t r = 0; r < rows; ++r) {
    Double sum = 0;
    for (size_t c = 0; c < cols; ++c) {
      sum += XData[r*cols + c];
    }
    sum /= cols;
    for (size_t c = 0; c < cols; ++c) {
      XData[r*cols + c] -= sum;
    }
  }
}

/** L2 normalize each row in data matrix */
inline void normalize(Double* const XData,
                      const size_t rows,
                      const size_t cols) {
#pragma omp parallel for
  for (size_t r = 0; r < rows; ++r) {
    Double sum = 0;
    for (size_t c = 0; c < cols; ++c) {
      Double x = XData[r*cols + c];
      sum += x * x;
    }
    sum = std::sqrt(sum);
    if (sum > 0) {
      for (size_t c = 0; c < cols; ++c) {
        XData[r*cols + c] /= sum;
      }
    }
  }
}
}
