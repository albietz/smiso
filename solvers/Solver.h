
#pragma once

#include <iostream>

#include "common.h"

namespace solvers {

class Solver {
 public:
  Solver(const size_t nfeatures,
         const std::string& loss)
    : nfeatures_(nfeatures),
      w_(Vector::Zero(nfeatures_)),
      loss_(loss) {
  }

  virtual ~Solver() {
  }

  // delete copy and move constructors
  Solver(const Solver&) = delete;
  Solver(Solver&&) = delete;
  Solver& operator=(const Solver&) = delete;
  Solver& operator=(Solver&&) = delete;

  size_t nfeatures() const {
    return nfeatures_;
  }

  const Vector& w() const {
    return w_;
  }

  Double* wdata() {
    return w_.data();
  }

 private:
  const size_t nfeatures_;

 protected:
  Vector w_;

  const std::string loss_;
};

template <typename SolverT>
void iterateBlock(SolverT& solver,
                  const size_t blockSize,
                  const Double* const XData,
                  const Double* const yData,
                  const int64_t* const idxData) {
  const MatrixMap Xmap(XData, blockSize, solver.nfeatures());
  for (size_t i = 0; i < blockSize; ++i) {
    solver.iterate(Xmap.row(i), yData[i], idxData[i]);
  }
}

template <typename SolverT>
void iterateBlockIndexed(SolverT& solver,
                         const size_t dataSize,
                         const Double* const XData,
                         const Double* const yData,
                         const size_t blockSize,
                         const int64_t* const idxData) {
  const MatrixMap Xmap(XData, dataSize, solver.nfeatures());
  for (size_t i = 0; i < blockSize; ++i) {
    solver.iterate(Xmap.row(idxData[i]), yData[idxData[i]], idxData[i]);
  }
}
}
