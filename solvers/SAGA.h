
#pragma once

#include <string>

#include "common.h"
#include "Solver.h"

namespace solvers {

class SAGABase : public Solver {
 public:
  SAGABase(const size_t nfeatures,
           const size_t nexamples,
           const Double lr,
           const Double lambda,
           const std::string& loss);

  size_t nexamples() const {
    return n_;
  }

  size_t t() const {
    return t_;
  }

 protected:
  const size_t n_; // number of examples/clusters in the dataset

  Double lr_; // step size

  const Double lambda_;

  size_t t_; // iteration
};

class SAGA : public SAGABase {
 public:
  SAGA(const size_t nfeatures,
       const size_t nexamples,
       const Double lr,
       const Double lambda,
       const std::string& loss);

  template <typename Derived>
  void iterate(const Eigen::MatrixBase<Derived>& x, // x is a row vector
               const Double y,
               const size_t idx);

 private:
  Matrix z_;

  Vector grad_;

  Vector zbar_;
};
}

#include "SAGA-inl.h"
