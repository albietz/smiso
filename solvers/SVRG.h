
#pragma once

#include <string>
#include <vector>

#include "common.h"
#include "Solver.h"

namespace solvers {

class SVRGBase : public Solver {
 public:
  SVRGBase(const size_t nfeatures,
           const size_t nexamples,
           const Double lr,
           const Double lambda,
           const std::string& loss,
           const std::string& prox = "none",
           const Double proxWeight = 0);

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

class SVRG : public SVRGBase {
 public:
  SVRG(const size_t nfeatures,
       const size_t nexamples,
       const Double lr,
       const Double lambda,
       const std::string& loss,
       const std::string& prox,
       const Double proxWeight);

  template <typename Derived>
  void iterate(const Eigen::MatrixBase<Derived>& x, // x is a row vector
               const Double y,
               const size_t idx);

  template <typename Derived>
  void computeSnapshot(const Eigen::MatrixBase<Derived>& X,
                       const Double* const y);

 private:
  Vector grad_;

  Vector gradSnap_; // average gradient snapshot

  Vector wSnap_; // snapshot parameter

  Vector gbarSnap_; // average gradient snapshot
};
}

#include "SVRG-inl.h"
