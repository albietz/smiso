
#include "SAGA.h"

namespace solvers {

SAGABase::SAGABase(const size_t nfeatures,
                   const size_t nexamples,
                   const Double lr,
                   const Double lambda,
                   const std::string& loss)
  : Solver(nfeatures, loss),
    n_(nexamples),
    lr_(lr),
    lambda_(lambda),
    decay_(false),
    t_(1),
    t0_(1) {
}

void SAGABase::startDecay() {
  decay_ = true;
  t0_ = t_;
  gamma_ = static_cast<size_t>(2.0 / (lambda_ * lr_)) + 1;
}

SAGA::SAGA(const size_t nfeatures,
           const size_t nexamples,
           const Double lr,
           const Double lambda,
           const std::string& loss)
  : SAGABase(nfeatures, nexamples, lr, lambda, loss),
    z_(Matrix::Zero(nexamples, nfeatures)),
    grad_(nfeatures),
    zbar_(Vector::Zero(nfeatures)) {
}
}
