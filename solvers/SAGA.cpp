
#include "SAGA.h"

namespace solvers {

SAGABase::SAGABase(const size_t nfeatures,
                   const size_t nexamples,
                   const Double lr,
                   const Double lambda,
                   const std::string& loss)
  : Solver(nfeatures, loss), n_(nexamples), lr_(lr), lambda_(lambda), t_(1) {
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
