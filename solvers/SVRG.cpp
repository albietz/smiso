
#include "SVRG.h"

namespace solvers {

SVRGBase::SVRGBase(const size_t nfeatures,
                   const size_t nexamples,
                   const Double lr,
                   const Double lambda,
                   const std::string& loss,
                   const std::string& prox,
                   const Double proxWeight)
  : Solver(nfeatures, loss, prox, proxWeight),
    n_(nexamples),
    lr_(lr),
    lambda_(lambda),
    t_(1) {
}

SVRG::SVRG(const size_t nfeatures,
           const size_t nexamples,
           const Double lr,
           const Double lambda,
           const std::string& loss,
           const std::string& prox,
           const Double proxWeight)
  : SVRGBase(nfeatures, nexamples, lr, lambda, loss, prox, proxWeight),
    grad_(nfeatures),
    gradSnap_(nfeatures),
    wSnap_(Vector::Zero(nfeatures)),
    gbarSnap_(Vector::Zero(nfeatures)) {
}
}
