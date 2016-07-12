
#include "MISO.h"

namespace solvers {

MISO::MISO(const size_t nfeatures,
           const size_t nexamples,
           const Double lambda,
           const std::string& loss)
  : Solver(nfeatures, loss),
    n_(nexamples),
    z_(nexamples, nfeatures),
    lambda_(lambda),
    decay_(false),
    t_(1),
    t0_(1) {
}

void MISO::startDecay() {
  decay_ = true;
  t0_ = t_;
}
}
