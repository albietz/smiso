
#include "SGD.h"

namespace solvers {

SGD::SGD(const size_t nfeatures,
         const Double lr,
         const Double lambda,
         const std::string& loss)
  : Solver(nfeatures, loss),
    lr_(lr),
    lambda_(lambda),
    decay_(false),
    t_(1),
    t0_(1) {
}

void SGD::startDecay() {
  decay_ = true;
  t0_ = t_;
}
}
