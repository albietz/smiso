
#include "SGD.h"

namespace solvers {

SGDBase::SGDBase(const size_t nfeatures,
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

void SGDBase::startDecay() {
  decay_ = true;
  t0_ = t_;
  gamma_ = static_cast<size_t>(2.0 / (lambda_ * lr_)) + 1;
}
}
