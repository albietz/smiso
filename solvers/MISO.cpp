
#include "MISO.h"

namespace solvers {

MISO::MISO(const size_t nfeatures,
           const size_t nexamples,
           const Double lambda,
           const std::string& loss,
           const bool computeLB)
  : Solver(nfeatures, loss),
    n_(nexamples),
    z_(Matrix::Zero(nexamples, nfeatures)),
    alpha_(1.0),
    lambda_(lambda),
    decay_(false),
    t_(1),
    t0_(1),
    computeLB_(computeLB) {
  if (computeLB_) {
    c_ = Vector::Zero(n_);
  }
}

void MISO::startDecay() {
  decay_ = true;
  t0_ = t_;
  gamma_ = static_cast<size_t>(2 * static_cast<Double>(n_) / alpha_) + 1;
}

void MISO::decay(const Double multiplier) {
  alpha_ *= multiplier;
}

Double MISO::lowerBound() const {
  if (computeLB_) {
    return (c_ - lambda_ * z_ * w_).mean() + 0.5 * lambda_ * w_.squaredNorm();
  } else {
    std::cerr << "computeLB is false!";
    return 0;
  }
}
}
