
#include "MISO.h"

namespace solvers {

MISOBase::MISOBase(const size_t nfeatures,
                   const size_t nexamples,
                   const Double lambda,
                   const std::string& loss,
                   const bool computeLB,
                   const std::string& prox,
                   const Double proxWeight)
  : Solver(nfeatures, loss, prox, proxWeight),
    n_(nexamples),
    alpha_(1.0),
    lambda_(lambda),
    decay_(false),
    t_(1),
    t0_(1),
    q_(0),
    computeLB_(computeLB) {
  if (computeLB_) {
    c_ = Vector::Zero(n_);
  }
}

void MISOBase::startDecay() {
  decay_ = true;
  t0_ = t_;
  gamma_ = static_cast<size_t>(2 * static_cast<Double>(n_) / alpha_) + 1;
}

void MISOBase::decay(const Double multiplier) {
  alpha_ *= multiplier;
}

MISO::MISO(const size_t nfeatures,
           const size_t nexamples,
           const Double lambda,
           const std::string& loss,
           const bool computeLB,
           const std::string& prox,
           const Double proxWeight)
  : MISOBase(nfeatures, nexamples, lambda, loss, computeLB, prox, proxWeight),
    z_(Matrix::Zero(nexamples, nfeatures)),
    grad_(nfeatures),
    zi_(nfeatures),
    zbar_(Vector::Zero(nfeatures)) {
}

Double MISO::lowerBound() const {
  if (computeLB_) {
    return (c_ - lambda_ * z_ * w_).mean() + 0.5 * lambda_ * w_.squaredNorm();
  } else {
    LOG(ERROR) << "computeLB is false!";
    return 0;
  }
}

SparseMISONaive::SparseMISONaive(const size_t nfeatures,
                                 const size_t nexamples,
                                 const Double lambda,
                                 const std::string& loss,
                                 const bool computeLB)
  : MISOBase(nfeatures, nexamples, lambda, loss, computeLB),
    z_(nexamples, nfeatures),
    grad_(nfeatures),
    zi_(nfeatures) {
}

Double SparseMISONaive::lowerBound() const {
  if (computeLB_) {
    return (c_ - lambda_ * z_ * w_).mean() + 0.5 * lambda_ * w_.squaredNorm();
  } else {
    LOG(ERROR) << "computeLB is false!";
    return 0;
  }
}

SparseMISO::SparseMISO(const size_t nfeatures,
                       const size_t nexamples,
                       const Double lambda,
                       const std::string& loss,
                       const bool computeLB)
  : MISOBase(nfeatures, nexamples, lambda, loss, computeLB),
    z_(nexamples, nfeatures) {
}

Double SparseMISO::lowerBound() const {
  LOG(ERROR) << "lowerBound() not implemented";
  return 0;
}
}
