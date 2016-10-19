
#include "MISO.h"

namespace solvers {

MISOBase::MISOBase(const size_t nfeatures,
                   const size_t nexamples,
                   const Double lambda,
                   const std::string& loss,
                   const bool computeLB)
  : Solver(nfeatures, loss),
    n_(nexamples),
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
           const bool computeLB)
  : MISOBase(nfeatures, nexamples, lambda, loss, computeLB),
    z_(Matrix::Zero(nexamples, nfeatures)),
    grad_(nfeatures),
    zi_(nfeatures) {
}

Double MISO::lowerBound() const {
  if (computeLB_) {
    return (c_ - lambda_ * z_ * w_).mean() + 0.5 * lambda_ * w_.squaredNorm();
  } else {
    std::cerr << "computeLB is false!";
    return 0;
  }
}

SparseMISO::SparseMISO(const size_t nfeatures,
                       const size_t nexamples,
                       const Double lambda,
                       const std::string& loss,
                       const bool computeLB)
  : MISOBase(nfeatures, nexamples, lambda, loss, computeLB),
    z_(nexamples, nfeatures),
    grad_(nfeatures),
    zi_(nfeatures) {
}

void SparseMISO::initZ(const size_t nnz,
                       const int32_t* const Xindptr,
                       const int32_t* const Xindices,
                       const Double* const Xvalues) {
  // this didn't seem to help much, even hurts
  // later iterations compared to just reserve(nnz)
  // const SpMatrixMap Xmap(n_, nfeatures(), nnz,
  //                        Xindptr, Xindices, Xvalues);
  // z_ = 0.0 * Xmap;

  z_.reserve(nnz);
}

Double SparseMISO::lowerBound() const {
  if (computeLB_) {
    return (c_ - lambda_ * z_ * w_).mean() + 0.5 * lambda_ * w_.squaredNorm();
  } else {
    std::cerr << "computeLB is false!";
    return 0;
  }
}
}
