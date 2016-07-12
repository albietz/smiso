
#pragma once

#include <string>

#include "common.h"

namespace solvers {

inline Double computeLoss(const std::string& loss,
                   const Double pred,
                   const Double y) {
  if (loss == "l2") {
    const auto err = pred - y;
    return 0.5 * err * err;
  } else if (loss == "logistic") {
    const auto sigm = 1.0 / (1 + std::exp(-pred));
    return - y * log(sigm) - (1 - y) * log(1 - sigm);
  } else {
    return 0;
  }
}

template <typename Derived>
Vector computeGradient(const std::string& loss,
                       const Eigen::MatrixBase<Derived>& x,
                       const Double pred,
                       const Double y) {
  if (loss == "l2") {
    return (pred - y) * x;
  } else if (loss == "logistic") {
    const auto sigm = 1.0 / (1 + std::exp(-pred));
    return (sigm - y) * x;
  } else {
    std::cerr << "loss not supported: " << loss;
    return Vector::Zero(x.size());
  }
}
}
