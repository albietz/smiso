
#pragma once

#include <random>
#include <string>

#include "common.h"

namespace solvers {

class Loss {
 public:
  static Double computeLoss(const std::string& loss,
                            const Double pred,
                            const Double y) {
    if (loss == "l2") {
      const auto err = pred - y;
      return 0.5 * err * err;
    } else if (loss == "logistic") {
      const auto sigm = 1.0 / (1 + std::exp(-pred));
      return - y * std::log(sigm) - (1 - y) * std::log(1 - sigm);
    } else if (loss == "squared_hinge") {
      const Double s = y > 0 ? pred : -pred;
      const Double hinge = std::max(0.0, 1.0 - s);
      return hinge * hinge;
    } else {
      std::cerr << "loss not supported: " << loss;
      return 0;
    }
  }

  template <typename Derived>
  static Vector computeGradient(const std::string& loss,
                                const Eigen::MatrixBase<Derived>& x,
                                const Double pred,
                                const Double y) {
    Vector g;
    if (loss == "l2") {
      g = (pred - y) * x.transpose();
    } else if (loss == "logistic") {
      const auto sigm = 1.0 / (1 + std::exp(-pred));
      g = (sigm - y) * x.transpose();
    } else if (loss == "squared_hinge") {
      const Double s = y > 0 ? pred : -pred;
      if (s > 1) {
        g = Vector::Zero(x.size());
      } else {
        g = (y > 0 ? -2 : 2) * (1.0 - s) * x.transpose();
      }
    } else {
      std::cerr << "loss not supported: " << loss;
      g = Vector::Zero(x.size());
    }

    if (gradSigma_ > 0) {
      std::normal_distribution<Double> distr(0.0, gradSigma_);
      g = g.unaryExpr([&distr](Double val) {
          return val + distr(gen_);
        });
    }
    return g;
  }

  static void setGradSigma(const Double gradSigma) {
    std::cerr << "setting gradient std dev to " << gradSigma << std::endl;
    gradSigma_ = gradSigma;
  }

  static Double gradSigma() {
    return gradSigma_;
  }

 private:
  static std::mt19937 gen_;

  static Double gradSigma_;
};
}
