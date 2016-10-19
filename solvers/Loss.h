
#pragma once

#include <random>
#include <string>
#include <type_traits>

#include "common.h"

namespace solvers {

namespace detail {

template <typename Derived, typename VecT>
using EBase = typename std::conditional<
  std::is_base_of<Eigen::MatrixBase<VecT>, VecT>::value,
  Eigen::MatrixBase<Derived>,
  Eigen::SparseMatrixBase<Derived>>::type;

inline void makeZero(SpVector& g, const size_t sz) {
  g.setZero();
}

inline void makeZero(Vector& g, const size_t sz) {
  g = Vector::Zero(sz);
}
}

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
      return 0.5 * hinge * hinge;
    } else {
      LOG(ERROR) << "loss not supported: " << loss;
      return 0;
    }
  }

  template <typename VectorT, typename Derived>
  static void computeGradient(VectorT& g,
                              const std::string& loss,
                              const detail::EBase<Derived, VectorT>& x,
                              const Double pred,
                              const Double y) {
    if (loss == "l2") {
      g = (pred - y) * x.transpose();
    } else if (loss == "logistic") {
      const auto sigm = 1.0 / (1 + std::exp(-pred));
      g = (sigm - y) * x.transpose();
    } else if (loss == "squared_hinge") {
      const Double s = y > 0 ? pred : -pred;
      if (s > 1) {
        detail::makeZero(g, x.size());
      } else {
        g = (y > 0 ? -1 : 1) * (1.0 - s) * x.transpose();
      }
    } else {
      LOG(ERROR) << "loss not supported: " << loss;
    }

    if (gradSigma_ > 0) {
      std::normal_distribution<Double> distr(0.0, gradSigma_);
      g = g.unaryExpr([&distr](Double val) {
          return val + distr(gen_);
        });
    }
  }

  static void setGradSigma(const Double gradSigma) {
    LOG(INFO) << "setting gradient std dev to " << gradSigma;
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
