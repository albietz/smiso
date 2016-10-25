
namespace solvers {

template <typename VectorT, typename Derived>
void Loss::computeGradient(VectorT& g,
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
    g = g.unaryExpr([&distr](Double val) { return val + distr(gen_); });
  }
}
}
