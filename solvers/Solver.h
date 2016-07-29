
#pragma once

#include <iostream>

#include "common.h"

namespace solvers {

class Solver {
 public:
  Solver(const size_t nfeatures,
         const std::string& loss)
    : nfeatures_(nfeatures),
      w_(Vector::Zero(nfeatures_)),
      loss_(loss) {
  }

  virtual ~Solver() {
  }

  // delete copy constructors
  Solver(const Solver&) = delete;
  Solver& operator=(const Solver&) = delete;
  Solver(Solver&&) = default;
  Solver& operator=(Solver&&) = default;

  size_t nfeatures() const {
    return nfeatures_;
  }

  const Vector& w() const {
    return w_;
  }

  Double* wdata() {
    return w_.data();
  }

  void predict(const size_t dataSize,
               Double* const outPreds,
               const Double* const XData) const {
    const MatrixMap Xmap(XData, dataSize, nfeatures_);
    Eigen::Map<Vector> preds(outPreds, dataSize);
    preds = Xmap * w_;
  }

  template <typename SolverT>
  static void iterateBlock(SolverT& solver,
                           const size_t blockSize,
                           const Double* const XData,
                           const Double* const yData,
                           const int64_t* const idxData) {
    const MatrixMap Xmap(XData, blockSize, solver.nfeatures());
    for (size_t i = 0; i < blockSize; ++i) {
      solver.iterate(Xmap.row(i), yData[i], idxData[i]);
    }
  }

  template <typename SolverT>
  static void iterateBlockIndexed(SolverT& solver,
                                  const size_t dataSize,
                                  const Double* const XData,
                                  const Double* const yData,
                                  const size_t blockSize,
                                  const int64_t* const idxData) {
    const MatrixMap Xmap(XData, dataSize, solver.nfeatures());
    for (size_t i = 0; i < blockSize; ++i) {
      solver.iterate(Xmap.row(idxData[i]), yData[idxData[i]], idxData[i]);
    }
  }

 private:
  const size_t nfeatures_;

 protected:
  Vector w_;

  const std::string loss_;
};

template <typename SolverT>
class OneVsRest {
 public:
  template <typename... Args>
  OneVsRest(const size_t nclasses, Args&&... args) : nclasses_(nclasses) {
    solvers_.reserve(nclasses_);
    for (size_t i = 0; i < nclasses_; ++i) {
      solvers_.emplace_back(std::forward<Args>(args)...);
    }
  }

  size_t nclasses() const {
    return nclasses_;
  }

  void startDecay() {
    for (auto& solver : solvers_) {
      solver.startDecay();
    }
  }

  void decay(const Double multiplier = 0.5) {
    for (auto& solver : solvers_) {
      solver.decay(multiplier);
    }
  }

  void iterateBlock(const size_t blockSize,
                    const Double* const XData,
                    const int32_t* const yData,
                    const int64_t* const idxData) {
    const MatrixMap Xmap(XData, blockSize, solvers_.front().nfeatures());
#pragma omp parallel for
    for (size_t c = 0; c < nclasses_; ++c) {
      for (size_t i = 0; i < blockSize; ++i) {
        solvers_[c].iterate(
            Xmap.row(i),
            static_cast<Double>(yData[i] == static_cast<int32_t>(c)),
            idxData[i]);
      }
    }
  }

  void iterateBlockIndexed(const size_t dataSize,
                           const Double* const XData,
                           const int32_t* const yData,
                           const size_t blockSize,
                           const int64_t* const idxData) {
    const MatrixMap Xmap(XData, dataSize, solvers_.front().nfeatures());
#pragma omp parallel for
    for (size_t c = 0; c < nclasses_; ++c) {
      for (size_t i = 0; i < blockSize; ++i) {
        solvers_[c].iterate(
            Xmap.row(idxData[i]),
            static_cast<Double>(yData[idxData[i]] == static_cast<int32_t>(c)),
            idxData[i]);
      }
    }
  }

  void predict(const size_t dataSize,
               int32_t* const out,
               const Double* const XData) const {
    Matrix preds(nclasses_, dataSize);
#pragma omp parallel for
    for (size_t c = 0; c < nclasses_; ++c) {
      solvers_[c].predict(dataSize, preds.row(c).data(), XData);
    }

#pragma omp parallel for
    for (size_t i = 0; i < dataSize; ++i) {
      out[i] = 0;
      Double m = preds(0, i);
      for (size_t c = 1; c < nclasses_; ++c) {
        if (preds(c, i) > m) {
          m = preds(c, i);
          out[i] = c;
        }
      }
    }
  }

 private:
  const size_t nclasses_;

  std::vector<SolverT> solvers_;
};
}
