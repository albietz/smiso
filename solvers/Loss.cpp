
#include "Loss.h"

namespace solvers {

std::mt19937 Loss::gen_ = std::mt19937(std::random_device()());

Double Loss::gradSigma_ = -1;
}
