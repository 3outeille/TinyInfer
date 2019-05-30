#pragma once

#include "node.hpp"

namespace tinyinfer {
namespace pass {
NodeVector convolve_relu_fusion(const NodeVector& nodes);
}
}