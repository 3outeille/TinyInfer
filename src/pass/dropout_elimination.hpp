#pragma once

#include "node.hpp"

namespace tinyinfer {
namespace pass {
NodeVector dropout_elimination(const NodeVector& nodes);
}
}