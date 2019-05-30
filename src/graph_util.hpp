#pragma once

#include "node.hpp"

namespace tinyinfer {
NodeVector topological_sort(const NodeVector& nodes);
}