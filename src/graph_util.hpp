#pragma once

#include <deque>
#include <unordered_map>

#include "node.hpp"

namespace tinyinfer {
NodeVector topological_sort(const NodeVector& nodes);
}