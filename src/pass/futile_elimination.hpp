#pragma once

#include "node.hpp"

namespace tinyinfer {
namespace pass {
/**
 * Eliminate the futile nodes like dropout, identity, etc
 * @param nodes: the node vector which stores the graph
 */
NodeVector futile_elimination(const NodeVector& nodes);
}
}