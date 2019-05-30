#pragma once

#include "node.hpp"

namespace tinyinfer {
    namespace pass {
/**
 * Fuse the Conv2d kernel with relu kernel
 * @param nodes: the node vector which stores the graph
 */
        NodeVector convolve_relu_fusion(const NodeVector &nodes);
    }
}