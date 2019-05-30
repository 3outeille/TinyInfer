#pragma once

#include <memory>

#include "op/op.hpp"

namespace tinyinfer {
    namespace op {
        /**
         * \brief Relu operation node
         * Implement the runtime op node for Relu activation function
         */
        class ReluOp : public Op {
        public:
            ReluOp(const std::shared_ptr<Node> &arg);

            void virtual forward();
        };
    }
}

