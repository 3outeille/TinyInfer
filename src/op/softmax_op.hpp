#pragma once

#include <memory>

#include "op/op.hpp"

namespace tinyinfer {
    namespace op {
        /**
         * \brief Softmax operation node
         * Implement the runtime op node for softmax activation function
         */
        class SoftmaxOp : public Op {
        public:
            SoftmaxOp(const std::shared_ptr<Node>& arg);
            void virtual forward();
        };
    }
}

