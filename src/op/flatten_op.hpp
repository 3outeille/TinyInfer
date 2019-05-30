#pragma once

#include <memory>

#include "op/op.hpp"

namespace tinyinfer {
    namespace op {
        /**
         * \brief Flatten operation node
         * Implement the runtime op node for flatten operation
         */
        class FlattenOp : public Op {
        private:

        public:
            /**
             * Construct the node with input nodes
             * @param arg: the input nodes
             */
            FlattenOp(const std::shared_ptr<Node> &arg);

            /**
             * The forword operation
             */
            void virtual forward();
        };
    }
}

