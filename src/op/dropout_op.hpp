#pragma once

#include <memory>

#include "op/op.hpp"

namespace tinyinfer {
    namespace op {
        /**
         * \brief Dropout operation node
         * Implement the runtime op node for Dropout Operation
         * For inference, it is an identity mapping
         */
        class DropoutOp : public Op {
        private:

        public:
            /**
             * Construct the node with input nodes
             * @param arg: the input nodes
             */
            DropoutOp(const std::shared_ptr<Node>& arg);

            /**
             * The forward operation
             * For runtime inference, it is simply an identity mapping
             */
            void virtual forward();
        };
    }
}

