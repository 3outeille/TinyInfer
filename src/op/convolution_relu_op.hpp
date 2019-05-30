#pragma once

#include <memory>

#include "op/op.hpp"
#include "op/convolution_op.hpp"

namespace tinyinfer {
    namespace op {
        /**
         * \brief Convolution Relu operation node
         * Implement the runtime op node for Conv2d and Relu Operation
         */
        class ConvReluOp : public Op {
            friend class ConvOp;

        protected:
            std::shared_ptr<runtime::Tensor> m_weights = nullptr;
            std::shared_ptr<runtime::Tensor> m_bias = nullptr;

            int m_stride_x = 1;       // default as 1
            int m_stride_y = 1;

        public:
            /**
             * Construct the node via a Conv2d node
             * @param arg: the Conv2d nodes
             */
            ConvReluOp(const std::shared_ptr<Node> &arg);

            /**
             * The forward operation
             */
            void virtual forward();
        };
    }
}

