#pragma once

#include <memory>

#include "op/op.hpp"

namespace tinyinfer {
    namespace op {
        /**
         * \brief Convolution Relu operation node
         * Implement the runtime op node for Conv2d and Relu Operation
         */
        class ConvReluOp : public Op {
        private:
            std::shared_ptr<runtime::Tensor> m_weights = nullptr;
            std::shared_ptr<runtime::Tensor> m_bias = nullptr;

            int m_stride_x = 1;       // default as 1
            int m_stride_y = 1;

        public:
            /**
             * Construct the node with input nodes
             * @param arg: the input nodes
             */
            ConvReluOp(const std::shared_ptr<Node>& arg);

            /**
             * Register all params
             * @param weight: the weight has to be 4-d tensor
             * @param bias: the bias has to be 1-d tensor
             * @param stride_x
             * @param stride_y
             */
            void register_params(const std::shared_ptr<runtime::Tensor>& weight,
                    const std::shared_ptr<runtime::Tensor>& bias,
                    int stride_x,
                    int stride_y);

            /**
             * Register weight
             * @param tensor: the weight has to be 4-d tensor
             */
            void register_weight(const std::shared_ptr<runtime::Tensor>& tensor);

            /**
             * Register bias
             * @param tensor: the bias has to be 1-d tensor
             */
            void register_bias(const std::shared_ptr<runtime::Tensor>& tensor);
            void set_stride_x(int stride);
            void set_stride_y(int stride);

            /**
             * The forward operation
             */
            void virtual forward();
        };
    }
}

