#pragma once

#include <memory>

#include "op/op.hpp"

namespace tinyinfer {
    namespace op {
        /**
         * \brief Dense operation node
         * Implement the runtime op node for dense (fully connect layer) Operation
         */
        class DenseOp : public Op {
        private:
            std::shared_ptr<runtime::Tensor> m_weights = nullptr;
            std::shared_ptr<runtime::Tensor> m_bias = nullptr;


        public:
            /**
             * Construct the node with input nodes
             * @param arg: the input nodes
             */
            DenseOp(const std::shared_ptr<Node> &arg);

            /**
             * Register all params
             * @param weights: the weight has to be 2-d tensor
             * @param bias: the bias has to be 1-d tensor
             */
            void register_params(const std::shared_ptr<runtime::Tensor> &weights,
                                 const std::shared_ptr<runtime::Tensor> &bias);

            /**
             * Register weight
             * @param tensor: the weight has to be 2-d tensor
             */
            void register_weight(const std::shared_ptr<runtime::Tensor> &tensor);

            /**
             * Register bias
             * @param tensor: the bias has to be 1-d tensor
             */
            void register_bias(const std::shared_ptr<runtime::Tensor> &tensor);

            /**
             * The forward operation
             */
            void virtual forward();
        };
    }
}

