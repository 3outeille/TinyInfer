#pragma once

#include "op/op.hpp"

namespace tinyinfer {
    class Function;
    namespace op {
/**
*\brief Parameter node: offer the input tensor to function
*/
        class Parameter : public op::Op {
        private:
            std::shared_ptr<runtime::Tensor> m_input = nullptr;

        public:
/**
 * Construct the node without input nodes
 */
            Parameter();

            /**
             * Register tensor to input
             * @param tensor: a shared pointer to runtime::Tensor object
             */
            void register_input(const std::shared_ptr<runtime::Tensor> &tensor);

            /**
             * The forward operation
             */
            void virtual forward();
        };
    }
    using ParameterVector = std::vector<std::shared_ptr<op::Parameter>>;
}