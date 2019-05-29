#pragma once

#include <memory>

#include "op/op.hpp"

namespace tinyinfer {
    namespace op {
        class DenseOp : public Op {
        private:
            std::shared_ptr<runtime::Tensor> m_weights = nullptr;
            std::shared_ptr<runtime::Tensor> m_bias = nullptr;


        public:
            DenseOp(const std::shared_ptr<Node>& arg);
            void register_weight(const std::shared_ptr<runtime::Tensor>& tensor);
            void register_bias(const std::shared_ptr<runtime::Tensor>& tensor);
            void virtual forward();
        };
    }
}

