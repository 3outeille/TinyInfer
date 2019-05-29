#pragma once

#include <memory>

#include "op/op.hpp"

namespace tinyinfer {
    namespace op {
        class ConvOp : public Op {
        private:
            std::shared_ptr<runtime::Tensor> m_weights = nullptr;
            std::shared_ptr<runtime::Tensor> m_bias = nullptr;

            int m_stride_x = 1;       // default as 1
            int m_stride_y = 1;

        public:
            ConvOp(const std::shared_ptr<Node>& arg);
            void register_weight(const std::shared_ptr<runtime::Tensor>& tensor);
            void register_bias(const std::shared_ptr<runtime::Tensor>& tensor);
            void set_stride_x(int stride);
            void set_stride_y(int stride);
            void virtual forward();
        };
    }
}

