#pragma once

#include <memory>

#include "op/op.hpp"

namespace tinyinfer {
    namespace op {
        class Maxpooling2dOp : public Op {
        private:
            int m_kernel_x = 2;         // the kernel size
            int m_kernel_y = 2;         // default as 2

            int m_stride_x = 2;         // the stride size
            int m_stride_y = 2;         // default as 2

        public:
            Maxpooling2dOp(const std::shared_ptr<Node>& arg);

            void register_params(int kernel_x, int kernel_y, int stride_x, int stride_y);

            void set_kernel_x(int x);
            void set_kernel_y(int y);
            void set_stride_x(int x);
            void set_stride_y(int y);
            void virtual forward();
        };
    }
}

