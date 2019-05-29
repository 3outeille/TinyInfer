#pragma once

#include <memory>

#include "op/op.hpp"

namespace tinyinfer {
    namespace op {
        class SoftmaxOp : public Op {
        public:
            SoftmaxOp(const std::shared_ptr<Node>& arg);
            void virtual forward();
        };
    }
}

