#pragma once

#include <memory>

#include "op/op.hpp"

namespace tinyinfer {
    namespace op {
        class ReluOp : public Op {
        public:
            ReluOp(const std::shared_ptr<Node>& arg);
            void virtual forward();
        };
    }
}

