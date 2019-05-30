#pragma once

#include <memory>

#include "op/op.hpp"

namespace tinyinfer {
    namespace op {
        class FlattenOp : public Op {
        private:

        public:
            FlattenOp(const std::shared_ptr<Node>& arg);
            void virtual forward();
        };
    }
}

