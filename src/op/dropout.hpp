#pragma once

#include <memory>

#include "op/op.hpp"

namespace tinyinfer {
    namespace op {
        class DropoutOp : public Op {
        private:

        public:
            DropoutOp(const std::shared_ptr<Node>& arg);
            void virtual forward();
        };
    }
}

