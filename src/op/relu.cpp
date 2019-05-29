#include "op/relu.hpp"

namespace tinyinfer {
    namespace op {
        ReluOp::ReluOp(const std::shared_ptr<tinyinfer::Node> &arg)
                : Op("Relu", check_args_single_output({arg})) {
            validate_and_infer();
        }

        // TODO: implement forward
        void ReluOp::forward() {

        }
    }
}

