#include "op/relu_op.hpp"
#include "runtime/kernel/relu.hpp"

namespace tinyinfer {

    namespace op {
        ReluOp::ReluOp(const std::shared_ptr<tinyinfer::Node> &arg)
                : Op("Relu", check_args_single_output({arg})) {
            validate_and_infer();
        }

        void ReluOp::forward() {
            this->get_outputs().at(0).set_tensor_ptr(
                    runtime::kernel::Relu(this->get_inputs().at(0).get_tensor_ptr()));
        }
    }
}

