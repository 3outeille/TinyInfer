#include "op/flatten_op.hpp"
#include "runtime/kernel/flatten.hpp"

namespace tinyinfer {
    namespace op {
        FlattenOp::FlattenOp(const std::shared_ptr<tinyinfer::Node> &arg)
                : Op("flatten", check_args_single_output({arg})) {
            validate_and_infer();
        }

        void FlattenOp::forward() {
            this->get_outputs().at(0).set_tensor_ptr(
                    runtime::kernel::Flatten(this->get_inputs().at(0).get_tensor_ptr()));
        }
    }
}