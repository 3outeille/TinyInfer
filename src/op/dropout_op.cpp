#include "op/dropout_op.hpp"
#include "runtime/kernel/dropout.hpp"

namespace tinyinfer{
    namespace op{
        DropoutOp::DropoutOp(const std::shared_ptr<tinyinfer::Node> &arg)
                : Op("Dropout", check_args_single_output({arg})) {
            validate_and_infer();
        }

        void DropoutOp::forward(){
            this->get_outputs().at(0).set_tensor_ptr(
                    runtime::kernel::Dropout(this->get_inputs().at(0).get_tensor_ptr()));
        }
    }
}