#include "op/softmax.hpp"
#include "runtime/kernel/softmax.hpp"

namespace tinyinfer{
    namespace op{
        SoftmaxOp::SoftmaxOp(const std::shared_ptr<tinyinfer::Node> &arg)
                : Op("softmax", check_args_single_output({arg})) {
            validate_and_infer();
        }

        void SoftmaxOp::forward() {
            this->get_outputs().at(0).set_tensor_ptr(
                    runtime::kernel::Softmax(this->get_inputs().at(0).get_tensor_ptr()));
        }
    }
}