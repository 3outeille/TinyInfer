#include "op/softmax.hpp"


namespace tinyinfer{
    namespace op{
        SoftmaxOp::SoftmaxOp(const std::shared_ptr<tinyinfer::Node> &arg)
                : Op("softmax", check_args_single_output({arg})) {
            validate_and_infer();
        }

        // TODO: implement forward()
        void SoftmaxOp::forward() {

        }
    }
}