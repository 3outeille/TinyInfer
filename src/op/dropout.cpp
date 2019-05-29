#include "dropout.hpp"

#include "op/convolution.hpp"


namespace tinyinfer{
    namespace op{
        DropoutOp::DropoutOp(const std::shared_ptr<tinyinfer::Node> &arg)
                : Op("Dropout", check_args_single_output({arg})) {
            validate_and_infer();
        }

        // TODO: implement forward()
        void DropoutOp::forward(){

        }
    }
}