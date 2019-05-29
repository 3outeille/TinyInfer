#include "op/flatten.hpp"


namespace tinyinfer{
    namespace op{
        FlattenOp::FlattenOp(const std::shared_ptr<tinyinfer::Node> &arg)
                : Op("flatten", check_args_single_output({arg})) {
            validate_and_infer();
        }

        // TODO: implement forward()
        void FlattenOp::forward(){

        }
    }
}