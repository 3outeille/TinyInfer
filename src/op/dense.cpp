#include "op/dense.hpp"


namespace tinyinfer{
    namespace op{
        DenseOp::DenseOp(const std::shared_ptr<tinyinfer::Node> &arg)
                : Op("dense", check_args_single_output({arg})) {
            validate_and_infer();
        }

        void DenseOp::register_weight(const std::shared_ptr<tinyinfer::runtime::Tensor> &tensor) {
            m_weights = tensor;
        }

        void DenseOp::register_bias(const std::shared_ptr<tinyinfer::runtime::Tensor> &tensor) {
            m_bias = tensor;
        }

        // TODO: implement forward()
        void DenseOp::forward() {

        }
    }
}