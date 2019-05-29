#include "op/dense_op.hpp"
#include "runtime/kernel/dense.hpp"

namespace tinyinfer{
    namespace op{
        DenseOp::DenseOp(const std::shared_ptr<tinyinfer::Node> &arg)
                : Op("dense", check_args_single_output({arg})) {
            validate_and_infer();
        }

        void DenseOp::register_params(const std::shared_ptr<tinyinfer::runtime::Tensor> &weights,
                                      const std::shared_ptr<tinyinfer::runtime::Tensor> &bias) {
            m_weights = weights;
            m_bias = bias;
        }

        void DenseOp::register_weight(const std::shared_ptr<tinyinfer::runtime::Tensor> &tensor) {
            m_weights = tensor;
        }

        void DenseOp::register_bias(const std::shared_ptr<tinyinfer::runtime::Tensor> &tensor) {
            m_bias = tensor;
        }

        void DenseOp::forward() {
            this->get_outputs().at(0).set_tensor_ptr(
                    runtime::kernel::Dense(this->m_weights, this->m_bias,
                                           this->get_inputs().at(0).get_tensor_ptr()));
        }
    }
}