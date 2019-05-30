#include "op/convolution_relu_op.hpp"
#include "runtime/kernel/convolution_relu.hpp"

namespace tinyinfer{
namespace op{
    ConvReluOp::ConvReluOp(const std::shared_ptr<tinyinfer::Node> &arg)
            : Op("Conv2d", check_args_single_output({arg})) {
        validate_and_infer();
    }

    void ConvReluOp::register_params(const std::shared_ptr<tinyinfer::runtime::Tensor> &weight,
                                 const std::shared_ptr<tinyinfer::runtime::Tensor> &bias, int stride_x, int stride_y) {
        m_weights = weight;
        m_bias = bias;
        m_stride_x = stride_x;
        m_stride_y = stride_y;
    }

    void ConvReluOp::register_weight(const std::shared_ptr<tinyinfer::runtime::Tensor> &tensor) {
        m_weights = tensor;
    }

    void ConvReluOp::register_bias(const std::shared_ptr<tinyinfer::runtime::Tensor> &tensor) {
        m_bias = tensor;
    }

    void ConvReluOp::set_stride_x(int stride) {
        m_stride_x = stride;
    }

    void ConvReluOp::set_stride_y(int stride) {
        m_stride_y = stride;
    }

    void ConvReluOp::forward() {
        this->get_outputs().at(0).set_tensor_ptr(
                runtime::kernel::Conv_Relu(this->m_weights, this->m_bias,
                                       this->get_inputs().at(0).get_tensor_ptr(),
                                       this->m_stride_x, this->m_stride_y));
    }
}
}