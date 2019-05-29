#include "op/convolution.hpp"


namespace tinyinfer{
namespace op{
    ConvOp::ConvOp(const std::shared_ptr<tinyinfer::Node> &arg)
            : Op("Conv2d", check_args_single_output({arg})) {
        validate_and_infer();
    }

    void ConvOp::register_weight(const std::shared_ptr<tinyinfer::runtime::Tensor> &tensor) {
        m_weights = tensor;
    }

    void ConvOp::register_bias(const std::shared_ptr<tinyinfer::runtime::Tensor> &tensor) {
        m_bias = tensor;
    }

    void ConvOp::set_stride_x(int stride) {
        m_stride_x = stride;
    }

    void ConvOp::set_stride_y(int stride) {
        m_stride_y = stride;
    }

    // TODO: implement forward()
    void ConvOp::forward() {

    }
}
}