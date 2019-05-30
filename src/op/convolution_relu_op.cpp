#include "op/convolution_relu_op.hpp"
#include "runtime/kernel/convolution_relu.hpp"

namespace tinyinfer {
namespace op {
ConvReluOp::ConvReluOp(const std::shared_ptr<tinyinfer::Node> &arg)
    : Op("ConvRelu", check_args_single_output(arg->get_arguments())) {
  assert(arg->get_description() == "Conv2d");
  const std::shared_ptr<ConvOp> &ptr = (const std::shared_ptr<ConvOp> &)arg;
  m_weights = ptr->m_weights;
  m_bias = ptr->m_bias;
  m_stride_x = ptr->m_stride_x;
  m_stride_y = ptr->m_stride_y;
  validate_and_infer();
}

void ConvReluOp::forward() {
  this->get_outputs().at(0).set_tensor_ptr(runtime::kernel::Conv_Relu(
      this->m_weights, this->m_bias, this->get_inputs().at(0).get_tensor_ptr(),
      this->m_stride_x, this->m_stride_y));
}
}
}