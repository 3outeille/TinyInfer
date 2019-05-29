#include "op/sampleOp.hpp"
#include "runtime/kernel/addconstant.hpp"

using namespace tinyinfer;
using namespace tinyinfer::runtime::kernel;

op::AddConstant::AddConstant(const std::shared_ptr<Node>& arg)
    : Op("AddConstant", check_args_single_output({arg})) {
  validate_and_infer();
}

void op::AddConstant::register_weight(const std::shared_ptr<runtime::Tensor>& tensor) {
  m_weights = tensor;
}

void op::AddConstant::forward() {
  // forward calculating and write the result back into output tensor
  addconstant();
}
