#include "op/parameter.hpp"

using namespace tinyinfer;

op::Parameter::Parameter() : Op("Parameter", {}) { validate_and_infer(); }

void op::Parameter::register_input(const std::shared_ptr<runtime::Tensor> &tensor) {
    m_input = tensor;
}

// put the input tensor onto output pipe
void op::Parameter::forward() {
    assert(this->get_output_num() == 1);
    this->get_outputs().at(0).set_tensor_ptr(m_input);
}