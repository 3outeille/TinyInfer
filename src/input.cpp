#include "input.hpp"
#include "output.hpp"
#include "node.hpp"

using namespace tinyinfer;

Input::Input(Node* node, size_t index, Output& output)
    : m_node(node), m_index(index), m_output(&output) {
  output.add_input(this);
}

std::shared_ptr<Node> Input::get_node() const {
  return m_node->shared_from_this();
}

void Input::set_output(Output& output) { m_output = &output; }

std::shared_ptr<runtime::Tensor> Input::get_tensor_ptr() {
  return m_output->get_tensor_ptr();
}
const Shape& Input::get_shape() const { return m_output->get_shape(); }
