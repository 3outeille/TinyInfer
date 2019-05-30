#include "output.hpp"
#include "node.hpp"

using namespace tinyinfer;

Output::Output(Node* node, size_t index) : m_node(node), m_index(index) {}

std::shared_ptr<Node> Output::get_node() const {
  return m_node->shared_from_this();
}
void Output::set_tensor_ptr(const std::shared_ptr<runtime::Tensor>& tensor) {
  m_tensor = tensor;
}
void Output::add_input(Input* input) { m_inputs.insert(input); }

void Output::remove_input(Input* input) {
  // crash and tell us if the case happen
  assert(m_inputs.count(input) != 0);
  m_inputs.erase(input);
}

const Shape& Output::get_shape() const { return m_tensor->get_shape(); }