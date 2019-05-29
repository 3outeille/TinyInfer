#include "function.hpp"

using namespace tinyinfer;

Function::Function(const NodeVector& graph, const ParameterVector& parameters)
    : m_graph(graph), m_parameters(parameters) {}

void Function::forward(const std::vector<runtime::Tensor> inputs) {
  assert(inputs.size() == m_parameters.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    m_parameters.at(i)->register_input(
        std::make_shared<runtime::Tensor>(inputs.at(i)));
  }

  for (auto node : m_graph) {
    node->forward();
  }
}