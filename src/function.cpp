#include "function.hpp"
#include "graph_util.hpp"

using namespace tinyinfer;

Function::Function(const NodeVector& graph, const ParameterVector& parameters, const NodeVector& targets)
    : m_graph(graph), m_parameters(parameters), m_targets(targets) {
  m_graph = topological_sort(m_graph);
}

const std::vector<runtime::Tensor> Function::forward(const std::vector<runtime::Tensor> inputs) {
  assert(inputs.size() == m_parameters.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    m_parameters.at(i)->register_input(
        std::make_shared<runtime::Tensor>(inputs.at(i)));
  }

  for (auto node : m_graph) {
    node->forward();
  }

  std::vector<runtime::Tensor> result;
  for (auto node : m_targets){
    result.push_back(*node->get_outputs().at(0).get_tensor_ptr().get());
  }

  return result;
}