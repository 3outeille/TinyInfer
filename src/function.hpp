#pragma once

#include "node.hpp"
#include "op/parameter.hpp"

namespace tinyinfer {
class Function {
 protected:
  NodeVector m_graph;
  ParameterVector m_parameters;
  NodeVector m_targets;

 public:
  Function(const NodeVector& graph, const ParameterVector& parameters,
           const NodeVector& targets);
  ~Function() {}

  void optimize_graph();

  NodeVector get_graph() { return m_graph; }

  // passing by runtime::Tensor here might be the most convinient interface
  const std::vector<runtime::Tensor> forward(
      const std::vector<runtime::Tensor> inputs);
};
}