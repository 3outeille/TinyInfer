#pragma once

#include "node.hpp"
#include "op/parameter.hpp"

namespace tinyinfer {
class Function {
 protected:
  NodeVector m_graph;
  ParameterVector m_parameters;

 public:
  Function(const NodeVector& graph, const ParameterVector& parameters);
  ~Function() {}

  // passing by runtime::Tensor here might be the most convinient interface
  void forward(const std::vector<runtime::Tensor> inputs);
};
}