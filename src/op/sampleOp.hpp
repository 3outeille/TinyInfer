#pragma once

#include <memory>

#include "op/op.hpp"

namespace tinyinfer {
namespace op {
class AddConstant : public Op {
 private:
  std::shared_ptr<runtime::Tensor> m_weights = nullptr;

 public:
  AddConstant(const std::shared_ptr<Node>& arg);
  void register_weight(const std::shared_ptr<runtime::Tensor>& tensor);
  void virtual forward();
};
}
}