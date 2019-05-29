#pragma once

#include "op/op.hpp"

namespace tinyinfer {
class Function;
namespace op {
class Parameter : public op::Op {
 private:
  std::shared_ptr<runtime::Tensor> m_input = nullptr;

 public:
  Parameter();
  void register_input(const std::shared_ptr<runtime::Tensor>& tensor);
  void virtual forward();
};
}
}