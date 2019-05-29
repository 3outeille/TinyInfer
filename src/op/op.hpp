#pragma once

#include "node.hpp"

namespace tinyinfer {
namespace op {
// root of all op nodes
// TODO: not clear the difference of roles between Op and Node
class Op : public Node {
 public:
  Op(const std::string& node_type, const NodeVector& arguments)
      : Node(node_type, arguments) {}
  // execute the exact computing and write the result into output
  void virtual forward() = 0;
  // validating correctness of the node
  virtual void validate_and_infer();
  bool is_same_op_type(const std::shared_ptr<Op>& op) const {
    return get_description() == op->get_description();
  }
};
}
}