#include "pass/convolve_relu_fusion.hpp"
#include "op/convolution_op.hpp"

using namespace tinyinfer;
using namespace tinyinfer::op;

NodeVector pass::convolve_relu_fusion(const NodeVector& nodes) {
  NodeVector result;
  size_t pos = 0;
  while (pos < nodes.size()) {
    auto node = nodes.at(pos);
    pos++;
    if (node->get_description() == "Conv2d" &&
        node->get_users().at(0)->get_description() == "Relu") {
      assert(node->get_input_num() == 1);

      auto new_node = std::make_shared<ConvReluOp>(node);

      Input* new_input =
          *(node->get_users().at(0)->get_outputs().at(0).get_inputs().begin());
      Output* new_output = &node->get_inputs().at(0).get_output();
      Output* own_output = &new_node->get_outputs().at(0);

      new_input->set_output(*own_output);
      own_output->add_input(new_input);
      new_output->remove_input(&node->get_inputs().at(0));
      pos++;
      result.push_back(new_node);
    } else {
      result.push_back(node);
    }
  }
  return result;
}