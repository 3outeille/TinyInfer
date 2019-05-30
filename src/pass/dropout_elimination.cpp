#include "pass/dropout_elimination.hpp"

using namespace tinyinfer;

NodeVector pass::dropout_elimination(const NodeVector& nodes) {
  NodeVector result;
  for (auto node : nodes) {
    if (node->get_description() == "Dropout") {
      assert(node->get_input_num() == 1);
      assert(node->get_output_num() == 1);
      Input* new_input = *(node->get_outputs().at(0).get_inputs().begin());
      Output* new_output = &node->get_inputs().at(0).get_output();

      new_output->remove_input(&node->get_inputs().at(0));
      new_output->add_input(new_input);
      new_input->set_output(*new_output);
    } else {
      result.push_back(node);
    }
  }
  return result;
}