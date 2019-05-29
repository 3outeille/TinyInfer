#include "node.hpp"

using namespace tinyinfer;

std::atomic<size_t> Node::m_next_instance_id(0);

Node::Node(const std::string& node_type)
    : m_node_type(node_type),
      m_instance_id(m_next_instance_id.fetch_add(1)),
      m_unique_name(get_description() + "_" + std::to_string(m_instance_id)) {}

Node::~Node() {
  for (auto& input : m_inputs) {
    input.get_output().remove_input(&input);
  }
}

NodeVector Node::get_arguments() const {
  NodeVector result;
  for (auto& i : m_inputs) {
    result.push_back(i.get_output().get_node());
  }
}

NodeVector Node::get_users() const {
  NodeVector result;
  for (size_t i = 0; i < get_output_num(); i++) {
    for (auto input : m_outputs.at(i).get_inputs()) {
      result.push_back(input->get_node());
    }
  }
}