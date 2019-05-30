#include <assert.h>

#include "node.hpp"

using namespace tinyinfer;

std::atomic<size_t> Node::m_next_instance_id(0);

Node::Node(const std::string &node_type, const NodeVector &arguments)
        : m_node_type(node_type),
          m_instance_id(m_next_instance_id.fetch_add(1)),
          m_unique_name(get_description() + "_" + std::to_string(m_instance_id)) {
    size_t i = 0;
    for (auto arg : arguments) {
        for (Output &output : arg->m_outputs) {
            m_inputs.emplace_back(this, i++, output);
        }
    }
    // set a default output for the node
    m_outputs.emplace_back(this, 0);
}

Node::~Node() {
    // here possible bug in some cases, commented out current
    //  for (auto& input : m_inputs) {
    //    input.get_output().remove_input(&input);
    //  }
}

std::shared_ptr<Node> Node::get_argument(size_t index) {
    return m_inputs.at(index).get_output().get_node();
}

NodeVector Node::get_arguments() {
    NodeVector result;
    for (auto &i : m_inputs) {
        result.push_back(i.get_output().get_node());
    }

    return result;
}

NodeVector Node::get_users() const {
    NodeVector result;
    for (size_t i = 0; i < get_output_num(); i++) {
        for (auto input : m_outputs.at(i).get_inputs()) {
            result.push_back(input->get_node());
        }
    }

    return result;
}

const NodeVector &Node::check_args_single_output(const NodeVector &args) {
    for (auto arg : args) {
        assert(arg->get_output_num() == 1);
    }
    return args;
}

const std::string& Node::get_description() const {
    return m_node_type;
}

size_t Node::get_instance_id() const {
    return m_instance_id;
}

const std::string& Node::get_name() const {
    return m_unique_name;
}

size_t Node::get_input_num() const {
    return m_inputs.size();
}

size_t Node::get_output_num() const {
    return m_outputs.size();
}

std::deque<Input>& Node::get_inputs() {
    return m_inputs;
}

std::deque<Output>& Node::get_outputs() {
    return m_outputs;
}
