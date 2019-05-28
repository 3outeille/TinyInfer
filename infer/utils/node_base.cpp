//
// Created by ernest on 19-5-27.
//

#include "node_base.h"

NodeBase::NodeBase(std::string name): m_name(name) {
    // dummy initialization
}

void NodeBase::set_op(std::string op) {
    m_op = op;
}

void NodeBase::set_input_node(std::string input_node) {
    m_input_node = input_node;
}

void NodeBase::print_debug() {
    std::cout << "Node name: " << m_name << std::endl;
    std::cout << "OP: " << m_op << std::endl;
    std::cout << "input_node: " << m_input_node << std::endl;
}