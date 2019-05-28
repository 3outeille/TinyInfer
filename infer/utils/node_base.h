//
// Created by ernest on 19-5-27.
//

#ifndef LOADER_TENSOR_BASE_H
#define LOADER_TENSOR_BASE_H

#include <string>
#include <iostream>

class NodeBase {
public:
    NodeBase(std::string name);


    void set_op(std::string op);
    void set_input_node(std::string input_node);

    void print_debug();
private:
    std::string m_name;
    std::string m_op;
    std::string m_input_node;
//    std::string m_input_tensor;
};


#endif //LOADER_TENSOR_BASE_H
