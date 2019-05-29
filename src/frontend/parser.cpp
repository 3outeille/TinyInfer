//
// Created by ernest on 19-5-29.
//

#include <tinyinfer.hpp>
#include <dropout.hpp>
#include <flatten.hpp>
#include "parser.h"

namespace tinyinfer {

    Parser::Parser(): m_nodes(), m_activations(), m_input_name(), m_input_node() {

    }

    void Parser::parse(const std::string &filename, const std::string &weights_dir) {
        std::ifstream input(filename);

        tensorflow::GraphDef graph_def;
        if (graph_def.ParseFromIstream(&input)){
//            std::cout << "Read Success" << std::endl;

            // go through the graph (list of nodes)
//        for (size_t i = 0; i < graph_def.node_size(); i++){
//            const tensorflow::NodeDef & node = graph_def.node(i);
//
//
//        }
            // initialize all node with node op and names
            parse_nodes(graph_def);


//            std::cout << "End of parsing iter 1" << std::endl;
//            std::cout << " ============================================= " << std::endl;
//            std::cout << " ============================================= " << std::endl;

            // For test only
            // Print out node info for validtion
//            for (auto p : m_nodes){
//                p.second->print_debug();
//                std::cout << "----------------------------" << std::endl;
//            }
            auto debug = true;
        }
        else{
            std::cerr << "[ERROR] can not read from: " << filename << std::endl;
        }
    }

    void Parser::parse_nodes(const tensorflow::GraphDef &graph_def) {
        for (size_t i = 0; i < graph_def.node_size(); i++){
            const tensorflow::NodeDef & node = graph_def.node(i);

            // filter out training-only node
            if ((node.name().find("training") != std::string::npos) ||
                ((node.name().find("Adam") != std::string::npos))){
                continue;
            }

            // Note: node_name is guaranteed to be unique for each layer
            const std::string& node_op = node.op();
            if (node_op == "Placeholder"){          // input (placeholder)
                std::string node_name = parse_node_name(node.name())[0];
                m_input_name = node_name;
                m_input_node = std::make_shared<op::Parameter>();
//                m_nodes.insert(std::make_pair(node_name, new_node));
            }
            else if (node_op == "Relu"){            // relu
                parse_relu(node);
            }
            else if (node_op == "Conv2D"){          // conv 2d
                parse_conv2d(node);
            }
            else if (node_op == "Dense"){
                parse_dense(node);
            }
            else if (node_op == "MaxPool"){
                parse_maxpooling2d(node);
            }
            else if ((node_op == "Switch") && (node.name().find("cond/mul/Switch") != std::string::npos)){
                // Dropout
                parse_dropout(node);
            }
            else if (node_op == "Reshape"){
                parse_flatten(node);
            }
        }
    }


    void Parser::parse_conv2d(const tensorflow::NodeDef &node_def) {
        std::string node_name = parse_node_name(node_def.name())[0];
        auto inputs = parse_node_inputs(node_def);

        if (inputs.size() != 1){
            // assert one input
            throw std::runtime_error("Conv2d only accept one input!");
        }

        // get input node
        std::shared_ptr<Node> input_node = get_input_node(inputs[0]);

        // create node and set attrs
        auto new_node = std::make_shared<op::ReluOp>(input_node);
        m_nodes.insert(std::make_pair(node_name, new_node));

        // TODO: set attr
    }

    void Parser::parse_relu(const tensorflow::NodeDef &node_def) {
        // make a new relu node
        std::string node_name = parse_node_name(node_def.name())[0];
        auto inputs = parse_activation_inputs(node_def);

        if (inputs.size() != 1){
            // assert one input
            throw std::runtime_error("Relu only accept one input!");
        }

        auto new_node = std::make_shared<op::ReluOp>(m_nodes[inputs[0]]);
        m_activations.insert(std::make_pair(node_name, new_node));
    }

    void Parser::parse_dense(const tensorflow::NodeDef &node_def) {
        std::string node_name = parse_node_name(node_def.name())[0];
        auto inputs = parse_node_inputs(node_def);

        if (inputs.size() != 1){
            // assert one input
            throw std::runtime_error("Dense only accept one input!");
        }

        // get input node
        std::shared_ptr<Node> input_node = get_input_node(inputs[0]);

        // create node and set attrs
        auto new_node = std::make_shared<op::ReluOp>(input_node);
        m_nodes.insert(std::make_pair(node_name, new_node));

        // TODO: set attr
    }

    void Parser::parse_maxpooling2d(const tensorflow::NodeDef &node_def) {
        std::string node_name = parse_node_name(node_def.name())[0];
        auto inputs = parse_node_inputs(node_def);

        if (inputs.size() != 1){
            // assert one input
            throw std::runtime_error("Dense only accept one input!");
        }

        // get input node
        std::shared_ptr<Node> input_node = get_input_node(inputs[0]);

        // create node and set attrs
        auto new_node = std::make_shared<op::Maxpooling2dOp>(input_node);
        m_nodes.insert(std::make_pair(node_name, new_node));

        // set attr
        auto node_attr = node_def.attr();
        auto ksize = node_attr.operator[]("ksize");
        int kernel_x = ksize.list().i(1);
        int kernel_y = ksize.list().i(2);
        new_node->set_kernel_x(kernel_x);
        new_node->set_kernel_x(kernel_y);

        auto strides = node_attr.operator[]("strides");
        int stride_x = strides.list().i(1);
        int stride_y = strides.list().i(2);
        new_node->set_stride_x(stride_x);
        new_node->set_stride_y(stride_y);
    }

    void Parser::parse_dropout(const tensorflow::NodeDef &node_def) {
        std::string node_name = parse_node_name(node_def.name())[0];
        auto inputs = parse_node_inputs(node_def);

        if (inputs.size() != 1){
            // assert one input
            throw std::runtime_error("Dense only accept one input!");
        }

        // get input node
        std::shared_ptr<Node> input_node = get_input_node(inputs[0]);

        // create node and set attrs
        auto new_node = std::make_shared<op::DropoutOp>(input_node);
        m_nodes.insert(std::make_pair(node_name, new_node));
    }

    void Parser::parse_flatten(const tensorflow::NodeDef &node_def) {
        std::string node_name = parse_node_name(node_def.name())[0];
        auto inputs = parse_node_inputs(node_def);

        if (inputs.size() != 1){
            // assert one input
            throw std::runtime_error("Dense only accept one input!");
        }

        // get input node
        std::shared_ptr<Node> input_node = get_input_node(inputs[0]);

        // create node and set attrs
        auto new_node = std::make_shared<op::FlattenOp>(input_node);
        m_nodes.insert(std::make_pair(node_name, new_node));
    }

    std::vector<std::string> Parser::parse_node_inputs(const NodeDef &node_def) {
        std::vector<std::string> results;

        std::string node_name = parse_node_name(node_def.name())[0];

        // if input is not self-contain (has the same name), it is an operation
        for (int i = 0; i < node_def.input_size(); i++){
            auto tmp = parse_node_name(node_def.input(i));
            if (tmp.size() == 1){
                // this is the input [the place holder]
                results.push_back(tmp[0]);
            }
            else{
                if (tmp[0] != node_name){
                    // this is the input
                    results.push_back(tmp[0]);
                }
            }
        }

        return results;
    }

    std::vector<std::string> Parser::parse_activation_inputs(const NodeDef &node_def) {
        std::vector<std::string> results;

        std::string node_name = parse_node_name(node_def.name())[0];

        // if input is not self-contain (has the same name), it is an operation
        for (int i = 0; i < node_def.input_size(); i++){
            auto tmp = parse_node_name(node_def.input(i));
            results.push_back(tmp[0]);
        }

        return results;
    }

    std::shared_ptr<Node> Parser::get_input_node(const std::string& node_name){
        // if it is the input
        if (node_name == m_input_name){
            return m_input_node;
        }

        // check if the node_name is in activation, use it if presented
        if (m_activations.count(node_name) == 1){
            return m_activations[node_name];
        }
        else{
            return m_nodes[node_name];
        }
    }

    std::vector<std::string> Parser::parse_node_name(const std::string& parse_node_name) {
//    std::stringstream stream(parse_node_name);
//    std::string word;
//    getline(stream, word, '/');
//
//    return word;
        std::vector<std::string> results;
        std::string delimiter = "/";
        std::string input = parse_node_name;

        size_t pos = 0;
        std::string token;
        // go through all delimter
//    while ((pos = input.find(delimiter)) != std::string::npos) {
//        token = input.substr(0, pos);
////        std::cout << token << std::endl;
//        results.push_back(token);
//        input.erase(0, pos + delimiter.length());
//    }
////    std::cout << input << std::endl;

        if ((pos = input.find(delimiter)) != std::string::npos) {
            token = input.substr(0, pos);
//        std::cout << token << std::endl;
            results.push_back(token);
            input.erase(0, pos + delimiter.length());
        }
//    std::cout << input << std::endl;

        results.push_back(input);

        for (auto v : results){
            std::cout << v << std::endl;
        }

        return results;
    }
}