//
// Created by ernest on 19-5-28.
//

#include "parser.h"

Parser::Parser() {}

void Parser::parse(const std::string & filename, const std::string & weights_dir) {
    std::ifstream input(filename);

    tensorflow::GraphDef graph_def;
    if (graph_def.ParseFromIstream(&input)){
        std::cout << "Read Success" << std::endl;

        // go through the graph (list of nodes)
//        for (size_t i = 0; i < graph_def.node_size(); i++){
//            const tensorflow::NodeDef & node = graph_def.node(i);
//
//
//        }
        // initialize all node with node op and names
        init_nodes(graph_def);
        build_dependency(graph_def);


        std::cout << "End of parsing iter 1" << std::endl;
        std::cout << " ============================================= " << std::endl;
        std::cout << " ============================================= " << std::endl;

        // For test only
        // Print out node info for validtion
        for (auto p : graph_){
            p.second->print_debug();
            std::cout << "----------------------------" << std::endl;
        }
    }
    else{
        std::cerr << "[ERROR] can not read from: " << filename << std::endl;
    }
}

void Parser::init_nodes(const GraphDef &graph_def) {
    for (size_t i = 0; i < graph_def.node_size(); i++){
        const tensorflow::NodeDef & node = graph_def.node(i);

        // filter out training-only node
        if ((node.name().find("training") != std::string::npos) ||
            ((node.name().find("Adam") != std::string::npos))){
            continue;
        }

        // Note: node_name is guaranteed to be unique for each layer
        const std::string& node_op = node.op();
        if (node_op == "Conv2D"){           // conv 2d
            std::string node_name = parse_node_name(node.name())[0];
            auto new_node = std::make_shared<NodeBase>(node_name);
            new_node->set_op("conv2d");
            graph_.insert(std::make_pair(node_name, new_node));
            std::cout << "Node name: " << node_name << "  " << "Node Type: conv2d" << std::endl;
        }
        else if (node_op == "MatMul"){      // dense
            std::string node_name = parse_node_name(node.name())[0];
            auto new_node = std::make_shared<NodeBase>(node_name);
            new_node->set_op("dense");
            graph_.insert(std::make_pair(node_name, new_node));
            std::cout << "Node name: " << node_name << "  " << "Node Type: dense" << std::endl;
        }
        else if (node_op == "Reshape"){     // flatten-like operations
            std::string node_name = parse_node_name(node.name())[0];
            auto new_node = std::make_shared<NodeBase>(node_name);
            new_node->set_op("flatten");
            graph_.insert(std::make_pair(node_name, new_node));
            std::cout << "Node name: " << node_name << "  " << "Node Type: flatten" << std::endl;
        }
        else if (node_op == "MaxPool"){     // maxPooling
            std::string node_name = parse_node_name(node.name())[0];
            auto new_node = std::make_shared<NodeBase>(node_name);
            new_node->set_op("maxpooling");
            graph_.insert(std::make_pair(node_name, new_node));
            std::cout << "Node name: " << node_name << "  " << "Node Type: maxpooling" << std::endl;
        }
        // TODO dropout
        // TODO activation functions
    }
}

void Parser::build_dependency(const GraphDef &graph_def) {
    for (size_t i = 0; i < graph_def.node_size(); i++){
        const tensorflow::NodeDef & node = graph_def.node(i);

        // filter out training-only node
        if ((node.name().find("training") != std::string::npos) ||
            ((node.name().find("Adam") != std::string::npos))){
            continue;
        }

        // Note: node_name is guaranteed to be unique for each layer
        const std::string& node_op = node.op();
        if (node_op == "Conv2D"){
            parse_dep(node);
        }
        else if (node_op == "MatMul"){      // dense
            parse_dep(node);
        }
        else if (node_op == "Reshape"){     // flatten-like operations
            parse_dep(node);
        }
        else if (node_op == "MaxPool"){     // maxPooling
            parse_dep(node);
        }
    }
}

void Parser::parse_dep(const NodeDef &node_def) {
    std::string node_name = parse_node_name(node_def.name())[0];
    for (int i = 0; i < node_def.input_size(); i++){
        auto tmp = parse_node_name(node_def.input(i));
        if (tmp.size() == 1){
            // this is the input [the place holder]
            graph_[node_name]->set_input_node(tmp[0]);
        }
        else{
            if (tmp[0] != node_name){
                // this is the input
                graph_[node_name]->set_input_node(tmp[0]);
            }
        }
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