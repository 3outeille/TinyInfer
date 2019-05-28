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
            const std::string& node_name = parse_node_name(node.name());
            graph_.insert(std::make_pair(node_name, std::make_shared<TensorBase>(node_name)));
            std::cout << "Node name: " << node_name << "  " << "Node Type: conv2d" << std::endl;
        }
        else if (node_op == "MatMul"){      // dense
            const std::string& node_name = parse_node_name(node.name());
            graph_.insert(std::make_pair(node_name, std::make_shared<TensorBase>(node_name)));
            std::cout << "Node name: " << node_name << "  " << "Node Type: dense" << std::endl;
        }
        else if (node_op == "Reshape"){     // flatten-like operations
            const std::string& node_name = parse_node_name(node.name());
            graph_.insert(std::make_pair(node_name, std::make_shared<TensorBase>(node_name)));
            std::cout << "Node name: " << node_name << "  " << "Node Type: flatten" << std::endl;
        }
        else if (node_op == "MaxPool"){     // maxPooling
            const std::string& node_name = parse_node_name(node.name());
            graph_.insert(std::make_pair(node_name, std::make_shared<TensorBase>(node_name)));
            std::cout << "Node name: " << node_name << "  " << "Node Type: maxpooling" << std::endl;
        }
        // TODO activation functions
    }
}

void Parser::parse_conv(const NodeDef &node_def) {

}

std::string Parser::parse_node_name(const std::string parse_node_name) {
    std::stringstream stream(parse_node_name);
    std::string word;
    getline(stream, word, '/');

    return word;
}