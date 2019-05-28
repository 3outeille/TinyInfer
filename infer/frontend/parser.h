//
// Created by ernest on 19-5-28.
//

#ifndef LOADER_PARSER_H
#define LOADER_PARSER_H

#include <iostream>
#include <fstream>
#include <sstream>

#include "graph.pb.h"
#include "node_def.pb.h"
#include "tensor_base.h"


using namespace tensorflow;

class Parser {
public:
    Parser();


    /**
     * Do the parsing
     * @param filename: the input .pb file (from tensorflow)
     * @param weights_dir: the directory for storing kernel weights
     */
    void parse(const std::string & filename, const std::string & weights_dir);

private:
    std::map<std::string, std::shared_ptr<TensorBase>> graph_;

    // ============== Helper Functions =============
    /**
     * Initial all nodes in graph_ by its op
     * @param graph_def: The input graph from tensorflow (protobuf)
     */
    void init_nodes(const GraphDef& graph_def);

    void parse_single_node(const NodeDef& node_def);

    void parse_input(const NodeDef& node_def);

    void parse_conv(const NodeDef& node_def);

    void parse_dense(const NodeDef& node_def);

    /**
     * The name from Protobuf contains two parts node_name + "/" + "operation"
     * e.g: conv2d_1/convolution, conv2d_1/kernel, conv2d_1/bias
     * @return: the actual node name. e.g. conv2d_1
     */
    std::string parse_node_name(const std::string parse_node_name);
    // ================== Attr =====================

};


#endif //LOADER_PARSER_H
