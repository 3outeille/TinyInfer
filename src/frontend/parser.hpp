//
// Created by ernest on 19-5-29.
//

#ifndef TINYINFER_PARSER_H
#define TINYINFER_PARSER_H

#include "node.hpp"

#include <iostream>
#include <fstream>
#include <map>

#include "graph.pb.h"
#include "node_def.pb.h"
#include "tinyinfer.hpp"
#include "io.hpp"
#include "runtime/tensor.hpp"


namespace tinyinfer {

    using namespace tensorflow;

    class Parser {
    public:
        Parser();
        ~Parser();

        /**
         * Do the parsing
         * @param filename: the input .pb file (from tensorflow)
         * @param weights_dir: the directory for storing kernel weights
         */
        std::vector<std::shared_ptr<Node>> parse(const std::string &filename, const std::string &weights_dir);

        /**
         * Get input node
         */
        std::shared_ptr<op::Parameter> get_input();

    private:
        // ============== Helper Functions =============
        /**
         * Parse all nodes in the graph
         * @param graph_def
         */
        void parse_nodes(const GraphDef& graph_def, const std::string& weights_dir);

        // ========================= Various Node Parsing =========================
        void parse_conv2d(const NodeDef& node_def, const std::string& weights_dir);
        void parse_dense(const NodeDef& node_def, const std::string& weights_dir);
        void parse_maxpooling2d(const NodeDef& node_def, const std::string& weights_dir);
        void parse_relu(const NodeDef& node_def, const std::string& weights_dir);
        void parse_softmax(const NodeDef& node_def, const std::string& weights_dir);
        void parse_dropout(const NodeDef& node_def, const std::string& weights_dir);
        void parse_flatten(const NodeDef& node_def, const std::string& weights_dir);

        // ======================= General Helper Functions =======================
        /**
         * Parse the node input
         * @param node_def
         * @return a list of inputs
         */
        std::vector<std::string> parse_node_inputs(const NodeDef &node_def);
        std::vector<std::string> parse_activation_inputs(const NodeDef &node_def);

        /**
         * The name from Protobuf contains two parts node_name + "/" + "operation"
         * e.g: conv2d_1/convolution, conv2d_1/kernel, conv2d_1/bias
         * @return: vector of node attributions {node_name, operation}
         */
        std::vector<std::string> parse_node_name(const std::string& parse_node_name);

        std::shared_ptr<Node> get_input_node(const std::string& node_name);

        std::vector<int> get_ksize(const NodeDef & node_def);
        std::vector<int> get_stride(const NodeDef & node_def);
        // ================== Attr =====================
        // the input tensor
        std::string m_input_name;
        std::shared_ptr<Node> m_input_node;
        std::shared_ptr<op::Parameter> m_input_node_param;
        // a mapping from node_name to the actual node
        std::map<std::string, std::shared_ptr<Node>> m_nodes;
        // each node should have one activation nodes
        std::map<std::string, std::shared_ptr<Node>> m_activations;

        // the result buffer
        std::vector<std::shared_ptr<Node>> m_results;
    };

}

#endif //TINYINFER_PARSER_H
