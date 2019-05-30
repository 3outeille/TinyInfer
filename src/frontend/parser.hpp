#pragma once

#include "node.hpp"

#include <iostream>
#include <fstream>
#include <map>

#include "graph.pb.h"
#include "node_def.pb.h"
#include "tinyinfer.hpp"
#include "io.hpp"
#include "runtime/tensor.hpp"
#include "frontend/parser_helper.hpp"


namespace tinyinfer {

    using namespace tensorflow;

    /**
     * Frontend parser for reading tensorflow model and construct the computation graph
     */
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
         * Get input node for the graph
         * @return shared pointer to the input node
         */
        std::shared_ptr<op::Parameter> get_input();

    private:
        /**
         * Parse all nodes in the input graph (protobuf)
         * @param graph_def: the input graph (protobuf)
         */
        void parse_nodes(const GraphDef& graph_def, const std::string& weights_dir);

        // ========================= Various Node Parsing =========================
        /**
         * Parse node (protobuf) for the conv2d operation and create new node in the result graph
         * @param node_def: input node (protobuf)
         * @param weights_dir: the weights directory
         */
        void parse_conv2d(const NodeDef& node_def, const std::string& weights_dir);

        /**
         * Parse node (protobuf) for the dense operation and create new node in the result graph
         * @param node_def: input node (protobuf)
         * @param weights_dir: the weights directory
         */
        void parse_dense(const NodeDef& node_def, const std::string& weights_dir);

        /**
         * Parse node (protobuf) for the maxpooling2d operation and create new node in the result graph
         * @param node_def: input node (protobuf)
         * @param weights_dir: the weights directory
         */
        void parse_maxpooling2d(const NodeDef& node_def, const std::string& weights_dir);

        /**
         * Parse node (protobuf) for the relu operation and create new node in the result graph
         * @param node_def: input node (protobuf)
         * @param weights_dir: the weights directory
         */
        void parse_relu(const NodeDef& node_def, const std::string& weights_dir);

        /**
         * Parse node (protobuf) for the softmax operation and create new node in the result graph
         * @param node_def: input node (protobuf)
         * @param weights_dir: the weights directory
         */
        void parse_softmax(const NodeDef& node_def, const std::string& weights_dir);

        /**
         * Parse node (protobuf) for the dropout operation and create new node in the result graph
         * @param node_def: input node (protobuf)
         * @param weights_dir: the weights directory
         */
        void parse_dropout(const NodeDef& node_def, const std::string& weights_dir);

        /**
         * Parse node (protobuf) for the flatten operation and create new node in the result graph
         * @param node_def: input node (protobuf)
         * @param weights_dir: the weights directory
         */
        void parse_flatten(const NodeDef& node_def, const std::string& weights_dir);

        // ======================= General Helper Functions =======================
        /**
         * Get the saved input node from the current name in the following order:
         * If the node_name is presented as input (Placeholder), return the input node
         * If the node_name is presented in m_activations, return m_activations[node_name]
         * Otherwise, return m_nodes[node_name]. (As no activation function is linked with the node)
         * @param node_name: the input node_name for lookup
         * @return the input node coresponding to the node_name
         */
        std::shared_ptr<Node> get_input_node(const std::string& node_name);

        // ================== Attr =====================
        // the input tensor
        std::string m_input_name;
        std::shared_ptr<op::Parameter> m_input_node;
        // a mapping from node_name to the actual node
        std::map<std::string, std::shared_ptr<Node>> m_nodes;
        // each node should have one activation nodes
        std::map<std::string, std::shared_ptr<Node>> m_activations;

        // the result buffer
        std::vector<std::shared_ptr<Node>> m_results;
    };

}
