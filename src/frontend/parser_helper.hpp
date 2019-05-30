#pragma once

#include <vector>
#include "node_def.pb.h"

namespace tinyinfer {

    namespace parser_helper {

        using namespace tensorflow;

        /**
         * Get the kernal size
         * @param node_def: the input node (protobuf)
         * @return A 2-d vector {ksize_x, ksize_y}
         */
        std::vector<int> get_ksize(const tensorflow::NodeDef &node_def);

        /**
         * Get the stride size
         * @param node_def: the input node (protobuf)
         * @return A 2-d vector {stride_x, stride_y}
         */
        std::vector<int> get_stride(const tensorflow::NodeDef &node_def);

        /**
         * The name from Protobuf contains two parts node_name + "/" + "operation"
         * e.g: conv2d_1/convolution, conv2d_1/kernel, conv2d_1/bias
         * @return: vector of node attributions {node_name, operation}
         */
        std::vector<std::string> parse_node_name(const std::string &parse_node_name);

        /**
         * Parse the probobuf Nodedef for graph nodes (Like Conv2d, Flatten)
         * @param node_def: the input Node (Protobuf)
         * @return a list of input nodes' names
         */
        std::vector<std::string> parse_node_inputs(const NodeDef &node_def);

        /**
         * Parse the probobuf Nodedef for activation function nodes (Like Relu, Softmax)
         * @param node_def: the input Node (Protobuf)
         * @return a list of input nodes' names
         */
        std::vector<std::string> parse_activation_inputs(const NodeDef &node_def);
    }

}