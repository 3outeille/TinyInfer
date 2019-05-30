#include "parser_helper.hpp"

namespace tinyinfer {

    namespace parser_helper {

        std::vector<int> get_ksize(const tensorflow::NodeDef &node_def) {
            auto node_attr = node_def.attr();
            auto ksize = node_attr.operator[]("ksize");
            int kernel_x = ksize.list().i(1);
            int kernel_y = ksize.list().i(2);
            return {kernel_x, kernel_y};
        }

        std::vector<int> get_stride(const tensorflow::NodeDef &node_def) {
            auto node_attr = node_def.attr();
            auto strides = node_attr.operator[]("strides");
            int stride_x = strides.list().i(1);
            int stride_y = strides.list().i(2);
            return {stride_x, stride_y};
        }

        std::vector<std::string> parse_node_name(const std::string &parse_node_name) {
            std::vector<std::string> results;
            std::string delimiter = "/";
            std::string input = parse_node_name;

            size_t pos = 0;
            std::string token;

            if ((pos = input.find(delimiter)) != std::string::npos) {
                token = input.substr(0, pos);
                results.push_back(token);
                input.erase(0, pos + delimiter.length());
            }

            results.push_back(input);

            return results;
        }

        std::vector<std::string> parse_node_inputs(const NodeDef &node_def) {
            std::vector<std::string> results;

            std::string node_name = parse_node_name(node_def.name())[0];

            // if input is not self-contain (has the same name), it is an operation
            for (int i = 0; i < node_def.input_size(); i++) {
                auto tmp = parse_node_name(node_def.input(i));
                if (tmp.size() == 1) {
                    // this is the input [the place holder]
                    results.push_back(tmp[0]);
                } else {
                    if (tmp[0] != node_name) {
                        // this is the input
                        results.push_back(tmp[0]);
                    }
                }
            }

            return results;
        }

        std::vector<std::string> parse_activation_inputs(const NodeDef &node_def) {
            std::vector<std::string> results;

            std::string node_name = parse_node_name(node_def.name())[0];

            // if input is not self-contain (has the same name), it is an operation
            for (int i = 0; i < node_def.input_size(); i++) {
                auto tmp = parse_node_name(node_def.input(i));
                results.push_back(tmp[0]);
            }

            return results;
        }

    }

}