#include <iostream>
#include <fstream>
#include <string>
#include <streambuf>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include "io.h"
#include "graph.pb.h"
#include "node_def.pb.h"
#include "attr_value.pb.h"

#include "parser.h"

using namespace tensorflow;
using nlohmann::json;

int main(int argc, const char * argv[]){
    // try eigen tensors
//    std::cout << "Hello World" << std::endl;
//
//    int x = 2;
//    int y =2;
//    int z =2;
//    Eigen::Tensor<float, 3> epsilon(x,y,z);
//    epsilon.setZero();
//    epsilon(0,1,1) = 1;
//
//    return 0;
    // For testing
//    Eigen::Tensor<float, 4> tensor_test = load_kernal_weight("/home/ernest/cs133_proj/data/tensor_weights/conv2d_1_kernel.kw");
//    return 0;

    std::cout << "Hello World" << std::endl;
    std::string input_pb = "/home/ernest/cs133_proj/data/model.pb";
    std::string tensor_weights_dir = "/home/ernest/cs133_proj/data/tensor_weights";

    Parser parser;
    parser.parse(input_pb, tensor_weights_dir);


    std::cout << "End of execution" << std::endl;
    return 0;

    std::ifstream input(input_pb);

    tensorflow::GraphDef graph_def;
    if (graph_def.ParseFromIstream(&input)){
        std::cout << "Read Success" << std::endl;
        for (size_t i = 0; i < graph_def.node_size(); i++){
            const tensorflow::NodeDef & node = graph_def.node(i);

            // filter out training-only node
            if ((node.name().find("training") != std::string::npos) ||
                    ((node.name().find("Adam") != std::string::npos))){
                continue;
            }

            std::cout << "Node: " << node.name() << std::endl;

            for (auto attr_tmp : node.attr()){
                std::string attr_key = attr_tmp.first;
                const tensorflow::AttrValue& attr_value = attr_tmp.second;
                std::cout << "  Attr: " << attr_tmp.first << std::endl;
                if (attr_key == "value"){
                    std::cout << "[INFO] caught attr value" << std::endl;
                    std::string tensor_content = attr_value.tensor().tensor_content();
                    if (!tensor_content.empty()){
                        std::cout << "[INFO] caught tensor content" << std::endl;
                    }
                    else{
                        std::cout << "[WARN] no tensor content available" << std::endl;
                    }
                }
            }

//            if (i > 10){
//                break;
//            }
        }
    }
    else{
        std::cerr << "[ERROR] can not read from: " << input_pb << std::endl;
    }

    std::cout << "Parsing ended" << std::endl;

    return 0;
}