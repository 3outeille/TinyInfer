#include <iostream>
#include <fstream>
#include <string>
#include <streambuf>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include "graph.pb.h"
#include "node_def.pb.h"
#include "attr_value.pb.h"

using namespace tensorflow;
using nlohmann::json;

Eigen::Tensor<float, 4> load_kernal_weight(std::string weight_file){
    // decode json file
    std::string encoded_json;
    std::ifstream input(weight_file);
    std::stringstream buffer;
    buffer << input.rdbuf();

    encoded_json = buffer.str();

    auto json_node = json::parse(encoded_json);
//    std::cout << json_node["name"] << std::endl;
//    std::cout << json_node["shape"] << std::endl;
//    std::cout << json_node["tensor_content"] << std::endl;
    auto tensor_shape = json_node["shape"];
    auto tensor_content = json_node["tensor_content"];

    // check tensor dimension
    if (tensor_shape.size() != 4){
        std::cerr << "[ERROR] Dimension Mismatch! Expected: 4  Actual: " <<  tensor_shape.size() << std::endl;
        throw std::runtime_error("[ERROR] Dimension Mismatch! Expected: 4");
    }

    // Loading tensor
    Eigen::Tensor<float, 4> result(tensor_shape[0],tensor_shape[1],tensor_shape[2],tensor_shape[3]);
    result.setZero();
    size_t counter = 0;
    for (int i = 0; i < tensor_shape[0]; i++){
        for (int j = 0; j < tensor_shape[1]; j++){
            for (int k = 0; k < tensor_shape[2]; k++){
                for (int w = 0; w < tensor_shape[3]; w++){
                    result(i,j,k,w) = tensor_content[counter];
                    counter ++;
                }
            }
        }
    }

    std::cout << result(0,0,0,0) << std::endl;
    std::cout << result(0,0,0,1) << std::endl;
    std::cout << result(0,0,0,2) << std::endl;
    std::cout << result(0,0,0,3) << std::endl;
    std::cout << result(0,0,0,4) << std::endl;



    return result;
}

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