//
// Created by ernest on 19-5-27.
//

#include "io.h"

namespace io{
    Eigen::Tensor<float, 1> load_kernel_weight_1d(std::string weight_file){
        // decode json file (read in)
        std::string encoded_json;
        std::ifstream input(weight_file);
        std::stringstream buffer;
        buffer << input.rdbuf();

        encoded_json = buffer.str();

        // decode json string
        auto json_node = json::parse(encoded_json);
        auto tensor_shape = json_node["shape"];
        auto tensor_content = json_node["tensor_content"];

        // check tensor dimension
        if (tensor_shape.size() != 4){
            std::cerr << "[ERROR] Dimension Mismatch! Expected: 4  Actual: " <<  tensor_shape.size() << std::endl;
            throw std::runtime_error("[ERROR] Dimension Mismatch! Expected: 4");
        }

        // Loading tensor
        Eigen::Tensor<float, 1> result(tensor_shape[0]);
        result.setZero();
        size_t counter = 0;
        for (int i = 0; i < tensor_shape[0]; i++){
                result(i) = tensor_content[counter];
                counter ++;
        }

        return result;
    }

    Eigen::Tensor<float, 2> load_kernel_weight_2d(std::string weight_file){
        // decode json file (read in)
        std::string encoded_json;
        std::ifstream input(weight_file);
        std::stringstream buffer;
        buffer << input.rdbuf();

        encoded_json = buffer.str();

        // decode json string
        auto json_node = json::parse(encoded_json);
        auto tensor_shape = json_node["shape"];
        auto tensor_content = json_node["tensor_content"];

        // check tensor dimension
        if (tensor_shape.size() != 4){
            std::cerr << "[ERROR] Dimension Mismatch! Expected: 4  Actual: " <<  tensor_shape.size() << std::endl;
            throw std::runtime_error("[ERROR] Dimension Mismatch! Expected: 4");
        }

        // Loading tensor
        Eigen::Tensor<float, 2> result(tensor_shape[0],tensor_shape[1]);
        result.setZero();
        size_t counter = 0;
        for (int i = 0; i < tensor_shape[0]; i++){
            for (int j = 0; j < tensor_shape[1]; j++){
                result(i,j) = tensor_content[counter];
                counter ++;
            }
        }

        return result;
    }

    Eigen::Tensor<float, 3> load_kernel_weight_3d(std::string weight_file){
        // decode json file (read in)
        std::string encoded_json;
        std::ifstream input(weight_file);
        std::stringstream buffer;
        buffer << input.rdbuf();

        encoded_json = buffer.str();

        // decode json string
        auto json_node = json::parse(encoded_json);
        auto tensor_shape = json_node["shape"];
        auto tensor_content = json_node["tensor_content"];

        // check tensor dimension
        if (tensor_shape.size() != 3){
            std::cerr << "[ERROR] Dimension Mismatch! Expected: 4  Actual: " <<  tensor_shape.size() << std::endl;
            throw std::runtime_error("[ERROR] Dimension Mismatch! Expected: 4");
        }

        // Loading tensor
        Eigen::Tensor<float, 3> result(tensor_shape[0],tensor_shape[1],tensor_shape[2]);
        result.setZero();
        size_t counter = 0;
        for (int i = 0; i < tensor_shape[0]; i++){
            for (int j = 0; j < tensor_shape[1]; j++){
                for (int k = 0; k < tensor_shape[2]; k++){
                    result(i,j,k) = tensor_content[counter];
                    counter ++;
                }
            }
        }

        return result;
    }

    Eigen::Tensor<float, 4> load_kernel_weight_4d(std::string weight_file){
        // decode json file (read in)
        std::string encoded_json;
        std::ifstream input(weight_file);
        std::stringstream buffer;
        buffer << input.rdbuf();

        encoded_json = buffer.str();

        // decode json string
        auto json_node = json::parse(encoded_json);
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

        return result;
    }


}