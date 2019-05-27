//
// Created by ernest on 19-5-27.
//

#ifndef TINYINFER_IO_H
#define TINYINFER_IO_H

#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

#include "graph.pb.h"
#include "node_def.pb.h"
#include "attr_value.pb.h"

using namespace tensorflow;
using nlohmann::json;

namespace io{
    /**
     * Load 1-d tensor from the disk
     * @param weight_file: the name of the file to be loaded
     * @return Eigen::Tensor<float, 1>
     */
    Eigen::Tensor<float, 1> load_kernel_weight_1d(std::string weight_file);

    /**
     * Load 2-d tensor from the disk
     * @param weight_file: the name of the file to be loaded
     * @return Eigen::Tensor<float, 1>
     */
    Eigen::Tensor<float, 2> load_kernel_weight_2d(std::string weight_file);

    /**
     * Load 3-d tensor from the disk
     * @param weight_file: the name of the file to be loaded
     * @return Eigen::Tensor<float, 1>
     */
    Eigen::Tensor<float, 3> load_kernel_weight_3d(std::string weight_file);

    /**
     * Load 4-d tensor from the disk
     * @param weight_file: the name of the file to be loaded
     * @return Eigen::Tensor<float, 1>
     */
    Eigen::Tensor<float, 4> load_kernel_weight_4d(std::string weight_file);
}


#endif //TINYINFER_IO_H
