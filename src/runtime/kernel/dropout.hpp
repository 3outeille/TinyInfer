#pragma once

#include <iostream>
#include <math.h>
#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

#include "runtime/tensor.hpp"

namespace tinyinfer {
namespace runtime {
namespace kernel {

typedef typename Eigen::Tensor<TENSOR_DATA_TYPE, 1, Eigen::RowMajor> Tensor1f;
typedef typename Eigen::Tensor<TENSOR_DATA_TYPE, 2, Eigen::RowMajor> Tensor2f;
typedef typename Eigen::Tensor<TENSOR_DATA_TYPE, 3, Eigen::RowMajor> Tensor3f;
typedef typename Eigen::Tensor<TENSOR_DATA_TYPE, 4, Eigen::RowMajor> Tensor4f;

/**
 * @brief drop out layer
 * @param _input
 */
std::shared_ptr<runtime::Tensor> Dropout(std::shared_ptr<runtime::Tensor> input) {
    if (input->get_rank() == 2)
        return std::make_shared<runtime::Tensor>(input->get_tensor_r2_ptr());
    else if (input->get_rank() == 3)
        return std::make_shared<runtime::Tensor>(input->get_tensor_r3_ptr());
    else if (input->get_rank() == 4)
        return std::make_shared<runtime::Tensor>(input->get_tensor_r4_ptr());
}
}
}
}
