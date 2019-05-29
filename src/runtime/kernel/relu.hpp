#pragma once

#include <iostream>
#include <math.h>
#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

#include "runtime/tensor.hpp"

typedef typename Eigen::Tensor<float, 1> Tensor1f;
typedef typename Eigen::Tensor<float, 2> Tensor2f;
typedef typename Eigen::Tensor<float, 3> Tensor3f;
typedef typename Eigen::Tensor<float, 4> Tensor4f;

namespace tinyinfer {
namespace runtime {
namespace kernel {

/**
 * @brief relu layer
 * @param _input
 */
Tensor2f Relu(Tensor2f _input) {
    return _input.cwiseMax(static_cast<float>(0));
}

Tensor3f Relu(Tensor3f _input) {
    return _input.cwiseMax(static_cast<float>(0));
}

Tensor4f Relu(Tensor4f _input) {
    return _input.cwiseMax(static_cast<float>(0));
}
}
}
}
