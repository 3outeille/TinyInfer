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
 * @brief drop out layer
 * @param _input
 */
Tensor2f Dropout(Tensor2f _input) {
    return _input;
}

Tensor3f Dropout(Tensor3f _input) {
    return _input;
}

Tensor4f Dropout(Tensor4f _input) {
    return _input;
}
}
}
}
