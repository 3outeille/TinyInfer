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
 * @brief flatten layer
 * @param _input
 */
Tensor2f Flatten(Tensor4f _input) {
    Tensor2f _output = Tensor2f(_input.dimension(0),
            _input.dimension(1) * _input.dimension(2) * _input.dimension(3)); /// return value

    for (auto i = 0; i < _output.dimension(0); ++i) {
        long temp = 0;

        for (auto j = 0; j < _input.dimension(1); ++j) {
            for (auto k = 0; k < _input.dimension(2); ++k) {
                for (auto l = 0; l < _input.dimension(3); ++l) {
                    _output(i, temp) = _input(i, j, k, l);
                    ++temp;
                }
            }
        }
    }

    return _output;
}
}
}
}
