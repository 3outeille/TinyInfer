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
 * @brief flatten layer
 * @param _input
 * @return a tensor after the flatten layer
 */
std::shared_ptr<runtime::Tensor> Flatten(std::shared_ptr<runtime::Tensor> input) {
    Tensor4f _input = input->get_tensor_r4_ptr();
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

    return std::make_shared<runtime::Tensor>(_output);
}
}
}
}
