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
 * @brief max pooling layer
 * @param _input
 * @param window_row
 * @param window_col
 * @param stride_row
 * @param stride_col
 */
Tensor4f Maxpool(Tensor4f _input, int window_row, int window_col, int stride_row, int stride_col) {
    Tensor4f _output = Tensor4f(_input.dimension(0), ceil(_input.dimension(1) / stride_row),
                                ceil(_input.dimension(2) / stride_col), _input.dimension(3)); /// return value

    for (auto i = 0; i < _output.dimension(0); ++i) {
        for (auto l = 0; l < _output.dimension(3); ++l) {

            for (auto j = 0; j < _output.dimension(1); ++j) {
                for (auto k = 0; k < _output.dimension(2); ++k) {

                    long index_row = stride_row * j;
                    long index_col = stride_col * k;
                    float max_val = _input(i, index_row, index_col, l);

                    for (auto m = 0; m < window_row; ++m)
                        for (auto n = 0; n < window_col; ++n)
                            if (index_row + m < _input.dimension(1) && index_col + n < _input.dimension(2))
                                max_val = _input(i, index_row + m, index_col + n, l) > max_val ?
                                        _input(i, index_row + m, index_col + n, l) : max_val;

                    _output(i, j, k, l) = max_val;

                }
            }
        }
    }

    return _output;
}
}
}
}
