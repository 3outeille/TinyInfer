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
 * @brief convolution layer
 * @param _kernel
 * @param _bias
 * @param _input
 * @param stride_row
 * @param stride_col
 * @return
 */
Tensor4f Conv(Tensor4f _kernel, Tensor1f _bias, Tensor4f _input, int stride_row, int stride_col) {
    long _output_row = ceil((_input.dimension(1) - _kernel.dimension(0)) / stride_row) + 1;
    long _output_col = ceil((_input.dimension(2) - _kernel.dimension(1)) / stride_col) + 1;
    Tensor4f _output = Tensor4f(_input.dimension(0), _output_row,
                                _output_col, _kernel.dimension(3)); /// return value

    for (auto i = 0; i < _output.dimension(0); ++i) {
        for (auto l = 0; l < _output.dimension(3); ++l) {

            for (auto j = 0; j < _output.dimension(1); ++j) {
                for (auto k = 0; k < _output.dimension(2); ++k) {

                    long index_row = stride_row * j;
                    long index_col = stride_col * k;
                    float val = 0.f;

                    if (index_row >= _input.dimension(1) || index_col >= _input.dimension(2))
                        throw("runtime error."); // padding (not implement)

                    for (auto init_channel = 0; init_channel < _input.dimension(3); ++init_channel)
                        for (auto m = 0; m < _kernel.dimension(0); ++m)
                            for (auto n = 0; n < _kernel.dimension(1); ++n)
                                val += _input(i, index_row + m, index_col + n,
                                              init_channel) * _kernel(m, n, init_channel, l);

                    _output(i, j, k, l) = val + _bias(l);
                }
            }

        }
    }

    return _output;
}
}
}
}
