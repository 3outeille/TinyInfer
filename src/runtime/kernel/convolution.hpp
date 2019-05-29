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
 * @brief convolution layer
 * @param _kernel
 * @param _bias
 * @param _input
 * @param stride_row
 * @param stride_col
 * @return
 */
const std::shared_ptr<runtime::Tensor>Conv(const std::shared_ptr<runtime::Tensor> kernel,
                                            const std::shared_ptr<runtime::Tensor> bias,
                                            const std::shared_ptr<runtime::Tensor> input,
                                            int stride_row, int stride_col) {
    Tensor4f _kernel = kernel->get_tensor_r2_ptr();
    Tensor1f _bias = bias->get_tensor_r1_ptr();
    Tensor4f _input = input->get_tensor_r2_ptr();

    long _output_row = ceil((float)(_input.dimension(1) - _kernel.dimension(0)) / stride_row) + 1;
    long _output_col = ceil((float)(_input.dimension(2) - _kernel.dimension(1)) / stride_col) + 1;
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

    return std::make_shared<runtime::Tensor>(_output);
}
}
}
}
