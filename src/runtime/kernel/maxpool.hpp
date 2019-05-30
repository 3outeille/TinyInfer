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
 * @brief max pooling layer
 * @param _input
 * @param window_row
 * @param window_col
 * @param stride_row
 * @param stride_col
 * @return a tensor after the max pooling layer
 */
            std::shared_ptr<runtime::Tensor> Maxpool(std::shared_ptr<runtime::Tensor> input,
                                                     int window_row, int window_col, int stride_row, int stride_col) {
                Tensor4f _input = input->get_tensor_r4_ptr();
                Tensor4f _output = Tensor4f(_input.dimension(0), ceil((float) _input.dimension(1) / stride_row),
                                            ceil((float) _input.dimension(2) / stride_col),
                                            _input.dimension(3)); /// return value

#pragma omp parallel for collapse(4)
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

                return std::make_shared<runtime::Tensor>(_output);
            }
        }
    }
}
