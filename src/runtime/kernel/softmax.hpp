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
 * @brief soft max layer
 * @param _input
 * @return a tensor after the soft max layer
 */
            std::shared_ptr<runtime::Tensor> Softmax(std::shared_ptr<runtime::Tensor> input) {
                Tensor2f _input = input->get_tensor_r2_ptr();
                auto shiftedInput = _input - _input.maximum(Eigen::array<long, 1>{1})
                        .eval().reshape(Eigen::array<long, 2>{_input.dimension(0), 1})
                        .broadcast(Eigen::array<long, 2>{1, _input.dimension(1)});

                auto exponentiated = shiftedInput.exp();
                Tensor2f _output = exponentiated * exponentiated.sum(Eigen::array<long, 1>{1})
                        .inverse().eval()
                        .reshape(Eigen::array<long, 2>({_input.dimension(0), 1}))
                        .broadcast(Eigen::array<long, 2>({1, _input.dimension(1)}));
                return std::make_shared<runtime::Tensor>(_output);
            }
        }
    }
}
