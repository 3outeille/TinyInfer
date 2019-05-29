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
 * @brief dense layer
 * @param _kernel
 * @param _bias
 * @param _input
 */
std::shared_ptr<runtime::Tensor> Dense(const std::shared_ptr<runtime::Tensor> kernel,
                                        const std::shared_ptr<runtime::Tensor> bias,
                                        const std::shared_ptr<runtime::Tensor> input) {
    Tensor2f _kernel = kernel->get_tensor_r2_ptr();
    Tensor1f _bias = bias->get_tensor_r1_ptr();
    Tensor2f _input = input->get_tensor_r2_ptr();

    Tensor2f _output = Tensor2f(_input.dimension(0), _kernel.dimension(1)); /// return value
    _output.setZero(); // init

    Eigen::array<Eigen::IndexPair<int>, 1> product_dims{ Eigen::IndexPair<int>(1, 0) };
    Eigen::array<Eigen::Index, 2> broadcast_dims{ _input.dimension(0), 1 };
    _output = _input.contract(_kernel, product_dims);

//    for (auto i = 0; i < _output.dimension(0); ++i) {
//        for (auto j = 0; j < _output.dimension(1); ++j) {
//            _output(i, j) += _bias(j);
//        }
//    }

    Tensor2f temp_bias = Tensor2f(1, _bias.dimension(0));
    for (auto i = 0; i < _bias.dimension(0); ++i)
        temp_bias(0, i) = _bias(i);

    _output += temp_bias.broadcast(broadcast_dims);

    return std::make_shared<runtime::Tensor>(_output);
}
}
}
}
