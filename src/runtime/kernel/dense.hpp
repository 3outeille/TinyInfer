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
 * @brief dense layer
 * @param _kernel
 * @param _bias
 * @param _input
 */
Tensor2f Dense(Tensor2f _kernel, Tensor1f _bias, Tensor2f _input) {
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

    return _output;
}
}
}
}
