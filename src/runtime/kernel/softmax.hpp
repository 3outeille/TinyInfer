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
 * @brief soft max layer
 * @param _input
 */
Tensor2f Softmax(Tensor2f _input) {
    auto shiftedInput = _input - _input.maximum(Eigen::array<long, 1>{1})
            .eval().reshape(Eigen::array<long, 2>{_input.dimension(0), 1})
            .broadcast(Eigen::array<long, 2>{1, _input.dimension(1)});

    auto exponentiated = shiftedInput.exp();
    Tensor2f _output = exponentiated * exponentiated.sum(Eigen::array<long, 1>{1})
            .inverse().eval()
            .reshape(Eigen::array<long, 2>({_input.dimension(0), 1}))
            .broadcast(Eigen::array<long, 2>({1, _input.dimension(1)}));
    return _output;
}
Tensor3f Softmax(Tensor3f _input) {
    auto shiftedInput = _input - _input.maximum(Eigen::array<long, 1>{1})
            .eval().reshape(Eigen::array<long, 2>{_input.dimension(0), 1})
            .broadcast(Eigen::array<long, 2>{1, _input.dimension(1)});

    auto exponentiated = shiftedInput.exp();
    Tensor2f _output = exponentiated * exponentiated.sum(Eigen::array<long, 1>{1})
            .inverse().eval()
            .reshape(Eigen::array<long, 2>({_input.dimension(0), 1}))
            .broadcast(Eigen::array<long, 2>({1, _input.dimension(1)}));
    return _output;
}
Tensor4f Softmax(Tensor4f _input) {
    auto shiftedInput = _input - _input.maximum(Eigen::array<long, 1>{1})
            .eval().reshape(Eigen::array<long, 2>{_input.dimension(0), 1})
            .broadcast(Eigen::array<long, 2>{1, _input.dimension(1)});

    auto exponentiated = shiftedInput.exp();
    Tensor2f _output = exponentiated * exponentiated.sum(Eigen::array<long, 1>{1})
            .inverse().eval()
            .reshape(Eigen::array<long, 2>({_input.dimension(0), 1}))
            .broadcast(Eigen::array<long, 2>({1, _input.dimension(1)}));
    return _output;
}
}
}
}
