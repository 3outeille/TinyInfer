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

//namespace tinyinfer {
//namespace runtime {
//namespace kernel {

/**
 * @brief dense layer
 * @param _kernel
 * @param _bias
 * @param _input
 */
Tensor2f Dense(Tensor2f _kernel, Tensor1f _bias, Tensor2f _input);

/**
 * @brief flatten layer
 * @param _input
 */
Tensor2f Flatten(Tensor4f _input);

/**
 * @brief max pooling layer
 * @param _input
 * @param window_row
 * @param window_col
 * @param stride_row
 * @param stride_col
 */
Tensor4f Maxpool(Tensor4f _input, int window_row, int window_col, int stride_row, int stride_col);

/**
 * @brief convolution layer
 * @param _kernel
 * @param _bias
 * @param _input
 * @param stride_row
 * @param stride_col
 * @return
 */
Tensor4f Conv(Tensor4f _kernel, Tensor1f _bias, Tensor4f _input, int stride_row, int stride_col);

/**
 * @brief relu layer
 * @param _input
 */
Tensor2f Relu(Tensor2f _input);
Tensor3f Relu(Tensor3f _input);
Tensor4f Relu(Tensor4f _input);

/**
 * @brief drop out layer
 * @param _input
 */
Tensor2f Dropout(Tensor2f _input);
Tensor3f Dropout(Tensor3f _input);
Tensor4f Dropout(Tensor4f _input);

/**
 * @brief soft max layer
 * @param _input
 */
Tensor2f Softmax(Tensor2f _input);
Tensor3f Softmax(Tensor3f _input);
Tensor4f Softmax(Tensor4f _input);

//}
//}
//}

//#include "dense.hpp"
//#include "flatten.hpp"
//#include "maxpool.hpp"
//#include "convolution.hpp"
//#include "relu.hpp"
//#include "dropout.hpp"
//#include "softmax.hpp"
