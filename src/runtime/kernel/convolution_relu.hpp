#pragma once

#include <iostream>
#include <math.h>
#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

#include "runtime/tensor.hpp"
#include "runtime/kernel/convolution.hpp"
#include "runtime/kernel/relu.hpp"

namespace tinyinfer {
namespace runtime {
namespace kernel {

typedef typename Eigen::Tensor<TENSOR_DATA_TYPE, 1, Eigen::RowMajor> Tensor1f;
typedef typename Eigen::Tensor<TENSOR_DATA_TYPE, 2, Eigen::RowMajor> Tensor2f;
typedef typename Eigen::Tensor<TENSOR_DATA_TYPE, 3, Eigen::RowMajor> Tensor3f;
typedef typename Eigen::Tensor<TENSOR_DATA_TYPE, 4, Eigen::RowMajor> Tensor4f;

/**
 * @brief convolution and relu layer
 * @param _kernel
 * @param _bias
 * @param _input
 * @param stride_row
 * @param stride_col
 * @return a tensor after the convolution and relu layer
 */
const std::shared_ptr<runtime::Tensor>Conv_Relu(const std::shared_ptr<runtime::Tensor> kernel,
                                            const std::shared_ptr<runtime::Tensor> bias,
                                            const std::shared_ptr<runtime::Tensor> input,
                                            int stride_row, int stride_col) {

    std::shared_ptr<runtime::Tensor> pre_output = Conv(kernel, bias, input, stride_row, stride_col);
    std::shared_ptr<runtime::Tensor> output = Relu(pre_output);

    return output;
}
}
}
}
