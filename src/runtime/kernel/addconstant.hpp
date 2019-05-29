#pragma once

#include "runtime/tensor.hpp"

namespace tinyinfer {
namespace runtime {
namespace kernel {
std::shared_ptr<runtime::Tensor> addconstant(
    const std::shared_ptr<runtime::Tensor> input,
    const std::shared_ptr<runtime::Tensor> constant) {
  Eigen::Tensor<TENSOR_DATA_TYPE, 1, Eigen::RowMajor> result =
      input->get_tensor_r1_ptr() + constant->get_tensor_r1_ptr();
  return std::make_shared<runtime::Tensor>(result);
}
}
}
}