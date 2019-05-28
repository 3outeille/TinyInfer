#pragma once

#include <memory>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>

#include "shape.hpp"

#define TENSOR_DATA_TYPE float

namespace tinyinfer {
class Tensor {
 private:
  Shape m_shape;
  size_t m_rank;
  std::unique_ptr<Eigen::Tensor<TENSOR_DATA_TYPE, 1, Eigen::RowMajor>>
      m_tensor_rank1;
  std::unique_ptr<Eigen::Tensor<TENSOR_DATA_TYPE, 2, Eigen::RowMajor>>
      m_tensor_rank2;
  std::unique_ptr<Eigen::Tensor<TENSOR_DATA_TYPE, 3, Eigen::RowMajor>>
      m_tensor_rank3;
  std::unique_ptr<Eigen::Tensor<TENSOR_DATA_TYPE, 4, Eigen::RowMajor>>
      m_tensor_rank4;

 public:
  Tensor(Eigen::Tensor<TENSOR_DATA_TYPE, 1, Eigen::RowMajor> tensor);
  Tensor(Eigen::Tensor<TENSOR_DATA_TYPE, 2, Eigen::RowMajor> tensor);
  Tensor(Eigen::Tensor<TENSOR_DATA_TYPE, 3, Eigen::RowMajor> tensor);
  Tensor(Eigen::Tensor<TENSOR_DATA_TYPE, 4, Eigen::RowMajor> tensor);
};
}  // namespace tinyinfer
