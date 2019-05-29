#pragma once

#include <memory>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>

#include "shape.hpp"

#define TENSOR_DATA_TYPE float

namespace tinyinfer {
namespace runtime {
class Tensor {
 private:
  Shape m_shape;
  size_t m_rank;
  Eigen::Tensor<TENSOR_DATA_TYPE, 1, Eigen::RowMajor> m_tensor_rank1;
  Eigen::Tensor<TENSOR_DATA_TYPE, 2, Eigen::RowMajor> m_tensor_rank2;
  Eigen::Tensor<TENSOR_DATA_TYPE, 3, Eigen::RowMajor> m_tensor_rank3;
  Eigen::Tensor<TENSOR_DATA_TYPE, 4, Eigen::RowMajor> m_tensor_rank4;

 public:
  Tensor(Eigen::Tensor<TENSOR_DATA_TYPE, 1, Eigen::RowMajor>& tensor);
  Tensor(Eigen::Tensor<TENSOR_DATA_TYPE, 2, Eigen::RowMajor>& tensor);
  Tensor(Eigen::Tensor<TENSOR_DATA_TYPE, 3, Eigen::RowMajor>& tensor);
  Tensor(Eigen::Tensor<TENSOR_DATA_TYPE, 4, Eigen::RowMajor>& tensor);

  const Shape& get_shape() const { return m_shape; }

  const Eigen::Tensor<TENSOR_DATA_TYPE, 1, Eigen::RowMajor>&
  get_tensor_r1_ptr() {
    return m_tensor_rank1;
  }
  const Eigen::Tensor<TENSOR_DATA_TYPE, 2, Eigen::RowMajor>&
  get_tensor_r2_ptr() {
    return m_tensor_rank2;
  }
  const Eigen::Tensor<TENSOR_DATA_TYPE, 3, Eigen::RowMajor>&
  get_tensor_r3_ptr() {
    return m_tensor_rank3;
  }
  const Eigen::Tensor<TENSOR_DATA_TYPE, 4, Eigen::RowMajor>&
  get_tensor_r4_ptr() {
    return m_tensor_rank4;
  }
};
}  // namespace runtime
}  // namespace tinyinfer
