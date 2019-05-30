#pragma once

#include <memory>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>

#include "shape.hpp"

#define TENSOR_DATA_TYPE float

namespace tinyinfer {
namespace runtime {

/**
 * @brief a class for runtime tensor.
 */
class Tensor {
 private:
  Shape m_shape;
  Eigen::Tensor<TENSOR_DATA_TYPE, 1, Eigen::RowMajor> m_tensor_rank1;
  Eigen::Tensor<TENSOR_DATA_TYPE, 2, Eigen::RowMajor> m_tensor_rank2;
  Eigen::Tensor<TENSOR_DATA_TYPE, 3, Eigen::RowMajor> m_tensor_rank3;
  Eigen::Tensor<TENSOR_DATA_TYPE, 4, Eigen::RowMajor> m_tensor_rank4;
  size_t m_rank; /// the rank of Tensor

 public:
  Tensor(const Eigen::Tensor<TENSOR_DATA_TYPE, 1, Eigen::RowMajor>& tensor); /// constructor
  Tensor(const Eigen::Tensor<TENSOR_DATA_TYPE, 2, Eigen::RowMajor>& tensor); /// constructor
  Tensor(const Eigen::Tensor<TENSOR_DATA_TYPE, 3, Eigen::RowMajor>& tensor); /// constructor
  Tensor(const Eigen::Tensor<TENSOR_DATA_TYPE, 4, Eigen::RowMajor>& tensor); /// constructor

  const Shape& get_shape() const { return m_shape; } /// simple getter

  const size_t& get_rank() const { return m_rank; } /// simple getter

  const Eigen::Tensor<TENSOR_DATA_TYPE, 1, Eigen::RowMajor>&
  get_tensor_r1_ptr() {
      if (m_rank != 1){
          throw std::runtime_error("[Tensor] m_rank != 1");
      }

    return m_tensor_rank1;
  } /// simple getter
  const Eigen::Tensor<TENSOR_DATA_TYPE, 2, Eigen::RowMajor>&
  get_tensor_r2_ptr() {
      if (m_rank != 2){
          throw std::runtime_error("[Tensor] m_rank != 2");
      }
    return m_tensor_rank2;
  } /// simple getter
  const Eigen::Tensor<TENSOR_DATA_TYPE, 3, Eigen::RowMajor>&
  get_tensor_r3_ptr() {
      if (m_rank != 3){
          throw std::runtime_error("[Tensor] m_rank != 3");
      }
    return m_tensor_rank3;
  } /// simple getter
  const Eigen::Tensor<TENSOR_DATA_TYPE, 4, Eigen::RowMajor>&
  get_tensor_r4_ptr() {
      if (m_rank != 4){
          throw std::runtime_error("[Tensor] m_rank != 4");
      }
    return m_tensor_rank4;
  } /// simple getter
};
}  // namespace runtime
}  // namespace tinyinfer
