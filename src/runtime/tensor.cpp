#include "runtime/tensor.hpp"

using namespace tinyinfer;

runtime::Tensor::Tensor(
    Eigen::Tensor<TENSOR_DATA_TYPE, 1, Eigen::RowMajor>& tensor)
    : m_tensor_rank1(tensor), m_rank(1) {}
runtime::Tensor::Tensor(
    Eigen::Tensor<TENSOR_DATA_TYPE, 2, Eigen::RowMajor>& tensor)
    : m_tensor_rank2(tensor), m_rank(2) {}
runtime::Tensor::Tensor(
    Eigen::Tensor<TENSOR_DATA_TYPE, 3, Eigen::RowMajor>& tensor)
    : m_tensor_rank3(tensor), m_rank(3) {}
runtime::Tensor::Tensor(
    Eigen::Tensor<TENSOR_DATA_TYPE, 4, Eigen::RowMajor>& tensor)
    : m_tensor_rank4(tensor), m_rank(4) {}