#include "runtime/tensor.hpp"

using namespace tinyinfer;

runtime::Tensor::Tensor(
    std::unique_ptr<Eigen::Tensor<TENSOR_DATA_TYPE, 1, Eigen::RowMajor>> tensor)
    : m_tensor_rank1(std::move(tensor)), m_rank(1) {}

runtime::Tensor::Tensor(
    std::unique_ptr<Eigen::Tensor<TENSOR_DATA_TYPE, 2, Eigen::RowMajor>> tensor)
    : m_tensor_rank2(std::move(tensor)), m_rank(2) {}

runtime::Tensor::Tensor(
    std::unique_ptr<Eigen::Tensor<TENSOR_DATA_TYPE, 3, Eigen::RowMajor>> tensor)
    : m_tensor_rank3(std::move(tensor)), m_rank(3) {}

runtime::Tensor::Tensor(
    std::unique_ptr<Eigen::Tensor<TENSOR_DATA_TYPE, 4, Eigen::RowMajor>> tensor)
    : m_tensor_rank4(std::move(tensor)), m_rank(4) {}