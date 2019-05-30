#include <iostream>
#include <memory>
#include "tinyinfer.hpp"

using namespace tinyinfer;
using namespace tinyinfer::op;

int main(int argc, const char* argv[]) {
  auto input_node = std::make_shared<Parameter>();
  auto sampleOp_node = std::make_shared<AddConstant>(input_node);
  Eigen::Tensor<TENSOR_DATA_TYPE, 1, Eigen::RowMajor> c_1d(2);
  c_1d(0) = 1;
  c_1d(1) = 1;
  sampleOp_node->register_weight(std::make_shared<runtime::Tensor>(c_1d));

  Eigen::Tensor<TENSOR_DATA_TYPE, 1, Eigen::RowMajor> t_1d(2);
  t_1d(0) = 1;
  t_1d(1) = 2;
  Function f({input_node, sampleOp_node}, {input_node}, {sampleOp_node});
  auto res = f.forward({runtime::Tensor(t_1d)}).at(0);

  // auto res = sampleOp_node->get_outputs().at(0).get_tensor_ptr();
  std::cout << res.get_tensor_r1_ptr()(0) << " got 2?\n";
  std::cout << res.get_tensor_r1_ptr()(1) << " got 3?\n";
}