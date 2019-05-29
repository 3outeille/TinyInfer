#include <iostream>
#include <memory>
#include "tinyinfer.hpp"

using namespace tinyinfer;
using namespace tinyinfer::op;

int main(int argc, const char* argv[]) {
  auto input_node = std::make_shared<Parameter>();
  Eigen::Tensor<TENSOR_DATA_TYPE, 4, Eigen::RowMajor> t_4d(2, 2, 3, 2);
  t_4d(0, 0, 0, 0) = 1;
  t_4d(0, 1, 0, 0) = -1;
  t_4d(1, 0, 0, 0) = 2;
  t_4d(1, 1, 0, 0) = 1.4;

  t_4d(0, 0, 1, 0) = 1;
  t_4d(0, 1, 1, 0) = -1;
  t_4d(1, 0, 1, 0) = 2;
  t_4d(1, 1, 1, 0) = 1.4;

  t_4d(0, 0, 2, 0) = 1;
  t_4d(0, 1, 2, 0) = -1;
  t_4d(1, 0, 2, 0) = 2;
  t_4d(1, 1, 2, 0) = 1.4;

  t_4d(0, 0, 0, 1) = 1;
  t_4d(0, 1, 0, 1) = -1;
  t_4d(1, 0, 0, 1) = 2;
  t_4d(1, 1, 0, 1) = 1.4;

  t_4d(0, 0, 1, 1) = 1;
  t_4d(0, 1, 1, 1) = -1;
  t_4d(1, 0, 1, 1) = 2;
  t_4d(1, 1, 1, 1) = 1.4;

  t_4d(0, 0, 2, 1) = 1;
  t_4d(0, 1, 2, 1) = -1;
  t_4d(1, 0, 2, 1) = 2;
  t_4d(1, 1, 2, 1) = 1.4;
  
  input_node->register_input(std::make_shared<runtime::Tensor>(t_4d));

  auto sampleOp_node = std::make_shared<Maxpooling2dOp>(input_node);
//  Eigen::Tensor<TENSOR_DATA_TYPE, 1, Eigen::RowMajor> c_1d(2);
//  c_1d(0) = 1;
//  c_1d(1) = 1;
//  sampleOp_node->register_weight(std::make_shared<runtime::Tensor>(c_1d));

  assert(input_node->get_outputs().at(0).get_tensor_ptr() == nullptr);
  input_node->forward();
  assert(input_node->get_outputs().at(0).get_tensor_ptr() != nullptr);

  assert(sampleOp_node->get_outputs().at(0).get_tensor_ptr() == nullptr);
  sampleOp_node->forward();
  assert(sampleOp_node->get_outputs().at(0).get_tensor_ptr() != nullptr);

  auto res = sampleOp_node->get_outputs().at(0).get_tensor_ptr();
  std::cout << res->get_tensor_r4_ptr() << std::endl;
}