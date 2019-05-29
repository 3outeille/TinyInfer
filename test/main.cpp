/***
 * Dummy File for testing cmake
 * Includes some header files
 */

#include <iostream>
#include "graph.pb.h"
#include "runtime/tensor.hpp"
#include "utils/io.hpp"

std::shared_ptr<tinyinfer::runtime::Tensor> foo(const std::shared_ptr<tinyinfer::runtime::Tensor>& tensor_in){
    std::shared_ptr<tinyinfer::runtime::Tensor> m_weights = nullptr;
    m_weights = tensor_in;
    return m_weights;
}

int main(int argc, const char * argv[]){
    std::cout << "Hello World" << std::endl;

    std::string input_tensor_1_fname = "/home/ernest/cs133_proj/data/tensor_weights/conv2d_1_bias.kw";
    std::string input_tensor_4_fname = "/home/ernest/cs133_proj/data/tensor_weights/conv2d_1_kernel.kw";

    auto tensor_ptr_1d = std::make_shared<tinyinfer::runtime::Tensor>(io::load_kernel_weight_1d(input_tensor_1_fname));
    auto tensor_ptr_4d = std::make_shared<tinyinfer::runtime::Tensor>(io::load_kernel_weight_4d(input_tensor_4_fname));

    auto a = foo(tensor_ptr_4d);

    return 0;
}