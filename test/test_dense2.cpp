#include <iostream>
#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../src/runtime/kernel/dense.hpp"
#include "../src/runtime/kernel/relu.hpp"
#include "io.h"

int main( int argc, char** argv ) {

    Eigen::Tensor<float, 2> in = io::load_kernel_weight_2d("../data/io_data/flatten_1_output");
    Eigen::Tensor<float, 2> ou = io::load_kernel_weight_2d("../data/io_data/dense_1_output");
    Eigen::Tensor<float, 2> ke = io::load_kernel_weight_2d("../data/tensor_weights/dense_1_kernel.kw");
    Eigen::Tensor<float, 1> bi = io::load_kernel_weight_1d("../data/tensor_weights/dense_1_bias.kw");


    std::cout << "~~~~~\n";

    Eigen::Tensor<float, 2> ou_res = tinyinfer::runtime::kernel::Dense(ke, bi, in);
    ou_res = tinyinfer::runtime::kernel::Relu(ou_res);

    for (auto i = 0; i < ou.dimension(0); ++i) {
        for (auto j = 0; j < ou.dimension(1); ++j) {

            if ( ou(i, j) - ou_res(i, j) > 0.00001 || ou(i, j) - ou_res(i, j) < -0.00001 ) {
                std::cout << i << " " << j << std::endl;
                std::cout << ou(i, j) << std::endl;
                std::cout << ou_res(i, j) << std::endl;
                return 0;
            }
        }
    }

    return 1;
}
