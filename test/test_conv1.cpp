#include <iostream>
#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../src/runtime/kernel/convolution.hpp"
#include "../src/runtime/kernel/relu.hpp"
#include "io.h"

int main( int argc, char** argv ) {

    Eigen::Tensor<float, 4> in = io::load_kernel_weight_4d("../data/io_data/conv2d_1_output");
    Eigen::Tensor<float, 4> ou = io::load_kernel_weight_4d("../data/io_data/conv2d_2_output");
    Eigen::Tensor<float, 4> ke = io::load_kernel_weight_4d("../data/tensor_weights/conv2d_2_kernel.kw");
    Eigen::Tensor<float, 1> bi = io::load_kernel_weight_1d("../data/tensor_weights/conv2d_2_bias.kw");


    std::cout << "~~~~~\n";

    Eigen::Tensor<float, 4> ou_res = tinyinfer::runtime::kernel::Conv(ke, bi, in, 1, 1);
    ou_res = tinyinfer::runtime::kernel::Relu(ou_res);

    for (auto i = 0; i < ou.dimension(0); ++i) {
        for (auto j = 0; j < ou.dimension(1); ++j) {
            for (auto k = 0; k < ou.dimension(2); ++k) {
                for (auto l = 0; l < ou.dimension(3); ++l) {

                    if (ou(i, j, k, l) - ou_res(i, j, k, l) > 0.00001 ||
                        ou(i, j, k, l) - ou_res(i, j, k, l) < -0.00001) {
                        std::cout << i << " " << j << " " << k << " " << l << std::endl;
                        std::cout << ou(i, j, k, l) << std::endl;
                        std::cout << ou_res(i, j, k, l) << std::endl;
                        return 0;
                    }
                }
            }
        }
    }

    return 1;
}
