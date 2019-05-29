#include <iostream>
#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../src/runtime/kernel/dense.hpp"

int main( int argc, char** argv ) {

    float a[4] = {1, 2, 3, 4};
    float b[4] = {1, 3, 2, 4};
    float c[2] = {6, 5};

    Eigen::TensorMap<Eigen::Tensor<float, 2>> in(a, 2, 2);
    Eigen::TensorMap<Eigen::Tensor<float, 2>> ke(b, 2, 2);
    Eigen::TensorMap<Eigen::Tensor<float, 1>> bi(c, 2);

    std::cout << in << std::endl;
    std::cout << ke << std::endl;
    std::cout << bi << std::endl;

    std::cout << "~~~~~\n";

    Eigen::Tensor<float, 2> ou_res = tinyinfer::runtime::kernel::Dense(ke, bi, in);
    std::cout << ou_res << std::endl;

    return 0;
}