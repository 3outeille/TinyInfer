#include <iostream>
#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../src/runtime/kernel/relu.hpp"

int main( int argc, char** argv ) {

    Eigen::Tensor<float, 3> in(2, 2, 3);
    in(0, 0, 0) = 1;
    in(0, 1, 0) = -32;
    in(1, 0, 0) = -9.2;
    in(1, 1, 0) = 0.2;

    in(0, 0, 1) = 1;
    in(0, 1, 1) = -32;
    in(1, 0, 1) = -6.23;
    in(1, 1, 1) = 0.2;

    in(0, 0, 2) = -54;
    in(0, 1, 2) = -52;
    in(1, 0, 2) = -9.23;
    in(1, 1, 2) = -23.2;

    std::cout << in << std::endl;
    std::cout << "~~~~~\n";

    Eigen::Tensor<float, 3> ou_res = Relu(in);
    std::cout << ou_res << std::endl;

    return 0;
}