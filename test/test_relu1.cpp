#include <iostream>
#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../src/runtime/kernel/relu.hpp"

int main( int argc, char** argv ) {

    Eigen::Tensor<float, 2> in(2, 2);
    in(0, 0) = 1;
    in(0, 1) = -32;
    in(1, 0) = -9.2;
    in(1, 1) = 0.2;


    std::cout << in << std::endl;
    std::cout << "~~~~~\n";

    Eigen::Tensor<float, 2> ou_res = Relu(in);
    std::cout << ou_res << std::endl;

    return 0;
}