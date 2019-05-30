/***
 * Parser test
 */

#include <iostream>
#include "frontend/parser.hpp"
#include "function.hpp"
#include "runtime/tensor.hpp"
#include "utils/io.hpp"
#include <chrono>

using namespace tinyinfer;

int main(int argc, const char * argv[]){
    // Eigen::initParallel();
    // parse graph
    std::string input_pb = "data/model.pb";
    std::string tensor_weights_dir = "data/tensor_weights";

    Parser parser;
    auto graph = parser.parse(input_pb, tensor_weights_dir);

    // load data
    std::string input_tensor_fname = "data/io_data/input.tensor";
    std::string expected_output_fname = "data/io_data/dense_2_output";
    runtime::Tensor input_tensor = io::load_kernel_weight_4d(input_tensor_fname);

    // fucntion
    Function func(graph, {parser.get_input()}, {graph[graph.size()-1]});
    std::cout << "original number of node " << func.get_graph().size() << "\n";
    func.optimize_graph();
    std::cout << "optimized graph, number of node " << func.get_graph().size() << "\n";
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 9; i++){
        func.forward({input_tensor});
    }
    auto output_tensors = func.forward({input_tensor});
    auto finish = std::chrono::high_resolution_clock::now();
    std::cout<<"time collapsed: " << std::chrono::duration_cast<std::chrono::seconds>(finish-start).count()/10.0 << " s\n";
    assert(output_tensors.size() == 1);
//    std::shared_ptr<runtime::Tensor> output_tensor = graph[graph.size()-1]->get_outputs().at(0).get_tensor_ptr();
    auto output_eigen = output_tensors.at(0).get_tensor_r2_ptr();
//    std::cout << output_tensor->get_rank() << std::endl;
//    std::cout << output_tensor->get_tensor_r2_ptr()(0,0) << std::endl;

    // load output data
    runtime::Tensor expected_output_tensor = io::load_kernel_weight_2d(expected_output_fname);
    auto expect_output_eigen = expected_output_tensor.get_tensor_r2_ptr();
    for (int i = 0; i < output_eigen.dimension(0); i++){
        for (int j = 0; j < output_eigen.dimension(1); j++){
            if (std::abs(output_eigen(i,j) - expect_output_eigen(i,j)) > 1e-6){
                std::cerr << "[WARN] " << output_eigen(i,j) << " does not match with expected " << expect_output_eigen(i,j) << std::endl;
            }
        }
    }
    std::cout << "Program Complete Execution" << std::endl;

    return 0;
}