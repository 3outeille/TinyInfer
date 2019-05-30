/***
 * Do the MNIST prediction
 * input arguemnts:
 *  1. input_pb: input model file e.g.(model.pb)
 *  2. tensor_weights_dir: input tensor weights directory e.g.(./tensor_weights)
 *  3. input tensor: input tensor file e.g.(input.tensor)
 */

#include <iostream>
#include "frontend/parser.hpp"
#include "function.hpp"
#include "runtime/tensor.hpp"
#include "utils/io.hpp"

using namespace tinyinfer;

typedef typename Eigen::Tensor<TENSOR_DATA_TYPE, 1, Eigen::RowMajor> Tensor1f;
typedef typename Eigen::Tensor<TENSOR_DATA_TYPE, 2, Eigen::RowMajor> Tensor2f;
typedef typename Eigen::Tensor<TENSOR_DATA_TYPE, 3, Eigen::RowMajor> Tensor3f;
typedef typename Eigen::Tensor<TENSOR_DATA_TYPE, 4, Eigen::RowMajor> Tensor4f;


int main(int argc, const char * argv[]){
    if (argc != 4){
        std::cerr << "[ERROR] invalid number of arguments. Expected 4" << std::endl;
        return 1;
    }

    // parse graph
    std::string input_pb = std::string(argv[1]);
    std::string tensor_weights_dir = std::string(argv[2]);

    Parser parser;
    auto graph = parser.parse(input_pb, tensor_weights_dir);

    // load data
    std::string input_tensor_frame = std::string(argv[3]);
    runtime::Tensor pre_input_tensor = io::load_kernel_weight_3d(input_tensor_frame);

    // load output data
    Tensor3f _pre_input = pre_input_tensor.get_tensor_r3_ptr();
    long total_number = _pre_input.dimension(0); // total number of images

    // preprocess data
    long num_batch = 1;
    long batch_size = total_number / num_batch;

    long fail_case = 0;

    for (auto b = 0; b < num_batch; ++b) {
//        std::cout << "-------------- Batch" << b <<  std::endl;
        Tensor4f _input = Tensor4f(batch_size, _pre_input.dimension(1), _pre_input.dimension(2), 1);
        long index_pre = b * batch_size;

        #pragma omp parallel for collapse(3)

        for (auto i = 0; i < _input.dimension(0); ++i) {
            for (auto j = 0; j < _input.dimension(1); ++j) {
                for (auto k = 0; k < _input.dimension(2); ++k) {
                    _input(i, j, k, 0) = _pre_input(index_pre + i, j ,k);
                }
            }
        }

        runtime::Tensor input_tensor = runtime::Tensor(_input);

        // function
        Function func(graph, {parser.get_input()}, {graph[graph.size()-1]});
        auto output_tensors = func.forward({input_tensor});
        assert(output_tensors.size() == 1);
        auto pre_output_eigen = output_tensors.at(0).get_tensor_r2_ptr();

        Tensor1f output_eigen = Tensor1f(batch_size);

        for (auto i = 0; i < batch_size; ++i) {
            float max_val = pre_output_eigen(i, 0);
            long max_index = 0;

            for (auto j = 0; j < pre_output_eigen.dimension(1); ++j) {
                if (pre_output_eigen(i, j) > max_val) {
                    max_val = pre_output_eigen(i, j);
                    max_index = j;
                }
            }

            output_eigen(i) = max_index;
        }

        //test
        for (auto i = 0; i < batch_size; ++i) {
            std::cout << output_eigen(i) << " ";
        }
        std::cout << std::endl;

    }

//    std::cout << "Program Complete Execution" << std::endl;
//    std::cout << "Fail cases number:" << fail_case << std::endl;

    return 0;
}