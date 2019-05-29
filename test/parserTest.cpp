/***
 * Parser test
 */

#include <iostream>
#include "frontend/parser.hpp"

int main(int argc, const char * argv[]){
    std::string input_pb = "/home/ernest/cs133_proj/data/model.pb";
    std::string tensor_weights_dir = "/home/ernest/cs133_proj/data/tensor_weights";

    tinyinfer::Parser parser;
    parser.parse(input_pb, tensor_weights_dir);

    return 0;
}