//
// Created by ernest on 19-5-27.
//

#ifndef LOADER_TENSOR_BASE_H
#define LOADER_TENSOR_BASE_H

#include <string>

class TensorBase {
public:
    TensorBase(std::string name);

    std::string name;
    std::string op;
    std::string input_tensor;
};


#endif //LOADER_TENSOR_BASE_H
