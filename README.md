# TinyInfer

## Dependency
#### Libraries
- [google protobuf](https://github.com/protocolbuffers/protobuf)
- [nlohmann json](https://github.com/nlohmann/json)
- eigen3 (auto-download on building)

Notes: 
- Please make sure your protobuf library supports proto3
- It is recommended to compile and install protobuf and nlohmann-json from source code


#### Local Test Environment
- Ubuntu 16.04
- Protobuf 3.7.1

## Usage 
### Compilation
```bash
mkdir build
cd build
cmake ..
make -j3
```

### using for MNIST inference

##### Generate input tensor and weights
Refer to jupyter nodebooks under ```train/jupyter```

##### Inference
```bash
./predict {PATH_TO_PB_MODEL} {DIR_TO_TENSOR_WEIGHTS} {PATH_TO_INPUT}
# for exmaple
./predict ../data/model.pb ../data/tensor_weights ../data/demo_input/input.tensor
```

### static library
```cmake
add_subdirectory(src)
```
This will invoke the cmake file in src and produce a static library ```TinyinferLib```, which can be included for inferring other network.

example usage:
```cmake
add_executable(predict test/predict.cpp)
target_link_libraries(predict TinyinferLib)
```


### Acknowledgement
This project is the course project for CS133 advanced C++ at ShanghaiTech University.

### Author
- Jianxiong Cai
- Zhiqiang Xie
- Jiadi Cui