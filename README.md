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
#### Data
- download data from the [realse page](https://github.com/jianxiongcai/cs133_tinyinfer_data/releases) of the data repo.
- untar the file in project root directory. This will have everything needed for running demo in ```data``` folder.
- Alternatively you may refer to the jupyter notebooks under [train/jupyter](train/jupyter) to generate new input data. 

#### Generate input tensor and weights
Refer to jupyter nodebooks under [demo/jupyter/demo.ipynb](demo/jupyter/demo.ipynb)

#### Inference
```bash
./predict {PATH_TO_PB_MODEL} {DIR_TO_TENSOR_WEIGHTS} {PATH_TO_INPUT}
# for exmaple
./predict ../data/model.pb ../data/tensor_weights ../data/demo_input/input.tensor
```

Alternatively, we provide a jupyter notebook for reference. (See jupyter Notebook [demo.ipynb](demo/jupyter/demo.ipynb))
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
